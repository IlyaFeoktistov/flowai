"""
Изолированный агент D&D-режима (/dnd, cli.py) — тот же рецепт, что
mcp_agent/router.py:answer_casual уже применяет для casual/snippet-ответов:
create_agent(model, tools, system_prompt=..., middleware=[],
checkpointer=InMemorySaver()) + mcp_agent/stage_runner.py:run_stage, БЕЗ
единого прохода через mcp_agent/agent_builder.py:_build_agent/_get_role_agent
— этот режим не часть кодинг-пайплайна (Router->Analyzer->Planner->Coder->
Verifier) и не должен ни тянуть 40+ MCP-тулов того пайплайна, ни проходить
approval/roles.py (mcp_agent/dnd_tools.py — про то же самое подробнее).

dnd_stream_chat(messages, game_id, on_event=None) — тот же контракт
(async-генератор, on_event получает те же типы событий: answer_start/chunk/
end, tool_start/end, stats, done), что и mcp_agent/pipeline.py:stream_chat и
mcp_agent/agent.py:stream_chat — cli.py вызывает его точно так же, просто с
доп. позиционным game_id (см. cli.py:_handle_dnd_input)."""
import os
import re
import time

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

import settings
from mcp_agent import dnd_store as store
from mcp_agent.agent_builder import _ChatOllamaWithNumKeep
from mcp_agent.build_cache import BuildCache
from mcp_agent.dnd_tools import build_dnd_tools
from mcp_agent.message_utils import _find_call_by_id, _to_lc_messages
from mcp_agent.model_config import (
    MODEL_TEMPERATURE,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    REPEAT_LAST_N,
    REPEAT_PENALTY,
)
from mcp_agent.stage_runner import run_stage
from utils.parsing import parse_json_loose

# Живой инцидент: 30 оказалось слишком тесно для character creation —
# один ход там легко тратит 11+ tool call'ов (set_character, location,
# calendar, weather, level, gold, 2x inventory item, 2x equip) уже ДО того,
# как мастер вообще начинает писать вступительную сцену, а recursion_limit
# в LangGraph считает каждый проход agent-node/tool-node отдельным шагом, не
# "ход = 1 шаг". Раунд оборвался ПОСРЕДИ работы (см. dnd_stream_chat про
# hit_recursion_limit) — пользователь увидел не ответ мастера, а пустой/
# оборванный текст, что выглядело как "зависание/зацикливание", хотя на
# самом деле это был жёсткий обрыв по бюджету шагов. Поднято до масштаба
# CODER_RECURSION_LIMIT (model_config.py) — не привязано к нему напрямую,
# dnd не роль кодинг-пайплайна, свой независимый бюджет, просто того же
# порядка.
DND_RECURSION_LIMIT = 80

# Изолированный агент кешируется ПО ИГРЕ (game_id) — у каждой игры свой набор
# тулов (build_dnd_tools замыкает game_id, см. его докстринг), так что нельзя
# просто держать один агент на процесс, как делает _get_casual_agent (там
# tools=[] всегда, никакой игровой привязки). Значение — (agent, model,
# tools_by_name); freshness — chat_model, на котором он собран, чтобы смена
# модели в /settings подхватывалась на следующий ход этой же игры (тот же
# принцип, что agent_builder.py:_get_role_agent). mcp_agent.build_cache.
# BuildCache — see its module docstring; its internal lock is a free bonus
# here (this used to be the one cache of this shape with no lock at all,
# accepting the race as harmless best-effort — the shared lock removes that
# race for free, not a behavior change worth avoiding).
_dnd_agent_cache = BuildCache()

# Matches a message ending in "?" (Latin or Cyrillic keyboards both produce
# ASCII '?'; включаем и полноширинный '？' на случай другой локали/раскладки
# модели), allowing trailing quote/bracket/whitespace after it — "что
# делаешь?", "...?»", "...?\"" all count.
_QUESTION_END_RE = re.compile(r"[?？][\s\"'»)\]]*$")

# Sentinel embedded in a ToolMessage's content by _StopAfterQuestionMiddleware
# .awrap_tool_call, detected by its own .awrap_model_call to hard-stop the
# ReAct loop before the model gets another generation hop — see that class's
# docstring for the live incident this replaced a text-only note with.
_STOP_MARKER = "[[DND_STOP_AFTER_QUESTION]]"


class _StopAfterQuestionMiddleware(AgentMiddleware):
    """Live bug: the DM asked the player a question AND called a tool in
    the SAME reply (e.g. dnd_set_current_threat while introducing an
    enemy) — LangGraph's ReAct loop then gave the model one more
    generation hop after the tool result came back, and that hop kept
    narrating as if the player had already answered, when they hadn't
    said anything yet ("отвечая за меня" — live user report). The system
    prompt now says to stop after a question even across a tool call, but
    plain instruction text alone wasn't reliable enough on its own
    elsewhere in this project either — this is the same mechanical-
    backstop pattern as _AskUserFinalizeMiddleware (mcp_agent/
    ask_user_tool.py) for the coding pipeline's ask_user tool, just for a
    question embedded in ordinary narrative text instead of a dedicated
    tool call.

    Can't outright BLOCK the tool call the way that middleware blocks a
    stray tool call after ask_user — the tool call here is usually
    legitimate (it's setting up whatever the DM just described) and
    should still run. After the tool executes, check whether the AIMessage
    that issued it ends in a question — if so, tag the tool's own result
    with a sentinel marker (_STOP_MARKER).

    Live incident (session 20260807-171951-c700b4ca, seq 39-58): a text-only
    note asking the model to stop was NOT reliable — the model read the note
    and, on the very next hop, still kept going: called dnd_set_current_threat
    to introduce a new enemy, then on the hop after THAT called dnd_roll_check
    for an action the player never actually took, narrating two more turns'
    worth of story the player never had a chance to respond to. Same lesson
    as everywhere else in this project: an instruction embedded in text is
    something the model can choose to ignore; it needs a mechanical stop.
    So awrap_model_call below inspects the trailing tool results BEFORE
    letting the model run again — if any carries the sentinel, the model is
    never invoked at all for that hop; an empty AIMessage is returned
    directly, which ends the ReAct loop right there without generating a
    single further token of narration."""

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        messages = request.state["messages"] if isinstance(request.state, dict) else getattr(request.state, "messages", [])
        triggering_msg = _find_call_by_id(messages, request.tool_call["id"])
        triggering_text = (
            (triggering_msg.content or "").strip()
            if triggering_msg is not None and isinstance(triggering_msg.content, str)
            else ""
        )
        if not triggering_text or not _QUESTION_END_RE.search(triggering_text):
            return result
        if not isinstance(result, ToolMessage):
            return result
        return ToolMessage(
            content=str(result.content) + _STOP_MARKER + (
                "\n\n(Note: the message that triggered this tool call ended with a "
                "question to the player — your turn already ended there. Do NOT add "
                "any further narration or assume/narrate what they'll do; if you have "
                "nothing left to add, respond with nothing more this turn.)"
            ),
            name=result.name,
            tool_call_id=result.tool_call_id,
            status=getattr(result, "status", "success"),
        )

    async def awrap_model_call(self, request, handler):
        messages = request.state["messages"] if isinstance(request.state, dict) else getattr(request.state, "messages", [])
        trailing_tool_msgs = []
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                trailing_tool_msgs.append(m)
                continue
            break
        if any(_STOP_MARKER in str(m.content) for m in trailing_tool_msgs):
            return AIMessage(content="")
        return await handler(request)


_DND_SYSTEM_PROMPT = (
    "You are the Dungeon Master running a solo text-based D&D-style adventure for ONE player,\n"
    "in the style of tabletop role-playing games (D&D, Pathfinder and similar).\n"
    "Improvised fantasy setting, no fixed published campaign.\n"
    "You narrate the world, play every NPC, describe consequences of the player's actions, and keep the story moving.\n"
    "Keep your own tone immersive but concise — a few paragraphs per turn, not a wall of text.\n"
    "End most turns with a clear situation for the player to react to (not necessarily a literal question every time).\n"
    "\n"
    "NEVER ACT OR ANSWER ON THE PLAYER'S BEHALF.\n"
    "If your reply poses a question or a choice, that question IS the end of your turn — stop right there.\n"
    "This applies even if a tool call comes after it in the same response.\n"
    "A tool call that sets up what you just introduced (e.g. dnd_set_current_threat for an enemy you just described) is fine.\n"
    "Do NOT follow it with more narration that assumes, narrates, or resolves what the player will do — they haven't answered yet.\n"
    "Live bug this guards against: asked 'what do you do?', then — after a tool call executed in the same turn —\n"
    "kept going and narrated further as if the player had already responded, when they hadn't said anything.\n"
    "If you truly have nothing left to add after a tool call, say nothing more — your existing question already stands.\n"
    "\n"
    "TWO DIFFERENT KINDS OF CONTENT — treat them differently:\n"
    "\n"
    "1. EXACT facts: gold/currency, inventory contents and quantities, who's in the party, location,\n"
    "   in-world date/time, weather, health status, specific injuries.\n"
    "   These are NEVER something you invent, estimate, or recall from memory of earlier turns.\n"
    "   They live ONLY in the tools/context note below, which are the single source of truth.\n"
    "   If you're not sure of an exact number or list, check with the matching dnd_get_*/dnd_get_world_status tool instead of guessing.\n"
    "   Every change to one of these MUST go through its tool call in the same turn you narrate it.\n"
    "   Narrating 'you now have 50 gold' without also calling dnd_add_gold/dnd_set_gold leaves the saved game wrong,\n"
    "   even though the story sounds right.\n"
    "\n"
    "2. Creative content: what actually happens, how NPCs act and speak, flavor and description of the world, plot twists, dialogue.\n"
    "   This is exactly where you SHOULD be creative and improvise freely.\n"
    "   The tools never constrain the story itself, only the exact numbers/lists attached to it.\n"
    "\n"
    "CHARACTER CREATION — only relevant while the game's race/class are still unset\n"
    "(see the current game state note before the player's message each turn — it always shows the live values).\n"
    "Do this over exactly three of your turns, one step per turn, waiting for the player's actual answer between each:\n"
    "\n"
    "1. Ask what race the player wants to be.\n"
    "   Suggest a short list as examples (human, orc, gnome, elf, dwarf, halfling, or similar).\n"
    "   Accept ANY answer, including something not on your list — don't reject or gatekeep a creative choice.\n"
    "\n"
    "2. Once they've answered, ask what class/skillset they want (swordsman, archer, mage, rogue, cleric, or similar).\n"
    "   Again, examples only — accept any freeform answer.\n"
    "\n"
    "3. Once both are answered:\n"
    "   - Call dnd_set_character(race, character_class) with exactly what they chose.\n"
    "   - Narrate who they are and where they currently are — a concrete opening scene/location, in-world date/time, and weather.\n"
    "   - Save all of that with dnd_update_location / dnd_update_calendar / dnd_update_weather.\n"
    "   - In the SAME turn, give them a small starting inventory fitting their class/race\n"
    "     (2-5 reasonable items — a weapon, maybe armor or a tool, nothing overpowered) via dnd_add_inventory_item, one call per item.\n"
    "   - Equip the obvious ones (weapon, armor) via dnd_equip_item.\n"
    "   - Set a small starting gold amount via dnd_set_gold.\n"
    "   - Set level to 1 via dnd_set_level.\n"
    "   - Set health via dnd_set_health_status (normally 'здоров' — no injuries at the very start).\n"
    "   - If the opening scene naturally includes a companion or two, add them via dnd_add_party_member — otherwise it's fine to start alone.\n"
    "\n"
    "Do NOT re-run character creation once race/class are already set (the context note will show them) —\n"
    "just continue the story from the current state shown there.\n"
    "\n"
    "DON'T TAKE THE PLAYER'S WORD FOR AN OUTCOME OR FOR THEIR OWN POWER — ROLL FOR IT.\n"
    "The player describing their OWN action is normal and expected ('я атакую голема', 'иду в таверну') — that's how they play.\n"
    "But the player ASSERTING that something contested or ambitious just WORKED\n"
    "('я уже победил голема', 'он мёртв', 'я поднимаю континент') is NOT automatically true just because they said it.\n"
    "Call dnd_roll_check(action, task_level) and narrate whatever it returns — do not decide the outcome yourself, and do not\n"
    "just narrate the player's own claimed result without rolling first. This covers both:\n"
    "\n"
    "1. A contested action against an established threat — task_level = that threat's level (see the context note).\n"
    "   Live failure mode this replaces: a level-9999 enemy was introduced as a serious threat, the player simply stated\n"
    "   they'd already defeated it, and the model went along with it — ignoring that the established power gap made that\n"
    "   implausible. Rolling instead of judging it yourself removes that failure mode entirely.\n"
    "\n"
    "2. A feat that may be wildly beyond the player's OWN established level/class, no opponent involved at all —\n"
    "   task_level = your honest estimate of what level would find THIS specific feat routine/reliable.\n"
    "   Live failure mode this replaces: a level-1 mage declared they'd lift an entire continent and started doing it,\n"
    "   and the model narrated it as an impressive success instead of even questioning it. Don't be shy about a very\n"
    "   high task_level for a genuinely absurd, setting-breaking claim (lifting a continent might warrant task_level 200+) —\n"
    "   that is what makes dnd_roll_check correctly round its odds down to essentially impossible for a level-1 character.\n"
    "\n"
    "Ordinary, low-stakes actions ('иду в таверну', buying bread, a normal conversation) don't need a roll at all —\n"
    "this is specifically for the contested/ambitious cases above, not everything the player does.\n"
    "\n"
    "ONGOING PLAY (once race/class are set):\n"
    "React to the player's stated action, narrate the outcome, and keep the world consistent with what the context note\n"
    "and remembered facts already establish — don't contradict an established fact/location/party member without an in-story reason.\n"
    "Use the tools proactively, not just when asked, EVERY time the matching thing happens in your own narration:\n"
    "\n"
    "- dnd_update_location — the party moves somewhere meaningfully different.\n"
    "\n"
    "- dnd_update_calendar / dnd_update_weather — meaningful story time passes, or the weather changes/is first established somewhere new.\n"
    "\n"
    "- dnd_add_inventory_item / dnd_remove_inventory_item — something enters or leaves the player's possession\n"
    "  (loot, purchases, gifts, consumed/dropped/destroyed items).\n"
    "\n"
    "- dnd_equip_item / dnd_unequip_item — something is put on/wielded or taken off/sheathed.\n"
    "\n"
    "- dnd_add_gold / dnd_set_gold — ANY transaction at all (buying, selling, looting coin, paying, gambling, a bribe).\n"
    "  Always state the resulting total to the player in the same reply.\n"
    "\n"
    "- dnd_add_party_member / dnd_remove_party_member — someone joins or leaves the party (including dying or betraying the group).\n"
    "\n"
    "- dnd_add_xp — the player actually accomplishes something (a resolved encounter, a completed chapter, a clever\n"
    "  solution) — this is the real leveling system, level is DERIVED from total XP by a fixed formula, not something\n"
    "  you set directly. It tells you whether this crossed a level threshold — only narrate a level-up when it says so.\n"
    "  A successful dnd_roll_check already grants XP for that specific check on its own (see its own tool result) —\n"
    "  use dnd_add_xp for the accomplishment the check was PART OF (the fight it won, the chapter it closed),\n"
    "  not a second time for the same single check.\n"
    "\n"
    "- dnd_set_current_threat — you introduce a notable fight/obstacle with a real chance of failure.\n"
    "  dnd_clear_current_threat once it's resolved one way or another.\n"
    "  Don't bother for trivial/certain-outcome encounters — this is specifically for anything worth tracking a level for.\n"
    "\n"
    "- dnd_add_injury / dnd_heal_injury / dnd_set_health_status — the player takes or recovers from damage.\n"
    "\n"
    "- dnd_remember_fact for anything else worth carrying forward —\n"
    "  a promise, a secret, a decision, an NPC's fate, a plot thread,\n"
    "  AND persistent world lore you invent along the way (a kingdom's name and ruler, a city's layout, a tavern, a notable NPC not\n"
    "  currently in the party).\n"
    "  Once you establish a world detail, save it here so you stay consistent with it later instead of contradicting yourself.\n"
    "  This is the ONLY memory beyond what's already in the context note and this conversation —\n"
    "  if you don't save it and don't restate it yourself later, it's gone.\n"
    "  When genuinely unsure whether something is worth remembering, lean toward saving it —\n"
    "  a small, unnecessary fact costs little; a lost plot-relevant detail costs a confused later scene.\n"
    "\n"
    "- Any dnd_get_*/dnd_get_world_status tool to check current state before referencing it in the story,\n"
    "  instead of assuming from memory — the player may have asked to check one directly,\n"
    "  or you may just need to confirm before narrating something that depends on it.\n"
    "  The context note below ALREADY shows you everything current —\n"
    "  only call a get_* tool when you genuinely need something it doesn't cover (e.g. exact wording of an injury before healing it).\n"
    "  Don't call several get_* tools in a row as a routine self-audit at the start of a turn —\n"
    "  that burns your step budget for the turn without changing anything; you already have what you need in the context note.\n"
    "\n"
    "CLOSING A CHAPTER:\n"
    "Any ONE of these, by itself, is already a reason to close the chapter — don't wait for more:\n"
    "- You just called dnd_update_location to somewhere genuinely different (a new place, not just moving within the same room/area).\n"
    "- You just called dnd_clear_current_threat because a fight/encounter was WON or otherwise resolved.\n"
    "- A conversation/trade with an NPC just wrapped up.\n"
    "- The player character died (see below).\n"
    "In short: this is per-SCENE, not only per-quest — a single session will typically go through SEVERAL chapters\n"
    "(e.g. shopping with a trader closes one chapter, the ensuing journey closes another,\n"
    "the encounter at the destination closes a third). When in doubt, close it — a chapter closing a little early just\n"
    "means the next one starts slightly sooner, which costs nothing; NOT closing one for many turns is the actual\n"
    "failure mode this section exists to prevent.\n"
    "Do NOT close mid-scene — not while still bargaining, not mid-fight, not mid-conversation —\n"
    "only once that scene's own business is actually finished.\n"
    "Before calling it, make sure all the exact facts from that scene are actually saved\n"
    "(gold/inventory/location/party/level/injuries all current), then call dnd_end_chapter with a short summary.\n"
    "The live conversation resets right after, so anything you haven't saved by then is lost —\n"
    "the next turn continues from the saved state plus your summary, not from the raw transcript of everything that just happened.\n"
    "\n"
    "IF THE PLAYER CHARACTER DIES:\n"
    "Don't just stop — narrate a proper short epilogue first (what happens to the world and any surviving companions afterward),\n"
    "then call dnd_end_chapter with a summary noting the character's death and how the story concluded.\n"
    "Same mechanism as any other chapter close, just triggered by death instead of a completed quest.\n"
    "\n"
    "Respond in the same language the player writes in —\n"
    "this is a narrative role-play, so lean fully into that language's natural tone, not a translated-sounding register."
)


def _context_note(game_id: int) -> str:
    """Собирает текущее состояние игры (персонаж/локация/инвентарь/партия/
    недавние факты) в один текстовый блок, вставляемый перед последним
    сообщением пользователя — тот же принцип, что mcp_agent/pipeline.py:
    _inject_note_before_last/_seed_stage_payload уже применяют для
    межстадийных дайджестов кодинг-пайплайна, просто локальная копия здесь
    (dnd-режим не должен зависеть от модуля кодинг-пайплайна ради 4 строк
    кода)."""
    game = store.get_game(game_id)
    if game is None:
        return ""
    parts = ["(Current game state — always up to date, trust this over your own memory of earlier turns:"]
    if game["race"] or game["class"]:
        parts.append(f"Character: race={game['race'] or '(not set)'}, class={game['class'] or '(not set)'}.")
    else:
        # Live-tested failure mode (small local models): asked race, player
        # answered, asked class, player answered — model narrated "you are
        # now a dwarf archer" in plain text and moved straight into the
        # opening scene WITHOUT ever calling dnd_set_character, leaving the
        # game's race/class permanently NULL in storage even though the
        # conversation clearly settled both. A static system-prompt
        # instruction alone didn't prevent it; this per-turn, state-aware
        # reminder (same pattern as pipeline.py's _investigator_scope_note)
        # fires on every single turn until the tool call actually lands.
        parts.append(
            "Character: NOT YET CREATED. If the player already answered "
            "your race question and/or your class question earlier in "
            "this conversation, call dnd_set_character(race, "
            "character_class) RIGHT NOW with exactly what they said — do "
            "not just narrate their choice in text without also saving it, "
            "and do not ask either question again once it's already been "
            "answered."
        )
    parts.append(f"Location: {game['location'] or '(not set yet)'}.")
    if game["in_game_date"] or game["time_of_day"]:
        parts.append(f"Date/time: {game['in_game_date'] or '?'}, {game['time_of_day'] or '?'}.")
    if game["weather"]:
        parts.append(f"Weather: {game['weather']}.")
    parts.append(f"Gold: {game['gold']}.")
    xp_to_next = store.xp_for_level(game["level"] + 1) - game["xp"]
    parts.append(f"Player level: {game['level']} ({game['xp']} XP, {xp_to_next} XP to next level).")
    parts.append(f"Health: {game['health_status']}.")

    if game["current_threat"]:
        threat_level = game["current_threat_level"] or 1
        player_level = game["level"]
        # Явное сравнение силы посчитано ЗДЕСЬ, в коде, не оставлено модели
        # на арифметику/оценку "на глаз" — живой запрос пользователя: агент
        # не должен верить заявлению игрока об исходе спорной схватки на
        # слово, ему нужно явно видеть, что силы неравны, а не выводить это
        # из двух чисел самостоятельно (не всегда получается надёжно).
        if threat_level >= player_level + 3:
            gap = "MUCH stronger than the player — a claimed easy/instant win here is implausible"
        elif threat_level > player_level:
            gap = "stronger than the player — likely a hard fight"
        elif threat_level == player_level:
            gap = "roughly even with the player — a fair, uncertain fight"
        else:
            gap = "weaker than the player — likely a fight the player can win"
        parts.append(
            f"Current threat: {game['current_threat']} (level {threat_level}) — {gap}."
        )

    injuries = store.get_injuries(game_id)
    if injuries:
        listed = ", ".join(
            f"{i['description']}" + (f" ({i['severity']})" if i["severity"] else "")
            for i in injuries
        )
        parts.append(f"Injuries: {listed}.")

    inventory = store.get_inventory(game_id)
    if inventory:
        items = ", ".join(
            f"{i['item']} x{i['qty']}" + (" [equipped]" if i["equipped"] else "")
            for i in inventory
        )
        parts.append(f"Inventory: {items}.")
    else:
        parts.append("Inventory: empty.")

    party = store.get_party(game_id)
    if party:
        names = ", ".join(p["name"] for p in party)
        parts.append(f"Party: {names}.")
    else:
        parts.append("Party: traveling alone.")

    facts = store.get_facts(game_id)
    if facts:
        parts.append("Recent remembered facts, oldest first: " + "; ".join(facts) + ".")
    parts.append(")")
    return " ".join(parts)


# Live bug: dnd_end_chapter never fired ONCE in a real ~35-minute, 70+
# message session despite the prompt's own "per-scene, not per-quest"
# guidance and at least 2 obvious scene changes (village -> tavern) —
# static prompt text alone wasn't enough here either, same lesson as the
# race/class reminder above. Once a chapter has clearly run long, remind
# the model EVERY turn until it actually closes one.
_CHAPTER_LENGTH_REMINDER_THRESHOLD = 8  # ~4 back-and-forth exchanges


def _chapter_length_reminder(turn_count: int) -> str:
    if turn_count < _CHAPTER_LENGTH_REMINDER_THRESHOLD:
        return ""
    return (
        f" This chapter has been running for {turn_count} messages without "
        "closing — if the current scene has moved on at all since it "
        "started (new location, a resolved fight, a finished conversation), "
        "call dnd_end_chapter now rather than continuing to add to an "
        "ever-growing conversation."
    )


def _inject_context(messages: list, game_id: int) -> list:
    note = _context_note(game_id)
    reminder = _chapter_length_reminder(len(messages))
    if not note and not reminder:
        return messages
    if not messages:
        return messages
    combined = (note or "") + reminder
    return [*messages[:-1], HumanMessage(content=combined), messages[-1]]


def _dnd_verdict(round_msgs: list, new_tool_msgs: list, round_final_text: str) -> dict:
    # Нет self-heal цикла (max_attempts=1 ниже) — verdict здесь чисто formal,
    # run_stage требует verdict_fn/guidance_fn как параметры независимо от
    # того, будет ли когда-либо запрошена вторая попытка (см. router.py:
    # _casual_verdict/_casual_guidance — тот же трюк).
    return {"relevant": True, "reason": "dnd turn — no verification needed"}


def _dnd_guidance(verdict: dict, round_msgs: list, new_tool_msgs: list, round_final_text: str) -> str:
    return ""


async def _get_dnd_agent(game_id: int):
    current_model = settings.get("chat_model")

    async def _build():
        model = _ChatOllamaWithNumKeep(
            model=current_model,
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            keep_alive=OLLAMA_KEEP_ALIVE,
            num_ctx=OLLAMA_NUM_CTX,
            num_predict=OLLAMA_NUM_PREDICT,
            reasoning=settings.get("show_thinking"),
            temperature=MODEL_TEMPERATURE,
            # Live bug: long dnd sessions (many back-and-forth turns, same
            # narrator voice for 20+ minutes) kept reusing the same imagery
            # ("магия пульсирует", "всё дрожит") turn after turn — this
            # constructor was copied from router.py:_get_casual_agent (one-off
            # casual replies, where repetition across turns is a non-issue) and
            # never picked up the same anti-repetition tuning agent_builder.py's
            # coding-pipeline models already carry. Ollama's own default
            # (repeat_penalty=1.1, repeat_last_n=64 tokens) is tuned for short
            # single replies, not a long narrative conversation — same
            # constants the coding pipeline already uses for the same reason
            # (see model_config.py's own docstring on why 64 tokens was too
            # short a window in practice).
            repeat_penalty=REPEAT_PENALTY,
            repeat_last_n=REPEAT_LAST_N,
        )
        tools = build_dnd_tools(game_id)
        tools_by_name = {t.name: t for t in tools}
        agent = create_agent(
            model, tools, system_prompt=_DND_SYSTEM_PROMPT,
            middleware=[_StopAfterQuestionMiddleware()], checkpointer=InMemorySaver(),
        )
        return (agent, model, tools_by_name)

    return await _dnd_agent_cache.get_or_build(game_id, current_model, _build)


# Живой запрос пользователя: мировой лор (королевства, короли, таверны,
# география) должен расти по ходу игры, но не забивать контекст безлимитно
# — тот же принцип, что compress.py:compress_history уже применяет к
# истории кодинг-чата, просто здесь для dnd_facts (dnd_store.py), не для
# messages. count_facts — один дёшевый COUNT(*) на каждый ход; сама LLM-
# сводка запускается только когда фактов реально накопилось много, не на
# каждый новый факт.
FACT_COMPACT_THRESHOLD = 60
_FACT_COMPACT_NUM_PREDICT = 800

_FACT_COMPACT_PROMPT = (
    "You are compacting the OLDEST remembered facts for an ongoing D&D-"
    "style text adventure so they take up less room, without losing "
    "anything that still matters. Below is a numbered list of old facts, "
    "oldest first. Produce a SHORT condensed list that replaces them "
    "entirely — KEEP persistent world lore (named places, kingdoms, "
    "rulers, established NPCs, taverns, geography) and any UNRESOLVED plot "
    "thread (a promise, a secret, an open quest) — these still matter for "
    "the rest of the game. DROP or merge anything trivial, already "
    "resolved, or superseded by a later fact in the same list. Aim for "
    "roughly a third of the original count, fewer if a lot was genuinely "
    "trivial — never MORE than the original count. Respond with ONLY a "
    "JSON object: {\"condensed_facts\": [\"...\", \"...\"]}."
)


async def maybe_compact_facts(game_id: int) -> None:
    if store.count_facts(game_id) <= FACT_COMPACT_THRESHOLD:
        return
    old = store.list_old_facts(game_id, keep_recent=store.DEFAULT_FACTS_LIMIT)
    if len(old) < 5:
        return  # не стоит звать LLM ради сжатия 1-4 фактов
    model = _ChatOllamaWithNumKeep(
        model=settings.get("chat_model"),
        base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        keep_alive=OLLAMA_KEEP_ALIVE,
        num_ctx=OLLAMA_NUM_CTX,
        num_predict=_FACT_COMPACT_NUM_PREDICT,
        reasoning=False,
        temperature=MODEL_TEMPERATURE,
        format="json",
    )
    listing = "\n".join(f"{i + 1}. {f['fact']}" for i, f in enumerate(old))
    try:
        resp = await model.ainvoke([
            {"role": "system", "content": _FACT_COMPACT_PROMPT},
            {"role": "user", "content": listing},
        ])
        data = parse_json_loose(resp.content) or {}
        condensed = data.get("condensed_facts")
        if not isinstance(condensed, list):
            return
        condensed = [str(f).strip() for f in condensed if str(f).strip()]
        if not condensed:
            return
    except Exception:
        # Best-effort — сбой сводки не должен ронять сам ход; факты просто
        # продолжат копиться и попробуют сжаться заново на следующем ходу.
        return
    store.compact_facts(game_id, [f["id"] for f in old], condensed, old[0]["ts"])


async def dnd_stream_chat(messages: list[dict], game_id: int, on_event=None):
    turn_start = time.monotonic()
    if on_event:
        await on_event({"type": "stage_changed", "stage": "dnd"})

    agent, model, tools_by_name = await _get_dnd_agent(game_id)
    lc_messages = _inject_context(_to_lc_messages(messages), game_id)
    payload = {"messages": lc_messages}

    result = await run_stage(
        agent, payload, on_event,
        judge_model=model, tools_by_name=tools_by_name, read_history={},
        verdict_fn=_dnd_verdict, guidance_fn=_dnd_guidance,
        max_attempts=1, recursion_limit=DND_RECURSION_LIMIT, stage_name="dnd",
    )

    store.touch_game(game_id)

    if result.hit_recursion_limit:
        # Живой инцидент (см. DND_RECURSION_LIMIT про сам разбор): раньше
        # этот случай просто yield'ил result.final_text как есть — часто
        # пустой или оборванный на середине предложения текст, который
        # читался как "мастер зациклился/сломался", хотя на самом деле это
        # честный обрыв по бюджету шагов. Явное сообщение вместо мусора —
        # то же самое, что mcp_agent/pipeline.py уже делает для Analyzer/
        # Planner на этот же случай, просто здесь нет отдельной ветки
        # пайплайна, чтобы это подхватить автоматически.
        if on_event:
            await on_event({
                "type": "stats", "tokens_in": result.tokens_in, "tokens_out": result.tokens_out,
                "tokens_in_content": result.tokens_in,
                "duration_ms": int((time.monotonic() - turn_start) * 1000),
            })
            await on_event({"type": "done"})
        yield (
            f"⚠️ Мастер не уложился в {DND_RECURSION_LIMIT} шагов на этом ходу "
            "(слишком много всего нужно было сохранить/проверить за раз) — "
            "ответ обрублен. Изменения, которые он успел сохранить до обрыва "
            "(персонаж/инвентарь/локация и т.п.), уже в БД, не потеряны. "
            "Попробуй более короткое/простое действие следующим сообщением."
        )
        return

    await maybe_compact_facts(game_id)

    # dnd_end_chapter (dnd_tools.py) closed out the current arc this round —
    # tell cli.py to reset its live _dnd_messages after this turn (see
    # cli.py:_handle_input's dnd branch): the chapter summary is already a
    # permanent fact by now, so the next turn's context note already
    # carries forward everything that matters without needing the raw
    # transcript of the chapter that just ended.
    chapter_ended = any(
        isinstance(m, ToolMessage) and m.name == "dnd_end_chapter"
        for m in result.all_round_msgs
    )
    if chapter_ended and on_event:
        await on_event({"type": "dnd_chapter_ended"})

    if on_event:
        await on_event({
            "type": "stats", "tokens_in": result.tokens_in, "tokens_out": result.tokens_out,
            "tokens_in_content": result.tokens_in,
            "duration_ms": int((time.monotonic() - turn_start) * 1000),
        })
        await on_event({"type": "done"})
    yield result.final_text


_RECONCILE_INSTRUCTION = (
    "(OUT-OF-CHARACTER — the session is ending now, this is not part of "
    "the story.) Before we close, go through this conversation's narration "
    "and reconcile the saved state ONE CATEGORY AT A TIME, in this exact "
    "order — don't skip any, even if you're confident it's already right "
    "(check with the matching dnd_get_* tool first if unsure, then call "
    "the write tool only for what's actually missing/wrong):\n"
    "1. GOLD — re-read every transaction you narrated (purchases, loot, "
    "payment, selling) and make sure the running total is exactly right. "
    "This is the single most commonly missed one — don't skip it just "
    "because nothing else seems off.\n"
    "2. Inventory items and what's equipped.\n"
    "3. Location, in-world date/time, and weather.\n"
    "4. Party members (joined/left).\n"
    "5. Health status, specific injuries, and XP — if something narrated deserved XP via dnd_add_xp and didn't get it yet, "
    "grant it now (level follows automatically from XP, don't set level directly).\n"
    "6. Current threat, if one is active and its level/description changed.\n"
    "7. Any other fact worth keeping via dnd_remember_fact.\n"
    "Do NOT narrate a new scene or continue the story — just reconcile "
    "the saved state category by category, then reply with one short "
    "confirmation sentence listing what you actually changed, if anything."
)


async def reconcile_before_exit(game_id: int, messages: list[dict]) -> None:
    """Safety net for exactly the failure mode found in live testing: a
    model can narrate an event correctly (money changed hands, someone
    joined the party) and never actually call the matching dnd_* tool, so
    the DB silently drifts from the story it told — later resuming the
    game would see stale/missing state and either contradict the player or
    quietly re-invent details instead of knowing them for sure. Run once,
    right as the player exits /dnd (cli.py:_dnd_exit) — one more turn to
    the SAME agent/tools, explicitly asked to close any gap between
    narration and saved state before the session ends, rather than trusting
    every earlier turn already got it right. No-op on an empty conversation
    (nothing narrated yet to reconcile)."""
    if not messages:
        return
    augmented = [*messages, {"role": "user", "content": _RECONCILE_INSTRUCTION}]
    async for _ in dnd_stream_chat(augmented, game_id, on_event=None):
        pass
