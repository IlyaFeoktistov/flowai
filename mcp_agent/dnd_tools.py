"""
Прямые (не MCP) тулы для изолированного D&D-агента (mcp_agent/dnd_agent.py) —
тот же рецепт, что уже применён для ask_user/mark_plan_step_current
(mcp_agent/ask_user_tool.py): обычные @tool-функции, никакого MCP-подпроцесса,
потому что это простое чтение/запись в SQLite (mcp_agent/dnd_store.py), а не
что-то, ради чего стоило бы поднимать отдельный процесс.

build_dnd_tools(game_id) — ФАБРИКА, не статичный список: game_id замыкается
в каждый тул при вызове, так что модели не нужно (и негде) ошибиться, указав
неверный/чужой game_id — тот же принцип, что _bind_constant_args
(tool_wrappers.py) применяет к repo_path у git-тулов, только через обычное
замыкание Python, а не пост-обработку JSON-схемы (эти тулы не MCP-тулы с
готовой dict-схемой, а свои StructuredTool — схему можно просто не включать
game_id в саму функцию).

Не проходят через TOOLS_REQUIRING_APPROVAL/roles.py — они собираются в
create_agent() изолированного dnd-агента напрямую (см. dnd_agent.py), минуя
agent_builder._build_agent/_get_role_agent целиком. Approval тут не нужен по
смыслу: это не файлы пользователя и не shell-команды, а изолированная
игровая БД, которую сам dnd-режим и создал."""
import math
import random

from langchain_core.tools import tool

from mcp_agent import dnd_store as store

# Live request: a level-1 mage declared they'd lift a continent, and the DM
# just narrated it as a success — prompt-only "is this plausible?" judgment
# kept failing in practice (see mcp_agent/dnd_agent.py's docstring on the
# same incident). dnd_roll_check replaces that judgment call with an actual
# random roll, computed here in code — the model still decides HOW HARD a
# feat is (task_level) and how to NARRATE the result, but not WHETHER it
# succeeds.
#
# Logistic curve, not a linear d20+modifier: a flat d20 always has a 1/20
# floor for a "natural 20" success no matter how absurd the modifier gap
# is, which is the opposite of what's needed here — an extreme mismatch
# (level 1 vs a level-1000 feat) must be able to round down to
# "essentially impossible", not floor out at 5%. _CHECK_STEEPNESS controls
# how fast the odds collapse per level of mismatch; player_level == task_level
# is an even coin flip by construction (gap=0 -> p=0.5), matching "hard but
# not impossible" for an evenly-matched attempt.
_CHECK_STEEPNESS = 0.5


def _check_probability(player_level: int, task_level: int) -> float:
    gap = player_level - task_level
    exponent = max(-700.0, min(700.0, -_CHECK_STEEPNESS * gap))
    p = 1.0 / (1.0 + math.exp(exponent))
    return max(1e-12, min(0.999, p))


# Auto-XP on a successful dnd_roll_check — the DM previously had to
# separately remember to call dnd_add_xp for this, which was exactly the
# kind of "relies on the model remembering an instruction" gap this project
# keeps hitting elsewhere (live report: DM never granted XP for won checks).
# Scaled by how much harder the beaten task was than the player's own level
# (_check_probability's own gap) rather than a flat amount, so a routine
# coin-flip check nets modest XP and beating something well above the
# player's level nets more — same "minor/moderate/major" bands dnd_add_xp's
# own docstring already calibrates to (10-30 / 50-150), just computed
# instead of left to the model's judgment.
_ROLL_XP_BASE = 15
_ROLL_XP_PER_GAP = 12


def _roll_xp_gain(player_level: int, task_level: int, critical: bool) -> int:
    gap = max(0, task_level - player_level)
    xp = _ROLL_XP_BASE + _ROLL_XP_PER_GAP * gap
    return round(xp * 1.5) if critical else xp


def build_dnd_tools(game_id: int) -> list:
    @tool
    async def dnd_set_character(race: str, character_class: str) -> str:
        """Save the player's chosen race and class for THIS game — call this
        ONCE, right after the player answers your race question and your
        class/skill question (e.g. race='эльф', character_class='лучник').
        Overwrites any previous value, so only call it again later if the
        character's race/class genuinely changes in the story (rare — a
        curse, a class change item, etc.), not on every turn. Both
        arguments are REQUIRED and must be the player's actual answer —
        never call this with a blank/placeholder value just because a
        reminder told you to; wait until you genuinely have both real
        answers first."""
        race, character_class = race.strip(), character_class.strip()
        if not race or not character_class:
            return (
                "Error: race and character_class must both be the player's "
                "actual answers, not blank — don't call this tool until you "
                "genuinely have both. If you don't have them yet, ask the "
                "missing one instead of calling this."
            )
        store.set_character(game_id, race, character_class)
        return f"Character saved: race={race!r}, class={character_class!r}."

    @tool
    async def dnd_update_location(location: str) -> str:
        """Save where the player's character currently is — call this every
        time the party moves to a meaningfully different place (a new room,
        town, forest, dungeon level), not for staying put within the same
        scene. Short, concrete description (e.g. 'тёмный лес у старой
        мельницы'), not a full narration — the narration itself goes in your
        normal reply text, this is just the persisted state."""
        location = location.strip()
        if not location:
            return "Error: location must not be blank."
        store.update_location(game_id, location)
        return f"Location saved: {location!r}."

    @tool
    async def dnd_update_calendar(in_game_date: str, time_of_day: str) -> str:
        """Save the current in-world date and time of day (e.g.
        in_game_date='3-й день похода', time_of_day='утро') — call this
        whenever a meaningful amount of story time passes (a new day, a
        long rest, a time skip), not for minutes passing within the same
        scene. This is IN-WORLD fantasy time, unrelated to the real clock."""
        in_game_date, time_of_day = in_game_date.strip(), time_of_day.strip()
        if not in_game_date or not time_of_day:
            return "Error: both in_game_date and time_of_day must be non-blank."
        store.update_calendar(game_id, in_game_date, time_of_day)
        return f"Calendar saved: {in_game_date!r}, {time_of_day!r}."

    @tool
    async def dnd_update_weather(weather: str) -> str:
        """Save the current in-world weather (e.g. 'лёгкий туман', 'ясно',
        'ливень') — call this whenever it changes or you first establish it
        for a new location."""
        weather = weather.strip()
        if not weather:
            return "Error: weather must not be blank."
        store.update_weather(game_id, weather)
        return f"Weather saved: {weather!r}."

    @tool
    async def dnd_get_world_status() -> str:
        """One-call overview of everything currently true about the world
        and the player: location, in-world date/time, weather, gold, and
        health status. Call this whenever you need to check several of
        these at once (e.g. narrating a new scene) instead of several
        separate reads — or whenever the player directly asks something
        like 'что сейчас происходит', 'какая погода', 'сколько сейчас
        времени'."""
        game = store.get_game(game_id)
        if game is None:
            return "Error: game not found."
        xp_to_next = store.xp_for_level(game["level"] + 1) - game["xp"]
        return (
            f"Location: {game['location'] or '(not set)'}. "
            f"Date/time: {game['in_game_date'] or '(not set)'}, "
            f"{game['time_of_day'] or '(not set)'}. "
            f"Weather: {game['weather'] or '(not set)'}. "
            f"Gold: {game['gold']}. "
            f"Level: {game['level']} ({game['xp']} XP, {xp_to_next} XP to next level). "
            f"Health: {game['health_status']}."
        )

    @tool
    async def dnd_add_inventory_item(item: str, qty: int = 1, description: str = "") -> str:
        """Add item(s) to the player's inventory — call this whenever the
        story has them pick up, buy, loot, or receive something. If the
        exact same item name is already in the inventory, qty is ADDED to
        the existing stack (e.g. picking up 3 more arrows when you already
        have 5 results in 8, not a second separate entry) — description on
        a merge is ignored, only the first description sticks, so keep it
        consistent if you add the same item more than once."""
        item = item.strip()
        if not item:
            return "Error: item must not be blank."
        if qty < 1:
            return "Error: qty must be at least 1 — use dnd_remove_inventory_item to take items away."
        store.add_inventory_item(game_id, item, qty, description)
        return f"Added {qty}x {item!r} to inventory."

    @tool
    async def dnd_remove_inventory_item(item: str, qty: int = 1) -> str:
        """Remove item(s) from the inventory — the player used, dropped,
        sold, or lost something. Matches the item name case-insensitively.
        If qty removed reaches or exceeds what's held, the item is dropped
        from the inventory entirely, not left at 0. Returns whether the item
        was actually found — if it wasn't, say so honestly in your reply
        (the player can't drop/use something they don't have), don't just
        assume it worked."""
        removed = store.remove_inventory_item(game_id, item, qty)
        if not removed:
            return f"Not found in inventory: {item!r} — nothing removed."
        return f"Removed {qty}x {item!r} from inventory (or fewer, if that's all that was left)."

    @tool
    async def dnd_get_inventory() -> str:
        """List everything currently in the player's inventory. Call this
        when the player asks what they're carrying, or when you (the DM)
        need to check what's available before letting them use/reference an
        item in the story — don't guess from memory what's in the
        inventory, it may have changed since you last checked."""
        items = store.get_inventory(game_id)
        if not items:
            return "Inventory is empty."
        return "\n".join(
            f"- {i['item']} x{i['qty']}"
            + (" [equipped]" if i["equipped"] else "")
            + (f" — {i['description']}" if i["description"] else "")
            for i in items
        )

    @tool
    async def dnd_equip_item(item: str, slot: str = "") -> str:
        """Mark an item ALREADY in the inventory as worn/wielded (e.g.
        equipping a sword, putting on armor) — slot is a free-form label
        for what it occupies ('оружие', 'голова', 'доспех'), optional.
        Doesn't create the item — if it's not in the inventory yet, add it
        with dnd_add_inventory_item first."""
        item = item.strip()
        if not item:
            return "Error: item must not be blank."
        ok = store.equip_item(game_id, item, slot)
        if not ok:
            return f"Not found in inventory: {item!r} — add it first with dnd_add_inventory_item."
        return f"Equipped: {item!r}" + (f" ({slot})" if slot else "") + "."

    @tool
    async def dnd_unequip_item(item: str) -> str:
        """Mark an equipped item as no longer worn/wielded — it stays in
        the inventory, just no longer equipped."""
        ok = store.unequip_item(game_id, item)
        if not ok:
            return f"Not found in inventory: {item!r} — nothing to unequip."
        return f"Unequipped: {item!r}."

    @tool
    async def dnd_get_equipment() -> str:
        """List only the items currently worn/wielded (a subset of the full
        inventory — see dnd_get_inventory for everything being carried).
        Call this when the story needs to know what the character has
        actively equipped right now (a fight starting, checking armor)."""
        equipped = store.get_equipped(game_id)
        if not equipped:
            return "Nothing currently equipped."
        return "\n".join(
            f"- {i['item']}" + (f" ({i['slot']})" if i["slot"] else "")
            for i in equipped
        )

    @tool
    async def dnd_set_gold(amount: int) -> str:
        """Set the player's exact gold/currency total — prefer
        dnd_add_gold for a gain/loss (buying, looting, paying), use this
        only when you need to set an absolute value directly (e.g.
        character creation's starting gold). Never negative — a value below
        0 is saved as 0."""
        final = max(0, amount)
        store.set_gold(game_id, final)
        return f"Gold set to {final}."

    @tool
    async def dnd_add_gold(delta: int) -> str:
        """Add (positive) or spend/lose (negative delta) gold — call this
        for EVERY transaction (purchases, loot, payment, bribes, gambling)
        so the total stays exact instead of you tracking it in your head
        across turns. Never goes below 0 — if delta would do that, it's
        clamped to 0 (the player can't end up with negative gold). Returns
        the resulting total — state it to the player so they always know
        exactly how much they have."""
        new_total = store.add_gold(game_id, delta)
        return f"Gold is now {new_total}."

    @tool
    async def dnd_add_party_member(name: str, description: str = "") -> str:
        """Add (or update the description of) a companion/NPC currently
        traveling with the player — call this whenever someone joins the
        party. Calling it again with the same name just updates that
        member's description instead of duplicating them."""
        name = name.strip()
        if not name:
            return "Error: name must not be blank."
        store.add_party_member(game_id, name, description)
        return f"Party member saved: {name!r}."

    @tool
    async def dnd_remove_party_member(name: str) -> str:
        """Remove a companion from the party — they left, died, or
        betrayed the group. Matches the name case-insensitively. Returns
        whether that name was actually in the party."""
        removed = store.remove_party_member(game_id, name)
        if not removed:
            return f"Not found in party: {name!r} — nothing removed."
        return f"Removed {name!r} from the party."

    @tool
    async def dnd_get_party() -> str:
        """List everyone currently traveling with the player. Call this
        when the player asks who's with them, or when you need to reference
        a companion by name/role and aren't sure who's actually still in
        the party right now."""
        party = store.get_party(game_id)
        if not party:
            return "The player is traveling alone — no party members."
        return "\n".join(
            f"- {p['name']}" + (f" — {p['description']}" if p["description"] else "")
            for p in party
        )

    @tool
    async def dnd_set_health_status(status: str) -> str:
        """Save the player's OVERALL health status in one short phrase
        (e.g. 'здоров', 'истощён', 'тяжело ранен', 'на грани смерти') —
        the quick-glance summary. For specific SIMULTANEOUS injuries (a cut
        on one arm AND a twisted ankle at the same time), use
        dnd_add_injury instead — this field is a single overall summary,
        not a list."""
        status = status.strip()
        if not status:
            return "Error: status must not be blank."
        store.set_health_status(game_id, status)
        return f"Health status saved: {status!r}."

    @tool
    async def dnd_add_injury(description: str, severity: str = "") -> str:
        """Record a specific injury (e.g. 'порез на левой руке', 'вывих
        правой ноги') — call this whenever the player takes damage that's
        specific enough to name, not for generic/vague harm. Multiple
        injuries can exist at once — this ADDS one, it doesn't replace
        others. severity is optional free text ('лёгкая', 'тяжёлая').
        Consider also calling dnd_set_health_status if this changes the
        overall picture."""
        description = description.strip()
        if not description:
            return "Error: description must not be blank."
        store.add_injury(game_id, description, severity)
        return f"Injury recorded: {description!r}" + (f" ({severity})" if severity else "") + "."

    @tool
    async def dnd_heal_injury(description: str) -> str:
        """Remove a specific injury once it's healed/treated — description
        must match exactly what was recorded (call dnd_get_injuries first
        if you're not sure of the exact wording). Consider also calling
        dnd_set_health_status if this changes the overall picture."""
        removed = store.heal_injury(game_id, description)
        if not removed:
            return f"Not found: {description!r} — check dnd_get_injuries for the exact wording."
        return f"Healed: {description!r}."

    @tool
    async def dnd_get_injuries() -> str:
        """List all of the player's current specific injuries (not the
        one-line overall status — see dnd_get_world_status for that).
        Call this before narrating a health-dependent scene, or when the
        player asks what's wrong with them, instead of guessing from
        memory."""
        injuries = store.get_injuries(game_id)
        if not injuries:
            return "No specific injuries recorded."
        return "\n".join(
            f"- {i['description']}" + (f" ({i['severity']})" if i["severity"] else "")
            for i in injuries
        )

    @tool
    async def dnd_set_level(level: int) -> str:
        """Set the player character's level DIRECTLY, bypassing XP —
        ONLY for character creation (always 1 there) or a genuine
        narrative exception (a cursed de-leveling, a magic item that
        grants power outright). For normal progression, use dnd_add_xp
        instead — it's the actual leveling system; calling this after
        character creation for an ordinary level-up SKIPS that system
        and will get silently overwritten the next time dnd_add_xp
        recomputes the level from accumulated XP anyway."""
        if level < 1:
            return "Error: level must be at least 1."
        store.set_level(game_id, level)
        return f"Level set to {level}."

    @tool
    async def dnd_add_xp(amount: int) -> str:
        """Grant experience for something the player actually accomplished
        — a resolved encounter, a completed quest/chapter, a clever
        solution to a real problem. This is the ONLY way the character
        levels up during normal play (dnd_set_level is for character
        creation/narrative exceptions only, see its own docstring) — the
        level number is entirely DERIVED from accumulated XP by a fixed
        formula, not something you set directly, so use this instead of
        deciding "they level up now" yourself.

        Rough calibration (tune to how big a deal the moment actually
        was, don't just default to the same number every time):
        - a minor task/skill check resolved: 10-30 XP
        - a real fight or moderate challenge won: 50-150 XP
        - a chapter-closing accomplishment (dnd_end_chapter territory) or a
          major fight against a serious threat: 200-500 XP
        Amount must be positive — there's no dnd_remove_xp, XP only goes up.

        Tells you exactly how many levels (if any) this crossed and the
        new level/XP total — narrate a level-up ONLY when this reports
        levels_gained > 0, never preemptively."""
        if amount <= 0:
            return "Error: amount must be positive — XP only increases."
        result = store.add_xp(game_id, amount)
        if result["levels_gained"] > 0:
            return (
                f"+{amount} XP (total {result['xp']}). LEVEL UP: "
                f"{result['old_level']} -> {result['level']} "
                f"({result['levels_gained']} level(s) gained). "
                f"{result['xp_to_next_level']} XP to next level."
            )
        return (
            f"+{amount} XP (total {result['xp']}, still level {result['level']}). "
            f"{result['xp_to_next_level']} XP to next level."
        )

    @tool
    async def dnd_set_current_threat(description: str, level: int) -> str:
        """Record the enemy/challenge the player is CURRENTLY facing, with
        its own level for comparison against the player's — call this when
        you introduce a notable fight or serious obstacle, so both you and
        the player have an honest, persistent reference for how dangerous
        it actually is instead of it being reinterpreted turn to turn. Call
        dnd_clear_current_threat once it's resolved (defeated, fled,
        avoided) — don't leave a stale threat sitting here after the scene
        moves on."""
        description = description.strip()
        if not description:
            return "Error: description must not be blank."
        if level < 1:
            return "Error: level must be at least 1."
        store.set_current_threat(game_id, description, level)
        return f"Current threat set: {description!r} (level {level})."

    @tool
    async def dnd_clear_current_threat() -> str:
        """Clear the current threat once it's been resolved one way or
        another (defeated, fled, avoided, negotiated away)."""
        store.clear_current_threat(game_id)
        return "Current threat cleared."

    @tool
    async def dnd_roll_check(action: str, task_level: int) -> str:
        """Roll the dice on an uncertain or risky action instead of just
        deciding yourself whether it works — use this for EXACTLY the
        situations covered by "don't take the player's word for an
        outcome" in your instructions: a contested action against an
        established threat (task_level = that threat's level), or the
        player attempting a feat that may be wildly beyond their own
        level/class (task_level = your honest estimate of what level
        would find this reliable/routine).

        Calibrate task_level on the same scale as player level: roughly
        equal to the player's own level means an even, uncertain coin
        flip; a few levels above means hard but doable; a HUGE gap (e.g.
        a level-1 character attempting something you'd peg at task_level
        200+, like reshaping a continent) should round down to
        essentially impossible — don't be shy about picking a very high
        task_level for a genuinely absurd, setting-breaking claim, that's
        exactly what makes the odds collapse correctly.

        The actual roll is a real random number generated here, not your
        own judgment — you don't decide or influence which outcome tier
        you get, only how far apart the two levels are. Narrate whatever
        tier comes back; do not re-roll, override it, or narrate a
        different tier than what's returned.

        A SUCCESS/CRITICAL SUCCESS also grants XP automatically (scaled to
        how hard the task was relative to the player's level) — this
        happens here, you don't need to separately call dnd_add_xp for it;
        that tool is still the right one for a bigger accomplishment this
        specific check was only part of (e.g. the fight it just won)."""
        action = action.strip()
        if not action:
            return "Error: action must not be blank."
        if task_level < 1:
            return "Error: task_level must be at least 1."
        game = store.get_game(game_id)
        player_level = game["level"] if game else 1
        p = _check_probability(player_level, task_level)
        roll = random.random()
        xp_note = ""
        if roll < p * 0.1:
            tier = "CRITICAL SUCCESS"
            guidance = (
                "it doesn't just work — it exceeds what was even attempted. Narrate an "
                "impressive, over-the-top success."
            )
            xp_gain = _roll_xp_gain(player_level, task_level, critical=True)
        elif roll < p:
            tier = "SUCCESS"
            guidance = (
                "it works, but scaled to what the character's ACTUAL level can support — "
                "not the grand version they described, a proportionate one."
            )
            xp_gain = _roll_xp_gain(player_level, task_level, critical=False)
        elif roll > 1 - (1 - p) * 0.1:
            tier = "CRITICAL FAILURE"
            guidance = (
                "it fails badly with a real negative consequence — don't just have it "
                "fizzle quietly, something should actually go wrong (backfire, draw "
                "attention, cost something, injure them — pick what fits the action)."
            )
            xp_gain = 0
        else:
            tier = "FAILURE"
            guidance = "it simply doesn't work — no progress, no side effect, the attempt just fails."
            xp_gain = 0
        if xp_gain > 0:
            xp_result = store.add_xp(game_id, xp_gain)
            if xp_result["levels_gained"] > 0:
                xp_note = (
                    f" +{xp_gain} XP (total {xp_result['xp']}). LEVEL UP: "
                    f"{xp_result['old_level']} -> {xp_result['level']} "
                    f"({xp_result['levels_gained']} level(s) gained). "
                    f"{xp_result['xp_to_next_level']} XP to next level. "
                    "Narrate this level-up too."
                )
            else:
                xp_note = f" +{xp_gain} XP (total {xp_result['xp']}, still level {xp_result['level']})."
        return (
            f"{tier} — player level {player_level} vs task level {task_level} "
            f"(success chance was {p * 100:.6g}%). Narrate accordingly: {guidance}{xp_note} "
            "This result is final — do not re-roll or narrate a different outcome."
        )

    @tool
    async def dnd_remember_fact(fact: str) -> str:
        """Record ONE concrete fact worth remembering for the rest of this
        game — a decision made, a promise given, a secret learned, an NPC's
        fate, a quest accepted. One short, self-contained sentence per call
        (call it multiple times for multiple facts, don't cram several
        unrelated facts into one). You decide what's worth it — this is the
        ONLY way anything you don't repeat in your own next reply survives
        to later turns (there is no separate automatic memory), so if a
        detail matters for later and you're not about to restate it
        yourself, save it here now, don't wait."""
        fact = fact.strip()
        if not fact:
            return "Error: fact must not be blank."
        store.remember_fact(game_id, fact)
        return f"Remembered: {fact!r}."

    @tool
    async def dnd_get_facts(limit: int = 30) -> str:
        """List remembered facts for this game, oldest first. The most
        recent ones are already shown to you automatically at the start of
        each turn (see your context) — call this only when you need MORE
        history than that, or the player explicitly asks to be reminded of
        something older."""
        facts = store.get_facts(game_id, limit=max(1, min(limit, 200)))
        if not facts:
            return "No facts recorded yet."
        return "\n".join(f"- {f}" for f in facts)

    @tool
    async def dnd_end_chapter(summary: str) -> str:
        """Close out the current story arc/quest — call this when the
        player reaches a natural conclusion (finished the dungeon, resolved
        the quest, left the area for good), NOT after every scene. summary
        is a few sentences covering what was accomplished and where things
        stand now — it's saved as a permanent fact, and the live
        conversation is reset right after this call to keep it from growing
        forever; the NEXT turn starts fresh, working from the saved game
        state (location/inventory/gold/party/level/facts, including this
        summary) instead of the full raw transcript of the chapter that
        just ended. Make sure location/inventory/gold/party/health are
        already accurate (call the matching tools first if anything from
        this chapter never got saved) — once the conversation resets,
        anything not in the DB is gone for good, only what you say here in
        summary (plus everything already saved) carries forward."""
        summary = summary.strip()
        if not summary:
            return "Error: summary must not be blank."
        store.remember_fact(game_id, f"[Chapter closed] {summary}")
        return f"Chapter closed: {summary!r}. Conversation resets after this turn."

    return [
        dnd_set_character, dnd_update_location, dnd_update_calendar, dnd_update_weather,
        dnd_get_world_status,
        dnd_add_inventory_item, dnd_remove_inventory_item, dnd_get_inventory,
        dnd_equip_item, dnd_unequip_item, dnd_get_equipment,
        dnd_set_gold, dnd_add_gold,
        dnd_add_party_member, dnd_remove_party_member, dnd_get_party,
        dnd_set_level, dnd_add_xp, dnd_set_current_threat, dnd_clear_current_threat, dnd_roll_check,
        dnd_set_health_status, dnd_add_injury, dnd_heal_injury, dnd_get_injuries,
        dnd_remember_fact, dnd_get_facts, dnd_end_chapter,
    ]
