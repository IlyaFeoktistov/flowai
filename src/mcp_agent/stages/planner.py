"""
Verdict/guidance для стадии Planner (mcp_agent/roles.py, mcp_agent/pipeline.py).

Planner обязан закончить раунд настоящим вызовом ask_user
(mcp_agent/prompts.py:_planner_system_prompt требует этого явно, даже для
узких однозначных планов — простое да/нет подтверждение), поэтому
единственная содержательная проверка здесь: был ли ask_user реально
вызван, а не отфутболен текстовым вопросом. Случай "текстовый вопрос без
ask_user" mcp_agent/stage_runner.py:run_stage уже отдельно спасает
настоящим интерактивным диалогом ДО того, как дело доходит до этого
verdict_fn (см. его "punt-to-user rescue" ветку, сработает раньше) — сюда
попадают только случаи, где финальный текст не похож на вопрос вообще
(например, Planner просто рассказал план и остановился, не спросив)."""


def _called_ask_user(new_tool_msgs: list) -> bool:
    return any(m.name == "ask_user" for m in new_tool_msgs)


def planner_verdict(round_msgs: list, new_tool_msgs: list, round_final_text: str) -> dict:
    if _called_ask_user(new_tool_msgs):
        return {"relevant": True, "reason": "presented a plan and got the user's confirmation via ask_user"}
    return {
        "relevant": False,
        "reason": "the round ended without calling ask_user — Planner must always confirm its plan with the user before finishing",
    }


def planner_guidance(verdict: dict, round_msgs: list, new_tool_msgs: list, round_final_text: str) -> str:
    return (
        "You did not call ask_user this round. Don't investigate further "
        "unless you're missing one specific fact — you likely already have "
        "enough from the Analyzer's findings above. State your concrete plan "
        "(which file(s), what specific change, in what order) and call "
        "ask_user NOW, by itself, presenting that plan as `recommended` "
        "along with any real alternative you considered."
    )
