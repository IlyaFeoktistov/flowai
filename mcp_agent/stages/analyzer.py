"""
Verdict/guidance для стадии Analyzer (mcp_agent/roles.py, mcp_agent/pipeline.py).

Analyzer — самая простая по вердикту роль: она не пишет код и не гоняет
bash, поэтому весь легаси self-heal-репертуар mcp_agent/agent.py
(execution_failure, syntax_only_verification, diff-review и т.д.) ей
попросту не о чем проверять — единственное, что имеет смысл: "было ли
вообще расследование" и "есть ли внятное саммари по итогу". Не нужен даже
LLM-судья (self_heal.py:_semantic_check) — то, что проверяется, полностью
решается по структуре tool_messages/тексту, без семантики."""


def analyzer_verdict(round_msgs: list, new_tool_msgs: list, round_final_text: str) -> dict:
    if not round_final_text.strip():
        return {"relevant": False, "reason": "no final summary text produced this round"}
    if not new_tool_msgs:
        # Пустой раунд без единого тула — валиден ТОЛЬКО если модель прямо
        # заявляет, что расследовать нечего (редкий, но легитимный случай,
        # например когда весь ответ уже был в auto-inject'нутом knowledge).
        # Отличить эту фразу от "просто разговорился вместо разведки"
        # детерминированно нельзя — полагаемся на длину/конкретность текста
        # как дешёвый эвристический сигнал вместо отдельного judge-вызова.
        if len(round_final_text.strip()) > 40:
            return {"relevant": True, "reason": "no tools needed — answered from already-available context"}
        return {
            "relevant": False,
            "reason": "no read/search tools were called and the final text is too short to be a real conclusion — investigate before summarizing",
        }
    return {"relevant": True, "reason": "investigated with tools and produced a summary"}


def analyzer_guidance(verdict: dict, round_msgs: list, new_tool_msgs: list, round_final_text: str) -> str:
    if not new_tool_msgs:
        return (
            "You didn't call any read/search tools this round and your answer "
            "was too short to be a real conclusion. Investigate using your "
            "tools before writing a summary — or if you're genuinely confident "
            "nothing relevant exists, say so explicitly and explain why, with "
            "specifics (what you'd expect to find and where, and why it isn't "
            "there)."
        )
    return "Your summary was empty — write a concrete summary of what you found, citing file:line."
