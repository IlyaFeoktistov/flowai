"""Пример скила, который не просто печатает что-то в консоль (как
todo.py рядом), а запускает НАСТОЯЩИЙ ход агента — возвращает
mcp_agent/plugins.py:SkillTask вместо None. cli.py подхватывает такой
возврат и скармливает задачу в тот же самый пайплайн, что и обычное
сообщение пользователя (см. docstring SkillTask/discover_project_skills).

Демонстрирует оба новых поля SkillTask:
  - allowed_tools — на ЭТОТ ход разрешены только read-only тулы; попытка
    вызвать что-то ещё (bash, write_file, ...) будет мгновенно отклонена
    SkillToolRestrictionMiddleware, а не просто "по-хорошему запрещена
    в промпте".
  - prefer_delegate — мягкий намёк использовать delegate() для широкого
    расследования, если он вообще доступен в этой сессии (легаси-агент;
    в новом пайплайне delegate нет вообще, намёк просто ни на что не
    влияет — см. его собственный докстринг)."""
from mcp_agent.plugins import SkillTask

_READ_ONLY_TOOLS = frozenset({
    "read_file", "grep_search", "glob_search", "search_code_semantic",
    "lsp", "get_knowledge", "list_deleted_paths",
})


def run(args, console):
    focus = args.strip()
    scope = f", focusing on: {focus}" if focus else ""
    console.print(f"[bold cyan]🔍 ищу узкие места по производительности{scope}[/]")
    task = (
        "Find narrow, concrete performance bottlenecks in this codebase"
        f"{scope}.\n\n"
        "For each one: cite the exact file:line, explain what specifically "
        "is slow/inefficient (CPU, I/O, memory, contention) and why it "
        "matters — not a vague \"this could be optimized\". This is a "
        "READ-ONLY investigation: report findings only, do not propose or "
        "make any code changes."
    )
    return SkillTask(task=task, allowed_tools=_READ_ONLY_TOOLS, prefer_delegate=True)
