"""Пример per-project скила — /todo в ЭТОМ проекте (без манифеста: имя
файла минус ".py" становится именем команды, см. mcp_agent/plugins.py's
docstring про discover_project_skills). Сигнатура фиксирована —
run(args: str, console) -> None | Awaitable, та же, что и у плагинных
команд (examples/plugins/hello-world/hello.py) — разница только в том, где
файл лежит и что не нужен plugin.json."""


def run(args, console):
    text = args.strip()
    if not text:
        console.print("[yellow]Использование: /todo <текст>[/]")
        return
    with open("TODO.local.md", "a", encoding="utf-8") as f:
        f.write(f"- [ ] {text}\n")
    console.print(f"[bold cyan]📝 добавлено в TODO.local.md:[/] {text}")
