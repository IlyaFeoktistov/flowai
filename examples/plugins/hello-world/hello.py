"""/hello command — see plugin.json's "commands" entry. Signature is fixed
by mcp_agent/plugins.py's docstring: (args: str, console) -> None | Awaitable.
Sync is fine — cli.py only awaits the result if it's awaitable."""


def run(args, console):
    name = args.strip() or "world"
    console.print(f"[bold cyan]👋 Hello, {name}![/]")
