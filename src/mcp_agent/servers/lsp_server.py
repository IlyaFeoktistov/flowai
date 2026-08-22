"""
Кастомный MCP-сервер: прокси к настоящему Language Server Protocol —
goToDefinition/findReferences/hover/documentSymbol/workspaceSymbol/
goToImplementation/prepareCallHierarchy/incomingCalls/outgoingCalls.

grep_search(pattern="def {query}"/"class {query}") — быстро и не требует
ничего, кроме rg/grep, но это угадайка по написанию. Она не отличает вызов
функции от переменной с тем же именем, не проследит импорт до реального
определения в другом файле, не покажет тип. lsp здесь общается с настоящим
языковым сервером (тем же протоколом, что использует любой нормальный
редактор) — результат семантически точный, а не текстовый.

Готовой Python-библиотеки с таким интерфейсом (per-language клиент +
единый generic API под произвольный сервер) не нашлось — есть
multilspy/pylsp-jsonrpc, но обе рассчитаны на другой сценарий
использования (embedding в конкретный tool, а не голый JSON-RPC поверх
stdio под наш MCP-контракт). Reinvent здесь минимальный: framing
(Content-Length) + request/response по id + notifications
(didOpen/didChange/publishDiagnostics) — самодостаточно и не тянет лишних
зависимостей.

Сама клиентская машинерия (LSPClient, запуск/маппинг серверов по
расширению, поднятие/переиспользование по (команда, корень репозитория))
живёт в _lsp_client.py — общий модуль, который использует ещё и
file_ops_server.py (диагностика после write_file/edit_file). Этот файл —
только форматирование под навигационные операции (goToDefinition/
findReferences/hover/...); за то, какие языковые сервера установлены и
почему, см. докстринг _lsp_client.py.

Для незамапленного расширения или отсутствующего бинарника тул возвращает
ошибку, а не падает молча — то же поведение, что у настоящего LSP-тула.

Запуск: python3 -m mcp_agent.servers.lsp_server
"""
import asyncio
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from mcp.server.fastmcp import FastMCP  # noqa: E402

from mcp_agent.servers._lsp_client import _get_client, _path_from_uri  # noqa: E402

mcp = FastMCP("lsp")

_OPS = (
    "goToDefinition", "findReferences", "hover", "documentSymbol",
    "workspaceSymbol", "goToImplementation", "prepareCallHierarchy",
    "incomingCalls", "outgoingCalls",
)

_OP_CAPABILITY = {
    "goToDefinition": "definitionProvider",
    "findReferences": "referencesProvider",
    "hover": "hoverProvider",
    "documentSymbol": "documentSymbolProvider",
    "workspaceSymbol": "workspaceSymbolProvider",
    "goToImplementation": "implementationProvider",
    "prepareCallHierarchy": "callHierarchyProvider",
    "incomingCalls": "callHierarchyProvider",
    "outgoingCalls": "callHierarchyProvider",
}


def _supports(capabilities: dict, operation: str) -> bool:
    key = _OP_CAPABILITY[operation]
    value = capabilities.get(key)
    return value is not None and value is not False

def _loc_to_str(uri: str, rng: dict) -> str:
    line = rng["start"]["line"] + 1
    char = rng["start"]["character"] + 1
    return f"{_path_from_uri(uri)}:{line}:{char}"


def _format_locations(result) -> str:
    if not result:
        return "No results"
    items = result if isinstance(result, list) else [result]
    lines = []
    for it in items:
        uri = it.get("uri") or it.get("targetUri")
        rng = it.get("range") or it.get("targetSelectionRange")
        if uri and rng:
            lines.append(_loc_to_str(uri, rng))
    return "\n".join(lines) if lines else "No results"


def _format_hover(result) -> str:
    if not result:
        return "No hover information"
    contents = result.get("contents")
    if isinstance(contents, dict):
        return contents.get("value", "") or "No hover information"
    if isinstance(contents, list):
        parts = [c.get("value", "") if isinstance(c, dict) else str(c) for c in contents]
        return "\n".join(p for p in parts if p) or "No hover information"
    return str(contents) if contents else "No hover information"


_KIND_NAMES = {
    1: "file", 2: "module", 3: "namespace", 4: "package", 5: "class",
    6: "method", 7: "property", 8: "field", 9: "constructor", 10: "enum",
    11: "interface", 12: "function", 13: "variable", 14: "constant",
    15: "string", 22: "struct",
}


def _format_symbols(result, fallback_uri: str | None = None) -> str:
    if not result:
        return "No symbols found"
    lines = []

    def walk(sym: dict, depth: int):
        name = sym.get("name", "?")
        kind = _KIND_NAMES.get(sym.get("kind"), str(sym.get("kind", "")))
        loc = sym.get("location") or {}
        uri = loc.get("uri") or fallback_uri
        rng = sym.get("selectionRange") or sym.get("range") or loc.get("range")
        pos = f" — {_loc_to_str(uri, rng)}" if uri and rng else ""
        lines.append(f"{'  ' * depth}{name} ({kind}){pos}")
        for child in sym.get("children") or []:
            walk(child, depth + 1)

    for sym in result:
        walk(sym, 0)
    return "\n".join(lines)


def _format_call_hierarchy(result) -> str:
    if not result:
        return "No results"
    lines = []
    for entry in result:
        item = entry.get("from") or entry.get("to") or entry
        name = item.get("name", "?")
        uri = item.get("uri")
        rng = item.get("selectionRange") or item.get("range")
        pos = f" — {_loc_to_str(uri, rng)}" if uri and rng else ""
        lines.append(f"{name}{pos}")
    return "\n".join(lines)


@mcp.tool()
async def lsp(operation: str, filePath: str, line: int = 1, character: int = 1, query: str = "") -> str:
    """Language Server Protocol code intelligence: goToDefinition,
    findReferences, hover, documentSymbol, workspaceSymbol,
    goToImplementation, prepareCallHierarchy, incomingCalls, outgoingCalls.

    Semantically exact — follows imports and type info via a real language
    server, unlike grep_search (text-based, language-agnostic but a guess by
    spelling). Use this when a grep_search for a name found a plausible
    match but you need the ACTUAL definition/callers, or when renaming/
    refactoring and you must find every real usage, not every string match.

    line/character are 1-based, as shown in editors/read_file output.
    workspaceSymbol ignores line/character but still needs a real filePath
    (only used to pick which language server to talk to) and requires a
    non-empty query — most servers return nothing for an empty one.

    Requires a language server installed for the file's language — errors
    out (rather than guessing) if none is configured. Currently wired up:
    Python (pylsp), Go (gopls), TypeScript/JavaScript
    (typescript-language-server), PHP (phpactor). Coverage differs per
    server — notably pylsp doesn't implement workspaceSymbol,
    goToImplementation, prepareCallHierarchy, incomingCalls or
    outgoingCalls for Python files; you'll get a clear error instead of
    empty results, don't retry those on .py files."""
    if operation not in _OPS:
        return f"Error: unknown operation '{operation}'. Valid: {', '.join(_OPS)}"
    if not os.path.isfile(filePath):
        return f"Error: file not found: {filePath}"

    try:
        client = await _get_client(filePath)
        uri, _ = await client.ensure_open(filePath)
    except Exception as e:
        return f"Error: {e}"

    if not _supports(client.server_capabilities, operation):
        return (
            f"Error: the language server for '{os.path.splitext(filePath)[1]}' "
            f"files doesn't support {operation}"
        )

    pos = {"line": max(0, line - 1), "character": max(0, character - 1)}
    text_doc = {"uri": uri}

    try:
        if operation == "workspaceSymbol":
            if not query.strip():
                return "Error: query is required for workspaceSymbol"
            result = await client.request("workspace/symbol", {"query": query})
            return _format_symbols(result)

        if operation == "documentSymbol":
            result = await client.request("textDocument/documentSymbol", {"textDocument": text_doc})
            return _format_symbols(result, fallback_uri=uri)

        if operation == "findReferences":
            result = await client.request("textDocument/references", {
                "textDocument": text_doc, "position": pos,
                "context": {"includeDeclaration": True},
            })
            return _format_locations(result)

        if operation in ("goToDefinition", "goToImplementation"):
            method = "textDocument/definition" if operation == "goToDefinition" else "textDocument/implementation"
            result = await client.request(method, {"textDocument": text_doc, "position": pos})
            return _format_locations(result)

        if operation == "hover":
            result = await client.request("textDocument/hover", {"textDocument": text_doc, "position": pos})
            return _format_hover(result)

        if operation == "prepareCallHierarchy":
            result = await client.request("textDocument/prepareCallHierarchy", {"textDocument": text_doc, "position": pos})
            return _format_call_hierarchy(result)

        # incomingCalls/outgoingCalls need a CallHierarchyItem, not a raw
        # position — prepare it transparently so the tool's own interface
        # stays (filePath, line, character) for every operation, same as
        # the reference LSP tool.
        items = await client.request("textDocument/prepareCallHierarchy", {"textDocument": text_doc, "position": pos})
        if not items:
            return "No callable symbol at this position"
        method = "callHierarchy/incomingCalls" if operation == "incomingCalls" else "callHierarchy/outgoingCalls"
        result = await client.request(method, {"item": items[0]})
        return _format_call_hierarchy(result)

    except asyncio.TimeoutError:
        return (
            "Error: language server timed out — a large project may still "
            "be indexing after startup, retry once in a few seconds instead "
            "of repeating the same call in a loop"
        )
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run()
