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
(Content-Length) + request/response по id + одна notification
(didOpen/didClose) — самодостаточно и не тянет лишних зависимостей.

Языковой сервер должен быть установлен и замаплен по расширению файла (см.
_server_command). На старте это:
  .py                → pylsp (python-lsp-server, requirements.txt)
  .go                → gopls (go install, уже стоит в системе)
  .ts/.tsx/.js/.jsx  → typescript-language-server --stdio (npm -g,
                       tsserver умеет и TS, и JS одним процессом — отдельного
                       JS-сервера не заводим, конфликтовать нечему)
  .php               → phpactor language-server (composer global require
                       phpactor/phpactor — не intelephense: у intelephense
                       findReferences/rename заперты за платной лицензией,
                       phpactor полностью открытый и бесплатный)
Для незамапленного расширения или отсутствующего бинарника тул возвращает
ошибку, а не падает молча — то же поведение, что у настоящего LSP-тула.

Один языковой сервер на (команда, корень репозитория) поднимается лениво
при первом обращении и живёт до конца процесса — переинициализация
(indexing) стоит секунды, гонять её на каждый вызов нельзя.

Запуск: python3 -m mcp_agent.servers.lsp_server
"""
import asyncio
import atexit
import itertools
import json
import os
import shutil
import sys
import urllib.parse

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("lsp")

_INIT_TIMEOUT = 20
_REQUEST_TIMEOUT = 15

_LANGUAGE_IDS = {
    ".py": "python",
    ".go": "go",
    ".ts": "typescript",
    ".tsx": "typescriptreact",
    ".js": "javascript",
    ".jsx": "javascriptreact",
    ".php": "php",
}

_COMPOSER_GLOBAL_BIN = os.path.expanduser("~/.config/composer/vendor/bin")

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

_child_procs: list[asyncio.subprocess.Process] = []


@atexit.register
def _cleanup_children():
    # Точечный terminate() по каждому известному процессу, а не killpg —
    # тот же принцип, что и в bash_server.py: осиротевший процесс —
    # меньшая беда, чем случайно затронуть что-то за пределами этого
    # дерева. atexit синхронный, await здесь невозможен — best-effort.
    for proc in _child_procs:
        if proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass


def _pylsp_path() -> str | None:
    path = os.path.join(os.path.dirname(sys.executable), "pylsp")
    return path if os.path.exists(path) else None


def _phpactor_path() -> str | None:
    found = shutil.which("phpactor")
    if found:
        return found
    local = os.path.join(_COMPOSER_GLOBAL_BIN, "phpactor")
    return local if os.path.exists(local) else None


def _server_command(ext: str) -> list[str] | None:
    if ext == ".py":
        pylsp = _pylsp_path()
        return [pylsp] if pylsp else None
    if ext == ".go":
        gopls = shutil.which("gopls")
        return [gopls] if gopls else None
    if ext in (".ts", ".tsx", ".js", ".jsx"):
        tsserver = shutil.which("typescript-language-server")
        return [tsserver, "--stdio"] if tsserver else None
    if ext == ".php":
        phpactor = _phpactor_path()
        return [phpactor, "language-server"] if phpactor else None
    return None


def _uri(path: str) -> str:
    return "file://" + urllib.parse.quote(os.path.abspath(path))


def _path_from_uri(uri: str) -> str:
    return urllib.parse.unquote(uri[len("file://"):]) if uri.startswith("file://") else uri


def _find_root(path: str) -> str:
    cur = os.path.dirname(os.path.abspath(path)) or "."
    start = cur
    while True:
        if os.path.isdir(os.path.join(cur, ".git")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return start
        cur = parent


class LSPClient:
    def __init__(self, cmd: list[str], root: str):
        self.cmd = cmd
        self.root = root
        self.proc: asyncio.subprocess.Process | None = None
        self._id_counter = itertools.count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self._opened: dict[str, int] = {}  # uri -> hash(content)
        self._call_lock = asyncio.Lock()
        self.server_capabilities: dict = {}

    async def start(self):
        self.proc = await asyncio.create_subprocess_exec(
            *self.cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        _child_procs.append(self.proc)
        asyncio.create_task(self._read_loop())
        result = await self._request("initialize", {
            "processId": os.getpid(),
            "rootUri": _uri(self.root),
            "capabilities": {
                "textDocument": {
                    "hover": {"contentFormat": ["plaintext", "markdown"]},
                    "definition": {},
                    "references": {},
                    "implementation": {},
                    "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                    "callHierarchy": {},
                },
                "workspace": {"symbol": {}},
            },
        }, timeout=_INIT_TIMEOUT)
        # Заявленные capabilities реально отличаются по серверам (живой
        # прогон: pylsp вообще не умеет workspace/symbol и callHierarchy —
        # запрос падает "Method Not Found" вместо пустого результата, а
        # phpactor репортит documentSymbolProvider как [] вместо true/объекта
        # при том, что метод у него РАБОТАЕТ). Сохраняем как есть, проверка
        # на поддержку — по ключу "объявлено вообще, не None/False", не по
        # truthiness значения, см. _supports().
        self.server_capabilities = (result or {}).get("capabilities", {})
        self._notify("initialized", {})

    async def _read_loop(self):
        reader = self.proc.stdout
        try:
            while True:
                headers = {}
                while True:
                    line = await reader.readline()
                    if not line or line in (b"\r\n", b"\n"):
                        break
                    key, _, value = line.decode("ascii", errors="replace").partition(":")
                    headers[key.strip().lower()] = value.strip()
                if not line:
                    break
                length = int(headers.get("content-length", 0) or 0)
                if length <= 0:
                    continue
                body = await reader.readexactly(length)
                try:
                    msg = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if isinstance(msg.get("id"), int) and ("result" in msg or "error" in msg):
                    fut = self._pending.pop(msg["id"], None)
                    if fut and not fut.done():
                        fut.set_result(msg)
                # Notifications (diagnostics, log messages, window/showMessage
                # и т.п.) нам не нужны — тул только читает, не подписывается
                # на изменения состояния.
        except (asyncio.IncompleteReadError, ValueError, ConnectionResetError):
            pass
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(RuntimeError("language server exited"))

    def _write(self, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
        self.proc.stdin.write(header + data)

    def _notify(self, method: str, params: dict):
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def _request(self, method: str, params: dict, timeout: float = _REQUEST_TIMEOUT):
        msg_id = next(self._id_counter)
        fut = asyncio.get_event_loop().create_future()
        self._pending[msg_id] = fut
        self._write({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
        await self.proc.stdin.drain()
        try:
            msg = await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(msg_id, None)
        if "error" in msg:
            raise RuntimeError(msg["error"].get("message", str(msg["error"])))
        return msg.get("result")

    async def request(self, method: str, params: dict, timeout: float = _REQUEST_TIMEOUT):
        # Один in-flight запрос за раз на сервер — упрощает жизнь придирчивым
        # реализациям (pylsp последовательно обрабатывает stdin), нагрузка
        # здесь не настолько высокая, чтобы это было узким местом.
        async with self._call_lock:
            return await self._request(method, params, timeout=timeout)

    async def ensure_open(self, path: str) -> str:
        uri = _uri(path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        digest = hash(text)
        if self._opened.get(uri) == digest:
            return uri
        if uri in self._opened:
            self._notify("textDocument/didClose", {"textDocument": {"uri": uri}})
        ext = os.path.splitext(path)[1]
        self._notify("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": _LANGUAGE_IDS.get(ext, "plaintext"),
                "version": 1,
                "text": text,
            },
        })
        self._opened[uri] = digest
        return uri


_clients: dict[tuple, LSPClient] = {}
_clients_lock = asyncio.Lock()


async def _get_client(path: str) -> LSPClient:
    ext = os.path.splitext(path)[1]
    cmd = _server_command(ext)
    if not cmd:
        raise RuntimeError(
            f"No LSP server configured/installed for '{ext or path}' files "
            "(currently wired up: Python/pylsp, Go/gopls, TS+JS/"
            "typescript-language-server, PHP/phpactor)"
        )
    root = _find_root(path)
    key = (tuple(cmd), root)
    async with _clients_lock:
        client = _clients.get(key)
        if client is None:
            client = LSPClient(cmd, root)
            await client.start()
            _clients[key] = client
    return client


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
        uri = await client.ensure_open(filePath)
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
