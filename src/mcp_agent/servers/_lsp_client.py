"""
Общая LSP-клиентская машинерия, вынесенная из lsp_server.py (навигация:
goToDefinition/findReferences/hover/...) — теперь используется ещё и
file_ops_server.py (диагностика после write_file/edit_file). Не сам
MCP-сервер, просто общий код.

Один языковой сервер на (команда, корень репозитория) поднимается лениво
при первом обращении и живёт до конца процесса своего вызывающего
(lsp_server.py или file_ops_server.py — это ДВА РАЗНЫХ процесса, каждый со
своим собственным пулом клиентов; общий модуль не даёт им шарить один и тот
же запущенный gopls/pylsp, только код).

Языковой сервер должен быть установлен и замаплен по расширению файла (см.
_server_command):
  .py                → pylsp (python-lsp-server, requirements.txt)
  .go                → gopls (go install, уже стоит в системе)
  .ts/.tsx/.js/.jsx  → typescript-language-server --stdio (npm -g,
                       tsserver умеет и TS, и JS одним процессом — отдельного
                       JS-сервера не заводим, конфликтовать нечему)
  .php               → phpactor language-server (composer global require
                       phpactor/phpactor — не intelephense: у intelephense
                       findReferences/rename заперты за платной лицензией,
                       phpactor полностью открытый и бесплатный)
Для незамапленного расширения или отсутствующего бинарника _get_client
бросает ошибку сразу, без запуска подпроцесса.
"""
import asyncio
import atexit
import itertools
import json
import os
import shutil
import sys
import urllib.parse

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
        self._versions: dict[str, int] = {}  # uri -> last didOpen/didChange version sent
        self._call_lock = asyncio.Lock()
        self.server_capabilities: dict = {}
        # uri -> latest textDocument/publishDiagnostics payload for it.
        self._diagnostics: dict[str, list[dict]] = {}
        # uri -> future resolved the next time a fresh publish for it lands
        # (registered by diagnostics_for BEFORE the triggering didOpen goes
        # out, so there's no window where the push could arrive unobserved).
        self._diag_waiters: dict[str, asyncio.Future] = {}

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
                    "publishDiagnostics": {},
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
        # truthiness значения, см. _supports() в lsp_server.py.
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
                elif msg.get("method") == "textDocument/publishDiagnostics":
                    params = msg.get("params") or {}
                    uri = params.get("uri")
                    if uri:
                        self._diagnostics[uri] = params.get("diagnostics") or []
                        waiter = self._diag_waiters.pop(uri, None)
                        if waiter and not waiter.done():
                            waiter.set_result(None)
                # Other notifications (log messages, window/showMessage,
                # etc.) still aren't needed by any caller — left dropped.
        except (asyncio.IncompleteReadError, ValueError, ConnectionResetError):
            pass
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(RuntimeError("language server exited"))
            for fut in self._diag_waiters.values():
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

    async def ensure_open(self, path: str) -> tuple[str, bool]:
        """Returns (uri, changed) — changed is True iff this call actually
        sent a fresh didOpen/didChange (new file, or content differs from
        what's already open), False if the already-open version is
        unchanged and no new diagnostics push is coming.

        Already-open + changed content goes through didChange (full-document
        sync, version bumped), NOT didClose+didOpen — a server is free to
        publish an empty "cleared" diagnostics list in response to didClose
        (LSP spec explicitly allows this), which would otherwise race with
        and falsely resolve diagnostics_for's wait for the REAL diagnostics
        of the reopened content."""
        uri = _uri(path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        digest = hash(text)
        if self._opened.get(uri) == digest:
            return uri, False
        if uri in self._opened:
            self._versions[uri] = self._versions.get(uri, 1) + 1
            self._notify("textDocument/didChange", {
                "textDocument": {"uri": uri, "version": self._versions[uri]},
                "contentChanges": [{"text": text}],
            })
        else:
            self._versions[uri] = 1
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
        return uri, True

    async def diagnostics_for(self, path: str, timeout: float = 4.0) -> list[dict] | None:
        """Diagnostics for `path`'s CURRENT on-disk content — freshly
        awaited from the server if that content isn't already open here,
        served instantly from the last known push otherwise. Returns None
        on timeout: "unknown", never treat that as "no issues"."""
        uri = _uri(path)
        waiter = self._diag_waiters.get(uri)
        if waiter is None or waiter.done():
            waiter = asyncio.get_event_loop().create_future()
            self._diag_waiters[uri] = waiter
        try:
            _, changed = await self.ensure_open(path)
        except Exception:
            self._diag_waiters.pop(uri, None)
            raise
        if not changed:
            self._diag_waiters.pop(uri, None)
            return self._diagnostics.get(uri, [])
        try:
            await asyncio.wait_for(waiter, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self._diag_waiters.pop(uri, None)
        return self._diagnostics.get(uri, [])


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
