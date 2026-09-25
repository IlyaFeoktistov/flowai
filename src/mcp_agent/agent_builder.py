"""
Сборка LangGraph-агента: поднимает MCP-серверы (mcp_agent/config.py),
грузит и оборачивает их тулы (mcp_agent/tool_wrappers.py,
mcp_agent/snapshots.py, mcp_agent/ask_user_tool.py), создаёт основную и
judge-модели (Ollama) и собирает всё через create_agent.

Тулы/MCP-подпроцессы (_get_tools/_tools_cache) и сам агент/модель
(_get_agent/_agent_cache) кешируются РАЗДЕЛЬНО: поднимать MCP-серверы и
грузить схемы тулов заново на каждый ход было бы неоправданно дорого, а
модель при этом должна уметь пересобраться на лету при смене settings.
chat_model (voice_mode) без пересоздания подпроцессов — см. docstring
_get_agent/_get_tools ниже.
"""
import asyncio
import logging
import os
import re

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

import expert_streaming
import settings
from mcp_agent import ollama_kv_cache
from ui.console import console, debug_print
from mcp_agent.ask_user_tool import (
    _AskUserFinalizeMiddleware,
    _AskUserFinalizeNumPredictMiddleware,
    _AskUserGuardMiddleware,
    _OutOfProjectWriteApprovalMiddleware,
    _ToolErrorGuardMiddleware,
    ask_user,
    mark_plan_step_current,
    submit_plan,
)
from mcp_agent.build_cache import BuildCache
from mcp_agent.compaction import _CompactResearchMiddleware, _DropStaleReadsMiddleware
from mcp_agent.config import build_mcp_connections, TOOLS_REQUIRING_APPROVAL
from mcp_agent.debug_log import log_event
from mcp_agent import plan_bash

# MCP tools open a stdio session per call; on close the mcp library kills the
# server's process group, and when the server already exited on its own it
# logs a WARNING ("Process group termination failed ... No such process")
# straight into the chat — nothing failed, the process is simply gone. Real
# kill failures are logged at ERROR and still show.
logging.getLogger("mcp.os.posix.utilities").setLevel(logging.ERROR)
from mcp_agent.delegate_tool import _DelegateNudgeMiddleware, build_delegate_tool
from mcp_agent.web_read_tool import build_web_read_tool
from mcp_agent import skills as md_skills
from mcp_agent import work_mode
from mcp_agent.message_utils import _DedupeToolResultsMiddleware
from mcp_agent.optimized_tools import build_optimized_tools
from mcp_agent import plugins
from mcp_agent.plugin_hooks import PluginHookMiddleware
from mcp_agent.roles import approval_tools, filter_tools
from mcp_agent.model_config import (
    DEBUG,
    JUDGE_NUM_PREDICT,
    OLLAMA_KEEP_ALIVE,
    TOOL_OUTPUT_CHAR_CAP,
)
import model_params
from mcp_agent import prompts
from mcp_agent.prompts import _build_optimized_system_prompt, _build_system_prompt, _build_voice_system_prompt


# Длинный ход (например ask_user + несколько раундов правок + верификация)
# может забить весь num_ctx, и тогда llama.cpp делает "context shift" (в
# логе Ollama это видно как "slot context shift, n_keep=..., n_discard=..."),
# выбрасывая всё, кроме первых n_keep токенов промпта и последней части
# истории — если системный промпт и исходная задача не уместились в n_keep,
# они улетают целиком. Если context shift происходит ПОСЕРЕДИНЕ генерации
# финального ответа, ответ обрывается на полуслове; более ранняя правка в
# том же ходе может тоже оказаться бессмысленной, если она была сделана
# сразу после предыдущего context shift, уже на урезанном контексте.
#
# ChatOllama (langchain_ollama) не имеет поля num_keep вообще — оно не входит
# ни в список полей модели, ни в дефолтный options_dict, который _chat_params
# строит из self.*, так что просто передать num_keep=... в конструктор
# неоткуда. num_keep — реальный параметр Ollama (есть в ollama.Options), его
# просто нужно докинуть в params["options"] на уровне API-вызова. Подкласс
# вместо monkey-patch/.bind(options=...): .bind() кладёт лишний kwarg на
# верхний уровень запроса (см. комментарий у _extract_ask_user_shape в
# self_heal.py — тот же способ подводит и там), а не в options{}, и, что
# важнее, create_agent() ниже сам вызывает model.bind_tools(...) — обычный
# Runnable, возвращённый .bind(), этот метод не реализует. Подкласс остаётся
# настоящим ChatOllama (bind_tools у него есть), только один хук (_chat_params)
# дописывает num_keep в уже готовый options-словарь.
class _ChatOllamaWithNumKeep(ChatOllama):
    num_keep: int | None = None
    # min_p — same gap as num_keep: a real Ollama option that ChatOllama has
    # no field for (per-model setting, see model_params.py).
    min_p: float | None = None

    def _chat_params(self, messages, stop=None, **kwargs):
        params = super()._chat_params(messages, stop, **kwargs)
        if isinstance(params.get("options"), dict):
            if self.num_keep is not None:
                params["options"]["num_keep"] = self.num_keep
            if self.min_p is not None:
                params["options"]["min_p"] = self.min_p
        return params


# gpt-oss:20b раньше требовал expert_streaming_enabled как единственный
# способ обойти GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor)) на
# обычном Ollama-пути (известный баг Ollama,
# github.com/ollama/ollama/issues/16946) — на самом деле краш триггерит не
# сам gpt-oss, а OLLAMA_KV_CACHE_TYPE=q8_0 конкретно: тот же gpt-oss:20b на
# том же железе успешно отрабатывает несколько ходов подряд под f16, и
# падает на первой же загрузке сразу после смены на q8_0. ollama_kv_cache.py
# переключает это автоматически перед каждой сборкой модели — больше не
# нужно принудительно гнать gpt-oss через экспериментальный
# expert-streaming только чтобы обойти эту переменную окружения.


def _build_chat_model(
    *, model_tag: str, num_predict: int | None, reasoning: bool, num_keep: int, format: str | None = None,
    has_tools: bool = True,
):
    """Единая точка сборки и для основной, и для judge-модели (обоих
    вызывающих — _build_agent ниже и _build_role_agent, см. их обе) —
    settings.expert_streaming_enabled переключает backend ЗДЕСЬ, один раз,
    вместо развилки в каждом из 4 мест конструктора. См.
    expert_streaming.py's docstring за полным разбором: что за форк, откуда,
    почему НЕ смёржен в апстрим, и какой trade-off (PP заметно медленнее, TG
    в среднем на треть быстрее по данным автора PR) он приносит.

    ensure_running возвращает (False, reason) вместо исключения на любой
    сбой (бинарник не собран, блоб не нашёлся, сервер не поднялся за
    таймаут) — в этом случае тихо (с предупреждением в консоль) откатываемся
    на обычный ChatOllama, а не роняем весь ход: экспериментальный backend
    не должен быть single point of failure для обычной работы агента.

    Параметры, которые Ollama пересчитывает НА КАЖДЫЙ вызов (num_keep,
    reasoning/show_thinking через chat template), здесь фиксируются один раз
    ПРИ СТАРТЕ процесса llama-server (см. ensure_running) — если модель тег
    не меняется, процесс просто переиспользуется как есть, даже если
    num_keep/reasoning с прошлого вызова успели измениться. Это осознанное
    огрубление экспериментального пути, не забытый баг — см.
    expert_streaming.py docstring, раздел "известные огрубления".

    num_predict=None — the model's own configured num_predict (per-model,
    model_params.py); callers with a deliberately short budget (judge,
    casual, finalize) pass their own number instead.

    has_tools — one of the two signals (with `format`) deciding whether
    model_params' family defaults apply at all for this call — see
    apply_repeat_override's own comment below for the mechanism and the
    two bugs that shaped it (naming kept as "has_tools" even though
    the gate is really has_tools OR format=="json", to avoid renaming
    every call site's kwarg over what's just an internal detail)."""
    # apply_repeat_override gates the WHOLE family-defaults bundle
    # (model_params._FAMILY_DEFAULTS: temperature/top_p/min_p/repeat_penalty
    # together), not just repeat_penalty alone: these are not independent
    # knobs. The community-recommended values are only validated as ONE
    # bundle together with repeat_penalty=1.0 — restoring plain
    # REPEAT_PENALTY=1.2 while still applying temperature=0.7/top_p=0.95/
    # min_p=0.01 is a combination nobody has validated, and gating
    # repeat_penalty alone produces exactly that combination on the
    # casual/no-tools path. That untested combination causes full
    # incoherent breakdown — garbled mixed-language text, random code
    # snippets — not just dull repetition. So every code path uses either
    # the full family bundle or the plain base defaults, never an invented
    # third combination. Values the user set for this model in /settings
    # apply on both paths.
    apply_repeat_override = has_tools or format == "json"
    params = model_params.effective(model_tag, include_family=apply_repeat_override)
    if num_predict is None:
        num_predict = params["num_predict"]
    if settings.get("expert_streaming_enabled"):
        # The server's own --reasoning default follows only the user's
        # show_thinking; THIS call's thinking goes per request via
        # chat_template_kwargs.enable_thinking (server-common.cpp honors it,
        # GLM's template reads it). Passing `reasoning` as the server config
        # instead made the main model (show_thinking) and the judge
        # (reasoning=False) disagree on it, so every agent build restarted
        # llama-server and reloaded the weights twice.
        ok, msg = expert_streaming.ensure_running(
            model_tag, num_ctx=params["num_ctx"], show_thinking=bool(settings.get("show_thinking")),
            parallel=settings.get("parallel_slots") or 1, no_mmap=bool(params["no_mmap"]),
        )
        if ok:
            extra_body = {
                "top_k": params["top_k"],
                "repeat_penalty": params["repeat_penalty"],
                "repeat_last_n": params["repeat_last_n"],
                "chat_template_kwargs": {"enable_thinking": bool(reasoning)},
            }
            if params["min_p"] is not None:
                extra_body["min_p"] = params["min_p"]
            if format == "json":
                extra_body["response_format"] = {"type": "json_object"}
            return ChatOpenAI(
                model=model_tag,
                base_url=f"http://{expert_streaming.DEFAULT_HOST}:{expert_streaming.DEFAULT_PORT}/v1",
                api_key="not-needed",
                max_tokens=num_predict,
                temperature=params["temperature"],
                top_p=params["top_p"],
                extra_body=extra_body,
                # langchain_openai только автовключает stream_usage=True для
                # ДЕФОЛТНОГО openai.com base_url — на кастомном (наш
                # localhost:8090) оно молча остаётся off, и итоговый
                # AIMessageChunk.usage_metadata пустой на каждом стриминговом
                # вызове, из-за чего agent.py's tokens_in/tokens_out (и,
                # соответственно, "stats"-событие/футер в вебе) всегда
                # считались нулём для этой модели — не "модель не отдаёт
                # токены", а просто клиент их не запрашивал. Форкнутый
                # llama.cpp-сервер (vendor/llama-expert-streaming) честно
                # умеет stream_options.include_usage (server-schema.cpp),
                # достаточно было включить это на клиенте.
                stream_usage=True,
                # langchain_openai's own watchdog assumes a real
                # OpenAI-class endpoint (first token in low single-digit
                # seconds even for large prompts) and fires
                # stream_chunk_timeout ("No streaming chunk received for
                # 120.0s ...") otherwise. This backend's own measured
                # prompt-processing throughput is ~2-5 tok/s (see
                # expert_streaming.py docstring) — a
                # multi-thousand-token system prompt alone can take several
                # MINUTES before the first generated token appears, which is
                # not a dead connection, just a slow (but alive) local
                # process. Disabled entirely rather than raised to a bigger
                # number — no fixed timeout is safe across every possible
                # prompt length/history size this agent might build up.
                stream_chunk_timeout=None,
            )
        # format != "json" — только основная модель печатает предупреждение,
        # не judge_model: тот же model_tag, тот же ensure_running (expert_
        # streaming.py's _last_failure кэширует и не повторяет сам провальный
        # запуск, но без этой проверки предупреждение печаталось бы дважды
        # подряд для одного и того же провала).
        if format != "json":
            console.print(
                f"[yellow]⚠ expert-streaming backend недоступен ({msg}) — "
                "использую обычный Ollama-путь[/]"
            )

    # См. модульный комментарий выше про gpt-oss:20b/q8_0 — best-effort:
    # если sudoers-правило не настроено (см. README), просто предупреждаем
    # и продолжаем на текущей (возможно неподходящей для этой модели)
    # конфигурации, а не роняем весь ход.
    kv_ok, kv_msg = ollama_kv_cache.ensure_kv_cache_type(model_tag)
    if not kv_ok and format != "json":
        console.print(f"[yellow]⚠ не удалось выставить OLLAMA_KV_CACHE_TYPE ({kv_msg})[/]")

    kwargs = dict(
        model=model_tag,
        base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        keep_alive=OLLAMA_KEEP_ALIVE,
        num_ctx=params["num_ctx"],
        num_predict=num_predict,
        reasoning=reasoning,
        temperature=params["temperature"],
        top_p=params["top_p"],
        top_k=params["top_k"],
        repeat_penalty=params["repeat_penalty"],
        repeat_last_n=params["repeat_last_n"],
        num_keep=num_keep,
    )
    if params["min_p"] is not None:
        kwargs["min_p"] = params["min_p"]
    if format is not None:
        kwargs["format"] = format
    return _ChatOllamaWithNumKeep(**kwargs)
async def preload_chat_model(on_event=None) -> None:
    """Loads the current chat model BEFORE the agent is built, reporting
    progress as "model_loading"/"model_loaded" events. Without it the load
    happens inside _build_chat_model (llama-server: a synchronous
    ensure_running that holds the event loop for the whole ~30s+, so the UI
    spinner freezes) or silently inside the first Ollama request — either
    way the user only sees a stalled turn.

    llama.cpp gets a real percentage (expert_streaming._load_progress);
    Ollama exposes no load progress, so only "loading" + elapsed time.
    Best-effort: any failure here is left for the normal build path to
    report (it already falls back/warns on its own)."""
    import time
    model_tag = settings.get("chat_model")
    params = model_params.effective(model_tag)
    loop = asyncio.get_running_loop()
    started = time.monotonic()

    async def _emit(event: dict) -> None:
        if on_event is not None:
            await on_event(event)

    try:
        if settings.get("expert_streaming_enabled") and expert_streaming.is_built():
            thinking = bool(settings.get("show_thinking"))
            parallel = settings.get("parallel_slots") or 1
            if await asyncio.to_thread(expert_streaming.is_serving, model_tag, params["num_ctx"], thinking, parallel,
                                     no_mmap=bool(params["no_mmap"])):
                return
            await _emit({"type": "model_loading", "model": model_tag, "backend": "llama.cpp", "percent": None})
            last_pct: list[int | None] = [None]

            def _progress(fraction: float | None) -> None:
                pct = None if fraction is None else int(fraction * 100)
                if pct != last_pct[0]:
                    last_pct[0] = pct
                    asyncio.run_coroutine_threadsafe(_emit({
                        "type": "model_loading", "model": model_tag, "backend": "llama.cpp", "percent": pct,
                    }), loop)

            ok, _msg = await asyncio.to_thread(
                expert_streaming.ensure_running, model_tag,
                num_ctx=params["num_ctx"], show_thinking=thinking, on_progress=_progress, parallel=parallel,
                no_mmap=bool(params["no_mmap"]),
            )
            if ok:
                await _emit({"type": "model_loaded", "model": model_tag, "seconds": round(time.monotonic() - started, 1)})
            return

        import ollama
        client = ollama.AsyncClient(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        loaded = {m.model for m in (await client.ps()).models}
        if model_tag in loaded:
            return
        await asyncio.to_thread(ollama_kv_cache.ensure_kv_cache_type, model_tag)
        await _emit({"type": "model_loading", "model": model_tag, "backend": "ollama", "percent": None})
        # Empty prompt = load only. num_ctx must match what the real request
        # sends, otherwise Ollama reloads the model again on that request.
        await client.generate(
            model=model_tag, prompt="", keep_alive=OLLAMA_KEEP_ALIVE,
            options={"num_ctx": params["num_ctx"]},
        )
        await _emit({"type": "model_loaded", "model": model_tag, "seconds": round(time.monotonic() - started, 1)})
    except Exception as e:
        log_event("preload_chat_model_failed", model=model_tag, error=str(e))


from mcp_agent.snapshots import _snapshot_before_write, list_file_snapshots, restore_file_snapshot
from mcp_agent.tool_wrappers import (
    _add_regex_warning,
    _add_verify_reminder,
    _cap_tool_output,
    _dedupe_read_tool,
    _require_fresh_read_tool,
    _track_read_mtime_tool,
    _wrap_read_invalidation,
)

# Both caches below are mcp_agent.build_cache.BuildCache — see its module
# docstring for why this shape (get-or-build, rebuild when a "freshness"
# tuple changes) lives in one place instead of being reimplemented by hand
# per cache. _tools_cache is keyed by repo_path with a constant freshness
# (True) — it only ever changes via invalidate_tool_caches() below, never
# automatically, since raising MCP subprocesses is expensive and unrelated
# to which model is selected (see _get_tools's own docstring).
_tools_cache = BuildCache()

# Кешируется ОТДЕЛЬНО от тулов, вместе с (chat_model, voice_mode), на
# которых он собран (см. _get_agent) — voice_mode (settings.py:set_value)
# переключает chat_model между qwen3-coder:30b и qwen3:8b на лету, и агент
# должен реально пересобраться на новой модели, а не молча продолжать
# работать на старой до перезапуска процесса. voice_mode — ОТДЕЛЬНАЯ часть
# ключа, не только chat_model: если voice_chat_model случайно совпадает с
# обычной chat_model (пользователь сам так настроил), тег при переключении
# voice_mode не меняется вообще — без voice_mode в ключе кеш решил бы, что
# пересобирать нечего, и агент остался бы с пустым тулсетом/голосовым
# промптом (или наоборот) до следующей смены МОДЕЛИ, а не режима. Single
# slot (key=None) — there's only ever one main monolith agent per process.
_agent_cache = BuildCache()


async def _load_tools_resilient(client: MultiServerMCPClient, server_names: list[str], plugin_server_names: frozenset[str] = frozenset()) -> tuple[list, frozenset[str]]:
    """client.get_tools() без server_name гребёт ВСЕ сервера через один
    asyncio.gather() без return_exceptions — если хотя бы один не
    поднимается (например npx недоступен), падает вся пачка и агент
    остаётся без единого тула. Грузим по серверам ОТДЕЛЬНЫМИ get_tools()
    вызовами — один сбойный сервер лишает нас только своих тулов, а не
    всех остальных — но всё равно ПАРАЛЛЕЛЬНО через свой asyncio.gather с
    return_exceptions=True: изоляция сбоя не требует последовательности,
    9-12 независимых подпроцессов (fetch/bash/web_search/memory/knowledge/
    rag/file_ops/vision/lsp/+gen*) не имеют друг с другом никакой data
    dependency — раньше каждый спавн+stdio-handshake+schema-fetch ждал
    предыдущий целиком, хотя мог идти рядом с ним.

    Возвращает (tools, plugin_tool_names) — второе — имена тулов, чьи
    сервера пришли от плагинов (mcp_agent/plugins.py). roles.py's
    composer-функции — статичные allowlist'ы ПО ИМЕНИ, собранные ДО того,
    как известно, какие тулы вообще предоставит установленный плагин, так
    что имя плагинского тула физически не может попасть ни в один из этих
    списков — без plugin_tool_names _build_role_agent не смог бы отличить
    "тул плагина, пропустить в любую роль" от "незнакомое имя, тул кто-то
    забыл добавить в allowlist" и был бы вынужден выбрать одно поведение
    для обоих случаев."""
    results = await asyncio.gather(
        *(client.get_tools(server_name=name) for name in server_names),
        return_exceptions=True,
    )
    tools = []
    plugin_tool_names = set()
    for name, result in zip(server_names, results):
        if isinstance(result, Exception):
            console.print(f"[yellow]⚠ MCP-сервер '{name}' не запустился — его инструменты недоступны: {result}[/]")
        else:
            tools.extend(result)
            if name in plugin_server_names:
                plugin_tool_names.update(t.name for t in result)
    return tools, frozenset(plugin_tool_names)


async def _build_tools(repo_path: str | None = None):
    resolved_repo_path = repo_path or os.getcwd()
    connections = build_mcp_connections(resolved_repo_path)
    client = MultiServerMCPClient(connections)
    plugin_server_names = frozenset(plugins.load_mcp_servers().keys())
    tools, plugin_tool_names = await _load_tools_resilient(client, list(connections.keys()), plugin_server_names)
    tools = [_cap_tool_output(t, TOOL_OUTPUT_CHAR_CAP) for t in tools]
    # Snapshot of the UNWRAPPED read_file, taken before _dedupe_read_tool
    # wraps it below with THIS role's read_history — build_delegate_tool
    # (delegate_tool.py) needs to wrap its own copy with an INDEPENDENT
    # read_history, not share this one. delegate runs as a fresh, isolated
    # sub-agent conversation; if it shared this dict, an EARLIER read the
    # outer agent already did (before delegate was ever called) would make
    # delegate's own first read of that same path hit "(You already read
    # `path`... reuse that earlier result)" — a stub telling it to reuse
    # content it never actually saw, since that read happened in a
    # DIFFERENT conversation. The model then has nothing real to reuse and
    # answers from nothing, reading as confident analysis of a file it
    # never actually looked at (live-run: delegate reported analysis that
    # didn't match the file's actual current content).
    raw_read_file_tool = next((t for t in tools if t.name == "read_file"), None)
    # read_history — {path: [key, ...]} для read_file, очищается в начале
    # каждого stream_chat (см. там же) и на каждом self-heal retry.
    read_history: dict = {}
    # read_mtimes — {path: mtime}, живёт весь срок жизни ЭТОГО _build_tools
    # (т.е. пока не инвалидирован _tools_cache) — в отличие от read_history
    # НЕ очищается по ходам, см. _track_read_mtime_tool/_require_fresh_read_tool
    # (tool_wrappers.py) за тем, почему этот трекинг живёт здесь, а не внутри
    # file_ops_server.py.
    read_mtimes: dict = {}
    tools = [_track_read_mtime_tool(t, read_mtimes) if t.name == "read_file" else t for t in tools]
    # For sub-agents that may write (delegate_tool.py): their own read_file
    # must still feed read_mtimes, or their write_file/edit_file would be
    # refused as "not read yet". Dedupe is added per sub-agent run there.
    tracked_read_file_tool = next((t for t in tools if t.name == "read_file"), None)
    tools = [_dedupe_read_tool(t, read_history) if t.name == "read_file" else t for t in tools]
    tools = [_require_fresh_read_tool(t, read_mtimes) if t.name in ("write_file", "edit_file") else t for t in tools]
    tools = [_wrap_read_invalidation(t, read_history) for t in tools]
    tools = [_add_verify_reminder(t) if t.name in ("write_file", "edit_file") else t for t in tools]
    tools = [_add_regex_warning(t) if t.name == "grep_search" else t for t in tools]
    # Снимок содержимого файла ДО мутации — outermost-обёртка, чтобы
    # захватить состояние прямо перед реальным изменением (см.
    # _snapshot_before_write). Даёт restore_file_snapshot точки возврата,
    # которых нет в git-истории (несколько незакоммиченных правок подряд).
    tools = [
        _snapshot_before_write(t, resolved_repo_path, path_key="path")
        if t.name in ("write_file", "edit_file") else t
        for t in tools
    ]
    # ask_user — не MCP-тул: ему нужен прямой доступ к TUI (tools/confirm.py:
    # ask_user_question), которого у отдельного subprocess-сервера нет.
    # list_file_snapshots/restore_file_snapshot — тоже не MCP-тулы (общее
    # sqlite-хранилище снимков, см. _save_file_snapshot), restore_file_snapshot
    # оборачиваем в инвалидацию read-кэша отдельно — он добавляется уже ПОСЛЕ
    # общего прохода _wrap_read_invalidation по остальным тулам выше.
    tools.append(ask_user)
    tools.append(mark_plan_step_current)
    tools.append(submit_plan)
    tools.append(list_file_snapshots)
    tools.append(_wrap_read_invalidation(restore_file_snapshot, read_history))
    # Для _execute_leaked_tool_call (см. выше) — те же самые объекты тулов,
    # что видит create_agent ниже (уже с _cap_tool_output/_bind_constant_args
    # обёртками), просто доступные по имени напрямую, в обход графа.
    tools_by_name = {t.name: t for t in tools}

    if DEBUG:
        debug_print(f"[dim][MCP-AGENT] Loaded {len(tools)} tools: {[t.name for t in tools]}[/]")
    log_event("tools_loaded", names=[t.name for t in tools])

    return tools, tools_by_name, read_history, resolved_repo_path, plugin_tool_names, raw_read_file_tool, tracked_read_file_tool


async def _get_tools(repo_path: str | None = None):
    """_build_tools() spawns several MCP server subprocesses (mcp-server-git,
    mcp-server-fetch, our own python servers including file_ops_server.py)
    and loads their tool schemas — независимо от того, какая chat_model
    выбрана. Кешируется
    ОТДЕЛЬНО от модели (см. _get_agent) — переключение chat_model (voice_mode
    ON/OFF) не должно заново поднимать все MCP-подпроцессы, это дорогая и
    никак не связанная с выбором модели часть.

    Ключ — resolved repo_path, а не просто "было — не важно, какое". Раньше
    это был единственный global-слот без ключа вообще: первый repo_path,
    на котором собрались тулы, оставался в силе НАВСЕГДА для всего
    процесса, даже когда следующий вызов приходил с другим repo_path (у
    пайплайна repo_path = os.getcwd() пересчитывается на каждый ход, см.
    pipeline.py) — file_ops/git-серверы у ВСЕХ последующих
    ролей молча продолжали бы работать в первом попавшемся проекте. Ключ по
    repo_path чинит это заранее, до того как ударит на реальной смене
    проекта в рамках одного процесса."""
    key = repo_path or os.getcwd()
    return await _tools_cache.get_or_build(key, True, lambda: _build_tools(repo_path))


_UNLOADABLE_SUBPROCESS_TOOLS = ("unload_image_gen_model", "unload_music_gen_model")


async def _unload_subprocess_models() -> list[str]:
    """Calls the unload_* tools on image_gen_server/music_server — the MCP
    subprocesses backing the AGENT's own generate_image/generate_music
    tool calls (see model_lifecycle.py — separate from that module's direct
    in-process unloads of tools/image_gen.py's /gen pipe and
    music_server's /music-streaming copy, which live in THIS process, not
    a subprocess). Deliberately reads the module-level _tools_cache
    directly instead of calling _get_tools() — that would spawn all 8 MCP
    subprocesses on demand, which is far more wasteful than the memory
    we're trying to free if the agent was never actually used this
    session. Returns [] with no subprocess call at all when _tools_cache
    is still empty. _tools_cache is now keyed by repo_path (see
    _get_tools) — image_gen/music_gen subprocesses don't actually vary per
    project, but nothing stops a second repo_path from having spun up its
    own copy, so unload every cached entry, not just "the" one."""
    if not _tools_cache:
        return []
    freed = []
    for _, tools_by_name, _, _ in _tools_cache.values():
        for tool_name in _UNLOADABLE_SUBPROCESS_TOOLS:
            tool = tools_by_name.get(tool_name)
            if tool is None:
                continue
            try:
                result = await tool.ainvoke({})
            except Exception:
                continue
            text = result[0] if isinstance(result, tuple) else result
            if isinstance(text, str) and not text.lower().startswith("nothing to unload"):
                freed.append(text)
    return freed


_GEN_MODEL_GPU_TOOLS = ("generate_3d_model", "generate_texture_for_model")


class _UnloadImageGenBeforeGenModelMiddleware(AgentMiddleware):
    """generate_3d_model/generate_texture_for_model (gen_model_server.py, its
    own MCP subprocess) launch a Hunyuan3D-2GP subprocess that wants the
    whole card to itself. gen3d/pipeline.py's own _free_gpu_for_subprocess
    already frees Ollama's chat model (reachable anywhere via its HTTP API)
    and THIS process's own /gen pipe (tools/image_gen.py, when it's the
    CLI-direct /gen_model path calling in-process) -- but it can't reach
    image_gen_server.py's resident SDXL/FLUX pipe, since that's a SEPARATE
    MCP subprocess with no direct IPC to gen_model_server.py's subprocess.
    If the agent generated a reference image itself earlier in this same
    conversation (its own generate_image tool call), that pipe is exactly
    what's left resident there. THIS process (agent_builder.py) already
    holds the MCP connection to image_gen_server.py, so unload it from here,
    right before dispatching either tool -- reusing _unload_subprocess_models's
    existing tool.ainvoke bridge (we're already in the agent's own async
    world in awrap_tool_call, no _run_coro_blocking thread-bridge needed --
    that one's only for model_lifecycle.py's sync curses-menu callers)."""

    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] in _GEN_MODEL_GPU_TOOLS:
            await _unload_subprocess_models()
        return await handler(request)


# In-place file mutation idioms bash could use to "fix" something itself
# instead of just checking it — sed/perl/awk in-place edit flags, tee, and
# the classic file-mutating coreutils/git commands. Dangerous regardless of
# WHERE they act (git mutations touch repo state; mv/cp/rm/chmod/chown/
# truncate/dd are destructive enough, and rare enough for a genuine "just
# check" task, that path-awareness isn't worth the complexity) — unlike a
# plain `>`/`>>` redirect (handled separately below), these are never
# treated as safe just because the target happens to be outside the
# project. Not trying to be a watertight sandbox (a determined model could
# still find a way around this with e.g. a python -c one-liner) — a
# backstop for the COMMON, easy way a shell command edits a file.
_UNCONDITIONAL_MUTATION_PATTERNS = [
    re.compile(r"\bsed\b[^|;&\n]*\s-i\b"),
    re.compile(r"\bperl\b[^|;&\n]*\s-i\b"),
    re.compile(r"\bgawk\b[^|;&\n]*-i\s*inplace\b"),
    re.compile(r"\btee\b"),
    re.compile(r"\b(mv|cp|rm|chmod|chown|truncate|dd)\b"),
    re.compile(r"\bgit\s+(apply|checkout|reset|restore|add|commit|stash)\b"),
    re.compile(r"\bpatch\b"),
]

# This pattern's ONLY exception used to be `/dev/null` — it didn't know
# about `N>&M` (duplicating one file descriptor onto another, e.g. `2>&1`
# to merge stderr into stdout), one of the single most common shell idioms
# for capturing a build/test command's FULL output, so a command like
# `gcc ... 2>&1` or `make 2>&1` got denied as "looks like it would modify a
# file in place" even though `2>&1` never touches a file at all, only
# redirects one already-open stream to another. Excluding `>&<digit>` (in
# addition to `/dev/null`) fixes this without opening a bypass — `.search()`
# still scans the WHOLE command, so `cmd 2>&1 > realfile.txt` is still
# caught by its second, real redirect.
_REDIRECT_PATTERN = re.compile(r">>?\s*(?!/dev/null\b)(?!&\d)\S")
_REDIRECT_TARGET_RE = re.compile(r">>?\s*(?!/dev/null\b)(?!&\d)(\S+)")

# Blocking EVERY redirect outright, regardless of target, would stop
# Verifier from writing a throwaway syntax-check file to /tmp when the
# real build is blocked by a missing system dependency (e.g. ncurses
# headers) it has no way to install (sudo needs an interactive password)
# — a genuinely reasonable way to
# verify what CAN be checked, not an attempt to fix the project. A redirect
# INTO the project (fixing the file being verified) and a redirect to
# scratch space elsewhere are not the same risk — only the former is a
# self-fix. See _looks_like_file_mutation for how the target is checked
# against repo_path.
_HEREDOC_START_RE = re.compile(r"<<-?\s*['\"]?\w")


def _shell_command_prefix(command: str) -> str:
    """The part of `command` before a heredoc body starts (`<<TAG`/
    `<<'TAG'`/`<<-TAG`) — real redirect/mutation syntax only appears here;
    everything after is heredoc CONTENT (e.g. the C source Verifier is
    writing to a scratch file), which can freely contain '>' as a
    comparison operator (`x > 5`) or a word like 'rm'/'tee' inside a
    comment/string without any of that being a real shell mutation.
    Without this, a heredoc body full of C comparisons (`player.vy > 10`,
    `x + w > p->x`, ...) can make the redirect-target extraction below
    think the command wrote garbage paths INSIDE the project, rejecting a
    command whose one REAL redirect (`cat > /tmp/test.c <<'EOF'`) is
    already safely outside it."""
    m = _HEREDOC_START_RE.search(command)
    return command[:m.start()] if m else command


def _redirect_targets(command: str) -> list[str]:
    return [m.group(1) for m in _REDIRECT_TARGET_RE.finditer(command)]


def _looks_like_file_mutation(command: str, repo_path: str | None = None) -> bool:
    """repo_path=None (Analyzer/Planner never call this — they use the
    stricter read-only allowlist instead) falls back to the old
    unconditional behavior: ANY redirect is treated as mutation, fail-safe.
    When repo_path IS given (Verifier — the only caller that has one),
    a `>`/`>>` redirect whose target(s) ALL resolve outside repo_path is
    scratch space, not a self-fix, and is allowed through."""
    prefix = _shell_command_prefix(command)
    if any(p.search(prefix) for p in _UNCONDITIONAL_MUTATION_PATTERNS):
        return True
    if not _REDIRECT_PATTERN.search(prefix):
        return False
    if repo_path is None:
        return True
    root = os.path.realpath(repo_path)
    for target in _redirect_targets(prefix):
        candidate = target if os.path.isabs(target) else os.path.join(root, target)
        resolved = os.path.realpath(candidate)
        if resolved == root or resolved.startswith(root + os.sep):
            return True  # at least one target lands inside the project
    return False  # every redirect target is outside the project


# Allowlist for Analyzer/Planner's bash (roles.py:investigator_tools/
# planner_tools — both keep bash unconditionally, for legitimate
# diagnostics per their own system prompt: "bash for READ-ONLY
# diagnostic commands"). _looks_like_file_mutation above is a DENYLIST,
# right for Verifier (which legitimately needs to run arbitrary builds/
# tests via bash, just not self-fix) — but Analyzer/Planner have no
# legitimate reason to run anything beyond inspection, so a default-DENY
# allowlist fits their narrower job instead: reject shell metacharacters
# outright (chaining/piping/substitution/redirection could launder a
# mutating command past a leading read-only token, e.g. `cat foo; rm -rf
# bar` or `cat foo | sh`), then only pass a curated set of read-only
# commands.
_READ_ONLY_BASH_METACHARS = re.compile(r"[;&$`<>(){}\n]|\|")

_READ_ONLY_BASH_ALLOWLIST = {
    "cat", "head", "tail", "less", "more", "wc", "ls", "grep", "rg", "awk",
    "sed", "echo", "printf", "which", "whoami", "pwd", "env", "printenv",
    "date", "df", "du", "free", "uptime", "uname", "file", "stat", "diff",
    "sort", "uniq", "tr", "cut", "paste", "test", "true", "false", "type",
    "readlink", "realpath", "basename", "dirname", "sha256sum", "md5sum",
    "tree", "jq", "nproc", "lscpu", "id",
}

_READ_ONLY_GIT_SUBCOMMANDS = {
    "status", "log", "diff", "show", "branch", "rev-parse", "ls-files",
    "blame", "describe", "tag", "remote", "shortlog",
    "symbolic-ref", "merge-base", "rev-list", "cat-file", "for-each-ref", "name-rev",
}

# ollama's OWN metadata (analyzer's system prompt explicitly recommends
# `ollama show <model>` for quantization/params/context length) — but
# `run`/`pull`/`rm`/`cp`/`create` all mutate local state, so only the
# read-only subcommands are allowed.
_READ_ONLY_OLLAMA_SUBCOMMANDS = {"show", "list", "ps"}

# node/php/python/etc — allowed ONLY as a bare version query (analyzer's
# own system prompt names exactly this: "a runtime's actual version/output
# (node --version, php -v)") — never with a script/file argument, since
# that executes arbitrary code rather than reporting metadata.
_VERSION_QUERY_INTERPRETERS = {"node", "php", "python", "python3", "ruby", "go", "java", "perl"}
_VERSION_QUERY_FLAGS = {"--version", "-v", "-V", "version"}

# Debugging a report of broken behavior requires REPRODUCING it, not just
# reading code that might cause it — the version-query-only rule above has
# no room for that. This tier allows RUNNING the project's own code to
# reproduce/observe behavior — still no package installs, no `-m
# pip`/`-c`/`-e` interpreter flags, no build tools with side-writing
# subcommands. NOT a sandbox against what the script itself does once
# running (same caveat as this function's own docstring) — Analyzer/
# Planner are trusted to run the thing being debugged, not to run anything.
_EXECUTE_SCRIPT_SPECS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "go":      (("run",), (".go",)),
    "python":  ((), (".py",)),
    "python3": ((), (".py",)),
    "ruby":    ((), (".rb",)),
    "node":    ((), (".js", ".mjs", ".cjs")),
    "php":     ((), (".php",)),
    "perl":    ((), (".pl",)),
}

# A short, curated list of well-known test-runner invocations, matched as
# a PREFIX (trailing flags/args of the tool's own choosing are fine — `go
# test -run TestFoo -v`, `pytest -k foo`, `npm test -- --watch=false`; none
# of these tools' test subcommand has a flag that installs/writes outside
# a build/coverage cache) — NOT a general "first token is a test tool"
# rule: most of these tools also have install/build subcommands that this
# must keep rejecting (plain `npm`/`go`/`cargo` with no further check would
# also let `npm install`/`go install`/`cargo build` straight through).
_EXECUTE_TEST_PREFIXES = {
    ("go", "test"), ("cargo", "test"), ("pytest",),
    ("npm", "test"), ("make", "test"),
}


def _is_execute_script_command(first: str, rest: list[str]) -> bool:
    """`go run file.go [args]`, `python3 file.py [args]` — see
    _EXECUTE_SCRIPT_SPECS above. The script's OWN arguments (`extra_args`,
    anything after the script path) are the program-under-test's business,
    not this function's — only the INTERPRETER position (`script`, the
    first token after any required subcommand) is checked for a leading
    '-', which blocks the interpreter's own escape hatches (`python3 -m
    pip install x`, `python3 -c "..."`, `node -e "..."`) without also
    blocking legitimate flags to the script itself (`go run main.go
    --verbose`, `python3 manage.py --settings=test`). The script argument
    must also carry the right extension for its interpreter."""
    spec = _EXECUTE_SCRIPT_SPECS.get(first)
    if spec is None:
        return False
    required_subcommand, exts = spec
    if required_subcommand:
        if tuple(rest[:len(required_subcommand)]) != required_subcommand:
            return False
        rest = rest[len(required_subcommand):]
    if not rest:
        return False
    script = rest[0]
    if script.startswith("-"):
        return False
    return script.endswith(exts)


def _is_execute_test_command(first: str, rest: list[str]) -> bool:
    tokens = (first, *rest)
    return any(tokens[:len(prefix)] == prefix for prefix in _EXECUTE_TEST_PREFIXES)


def _is_read_only_bash_command(command: str) -> bool:
    """Conservative allowlist heuristic — without it, Analyzer can
    correctly investigate a task (get_knowledge, project_tree, ls -la),
    then use bash to `cat > file << EOF`, `make`, or `apt-get install ...`
    directly instead of reporting back — none of that is a diagnostic, but
    nothing before this stopped it. Also allows a narrow "run to reproduce"
    tier (_is_execute_script_command/_is_execute_test_command) on top of
    the read-only-diagnostics allowlist below — see _EXECUTE_SCRIPT_SPECS'
    own comment for why. Not a watertight sandbox (a determined model
    could still find a way around this, e.g. a quoted one-liner some
    allowed command happens to interpret, or a script that itself mutates
    something once run) — a backstop for the common, easy ways a shell
    command writes something."""
    command = command.strip()
    if not command:
        return False
    segments = _split_read_only_pipeline(command)
    if segments is None:
        return False
    return all(_is_read_only_simple_command(seg) for seg in segments)


# Redirections that only discard/merge output — `git ... 2>/dev/null` is
# how models routinely probe; rejecting it made them conclude that even
# `git diff` is forbidden.
_HARMLESS_REDIRECT = re.compile(r"(?<!\S)(?:[12]?>|&>)\s*/dev/null(?!\S)|(?<!\S)2>&1(?!\S)")


def _split_read_only_pipeline(command: str) -> list[str] | None:
    """Splits on |, ||, &&, ; outside quotes. None if the command uses
    anything else the shell would interpret — redirection to a file,
    substitution ($, backticks, also inside double quotes), subshells,
    background &, newlines. Single-quoted text is literal and never split."""
    command = _HARMLESS_REDIRECT.sub(" ", command)
    segments: list[str] = []
    current: list[str] = []
    quote = ""
    i = 0
    while i < len(command):
        ch = command[i]
        if quote == "'":
            if ch == "'":
                quote = ""
            current.append(ch)
        elif quote == '"':
            if ch in "$`":
                return None
            if ch == "\\" and i + 1 < len(command):
                current.append(command[i:i + 2])
                i += 2
                continue
            if ch == '"':
                quote = ""
            current.append(ch)
        elif ch in "'\"":
            quote = ch
            current.append(ch)
        elif command.startswith(("&&", "||"), i):
            segments.append("".join(current))
            current = []
            i += 2
            continue
        elif ch in "|;":
            segments.append("".join(current))
            current = []
        elif _READ_ONLY_BASH_METACHARS.match(ch):
            return None
        else:
            current.append(ch)
        i += 1
    if quote:
        return None
    segments.append("".join(current))
    segments = [seg.strip() for seg in segments]
    if any(not seg for seg in segments):
        return None
    return segments


def _is_read_only_simple_command(command: str) -> bool:
    tokens = command.split()
    first = tokens[0].rsplit("/", 1)[-1]
    rest = tokens[1:]

    if _is_execute_test_command(first, rest):
        return True
    if first == "git":
        return bool(rest) and rest[0] in _READ_ONLY_GIT_SUBCOMMANDS
    if first == "ollama":
        return bool(rest) and rest[0] in _READ_ONLY_OLLAMA_SUBCOMMANDS
    if first in _VERSION_QUERY_INTERPRETERS or first in _EXECUTE_SCRIPT_SPECS:
        if rest and all(tok in _VERSION_QUERY_FLAGS for tok in rest):
            return True
        return _is_execute_script_command(first, rest)
    if first == "find":
        return not any(x in command for x in ("-exec", "-execdir", "-delete", "-ok", "-fprintf"))

    if first not in _READ_ONLY_BASH_ALLOWLIST:
        return False
    # awk programs are quoted, so the shell-level checks above never see
    # their own output redirection/pipes/system() calls.
    if first == "awk" and any(x in command for x in (">", "|", "system")):
        return False
    # sed/awk's own in-place edit flags.
    return " -i " not in f" {command} " and "--in-place" not in command


class _InvestigationReadOnlyBashMiddleware(AgentMiddleware):
    """Analyzer/Planner keep bash unconditionally (roles.py:
    investigator_tools/planner_tools) for legitimate read-only diagnostics
    — their own system prompt already says so in plain English ("bash
    for READ-ONLY diagnostic commands... never write/delete/mutate
    anything, that is entirely the Coder stage's job", prompts.py:
    _analyzer_system_prompt) — but nothing enforced that boundary at the
    tool level, only the prompt sentence. See
    _is_read_only_bash_command's docstring above for the failure mode this
    backstops. Mechanical backstop, only
    attached to roles whose tool set includes bash but no legitimate
    reason to ever mutate anything (analyzer, planner — see
    _build_role_agent; verifier/coder/quick_fix get
    _NoBashSelfFixMiddleware instead, a denylist, since they legitimately
    need to run arbitrary builds/tests):
    reject any bash/bash_bg call whose command isn't on the
    narrow read-only allowlist, same "final, don't retry" contract as a
    real permission denial."""

    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] not in ("bash", "bash_bg"):
            return await handler(request)
        command = str((request.tool_call.get("args") or {}).get("command", ""))
        if _is_read_only_bash_command(command):
            return await handler(request)
        return ToolMessage(
            content=(
                f"Denied: {command!r} is not a recognized read-only "
                "command — this stage only investigates and reports, it "
                "never writes files, builds, or installs anything (that's "
                "the Coder stage's job, after Planner turns your summary "
                "into a plan). Finish your investigation and report your "
                "findings as your final summary instead of retrying this "
                "or a similar command."
            ),
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
            status="error",
        )


class _PlanModeMiddleware(AgentMiddleware):
    """plan mode (mcp_agent/work_mode.py): only read-only tools; bash runs
    unless it's recognizably mutating (plan_bash.py's denylist — an
    allowlist kept rejecting harmless diagnostics like `ps`). No-op in build
    mode (the default, and always for pipeline roles, which never set
    work_mode)."""

    async def awrap_tool_call(self, request, handler):
        if work_mode.current_work_mode.get() != work_mode.PLAN:
            return await handler(request)
        name = request.tool_call["name"]
        allowed = name in work_mode.PLAN_ALLOWED_TOOLS
        if allowed and name in ("bash", "bash_bg"):
            allowed = not plan_bash.is_mutating_bash(str((request.tool_call.get("args") or {}).get("command", "")))
        if allowed:
            return await handler(request)
        if name in ("bash", "bash_bg"):
            content = (
                "Denied in PLAN mode: this command changes something (writes/deletes "
                "files, installs, mutating git, kills or starts processes). Any command "
                "that only reads or inspects still works — ps, ls, cat, grep, find, "
                "git log/diff/status, docker ps, systemctl status..., chained with "
                "| && || ; and with 2>/dev/null; writing scratch files under /tmp is "
                "allowed too. Use a read-only command for what you need, or "
                "describe the change in your answer instead."
            )
        else:
            content = (
                f"Denied: '{name}' is not available in PLAN mode — this turn is "
                "read-only research, it never writes files or runs mutating "
                "commands. Describe the change in your answer instead."
            )
        return ToolMessage(
            content=content,
            name=name,
            tool_call_id=request.tool_call["id"],
            status="error",
        )


class _SubagentApprovalMiddleware(AgentMiddleware):
    """Permission dialogs for a writing sub-agent (delegate_tool.py). The
    main agent gets them through HumanInTheLoopMiddleware's graph interrupt,
    which agent.py resumes — a sub-agent's graph runs inside a tool call and
    nobody resumes its interrupts, so it asks directly through the same
    tools/confirm.ask_permission the interrupt path ends up in (same "always
    allow" memory, same ask_permissions switch)."""

    async def awrap_tool_call(self, request, handler):
        name = request.tool_call["name"]
        if name not in TOOLS_REQUIRING_APPROVAL:
            return await handler(request)
        from tools.confirm import ask_permission
        from mcp_agent.ask_user_tool import _action_and_detail
        action, detail = _action_and_detail(name, request.tool_call.get("args") or {})
        if await ask_permission(action, detail):
            return await handler(request)
        return ToolMessage(
            content="Rejected by the user — do not retry this call; report what you wanted to do instead.",
            name=name, tool_call_id=request.tool_call["id"], status="error",
        )


def _subagent_writer_middleware(resolved_repo_path: str) -> list:
    """Safety middleware for a sub-agent that can modify things — the same
    guards the main agent has (_base_agent_middleware), with the approval
    step done in-process instead of via a graph interrupt."""
    return [
        plugins.SkillToolRestrictionMiddleware(),
        _PlanModeMiddleware(),
        _SubagentApprovalMiddleware(),
        _OutOfProjectWriteApprovalMiddleware(resolved_repo_path),
        PluginHookMiddleware(resolved_repo_path),
    ]


class _NoBashSelfFixMiddleware(AgentMiddleware):
    """Every role with both bash and real write tools (coder, quick_fix)
    or bash alone with no write tools at all (verifier) must still make
    every actual edit through write_file/edit_file, which have a pre-write
    snapshot for safety (_snapshot_before_write) — bash is only for
    RUNNING checks (build/test/run), never for patching a file in place.
    Verifier's own system prompt already says this in plain English ("You
    have NO write/edit tools... a failure goes back to the Coder stage",
    prompts.py:_verifier_system_prompt) and Coder/quick_fix's prompts now
    describe the same bash-for-checks-only boundary — that alone isn't
    enough: given e.g. a `go build` failure (unused import), a model can
    run `sed -i '/strconv/d' snake.go && go build snake.go` via bash
    instead of fixing it the safe way. Such an edit lands with NO pre-write
    snapshot (bash was never in _snapshot_before_write's tool list, on
    purpose, since most bash calls aren't edits) — for Verifier specifically
    it also skips the whole Coder-Verifier retry loop the pipeline is built
    around, since Verifier itself has no write tools to have made the
    edit through. Mechanical backstop, attached whenever role is
    "verifier"/"coder"/"quick_fix" (see _build_role_agent): reject
    bash/bash_bg calls whose command looks like an in-place file edit,
    pointing the model back at write_file/edit_file (or, for Verifier,
    at reporting the failure) instead of retrying with a different shell
    trick — same "final, don't retry" contract as a real permission
    denial (prompts.py already tells every role to treat a rejected/
    denied call as final).

    repo_path is passed through to _looks_like_file_mutation so a redirect
    to scratch space OUTSIDE the project (e.g. /tmp) isn't treated as a
    self-fix — see that function's module-level comments: without this,
    legitimately writing a throwaway /tmp syntax-check file (say the real
    build is blocked by a missing system dependency there's no way to
    install) would be denied outright alongside real in-project edits,
    leaving the role unable to verify anything it COULD have checked."""

    def __init__(self, repo_path: str, has_write_tools: bool):
        self._repo_path = repo_path
        # Verifier has no write tools at all (roles.py:verifier_tools) —
        # telling it to "use write_file/edit_file instead" would name
        # tools it doesn't have; coder/quick_fix DO have them, so the
        # denial should point there instead of at reporting a failure to
        # a separate stage that, for them, IS this same stage.
        self._has_write_tools = has_write_tools

    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] not in ("bash", "bash_bg"):
            return await handler(request)
        command = str((request.tool_call.get("args") or {}).get("command", ""))
        if not _looks_like_file_mutation(command, self._repo_path):
            return await handler(request)
        fix_instruction = (
            "Use write_file/edit_file instead — they have a pre-write "
            "snapshot for safety, unlike an unsnapshotted shell edit."
            if self._has_write_tools else
            "Report this as a failure in your verdict instead (which file, "
            "what the real error/output was) — you have no write tools on "
            "purpose, you check, you don't fix; the Coder stage does the "
            "fix, with a pre-write snapshot for safety, unlike an "
            "unsnapshotted shell edit here."
        )
        return ToolMessage(
            content=(
                f"Denied: this command looks like it would modify a file "
                f"inside the project ({command!r}) — bash here is for "
                f"RUNNING checks (build/test/run), never for editing "
                f"files. {fix_instruction} Do not retry this or a similar "
                "command. (Writing a throwaway scratch file OUTSIDE the "
                "project, e.g. under /tmp, to help you verify something IS "
                "allowed — this denial means the target resolves inside "
                "the project.)"
            ),
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
            status="error",
        )


async def _evict_ollama_model(model_name: str) -> None:
    """Немедленно выгружает model_name из Ollama вместо того, чтобы ждать
    истечения его OLLAMA_KEEP_ALIVE (2h, см. model_config.py) — без этого
    переключение chat_model (voice_mode) оставляло бы СТАРУЮ модель висеть в
    памяти ещё два часа впустую, пока грузится новая. Best-effort: сбой здесь
    не должен ронять сам свитч модели, просто старая модель проживёт дольше."""
    try:
        import ollama
        client = ollama.AsyncClient(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        await client.generate(model=model_name, prompt="", keep_alive=0)
    except Exception:
        pass


def _compute_num_keep(system_prompt_tokens_estimate: int) -> int:
    """+1500 tokens of slack for the user's own task text and the short
    digest/correction messages injected between attempts — neither is part
    of the system prompt estimate itself. Capped at half of num_ctx:
    num_keep only helps if there's still enough ROOM left after it for the
    part of the history llama.cpp actually discards on a context shift;
    keeping more than half would defeat that. Same formula for both the
    main monolith (_build_agent) and every pipeline role (_build_role_agent)
    — they differ only in WHICH prompt's token estimate they pass in."""
    return min(model_params.num_ctx() // 2, system_prompt_tokens_estimate + 1500)


def _base_agent_middleware(
    resolved_repo_path: str, hitl_middleware: HumanInTheLoopMiddleware,
    pre_hitl: list | None = None,
) -> list:
    """The 7-middleware base both the main monolith agent (_build_agent)
    and every pipeline role agent (_build_role_agent) start from, before
    their own mode/role-specific extras (delegate-nudge/voice for the
    former; ask-user-finalize for the latter — see each function's own
    tail). `hitl_middleware` is the caller's own HumanInTheLoopMiddleware
    instance (its interrupt_on set differs per caller, so it's built by
    the caller, not here) — everything else in this list is identical
    across every caller.

    `pre_hitl` (role-specific mechanical bash-content rejectors —
    _NoBashSelfFixMiddleware/_InvestigationReadOnlyBashMiddleware, see
    _build_role_agent's own tail) MUST be inserted BEFORE hitl_middleware,
    not appended after it: langchain's middleware order is outermost-first
    (langchain.agents.factory._chain_tool_call_wrappers's own docstring,
    "first = outermost" — request flows through them in list order before
    ever reaching the tool), so a middleware appended after hitl_middleware
    only runs once its approval interrupt has already fired and been
    answered — a command doomed to be mechanically rejected must never
    make the user sit through approving it first.

    PluginHookMiddleware is unconditional (not gated behind whether any
    plugin is even installed) — mcp_agent.plugins.load_hooks() is itself
    the check (empty list, no-op, if nothing declared a given hook), so
    there's no meaningful "role doesn't need it" case to special-case.
    Same reasoning for plugins.SkillToolRestrictionMiddleware — it's a
    no-op whenever current_skill_restriction is unset (the common case),
    and placed alongside the other pre_hitl-style mechanical rejectors
    (before hitl_middleware) for the same reason: a tool call that's going
    to be refused outright must never make the user approve it first."""
    return [
        _ToolErrorGuardMiddleware(),
        _UnloadImageGenBeforeGenModelMiddleware(),
        plugins.SkillToolRestrictionMiddleware(),
        _PlanModeMiddleware(),
        *(pre_hitl or ()),
        hitl_middleware,
        _AskUserGuardMiddleware(),
        _OutOfProjectWriteApprovalMiddleware(resolved_repo_path),
        _DedupeToolResultsMiddleware(),
        PluginHookMiddleware(resolved_repo_path),
    ]


async def _build_agent(repo_path: str | None = None):
    tools, tools_by_name, read_history, resolved_repo_path, plugin_tool_names, raw_read_file_tool, tracked_read_file_tool = await _get_tools(repo_path)

    # Модель берётся из settings.py ("тяжёлая модель" — chat_model), а не из
    # отдельных SPIKE_*-переменных, как было раньше (см. историю: этот файл
    # начинался как эксперимент для сравнения со старым orchestrator.py, со
    # своим изолированным конфигом — но давно уже прод-путь, cli.py импортирует
    # именно stream_chat отсюда, и держать конфиг отдельно от settings.py,
    # который реально виден и редактируется в TUI-настройках, значит держать
    # две несвязанные системы выбора модели).
    #
    # Осознанный выбор — ОДИН тег на всё (и основную модель, и judge, см.
    # ниже), без переключения между уровнями "сильная/средняя/слабая" внутри
    # сессии: на этой машине всего 6 GB VRAM, и загрузка ВТОРОЙ пары весов
    # вместо переиспользования резидентной означает evict/reload у Ollama —
    # само переключение стоит времени, которое routing должен был сэкономить.
    # qwen2.5-coder:7b в этой же роли не заворачивает tool-call в свои же
    # теги <tool_call>...</tool_call> надёжно — не разовая случайность, а
    # системная ненадёжность именно этой модели тут.
    MAIN_MODEL = settings.get("chat_model")
    voice_mode = settings.get("voice_mode")

    # optimized_tools (mcp_agent/optimized_tools.py, settings.py) — урезанный
    # (БЕЗ переименования — см. модульный docstring про то, почему) список
    # тулов для этого, основного агента: один тул на смысл (bash/read/grep/
    # glob/write/edit) вместо 5-6 читающих и 5 пишущих вариантов сразу.
    # full_tools сохраняется отдельно — delegate (ниже, build_delegate_tool)
    # строится из НЕГО, не из урезанного списка: его собственное read-only
    # investigation-поведение не должно зависеть от этого тумблера.
    full_tools = tools
    use_optimized_tools = not voice_mode and settings.get("optimized_tools")
    if use_optimized_tools:
        tools, tools_by_name = build_optimized_tools(tools, plugin_tool_names)

    # Built BEFORE the models below (not in its usual place further down)
    # specifically so prompts._SYSTEM_PROMPT_TOKENS_ESTIMATE is already the
    # real size of THIS session's prompt (repo_path/env_block/FLOWAI.md all
    # affect it) by the time num_keep is computed from it — see
    # _ChatOllamaWithNumKeep above for why num_keep needs to cover it.
    # optimized_tools gets its OWN template (_build_optimized_system_prompt),
    # not a conditional branch inside _build_system_prompt — see that
    # function's docstring in prompts.py for why (the default template
    # documents tools this mode doesn't have at all).
    if voice_mode:
        system_prompt = _build_voice_system_prompt()
    elif use_optimized_tools:
        system_prompt = _build_optimized_system_prompt(resolved_repo_path)
    else:
        system_prompt = _build_system_prompt(resolved_repo_path)
    plan_mode = not voice_mode and work_mode.current_work_mode.get() == work_mode.PLAN
    if plan_mode:
        system_prompt += work_mode.PLAN_SYSTEM_PROMPT_SECTION
        prompts._SYSTEM_PROMPT_TOKENS_ESTIMATE = len(system_prompt) // 4
    num_keep = _compute_num_keep(prompts._SYSTEM_PROMPT_TOKENS_ESTIMATE)

    # reasoning=... -> Ollama's "think" API field. Живой замер на этой машине:
    # qwen3:14b на простом промпте сгенерировала 1722 токена невидимого
    # <think>-рассуждения и потратила 368с только на генерацию; с
    # reasoning=False — 21 токен, 2.6с. 66x разница на пустом месте — thinking
    # включён у Qwen3 по умолчанию. Основная модель слушается настройки
    # show_thinking (дефолт — False, см. settings.py) — если включить, её
    # рассуждение реально показывается в UI (см. reasoning_content в
    # _stream_round), а не тратится молча впустую, как раньше, когда эта
    # настройка нигде не была подключена.
    #
    # voice_mode ИГНОРИРУЕТ show_thinking и всегда идёт с reasoning=False —
    # эта настройка задумана для отладки основной кодовой модели, а не для
    # голосового диалога: если она включена (персистится в settings.py между
    # сессиями), voice_chat_model честно рассуждала бы вслух ПЕРЕД коротким
    # ответом (тот самый репортнутый "он размышляет"), и это рассуждение —
    # ровно то место, где маленькая модель дословно пересказывает себе
    # системный промпт, обдумывая, как на него ответить, прежде чем такая
    # фраза-пересказ иногда просачивается в сам ответ.
    model = _build_chat_model(
        model_tag=MAIN_MODEL,
        num_predict=None,
        reasoning=False if voice_mode else settings.get("show_thinking"),
        num_keep=num_keep,
        # voice_mode gets agent_tools=[] below (see its comment) — no
        # tool-call syntax will ever be generated on this model, so the
        # repeat_penalty override some models carry FOR tool-call syntax
        # (model_params._FAMILY_DEFAULTS, see _build_chat_model's docstring)
        # doesn't apply and would only remove a real defense against plain
        # prose repeating itself.
        has_tools=not voice_mode,
    )

    # _semantic_check — бинарный вердикт по готовым tool-результатам, а не
    # рассуждение с нуля, так что модель поменьше сюда просилась ради скорости
    # — но на практике 3b-судья дважды подряд ошибочно забраковал ЯВНО
    # релевантные результаты (git_status и git_diff с реальными изменениями),
    # впустую спалив весь MAX_ATTEMPTS. ВСЕГДА тот же тег, что и MAIN_MODEL
    # (не отдельный слабый/средний уровень) — осознанно, две причины разом:
    # (1) описанная выше ненадёжность слабого судьи, (2) другой тег означает
    # вторую пару весов в те же 6 GB VRAM и evict/reload на каждое
    # переключение — судья просто переиспользует уже резидентные веса с более
    # коротким num_predict.
    #
    # reasoning всегда False здесь, НЕЗАВИСИМО от show_thinking — судья
    # никогда не показывается пользователю (внутренний бинарный чек), так что
    # его размышление негде отображать и нечего с этого получить, только
    # тратить время впустую.
    #
    # format="json" безопасен именно здесь (и только здесь): judge_model
    # используется ИСКЛЮЧИТЕЛЬНО в _semantic_check и _extract_ask_user_shape
    # — оба места просят "Respond with ONLY this JSON" и парсят ответ через
    # parse_json_loose. В отличие от MAIN_MODEL (там реальный текст вперемешку
    # с tool-calls, format="json" сломал бы обычные ответы), тут результат
    # ВСЕГДА обязан быть чистым JSON — Ollama валидирует это на уровне
    # сэмплинга, а не только просьбой в промпте.
    judge_model = _build_chat_model(
        model_tag=MAIN_MODEL,
        num_predict=JUDGE_NUM_PREDICT,
        reasoning=False,
        num_keep=num_keep,
        format="json",
        # judge_model is never handed to create_agent/bound to a tools
        # list — it only ever answers a raw .ainvoke() for a JSON verdict
        # (_semantic_check/_extract_ask_user_shape) — see has_tools's
        # docstring in _build_chat_model.
        has_tools=False,
    )

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={name: True for name in TOOLS_REQUIRING_APPROVAL},
    )

    # voice_mode ходит на слабую voice_chat_model (qwen3:8b, см. settings.py)
    # ради скорости голосового диалога — она объективно ненадёжна с 35+
    # тулами и их схемами (см. комментарий про qwen2.5-coder:7b выше про
    # рвущийся tool-call формат даже у модели покрупнее). Отдавать их ей
    # всё равно, "а вдруг пригодится", значило бы просто плодить сорванные
    # tool-call'ы и путаницу вместо короткого голосового ответа — здесь агент
    # получает ПУСТОЙ список тулов и написанный отдельно короткий разговорный
    # промпт (_build_voice_system_prompt), а не урезанную копию кодового.
    # tools_by_name (для _execute_leaked_tool_call, см. agent.py) остаётся
    # полным независимо от режима — это просто карта имя->объект, не то, что
    # реально показывается модели через create_agent.
    #
    # delegate (mcp_agent/delegate_tool.py) собирается ЗДЕСЬ, а не в
    # _build_tools — ему нужен уже готовый `model` (тот же резидентный
    # инстанс, никакой второй подгрузки весов), а tools-кэш собирается один
    # раз на процесс, ещё до выбора модели. voice_mode пропускает его по
    # той же причине, что и остальные тулы, — пустой список.
    #
    # ПЕРВЫМ в списке (не последним, как было) — тот же эффект, что уже
    # оправдал переупорядочивание write_file в optimized_tools.py:
    # tools_available залогировал delegate ПОСЛЕДНИМ из ~20 тулов (просто
    # дописан в конец списка), а delegate_tool.py's
    # docstring (см. delegate_nudge middleware ниже) отдельно фиксирует,
    # что модель почти никогда не вызывает его сама, даже когда задача явно
    # многофайловая. Позиционное смещение LLM tool-choice к тулам В НАЧАЛЕ
    # списка — тот же эффект, что подтолкнул write_file быть overused —
    # здесь стоит развернуть в пользу delegate, а не оставлять его в самом
    # невыгодном месте.
    # web_read (web_read_tool.py) isn't an MCP tool (needs THIS `model`, see
    # its own docstring) — added here for the same reason delegate is:
    # tools/full_tools only ever hold what _get_tools returned, and that
    # cache is built before any model is chosen. voice_mode gets neither
    # (empty agent_tools — nothing web-related to add web_read alongside).
    # judge_model passed into delegate — see build_delegate_tool's own
    # docstring on why (compact_research, same judge_model as this agent's
    # own self-heal, not a second one).
    agent_tools = [] if voice_mode else [
        *build_delegate_tool(
            model, full_tools, raw_read_file_tool, judge_model,
            repo_path=resolved_repo_path, tracked_read_file_tool=tracked_read_file_tool,
            writer_middleware=lambda: _subagent_writer_middleware(resolved_repo_path),
        ),
        build_web_read_tool(model),
    ] + tools
    # SKILL.md skills (mcp_agent/skills.py) — the tool only exists when
    # there's at least one skill, so its schema doesn't cost tokens for nothing.
    if not voice_mode and md_skills.discover_skills(resolved_repo_path):
        agent_tools.append(md_skills.build_skill_tool(resolved_repo_path))
    # plan mode: write tools aren't in the schema at all, so the model plans
    # instead of reaching for them (_PlanModeMiddleware stays as a backstop,
    # e.g. for bash commands, which stay available for read-only use).
    if plan_mode:
        agent_tools = [t for t in agent_tools if t.name in work_mode.PLAN_ALLOWED_TOOLS]

    # Отдельный от "tools_loaded" в _build_tools лог — тот пишется ДО
    # optimized_tools/voice_mode/delegate, то есть показывает "что подняли из
    # MCP-серверов", а не "что реально видит модель в схеме этого хода".
    # Здесь — уже финальный agent_tools, ровно тот список объектов, что идёт
    # в create_agent() ниже.
    if DEBUG:
        debug_print(f"[dim][MCP-AGENT] Model has {len(agent_tools)} tools available this turn: {[t.name for t in agent_tools]}[/]")
    log_event("tools_available", names=[t.name for t in agent_tools])

    # Last in the list — see compaction.py's module docstring on why it
    # wants the already-deduped view from _DedupeToolResultsMiddleware
    # rather than raw request.messages.
    compact_research = _CompactResearchMiddleware(judge_model)

    # Только не в voice_mode — там agent_tools пуст, delegate самого нет в
    # наборе, подталкивать не к чему (см. delegate_tool.py про то, почему
    # эта миддлварь вообще существует — промпт-совета оказалось недостаточно).
    agent_middleware = _base_agent_middleware(resolved_repo_path, middleware)
    if not voice_mode:
        agent_middleware.append(_DelegateNudgeMiddleware())
    agent_middleware.append(_DropStaleReadsMiddleware())
    agent_middleware.append(compact_research)

    agent = create_agent(
        model,
        agent_tools,
        system_prompt=system_prompt,
        middleware=agent_middleware,
        checkpointer=InMemorySaver(),
    )
    return agent, model, judge_model, tools_by_name, read_history, compact_research


# Freshness key: (chat_model, resolved_repo_path, expert_streaming_enabled,
# num_ctx) — see _get_role_agent for why each element is there.
_role_agent_cache = BuildCache()


def invalidate_tool_caches() -> None:
    """Drops every cache keyed off the loaded MCP tool list: _tools_cache
    (schemas/connections, see _get_tools), _agent_cache (main monolith
    agent — holds a direct reference to the OLD tools list baked in at
    build time via _build_agent's agent_tools = tools + [...]), and
    _role_agent_cache (pipeline roles, same problem one level up). None of
    these caches' freshness keys — (chat_model, voice_mode, ...) for the
    agent, (chat_model, repo_path, ...) for role agents, plain repo_path
    for tools — include settings.gen_agent_tools, so toggling it in
    /settings would otherwise sit inert until the NEXT flowai process even
    though build_mcp_connections (mcp_agent/config.py) already reacts to it
    immediately. Called from settings.py:set_value right when the setting
    actually changes. No lock needed: the worst case is one in-flight tool
    call finishing against the stale (still perfectly valid) tools list —
    every call issued after this one sees the fresh set, spawning/dropping
    the image_gen/music/gen_model MCP subprocesses as needed on next use."""
    _tools_cache.clear()
    invalidate_agent_caches()


def invalidate_agent_caches() -> None:
    """Drops built agents but keeps the (expensive) MCP tool connections —
    for changes that only affect the chat model client itself: a model
    unloaded via /instances (the cached client points at a dead llama-server
    port) or edited per-model sampling params (baked into the client at
    build time, see _build_chat_model)."""
    _agent_cache.clear()
    _role_agent_cache.clear()


async def _build_role_agent(role: str, tool_names: frozenset[str], repo_path: str | None = None):
    """Аналог _build_agent, но для ОДНОЙ роли пайплайна Router->Analyzer->
    Planner->Coder->Verifier (mcp_agent/roles.py) вместо единого агента со
    всеми ~60 тулами: свой набор тулов — tool_names передаёт ВЫЗЫВАЮЩИЙ КОД
    (mcp_agent/pipeline.py), составленный из текущих флагов router.py через
    roles.py:investigator_tools/planner_tools/executor_tools/coder_tools/
    verifier_tools, а не выведенный здесь из статичной таблицы роль->тулы
    (см. docstring roles.py про то, почему). Свой промпт (prompts.py:
    _build_role_system_prompt — НЕ мутирует общий
    _SYSTEM_PROMPT_TOKENS_ESTIMATE, см. его докстринг), свой approval-список
    (roles.py:approval_tools(tool_names)). recursion_limit/max_attempts для
    роли берёт вызывающий код (mcp_agent/stage_runner.py) напрямую из
    roles.py:ROLE_RECURSION_LIMIT/ROLE_MAX_ATTEMPTS — не параметр сборки
    агента, а параметр КОНКРЕТНОГО astream/ainvoke вызова.

    Никакой voice_mode-развилки здесь нет: роли пайплайна не участвуют в
    голосовом режиме — тот идёт по основному _get_agent (пустой tools=[],
    отдельный _build_voice_system_prompt)."""
    tools, tools_by_name, read_history, resolved_repo_path, plugin_tool_names, _raw_read_file_tool, _tracked_read_file_tool = await _get_tools(repo_path)

    MAIN_MODEL = settings.get("chat_model")

    # Та же _compute_num_keep, что в _build_agent, но от ЛОКАЛЬНОЙ оценки
    # токенов ЭТОЙ роли, не от общего мутируемого prompts.
    # _SYSTEM_PROMPT_TOKENS_ESTIMATE — иначе 4 разных по размеру промпта
    # затирали бы одно и то же значение друг у друга.
    system_prompt, system_prompt_tokens_estimate = prompts._build_role_system_prompt(role, resolved_repo_path)
    num_keep = _compute_num_keep(system_prompt_tokens_estimate)

    model = _build_chat_model(
        model_tag=MAIN_MODEL,
        num_predict=None,
        reasoning=settings.get("show_thinking"),
        num_keep=num_keep,
    )
    # Тот же принцип, что у judge_model в _build_agent: ОДИН тег с MAIN_MODEL
    # (не отдельная слабая модель — слабый судья ненадёжен), format="json"
    # безопасен именно здесь, потому что
    # judge_model нигде в этой роли не используется для обычного текста.
    judge_model = _build_chat_model(
        model_tag=MAIN_MODEL,
        num_predict=JUDGE_NUM_PREDICT,
        reasoning=False,
        num_keep=num_keep,
        format="json",
        has_tools=False,  # see has_tools's docstring in _build_chat_model
    )

    # delegate (delegate_tool.py) сознательно НЕ даётся ни одной роли
    # пайплайна, даже инвестигатору — у него ТОТ ЖЕ read-only
    # набор тулов (roles.py:MAIN_INVESTIGATION_TOOL_NAMES), то есть это
    # вложенный саб-агент внутри уже отдельно бюджетируемой
    # исследовательской роли — лишняя матрёшка, а не разделение труда.
    # delegate был оправдан для ОСНОВНОГО монолитного агента (mcp_agent/
    # agent.py:_build_agent, тот случай ниже остаётся с ним), где всё
    # расследование делило ОДИН общий бюджет шагов на весь ход — здесь
    # инвестигатор получает свой собственный, достаточно большой бюджет
    # напрямую (см. mcp_agent/roles.py:ROLE_RECURSION_LIMIT) вместо того,
    # чтобы прятать часть его в непрозрачный (может идти несколько минут
    # без единого признака прогресса) вложенный вызов.
    # plugin_tool_names unioned in AFTER roles.py's static allowlist, not
    # inside it — roles.py can't possibly name a plugin's tools ahead of
    # time (see _load_tools_resilient's docstring), so every plugin tool
    # is available to every pipeline role regardless of tool_names, same
    # blanket-access principle as flowai_guide's _META_TOOLS group.
    agent_tools = filter_tools(tool_names | plugin_tool_names, tools)

    # web_read (web_read_tool.py) isn't an MCP tool — it needs THIS role's
    # own `model` (see build_web_read_tool's docstring), so it can't live in
    # `tools` (agent_builder.py:_get_tools, cached per repo_path independent
    # of model) for filter_tools above to have picked up — added here
    # instead, gated on the SAME roles.py:_WEB_TOOLS membership that
    # decided whether "web_read" is even in tool_names.
    if "web_read" in tool_names:
        agent_tools.append(build_web_read_tool(model))
    # Read-only and available to every role, same as plugin tools above.
    if md_skills.discover_skills(resolved_repo_path):
        agent_tools.append(md_skills.build_skill_tool(resolved_repo_path))

    if DEBUG:
        debug_print(f"[dim][MCP-AGENT] Role '{role}' has {len(agent_tools)} tools available this turn: {[t.name for t in agent_tools]}[/]")
    log_event("tools_available", role=role, names=[t.name for t in agent_tools])

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={name: True for name in approval_tools(tool_names)},
    )
    compact_research = _CompactResearchMiddleware(judge_model)

    # These two run BEFORE hitl_middleware (see _base_agent_middleware's
    # pre_hitl parameter) — a command they're going to reject mechanically
    # must never reach the human approval prompt in the first place.
    pre_hitl: list = []
    if role in ("verifier", "coder", "quick_fix"):
        # Все три держат bash способный подменить собой запись файла
        # (sed -i и т.п., см. докстринг мидлвари) — Verifier БЕЗ единого
        # write-тула рядом, Coder/quick_fix С write_file/edit_file (bash
        # у них только для запуска проверок, не для правок в объезд их
        # snapshot-защиты). resolved_repo_path — чтобы отличить редирект
        # В ПРОЕКТ (самопочинка) от редиректа в /tmp и подобное
        # (одноразовый scratch для проверки).
        pre_hitl.append(_NoBashSelfFixMiddleware(resolved_repo_path, has_write_tools=role != "verifier"))
    if role in ("analyzer", "planner"):
        # Обе роли держат bash безусловно (roles.py:investigator_tools/
        # planner_tools) только для диагностики, никогда для мутации — см.
        # докстринг мидлвари про сценарий, который она блокирует (Analyzer,
        # вместо сводки для Planner, сам пишет файлы через `cat > file`).
        pre_hitl.append(_InvestigationReadOnlyBashMiddleware())

    agent_middleware = _base_agent_middleware(resolved_repo_path, middleware, pre_hitl)
    if role == "planner":
        # Только Planner имеет ask_user (roles.py:planner_tools) — эта
        # мидлварь не даёт ему звать что-либо ещё после первого
        # подтверждения, см. её докстринг в ask_user_tool.py про паттерн
        # зацикленных "готов ли я...?"-переспросов, который она блокирует.
        agent_middleware.append(_AskUserFinalizeMiddleware())
        agent_middleware.append(_AskUserFinalizeNumPredictMiddleware())
    agent_middleware.append(_DropStaleReadsMiddleware())
    agent_middleware.append(compact_research)

    agent = create_agent(
        model,
        agent_tools,
        system_prompt=system_prompt,
        middleware=agent_middleware,
        checkpointer=InMemorySaver(),
    )
    return agent, model, judge_model, tools_by_name, read_history, compact_research, system_prompt_tokens_estimate


async def _get_role_agent(role: str, tool_names: frozenset[str], repo_path: str | None = None):
    """Кеш ПО (РОЛЬ, НАБОР ТУЛОВ) — в отличие от единственного _agent_cache
    пути основного агента, здесь словарь, значение-ключ для сравнения свежести —
    (chat_model, resolved_repo_path) (у ролей пайплайна нет своего
    voice_mode-переключения, см. _build_role_agent). tool_names — ЧАСТЬ
    КЛЮЧА словаря САМОГО (не просто значения для сравнения свежести), а не
    статичный атрибут роли: с переходом на композицию тулов из флагов
    router.py (см. roles.py) одна и та же роль ("analyzer") может прийти с
    РАЗНЫМИ наборами тулов в разных ходах (needs_project=false в одном,
    true в другом) — без tool_names в ключе второй ход получил бы
    закешированный агент первого, собранный под другой набор тулов.
    repo_path — часть ключа-значения, а не только chat_model:
    pipeline.py пересчитывает repo_path = os.getcwd() на каждый ход, и без
    этого агент, один раз собранный под первый repo_path, молча продолжал
    бы отвечать (и писать/выполнять bash) в НЕМ, даже когда следующий
    ход пришёл с другим repo_path и той же моделью — тот же класс бага,
    что и в _get_tools (см. его докстринг), только уровнем выше: там
    чинится MCP-подпроцессы/cwd, здесь — system-prompt с зашитым repo_path
    (prompts.py:_build_role_system_prompt). Смена ТОЛЬКО repo_path (модель
    та же) не должна выгружать модель из Ollama — _evict_ollama_model
    сравнивается по chat_model отдельно, не по всему ключу. При смене
    chat_model каждый уже закешированный (role, tool_names) при следующем
    обращении пересоберётся и вызовет _evict_ollama_model для старого тега
    — если несколько уже закешированы, это может вызвать evict несколько
    раз подряд, что безопасно (best-effort, см. _evict_ollama_model),
    просто не отличается по цене от одного вызова."""
    current_model = settings.get("chat_model")
    cache_key = (role, tool_names)
    current_value = (
        current_model, repo_path or os.getcwd(),
        settings.get("expert_streaming_enabled"), tuple(sorted(model_params.effective(current_model).items())),
        settings.get("parallel_slots"),
        md_skills.fingerprint(repo_path or os.getcwd()),
    )

    async def _on_stale(old_freshness, _new_freshness):
        old_model = old_freshness[0]
        if old_model != current_model:
            await _evict_ollama_model(old_model)

    return await _role_agent_cache.get_or_build(
        cache_key, current_value,
        lambda: _build_role_agent(role, tool_names, repo_path),
        on_stale=_on_stale,
    )


async def _get_agent(repo_path: str | None = None):
    """MCP-серверы/тулы кешируются вечно на процесс (см. _get_tools — дорогие,
    не зависят от модели). Сам agent/model/judge_model кешируются ОТДЕЛЬНО и
    пересобираются, если (chat_model, voice_mode, resolved_repo_path,
    optimized_tools, always_delegate_search) поменялись с прошлого раза (см.
    _agent_cache_key) —
    иначе voice_mode (settings.py:set_value переключает chat_model на
    qwen3:8b и обратно) молча продолжал бы работать на старой модели/
    тулсете/промпте до перезапуска процесса; repo_path в ключе — тот же
    повод, что у _get_role_agent (см. его докстринг) — агент не должен молча
    остаться привязан к первому repo_path, увиденному этим процессом.
    optimized_tools/always_delegate_search в ключе — тот же принцип: тумблер
    /settings должен подхватиться со следующего хода, а не молча висеть до
    перезапуска (always_delegate_search меняет только текст system_prompt,
    не набор тулов, но собирается той же _build_system_prompt, что кеширует
    этот agent).
    Пересборка агента здесь дешёвая (тулы уже в кеше) — новая ChatOllama +
    create_agent, без повторного подъёма подпроцессов."""
    current_model = settings.get("chat_model")
    current_key = (
        current_model, settings.get("voice_mode"), repo_path or os.getcwd(), work_mode.current_work_mode.get(),
        settings.get("optimized_tools"), settings.get("always_delegate_search"),
        settings.get("expert_streaming_enabled"), tuple(sorted(model_params.effective(current_model).items())),
        settings.get("parallel_slots"),
        md_skills.fingerprint(repo_path or os.getcwd()),
    )

    async def _on_stale(old_freshness, _new_freshness):
        old_model = old_freshness[0]
        if old_model != current_model:
            await _evict_ollama_model(old_model)

    return await _agent_cache.get_or_build(
        None, current_key, lambda: _build_agent(repo_path), on_stale=_on_stale,
    )


