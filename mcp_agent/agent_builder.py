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
from ui.console import console
from mcp_agent.ask_user_tool import (
    _AskUserFinalizeMiddleware,
    _AskUserGuardMiddleware,
    _OutOfProjectWriteApprovalMiddleware,
    _ToolErrorGuardMiddleware,
    ask_user,
    mark_plan_step_current,
)
from mcp_agent.compaction import _CompactResearchMiddleware, _DropStaleReadsMiddleware
from mcp_agent.config import build_mcp_connections, TOOLS_REQUIRING_APPROVAL
from mcp_agent.debug_log import log_event
from mcp_agent.delegate_tool import _DelegateNudgeMiddleware, build_delegate_tool
from mcp_agent.message_utils import _DedupeToolResultsMiddleware
from mcp_agent.optimized_tools import build_optimized_tools
from mcp_agent.roles import approval_tools, filter_tools
from mcp_agent.model_config import (
    DEBUG,
    JUDGE_NUM_PREDICT,
    MODEL_TEMPERATURE,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_NUM_PREDICT,
    REPEAT_LAST_N,
    REPEAT_PENALTY,
    TOOL_OUTPUT_CHAR_CAP,
    TOP_K,
    TOP_P,
)
from mcp_agent import prompts
from mcp_agent.prompts import _build_optimized_system_prompt, _build_system_prompt, _build_voice_system_prompt


# Живой прогон (сессия 20260707-135011-31723e6e, journalctl -u ollama):
# длинный ход (ask_user + два раунда правок + верификация) забил весь
# num_ctx=32768, и llama.cpp сделал "context shift" — ДВАЖДЫ в этом же ходе
# ("slot context shift, n_keep=4, n_discard=16381" в логе Ollama), выбросив
# ВСЁ, кроме первых 4 токенов промпта (системный промпт + исходная задача
# улетают целиком) и последней половины истории. Второй раз это случилось
# прямо ПОСЕРЕДИНЕ генерации финального ответа — отсюда обрыв на полуслове
# ("Таким образом, "), и, вероятно, тем же объясняется бессмысленная вторая
# правка чуть раньше в том же ходе (тоже сразу после context shift).
#
# ChatOllama (langchain_ollama) не имеет поля num_keep вообще — оно не входит
# ни в список полей модели, ни в дефолтный options_dict, который _chat_params
# строит из self.*, так что просто передать num_keep=... в конструктор
# неоткуда. num_keep — реальный параметр Ollama (есть в ollama.Options), его
# просто нужно докинуть в params["options"] на уровне API-вызова. Подкласс
# вместо monkey-patch/.bind(options=...): .bind() кладёт лишний kwarg на
# верхний уровень запроса (см. комментарий у _extract_ask_user_shape в
# self_heal.py, там же живой баг с ЭТИМ способом), а не в options{}, и, что
# важнее, create_agent() ниже сам вызывает model.bind_tools(...) — обычный
# Runnable, возвращённый .bind(), этот метод не реализует. Подкласс остаётся
# настоящим ChatOllama (bind_tools у него есть), только один хук (_chat_params)
# дописывает num_keep в уже готовый options-словарь.
class _ChatOllamaWithNumKeep(ChatOllama):
    num_keep: int | None = None

    def _chat_params(self, messages, stop=None, **kwargs):
        params = super()._chat_params(messages, stop, **kwargs)
        if self.num_keep is not None and isinstance(params.get("options"), dict):
            params["options"]["num_keep"] = self.num_keep
        return params


# gpt-oss:20b раньше требовал expert_streaming_enabled как единственный
# способ обойти GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor)) на
# обычном Ollama-пути (известный баг Ollama,
# github.com/ollama/ollama/issues/16946) — живой journalctl-трейс
# (2026-08-11) показал, что этот краш триггерится НЕ самим gpt-oss, а
# OLLAMA_KV_CACHE_TYPE=q8_0 конкретно: тот же gpt-oss:20b на том же
# железе успешно отработал несколько ходов подряд под f16, и упал на
# первой же загрузке сразу после смены на q8_0. ollama_kv_cache.py
# переключает это автоматически перед каждой сборкой модели — больше не
# нужно принудительно гнать gpt-oss через экспериментальный
# expert-streaming только чтобы обойти эту переменную окружения.


# Per-model sampling overrides for models whose recommended settings differ
# from this app's Qwen-tuned defaults (MODEL_TEMPERATURE/TOP_P/TOP_K/
# REPEAT_PENALTY above) -- keyed by model_tag prefix before the ':'. Applied
# on top of those defaults in _build_chat_model, never touching them for any
# other model.
#
# glm-4.7-flash: live test (2026-08-14, expert-streaming backend) reproduced
# this app's default REPEAT_PENALTY=1.2/REPEAT_LAST_N=512 causing the
# model's tool-call arguments to degenerate into incoherent word-soup on any
# non-trivial prompt (a long real system prompt + several tools was enough --
# a trivial one-line prompt didn't trigger it). Root cause: GLM's own
# tool-call syntax (<tool_call>name<arg_key>...<arg_value>...</tool_call>)
# repeats the same structural tokens on every argument, and a repeat
# penalty this strong fights that, pushing the model into increasingly
# desperate synonym-hunting instead of clean structural output. These are
# the community-recommended values instead (HF discussion
# unsloth/GLM-4.7-Flash-GGUF#23) -- confirmed clean on the same reproduction
# (same prompt/tools, only sampling changed).
_MODEL_SAMPLING_OVERRIDES: dict[str, dict] = {
    "glm-4.7-flash": {"temperature": 0.7, "top_p": 0.95, "min_p": 0.01, "repeat_penalty": 1.0},
}


def _sampling_overrides_for(model_tag: str) -> dict:
    return _MODEL_SAMPLING_OVERRIDES.get(model_tag.partition(":")[0], {})


def _build_chat_model(
    *, model_tag: str, num_predict: int, reasoning: bool, num_keep: int, format: str | None = None,
):
    """Единая точка сборки и для основной, и для judge-модели (обоих
    вызывающих — _build_agent ниже и _build_role_agent, см. их обе) —
    settings.expert_streaming_enabled переключает backend ЗДЕСЬ, один раз,
    вместо развилки в каждом из 4 мест конструктора. См.
    expert_streaming.py's docstring за полным разбором: что за форк, откуда,
    почему НЕ смёржен в апстрим, и какой trade-off (PP заметно медленнее, TG
    в среднем на треть быстрее по живому отзыву автора PR) он приносит.

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
    expert_streaming.py docstring, раздел "известные огрубления"."""
    if settings.get("expert_streaming_enabled"):
        ok, msg = expert_streaming.ensure_running(
            model_tag, num_ctx=settings.get("num_ctx"), show_thinking=reasoning,
        )
        if ok:
            sampling = _sampling_overrides_for(model_tag)
            extra_body = {
                "top_k": TOP_K,
                "repeat_penalty": sampling.get("repeat_penalty", REPEAT_PENALTY),
                "repeat_last_n": REPEAT_LAST_N,
            }
            if "min_p" in sampling:
                extra_body["min_p"] = sampling["min_p"]
            if format == "json":
                extra_body["response_format"] = {"type": "json_object"}
            return ChatOpenAI(
                model=model_tag,
                base_url=f"http://{expert_streaming.DEFAULT_HOST}:{expert_streaming.DEFAULT_PORT}/v1",
                api_key="not-needed",
                max_tokens=num_predict,
                temperature=sampling.get("temperature", MODEL_TEMPERATURE),
                top_p=sampling.get("top_p", TOP_P),
                extra_body=extra_body,
                # Live bug (user report): "No streaming chunk received for
                # 120.0s ... stream_chunk_timeout fired" — langchain_openai's
                # own watchdog assumes a real OpenAI-class endpoint (first
                # token in low single-digit seconds even for large prompts).
                # This backend's own measured prompt-processing throughput
                # is ~2-5 tok/s (see expert_streaming.py docstring) — a
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
        # запуск, но без этой проверки предупреждение всё равно печаталось
        # бы дважды подряд для одного и того же провала — живой баг, ровно
        # так и было в отчёте пользователя).
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

    sampling = _sampling_overrides_for(model_tag)
    kwargs = dict(
        model=model_tag,
        base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        keep_alive=OLLAMA_KEEP_ALIVE,
        num_ctx=settings.get("num_ctx"),
        num_predict=num_predict,
        reasoning=reasoning,
        temperature=sampling.get("temperature", MODEL_TEMPERATURE),
        top_p=sampling.get("top_p", TOP_P),
        top_k=TOP_K,
        repeat_penalty=sampling.get("repeat_penalty", REPEAT_PENALTY),
        repeat_last_n=REPEAT_LAST_N,
        num_keep=num_keep,
    )
    if format is not None:
        kwargs["format"] = format
    return _ChatOllamaWithNumKeep(**kwargs)
from mcp_agent.snapshots import _snapshot_before_write, list_file_snapshots, restore_file_snapshot
from mcp_agent.tool_wrappers import (
    _add_glob_warning,
    _add_regex_warning,
    _add_verify_reminder,
    _autofill_expected_lines,
    _bind_constant_args,
    _cache_line_content_tool,
    _cap_tool_output,
    _dedupe_read_tool,
    _normalize_edit_file_args,
    _require_expected_lines,
    _split_head_tail_tool,
    _wrap_read_invalidation,
)

_tools_cache: dict[str, tuple] = {}
_tools_cache_lock = asyncio.Lock()

# Кешируется ОТДЕЛЬНО от тулов, вместе с (chat_model, voice_mode), на
# которых он собран (см. _get_agent) — voice_mode (settings.py:set_value)
# переключает chat_model между qwen3-coder:30b и qwen3:8b на лету, и агент
# должен реально пересобраться на новой модели, а не молча продолжать
# работать на старой до перезапуска процесса. voice_mode — ОТДЕЛЬНАЯ часть
# ключа, не только chat_model: если voice_chat_model случайно совпадает с
# обычной chat_model (пользователь сам так настроил), тег при переключении
# voice_mode не меняется вообще — без voice_mode в ключе кеш решил бы, что
# пересобирать нечего, и агент остался бы с пустым тулсетом/голосовым
# промптом (или наоборот) до следующей смены МОДЕЛИ, а не режима.
_agent_cache: tuple | None = None
_agent_cache_key: tuple | None = None  # (chat_model, voice_mode)
_agent_cache_lock = asyncio.Lock()


async def _load_tools_resilient(
    client: MultiServerMCPClient, server_names: list[str], repo_path: str
) -> list:
    """client.get_tools() без server_name гребёт ВСЕ сервера через один
    asyncio.gather() без return_exceptions — если хотя бы один не
    поднимается (например mcp-server-git запущен не в git-репозитории,
    или npx недоступен), падает вся пачка и агент остаётся без единого
    тула. Грузим по серверам отдельно: один сбойный сервер лишает нас
    только своих тулов, а не всех 35."""
    tools = []
    for name in server_names:
        try:
            server_tools = await client.get_tools(server_name=name)
            if name == "git":
                # repo_path у git-тулов константен для всей сессии — модели
                # его вообще не показываем, см. _bind_constant_args.
                server_tools = [
                    _bind_constant_args(t, {"repo_path": repo_path}) for t in server_tools
                ]
            tools.extend(server_tools)
        except Exception as e:
            console.print(f"[yellow]⚠ MCP-сервер '{name}' не запустился — его инструменты недоступны: {e}[/]")
    return tools


async def _build_tools(repo_path: str | None = None):
    resolved_repo_path = repo_path or os.getcwd()
    connections = build_mcp_connections(resolved_repo_path)
    client = MultiServerMCPClient(connections)
    tools = await _load_tools_resilient(client, list(connections.keys()), resolved_repo_path)
    # search_files (filesystem MCP-сервер) временно убран: excludePatterns у
    # него по умолчанию пустой (index.js:124 в самом пакете), так что без
    # явной передачи он рекурсивно обходит ВСЁ дерево repo_path без единого
    # исключения — на живом прогоне это означало обход vendor/+venv-tts
    # (42 ГБ, 322k файлов в этом самом репозитории). find_files_by_name
    # (code_search_server.py) закрывает тот же случай — умеет настоящий glob
    # и уже пропускает vendor/node_modules/.venv/venv/.git и т.п.
    tools = [t for t in tools if t.name != "search_files"]
    tools = [_cap_tool_output(t, TOOL_OUTPUT_CHAR_CAP) for t in tools]
    tools = [_normalize_edit_file_args(t) if t.name == "edit_file" else t for t in tools]
    # read_history — общий для read_file/read_text_file (ключ — путь) И для
    # _require_expected_lines ниже (ключ — кортеж ("__line_edit_failures__",
    # tool_name), namespace не пересекается с путями), очищается в начале
    # каждого stream_chat (см. там же) и на каждом self-heal retry.
    read_history: dict = {}
    # line_content_cache — реальные строки с диска по каждому успешному
    # read_file/read_text_file/read_file_range/read_multiple_files (см.
    # tool_wrappers.py:_cache_line_content_tool), тот же жизненный цикл,
    # что у read_history (очистка/инвалидация — см. ниже). Даёт
    # _autofill_expected_lines материал, чтобы не полагаться на то, что
    # модель верно перепечатает expected_first_line/expected_last_line/
    # expected_line сама.
    line_content_cache: dict = {}
    tools = [
        _cache_line_content_tool(t, line_content_cache)
        if t.name in ("read_file", "read_text_file", "read_file_range", "read_multiple_files") else t
        for t in tools
    ]
    tools = [
        _dedupe_read_tool(t, read_history) if t.name in ("read_file", "read_text_file", "read_file_range") else t
        for t in tools
    ]
    tools = [_wrap_read_invalidation(t, read_history) for t in tools]
    tools = [_wrap_read_invalidation(t, line_content_cache) for t in tools]
    # Outermost of the read wrappers (added last) — see its own docstring
    # for why: the inner layers above (cache/dedupe/invalidation) each see
    # two ordinary single-param calls instead of one with both set.
    tools = [_split_head_tail_tool(t) if t.name in ("read_file", "read_text_file") else t for t in tools]
    tools = [_add_verify_reminder(t) if t.name in ("write_file", "edit_file", "replace_lines", "copy_lines", "insert_lines") else t for t in tools]
    # _require_expected_lines TEMPORARILY DISABLED (20260812): now that
    # replace_lines/insert_lines/copy_lines check expected_*_hash against
    # the real line content (fs_extra_server.py) instead of full line text,
    # live runs show the model landing on the right lines far more often —
    # but it also frequently just omits expected_*_hash on calls where
    # _autofill_expected_lines below has no cache hit (range not read this
    # turn), and the hard "required" rejection this wrapper used to add on
    # top burned a turn on an edit that would otherwise have gone through
    # fine unverified. fs_extra_server.py's own hash check still runs
    # whenever a value IS passed (by the model or by autofill below) — this
    # only removes the wrapper that made passing one mandatory. Re-enable
    # by uncommenting the line below if blind/stale edits become a problem
    # again without it.
    # tools = [_require_expected_lines(t, read_history) if t.name in ("replace_lines", "copy_lines", "insert_lines") else t for t in tools]
    tools = [
        _autofill_expected_lines(t, line_content_cache) if t.name in ("replace_lines", "copy_lines", "insert_lines") else t
        for t in tools
    ]
    tools = [_add_glob_warning(t) if t.name == "search_files" else t for t in tools]
    tools = [_add_regex_warning(t) if t.name == "search_code" else t for t in tools]
    # Снимок содержимого файла ДО мутации — outermost-обёртка, чтобы
    # захватить состояние прямо перед реальным изменением, а не до
    # нормализации/дедупа/verify-хинта выше по цепочке (см.
    # _snapshot_before_write). Даёт restore_file_snapshot точки возврата,
    # которых нет в git-истории (несколько незакоммиченных правок подряд).
    # copy_lines мутирует dest_path, а не path — другой path_key.
    tools = [
        _snapshot_before_write(t, resolved_repo_path, path_key="dest_path" if t.name == "copy_lines" else "path")
        if t.name in ("write_file", "edit_file", "git_restore_file", "replace_lines", "copy_lines", "insert_lines") else t
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
    tools.append(list_file_snapshots)
    restore_file_snapshot_wrapped = _wrap_read_invalidation(restore_file_snapshot, read_history)
    restore_file_snapshot_wrapped = _wrap_read_invalidation(restore_file_snapshot_wrapped, line_content_cache)
    tools.append(restore_file_snapshot_wrapped)
    # Для _execute_leaked_tool_call (см. выше) — те же самые объекты тулов,
    # что видит create_agent ниже (уже с _cap_tool_output/_bind_constant_args
    # обёртками), просто доступные по имени напрямую, в обход графа.
    tools_by_name = {t.name: t for t in tools}

    if DEBUG:
        console.print(f"[dim][MCP-AGENT] Loaded {len(tools)} tools: {[t.name for t in tools]}[/]")
    log_event("tools_loaded", names=[t.name for t in tools])

    return tools, tools_by_name, read_history, resolved_repo_path


async def _get_tools(repo_path: str | None = None):
    """_build_tools() spawns 8 MCP server subprocesses (npx filesystem,
    mcp-server-git, mcp-server-fetch, our own python servers) and loads 35+
    tool schemas — независимо от того, какая chat_model выбрана. Кешируется
    ОТДЕЛЬНО от модели (см. _get_agent) — переключение chat_model (voice_mode
    ON/OFF) не должно заново поднимать все MCP-подпроцессы, это дорогая и
    никак не связанная с выбором модели часть.

    Ключ — resolved repo_path, а не просто "было — не важно, какое". Раньше
    это был единственный global-слот без ключа вообще: первый repo_path,
    на котором собрались тулы, оставался в силе НАВСЕГДА для всего
    процесса, даже когда следующий вызов приходил с другим repo_path (у
    пайплайна repo_path = os.getcwd() пересчитывается на каждый ход, см.
    pipeline.py) — bash_exec/filesystem/git-серверы у ВСЕХ последующих
    ролей молча продолжали бы работать в первом попавшемся проекте. В
    живом тесте (mail-server) не проявилось напрямую (repo_path был один
    и тот же весь прогон), но найдено при разборе того же прогона —
    чинится заранее, до того как ударит на реальной смене проекта в
    рамках одного процесса."""
    global _tools_cache
    key = repo_path or os.getcwd()
    if key in _tools_cache:
        return _tools_cache[key]
    async with _tools_cache_lock:
        if key not in _tools_cache:
            _tools_cache[key] = await _build_tools(repo_path)
    return _tools_cache[key]


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


# In-place file mutation idioms bash_exec could use to "fix" something
# itself instead of just checking it — sed/perl/awk in-place edit flags,
# any redirection into a real path (writes/overwrites — /dev/null is the
# one legitimate exception, e.g. `cmd >/dev/null` to silence output), tee,
# and the classic file-mutating coreutils/git commands. Not trying to be a
# watertight sandbox (a determined model could still find a way around
# this with e.g. a python -c one-liner) — this is a backstop for the
# COMMON, easy way a shell command edits a file, matching the actual live
# incident below, not exhaustive security.
_FILE_MUTATION_PATTERNS = [
    re.compile(r"\bsed\b[^|;&\n]*\s-i\b"),
    re.compile(r"\bperl\b[^|;&\n]*\s-i\b"),
    re.compile(r"\bgawk\b[^|;&\n]*-i\s*inplace\b"),
    re.compile(r">>?\s*(?!/dev/null\b)\S"),
    re.compile(r"\btee\b"),
    re.compile(r"\b(mv|cp|rm|chmod|chown|truncate|dd)\b"),
    re.compile(r"\bgit\s+(apply|checkout|reset|restore|add|commit|stash)\b"),
    re.compile(r"\bpatch\b"),
]


def _looks_like_file_mutation(command: str) -> bool:
    return any(p.search(command) for p in _FILE_MUTATION_PATTERNS)


class _VerifierNoSelfFixMiddleware(AgentMiddleware):
    """Verifier has no write tools on purpose (roles.py:verifier_tools —
    only read + shell) precisely so a failed check turns into a REPORTED
    failure that sends the change back to Coder for a proper, snapshotted
    fix (pipeline.py's Coder<->Verifier retry loop) — not something
    Verifier patches itself. Its own system prompt already says so in
    plain English ("You have NO write/edit tools... a failure goes back to
    the Coder stage", prompts.py:_verifier_system_prompt) — live bug
    anyway: given a `go build` failure (unused import), qwen3-coder ran
    `sed -i '/strconv/d' snake.go && go build snake.go` via bash_exec
    instead of reporting it. bash_exec is the ONE tool Verifier keeps that
    can still write to disk (it needs it to run builds/tests), and the
    prompt sentence alone didn't stop the model from using it to edit
    instead of just check. That edit landed with NO pre-write snapshot
    (_snapshot_before_write only wraps write_file/edit_file/replace_lines/
    insert_lines/copy_lines/git_restore_file — bash_exec was never in that
    list, on purpose, since most bash_exec calls aren't edits) and
    skipped the whole Coder-Verifier retry loop the pipeline is built
    around. Mechanical backstop, only attached when role == "verifier"
    (see _build_role_agent): reject bash_exec/bash_exec_bg calls whose
    command looks like an in-place file edit, pointing the model back at
    reporting the failure instead of retrying with a different shell
    trick — same "final, don't retry" contract as a real permission
    denial (prompts.py already tells every role to treat a rejected/
    denied call as final)."""

    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] not in ("bash_exec", "bash_exec_bg"):
            return await handler(request)
        command = str((request.tool_call.get("args") or {}).get("command", ""))
        if not _looks_like_file_mutation(command):
            return await handler(request)
        return ToolMessage(
            content=(
                f"Denied: this command looks like it would modify a file "
                f"in place ({command!r}) — Verifier has no write tools on "
                "purpose, you check, you don't fix. Report this as a "
                "failure in your verdict instead (which file, what the "
                "real error/output was) so it goes back to the Coder "
                "stage — which has real write tools AND a pre-write "
                "snapshot for safety, unlike an unsnapshotted shell edit "
                "here. Do not retry this or a similar command."
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


async def _build_agent(repo_path: str | None = None):
    tools, tools_by_name, read_history, resolved_repo_path = await _get_tools(repo_path)

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
    # qwen2.5-coder:7b в этой же роли на ДВУХ живых прогонах подряд не
    # завернула tool-call в свои же теги <tool_call>...</tool_call> — не
    # разовая случайность, а системная ненадёжность именно этой модели тут.
    MAIN_MODEL = settings.get("chat_model")
    voice_mode = settings.get("voice_mode")

    # optimized_tools (mcp_agent/optimized_tools.py, settings.py) — урезанный
    # (БЕЗ переименования — см. модульный docstring про то, почему) список
    # тулов для этого, легаси-агента: один тул на смысл (bash/read/grep/
    # glob/write/edit) вместо 5-6 читающих и 5 пишущих вариантов сразу.
    # full_tools сохраняется отдельно — delegate (ниже, build_delegate_tool)
    # строится из НЕГО, не из урезанного списка: его собственное read-only
    # investigation-поведение не должно зависеть от этого тумблера.
    full_tools = tools
    use_optimized_tools = not voice_mode and settings.get("optimized_tools")
    if use_optimized_tools:
        tools, tools_by_name = build_optimized_tools(tools)

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
    # +1500 tokens of slack for the user's own task text and the short
    # digest/correction messages stream_chat injects between attempts
    # (_start_next_attempt) — neither is part of the system prompt itself.
    # Capped at half of num_ctx: num_keep only helps if there's still enough
    # ROOM left after it for the part of the history llama.cpp actually
    # discards on a context shift; keeping more than half would defeat that.
    num_keep = min(settings.get("num_ctx") // 2, prompts._SYSTEM_PROMPT_TOKENS_ESTIMATE + 1500)

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
        num_predict=OLLAMA_NUM_PREDICT,
        reasoning=False if voice_mode else settings.get("show_thinking"),
        num_keep=num_keep,
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
    # ПЕРВЫМ в списке (не последним, как было) — тот же живой прогон
    # (20260812, XOR-в-Go задача), что и переупорядочивание write_file в
    # optimized_tools.py: tools_available залогировал delegate ПОСЛЕДНИМ
    # из ~20 тулов (просто дописан в конец списка), а delegate_tool.py's
    # docstring (см. delegate_nudge middleware ниже) отдельно фиксирует,
    # что модель почти никогда не вызывает его сама, даже когда задача явно
    # многофайловая. Позиционное смещение LLM tool-choice к тулам В НАЧАЛЕ
    # списка — тот же эффект, что подтолкнул write_file быть overused —
    # здесь стоит развернуть в пользу delegate, а не оставлять его в самом
    # невыгодном месте.
    agent_tools = [] if voice_mode else [build_delegate_tool(model, full_tools)] + tools

    # Отдельный от "tools_loaded" в _build_tools лог — тот пишется ДО
    # optimized_tools/voice_mode/delegate, то есть показывает "что подняли из
    # MCP-серверов", а не "что реально видит модель в схеме этого хода".
    # Здесь — уже финальный agent_tools, ровно тот список объектов, что идёт
    # в create_agent() ниже.
    if DEBUG:
        console.print(f"[dim][MCP-AGENT] Model has {len(agent_tools)} tools available this turn: {[t.name for t in agent_tools]}[/]")
    log_event("tools_available", names=[t.name for t in agent_tools])

    # Last in the list — see compaction.py's module docstring on why it
    # wants the already-deduped view from _DedupeToolResultsMiddleware
    # rather than raw request.messages.
    compact_research = _CompactResearchMiddleware(judge_model)

    # Только не в voice_mode — там agent_tools пуст, delegate самого нет в
    # наборе, подталкивать не к чему (см. delegate_tool.py про то, почему
    # эта миддлварь вообще существует — промпт-совета оказалось недостаточно).
    agent_middleware = [
        _ToolErrorGuardMiddleware(),
        _UnloadImageGenBeforeGenModelMiddleware(),
        middleware,
        _AskUserGuardMiddleware(),
        _OutOfProjectWriteApprovalMiddleware(resolved_repo_path),
        _DedupeToolResultsMiddleware(),
    ]
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


_role_agent_cache: dict[tuple, tuple] = {}
_role_agent_cache_key: dict[tuple, tuple] = {}  # (role, tool_names) -> (chat_model, repo_path)
_role_agent_cache_lock = asyncio.Lock()


def invalidate_tool_caches() -> None:
    """Drops every cache keyed off the loaded MCP tool list: _tools_cache
    (schemas/connections, see _get_tools), _agent_cache (legacy monolith
    agent — holds a direct reference to the OLD tools list baked in at
    build time via _build_agent's agent_tools = tools + [...]), and
    _role_agent_cache (pipeline roles, same problem one level up). None of
    these caches' keys — (chat_model, voice_mode) for the agent, (role,
    tool_names) -> (chat_model, repo_path) for role agents, plain repo_path
    for tools — include settings.gen_agent_tools, so toggling it in
    /settings would otherwise sit inert until the NEXT flowai process even
    though build_mcp_connections (mcp_agent/config.py) already reacts to it
    immediately. Called from settings.py:set_value right when the setting
    actually changes. No lock needed: the worst case is one in-flight tool
    call finishing against the stale (still perfectly valid) tools list —
    every call issued after this one sees the fresh set, spawning/dropping
    the image_gen/music/gen_model MCP subprocesses as needed on next use."""
    global _tools_cache, _agent_cache, _agent_cache_key, _role_agent_cache, _role_agent_cache_key
    _tools_cache = {}
    _agent_cache = None
    _agent_cache_key = None
    _role_agent_cache = {}
    _role_agent_cache_key = {}


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
    голосовом режиме — тот идёт по легаси _get_agent (пустой tools=[],
    отдельный _build_voice_system_prompt)."""
    tools, tools_by_name, read_history, resolved_repo_path = await _get_tools(repo_path)

    MAIN_MODEL = settings.get("chat_model")

    # Тот же num_keep-расчёт, что в _build_agent, но от ЛОКАЛЬНОЙ оценки
    # токенов ЭТОЙ роли, не от общего мутируемого prompts.
    # _SYSTEM_PROMPT_TOKENS_ESTIMATE — иначе 4 разных по размеру промпта
    # затирали бы одно и то же значение друг у друга.
    system_prompt, system_prompt_tokens_estimate = prompts._build_role_system_prompt(role, resolved_repo_path)
    num_keep = min(settings.get("num_ctx") // 2, system_prompt_tokens_estimate + 1500)

    model = _build_chat_model(
        model_tag=MAIN_MODEL,
        num_predict=OLLAMA_NUM_PREDICT,
        reasoning=settings.get("show_thinking"),
        num_keep=num_keep,
    )
    # Тот же принцип, что у judge_model в _build_agent: ОДИН тег с MAIN_MODEL
    # (не отдельная слабая модель — живой прогон показал ненадёжность
    # слабого судьи), format="json" безопасен именно здесь, потому что
    # judge_model нигде в этой роли не используется для обычного текста.
    judge_model = _build_chat_model(
        model_tag=MAIN_MODEL,
        num_predict=JUDGE_NUM_PREDICT,
        reasoning=False,
        num_keep=num_keep,
        format="json",
    )

    # delegate (delegate_tool.py) сознательно НЕ даётся ни одной роли
    # пайплайна, даже инвестигатору — живой прогон: у него ТОТ ЖЕ read-only
    # набор тулов (roles.py:LEGACY_INVESTIGATION_TOOL_NAMES), то есть это
    # вложенный саб-агент внутри уже отдельно бюджетируемой
    # исследовательской роли — лишняя матрёшка, а не разделение труда.
    # delegate был оправдан для ЛЕГАСИ монолитного агента (mcp_agent/
    # agent.py:_build_agent, тот случай ниже остаётся с ним), где всё
    # расследование делило ОДИН общий бюджет шагов на весь ход — здесь
    # инвестигатор получает свой собственный, достаточно большой бюджет
    # напрямую (см. mcp_agent/roles.py:ROLE_RECURSION_LIMIT) вместо того,
    # чтобы прятать часть его в непрозрачный (на живом прогоне — до 6+
    # минут без единого признака прогресса) вложенный вызов.
    agent_tools = filter_tools(tool_names, tools)

    if DEBUG:
        console.print(f"[dim][MCP-AGENT] Role '{role}' has {len(agent_tools)} tools available this turn: {[t.name for t in agent_tools]}[/]")
    log_event("tools_available", role=role, names=[t.name for t in agent_tools])

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={name: True for name in approval_tools(tool_names)},
    )
    compact_research = _CompactResearchMiddleware(judge_model)
    agent_middleware = [
        _ToolErrorGuardMiddleware(),
        _UnloadImageGenBeforeGenModelMiddleware(),
        middleware,
        _AskUserGuardMiddleware(),
        _OutOfProjectWriteApprovalMiddleware(resolved_repo_path),
        _DedupeToolResultsMiddleware(),
    ]
    if role == "planner":
        # Только Planner имеет ask_user (roles.py:planner_tools) — эта
        # мидлварь не даёт ему звать что-либо ещё после первого
        # подтверждения, см. её докстринг в ask_user_tool.py про живой
        # инцидент с 8 циклами "готов ли я...?".
        agent_middleware.append(_AskUserFinalizeMiddleware())
    if role == "verifier":
        # Только Verifier держит bash_exec БЕЗ единого write-тула рядом —
        # единственная роль, где bash_exec реально способен подменить
        # собой запись файла (sed -i и т.п.), см. докстринг мидлвари.
        agent_middleware.append(_VerifierNoSelfFixMiddleware())
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
    легаси-пути, здесь словарь, значение-ключ для сравнения свежести —
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
    бы отвечать (и писать/выполнять bash_exec) в НЕМ, даже когда следующий
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
    global _role_agent_cache, _role_agent_cache_key
    current_model = settings.get("chat_model")
    cache_key = (role, tool_names)
    current_value = (
        current_model, repo_path or os.getcwd(),
        settings.get("expert_streaming_enabled"), settings.get("num_ctx"),
    )
    cached_value = _role_agent_cache_key.get(cache_key)
    if cache_key in _role_agent_cache and cached_value == current_value:
        return _role_agent_cache[cache_key]
    async with _role_agent_cache_lock:
        cached_value = _role_agent_cache_key.get(cache_key)
        if cache_key not in _role_agent_cache or cached_value != current_value:
            old_model = cached_value[0] if cached_value is not None else None
            _role_agent_cache[cache_key] = await _build_role_agent(role, tool_names, repo_path)
            _role_agent_cache_key[cache_key] = current_value
            if old_model is not None and old_model != current_model:
                await _evict_ollama_model(old_model)
    return _role_agent_cache[cache_key]


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
    global _agent_cache, _agent_cache_key
    current_model = settings.get("chat_model")
    current_key = (
        current_model, settings.get("voice_mode"), repo_path or os.getcwd(),
        settings.get("optimized_tools"), settings.get("always_delegate_search"),
        settings.get("expert_streaming_enabled"), settings.get("num_ctx"),
    )
    if _agent_cache is not None and _agent_cache_key == current_key:
        return _agent_cache
    async with _agent_cache_lock:
        if _agent_cache is None or _agent_cache_key != current_key:
            old_model = _agent_cache_key[0] if _agent_cache_key is not None else None
            _agent_cache = await _build_agent(repo_path)
            _agent_cache_key = current_key
            if old_model is not None and old_model != current_model:
                await _evict_ollama_model(old_model)
    return _agent_cache


