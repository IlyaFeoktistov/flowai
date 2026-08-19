"""
Экспериментальный альтернативный backend для основной кодовой модели —
собственный llama-server (не бандл Ollama, а собранный из vendor/
llama-expert-streaming) с настоящим dynamic per-token expert caching для
MoE-моделей, вместо статичного once-at-load CPU/GPU сплита, который Ollama
использует всегда (см. ui/tui/settings.py:_MEASURED_GPU_SHARE — тот сплит
решается один раз при загрузке и не меняется по ходу генерации).

## Что за форк, откуда и почему именно он

Апстрим llama.cpp (ggml-org/llama.cpp) НЕ имеет в mainline никакого dynamic
expert offloading — только статичный `-ngl`/`-ncmoe`/`-ot`. Реальный
dynamic-caching код существует, но не влит: PR #26824 "Expert Caching..."
(автор miltos22, github.com/ggml-org/llama.cpp/pull/26824, открыт и закрыт
2026-08-10, наследник закрытого PR #26563) добавляет настоящий per-token
hot/cold кэш экспертов — heatmap с decay, кто "горячий" остаётся в VRAM,
кто "холодный" — в RAM, с real-time переносом между ними по мере того, как
роутер модели фактически выбирает экспертов, а не один раз решённым при
загрузке сплитом. Закрыт мейнтейнерами НЕ потому что не работает — по
процессным причинам (несколько backend'ов в одном PR, не было заранее
RFC-issue, см. github.com/ggml-org/llama.cpp/discussions/24528) — просто
альтернативная реализация той же идеи, которая уже обсуждалась отдельно.

На похожей MoE-модели того же класса (A3B) на 16GB карте: PP падает
примерно втрое от обычного, TG ускоряется — оценочно на треть. То есть
это НЕ чистый выигрыш, а обмен: prompt-processing проседает в разы, зато
генерация ускоряется. На 6GB карте, где и без того почти всё уходит на
CPU, соотношение может отличаться в любую сторону — отсюда сам смысл этого
модуля: дать реальный, воспроизводимый способ включить и измерить это на
СВОЁМ железе и модели, а не поверить на слово чужому бенчмарку.

На коротком тестовом промпте (не полноценный бенчмарк) эффект сильно
зависит от num_ctx: при num_ctx=8192 (меньше KV-cache, больше VRAM
остаётся под hot-store экспертов) PP=4.85 tok/s (не хуже Ollama),
TG=8.84 tok/s (+105% к Ollama). При РЕАЛЬНОМ проектном num_ctx=65536
(mcp_agent/model_config.py:OLLAMA_NUM_CTX) — PP=1.96, TG=3.95 tok/s, то
есть выигрыш заметно меньше (KV-cache отъедает VRAM, которая иначе пошла
бы в hot-store, autofit находит меньше горячих слотов). Цифры на
маленьком num_ctx не репрезентативны для реальной работы агента — мерить
нужно именно на своём num_ctx.

С РЕАЛЬНЫМ системным промптом Analyzer'а из prompts.py (2686 токенов, не
игрушечный) на num_ctx=65536 autofit находит 0 hot-slots ВООБЩЕ независимо
от типа KV-cache (проверено и на f16, и на -ctk/-ctv q8_0): fit.cpp жадно
отдаёт освободившуюся от квантования KV-cache память под ДОПОЛНИТЕЛЬНЫЕ
dense-слои на GPU (шаг, идущий ДО расчёта hot-store), а не резервирует её
под экспертов. На num_ctx=30000 с явным -ctk/-ctv q8_0 autofit находит
**10 hot-slots** (1438 МиБ) — но контрольный запуск (тот же промпт, тот
же num_ctx=30000, тот же -ctk/-ctv q8_0, но -ehs 0 — кэш экспертов
выключен ВООБЩЕ) даёт РЕЗУЛЬТАТ ЛУЧШЕ, а не хуже: PP 203 tok/s против 59
tok/s с 10 слотами, TG 16-20 tok/s против 13-21 (то есть не хуже, часто
лучше). common_memory_breakdown_print объясняет почему: без hot-store
резервации fit отдаёт под саму модель на GPU 2277 МиБ вместо 1055-737 МиБ
с hot-cache — сама фича dynamic per-token expert caching, ради которой
этот форк вообще существует, на ЭТОЙ карте (6 GB) и ЭТОЙ модели (30B MoE)
оказывается ЧИСТЫМ МИНУСОМ:
накладные расходы на подкачку экспертов в реальном времени дороже, чем
просто отдать fit'у больше GPU под статичные слои без всякого кэша. Реальный
выигрыш этого бэкенда над Ollama на данном железе — не в dynamic caching, а
в том, что сам этот билд llama.cpp иначе (эффективнее) считает -fit/-ngl,
никак не связано с самим PR26824. Отсюда — рабочий дефолт ниже: -ehs 0
(кэш выключен), не -ehs -1. Если апстрим когда-нибудь доведёт этот PR до
более лёгкого свопа — стоит перемерить; на сегодняшних числах включать его
на этой машине не за что.

Собирается через `python3 setup.py --only expert-streaming` (см. setup.py)
— клонирует ggml-org/llama.cpp в vendor/llama-expert-streaming, переключает
на ветку `expert-streaming-pr26824` (fetch pull/26824/head), собирает с
CUDA через отдельный venv-build-tools (cmake/ninja через pip --user --
никакого sudo, та же философия, что и остальной setup.py). Ничего из этого
не пересобирается на каждый запуск — один раз собранный бинарник лежит в
vendor/llama-expert-streaming/build/bin/llama-server и переиспользуется.

## GLM-4.7-Flash

Эта же ветка форка (`vendor/llama-expert-streaming`) несёт ещё два
собственных коммита — не про expert-caching, про совместимость с моделью
`glm-4.7-flash` (обычный `ollama pull glm-4.7-flash:q4_K_M`): её Ollama-GGUF
несёт `general.architecture = "glm4moelite"`, архитектуру, которую апстрим
llama.cpp до сих пор не поддерживает нативно (issue #18931) — форк алиасит
её на `deepseek2` прямо в загрузчике GGUF (тензоры и MLA-хпараметры почти
1:1 совпадают, кроме пары свопнутых ключей и принудительного
`head_count_kv=1`), плюс фиксит реальный баг остановки генерации (модель
зацикливалась / не умела кончить ход — GGUF нёс массив
`tokenizer.ggml.eos_token_ids` из 3 кандидатов, а код читал только
единственный, из-за чего 2 реальных стоп-токена GLM никогда не
регистрировались). Подробности расследования — история коммитов на
`flowai-expert-streaming` (git@github.com:IlyaFeoktistov/llama.cpp.git) и
шапка `vendor/llama-expert-streaming/README.md`. У этой модели нет
embedded chat-template в Ollama-GGUF вообще — `--chat-template-file`
передаётся отсюда автоматически (см. `_chat_template_file_for`) на уже
существующую тестовую фикстуру форка
(`models/templates/GLM-4.7-Flash.jinja`, байт-в-байт совпадает с
официальным zai-org/GLM-4.7-Flash).

## Точные флаги (проверено по исходникам ветки, common/arg.cpp)

  -ehs N / --expert-hot-s N   -1 = autofit слотов по свободной VRAM,
                               0 = выключено (обычное поведение), N = вручную
  --expert-sidecar            сохраняет/загружает heatmap рядом с моделью
                               (<model>.tier) — прогретый старт между запусками
  --expert-pin N              % холодных экспертов, закреплённых в RAM

ВАЖНО (из описания самого PR): `-ehs` НЕЛЬЗЯ сочетать с `-ncmoe` — второй
отбирает VRAM у статичных экспертов, которые нужны первому для кэша.
Использовать только `-ehs`, без `-ncmoe`/`-ot`.

## Чем этот backend отличается от обычного Ollama-пути — известные огрубления

agent_builder.py дополнительно теряет/огрубляет несколько вещей, которые
Ollama-путь считает per-запросно:
  - num_keep (см. agent_builder._ChatOllamaWithNumKeep) — здесь становится
    статичным значением при СТАРТЕ сервера (--keep), не пересчитывается на
    каждый вызов модели.
  - show_thinking/reasoning (Ollama's think=...) — тоже фиксируется при
    старте через --chat-template-kwargs, переключение в /settings во время
    работы уже запущенного экспериментального сервера не подхватится без
    перезапуска (ensure_running перезапускает сервер только при смене
    МОДЕЛИ, не при смене этого параметра — см. ensure_running).
  - keep_alive/автовыгрузка Ollama — не применимо, жизненным циклом
    процесса управляет этот модуль (start/stop/ensure_running), а не демон
    Ollama.

Покрывает ОБА пути через agent_builder._build_chat_model — легаси-агент
(_build_agent, pipeline_mode=ВЫКЛ) И роли нового пайплайна (_build_role_agent,
вызывается из mcp_agent/pipeline.py при pipeline_mode=ВКЛ) — обе функции
строят модель с одним и тем же тегом settings.chat_model через один и тот
же _build_chat_model, так что тумблер действует независимо от pipeline_mode.
"""
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import storage

# llama-server's own stdout/stderr — captured to a file (not DEVNULL, not a
# subprocess.PIPE we never read) so a startup failure can show the model's
# OWN error line instead of a guess. A generic fallback message like
# "процесс завершился сам (код 1) до готовности — модель
# большая, загрузка с диска не быстрая, но не столько" is meaningless filler
# text for EVERY failure, whatever the real cause — e.g. a stale process
# from manual testing still holding the port fails with
# "couldn't bind HTTP server socket", a one-line, instantly diagnosable
# reason that generic text would hide entirely. One shared file, reused every run — this
# is a debug aid, not a durable log a user would want to keep across runs.
_LOG_PATH = storage.data_dir() / "expert_streaming_server.log"

# Written by whichever flowai process actually starts the server, read by any
# OTHER flowai process that later finds the port already healthy (see
# ensure_running's "занят каким-то другим процессом" branch below) — a
# second terminal running flowai concurrently used to always hit that branch
# and fall back to the plain Ollama path, which glm-4.7-flash can't actually
# run on (see CLAUDE.md). One shared file, same durability tier as
# _LOG_PATH — a live server's identity, not data worth keeping across
# reboots. _proc.pid recorded here is the SERVER's own pid (this module's
# Popen child IS the server), not the flowai process's — deliberately, so a
# server that outlives a crashed/restarted flowai (see ensure_running's own
# comment on that) is still correctly recognized as alive by its own pid.
_STATE_PATH = storage.data_dir() / "expert_streaming_server.json"

# .parent.parent, not .parent — expert_streaming.py lives one level deep
# under src/ (src/expert_streaming.py), but vendor/ is a real repo-root
# directory (the llama.cpp fork's build tree) that never moved there.
FLOWAI_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = FLOWAI_ROOT / "vendor" / "llama-expert-streaming"
SERVER_BINARY = VENDOR_DIR / "build" / "bin" / "llama-server"

DEFAULT_PORT = 8090
DEFAULT_HOST = "127.0.0.1"

# Ollama хранит блобы под /usr/share/ollama/.ollama (систем-юнит) ИЛИ
# ~/.ollama (обычный локальный демон) в зависимости от того, как его
# поставили — не угадываем одно, пробуем оба по порядку, первый существующий
# выигрывает. Переопределяемо через OLLAMA_MODELS (тот же env var, что
# понимает сам `ollama serve`), если пользователь держит модели ещё где-то.
def _candidate_ollama_models_dirs() -> list[Path]:
    override = os.getenv("OLLAMA_MODELS")
    if override:
        return [Path(override)]
    return [
        Path("/usr/share/ollama/.ollama/models"),
        Path.home() / ".ollama" / "models",
    ]


def _find_manifest_path(model_tag: str) -> Path | None:
    """model_tag вида "qwen3-coder:30b" -> ищем файл манифеста, оканчивающийся
    на .../<name>/<tag>, под manifests/ — НЕ хардкодим registry.ollama.ai/
    library (только это покрывало бы исключительно `ollama pull <name>:<tag>`
    из официального реестра): раскладка Ollama на диске всегда
    manifests/<registry-host>/<namespace>/<name>/<tag> — ровно 4 уровня,
    независимо от того, откуда модель реально притянута (обычный library,
    кастомный registry, `ollama pull hf.co/...`), так что glob по этим
    последним двум компонентам находит манифест вне зависимости от первых
    двух — то, что реально нужно другому человеку на другом компе с другим
    способом установки той же модели, а не только "как поставил я"."""
    name, _, tag = model_tag.partition(":")
    tag = tag or "latest"
    for models_dir in _candidate_ollama_models_dirs():
        manifests_root = models_dir / "manifests"
        if not manifests_root.is_dir():
            continue
        for candidate in manifests_root.glob(f"*/*/{name}/{tag}"):
            if candidate.is_file():
                return candidate
    return None


def resolve_model_blob_path(model_tag: str) -> Path | None:
    """Читает манифест Ollama и достаёт путь до реального GGUF-блоба веса
    модели (не license/params — слой mediaType
    application/vnd.ollama.image.model) — тот же файл, что грузит `ollama
    run`, просто напрямую, без демона Ollama между нами и им."""
    manifest_path = _find_manifest_path(model_tag)
    if manifest_path is None:
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            digest = layer.get("digest", "")
            if not digest.startswith("sha256:"):
                continue
            blob_name = "sha256-" + digest.removeprefix("sha256:")
            blob_path = manifest_path.parents[4] / "blobs" / blob_name
            return blob_path if blob_path.is_file() else None
    return None


def is_built() -> bool:
    return SERVER_BINARY.is_file()


# GLM-4.7-Flash's Ollama GGUF has no embedded chat template at all (Ollama
# renders it separately, in its own Go code — see the fork's own commit
# history on the flowai-expert-streaming branch, vendor/llama-expert-
# streaming, for the full investigation) — --jinja alone has nothing to
# render for this one model, unlike every other model this module runs,
# whose GGUF already carries its own tokenizer.chat_template. The real
# template (byte-identical to zai-org/GLM-4.7-Flash's official
# chat_template.jinja) already ships in the fork itself as a test fixture —
# reused directly rather than duplicating it into this repo.
_GLM_4_7_FLASH_CHAT_TEMPLATE = VENDOR_DIR / "models" / "templates" / "GLM-4.7-Flash.jinja"


def _chat_template_file_for(model_tag: str) -> Path | None:
    if model_tag.partition(":")[0] == "glm-4.7-flash":
        return _GLM_4_7_FLASH_CHAT_TEMPLATE
    return None


# Per-model llama-server launch overrides -- same idea as
# _chat_template_file_for above, for flags instead of the template.
#
# glm-4.7-flash: live measurement (2026-08-14, this machine, a ~2.5k-token
# prompt) -- "--no-mmap --direct-io" took prompt processing from 197.7 to
# 288.9 tok/s (+46%), generation unchanged (~18-19 tok/s either way).
# llama.cpp's own loader already warns about exactly this case for this
# model ("tensor overrides to CPU are used with mmap enabled - consider
# using --no-mmap for better performance" -- autofit here always partially
# offloads GLM's MoE layers to CPU on a 6 GB card), just never wired up
# before. Community-recommended for this model too (HF discussion
# zai-org/GLM-4.7-Flash#66). q4_0 KV cache (vs this project's usual q8_0
# elsewhere) is also from that same discussion -- not applied to any other
# model here, since it's a precision/quality tradeoff never measured for
# them specifically. -ehs -1 (dynamic expert hot-store) was re-measured
# for this model too, same live test: WORSE across the board (prompt
# 288.9 -> 65.8 tok/s, generation 18.2 -> 12.8 tok/s) -- same conclusion as
# qwen3-coder:30b (see the -ehs 0 comment below), so it stays off here too,
# nothing to override.
_GLM_4_7_FLASH_EXTRA_FLAGS = ["--no-mmap", "--direct-io"]
_GLM_4_7_FLASH_CACHE_TYPE = "q4_0"


def _extra_server_flags_for(model_tag: str) -> list[str]:
    if model_tag.partition(":")[0] == "glm-4.7-flash":
        return _GLM_4_7_FLASH_EXTRA_FLAGS
    return []


def _cache_type_for(model_tag: str) -> str:
    if model_tag.partition(":")[0] == "glm-4.7-flash":
        return _GLM_4_7_FLASH_CACHE_TYPE
    return "q8_0"


_proc: subprocess.Popen | None = None
# (model_tag, num_ctx, show_thinking) the CURRENT _proc was actually started
# with — not just model_tag. num_ctx is fixed at process startup (-c), so a
# num_ctx-only settings change (e.g. testing 32768 vs 65536) with the SAME
# model_tag used to be silently ignored — is_running()/this tag matched, so
# ensure_running kept serving the OLD context instead of restarting.
_proc_config: tuple[str, int, bool] | None = None

# agent_builder._build_chat_model calls ensure_running TWICE per agent build
# (main model, then judge_model — same model_tag) — without this, a failure
# means spawning the ~18 GB model process twice in a row and printing the
# same warning twice, back to back. Short-lived negative cache: a failure
# for the same model_tag within
# _FAILURE_COOLDOWN_S returns the SAME (False, msg) instantly instead of
# repeating a doomed multi-second startup attempt. Not cached on success —
# is_running()/_proc_config above already short-circuits that case.
_FAILURE_COOLDOWN_S = 30.0
_last_failure: tuple[str, float, str] | None = None


def _health_check(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{DEFAULT_HOST}:{port}/health", timeout=1) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def is_running() -> bool:
    return _proc is not None and _proc.poll() is None


def _write_state(pid: int, port: int, model_tag: str, num_ctx: int, show_thinking: bool) -> None:
    try:
        _STATE_PATH.write_text(json.dumps({
            "pid": pid, "port": port, "model_tag": model_tag,
            "num_ctx": num_ctx, "show_thinking": show_thinking,
        }))
    except OSError:
        pass  # best-effort — worst case, the NEXT process to find this port occupied fails loud instead of adopting it


def _clear_state() -> None:
    try:
        _STATE_PATH.unlink()
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False  # os.kill(0, ...)/negative pid target a process GROUP, not a single pid
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _adoptable_state(port: int, model_tag: str, num_ctx: int, show_thinking: bool) -> dict | None:
    """None unless _STATE_PATH names a server that's (a) still alive by its
    own recorded pid and (b) configured EXACTLY like what THIS call is
    asking for — same guard the module already applies to its OWN _proc via
    _proc_config, just readable across process boundaries. A mismatch on
    any field (including a state file from a different port/model
    altogether) means "not proven safe to adopt", same treatment as no
    state file at all — the caller falls through to the existing
    port-occupied failure, never guesses."""
    try:
        state = json.loads(_STATE_PATH.read_text())
    except (OSError, ValueError):
        return None
    if (state.get("port"), state.get("model_tag"), state.get("num_ctx"), state.get("show_thinking")) != (
        port, model_tag, num_ctx, show_thinking
    ):
        return None
    pid = state.get("pid")
    if not isinstance(pid, int) or not _pid_alive(pid):
        return None
    return state


def _find_port_holder_pids(port: int) -> list[str]:
    """Best-effort PID lookup for whatever's LISTENING on `port` — never
    kills anything (a process on this port could be unrelated to flowai
    entirely; guessing wrong and killing it would be worse than the
    original bug). -sTCP:LISTEN — plain `lsof -ti tcp:{port}` also matches
    our OWN outbound client connection to this same port (the ChatOpenAI
    client agent_builder.py builds), which would misleadingly show up as
    if "someone's holding the port" even when it's just this same flowai
    process talking to its own already-running server; only the listening
    socket is the actual holder worth reporting. Returns [] if `lsof`
    isn't installed or nothing's listening — callers treat that the same
    as "unknown holder"."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"], capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [p for p in result.stdout.split() if p.strip()]


def stop_server() -> None:
    global _proc, _proc_config
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _proc.kill()
        _clear_state()  # only OUR OWN tracked process's death invalidates the state file — an adopted server we never owned is left running, its own state untouched
    _proc = None
    _proc_config = None


def ensure_running(
    model_tag: str,
    port: int = DEFAULT_PORT,
    num_ctx: int = 65536,
    show_thinking: bool = False,
    wait_seconds: float = 120.0,
) -> tuple[bool, str]:
    """Запускает llama-server (если ещё не запущен с ТЕМИ ЖЕ model_tag/
    num_ctx/show_thinking — смена любого из них перезапускает процесс,
    иначе переиспользует живой) с -ehs -1 (autofit кэша экспертов по
    свободной VRAM, см. модульный docstring про то, почему НЕ -ncmoe рядом).
    Возвращает
    (ok, message) вместо исключения — вызывающий код (agent_builder) должен
    суметь откатиться на обычный Ollama-путь, если что-то пошло не так,
    а не уронить весь ход.

    "Переиспользует живой" относится и к серверу, поднятому ДРУГИМ flowai-
    процессом, не только этим самым (см. _STATE_PATH/_adoptable_state) — до
    этого второй параллельный `flowai` с той же дефолтной моделью (требующей
    expert_streaming_enabled=ВКЛ, см. CLAUDE.md) всегда попадал в ветку
    "порт занят" ниже и откатывался на обычный Ollama-путь, на котором эта
    архитектура физически не работает.

    Ollama сама держит модель резидентной и перезагружает её по своему
    keep_alive независимо от этого модуля — если наш процесс пытается
    стартовать, пока Ollama-инстанс всё ещё занимает VRAM под ту же
    модель, он падает сразу же (autofit не находит места) на GPU с
    ограниченным объёмом памяти, где обе копии одновременно попросту не
    помещаются. Выгружаем Ollama-копию ПЕРЕД стартом, а не полагаемся на
    то, что вызывающий код сделал это сам."""
    global _proc, _proc_config, _last_failure

    if not is_built():
        return False, (
            f"бинарник не собран: {SERVER_BINARY} не существует — запусти "
            "`python3 setup.py --only expert-streaming`"
        )

    if is_running() and _proc_config == (model_tag, num_ctx, show_thinking):
        return True, "already running"

    if _last_failure is not None:
        failed_tag, failed_at, failed_msg = _last_failure
        if failed_tag == model_tag and time.monotonic() - failed_at < _FAILURE_COOLDOWN_S:
            return False, failed_msg

    stop_server()

    def _fail(msg: str) -> tuple[bool, str]:
        global _last_failure
        _last_failure = (model_tag, time.monotonic(), msg)
        return False, msg

    # A llama-server this module started earlier can
    # outlive a crashed/restarted flowai process (Popen below has no
    # start_new_session, but that's not even the point — a FRESH flowai run
    # has its own empty _proc/_proc_config regardless, no memory of any
    # earlier process either way). If something ELSE is still answering on
    # `port` right after stop_server() above (which only ever stops a
    # process THIS run itself tracks), starting our own new server would
    # just fail to bind — but worse, the health-check loop below can't tell
    # "my new process is up" apart from "someone else already answers here"
    # and would wrongly report success while every request silently keeps
    # going to that OTHER process's actual (possibly stale) config — e.g. a
    # num_ctx change via /settings looked like it had no effect at all,
    # because an orphaned process from a previous run was still serving the
    # old context size the whole time. Fail loudly and specifically instead
    # of guessing — never kill it ourselves, it might not even be ours.
    #
    # EXCEPT when _STATE_PATH proves it's safe: a server that's (a) alive by
    # its own recorded pid and (b) configured EXACTLY like what we're asking
    # for right now (see _adoptable_state) — most likely another flowai
    # process's server, started with the same settings.py config this one
    # just loaded from the same shared SQLite file. That's "already running"
    # in every way that matters, not a stale/foreign process to fail on.
    if _health_check(port):
        adopted = _adoptable_state(port, model_tag, num_ctx, show_thinking)
        if adopted is not None:
            # _proc stays None on purpose — we didn't start this process, so
            # stop_server() must never try to kill it.
            _proc_config = (model_tag, num_ctx, show_thinking)
            return True, f"already running (started by another flowai process, pid {adopted['pid']})"
        pids = _find_port_holder_pids(port)
        if pids:
            how_to_stop = f"останови вручную: `kill {' '.join(pids)}`"
        else:
            how_to_stop = f"найди его (`lsof -ti tcp:{port}`) и останови вручную"
        return _fail(
            f"порт {port} уже занят каким-то другим процессом"
            + (f" (PID {', '.join(pids)})" if pids else "")
            + " — не отслеживается этим запуском flowai, возможно осиротевший "
            f"llama-server от предыдущего запуска; {how_to_stop}, если он не нужен"
        )

    blob_path = resolve_model_blob_path(model_tag)
    if blob_path is None:
        return _fail(f"не нашёл GGUF-блоб для '{model_tag}' в манифестах Ollama")

    try:
        import model_lifecycle
        model_lifecycle.unload_ollama_model(model_tag)
    except Exception:
        pass  # best-effort — если не вышло, autofit ниже просто увидит меньше свободной VRAM

    cmd = [
        str(SERVER_BINARY),
        "-m", str(blob_path),
        "-c", str(num_ctx),
        # -np 1 — по умолчанию (-1 = auto) сервер сам завёл 4 параллельных
        # слота (подтверждено живым /slots на этой машине), каждый со своим
        # куском KV-cache, хотя agent_builder.py никогда не шлёт больше
        # одного запроса одновременно на этот сервер — 3 неиспользуемых
        # слота отъедали VRAM/RAM, которые autofit мог бы отдать под hot-store
        # экспертов вместо этого.
        "-np", "1",
        # НЕ передавать -ngl здесь — с явным "-ngl 999" рядом autofit
        # безусловно бросает
        # "-ehs -1 autofit aborted (explicit -ngl/-ncmoe or fit error);
        # expert cache is OFF" — -fit (авто-подбор -ngl под свободную VRAM,
        # common/common.h: fit_params=true) обязан остаться единственным,
        # кто решает -ngl.
        #
        # -ehs 0 — кэш экспертов ВЫКЛЮЧЕН НАМЕРЕННО, не забыли включить.
        # Замер (эта машина, реальный системный промпт
        # Analyzer'а, num_ctx=30000, q8_0 KV-cache): -ehs -1 нашёл автофитом
        # 10 hot-slots (1438 МиБ) и дал PP=59 tok/s, TG=13-21 tok/s; тот же
        # прогон с -ehs 0 дал PP=203 tok/s, TG=16-20 tok/s — то есть БЕЗ
        # кэша быстрее почти по всем метрикам, не только по PP. Причина —
        # common_memory_breakdown_print: без резервации под hot-store fit
        # отдаёт под саму модель на GPU 2277 МиБ вместо 1055-737 — накладные
        # расходы на живую подкачку экспертов дороже эффекта от кэша именно
        # на этой карте (6 GB) и этой модели (30B MoE). Не включать -ehs -1
        # обратно без повторного замера на актуальном железе/модели/промпте.
        "-ehs", "0",
        "-ctk", _cache_type_for(model_tag), "-ctv", _cache_type_for(model_tag),
        "--host", DEFAULT_HOST,
        "--port", str(port),
        "--jinja",
        # --chat-template-kwargs '{"enable_thinking":...}' устарел (сервер
        # печатает deprecation warning) — актуальный способ тот же переключатель.
        "--reasoning", "on" if show_thinking else "off",
    ]
    cmd += _extra_server_flags_for(model_tag)
    chat_template_file = _chat_template_file_for(model_tag)
    if chat_template_file is not None:
        cmd += ["--chat-template-file", str(chat_template_file)]
    log_file = open(_LOG_PATH, "wb")
    try:
        _proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    except OSError as e:
        log_file.close()
        return _fail(f"не удалось запустить процесс: {e}")
    finally:
        log_file.close()  # child inherited its own fd on Popen; safe to close ours

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if _proc.poll() is not None:
            return _fail(f"процесс завершился сам (код {_proc.returncode}) — {_tail_log()}")
        if _health_check(port):
            _proc_config = (model_tag, num_ctx, show_thinking)
            _write_state(_proc.pid, port, model_tag, num_ctx, show_thinking)
            return True, "started"
        time.sleep(0.5)

    stop_server()
    return _fail(f"не поднялся за {wait_seconds:.0f}с (health-check не отвечает) — {_tail_log()}")


def _tail_log(n_lines: int = 3) -> str:
    """Последние строки log-файла llama-server — реальная причина сбоя
    (например "couldn't bind HTTP server socket... port: 8090" вместо
    молчаливого кода возврата) — см. модульный комментарий про _LOG_PATH."""
    try:
        lines = _LOG_PATH.read_text(errors="replace").splitlines()
    except OSError:
        return f"лог не прочитать: {_LOG_PATH}"
    if not lines:
        return f"лог пуст: {_LOG_PATH}"
    return f"последние строки лога ({_LOG_PATH}):\n" + "\n".join(lines[-n_lines:])
