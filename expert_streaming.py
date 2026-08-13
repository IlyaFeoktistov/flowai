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

Живой отзыв прямо в треде PR (miltos22, 2026-08-10T18:27:13Z, MoE-модель
того же класса, A3B, на 16GB карте): "PP is very low, around a third of
usual. TG seems faster... I estimate TG to be about 33% faster" — то есть
это НЕ чистый выигрыш, а обмен: prompt-processing проседает в разы, зато
генерация ускоряется. На 6GB карте, где и без того почти всё уходит на
CPU, соотношение может отличаться в любую сторону — отсюда сам смысл этого
модуля: дать реальный, воспроизводимый способ включить и измерить это на
СВОЁМ железе и модели, а не поверить на слово чужому бенчмарку.

Живые замеры на этой машине (2026-08-11, короткий тестовый промпт, НЕ
полноценный бенчмарк) — эффект сильно зависит от num_ctx: при num_ctx=8192
(меньше KV-cache, больше VRAM остаётся под hot-store экспертов) PP=4.85
tok/s (не хуже Ollama), TG=8.84 tok/s (+105% к Ollama). При РЕАЛЬНОМ
проектном num_ctx=65536 (mcp_agent/model_config.py:OLLAMA_NUM_CTX) —
PP=1.96, TG=3.95 tok/s, то есть выигрыш заметно меньше (KV-cache отъедает
VRAM, которая иначе пошла бы в hot-store, autofit находит меньше горячих
слотов). Не считать цифры на маленьком num_ctx репрезентативными для
реальной работы агента — мерить нужно именно на своём num_ctx.

Повторный живой замер (2026-08-13, эта же машина, РЕАЛЬНЫЙ системный промпт
Analyzer'а из prompts.py, 2686 токенов, не игрушечный) — на num_ctx=65536
autofit находил 0 hot-slots ВООБЩЕ независимо от типа KV-cache (проверено и
на f16, и на -ctk/-ctv q8_0): fit.cpp жадно отдаёт освободившуюся от
квантования KV-cache память под ДОПОЛНИТЕЛЬНЫЕ dense-слои на GPU (шаг,
идущий ДО расчёта hot-store), а не резервирует её под экспертов. На
num_ctx=30000 с явным -ctk/-ctv q8_0 autofit нашёл **10 hot-slots** (1438
МиБ) — но решающий контрольный замер (тот же промпт, тот же num_ctx=30000,
тот же -ctk/-ctv q8_0, но -ehs 0 — кэш экспертов выключен ВООБЩЕ) дал
РЕЗУЛЬТАТ ЛУЧШЕ, а не хуже: PP 203 tok/s против 59 tok/s с 10 слотами, TG
16-20 tok/s против 13-21 (то есть не хуже, часто лучше). common_memory_
breakdown_print объясняет почему: без hot-store резервации fit отдаёт под
саму модель на GPU 2277 МиБ вместо 1055-737 МиБ с hot-cache — сама фича
dynamic per-token expert caching, ради которой этот форк вообще существует,
на ЭТОЙ карте (6 GB) и ЭТОЙ модели (30B MoE) оказалась ЧИСТЫМ МИНУСОМ:
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
# OWN error line instead of a guess. Live bug (user report): the fallback
# message used to be "процесс завершился сам (код 1) до готовности — модель
# большая, загрузка с диска не быстрая, но не столько" — meaningless filler
# text for EVERY failure, whatever the real cause (that specific live
# failure was a stale process from manual testing still holding the port —
# "couldn't bind HTTP server socket" — a one-line, instantly diagnosable
# reason that text hid entirely). One shared file, reused every run — this
# is a debug aid, not a durable log a user would want to keep across runs.
_LOG_PATH = storage.data_dir() / "expert_streaming_server.log"

FLOWAI_ROOT = Path(__file__).resolve().parent
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


_proc: subprocess.Popen | None = None
# (model_tag, num_ctx, show_thinking) the CURRENT _proc was actually started
# with — not just model_tag. num_ctx is fixed at process startup (-c), so a
# num_ctx-only settings change (e.g. testing 32768 vs 65536) with the SAME
# model_tag used to be silently ignored — is_running()/this tag matched, so
# ensure_running kept serving the OLD context instead of restarting.
_proc_config: tuple[str, int, bool] | None = None

# agent_builder._build_chat_model calls ensure_running TWICE per agent build
# (main model, then judge_model — same model_tag) — without this, a failure
# meant spawning the ~18 GB model process twice in a row and printing the
# same warning twice (live bug report: exact duplicate warning line back to
# back). Short-lived negative cache: a failure for the same model_tag within
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

    Живой прогон (2026-08-11, эта же машина, 6 GB VRAM): Ollama сама
    держит qwen3-coder:30b резидентной и перезагружает её по своему
    keep_alive независимо от этого модуля — второй процесс (наш) пытался
    стартовать, пока Ollama-инстанс всё ещё занимал ~4 GB VRAM, и падал
    сразу же (autofit не находит места). Обе копии одной и той же модели
    в 6 GB одновременно попросту не помещаются — выгружаем Ollama-копию
    ПЕРЕД стартом, а не полагаемся на то, что вызывающий код сделал это сам."""
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

    # Live bug (user report): a llama-server this module started earlier can
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
    if _health_check(port):
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
        # НЕ передавать -ngl здесь — живой прогон (2026-08-11, эта же
        # машина): с явным "-ngl 999" рядом autofit безусловно бросает
        # "-ehs -1 autofit aborted (explicit -ngl/-ncmoe or fit error);
        # expert cache is OFF" — -fit (авто-подбор -ngl под свободную VRAM,
        # common/common.h: fit_params=true) обязан остаться единственным,
        # кто решает -ngl.
        #
        # -ehs 0 — кэш экспертов ВЫКЛЮЧЕН НАМЕРЕННО, не забыли включить.
        # Живой замер (2026-08-13, эта же машина, реальный системный промпт
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
        "-ctk", "q8_0", "-ctv", "q8_0",
        "--host", DEFAULT_HOST,
        "--port", str(port),
        "--jinja",
        # --chat-template-kwargs '{"enable_thinking":...}' устарел (сервер
        # печатает deprecation warning) — актуальный способ тот же переключатель.
        "--reasoning", "on" if show_thinking else "off",
    ]
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
