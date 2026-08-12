import curses
from typing import Callable
import settings
from ui.tui.curses_util import flush_pending_input

# (model_id, описание, steps, guidance)
_IMGGEN_MODELS = [
    ("black-forest-labs/FLUX.1-schnell",         "лучшее качество, ~33GB скачать",  4,  0.0),
    ("stabilityai/sdxl-turbo",                  "быстрая, ~6.5GB скачать",          4,  0.0),
    ("ByteDance/SDXL-Lightning",                 "быстрая + качество, ~7GB скачать", 4,  0.0),
    ("stabilityai/stable-diffusion-xl-base-1.0", "высокое качество, ~6.5GB скачать",25,  7.5),
    ("stabilityai/stable-diffusion-2-1",         "хорошее качество, ~5.2GB скачать",25,  7.5),
    ("runwayml/stable-diffusion-v1-5",           "лёгкая классика, ~4GB скачать",   25,  7.5),
]

_GUIDANCE_PRESETS = [
    (0.0,  "distilled-модели: turbo, Lightning"),
    (3.5,  "мягкое следование промпту"),
    (7.0,  "стандарт для SD 1.5 / SD 2.1"),
    (7.5,  "стандарт для SDXL"),
    (10.0, "строгое следование промпту"),
    (15.0, "очень строгое, может перенасытить"),
]

# Только 4 и 8 — единственные шаги, под которые реально есть скачанный
# чекпойнт SDXL-Lightning (см. mcp_agent/servers/image_gen_server.py:
# _LIGHTNING_CKPTS) — любое другое число для Lightning тихо откатится на
# 4-шаговые веса, оставаясь на этом числе шагов денойзинга, то есть веса и
# шаги разъедутся. "своё значение" ниже — для остальных моделей (SDXL
# base/SD 1.5/2.1 в _IMGGEN_MODELS выше берут 25).
_STEPS_PRESETS = [
    (4, "быстрее — чекпойнт Lightning 4-step"),
    (8, "детальнее — чекпойнт Lightning 8-step"),
]

# 0.05 шагом от почти "без изменений" до почти "с нуля" — весь диапазон,
# который в принципе принимает diffusers img2img (0 не имеет смысла: пайплайн
# не тронет картинку вообще; >1 тоже не имеет смысла для этой формулы).
_STRENGTH_PRESETS = [
    (round(0.05 * i, 2), {1: "почти без изменений", 10: "среднее", 20: "почти с нуля"}.get(i, ""))
    for i in range(1, 21)
]

# faster-whisper модели по имени — компромисс скорость/точность, см.
# https://github.com/SYSTRAN/faster-whisper. "своё значение" ниже покрывает
# составные теги вроде distil-large-v3, которых нет в этом коротком списке.
_STT_MODEL_PRESETS = [
    ("tiny",     "самая быстрая, ниже точность"),
    ("base",     "быстро, базовая точность"),
    ("small",    "баланс"),
    ("medium",   "точнее, медленнее (дефолт)"),
    ("large-v3", "максимальная точность, медленно на CPU"),
]

# /gen_model (gen3d/) — см. settings.py про сами дефолты и 3dtodo.md про
# замеры VRAM, откуда взяты цифры в описаниях.
_GEN3D_FACES_PRESETS = [
    (1000,  "low-poly стиль, минимальный размер"),
    (5000,  "лёгкий игровой ассет"),
    (15000, "дефолт — детально, но не избыточно"),
    (30000, "высокая детализация"),
    (50000, "максимум, почти без сжатия исходного меша"),
]
_GEN3D_SKIN_SOURCE_PRESETS = [
    ("auto_weights", "Blender Automatic Weights — легче, без доп. VRAM (дефолт)"),
    ("unirig",        "UniRig skin-модель — точнее, но пик VRAM впритык (+1.6 ГБ)"),
]
_GEN3D_PROFILE_PRESETS = [
    (4, "дефолт — больше запаса VRAM"),
    (3, "чуть быстрее (~3%), но пик VRAM почти вдвое выше"),
]

# Контекст ГЛАВНОЙ чат/judge-модели (settings.num_ctx — см. его докстринг в
# settings.py про то, почему это ОТДЕЛЬНЫЙ тумблер от model_config.OLLAMA_NUM_CTX,
# и на что он НЕ распространяется). 65536 — посчитанный потолок под 24 GB
# RAM этой машины (см. model_config.py) — меньшие значения безопасны
# (меньше RAM/VRAM под KV-cache), больше — риск свопа на длинных ходах.
_NUM_CTX_PRESETS = [
    (8192,   "минимум — рискует обрезать длинную историю тул-вызовов"),
    (16384,  ""),
    (32768,  "половина текущего дефолта"),
    (65536,  "дефолт этого проекта — посчитанный потолок под 24 GB RAM"),
    (131072, "выше проверенного потолка — риск свопа на длинных ходах"),
]

# key -> (заголовок экрана выбора, список пресетов, тип для "своего значения")
_PRESET_CONFIGS = {
    "imggen_guidance": ("guidance scale",           _GUIDANCE_PRESETS,  float),
    "imggen_steps":    ("imggen шаги",               _STEPS_PRESETS,     int),
    "imggen_strength": ("imggen сила (0-1)",         _STRENGTH_PRESETS,  float),
    "stt_model":       ("распознавание речи (whisper)", _STT_MODEL_PRESETS, str),
    "gen3d_target_faces":   ("gen_model целевой полигонаж", _GEN3D_FACES_PRESETS, int),
    "gen3d_skin_source":    ("gen_model источник скиннинга (--rig)", _GEN3D_SKIN_SOURCE_PRESETS, str),
    "gen3d_hunyuan_profile": ("gen_model профиль offload", _GEN3D_PROFILE_PRESETS, int),
    "num_ctx":         ("контекст чат-модели (num_ctx)", _NUM_CTX_PRESETS, int),
}


# (model_id | None, описание)  — None = разделитель-заголовок
# показываются даже если не установлены. Размер VRAM в списке НЕ отсюда —
# он дописывается динамически при отрисовке (см. _size_suffix): реальный
# для установленных (из `ollama list`), проверенный по ollama.com/library
# для остальных — вместо того чтобы гадать словами вроде "впритык"/
# "вероятно" пользователь сам видит цифры и решает сам.
#
# ВАЖНО про "влезает": просто вес_модели/TOTAL_VRAM_GB (см. _size_suffix)
# НЕ означает "влезает целиком на GPU" — это доля ТОЛЬКО весов, без
# KV-cache. Агент всегда грузит модель с OLLAMA_NUM_CTX=65536
# (mcp_agent/model_config.py), и KV-cache на таком контексте у большинства
# моделей сам по себе больше пары ГБ — реальную долю на GPU/CPU показывает
# только `ollama ps` ПОСЛЕ фактической загрузки модели. Живой замер на этой
# машине (RTX 4050 Laptop, 5.9 GB VRAM, num_ctx=65536, 2026-08-11) — см.
# _MEASURED_GPU_SHARE ниже: из установленных моделей ПОЛНОСТЬЮ на GPU
# влезает только qwen2.5:3b (100%), qwen3:8b — только 48%, а не "влезает,
# потому что 5.2 < 5.9". Если поменяется OLLAMA_NUM_CTX или сама модель —
# эти цифры устареют, переизмерить тем же способом (curl .../api/generate
# с нужным num_ctx, затем `ollama ps`).
_MEASURED_GPU_SHARE = {
    "qwen2.5:3b":      100,  # CONTEXT capped 32768, итого 2.8GB
    "qwen3:4b":         50,  # CONTEXT 65536, итого 8.3GB
    "qwen3:8b":         48,  # CONTEXT capped 40960, итого 8.8GB
    "qwen2.5vl:7b":     45,  # CONTEXT 65536, итого 8.1GB
    "dolphin3:latest":  40,  # CONTEXT 65536, итого 10GB
    "llava:13b":        44,  # CONTEXT capped 4096, итого 9.4GB
    "qwen2.5:14b":      32,  # CONTEXT capped 32768, итого 12GB
    "qwen3:14b":        30,  # CONTEXT capped 40960, итого 13GB
    "qwen3-coder:30b":  19,  # CONTEXT 65536, итого 22GB
}
_SUGGESTED: dict[str, list[tuple]] = {
    "chat_model": [
        (None,                   "── тяжёлые ─────────────────"),
        ("qwen3:14b",            "reasoning"),
        ("qwen2.5:14b",          "стабильный"),
        ("qwen2.5-coder:14b",    "кодинг"),
        ("qwq:32b",              "глубокое мышление"),
        (None,                   "── MoE (тяжёлая VRAM, лёгкое вычисление) ──"),
        ("gpt-oss:20b",          "★ рекомендуется (с expert_streaming_enabled=ВКЛ) · "
                                  "agentic/тул-коллы, MoE ~3.6B активных из 20B — на "
                                  "expert-streaming тот же класс скорости, что и "
                                  "qwen3-coder:30b там же, но заметно меньший суммарный "
                                  "вес (13 GB против 18-19 GB), значит меньше данных "
                                  "стримить на каждое переключение эксперта. На ОБЫЧНОМ "
                                  "Ollama-пути (expert_streaming_enabled=ВЫКЛ) падает при "
                                  "OLLAMA_KV_CACHE_TYPE=q8_0 (GGML_ASSERT, известный баг "
                                  "Ollama #16946) — без expert-streaming лучше не выбирать "
                                  "эту модель, agent_builder.py явно откажет с понятной "
                                  "ошибкой вместо тихого краха (проверено 2026-08-11)"),
        ("qwen3-coder:30b",      "agentic-кодинг · MoE, ~3.3B активных из 30B — "
                                  "тул-коллы заточены под агентный кодинг, работает и без "
                                  "expert-streaming (обычный Ollama-путь), берёт больше VRAM"),
        (None,                   "── средние ─────────────────"),
        ("qwen3.5:9b",           "новее"),
        ("qwen3:8b",             "быстрый"),
        ("qwen2.5:7b",           "стабильный"),
        ("qwen2.5-coder:7b",     "кодинг"),
        ("dolphin3:latest",      ""),
        (None,                   "── лёгкие ──────────────────"),
        ("qwen3:4b",             "быстрый"),
        ("qwen2.5:3b",           "минимальный"),
        ("qwen2.5-coder:3b",     "кодинг лёгкий"),
    ],
    "vision_model": [
        ("qwen2.5vl:7b",    "лучший OCR/анализ · ~5 GB"),
        ("qwen2.5vl:32b",   "мощный · ~18 GB"),
        ("llava:13b",       "проверенный · ~8 GB"),
    ],
    # В голосовом ходе модель — не единственное узкое место (ещё STT + TTS
    # последовательно, см. agent_builder.py про eviction между переключениями),
    # так что здесь важнее скорость ответа, чем глубина рассуждения или
    # качество кода — reasoning-модели и тяжёлые coder-теги из chat_model
    # сюда специально не включены.
    "voice_chat_model": [
        (None,               "── здесь важна СКОРОСТЬ, не глубина ──"),
        ("qwen3:8b",         "быстрый, общего назначения (дефолт)"),
        ("qwen2.5:7b",       "стабильный, общего назначения"),
        ("qwen3:4b",         "самый быстрый из этого списка"),
        ("qwen2.5:3b",       "минимальный, для слабого железа"),
    ],
}

# Размер весов (ГБ) для моделей из _SUGGESTED, которые ЕЩЁ НЕ установлены —
# используется только как fallback в _size_suffix(). Для уже установленных
# моделей всегда берётся точный размер из `ollama list` (_fetch_ollama_models),
# он и так под рукой и надёжнее любого захардкоженного числа. Значения сверены
# напрямую по ollama.com/library на момент написания — если модель обновится
# в реестре, здесь могут устареть, но это лучше, чем вообще не показывать
# ничего до скачивания.
_MODEL_SIZE_GB = {
    "qwen3:14b": 9.3,
    "qwen2.5:14b": 9.0,
    "qwen2.5-coder:14b": 9.0,
    "qwq:32b": 20.0,
    "qwen3-coder:30b": 19.0,
    "gpt-oss:20b": 13.0,
    "qwen3.5:9b": 6.6,
    "qwen3:8b": 5.2,
    "qwen2.5:7b": 4.7,
    "qwen2.5-coder:7b": 4.7,
    "dolphin3:latest": 4.9,
    "qwen3:4b": 2.5,
    "qwen2.5:3b": 1.9,
    "qwen2.5-coder:3b": 1.9,
}


def _size_suffix(model_id: str, installed_sizes: dict[str, float]) -> str:
    """Если для этой модели есть живой замер _MEASURED_GPU_SHARE (см. его
    комментарий выше про num_ctx/KV-cache) — показываем ЕГО, это единственная
    цифра, которая реально отвечает на "влезает или нет". Без замера
    (модель не установлена, никто её ещё не грузил и не смотрел `ollama ps`)
    откатываемся на вес_модели/TOTAL_VRAM_GB, как раньше — это НЕ означает
    "влезает", только оценка веса относительно объёма карты, без KV-cache.
    Если GPU не определился (settings.TOTAL_VRAM_GB is None — нет карты,
    nvidia-smi недоступен и т.п.) — делить не на что, показываем просто
    размер модели с "~", а не лезем строить дробь и не падаем на None."""
    size_gb = installed_sizes.get(model_id) or _MODEL_SIZE_GB.get(model_id)
    if size_gb is None:
        return ""
    gpu_share = _MEASURED_GPU_SHARE.get(model_id)
    if gpu_share is not None:
        return f"{size_gb:.1f}GB · {gpu_share}% GPU"
    if settings.TOTAL_VRAM_GB:
        return f"{size_gb:.1f}/{settings.TOTAL_VRAM_GB:.1f}GB"
    return f"~{size_gb:.1f}GB"


_ITEMS = [
    ("модель",            "chat_model",        "ollama_model"),
    ("спрашивать разрешения", "ask_permissions", "toggle"),
    ("автопроверка ответа (ретраи)", "self_heal_enabled", "toggle"),
    ("новый пайплайн (Router→Analyzer→Planner→Coder→Verifier)", "pipeline_mode", "toggle"),
    ("оптимизированные тулы (урезанный список для простого пайплайна)", "optimized_tools", "toggle"),
    ("всегда делегировать поиск по коду сабагенту", "always_delegate_search", "toggle"),
    ("подсказка \"делегируй\" после долгой разведки", "delegate_nudge_enabled", "toggle"),
    ("экспериментальный expert-streaming backend", "expert_streaming_enabled", "toggle"),
    ("контекст чат-модели",  "num_ctx",            "preset"),
    ("размышления",       "show_thinking",    "toggle"),
    ("recap",             "recap_enabled",    "toggle"),
    ("сжатие истории тулов в ходе", "compact_history_enabled", "toggle"),
    ("vision модель",     "vision_model",     "ollama_model"),
    ("голосовой режим",   "voice_mode",       "toggle"),
    ("модель голос. режима", "voice_chat_model", "ollama_model"),
    ("распознавание речи", "stt_model",       "preset"),
    ("клонирование голоса", "tts_voice_clone_path", "voice_clone"),
    ("music устройство",  "music_gen_device", "device"),
    ("imggen модель",     "image_gen_model",  "imggen_model"),
    ("imggen устройство", "image_gen_device", "device"),
    ("imggen шаги",       "imggen_steps",     "preset"),
    ("imggen guidance",   "imggen_guidance",  "preset"),
    ("imggen сила",       "imggen_strength",  "preset"),
    ("imggen ширина",     "imggen_width",             "int"),
    ("imggen высота",     "imggen_height",            "int"),
    ("imggen фильтр",     "imggen_safety",            "toggle"),
    ("imggen enhance",    "imggen_enhance_prompt",    "toggle"),
    ("imggen prefix",     "imggen_prompt_prefix",     "str"),
    ("imggen negative",   "imggen_negative_prompt",   "str"),
    ("генеративные тулы у агента", "gen_agent_tools",  "toggle"),
    ("gen_model включён", "gen3d_enabled",             "toggle"),
    ("gen_model полигонаж", "gen3d_target_faces",     "preset"),
    ("gen_model скиннинг", "gen3d_skin_source",        "preset"),
    ("gen_model профиль",  "gen3d_hunyuan_profile",    "preset"),
    ("gen_model AI PBR (roughness/metallic)", "gen3d_pbr_ai", "toggle"),
    ("debug",             "debug",                    "toggle"),
    ("выгрузить модели",  "_unload_models",            "action"),
]

# Раньше был захардкожен на 22 — часть названий настроек (например,
# "оптимизированные тулы (урезанный список для простого пайплайна)", 63
# символа) шире этого, и колонка значения (xv в _draw ниже) рисовалась
# ПРЯМО ПОВЕРХ хвоста названия, съедая половину текста. Считаем от
# реального самого длинного названия — растёт сам по себе, когда в _ITEMS
# добавляется новая длинная строка, вместо того чтобы снова упираться в
# застывшее число.
_LABEL_COL_WIDTH = max(22, max(len(label) for label, _, _ in _ITEMS) + 2)

_TOGGLE_HINTS = {
    "ask_permissions":        "(bash/запись файлов/git — ВЫКЛ = без подтверждения, ОПАСНО)",
    "self_heal_enabled":      "(ВЫКЛ = 1 попытка без ретраев; ask_user-диалог всё равно работает; сама проверка ответа не выключается — она же фильтр качества датасета для finetune/, см. finetune/README.md)",
    "pipeline_mode":          "(ВЫКЛ = простой пайплайн — один агент с планированием, без разделения на стадии; голосовой режим всегда идёт через простой пайплайн)",
    "always_delegate_search": "(ВЫКЛ = делегирует поиск сабагенту только для больших/незнакомых деревьев; ВКЛ = делегирует ЛЮБОЙ поиск по коду, даже мелкий в этом проекте — медленнее на простых задачах)",
    "delegate_nudge_enabled": "(ВЫКЛ = не подсказывать delegate после долгой разведки — независимо от always_delegate_search выше)",
    "expert_streaming_enabled": "(экспериментальный, незамерженный форк llama.cpp с настоящим dynamic MoE expert-кэшем вместо статичного сплита Ollama — TG быстрее, PP заметно медленнее, нужен `setup.py --only expert-streaming`; действует независимо от pipeline_mode — см. expert_streaming.py)",
    "optimized_tools":        "(работает только когда 'новый пайплайн' ВЫКЛ и не в голосовом режиме — по одному тулу на смысл: bash/read/grep/glob/write/edit, без git-тулов и вариантов read/write)",
    "show_thinking": "(цепочка мыслей)",
    "recap_enabled":          "(краткая память в шапке)",
    "compact_history_enabled": "(ВЫКЛ = никогда не пересказывать историю тул-вызовов внутри хода — риск переполнить num_ctx на очень долгих ходах, зато без потери деталей)",
    "voice_mode":             "(озвучивает ответы, переключает модель на qwen3:8b)",
    "imggen_safety":          "(safety checker)",
    "imggen_enhance_prompt":  "(LLM улучшает промпт перед генерацией)",
    "debug":                  "(лог tool-коллов в episodic; DEBUG в .env приоритетнее этого тумблера)",
    "gen_agent_tools":        "(даёт агенту generate_image/music/3d_model напрямую; /gen /music /gen_model работают всегда; применяется сразу, со следующего хода)",
    "gen3d_enabled":          "(/gen_model, /anim и агентные тулы — ВЫКЛ прячет их совсем)",
    "gen3d_pbr_ai":           "(SuperMat, отдельная SD2.1-модель, +6-8 мин к /gen_model; нужен vendor/supermat)",
}


# Соответствие русских клавиш латинским (та же физическая позиция)
_RU_TO_EN: dict[str, str] = {
    'й':'q','ц':'w','у':'e','к':'r','е':'t','н':'y','г':'u','ш':'i','щ':'o','з':'p',
    'ф':'a','ы':'s','в':'d','а':'f','п':'g','р':'h','о':'j','л':'k','д':'l',
    'я':'z','ч':'x','с':'c','м':'v','и':'b','т':'n','ь':'m',
    'Й':'Q','Ц':'W','У':'E','К':'R','Е':'T','Н':'Y','Г':'U','Ш':'I','Щ':'O','З':'P',
    'Ф':'A','Ы':'S','В':'D','А':'F','П':'G','Р':'H','О':'J','Л':'K','Д':'L',
    'Я':'Z','Ч':'X','С':'C','М':'V','И':'B','Т':'N','Ь':'M',
}


def _getch(stdscr) -> int:
    """getch с поддержкой Unicode и русской раскладки."""
    try:
        ch = stdscr.get_wch()
    except Exception:
        return -1
    if isinstance(ch, str):
        ch = _RU_TO_EN.get(ch, ch)
        return ord(ch)
    return ch


def _fetch_ollama_models() -> dict[str, float]:
    """{model_name: size_gb} для локально установленных моделей — реальный
    размер из самого Ollama, не оценка."""
    try:
        import ollama as _ol
        return {m.model: m.size / (1024 ** 3) for m in _ol.list().models}
    except Exception:
        return {}


def settings_menu(print_header: Callable) -> None:

    def _run(stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)
        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN,   -1)
            curses.init_pair(2, curses.COLOR_GREEN,  -1)
            curses.init_pair(3, curses.COLOR_YELLOW, -1)
        except Exception:
            pass

        sel = 0
        status_msg = ""  # короткая обратная связь после "action"-пунктов (см. unload_models)

        # ── Отрисовка главного меню ───────────────────────────────────────────

        def _draw():
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            title = "  настройки  "
            try:
                stdscr.addstr(0, 0, "─" * (w - 1))
                stdscr.addstr(0, max(0, (w - len(title)) // 2), title,
                              curses.A_BOLD | curses.color_pair(1))
            except curses.error:
                pass

            visible = h - 4
            start = max(0, min(sel - visible // 2, max(0, len(_ITEMS) - visible)))
            end = min(len(_ITEMS), start + visible)

            for i, (label, key, kind) in enumerate(_ITEMS[start:end], start=start):
                y = 2 + (i - start)
                val = settings.get(key)
                is_sel = (i == sel)
                base = curses.color_pair(1) | curses.A_BOLD if is_sel else 0
                xv = 4 + _LABEL_COL_WIDTH

                try:
                    stdscr.addstr(y, 2, "▶ " if is_sel else "  ", base)
                    stdscr.addstr(y, 4, f"{label:<{_LABEL_COL_WIDTH}}", base)

                    if kind == "toggle":
                        if val:
                            stdscr.addstr(y, xv, "● ВКЛ  ", curses.color_pair(2) | curses.A_BOLD)
                        else:
                            stdscr.addstr(y, xv, "○ ВЫКЛ ", curses.A_DIM)
                        hint = _TOGGLE_HINTS.get(key, "")
                        if hint:
                            stdscr.addstr(y, xv + 7, hint, curses.A_DIM)
                    elif kind == "device":
                        if not settings.CUDA_AVAILABLE:
                            stdscr.addstr(y, xv, f"{val.upper()}  (GPU недоступен)", curses.A_DIM)
                        else:
                            attr = curses.color_pair(2) if val == "cuda" else curses.color_pair(3)
                            gpu = f"  ({settings.CUDA_DEVICE_NAME})" if settings.CUDA_DEVICE_NAME else ""
                            stdscr.addstr(y, xv, val.upper() + gpu, attr | curses.A_BOLD)
                    elif kind == "voice_clone":
                        if val:
                            stdscr.addstr(y, xv, f"кастомный ({val})", curses.color_pair(2))
                        else:
                            stdscr.addstr(y, xv, "стандартный", curses.A_DIM)
                    elif kind == "action":
                        stdscr.addstr(y, xv, "[Enter] выполнить", curses.color_pair(3))
                    else:
                        stdscr.addstr(y, xv, str(val), curses.color_pair(3))
                except curses.error:
                    pass

            if status_msg:
                try:
                    stdscr.addstr(h - 3, 4, status_msg, curses.color_pair(2))
                except curses.error:
                    pass

            try:
                stdscr.addstr(h - 2, 0, "─" * (w - 1))
                foot = " ↑↓  навигация    Enter / Space  изменить    Esc / q  выход "
                stdscr.addstr(h - 1, max(0, (w - len(foot)) // 2), foot, curses.A_DIM)
            except curses.error:
                pass
            stdscr.refresh()

        # ── Выбор Ollama-модели из списка ─────────────────────────────────────

        def _pick_model(skey: str) -> str | None:
            current        = settings.get(skey)
            installed_sizes = _fetch_ollama_models()
            installed      = set(installed_sizes)

            # Build unified list: suggested first (with section headers), then extras
            suggested_map = {m: d for m, d in _SUGGESTED.get(skey, []) if m is not None}
            extra = [m for m in sorted(installed) if m not in suggested_map]
            # entries: (model_id | None, description, is_installed)
            # None model_id = non-selectable section header
            entries: list[tuple] = []
            for mid, desc in _SUGGESTED.get(skey, []):
                if mid is None:
                    entries.append((None, desc, False))  # section header
                else:
                    size = _size_suffix(mid, installed_sizes)
                    if desc and size:
                        full_desc = f"{desc} · {size}"
                    else:
                        full_desc = desc or size
                    entries.append((mid, full_desc, mid in installed))
            if extra:
                entries.append((None, "── установленные ───────────────", False))
                for mid in extra:
                    size = _size_suffix(mid, installed_sizes)
                    entries.append((mid, size, True))
            # fallback: no selectable entries
            if not any(m for m, _, _ in entries):
                return _edit_str(skey)

            # Start selection on current model or first selectable entry
            def _first_selectable(start=0):
                for i in range(start, len(entries)):
                    if entries[i][0] is not None:
                        return i
                return start

            sub_sel = next(
                (i for i, (m, _, _) in enumerate(entries) if m == current),
                _first_selectable()
            )
            pull_hint = ""

            while True:
                stdscr.erase()
                h, w = stdscr.getmaxyx()
                title = "  выбор модели  "
                try:
                    stdscr.addstr(0, 0, "─" * (w - 1))
                    stdscr.addstr(0, max(0, (w - len(title)) // 2), title,
                                  curses.A_BOLD | curses.color_pair(1))
                except curses.error:
                    pass

                visible = h - 4
                start = max(0, sub_sel - visible // 2)
                end   = min(len(entries), start + visible)

                for i, (mid, desc, inst) in enumerate(entries[start:end]):
                    y = 2 + i
                    is_cur    = (i + start == sub_sel)
                    is_active = (mid == current)
                    base_attr = curses.color_pair(1) | curses.A_BOLD if is_cur else 0
                    try:
                        if mid is None:
                            # Section header — dim, non-selectable
                            stdscr.addstr(y, 2, f"  {desc}", curses.A_DIM)
                            continue
                        stdscr.addstr(y, 2, "▶ " if is_cur else "  ", base_attr)
                        if inst:
                            stdscr.addstr(y, 4, mid, base_attr)
                        else:
                            stdscr.addstr(y, 4, mid, curses.A_DIM if not is_cur else curses.color_pair(3))
                        x = 5 + len(mid)
                        if desc:
                            stdscr.addstr(y, x, f"  {desc}", curses.A_DIM)
                            x += 2 + len(desc)
                        if not inst:
                            stdscr.addstr(y, x + 1, "⬇", curses.color_pair(3))
                        elif is_active:
                            stdscr.addstr(y, x + 1, "←", curses.color_pair(2))
                    except curses.error:
                        pass

                try:
                    stdscr.addstr(h - 2, 0, "─" * (w - 1))
                    if pull_hint:
                        hint_str = pull_hint[:w - 2]
                        stdscr.addstr(h - 1, 0, " " * (w - 1))
                        stdscr.addstr(h - 1, 2, hint_str, curses.color_pair(3))
                    else:
                        foot = " ↑↓  выбор    Enter  применить    Esc  отмена "
                        stdscr.addstr(h - 1, max(0, (w - len(foot)) // 2), foot, curses.A_DIM)
                except curses.error:
                    pass
                stdscr.refresh()

                k = _getch(stdscr)
                pull_hint = ""
                if k in (curses.KEY_UP, ord('k')):
                    i = (sub_sel - 1) % len(entries)
                    while entries[i][0] is None:
                        i = (i - 1) % len(entries)
                    sub_sel = i
                elif k in (curses.KEY_DOWN, ord('j')):
                    i = (sub_sel + 1) % len(entries)
                    while entries[i][0] is None:
                        i = (i + 1) % len(entries)
                    sub_sel = i
                elif k in (curses.KEY_ENTER, ord('\n'), ord('\r')):
                    mid, _, inst = entries[sub_sel]
                    if mid is None:
                        pass
                    elif not inst:
                        pull_hint = f"не установлена — скачай: ollama pull {mid}"
                    else:
                        return mid
                elif k in (27, ord('q')):
                    return None

        # ── Выбор imggen модели из предустановленного списка ─────────────────

        def _pick_imggen_model() -> str | None:
            current = settings.get("image_gen_model")
            known = _IMGGEN_MODELS
            custom_row = ("✏  своя модель", "введи HuggingFace ID вручную", None, None)
            options = known + [custom_row]
            sub_sel = next((i for i, (m, *_) in enumerate(options) if m == current), 0)

            while True:
                stdscr.erase()
                h, w = stdscr.getmaxyx()
                title = "  imggen модель  "
                try:
                    stdscr.addstr(0, 0, "─" * (w - 1))
                    stdscr.addstr(0, max(0, (w - len(title)) // 2), title,
                                  curses.A_BOLD | curses.color_pair(1))
                except curses.error:
                    pass

                for i, (model, desc, steps, guidance) in enumerate(options):
                    y = 2 + i
                    if y >= h - 2:
                        break
                    is_cur    = (i == sub_sel)
                    is_active = (model == current)
                    attr = curses.color_pair(1) | curses.A_BOLD if is_cur else 0
                    badge = f"  {steps}шг · cfg{guidance}" if steps is not None else ""
                    try:
                        stdscr.addstr(y, 2, "▶ " if is_cur else "  ", attr)
                        stdscr.addstr(y, 4, model, attr)
                        stdscr.addstr(y, 5 + len(model), f"  {desc}", curses.A_DIM)
                        if badge:
                            stdscr.addstr(y, 5 + len(model) + 2 + len(desc), badge, curses.color_pair(3))
                        if is_active:
                            stdscr.addstr(y, 5 + len(model) + 2 + len(desc) + len(badge), " ←",
                                          curses.color_pair(2))
                    except curses.error:
                        pass

                try:
                    stdscr.addstr(h - 2, 0, "─" * (w - 1))
                    foot = " ↑↓  выбор    Enter  применить    Esc  отмена "
                    stdscr.addstr(h - 1, max(0, (w - len(foot)) // 2), foot, curses.A_DIM)
                except curses.error:
                    pass
                stdscr.refresh()

                k = _getch(stdscr)
                if k in (curses.KEY_UP, ord('k')):
                    sub_sel = (sub_sel - 1) % len(options)
                elif k in (curses.KEY_DOWN, ord('j')):
                    sub_sel = (sub_sel + 1) % len(options)
                elif k in (curses.KEY_ENTER, ord('\n'), ord('\r')):
                    model, _, steps, guidance = options[sub_sel]
                    if model.startswith("✏"):
                        return _edit_str("image_gen_model")
                    # Автоматически применяем шаги и guidance для выбранной модели
                    if steps is not None:
                        settings.set_value("imggen_steps", steps)
                    if guidance is not None:
                        settings.set_value("imggen_guidance", guidance)
                    return model
                elif k in (27, ord('q')):
                    return None

        def _flash(msg: str) -> None:
            """Короткое сообщение под меню на секунду — не отдельный
            попап/диалог, просто сразу видимая обратная связь после
            действия (запись/удаление), прежде чем экран перерисуется."""
            h, w = stdscr.getmaxyx()
            try:
                stdscr.addstr(h - 2, 0, " " * (w - 1))
                stdscr.addstr(h - 2, 2, msg, curses.color_pair(2) | curses.A_BOLD)
                stdscr.refresh()
            except curses.error:
                pass
            curses.napms(900)

        def _pick_voice_clone() -> None:
            """Клонирование голоса для TTS (Chatterbox audio_prompt_path) —
            не выбор значения из списка, а меню ДЕЙСТВИЙ: записать новый
            референс-клип прямо здесь, указать путь к готовому файлу, или
            вернуть стандартный голос. 'Удалить' показывается только когда
            кастомный голос реально задан — нечего удалять у дефолта."""
            while True:
                current = settings.get("tts_voice_clone_path")
                options = [("record", "🎤  Записать новый (15 сек)"),
                           ("file",   "📁  Указать файл вручную")]
                if current:
                    options.append(("delete", "🗑  Вернуть стандартный голос"))
                options.append(("cancel", "Esc — назад"))
                sub_sel = 0

                while True:
                    stdscr.erase()
                    h, w = stdscr.getmaxyx()
                    title = "  клонирование голоса  "
                    try:
                        stdscr.addstr(0, 0, "─" * (w - 1))
                        stdscr.addstr(0, max(0, (w - len(title)) // 2), title,
                                      curses.A_BOLD | curses.color_pair(1))
                        status = f"сейчас: {current}" if current else "сейчас: стандартный голос Chatterbox"
                        stdscr.addstr(2, 2, status, curses.A_DIM)
                    except curses.error:
                        pass

                    for i, (_, label) in enumerate(options):
                        y = 4 + i
                        if y >= h - 2:
                            break
                        is_cur = (i == sub_sel)
                        attr = curses.color_pair(1) | curses.A_BOLD if is_cur else 0
                        try:
                            stdscr.addstr(y, 2, "▶ " if is_cur else "  ", attr)
                            stdscr.addstr(y, 4, label, attr)
                        except curses.error:
                            pass

                    try:
                        stdscr.addstr(h - 2, 0, "─" * (w - 1))
                        foot = " ↑↓  выбор    Enter  выполнить    Esc  назад "
                        stdscr.addstr(h - 1, max(0, (w - len(foot)) // 2), foot, curses.A_DIM)
                    except curses.error:
                        pass
                    stdscr.refresh()

                    k = _getch(stdscr)
                    if k in (curses.KEY_UP, ord('k')):
                        sub_sel = (sub_sel - 1) % len(options)
                    elif k in (curses.KEY_DOWN, ord('j')):
                        sub_sel = (sub_sel + 1) % len(options)
                    elif k in (curses.KEY_ENTER, ord('\n'), ord('\r')):
                        action = options[sub_sel][0]
                        if action == "cancel":
                            return
                        if action == "delete":
                            settings.set_value("tts_voice_clone_path", None)
                            _flash("Возвращён стандартный голос")
                            break
                        if action == "file":
                            path = _edit_str("tts_voice_clone_path")
                            if path:
                                import os as _os
                                if _os.path.isfile(path):
                                    settings.set_value("tts_voice_clone_path", path)
                                    _flash("Файл подключён")
                                else:
                                    _flash("Файл не найден — не сохранено")
                            break
                        if action == "record":
                            try:
                                stdscr.addstr(h - 2, 0, " " * (w - 1))
                                stdscr.addstr(h - 2, 2, "🎤 Запись 15 сек — говори сейчас...",
                                              curses.color_pair(3) | curses.A_BOLD)
                                stdscr.refresh()
                            except curses.error:
                                pass
                            from ui.audio import record_voice_sample
                            new_path = record_voice_sample(15)
                            if new_path:
                                settings.set_value("tts_voice_clone_path", new_path)
                                _flash("Записано и подключено")
                            else:
                                _flash("Не удалось записать — проверь микрофон")
                            break
                    elif k in (27, ord('q')):
                        return

        def _pick_preset(skey: str) -> None:
            title_label, presets, caster = _PRESET_CONFIGS[skey]
            current = settings.get(skey)
            options = presets + [(None, "✏  своё значение")]
            sub_sel = next((i for i, (v, _) in enumerate(options) if v == current), 0)

            while True:
                stdscr.erase()
                h, w = stdscr.getmaxyx()
                title = f"  {title_label}  "
                try:
                    stdscr.addstr(0, 0, "─" * (w - 1))
                    stdscr.addstr(0, max(0, (w - len(title)) // 2), title,
                                  curses.A_BOLD | curses.color_pair(1))
                except curses.error:
                    pass

                visible = h - 4
                start = max(0, min(sub_sel - visible // 2, max(0, len(options) - visible)))
                end = min(len(options), start + visible)

                for i, (val, desc) in enumerate(options[start:end], start=start):
                    y = 2 + (i - start)
                    is_cur    = (i == sub_sel)
                    is_active = (val == current)
                    attr = curses.color_pair(1) | curses.A_BOLD if is_cur else 0
                    label = str(val) if val is not None else desc
                    extra = f"  {desc}" if val is not None else ""
                    try:
                        stdscr.addstr(y, 2, "▶ " if is_cur else "  ", attr)
                        stdscr.addstr(y, 4, label, attr)
                        stdscr.addstr(y, 5 + len(label), extra, curses.A_DIM)
                        if is_active:
                            stdscr.addstr(y, 5 + len(label) + len(extra), " ←", curses.color_pair(2))
                    except curses.error:
                        pass

                try:
                    stdscr.addstr(h - 2, 0, "─" * (w - 1))
                    foot = " ↑↓  выбор    Enter  применить    Esc  отмена "
                    stdscr.addstr(h - 1, max(0, (w - len(foot)) // 2), foot, curses.A_DIM)
                except curses.error:
                    pass
                stdscr.refresh()

                k = _getch(stdscr)
                if k in (curses.KEY_UP, ord('k')):
                    sub_sel = (sub_sel - 1) % len(options)
                elif k in (curses.KEY_DOWN, ord('j')):
                    sub_sel = (sub_sel + 1) % len(options)
                elif k in (curses.KEY_ENTER, ord('\n'), ord('\r')):
                    val, _ = options[sub_sel]
                    if val is None:
                        raw = _edit_str(skey)
                        if raw:
                            try:
                                settings.set_value(skey, caster(raw))
                            except ValueError:
                                pass
                    else:
                        settings.set_value(skey, val)
                    return
                elif k in (27, ord('q')):
                    return

        # ── Ввод произвольной строки ──────────────────────────────────────────

        def _edit_str(skey: str) -> str | None:
            h, w = stdscr.getmaxyx()
            current = settings.get(skey)
            hint   = f"  (текущее: {current})  "
            prompt = " › "
            try:
                stdscr.addstr(h - 2, 0, " " * (w - 1))
                stdscr.addstr(h - 2, 2, hint, curses.A_DIM)
                stdscr.addstr(h - 1, 0, " " * (w - 1))
                stdscr.addstr(h - 1, 0, prompt, curses.color_pair(1) | curses.A_BOLD)
            except curses.error:
                pass
            curses.curs_set(1)
            curses.echo()
            stdscr.refresh()
            try:
                raw = stdscr.getstr(h - 1, len(prompt), w - len(prompt) - 2)
                val = raw.decode("utf-8", errors="replace").strip()
            except Exception:
                val = ""
            curses.noecho()
            curses.curs_set(0)
            return val or None

        # ── Основной цикл ─────────────────────────────────────────────────────

        while True:
            _draw()
            key = _getch(stdscr)

            if key in (curses.KEY_UP, ord('k')):
                sel = (sel - 1) % len(_ITEMS)
            elif key in (curses.KEY_DOWN, ord('j')):
                sel = (sel + 1) % len(_ITEMS)
            elif key in (curses.KEY_ENTER, ord('\n'), ord('\r'), ord(' ')):
                _, skey, kind = _ITEMS[sel]
                status_msg = ""
                if kind == "toggle":
                    settings.set_value(skey, not settings.get(skey))
                elif kind == "device":
                    if settings.CUDA_AVAILABLE:
                        cur = settings.get(skey)
                        settings.set_value(skey, "cpu" if cur == "cuda" else "cuda")
                elif kind == "ollama_model":
                    new_val = _pick_model(skey)
                    if new_val:
                        settings.set_value(skey, new_val)
                elif kind == "imggen_model":
                    new_val = _pick_imggen_model()
                    if new_val:
                        settings.set_value(skey, new_val)
                elif kind == "preset":
                    _pick_preset(skey)
                elif kind == "voice_clone":
                    _pick_voice_clone()
                elif kind == "str":
                    new_val = _edit_str(skey)
                    if new_val:
                        settings.set_value(skey, new_val)
                elif kind == "int":
                    new_val = _edit_str(skey)
                    if new_val:
                        try:
                            settings.set_value(skey, int(new_val))
                        except ValueError:
                            pass
                elif kind == "action" and skey == "_unload_models":
                    from model_lifecycle import unload_idle_models
                    freed = unload_idle_models()
                    status_msg = ("выгружено: " + ", ".join(freed)) if freed else "ничего лишнего не висело в памяти"
            elif key in (27, ord('q')):
                break

    curses.wrapper(_run)
    flush_pending_input()
    print_header()
