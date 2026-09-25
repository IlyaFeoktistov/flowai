import curses
from typing import Callable
import model_params
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

# Слоты llama-server на ОДНОЙ загруженной модели (settings.parallel_slots) —
# каждый слот сверх первого = ещё один KV-cache окна num_ctx в памяти, а
# скорость генерации делится между одновременными запросами.
_PARALLEL_PRESETS = [
    (1, "дефолт — агенты по очереди, память как без них"),
    (2, "два запроса одновременно — +1 KV-cache окна num_ctx, скорость делится"),
    (3, "+2 KV-cache окна num_ctx — только при запасе RAM/VRAM"),
    (4, "+3 KV-cache окна num_ctx — риск нехватки памяти"),
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
    "parallel_slots":  ("параллельные потоки (одна модель, N слотов)", _PARALLEL_PRESETS, int),
}


_ITEMS = [
    ("модель",            "chat_model",        "ollama_model"),
    ("спрашивать разрешения", "ask_permissions", "toggle"),
    ("автопроверка ответа", "self_heal_enabled", "toggle"),
    ("агентный режим", "pipeline_mode", "toggle"),
    ("простые ответы без тулов", "casual_answers_enabled", "toggle"),
    ("оптимизированные тулы", "optimized_tools", "toggle"),
    ("поиск кода через агента", "always_delegate_search", "toggle"),
    ("параллельные потоки",  "parallel_slots",   "preset"),
    ("подсказка про агентов", "delegate_nudge_enabled", "toggle"),
    ("expert-streaming backend", "expert_streaming_enabled", "toggle"),
    ("параметры модели",  "_model_params",    "model_params"),
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
    ("gen_model AI PBR", "gen3d_pbr_ai", "toggle"),
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
    "ask_permissions":        "подтверждение перед bash/файлами/git — ВЫКЛ опасно",
    "self_heal_enabled":      "автоматический повтор при неудачном ответе",
    "pipeline_mode":          "Router→Analyzer→Planner→Coder→Verifier",
    "casual_answers_enabled": "прямой ответ без верификации (Требуется pipeline_mode)",
    "always_delegate_search": "всегда, даже мелкий — не только большие деревья",
    "delegate_nudge_enabled": "предложить explore-агента после долгой разведки",
    "expert_streaming_enabled": "MoE expert-кэш (Требуется собранный expert-streaming)",
    "optimized_tools":        "по одному тулу на смысл, для всех агентов",
    "show_thinking":          "цепочка мыслей модели",
    "recap_enabled":          "краткая память в шапке",
    "compact_history_enabled": "сжимать историю тул-вызовов внутри хода",
    "voice_mode":             "озвучивает ответы, переключает модель",
    "imggen_safety":          "safety checker",
    "imggen_enhance_prompt":  "LLM улучшает промпт перед генерацией",
    "debug":                  "лог тул-коллов в episodic",
    "gen_agent_tools":        "агент сам вызывает генерацию картинок/музыки/3D",
    "gen3d_enabled":          "включает /gen_model, /anim и агентные тулы",
    "gen3d_pbr_ai":           "roughness/metallic, +6-8 мин (Требуется vendor/supermat)",
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


def _fit(text: str, max_len: int) -> str:
    """Truncates with a trailing ellipsis if text would overflow max_len —
    curses.addstr on a string that runs past the window's right edge
    either wraps onto the NEXT row (visually merging two settings' text
    together — the exact garbled overlap a too-long _TOGGLE_HINTS entry
    used to produce on anything narrower than a very wide terminal) or
    raises curses.error partway through the write (silently caught
    elsewhere in this module, leaving an arbitrary mid-word cutoff).
    Drawing an already-bounded string avoids both failure modes outright,
    regardless of how long any given label/hint/value happens to be."""
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len == 1:
        return "…"
    return text[:max_len - 1] + "…"


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

                # w - xv (1 col margin) — сколько ЕЩЁ можно вообще нарисовать в
                # этой строке правее значения (общий бюджет под "○ ВЫКЛ " +
                # подсказку/значение); room_after_toggle — то, что остаётся
                # под саму подсказку уже после 7-символьного индикатора
                # ВКЛ/ВЫКЛ. На отрицательный бюджет (совсем узкий терминал)
                # _fit сама отдаёт "" вместо попытки нарисовать что-то с
                # отрицательной длиной.
                room = max(0, w - xv - 1)
                room_after_toggle = max(0, room - 7)

                try:
                    stdscr.addstr(y, 2, "▶ " if is_sel else "  ", base)
                    stdscr.addstr(y, 4, _fit(label, _LABEL_COL_WIDTH), base)

                    if kind == "toggle":
                        if val:
                            stdscr.addstr(y, xv, "● ВКЛ  ", curses.color_pair(2) | curses.A_BOLD)
                        else:
                            stdscr.addstr(y, xv, "○ ВЫКЛ ", curses.A_DIM)
                        hint = _TOGGLE_HINTS.get(key, "")
                        if hint:
                            stdscr.addstr(y, xv + 7, _fit(hint, room_after_toggle), curses.A_DIM)
                    elif kind == "device":
                        if not settings.CUDA_AVAILABLE:
                            stdscr.addstr(y, xv, _fit(f"{val.upper()}  (GPU недоступен)", room), curses.A_DIM)
                        else:
                            attr = curses.color_pair(2) if val == "cuda" else curses.color_pair(3)
                            gpu = f"  ({settings.CUDA_DEVICE_NAME})" if settings.CUDA_DEVICE_NAME else ""
                            stdscr.addstr(y, xv, _fit(val.upper() + gpu, room), attr | curses.A_BOLD)
                    elif kind == "voice_clone":
                        if val:
                            stdscr.addstr(y, xv, _fit(f"кастомный ({val})", room), curses.color_pair(2))
                        else:
                            stdscr.addstr(y, xv, "стандартный", curses.A_DIM)
                    elif kind == "action":
                        stdscr.addstr(y, xv, "[Enter] выполнить", curses.color_pair(3))
                    elif kind == "model_params":
                        n = sum(1 for p in model_params.describe(settings.get("chat_model")) if p["overridden"])
                        summary = f"{settings.get('chat_model')} · " + (f"изменено: {n}" if n else "по умолчанию")
                        stdscr.addstr(y, xv, _fit(summary, room), curses.color_pair(3))
                    else:
                        stdscr.addstr(y, xv, _fit(str(val), room), curses.color_pair(3))
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

        def _pick_model(skey: str | None, title_text: str = "выбор модели", current: str | None = None) -> str | None:
            """Только установленные модели (`ollama list`) с размером весов —
            что реально загружено и сколько занимает, показывает /instances."""
            current = settings.get(skey) if skey else current
            installed = _fetch_ollama_models()
            entries = sorted(installed.items())
            if current and current not in installed:
                entries.insert(0, (current, None))
            if not entries:
                return _edit_str(skey) if skey else None
            sub_sel = next((i for i, (m, _) in enumerate(entries) if m == current), 0)

            while True:
                stdscr.erase()
                h, w = stdscr.getmaxyx()
                title = f"  {title_text}  "
                try:
                    stdscr.addstr(0, 0, "─" * (w - 1))
                    stdscr.addstr(0, max(0, (w - len(title)) // 2), title,
                                  curses.A_BOLD | curses.color_pair(1))
                except curses.error:
                    pass

                visible = h - 4
                start = max(0, min(sub_sel - visible // 2, max(0, len(entries) - visible)))
                end = min(len(entries), start + visible)
                name_w = max(len(m) for m, _ in entries) + 2
                for i, (mid, size) in enumerate(entries[start:end], start=start):
                    y = 2 + (i - start)
                    is_cur = i == sub_sel
                    attr = curses.color_pair(1) | curses.A_BOLD if is_cur else 0
                    size_txt = f"{size:.1f} GB" if size is not None else "не установлена"
                    try:
                        stdscr.addstr(y, 2, "▶ " if is_cur else "  ", attr)
                        stdscr.addstr(y, 4, _fit(mid, w - 6), attr)
                        stdscr.addstr(y, 4 + name_w, _fit(size_txt, max(0, w - name_w - 8)), curses.A_DIM)
                        if mid == current:
                            stdscr.addstr(y, 4 + name_w + len(size_txt) + 1, "←", curses.color_pair(2))
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
                    sub_sel = (sub_sel - 1) % len(entries)
                elif k in (curses.KEY_DOWN, ord('j')):
                    sub_sel = (sub_sel + 1) % len(entries)
                elif k in (curses.KEY_ENTER, ord('\n'), ord('\r')):
                    return entries[sub_sel][0]
                elif k in (27, ord('q')):
                    return None

        # ── Параметры генерации выбранной модели (model_params.py) ───────────

        def _edit_model_params() -> None:
            model = settings.get("chat_model")
            sub_sel = 0
            note = ""
            while True:
                rows = model_params.describe(model)
                stdscr.erase()
                h, w = stdscr.getmaxyx()
                title = "  параметры модели  "
                try:
                    stdscr.addstr(0, 0, "─" * (w - 1))
                    stdscr.addstr(0, max(0, (w - len(title)) // 2), title,
                                  curses.A_BOLD | curses.color_pair(1))
                    stdscr.addstr(2, 4, "модель: ", curses.A_DIM)
                    stdscr.addstr(2, 12, _fit(model, w - 40), curses.color_pair(3) | curses.A_BOLD)
                    stdscr.addstr(2, 13 + min(len(model), w - 40), "  [m] другая модель", curses.A_DIM)
                except curses.error:
                    pass

                for i, row in enumerate(rows):
                    y = 4 + i
                    if y >= h - 4:
                        break
                    is_cur = i == sub_sel
                    attr = curses.color_pair(1) | curses.A_BOLD if is_cur else 0
                    value = "по умолч. бэкенда" if row["value"] is None else str(row["value"])
                    mark = "своё" if row["overridden"] else "по умолч."
                    try:
                        stdscr.addstr(y, 2, "▶ " if is_cur else "  ", attr)
                        stdscr.addstr(y, 4, f"{row['label']:<16}", attr)
                        stdscr.addstr(y, 21, f"{value:<18}",
                                      (curses.color_pair(2) | curses.A_BOLD) if row["overridden"] else curses.color_pair(3))
                        stdscr.addstr(y, 40, f"{mark:<10}", curses.A_DIM)
                        stdscr.addstr(y, 51, _fit(row["hint"], max(0, w - 53)), curses.A_DIM)
                    except curses.error:
                        pass

                if note:
                    try:
                        stdscr.addstr(h - 3, 4, _fit(note, w - 6), curses.color_pair(2))
                    except curses.error:
                        pass
                try:
                    stdscr.addstr(h - 2, 0, "─" * (w - 1))
                    foot = " ↑↓  выбор    Enter  изменить    r  сбросить    m  модель    Esc  назад "
                    stdscr.addstr(h - 1, max(0, (w - len(foot)) // 2), _fit(foot, w - 1), curses.A_DIM)
                except curses.error:
                    pass
                stdscr.refresh()

                k = _getch(stdscr)
                note = ""
                if k in (curses.KEY_UP, ord('k')):
                    sub_sel = (sub_sel - 1) % len(rows)
                elif k in (curses.KEY_DOWN, ord('j')):
                    sub_sel = (sub_sel + 1) % len(rows)
                elif k in (ord('m'), ord('M')):
                    picked = _pick_model(None, "параметры какой модели", current=model)
                    if picked:
                        model = picked
                        sub_sel = 0
                elif k in (ord('r'), ord('R')):
                    row = rows[sub_sel]
                    model_params.set_param(model, row["key"], None)
                    note = f"{row['label']} сброшен к {row['default']}"
                elif k in (curses.KEY_ENTER, ord('\n'), ord('\r'), ord(' ')):
                    row = rows[sub_sel]
                    raw = _edit_str(None, f"{row['label']} (текущее: {row['value']}, пусто — по умолчанию)")
                    if raw is None:
                        model_params.set_param(model, row["key"], None)
                        note = f"{row['label']}: по умолчанию ({row['default']})"
                        continue
                    try:
                        model_params.set_param(model, row["key"], model_params.parse_value(row["key"], raw))
                        note = f"{row['label']} = {raw}"
                    except ValueError as e:
                        note = f"не сохранено: {e}"
                elif k in (27, ord('q')):
                    return

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

        def _edit_str(skey: str | None, hint_text: str | None = None) -> str | None:
            h, w = stdscr.getmaxyx()
            hint = f"  ({hint_text or f'текущее: {settings.get(skey)}'})  "
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
                elif kind == "model_params":
                    _edit_model_params()
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
