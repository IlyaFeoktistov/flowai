import json
import os
import subprocess

from storage import connect


def _detect_gpu() -> tuple[bool, str | None, float | None]:
    # Вариант 1: torch (нужен для diffusers)
    try:
        import torch as _torch
        if _torch.cuda.is_available() and _torch.cuda.device_count() > 0:
            name = _torch.cuda.get_device_name(0)
            total_gb = _torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return True, name, total_gb
    except Exception:
        pass

    # Вариант 2: nvidia-smi (GPU есть, но torch-cuda не установлен)
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            name, mem_mib = r.stdout.strip().splitlines()[0].split(",")
            return True, name.strip(), int(mem_mib.strip()) / 1024
    except Exception:
        pass

    return False, None, None


# TOTAL_VRAM_GB — реальный объём VRAM ЭТОЙ машины (не догадка/константа из
# CLAUDE.md, которая может устареть на другом железе) — нужен, чтобы в
# выборе модели (ui/tui/settings.py) показывать не абстрактное "~9 GB", а
# факт "9.3/6.0GB" — сколько модель займёт из того, что реально доступно.
CUDA_AVAILABLE, CUDA_DEVICE_NAME, TOTAL_VRAM_GB = _detect_gpu()

_conn = connect()
_conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
_conn.commit()

_state: dict = {
    # glm-4.7-flash:q4_K_M (не qwen3-coder:30b) — признанный дефолт с
    # 2026-08-14: живыми прогонами (см. expert_streaming.py, раздел
    # «GLM-4.7-Flash») подтверждены и загрузка, и корректная остановка
    # генерации на expert-streaming backend'е, при близком к qwen3-coder:30b
    # весе (17.7 GB против 18 GB) и заметно лучшей скорости на этом железе.
    # Требует expert_streaming_enabled=ВКЛ (см. ниже) — обычный Ollama-путь
    # не поддерживает архитектуру glm4moelite без патча этого форка.
    "chat_model":        os.getenv("OLLAMA_MODEL",        "glm-4.7-flash:q4_K_M"),
    "show_thinking":    os.getenv("SHOW_THINKING",    "0") == "1",
    # Глобальный выключатель permission-диалогов (bash, write_file,
    # git-мутации, ...) — см. tools/confirm.py:ask_permission, единая точка
    # входа для обоих путей подтверждения. Дефолт ВКЛ (спрашивать) — тот же
    # безопасный дефолт, что был всегда; выключение — осознанный шаг
    # пользователя, не тихая смена поведения при обновлении.
    "ask_permissions":  os.getenv("ASK_PERMISSIONS",  "1") == "1",
    # Выключает автоматический self-heal retry-цикл (mcp_agent/agent.py:
    # stream_chat, MAX_ATTEMPTS=3 попытки, судит _semantic_check) — модель
    # получает ровно ОДНУ попытку на ход и её первый ответ становится
    # финальным, даже если внутренний судья считает его "не relevant".
    # НЕ отключает ask_user-спасение (когда модель САМА оставила вопрос
    # текстом вместо настоящего тула) — это не слепой ретрай, а реальный
    # диалог с пользователем, и он остаётся полезен независимо от этой
    # настройки. Дефолт ВКЛ (ретраи разрешены) — прежнее поведение.
    "self_heal_enabled": os.getenv("SELF_HEAL_ENABLED", "1") == "1",
    # Новый пайплайн Router->Analyzer->Planner->Coder->Verifier
    # (mcp_agent/pipeline.py) вместо простого пайплайна — единого агента с
    # планированием (mcp_agent/agent.py) — см. cli.py, где по этому флагу
    # выбирается, чей stream_chat вызывать.
    # Не участвует в voice_mode (у пайплайна нет голосовой ветки) — cli.py
    # проверяет оба флага на каждый ход, не только при старте. Дефолт ВКЛ:
    # легаси-путь на реальных задачах уже наглядно ломался (план текстом
    # вместо ask_user, self-heal ловит, ретрай уходит в повторную разведку
    # вместо того чтобы взять план и делать) — именно это новый пайплайн
    # решает архитектурно. Эскейп-люк на случай регрессии — /settings.
    "pipeline_mode":    os.getenv("PIPELINE_MODE",    "1") == "1",
    # Гейтит ветку router.py:answer_casual в НОВОМ пайплайне (mcp_agent/
    # pipeline.py:stream_chat) — прямой ответ без единого тула, без
    # stage_runner'а вокруг (значит без verdict/guidance ретраев и без
    # compaction истории, см. router.py:_CASUAL_HISTORY_WINDOW). Эскейп-люк
    # на случай регрессии: если модель на этой ветке ведёт себя странно
    # (уходит в бесконечный повтор/слоп без верификации ответа — живой
    # инцидент, см. коммит "Fix casual-chat coherence collapse"), выключить
    # тут — тогда те же сообщения (needs_change=false) идут в обычную
    # analyzer-ветку, у которой есть verdict/guidance/recursion-machinery
    # stage_runner'а, просто без Planner/Coder/Verifier после неё. Дефолт
    # ВЫКЛ — analyzer у пользователя стабильно ведёт себя лучше на простых
    # ответах, чем голый answer_casual. Не действует на легаси-агент (у
    # того свой, отдельный casual-путь, не связанный с этим тумблером) и не
    # действует, если pipeline_mode выключен.
    "casual_answers_enabled": os.getenv("CASUAL_ANSWERS_ENABLED", "0") == "1",
    # Урезанный (без переименования) набор тулов для ПРОСТОГО пайплайна
    # (единый агент с планированием, без разделения на стадии) —
    # mcp_agent/optimized_tools.py: один тул на смысл (bash/read/
    # grep/glob/write/edit), без генеративных/git-тулов, если они выключены
    # отдельными тумблерами. Работает ТОЛЬКО когда pipeline_mode выключен (новый
    # пайплайн уже решает эту же задачу иначе, per-request композицией —
    # см. router.py/roles.py) и не участвует в voice_mode (там свой пустой
    # tools=[] путь). Дефолт ВЫКЛ — не меняет поведение легаси-агента, пока
    # пользователь не включит явно.
    "optimized_tools":  os.getenv("OPTIMIZED_TOOLS",  "0") == "1",
    # ВКЛ заставляет простой пайплайн ВСЕГДА делегировать любой поиск по коду
    # (grep_search/glob_search) сабагенту через delegate, а не только для
    # больших/незнакомых деревьев — см. prompts.py:_SYSTEM_PROMPT_TEMPLATE,
    # ветка про delegate. Дефолт ВКЛ по прямому запросу пользователя
    # (2026-08-11, после живого таймаута на большом внешнем репозитории) —
    # эскейп-люк на случай
    # регрессии (лишний полный раунд сабагента там, где хватило бы 1 прямого
    # вызова) — тумблер в /settings.
    "always_delegate_search": os.getenv("ALWAYS_DELEGATE_SEARCH", "1") == "1",
    # _DelegateNudgeMiddleware (mcp_agent/delegate_tool.py) — независимый от
    # always_delegate_search механизм: он не запрещает прямые search-вызовы,
    # а вставляет в историю подсказку "хватит копать самому, вызови delegate"
    # после N read/search-вызовов подряд без delegate (см. её собственный
    # докстринг). Дефолт ВКЛ (текущее поведение) — тумблер только чтобы
    # пользователь мог выключить, если подсказка мешает на конкретной задаче.
    "delegate_nudge_enabled": os.getenv("DELEGATE_NUDGE_ENABLED", "1") == "1",
    # Отдельный, собранный из vendor/llama-expert-streaming бинарник (см.
    # expert_streaming.py's docstring — какой именно незамерженный PR, зачем
    # он тут, и trade-off, который он приносит) вместо Ollama для основной
    # кодовой модели — настоящее dynamic per-token expert offloading (-ehs)
    # для MoE-моделей, а не статичный CPU/GPU сплит, который использует
    # Ollama всегда. Дефолт ВКЛ с 2026-08-14 (был ВЫКЛ) — признанный дефолт
    # chat_model выше, glm-4.7-flash:q4_K_M, физически НЕ работает на обычном
    # Ollama-пути (архитектура glm4moelite не поддерживается апстримом без
    # патча этого форка, см. expert_streaming.py), так что для дефолтной
    # модели этот тумблер обязан быть включён, иначе чат из коробки не
    # заработает. Остаётся безопасным дефолтом даже без сборки: `python3
    # setup.py` (без --only) собирает expert-streaming автоматически (см.
    # setup.py:SETUP_FUNCS), а если бинарник всё же не собран —
    # agent_builder тихо откатывается на обычный Ollama-путь (см.
    # expert_streaming.ensure_running, которая возвращает (False, reason)
    # вместо исключения) — просто с другой моделью, а не падением.
    # Остальные аргументы за прежний дефолт ВЫКЛ остаются в силе для ЛЮБОЙ
    # другой модели (не glm-4.7-flash): это чужой недособранный community-PR,
    # без гарантий совместимости с будущими версиями llama.cpp, и живой
    # отзыв автора PR (см. docstring expert_streaming.py) — prompt-processing
    # падает в разы, генерация ускоряется в среднем на треть, то есть
    # компромисс, не безусловное ускорение. Действует независимо от
    # pipeline_mode — легаси-агент и роли пайплайна строят модель через один
    # и тот же agent_builder._build_chat_model.
    "expert_streaming_enabled": os.getenv("EXPERT_STREAMING_ENABLED", "1") == "1",
    # Живой прогон живьём показал, зачем это нужно как отдельный, редактируемый
    # тумблер, а не только mcp_agent/model_config.py:OLLAMA_NUM_CTX (тот
    # читается ОДИН раз при импорте и раздаётся через `from ... import
    # OLLAMA_NUM_CTX` в добрый десяток модулей — поменять его runtime'ом
    # означало бы менять сам механизм чтения константы во всех них, см. её
    # докстринг про то, что она НИГДЕ не переприсваивается через global):
    # тестируя expert_streaming_enabled на 6 GB VRAM, меньший num_ctx
    # оставляет больше свободной памяти под hot-store экспертов (KV-cache и
    # store делят один и тот же бюджет) — 8192 дал +105% к генерации, а
    # проектный дефолт 65536 в этом же тесте — всего +0% (в моменте даже чуть
    # хуже голого Ollama). ВАЖНО: применяется ТОЛЬКО к основной чат/judge-
    # модели, собираемой в agent_builder.py (_build_chat_model,
    # _build_agent, _build_role_agent) — router.py/dnd_agent.py/self_heal.py
    # по-прежнему используют неизменный model_config.OLLAMA_NUM_CTX (см. их
    # собственные импорты) — намеренное, узкое разделение, а не забытое
    # место: те не имеют своей копии этого тумблера и не участвуют в том,
    # что тут тестируется. mcp_agent/compaction.py раньше тоже был в этом
    # списке — но её единственная задача — не дать переполниться контексту
    # ИМЕННО этой чат/judge-модели, так что использование там
    # OLLAMA_NUM_CTX вместо settings.get("num_ctx") было не намеренным
    # разделением, а багом: живой прогон (20260812, XOR-в-Go задача) —
    # num_ctx был занижен до 16384 под этот же тест, а порог компакта
    # по-прежнему считался от OLLAMA_NUM_CTX=65536, так что компакт ждал
    # вдвое больше токенов, чем реально помещалось в модель, и ни разу не
    # сработал за весь ход. compaction.py теперь тоже читает
    # settings.get("num_ctx") — см. её собственный докстринг у
    # _needs_compaction. Понижение сильно ниже дефолта — это ОСОЗНАННЫЙ
    # риск возврата бага "n_keep=4, промпт обрезается, модель забывает
    # задачу" (см. подробный разбор в model_config.py прямо над
    # OLLAMA_NUM_CTX) — 65536 там был не произвольным числом, а посчитанным
    # потолком под 24 GB RAM этой машины, так что при использовании (не
    # только тестировании!) меньшего значения эффективный
    # TOOL_OUTPUT_CHAR_CAP/длина истории должны соответствовать новому
    # потолку, тумблер сам этого не гарантирует.
    #
    # Дефолт понижен до 30000 (2026-08-13, эта же машина, RTX 4050 6 GB) —
    # см. expert_streaming.py про живой замер: на 65536 autofit не находил
    # ни одного hot-slot экспертов вообще ни при f16, ни при q8_0 KV-cache,
    # а на 30000 — 10 слотов (хотя дефолтный бэкенд в итоге держит их
    # выключенными, -ehs 0, сам кэш экспертов там оказался медленнее, чем
    # без него — см. тот же докстринг). Число всё равно важно как нижняя
    # граница реального запаса VRAM под compute-буферы на этой карте.
    "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "30000")),
    "vision_model":     os.getenv("VISION_MODEL",      "llava:13b"),
    "stt_model":        os.getenv("STT_MODEL",         "medium"),
    "voice_mode":       os.getenv("VOICE_MODE",        "0") == "1",
    "voice_chat_model": os.getenv("VOICE_CHAT_MODEL",  "qwen3:8b"),
    # Путь к референс-клипу для voice cloning (Chatterbox audio_prompt_path).
    # None = стандартный голос. Файл лежит в storage.data_dir() (общее место,
    # ~/.local/share/flowai/), НЕ внутри проекта — тот же принцип, что уже
    # применён к settings/memory/usage (см. storage.py).
    "tts_voice_clone_path": None,
    "image_gen_model":  os.getenv("IMAGE_GEN_MODEL",   "stabilityai/sdxl-turbo"),
    "image_gen_device": "cuda" if CUDA_AVAILABLE else "cpu",
    # Дефолт CPU (а не auto-cuda, как у image_gen_device) — сознательно:
    # /music может крутиться долго (потоковая генерация), а SDXL/Ollama и так
    # соревнуются за те же 5.9 GB VRAM, см. music_server.py. Опция GPU — по
    # запросу пользователя, не тихая смена дефолтного поведения.
    "music_gen_device": "cpu",
    "imggen_safety":    os.getenv("IMGGEN_SAFETY",    "0") == "1",
    "imggen_steps":     int(os.getenv("IMGGEN_STEPS", "4")),
    "imggen_guidance":  float(os.getenv("IMGGEN_GUIDANCE", "0.0")),
    "imggen_strength":  float(os.getenv("IMGGEN_STRENGTH", "0.6")),
    "imggen_width":     int(os.getenv("IMGGEN_WIDTH", "1024")),
    "imggen_height":    int(os.getenv("IMGGEN_HEIGHT", "1024")),
    "imggen_prompt_prefix":   os.getenv("IMGGEN_PROMPT_PREFIX",   "masterpiece, best quality, high resolution, detailed"),
    "imggen_negative_prompt": os.getenv("IMGGEN_NEGATIVE_PROMPT", "bad anatomy, deformed, distorted, disfigured, mutated, extra fingers, missing fingers, fused fingers, bad hands, poorly drawn face, asymmetrical face, unrealistic eyes, blurry, low resolution, pixelated, jpeg artifacts, noise, overexposed, underexposed, bad lighting, plastic skin, waxy skin, doll-like, text, watermark, logo, signature, duplicate body, cloned face, extra limbs, cropped, out of frame, cut off, worst quality, low quality, normal quality"),
    "imggen_enhance_prompt":  os.getenv("IMGGEN_ENHANCE_PROMPT",  "0") == "1",
    # Гейтит МОДЕЛЬНЫЙ (агентный tool-calling) доступ к generate_image/
    # edit_image/generate_music/generate_3d_model/animate_3d_model/
    # generate_texture_for_model — НЕ слэш-команды: /gen, /music,
    # /gen_model, /anim, /gen_texture идут по отдельному CLI-прямому пути
    # (tools/image_gen.py, tools/gen_model.py, mcp_agent/servers/
    # music_server.py импортируется напрямую в cli.py) и работают
    # ВСЕГДА, независимо от этого тумблера — см. их докстринги про "outside
    # the agent tool-calling pipeline". Дефолт ВЫКЛ — по прямому запросу
    # пользователя: во время обычного кодинга модель не должна ни видеть
    # эти тулы в схеме, ни путать их с проектными read/write; ВЫКЛ также
    # останавливает подъём самих MCP-подпроцессов image_gen/music/gen_model
    # (см. mcp_agent/config.py:build_mcp_connections) — реальная экономия
    # времени старта и VRAM/RAM на процесс, не только косметика списка
    # тулов. web_search/fetch/analyze_image НЕ гейтятся этим тумблером —
    # они читающие и полезны независимо от того, кодим мы или генерируем
    # медиа (см. roles.py). set_value() ниже сбрасывает _tools_cache/
    # _agent_cache/_role_agent_cache (agent_builder.py:invalidate_tool_caches)
    # на КАЖДОЕ реальное изменение этого флага — без этого смена тумблера
    # молча ждала бы следующего запуска flowai, хотя build_mcp_connections
    # сам по себе уже реагирует немедленно.
    "gen_agent_tools":  os.getenv("GEN_AGENT_TOOLS", "0") == "1",
    # Общий выключатель /gen_model, /anim, /gen_texture и агентных
    # generate_3d_model/animate_3d_model/generate_texture_for_model — не про
    # качество, а про "весь пайплайн не установлен
    # / не нужен на этой машине" (setup.py). Дефолт ВКЛ — если vendor/
    # не настроен, gen3d/pipeline.py всё равно откажет с понятной ошибкой
    # ("Run: python3 setup.py"), этот тумблер просто позволяет спрятать
    # команды/тулы совсем, а не полагаться каждый раз на такую ошибку.
    "gen3d_enabled":        os.getenv("GEN3D_ENABLED", "1") == "1",
    # /gen_model (gen3d/pipeline.py) — целевой полигонаж после ретопологии.
    # 15000 — компромисс из 3dtodo.md: достаточно детально для большинства
    # ассетов, но проверяли и 1000 (низкополигональный стиль) без потери
    # силуэта — можно смело уменьшать через /settings под конкретную нужду.
    "gen3d_target_faces":   int(os.getenv("GEN3D_TARGET_FACES", "15000")),
    # Профиль оффлоада mmgp у Hunyuan3D-2GP. Замеряли 3 и 4 на 6 ГБ VRAM
    # (3dtodo.md): профиль 3 даёт ~3% скорости за счёт пика VRAM почти вдвое
    # (2.2→4.34 ГБ) — профиль 4 остаётся дефолтом, запас важнее.
    "gen3d_hunyuan_profile": int(os.getenv("GEN3D_HUNYUAN_PROFILE", "4")),
    # Источник скиннинга для --rig: "auto_weights" (встроенный в Blender
    # Automatic Weights, без доп. VRAM/чекпоинта) или "unirig" (собственная
    # skin-модель UniRig, точнее на сложной топологии, но пик VRAM ~5.87 ГБ
    # из 6 — запас всего ~274 МБ, см. 3dtodo.md). Дефолт auto_weights —
    # укладывается с большим запасом, unirig — осознанный выбор пользователя.
    "gen3d_skin_source":    os.getenv("GEN3D_SKIN_SOURCE", "auto_weights"),
    # AI-оценка roughness/metallic (SuperMat, vendor/supermat) поверх
    # gen3d_target_faces-меша с уже запечённым albedo -- добавляет реальный
    # metallicRoughnessTexture вместо плоских rebake_texture.py-констант
    # (ROUGHNESS_DEFAULT/METALLIC_DEFAULT). Дефолт ВКЛ (по явному запросу) --
    # добавляет ~6-8 минут к /gen_model (отдельный подпроцесс со своей SD2.1-
    # моделью: рендер 6 ракурсов + инференс + обратная проекция на UV).
    # ВАЖНО: требует vendor/supermat (python3 setup.py --only supermat) --
    # на инсталляции без него /gen_model без --raw будет падать с понятной
    # PipelineError на этом шаге, пока supermat не поставлен или это не
    # выключено обратно через /settings. См. gen3d/pipeline.py:estimate_material.
    "gen3d_pbr_ai":         os.getenv("GEN3D_PBR_AI", "1") == "1",
    "recap_enabled":    os.getenv("RECAP_ENABLED",    "1") == "1",
    # Держать в синхроне с mcp_agent/model_config.py:OLLAMA_NUM_CTX вручную —
    # не импортируется напрямую (model_config.py сам импортирует settings,
    # обратный импорт был бы циклическим — тот же паттерн, что у любой
    # другой model_config.py-константы, которую settings.py не может
    # импортировать напрямую). Расхождение не ломает ничего катастрофично
    # (это порог для МЕЖходового compress_history в cli.py, отдельная система
    # от per-turn _CompactResearchMiddleware), но означает, что "сжать при
    # 70% контекста" тихо считает контекст меньше, чем он есть на самом деле.
    "context_limit":    int(os.getenv("CONTEXT_LIMIT", "65536")),
    "compress_at":      float(os.getenv("COMPRESS_AT", "0.70")),
    # Общий выключатель _CompactResearchMiddleware (mcp_agent/compaction.py) —
    # ВНУТРИ одного хода сжимает историю тул-вызовов в дайджест, когда она
    # реально приближается к num_ctx (см. _needs_compaction в
    # compaction.py). Дефолт ВКЛ — без этого длинный тул-цикл рискует
    # переполнить контекст (см. модуль compaction.py про live-run с
    # "context shift" и обрывом ответа на полуслове). ВЫКЛ — эскейп-люк для
    # случая, когда дайджест-пересказ теряет что-то важное для конкретной
    # задачи (см. compaction.py live-run #3) и пользователь предпочитает
    # риск переполнения риску пересказа.
    "compact_history_enabled": os.getenv("COMPACT_HISTORY_ENABLED", "1") == "1",
    # update.py — кэш фоновой проверки git-обновлений (см. её докстринг).
    # last_update_check: ISO-таймстамп последнего реального `git fetch`
    # (None = ещё ни разу); update_commits_behind: сколько коммитов
    # origin/<ветка> впереди HEAD на момент последней проверки — 0, пока не
    # доказано обратное. Оба обновляются ТОЛЬКО из update.py — /update
    # (реальный pull) и фоновая проверка при старте cli.py (только
    # fetch+сравнение, без pull) читают/пишут эти же два ключа.
    "last_update_check": None,
    "update_commits_behind": 0,
    "debug":            False,
}

# Ключи, которые сохраняются (без device — определяется автоматически)
_PERSIST_KEYS = {
    "chat_model", "show_thinking", "ask_permissions", "self_heal_enabled", "pipeline_mode", "casual_answers_enabled", "optimized_tools", "always_delegate_search", "delegate_nudge_enabled", "expert_streaming_enabled", "num_ctx",
    "vision_model", "stt_model", "voice_mode", "voice_chat_model", "tts_voice_clone_path", "image_gen_model", "image_gen_device", "music_gen_device", "imggen_safety",
    "imggen_steps", "imggen_guidance", "imggen_strength", "imggen_width", "imggen_height",
    "imggen_prompt_prefix", "imggen_negative_prompt", "imggen_enhance_prompt", "recap_enabled",
    "gen3d_enabled", "gen3d_target_faces", "gen3d_hunyuan_profile", "gen3d_skin_source",
    "gen_agent_tools", "compact_history_enabled",
    "last_update_check", "update_commits_behind",
    "debug",
}

# Загружаем сохранённые настройки поверх дефолтов
try:
    for _key, _value in _conn.execute("SELECT key, value FROM settings"):
        if _key in _state:
            _state[_key] = json.loads(_value)
except Exception:
    pass

# DEBUG из env — ЕДИНСТВЕННОЕ исключение из "БД поверх дефолтов" выше: env
# нужен как временный флаг на один конкретный прогон (не хочется, чтобы он
# тихо перекрывался тем, что было переключено через /settings в прошлый
# раз), поэтому он приоритетнее сохранённого значения, а не наоборот. Когда
# DEBUG не задан в окружении вообще — используется то, что переключено через
# /settings (или дефолт False), и он же остаётся между запусками.
if os.getenv("DEBUG") is not None:
    _state["debug"] = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")


def get(key: str):
    return _state.get(key)


def set_value(key: str, value) -> None:
    # voice_mode ON/OFF также переключает chat_model на voice_chat_model
    # (дефолт qwen3:8b, но настраивается отдельно, см. _ITEMS в
    # ui/tui/settings.py — там же пояснение, почему тут желательна именно
    # быстрая модель) — qwen3-coder:30b слишком тяжёлая/заточенная под кодинг
    # для голосового общения. Запоминаем, какая модель была ДО включения
    # voice_mode (в памяти, не персистентно — временное состояние сессии),
    # чтобы выключение вернуло её обратно, а не оставило пользователя на
    # voice_chat_model навсегда.
    if key == "voice_mode" and value != _state.get("voice_mode"):
        if value:
            _state["_pre_voice_chat_model"] = _state.get("chat_model")
            _state["chat_model"] = _state.get("voice_chat_model")
        else:
            prev = _state.pop("_pre_voice_chat_model", None)
            if prev:
                _state["chat_model"] = prev
            # Возврат к тяжёлой кодовой модели должен освобождать всё, чем
            # обзавёлся голосовой режим (Whisper, voice_chat_model, ...) —
            # иначе она грузится ПОВЕРХ уже занятой памяти/VRAM вместо того,
            # чтобы получить её всю. _state["chat_model"] уже восстановлен
            # выше — model_lifecycle читает его как "текущую", не выгружая.
            try:
                from model_lifecycle import unload_idle_models
                unload_idle_models()
            except Exception:
                pass
    # Если голосовой режим уже включён и пользователь меняет саму
    # voice_chat_model (а не voice_mode) — переключить chat_model сразу,
    # не только при следующем ON/OFF.
    if key == "voice_chat_model" and _state.get("voice_mode"):
        _state["chat_model"] = value
    # gen_agent_tools flips which MCP servers build_mcp_connections
    # (mcp_agent/config.py) includes — but _get_tools()/_get_agent()/
    # _get_role_agent() (mcp_agent/agent_builder.py) all cache their build
    # results on keys that don't include this setting at all, so without
    # explicitly busting those caches here, the change would silently sit
    # inert until the next flowai process instead of the very next turn.
    if key == "gen_agent_tools" and value != _state.get("gen_agent_tools"):
        try:
            from mcp_agent.agent_builder import invalidate_tool_caches
            invalidate_tool_caches()
        except Exception:
            pass
    _state[key] = value
    _save()


def _save() -> None:
    try:
        _conn.executemany(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [(k, json.dumps(_state[k])) for k in _PERSIST_KEYS if k in _state],
        )
        _conn.commit()
    except Exception:
        pass
