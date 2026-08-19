# Генеративные фичи

Картинки, музыка, 3D-модели, голос — всё локально, никаких внешних API.
Каждая фича доступна двумя путями:

1. **Слэш-команда** (`/gen`, `/music`, `/gen_model`, `/anim`,
   `/gen_texture`, `/talk`) — прямой CLI-путь, `cli.py` вызывает
   `tools/`/`gen3d/`-код напрямую, минуя модель и tool-calling целиком.
   Работает всегда, независимо от настроек ниже.
2. **Тул модели** (`generate_image`/`edit_image`/`generate_music`/
   `generate_3d_model`/`animate_3d_model`/`generate_texture_for_model`) —
   модель сама решает вызвать, во время обычного диалога. Гейтится
   `gen_agent_tools` (дефолт ВЫКЛ, `/settings`) — во время обычного
   кодинга модель не должна ни видеть эти тулы в схеме, ни путать их с
   проектными read/write. ВЫКЛ также останавливает сами MCP-подпроцессы
   (`build_mcp_connections`) — экономия времени старта и VRAM/RAM, не
   только косметика списка тулов. `web_search`/`fetch`/`analyze_image`
   этим тумблером не гейтятся — они читающие и полезны всегда.

## Куда сохраняются результаты

**В открытый проект**, а не в служебную папку flowAI — сгенерированный
файл — результат работы ПОЛЬЗОВАТЕЛЯ над его проектом, не служебные
данные flowAI (в отличие от `data_dir()`, см.
[persistence.md](persistence.md)):

- `<cwd>/generated/` — картинки (`tools/image_gen.py`,
  `mcp_agent/servers/image_gen_server.py`), музыка
  (`mcp_agent/servers/music_server.py`).
- `<cwd>/generated/models/` — 3D-модели (`gen3d/pipeline.py`).
- `<cwd>/img-refs/` — референсные картинки пользователя для `/gen_model`
  (`@имя.png`), не коммитятся.

Оба пути вычисляются от `Path.cwd()` — той директории, откуда запущен
`flowai`, то есть открытого проекта.

## Изображения

- **Генерация** — SDXL/FLUX (`settings.image_gen_model`, дефолт
  `stabilityai/sdxl-turbo`), устройство автоопределяется
  (`image_gen_device`: `cuda`, если детектится GPU через torch/
  `nvidia-smi`, иначе `cpu`). Требует GPU для практической скорости.
- **Редактирование** — img2img по существующей картинке, сохраняет
  композицию.
- **Чтение** — `analyze_image` через vision-модель (`vision_model`,
  дефолт `llava:13b`) — описание, OCR, ответ на вопрос про картинку.
- Тонкая настройка — `imggen_steps`/`imggen_guidance`/`imggen_strength`/
  `imggen_width`/`imggen_height`/`imggen_prompt_prefix`/
  `imggen_negative_prompt`/`imggen_enhance_prompt`/`imggen_safety`
  (все в `/settings`).

## Музыка

`generate_music` / `/music` (потоковая, повторный `/music` или Ctrl+C —
стоп) / `/music_gen` (один трек напрямую) — `facebook/musicgen-small`
через HF `transformers` (не `audiocraft` — более тяжёлая/капризная
цепочка зависимостей, а `transformers` уже установлен ради
diffusers/SDXL). Устройство — `music_gen_device`, дефолт **CPU** (в
отличие от картинок): `/music` может крутиться долго, а SDXL/Ollama и
так соревнуются за те же 5.9 GB VRAM — GPU доступен по явному запросу
через `/settings`, не тихий дефолт.

## 3D-модели (`gen3d/pipeline.py`)

`/gen_model` — image-to-3D (Hunyuan3D-2GP) → ретопология → перепечка
текстуры → опциональный риг (скелет UniRig + Blender Automatic Weights,
либо полная UniRig skin-модель). Каждый тяжёлый шаг — отдельный
подпроцесс в своём `vendor/*/venv` или через системный Blender (см.
`setup.py` — почему они не могут делить venv с основным flowAI: разные,
несовместимые версии torch/CUDA).

Флаги: `--rig` (со скелетом), `--raw` (без ретопологии), `--lod N`
(ещё N моделей с более низким полигонажем), `@имя.png` (референс из
`img-refs/`, несколько `@` — батч, по модели на картинку). Мультиракурсный
режим (`--front`/`--left`/`--back`/`--right`, Hunyuan3D-2mv) — одна
модель из нескольких фиксированных ракурсов одного объекта.

Настройки (`/settings`): `gen3d_enabled` (общий выключатель — не про
качество, а про "пайплайн не установлен на этой машине"),
`gen3d_target_faces` (целевой полигонаж после ретопологии, дефолт
15000), `gen3d_hunyuan_profile` (профиль offload'а mmgp, дефолт 4 —
больший VRAM-запас, чем профиль 3), `gen3d_skin_source`
(`auto_weights`/`unirig`), `gen3d_pbr_ai` (AI-оценка
roughness/metallic поверх меша — требует `vendor/supermat`, добавляет
~6-8 минут).

3D-генерация и Ollama-чат соревнуются за одну и ту же VRAM — если во
время `/gen_model`/`/anim` открыт активный чат, пайплайн сам выгружает
текущую chat-модель перед тяжёлыми GPU-этапами (она подгрузится заново
на следующее сообщение).

- `/anim` — оживляет уже риггованную модель по описанию движения
  (Animato), `@имя` — конкретную модель.
- `/gen_texture` — перегенерирует текстуру готовой модели по референсу
  (`@модель @картинка`, порядок аргументов не важен).

## Голос

- **Ввод** (`Alt+R`) — `faster-whisper` (`stt_model`, дефолт `medium`),
  CPU. На WSL2 идёт через мост на `powershell.exe` (см. `ui/audio.py`) —
  на «чистом» Linux мост не нужен, но пока не выделен в отдельный путь.
- **Ответ** (`voice_mode`) — Chatterbox (отдельный `venv-tts`,
  Python 3.11), CPU-only, заметно медленнее реального времени. Все
  выходные аудиофайлы несут неслышимый цифровой водяной знак (`perth`,
  офлайн, не отключается через публичный API — лицензия Coqui CPML,
  некоммерческая). `tts_voice_clone_path` — референс-клип для voice
  cloning, хранится в `data_dir()` (не в проекте — общий ресурс, тот же
  принцип, что у settings/memory).
- `/talk текст` — озвучить текст напрямую, без модели.
