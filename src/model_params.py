"""
Параметры генерации чат-модели — отдельно для каждой модели, редактируются
в /settings (ui/tui/settings.py) и в вебе (GET/POST /api/v1/model_params).

Значение параметра для модели собирается в три слоя, каждый перекрывает
предыдущий:
  1. _BASE_DEFAULTS — общие дефолты (прежние константы model_config.py,
     подобранные под Qwen: низкая temperature ради точного формата
     tool-call аргументов, сильный repeat_penalty против зацикливания
     длинных инвентарей Analyzer'а — см. там же почему).
  2. _FAMILY_DEFAULTS — дефолты семейства модели (тег до ':'), для моделей,
     которым базовые дефолты вредят.
  3. Значения пользователя — settings "model_params" = {model_tag: {key: value}}.

Семейные дефолты — связка, проверенная только целиком (см. комментарий у
_FAMILY_DEFAULTS), поэтому effective(include_family=False) даёт базовые
дефолты + пользовательские значения без семейных: так вызывает
agent_builder._build_chat_model там, где связка не проверялась
(casual-ответ без тулов).

Пользовательские значения ВСЕГДА применяются: это явный выбор человека,
а не угаданная комбинация.
"""
import settings
from mcp_agent.model_config import (
    MODEL_TEMPERATURE,
    OLLAMA_NUM_PREDICT,
    REPEAT_LAST_N,
    REPEAT_PENALTY,
    TOP_K,
    TOP_P,
)

# (key, label, type, hint). None у значения = не передавать параметр вовсе,
# бэкенд берёт свой дефолт (например, min_p у llama.cpp — 0.05, у Ollama — 0).
PARAMS: list[tuple[str, str, type, str]] = [
    ("temperature",    "temperature",    float, "случайность выбора токена; ниже — точнее формат tool-call"),
    ("top_p",          "top_p",          float, "nucleus sampling: доля вероятностной массы"),
    ("top_k",          "top_k",          int,   "выбор только из K самых вероятных токенов"),
    ("min_p",          "min_p",          float, "отсекает токены с вероятностью < min_p × max"),
    ("repeat_penalty", "repeat_penalty", float, "штраф за повтор; 1.0 — выключен"),
    ("repeat_last_n",  "repeat_last_n",  int,   "сколько последних токенов учитывает штраф"),
    ("num_predict",    "num_predict",    int,   "максимум токенов в одном ответе"),
    ("num_ctx",        "num_ctx",        int,   "окно контекста; больше — больше RAM/VRAM под KV-cache"),
    ("no_mmap",        "no_mmap",        bool,  "только llama.cpp: веса целиком в RAM — промпт быстрее, но RAM не освобождается (риск OOM)"),
]
PARAM_KEYS = [k for k, *_ in PARAMS]
_TYPES = {k: t for k, _, t, _ in PARAMS}


def _base_defaults() -> dict:
    return {
        "temperature": MODEL_TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "min_p": None,
        "repeat_penalty": REPEAT_PENALTY,
        "repeat_last_n": REPEAT_LAST_N,
        "num_predict": OLLAMA_NUM_PREDICT,
        # Глобальный settings.num_ctx остаётся дефолтом для моделей без
        # своего значения (см. его докстринг в settings.py про expert-streaming).
        "num_ctx": settings.get("num_ctx"),
        # Off by default: with --no-mmap the whole GGUF sits in RAM that the
        # kernel can't evict, which OOM-kills a VM too small for model + IDE.
        "no_mmap": False,
    }


# glm-4.7-flash: базовые REPEAT_PENALTY=1.2/REPEAT_LAST_N=512 разваливают
# аргументы tool-call в бессвязный набор слов на любом нетривиальном промпте
# (длинный системный промпт + несколько тулов): синтаксис вызова GLM
# (<tool_call>name<arg_key>...<arg_value>...</tool_call>) повторяет одни и те
# же структурные токены на каждом аргументе, и сильный штраф за повтор
# борется именно с ними. Значения — рекомендованные сообществом (HF
# discussion unsloth/GLM-4.7-Flash-GGUF#23); проверены только ВМЕСТЕ —
# temperature/top_p/min_p от этой связки при repeat_penalty=1.2 дают полный
# распад текста, поэтому include_family применяет её целиком или никак.
_FAMILY_DEFAULTS: dict[str, dict] = {
    "glm-4.7-flash": {"temperature": 0.7, "top_p": 0.95, "min_p": 0.01, "repeat_penalty": 1.0},
}


def _family(model_tag: str) -> str:
    return (model_tag or "").partition(":")[0]


def _user_overrides(model_tag: str) -> dict:
    return dict((settings.get("model_params") or {}).get(model_tag) or {})


def defaults(model_tag: str, include_family: bool = True) -> dict:
    values = _base_defaults()
    if include_family:
        values.update(_FAMILY_DEFAULTS.get(_family(model_tag), {}))
    return values


def effective(model_tag: str | None = None, include_family: bool = True) -> dict:
    model_tag = model_tag or settings.get("chat_model")
    values = defaults(model_tag, include_family)
    values.update({k: v for k, v in _user_overrides(model_tag).items() if k in values})
    return values


def num_ctx(model_tag: str | None = None) -> int:
    """num_ctx текущей (или указанной) чат-модели — единственный источник для
    всего, что считает бюджет окна основной модели (compaction, compress,
    индикатор контекста, expert-streaming -c)."""
    return int(effective(model_tag)["num_ctx"])


def describe(model_tag: str) -> list[dict]:
    """Для UI: каждый параметр с текущим значением, дефолтом и флагом
    "задан пользователем"."""
    overrides = _user_overrides(model_tag)
    base = defaults(model_tag)
    eff = effective(model_tag)
    return [
        {
            "key": key, "label": label, "type": t.__name__, "hint": hint,
            "value": eff[key], "default": base[key], "overridden": key in overrides,
        }
        for key, label, t, hint in PARAMS
    ]


_TRUE = {"1", "true", "yes", "on", "да", "вкл"}
_FALSE = {"0", "false", "no", "off", "нет", "выкл"}


def _parse_bool(raw) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"expected on/off, got {raw!r}")


def parse_value(key: str, raw):
    """Строка/число из UI -> значение нужного типа. None/"" — сброс к
    дефолту. ValueError на мусор или неизвестный ключ."""
    if key not in _TYPES:
        raise ValueError(f"unknown parameter: {key}")
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return None
    if _TYPES[key] is bool:
        return _parse_bool(raw)
    value = _TYPES[key](raw)
    if key == "num_ctx" and value < 512:
        raise ValueError("num_ctx must be >= 512")
    if key in ("num_predict", "top_k", "repeat_last_n") and value < 0:
        raise ValueError(f"{key} must be >= 0")
    return value


def set_param(model_tag: str, key: str, value) -> None:
    """value=None сбрасывает параметр к дефолту. Сбрасывает кэш собранных
    агентов: параметры зашиваются в клиент модели при сборке
    (agent_builder._build_chat_model), без этого изменение ждало бы
    перезапуска."""
    if key not in _TYPES:
        raise ValueError(f"unknown parameter: {key}")
    all_params = {m: dict(p) for m, p in (settings.get("model_params") or {}).items()}
    per_model = all_params.setdefault(model_tag, {})
    if value is None:
        per_model.pop(key, None)
    else:
        per_model[key] = _parse_bool(value) if _TYPES[key] is bool else _TYPES[key](value)
    if not per_model:
        all_params.pop(model_tag, None)
    settings.set_value("model_params", all_params)
    try:
        from mcp_agent.agent_builder import invalidate_agent_caches
        invalidate_agent_caches()
    except Exception:
        pass
