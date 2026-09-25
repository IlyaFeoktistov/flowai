"""model_params.py — layering (base <- family <- user) and parsing. settings
persistence is stubbed out so tests never touch the real SQLite file."""
import pytest

import model_params
import settings


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    state = {"chat_model": "qwen3:8b", "num_ctx": 30000, "model_params": {}}
    monkeypatch.setattr(settings, "get", lambda k: state.get(k))
    monkeypatch.setattr(settings, "set_value", lambda k, v: state.__setitem__(k, v))
    return state


def test_family_defaults_apply_only_with_include_family():
    with_family = model_params.effective("glm-4.7-flash:q4_K_M")
    without = model_params.effective("glm-4.7-flash:q4_K_M", include_family=False)
    assert with_family["repeat_penalty"] == 1.0 and with_family["min_p"] == 0.01
    assert without["repeat_penalty"] == model_params.REPEAT_PENALTY and without["min_p"] is None


def test_user_value_is_per_model_and_overrides_family_on_both_paths():
    model_params.set_param("glm-4.7-flash:q4_K_M", "temperature", 0.3)
    assert model_params.effective("glm-4.7-flash:q4_K_M")["temperature"] == 0.3
    assert model_params.effective("glm-4.7-flash:q4_K_M", include_family=False)["temperature"] == 0.3
    assert model_params.effective("qwen3:8b")["temperature"] == model_params.MODEL_TEMPERATURE


def test_reset_removes_override_and_empty_model_entry(_isolated_settings):
    model_params.set_param("qwen3:8b", "num_ctx", 8192)
    assert model_params.num_ctx() == 8192
    model_params.set_param("qwen3:8b", "num_ctx", None)
    assert model_params.num_ctx() == 30000
    assert _isolated_settings["model_params"] == {}


def test_parse_value():
    assert model_params.parse_value("top_k", "20") == 20
    assert model_params.parse_value("temperature", "0.5") == 0.5
    assert model_params.parse_value("top_p", "") is None
    with pytest.raises(ValueError):
        model_params.parse_value("num_ctx", "100")
    with pytest.raises(ValueError):
        model_params.parse_value("bogus", "1")


def test_no_mmap_is_off_by_default_and_parses_on_off():
    assert model_params.effective("glm-4.7-flash:q4_K_M")["no_mmap"] is False
    assert model_params.parse_value("no_mmap", "вкл") is True
    assert model_params.parse_value("no_mmap", "false") is False
    with pytest.raises(ValueError):
        model_params.parse_value("no_mmap", "maybe")
    model_params.set_param("glm-4.7-flash:q4_K_M", "no_mmap", True)
    assert model_params.effective("glm-4.7-flash:q4_K_M")["no_mmap"] is True
    assert model_params.effective("qwen3:8b")["no_mmap"] is False
