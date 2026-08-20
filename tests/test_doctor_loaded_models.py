"""doctor.py's _check_loaded_models — reports every currently-resident
model instance, not just what `ollama ps` shows: with
expert_streaming_enabled=ON (this project's default, see CLAUDE.md), the
chat model is served by expert-streaming's own separate llama.cpp process,
entirely outside the Ollama daemon — `ollama ps` alone would under-count
active instances in exactly that default configuration."""
import pytest

import doctor


class _FakeModel:
    def __init__(self, model, size, size_vram):
        self.model = model
        self.size = size
        self.size_vram = size_vram


class _FakePsResponse:
    def __init__(self, models):
        self.models = models


class _FakeOllamaClient:
    def __init__(self, models):
        self._models = models

    async def ps(self):
        return _FakePsResponse(self._models)


def _patch_ollama_ps(monkeypatch, models):
    import ollama
    monkeypatch.setattr(ollama, "AsyncClient", lambda host=None: _FakeOllamaClient(models))


@pytest.mark.asyncio
async def test_no_models_resident_warns(monkeypatch):
    _patch_ollama_ps(monkeypatch, [])
    monkeypatch.setattr(doctor.expert_streaming, "live_state", lambda: None)

    check = await doctor._check_loaded_models()

    assert check.level == doctor._WARN
    assert "не резидентна" in check.summary


@pytest.mark.asyncio
async def test_lists_ollama_model_with_size_and_gpu_percent(monkeypatch):
    _patch_ollama_ps(monkeypatch, [_FakeModel("nomic-embed-text", 300 * 1024**2, 300 * 1024**2)])
    monkeypatch.setattr(doctor.expert_streaming, "live_state", lambda: None)
    monkeypatch.setattr(doctor, "_nvidia_smi_gpu_memory", lambda: (None, None))

    check = await doctor._check_loaded_models()

    assert check.level == doctor._OK
    assert "1 активно" in check.summary
    assert "nomic-embed-text" in check.summary
    assert "100% GPU" in check.summary


@pytest.mark.asyncio
async def test_counts_expert_streaming_as_its_own_instance_alongside_ollama(monkeypatch):
    _patch_ollama_ps(monkeypatch, [_FakeModel("nomic-embed-text", 300 * 1024**2, 300 * 1024**2)])
    monkeypatch.setattr(doctor.expert_streaming, "live_state", lambda: {
        "pid": 4242, "port": 8090, "model_tag": "glm-4.7-flash:q4_K_M",
    })
    monkeypatch.setattr(doctor, "_nvidia_smi_gpu_memory", lambda: (None, None))

    check = await doctor._check_loaded_models()

    assert "2 активно" in check.summary
    assert "glm-4.7-flash:q4_K_M" in check.summary
    assert "expert-streaming" in check.summary
    assert "pid 4242" in check.summary


@pytest.mark.asyncio
async def test_reports_total_gpu_usage_and_expert_streaming_share_when_available(monkeypatch):
    _patch_ollama_ps(monkeypatch, [_FakeModel("nomic-embed-text", 1024**3, 1024**3)])  # 1GB, 100% GPU
    monkeypatch.setattr(doctor.expert_streaming, "live_state", lambda: {
        "pid": 4242, "port": 8090, "model_tag": "glm-4.7-flash:q4_K_M",
    })
    monkeypatch.setattr(doctor, "_nvidia_smi_gpu_memory", lambda: (5.0, 6.1))

    check = await doctor._check_loaded_models()

    assert "GPU занято 5.0/6.1GB" in check.summary
    assert "expert-streaming" in check.summary  # the ~4GB unaccounted-for note


@pytest.mark.asyncio
async def test_ollama_ps_failure_with_no_expert_streaming_warns(monkeypatch):
    import ollama

    class _Broken:
        def __init__(self, host=None):
            pass

        async def ps(self):
            raise ConnectionError("refused")

    monkeypatch.setattr(ollama, "AsyncClient", _Broken)
    monkeypatch.setattr(doctor.expert_streaming, "live_state", lambda: None)

    check = await doctor._check_loaded_models()

    assert check.level == doctor._WARN
    assert "ollama ps" in check.summary


@pytest.mark.asyncio
async def test_ollama_ps_failure_but_expert_streaming_alive_still_reports_ok(monkeypatch):
    import ollama

    class _Broken:
        def __init__(self, host=None):
            pass

        async def ps(self):
            raise ConnectionError("refused")

    monkeypatch.setattr(ollama, "AsyncClient", _Broken)
    monkeypatch.setattr(doctor.expert_streaming, "live_state", lambda: {
        "pid": 4242, "port": 8090, "model_tag": "glm-4.7-flash:q4_K_M",
    })
    monkeypatch.setattr(doctor, "_nvidia_smi_gpu_memory", lambda: (None, None))

    check = await doctor._check_loaded_models()

    assert check.level == doctor._OK
    assert "1 активно" in check.summary


def test_nvidia_smi_gpu_memory_parses_csv(monkeypatch):
    class _Result:
        returncode = 0
        stdout = "5120, 6141\n"

    monkeypatch.setattr(doctor.subprocess, "run", lambda *a, **k: _Result())

    used, total = doctor._nvidia_smi_gpu_memory()

    assert used == pytest.approx(5.0, abs=0.01)
    assert total == pytest.approx(6.0, abs=0.01)


def test_nvidia_smi_gpu_memory_none_when_command_missing(monkeypatch):
    def _raise(*a, **k):
        raise FileNotFoundError()
    monkeypatch.setattr(doctor.subprocess, "run", _raise)

    assert doctor._nvidia_smi_gpu_memory() == (None, None)


def test_nvidia_smi_gpu_memory_none_on_empty_output(monkeypatch):
    class _Result:
        returncode = 0
        stdout = ""
    monkeypatch.setattr(doctor.subprocess, "run", lambda *a, **k: _Result())

    assert doctor._nvidia_smi_gpu_memory() == (None, None)
