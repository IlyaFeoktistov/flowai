"""expert_streaming.live_state() — is_running() only reflects whether THIS
flowai process's own _proc launched the server; a server ADOPTED from
another flowai instance (see the module's adoption mechanism) is just as
alive but is_running() would say False for it. doctor.py needs the real
answer regardless of ownership, hence this separate, pid-liveness-checked
reader of the shared state file."""
import json
import os

import expert_streaming


def test_live_state_none_when_no_state_file(tmp_path, monkeypatch):
    monkeypatch.setattr(expert_streaming, "_STATE_PATH", tmp_path / "missing.json")
    assert expert_streaming.live_state() is None


def test_live_state_none_when_file_is_not_valid_json(tmp_path, monkeypatch):
    p = tmp_path / "state.json"
    p.write_text("not json")
    monkeypatch.setattr(expert_streaming, "_STATE_PATH", p)
    assert expert_streaming.live_state() is None


def test_live_state_none_when_recorded_pid_is_dead(tmp_path, monkeypatch):
    # A pid essentially guaranteed not to be alive right now.
    dead_pid = 2**30
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"pid": dead_pid, "port": 8090, "model_tag": "x", "num_ctx": 1, "show_thinking": False}))
    monkeypatch.setattr(expert_streaming, "_STATE_PATH", p)
    assert expert_streaming.live_state() is None


def test_live_state_returns_dict_when_pid_is_alive(tmp_path, monkeypatch):
    p = tmp_path / "state.json"
    p.write_text(json.dumps({
        "pid": os.getpid(), "port": 8090, "model_tag": "glm-4.7-flash:q4_K_M",
        "num_ctx": 65536, "show_thinking": False,
    }))
    monkeypatch.setattr(expert_streaming, "_STATE_PATH", p)

    state = expert_streaming.live_state()

    assert state["pid"] == os.getpid()
    assert state["model_tag"] == "glm-4.7-flash:q4_K_M"
