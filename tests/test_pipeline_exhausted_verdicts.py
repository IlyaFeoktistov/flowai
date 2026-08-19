"""mcp_agent/pipeline.py:stream_chat — Coder/Verifier must not report
success when they exhausted every self-heal attempt without ever getting a
relevant=True verdict. Without this: if quick_fix runs out of retries
after repeatedly failing to call write/edit, the pipeline hands the
(untouched) round to Verifier as if quick_fix had succeeded, and Verifier
re-runs the same doomed check a second time before giving up."""
import pytest

import mcp_agent.pipeline as pipeline_mod
from mcp_agent.stage_runner import StageResult


async def _fake_classify_intent(messages):
    # Routes straight to the quick_fix branch (no Analyzer/Planner) — the
    # shortest path to the Coder<->Verifier loop this test targets.
    return {
        "needs_project": True, "needs_shell": False,
        "needs_change": True, "change_is_ambiguous": False,
    }


async def _fake_load_knowledge(repo_path):
    return None


async def _fake_get_role_agent(role, tool_names, repo_path=None):
    return (f"AGENT[{role}]", "MODEL", "JUDGE", {}, {}, None, 0)


@pytest.fixture(autouse=True)
def common_mocks(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "classify_intent", _fake_classify_intent)
    monkeypatch.setattr(pipeline_mod, "load_knowledge", _fake_load_knowledge)
    monkeypatch.setattr(pipeline_mod, "_get_role_agent", _fake_get_role_agent)
    yield


async def _collect(gen):
    return [chunk async for chunk in gen]


@pytest.mark.asyncio
async def test_coder_exhausted_without_success_is_reported_as_failure(monkeypatch):
    async def fake_run_stage(agent, payload, on_event, *, stage_name, **kw):
        assert stage_name == "quick_fix"  # Verifier must never be reached
        return StageResult(
            final_text="I tried but couldn't find the right approach.",
            verdict={"relevant": False, "reason": "no write/edit tool was called this round"},
        )

    monkeypatch.setattr(pipeline_mod, "run_stage", fake_run_stage)

    outputs = await _collect(pipeline_mod.stream_chat([{"role": "user", "content": "fix the bug"}]))
    assert len(outputs) == 1
    assert "не справился с задачей" in outputs[0]
    assert "no write/edit tool was called this round" in outputs[0]
    assert "Готово" not in outputs[0]


@pytest.mark.asyncio
async def test_verifier_exhausted_without_a_real_check_is_not_silently_treated_as_pass(monkeypatch):
    calls = {"n": 0}

    async def fake_run_stage(agent, payload, on_event, *, stage_name, **kw):
        calls["n"] += 1
        if stage_name == "quick_fix":
            return StageResult(
                final_text="Applied the fix.",
                verdict={"relevant": True, "reason": "wrote code and reported"},
            )
        assert stage_name == "verifier"
        return StageResult(
            final_text="I looked at the diff but didn't run anything.",
            verdict={"relevant": False, "reason": "no bash call was made this round"},
        )

    monkeypatch.setattr(pipeline_mod, "run_stage", fake_run_stage)

    outputs = await _collect(pipeline_mod.stream_chat([{"role": "user", "content": "fix the bug"}]))
    assert len(outputs) == 1
    assert "Verifier не смог завершить проверку" in outputs[0]
    assert "no bash call was made this round" in outputs[0]
    assert "✅ Готово" not in outputs[0]
    # Confirms it stopped after ONE verifier attempt instead of silently
    # passing through to a second Coder<->Verifier round.
    assert calls["n"] == 2  # 1 quick_fix + 1 verifier


@pytest.mark.asyncio
async def test_verifier_execution_failure_still_sends_the_change_back_to_coder(monkeypatch):
    """Sanity check that the fix above didn't overreach: a REAL, relevant
    execution_failure verdict (Verifier did its job and the check failed)
    must still loop back to Coder, not be treated as "exhausted"."""
    calls = []

    async def fake_run_stage(agent, payload, on_event, *, stage_name, **kw):
        calls.append(stage_name)
        if stage_name == "quick_fix":
            return StageResult(final_text="Applied the fix.", verdict={"relevant": True, "reason": "ok"})
        # verifier: first round reports a real, relevant execution failure;
        # second round (after the "fix" is retried) passes.
        if calls.count("verifier") == 1:
            return StageResult(
                final_text="Ran the build, it failed.",
                verdict={"relevant": True, "kind": "execution_failure", "reason": "ran a real check and it failed"},
            )
        return StageResult(final_text="Ran the build, it passed.", verdict={"relevant": True, "reason": "ran real checks and they passed"})

    monkeypatch.setattr(pipeline_mod, "run_stage", fake_run_stage)

    outputs = await _collect(pipeline_mod.stream_chat([{"role": "user", "content": "fix the bug"}]))
    assert len(outputs) == 1
    assert "✅ Готово" in outputs[0]
    assert calls == ["quick_fix", "verifier", "quick_fix", "verifier"]
