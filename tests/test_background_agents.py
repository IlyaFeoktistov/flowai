"""delegate_tool's background-run registry — stage_runner.run_stage collects
a stage's uncollected background agents before the turn ends and cancels
them when the stage exits; one stage never touches another's."""
import asyncio

from mcp_agent import delegate_tool


def _add(agent_id, owner, coro):
    task = asyncio.get_running_loop().create_task(coro)
    delegate_tool._background[agent_id] = {"task": task, "type": "explore", "label": agent_id, "owner": owner}
    return task


def test_collect_returns_reports_and_forgets_only_own_runs():
    async def go():
        mine, other = object(), object()

        async def report(text):
            await asyncio.sleep(0.01)
            return text

        _add("a1", mine, report("found X"))
        _add("a2", mine, report("found Y"))
        foreign = _add("b1", other, asyncio.sleep(10))
        text = await delegate_tool.collect_background(mine)
        assert "found X" in text and "found Y" in text
        assert delegate_tool.pending_background(mine) == []
        assert delegate_tool.pending_background(other) == ["b1"]
        delegate_tool.cancel_background(other)
        await asyncio.sleep(0)
        assert foreign.cancelled()
    asyncio.run(go())


def test_failed_run_is_reported_not_raised():
    async def go():
        owner = object()

        async def boom():
            raise RuntimeError("model died")

        _add("c1", owner, boom())
        assert "Agent failed: model died" in await delegate_tool.collect_background(owner)
    asyncio.run(go())


def test_run_stage_collects_uncollected_background_agents_before_ending(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage
    import mcp_agent.agent as agent_mod
    from mcp_agent import stage_runner

    seen_payloads = []

    async def fake_round(agent, payload, config, on_event, emitted, mid_turn_queue=None):
        seen_payloads.append(payload["messages"][-1].content)
        if len(seen_payloads) == 1:
            # the model starts a background agent and answers right away
            async def report():
                await asyncio.sleep(0.01)
                return "agent report: 3 bugs"
            _add("bg1", delegate_tool.current_stage_owner.get(), report())
            msgs = [AIMessage(content="Готово, агенты работают.")]
        else:
            msgs = [AIMessage(content="Итог: 3 бага.")]
        return {"messages": msgs}, 0, 1, 1, 1, False, False, 0

    monkeypatch.setattr(agent_mod, "_stream_round", fake_round)
    events = []

    async def on_event(e):
        events.append(e)

    result = asyncio.run(stage_runner.run_stage(
        None, {"messages": [HumanMessage(content="task")]}, on_event,
        judge_model=None, tools_by_name={}, read_history={},
        verdict_fn=lambda *a: {"relevant": True, "reason": ""}, guidance_fn=lambda *a: "",
        max_attempts=1, recursion_limit=10, stage_name="main",
    ))
    assert result.final_text == "Итог: 3 бага."
    assert "agent report: 3 bugs" in seen_payloads[1]
    assert any(e.get("type") == "tool_start" and e.get("name") == "agent_result" for e in events)
    assert delegate_tool._background == {}
