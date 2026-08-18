"""mcp_agent/build_cache.py:BuildCache — the shared get-or-build cache used
by agent_builder.py (x3), router.py and dnd_agent.py (see that module's
docstring for why one implementation replaced five)."""
import asyncio

import pytest

from mcp_agent.build_cache import BuildCache


@pytest.mark.asyncio
async def test_builds_once_for_same_key_and_freshness():
    cache = BuildCache()
    calls = []

    async def build():
        calls.append(1)
        return "value"

    assert await cache.get_or_build("k", "fresh", build) == "value"
    assert await cache.get_or_build("k", "fresh", build) == "value"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_rebuilds_when_freshness_changes():
    cache = BuildCache()
    calls = []

    async def build():
        calls.append(1)
        return f"value{len(calls)}"

    v1 = await cache.get_or_build("k", "fresh1", build)
    v2 = await cache.get_or_build("k", "fresh2", build)
    assert v1 != v2
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_on_stale_fires_with_old_and_new_freshness_only_after_a_prior_build():
    cache = BuildCache()
    stale_calls = []

    async def on_stale(old, new):
        stale_calls.append((old, new))

    async def build():
        return "v"

    # First build for this key — no prior entry, on_stale must NOT fire.
    await cache.get_or_build("k", "fresh1", build, on_stale=on_stale)
    assert stale_calls == []

    # Second build, freshness changed — on_stale fires with (old, new).
    await cache.get_or_build("k", "fresh2", build, on_stale=on_stale)
    assert stale_calls == [("fresh1", "fresh2")]


@pytest.mark.asyncio
async def test_different_keys_are_independent():
    cache = BuildCache()
    calls = []

    async def build():
        calls.append(1)
        return len(calls)

    v1 = await cache.get_or_build("a", "f", build)
    v2 = await cache.get_or_build("b", "f", build)
    assert v1 != v2
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_concurrent_build_of_same_new_key_happens_once():
    cache = BuildCache()
    call_count = 0

    async def slow_build():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.02)
        return "built"

    results = await asyncio.gather(*[cache.get_or_build("x", "f", slow_build) for _ in range(5)])
    assert all(r == "built" for r in results)
    assert call_count == 1


@pytest.mark.asyncio
async def test_values_bool_and_clear_mirror_dict_semantics():
    cache = BuildCache()
    assert bool(cache) is False
    assert list(cache.values()) == []

    async def build():
        return "v"

    await cache.get_or_build("k", "f", build)
    assert bool(cache) is True
    assert list(cache.values()) == ["v"]

    cache.clear()
    assert bool(cache) is False
    assert list(cache.values()) == []
