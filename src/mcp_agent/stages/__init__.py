"""Per-роль verdict_fn/guidance_fn для mcp_agent/stage_runner.py:run_stage —
по одному модулю на роль пайплайна Router->Analyzer->Planner->Coder->
Verifier (mcp_agent/roles.py). Собраны из тех же чистых функций
mcp_agent/self_heal.py, что раньше жили одним монолитным деревом внутри
mcp_agent/agent.py:stream_chat, просто распределены по стадиям."""
