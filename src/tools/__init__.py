"""
Tools module.

После миграции на MCP (mcp_agent/) агентные тулы живут в mcp_agent/servers/
и подключаются через MCP-протокол, а не через tools/registry.py (удалён
при cutover).

Что осталось здесь и почему:
  - base.py     — ToolResult/ok/fail, используется image_gen.py
  - confirm.py  — ask_permission(), используется напрямую cli.py и mcp_agent/agent.py
  - image_gen.py — используется напрямую cli.py (команда /gen), вне агентного пайплайна
"""
