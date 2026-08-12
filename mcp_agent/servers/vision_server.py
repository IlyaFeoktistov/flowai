"""
Кастомный MCP-сервер: анализ/описание изображений через локальную
vision-модель Ollama (settings.vision_model, напр. qwen2.5vl:7b) — отдельно
от chat_model, который эту роль не выполняет (см. settings.py: осознанно
разные теги, чтобы не грузить в основную "тяжёлую" модель мультимодальные
веса).

Готового MCP-сервера под наш локальный Ollama-инстанс в реестре нет — свой,
по тому же паттерну, что image_gen_server.py (свой пайплайн под конкретное
локальное железо/модели).

Запуск: python3 -m mcp_agent.servers.vision_server
"""
import os
import sys
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import ollama  # noqa: E402
import settings  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("vision")

_client = ollama.AsyncClient(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))


@mcp.tool()
async def analyze_image(path: str, question: str = "") -> str:
    """Look at an EXISTING local image file and answer a question about it
    (or, with no question, give a general description) — OCR, what's
    depicted, colors/composition, spotting a specific detail. `path` must be
    a real, already-existing image file (one the user pasted/attached, or a
    previous generate_image/edit_image result) — this tool only reads
    images, it never creates or modifies one (that's generate_image/
    edit_image)."""
    src = Path(path)
    if not src.is_file():
        return f"Error: file not found: {path}"

    model = settings.get("vision_model")
    prompt = question.strip() or "Describe this image in detail."

    try:
        response = await _client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [str(src)]}],
        )
    except ollama.ResponseError as e:
        return f"Error: {e}"

    return response["message"]["content"]


if __name__ == "__main__":
    mcp.run()
