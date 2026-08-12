import os

import ollama

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

_BATCH_SIZE = 32


async def embed_texts(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Батчами через ollama.AsyncClient().embed() — прямой клиент, не
    langchain_ollama.OllamaEmbeddings: rag_server.py — самостоятельный
    FastMCP-процесс без LangGraph, а `ollama` — единственный реально
    задекларированный в requirements.txt пакет."""
    if not texts:
        return []

    client = ollama.AsyncClient()
    result: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i:i + _BATCH_SIZE]
        response = await client.embed(model=model, input=batch)
        result.extend(response["embeddings"])
    return result
