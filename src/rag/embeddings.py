import os
from typing import Callable

import ollama

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

_BATCH_SIZE = 32


async def embed_texts(
    texts: list[str], model: str = EMBED_MODEL,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    """Батчами через ollama.AsyncClient().embed() — прямой клиент, не
    langchain_ollama.OllamaEmbeddings: rag_server.py — самостоятельный
    FastMCP-процесс без LangGraph, а `ollama` — единственный реально
    задекларированный в requirements.txt пакет.

    on_progress(done, total) — вызывается синхронно после КАЖДОГО батча
    (не после каждого текста — Ollama сам не отдаёт progress внутри одного
    embed()-вызова), для /reindex (cli.py), который на большом проекте
    может идти минуты, а до этого не показывал вообще никакой обратной
    связи о том, сколько ещё осталось."""
    if not texts:
        return []

    client = ollama.AsyncClient()
    result: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i:i + _BATCH_SIZE]
        response = await client.embed(model=model, input=batch)
        result.extend(response["embeddings"])
        if on_progress is not None:
            on_progress(len(result), len(texts))
    return result
