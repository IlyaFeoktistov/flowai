from datetime import datetime

from crawl4ai import AsyncWebCrawler

from .chunking import chunk_text
from .embeddings import embed_texts
from .store import VectorStore


async def remember_url(url: str, store: VectorStore) -> str:
    """Явное, осознанное сохранение страницы — не пассивная индексация
    каждого web_search/fetch результата. Большая часть таких результатов —
    шум, который модель сразу отбрасывает; пассивная индексация тратила бы
    эмбеддинги и засоряла индекс. Сохранение — всегда явное действие модели,
    как update_memory/update_knowledge."""
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    if not result.success:
        return f"Failed to fetch {url}: {result.error_message}"

    text = str(result.markdown).strip()
    if not text:
        return f"No content extracted from {url}"

    chunks = chunk_text(text, lines_per_chunk=40, overlap=5)
    embeddings = await embed_texts(chunks)
    fetched_at = datetime.now().isoformat(timespec="seconds")

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        store.add(
            f"{url}:{i}",
            chunk,
            embedding,
            {"source_type": "external", "source": url, "chunk_idx": i, "fetched_at": fetched_at},
        )
    store.save()

    return f"Remembered {len(chunks)} chunks from {url}"
