def chunk_text(text: str, lines_per_chunk: int = 60, overlap: int = 10) -> list[str]:
    """Разбивка по окну строк, без AST — язык-агностично, работает
    одинаково для кода, markdown и обычного текста."""
    lines = text.splitlines()
    if not lines:
        return []

    chunks = []
    step = max(lines_per_chunk - overlap, 1)
    for start in range(0, len(lines), step):
        chunk = "\n".join(lines[start:start + lines_per_chunk])
        if chunk.strip():
            chunks.append(chunk)
        if start + lines_per_chunk >= len(lines):
            break
    return chunks
