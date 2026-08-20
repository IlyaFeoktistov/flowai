from .store import VectorStore
from .chunking import chunk_text
from .embeddings import embed_texts, EMBED_MODEL

# code_store_path/reindex_code (index_code.py) deliberately NOT re-exported
# here — that module pulls in mcp_agent.servers.file_ops_server (for
# SKIP_DIRS) as a transitive dependency, and this package's other three
# names are imported by cli.py on every single session start (the dialog
# store) — no reason to drag a whole MCP-server module into that path for
# the two names only cli.py's /reindex command and rag_server.py actually
# need. Import them directly from rag.index_code instead.
__all__ = ["VectorStore", "chunk_text", "embed_texts", "EMBED_MODEL"]
