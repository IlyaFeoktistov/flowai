import os
from .base import MemoryStore
from .sqlite_store import SQLiteMemoryStore

_store: MemoryStore = SQLiteMemoryStore()

DEFAULT_USER = os.getenv("MEMORY_USER", "default")


def get_store() -> MemoryStore:
    return _store
