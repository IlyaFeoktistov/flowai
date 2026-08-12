from abc import ABC, abstractmethod


class MemoryStore(ABC):
    @abstractmethod
    async def load(self, user_id: str) -> dict:
        """Загружает память пользователя. Возвращает {} если нет."""

    @abstractmethod
    async def save(self, user_id: str, data: dict) -> None:
        """Сохраняет память пользователя."""
