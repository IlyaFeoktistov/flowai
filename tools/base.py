from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Standardized contract every tools/*.py handler returns (via to_dict())."""
    output: Any
    error: str | None = None
    status: str = "ok"

    def to_dict(self) -> dict:
        return {
            "output": self.output,
            "error": self.error,
            "status": self.status
        }


def ok(output: Any) -> dict:
    """Build a successful tool result."""
    return ToolResult(output=output, error=None, status="ok").to_dict()


def fail(error: str, output: Any = None) -> dict:
    """Build a failed tool result."""
    return ToolResult(output=output, error=error, status="error").to_dict()