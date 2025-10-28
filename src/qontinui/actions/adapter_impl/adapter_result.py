"""Result type for adapter action execution.

Provides a consistent return type for all adapter operations.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class AdapterResult:
    """Result from adapter action execution."""

    success: bool
    data: Any | None = None
    error: str | None = None
