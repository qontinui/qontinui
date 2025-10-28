"""Base exception classes for Qontinui framework.

This module contains the root exception hierarchy that all other
Qontinui exceptions inherit from.
"""

from typing import Any


class QontinuiException(Exception):
    """Base exception for all Qontinui errors.

    Attributes:
        message: Human-readable error message
        error_code: Optional error code for programmatic handling
        context: Additional context information
    """

    def __init__(
        self, message: str, error_code: str | None = None, context: dict[str, Any] | None = None
    ) -> None:
        """Initialize the exception.

        Args:
            message: Error message
            error_code: Optional error code
            context: Optional context dictionary
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}

    def __str__(self) -> str:
        """Return string representation."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
