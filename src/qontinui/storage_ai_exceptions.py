"""Storage and AI/ML exceptions.

This module contains exceptions for storage operations and
AI/ML model operations.
"""

from .base_exceptions import QontinuiException


class StorageException(QontinuiException):
    """Base exception for storage/persistence errors."""

    pass


class StorageReadException(StorageException):
    """Raised when reading from storage fails."""

    def __init__(self, key: str, storage_type: str, reason: str, **kwargs) -> None:
        """Initialize with read details."""
        super().__init__(
            f"Failed to read '{key}' from {storage_type}: {reason}",
            error_code="STORAGE_READ_FAILED",
            context={
                "key": key,
                "storage_type": storage_type,
                "reason": reason,
                **kwargs,
            },
        )


class StorageWriteException(StorageException):
    """Raised when writing to storage fails."""

    def __init__(self, key: str, storage_type: str, reason: str, **kwargs) -> None:
        """Initialize with write details."""
        super().__init__(
            f"Failed to write '{key}' to {storage_type}: {reason}",
            error_code="STORAGE_WRITE_FAILED",
            context={
                "key": key,
                "storage_type": storage_type,
                "reason": reason,
                **kwargs,
            },
        )


class AIException(QontinuiException):
    """Base exception for AI/ML errors."""

    pass


class ModelLoadException(AIException):
    """Raised when model loading fails."""

    def __init__(self, model_name: str, reason: str, **kwargs) -> None:
        """Initialize with model details."""
        super().__init__(
            f"Failed to load model '{model_name}': {reason}",
            error_code="MODEL_LOAD_FAILED",
            context={"model_name": model_name, "reason": reason, **kwargs},
        )


class InferenceException(AIException):
    """Raised when model inference fails."""

    def __init__(self, model_name: str, reason: str, **kwargs) -> None:
        """Initialize with inference details."""
        super().__init__(
            f"Inference failed for model '{model_name}': {reason}",
            error_code="INFERENCE_FAILED",
            context={"model_name": model_name, "reason": reason, **kwargs},
        )


class VectorDatabaseException(AIException):
    """Raised when vector database operation fails."""

    def __init__(self, operation: str, reason: str, **kwargs) -> None:
        """Initialize with database details."""
        super().__init__(
            f"Vector database operation '{operation}' failed: {reason}",
            error_code="VECTOR_DB_FAILED",
            context={"operation": operation, "reason": reason, **kwargs},
        )
