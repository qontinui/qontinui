"""Serialization handlers for different data formats.

This module provides a unified interface for serializing and deserializing
data in various formats (JSON, Pickle, etc.).
"""

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..exceptions import StorageReadException, StorageWriteException
from ..logging import get_logger

logger = get_logger(__name__)


class Serializer(ABC):
    """Base interface for data serializers."""

    @abstractmethod
    def serialize(self, data: Any, path: Path) -> None:
        """Serialize data to file.

        Args:
            data: Data to serialize
            path: Target file path

        Raises:
            StorageWriteException: If serialization fails
        """
        pass

    @abstractmethod
    def deserialize(self, path: Path) -> Any:
        """Deserialize data from file.

        Args:
            path: Source file path

        Returns:
            Deserialized data

        Raises:
            StorageReadException: If deserialization fails
        """
        pass

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Get the file extension for this serializer."""
        pass


class JsonSerializer(Serializer):
    """JSON serialization handler."""

    def __init__(self, indent: int = 2, ensure_ascii: bool = False) -> None:
        """Initialize JSON serializer.

        Args:
            indent: Indentation level for pretty printing
            ensure_ascii: Whether to escape non-ASCII characters
        """
        self.indent = indent
        self.ensure_ascii = ensure_ascii

    def serialize(self, data: Any, path: Path) -> None:
        """Serialize data to JSON file.

        Args:
            data: Data to serialize (must be JSON serializable)
            path: Target file path

        Raises:
            StorageWriteException: If serialization fails
        """
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    data,
                    f,
                    indent=self.indent,
                    ensure_ascii=self.ensure_ascii,
                    default=str,
                )

            logger.debug("json_serialized", path=str(path), size=path.stat().st_size)

        except Exception as e:
            raise StorageWriteException(key=path.stem, storage_type="JSON", reason=str(e)) from e

    def deserialize(self, path: Path) -> Any:
        """Deserialize data from JSON file.

        Args:
            path: Source file path

        Returns:
            Deserialized data

        Raises:
            StorageReadException: If deserialization fails
        """
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            logger.debug("json_deserialized", path=str(path))

            return data

        except Exception as e:
            raise StorageReadException(key=path.stem, storage_type="JSON", reason=str(e)) from e

    @property
    def file_extension(self) -> str:
        """Get the file extension for JSON files."""
        return ".json"


class PickleSerializer(Serializer):
    """Pickle serialization handler.

    SECURITY WARNING:
    Pickle is inherently insecure when loading untrusted data. This serializer
    should only be used for internal state files in trusted locations.

    Safe usage:
    - Saving/loading Qontinui-generated state files
    - Files stored in project-controlled directories
    - Local file system with appropriate permissions

    Unsafe usage (DO NOT):
    - Loading files from network sources
    - Loading user-uploaded files
    - Loading files from shared/world-writable directories
    - Any file that could be modified by untrusted parties

    For untrusted data exchange, use JsonSerializer instead.
    See docs/SECURITY.md for comprehensive security guidance.
    """

    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
        """Initialize Pickle serializer.

        Args:
            protocol: Pickle protocol version to use
        """
        self.protocol = protocol

    def serialize(self, data: Any, path: Path) -> None:
        """Serialize data to Pickle file.

        Saves data to a trusted location using pickle. Ensure the target path
        is in a directory you control with appropriate permissions.

        Args:
            data: Data to serialize
            path: Target file path (should be in a trusted directory)

        Raises:
            StorageWriteException: If serialization fails
        """
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f, protocol=self.protocol)

            logger.debug(
                "pickle_serialized",
                path=str(path),
                type=type(data).__name__,
            )

        except Exception as e:
            raise StorageWriteException(key=path.stem, storage_type="Pickle", reason=str(e)) from e

    def deserialize(self, path: Path) -> Any:
        """Deserialize data from Pickle file.

        SECURITY WARNING:
        Loading pickle files can execute arbitrary code. Only load files that:
        - Were created by your own Qontinui installation
        - Are stored in trusted locations you control
        - Have not been modified by untrusted parties

        Never load pickle files from:
        - Network downloads or API responses
        - User uploads or external sources
        - Shared directories accessible by untrusted users
        - Any location that could be compromised

        Args:
            path: Source file path (must be from a trusted source)

        Returns:
            Deserialized data

        Raises:
            StorageReadException: If deserialization fails
        """
        try:
            with open(path, "rb") as f:
                # Security: pickle.load is safe here as files are local-only and user-controlled.
                # Network-transmitted or user-uploaded files should never use pickle.
                data = pickle.load(f)  # noqa: S301

            logger.debug(
                "pickle_deserialized",
                path=str(path),
                type=type(data).__name__,
            )

            return data

        except Exception as e:
            raise StorageReadException(key=path.stem, storage_type="Pickle", reason=str(e)) from e

    @property
    def file_extension(self) -> str:
        """Get the file extension for Pickle files."""
        return ".pkl"
