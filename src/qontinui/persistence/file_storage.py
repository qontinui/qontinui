"""File-based storage backend.

This module provides file system storage operations with support for
versioning, backups, and multiple serialization formats.
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..logging import get_logger
from .serializers import JsonSerializer, Serializer

logger = get_logger(__name__)


class FileStorage:
    """File-based storage with versioning and backup support.

    Features:
        - Multiple serialization formats (JSON, Pickle)
        - Automatic directory creation
        - Versioning support
        - Backup functionality
        - File metadata queries
    """

    def __init__(
        self,
        base_path: Path | None = None,
        default_serializer: Serializer | None = None,
    ) -> None:
        """Initialize file storage.

        Args:
            base_path: Base path for storage (defaults to settings)
            default_serializer: Default serializer to use (defaults to JSON)
        """
        settings = get_settings()
        self.base_path: Path = Path(base_path or settings.dataset.path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.default_serializer = default_serializer or JsonSerializer()

        # Create backups directory
        self.backups_path = self.base_path / "backups"
        self.backups_path.mkdir(exist_ok=True)

        logger.info("file_storage_initialized", base_path=str(self.base_path))

    def save(
        self,
        key: str,
        data: Any,
        subfolder: str = "",
        serializer: Serializer | None = None,
        version: bool = False,
        backup: bool = False,
    ) -> Path:
        """Save data to file.

        Args:
            key: Storage key/filename (without extension)
            data: Data to save
            subfolder: Optional subfolder within base path
            serializer: Serializer to use (defaults to default_serializer)
            version: Add timestamp to filename
            backup: Create backup of existing file

        Returns:
            Path where data was saved
        """
        serializer = serializer or self.default_serializer

        # Prepare path
        folder = self._resolve_folder(subfolder)
        folder.mkdir(parents=True, exist_ok=True)

        # Build filename with optional versioning
        filename = self._build_filename(key, serializer.file_extension, version)
        path = folder / filename

        # Backup existing file if requested
        if backup and path.exists():
            self._create_backup(path)

        # Serialize data
        serializer.serialize(data, path)

        return path

    def load(
        self,
        key: str,
        subfolder: str = "",
        serializer: Serializer | None = None,
        version: str | None = None,
        default: Any = None,
    ) -> Any:
        """Load data from file.

        Args:
            key: Storage key/filename (without extension)
            subfolder: Optional subfolder within base path
            serializer: Serializer to use (defaults to default_serializer)
            version: Optional version timestamp
            default: Default value if file not found

        Returns:
            Loaded data or default value
        """
        serializer = serializer or self.default_serializer

        # Prepare path
        folder = self._resolve_folder(subfolder)

        # Build filename
        if version:
            filename = f"{key}_{version}{serializer.file_extension}"
        else:
            filename = f"{key}{serializer.file_extension}"

        path = folder / filename

        if not path.exists():
            return default

        # Deserialize data
        try:
            return serializer.deserialize(path)
        except Exception:
            if default is not None:
                return default
            raise

    def list_files(
        self,
        subfolder: str = "",
        pattern: str = "*",
        serializer: Serializer | None = None,
    ) -> list[Path]:
        """List files in storage.

        Args:
            subfolder: Optional subfolder
            pattern: Glob pattern for matching (without extension)
            serializer: Serializer to filter by extension

        Returns:
            List of file paths
        """
        folder = self._resolve_folder(subfolder)
        if not folder.exists():
            return []

        # Build full pattern with extension
        if serializer:
            full_pattern = f"{pattern}{serializer.file_extension}"
        else:
            full_pattern = pattern

        return sorted(folder.glob(full_pattern))

    def delete(self, key: str, subfolder: str = "") -> bool:
        """Delete stored file.

        Attempts to delete files with common extensions (.json, .pkl).

        Args:
            key: Storage key
            subfolder: Optional subfolder

        Returns:
            True if deleted, False if not found
        """
        folder = self._resolve_folder(subfolder)

        # Try common extensions
        for ext in [".json", ".pkl"]:
            path = folder / f"{key}{ext}"
            if path.exists():
                path.unlink()
                logger.debug("file_deleted", path=str(path))
                return True

        return False

    def exists(self, key: str, subfolder: str = "") -> bool:
        """Check if key exists.

        Args:
            key: Storage key
            subfolder: Optional subfolder

        Returns:
            True if exists
        """
        folder = self._resolve_folder(subfolder)

        for ext in [".json", ".pkl"]:
            if (folder / f"{key}{ext}").exists():
                return True

        return False

    def get_size(self, key: str, subfolder: str = "") -> int | None:
        """Get file size in bytes.

        Args:
            key: Storage key
            subfolder: Optional subfolder

        Returns:
            Size in bytes or None if not found
        """
        folder = self._resolve_folder(subfolder)

        for ext in [".json", ".pkl"]:
            path = folder / f"{key}{ext}"
            if path.exists():
                return path.stat().st_size

        return None

    def _resolve_folder(self, subfolder: str) -> Path:
        """Resolve folder path.

        Args:
            subfolder: Optional subfolder

        Returns:
            Resolved path
        """
        if subfolder:
            return self.base_path / subfolder
        return self.base_path

    def _build_filename(self, key: str, extension: str, version: bool) -> str:
        """Build filename with optional versioning.

        Args:
            key: Base filename
            extension: File extension (including dot)
            version: Whether to add version timestamp

        Returns:
            Complete filename
        """
        if version:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{key}_{timestamp}{extension}"
        return f"{key}{extension}"

    def _create_backup(self, path: Path) -> Path:
        """Create backup of file.

        Args:
            path: File to backup

        Returns:
            Backup path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{path.stem}_backup_{timestamp}{path.suffix}"
        backup_path = self.backups_path / backup_name

        shutil.copy2(path, backup_path)

        logger.debug("backup_created", original=str(path), backup=str(backup_path))

        return backup_path
