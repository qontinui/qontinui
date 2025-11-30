"""Configuration management for Qontinui.

This module provides specialized storage for application configurations
with JSON-based persistence.
"""

from pathlib import Path
from typing import Any

from ..logging import get_logger
from .file_storage import FileStorage
from .serializers import JsonSerializer

logger = get_logger(__name__)


class ConfigManager:
    """Configuration manager for application settings.

    Features:
        - JSON-based storage for readability
        - Simple save/load interface
        - Configuration validation support
        - Environment-specific configs
    """

    def __init__(self, base_path: Path | None = None) -> None:
        """Initialize config manager.

        Args:
            base_path: Base path for storage (defaults to settings)
        """
        self.storage = FileStorage(base_path=base_path)
        self.configs_folder = "configs"

        # Ensure configs directory exists
        configs_path = self.storage.base_path / self.configs_folder
        configs_path.mkdir(exist_ok=True)

        logger.info("config_manager_initialized", path=str(configs_path))

    def save_config(self, config_name: str, config_data: dict[str, Any]) -> Path:
        """Save configuration.

        Args:
            config_name: Config name
            config_data: Config data dictionary

        Returns:
            Path where config was saved
        """
        path = self.storage.save(
            key=config_name,
            data=config_data,
            subfolder=self.configs_folder,
            serializer=JsonSerializer(),
        )

        logger.info("config_saved", config_name=config_name, path=str(path))

        return path

    def load_config(
        self,
        config_name: str,
        default: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Load configuration.

        Args:
            config_name: Config name
            default: Default config if not found

        Returns:
            Config data or default
        """
        data = self.storage.load(
            key=config_name,
            subfolder=self.configs_folder,
            serializer=JsonSerializer(),
            default=default,
        )

        if data is not None:
            logger.info("config_loaded", config_name=config_name)

        return data  # type: ignore[no-any-return]

    def delete_config(self, config_name: str) -> bool:
        """Delete a configuration.

        Args:
            config_name: Config name

        Returns:
            True if deleted, False if not found
        """
        result = self.storage.delete(key=config_name, subfolder=self.configs_folder)

        if result:
            logger.info("config_deleted", config_name=config_name)

        return result

    def config_exists(self, config_name: str) -> bool:
        """Check if config exists.

        Args:
            config_name: Config name

        Returns:
            True if exists
        """
        return self.storage.exists(key=config_name, subfolder=self.configs_folder)

    def list_configs(self) -> list[str]:
        """List all saved configurations.

        Returns:
            List of config names
        """
        files = self.storage.list_files(
            subfolder=self.configs_folder,
            pattern="*.json",
        )
        return [f.stem for f in files]

    def update_config(
        self,
        config_name: str,
        updates: dict[str, Any],
        create_if_missing: bool = True,
    ) -> Path:
        """Update existing config with new values.

        Args:
            config_name: Config name
            updates: Dictionary with values to update
            create_if_missing: Create new config if it doesn't exist

        Returns:
            Path where config was saved

        Raises:
            FileNotFoundError: If config doesn't exist and create_if_missing is False
        """
        # Load existing config
        existing = self.load_config(config_name, default={})

        if existing is None and not create_if_missing:
            raise FileNotFoundError(f"Config '{config_name}' not found")

        # Merge updates
        merged = {**(existing or {}), **updates}

        # Save merged config
        return self.save_config(config_name, merged)
