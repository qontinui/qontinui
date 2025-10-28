"""State management for Qontinui.

This module provides specialized storage for application states with
metadata tracking and versioning support.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from ..logging import get_logger
from .file_storage import FileStorage
from .serializers import JsonSerializer

logger = get_logger(__name__)


class StateManager:
    """State manager for saving and loading application states.

    Features:
        - Automatic metadata injection (timestamps, state name)
        - Automatic backups on save
        - JSON-based storage for readability
        - State listing and enumeration
    """

    def __init__(self, base_path: Path | None = None) -> None:
        """Initialize state manager.

        Args:
            base_path: Base path for storage (defaults to settings)
        """
        self.storage = FileStorage(base_path=base_path)
        self.states_folder = "states"

        # Ensure states directory exists
        states_path = self.storage.base_path / self.states_folder
        states_path.mkdir(exist_ok=True)

        logger.info("state_manager_initialized", path=str(states_path))

    def save_state(self, state_name: str, state_data: dict[str, Any]) -> Path:
        """Save state with metadata.

        Args:
            state_name: State name
            state_data: State data dictionary

        Returns:
            Path where state was saved
        """
        # Inject metadata
        enriched_data = {
            **state_data,
            "_saved_at": datetime.now().isoformat(),
            "_name": state_name,
        }

        path = self.storage.save(
            key=state_name,
            data=enriched_data,
            subfolder=self.states_folder,
            serializer=JsonSerializer(),
            backup=True,
        )

        logger.info("state_saved", state_name=state_name, path=str(path))

        return path

    def load_state(self, state_name: str) -> dict[str, Any] | None:
        """Load state data.

        Args:
            state_name: State name

        Returns:
            State data or None if not found
        """
        data = self.storage.load(
            key=state_name,
            subfolder=self.states_folder,
            serializer=JsonSerializer(),
            default=None,
        )

        if data is not None:
            logger.info("state_loaded", state_name=state_name)

        return data

    def delete_state(self, state_name: str) -> bool:
        """Delete a state.

        Args:
            state_name: State name

        Returns:
            True if deleted, False if not found
        """
        result = self.storage.delete(key=state_name, subfolder=self.states_folder)

        if result:
            logger.info("state_deleted", state_name=state_name)

        return result

    def state_exists(self, state_name: str) -> bool:
        """Check if state exists.

        Args:
            state_name: State name

        Returns:
            True if exists
        """
        return self.storage.exists(key=state_name, subfolder=self.states_folder)

    def list_states(self) -> list[str]:
        """List all saved states.

        Returns:
            List of state names
        """
        files = self.storage.list_files(
            subfolder=self.states_folder,
            pattern="*.json",
        )
        return [f.stem for f in files]

    def get_state_info(self, state_name: str) -> dict[str, Any] | None:
        """Get state metadata without loading full data.

        Args:
            state_name: State name

        Returns:
            Dictionary with state info or None
        """
        data = self.load_state(state_name)
        if data is None:
            return None

        return {
            "name": data.get("_name", state_name),
            "saved_at": data.get("_saved_at"),
            "size": self.storage.get_size(state_name, self.states_folder),
        }
