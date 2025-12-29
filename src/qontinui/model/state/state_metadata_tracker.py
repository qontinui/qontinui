"""StateMetadataTracker - Tracks state metadata and statistics.

Manages access counts, timestamps, tags, and custom metadata for states.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from qontinui_schemas.common import utc_now

logger = logging.getLogger(__name__)


@dataclass
class StateMetadata:
    """Metadata about a stored state."""

    created_at: datetime = field(default_factory=utc_now)
    last_accessed: datetime = field(default_factory=utc_now)
    access_count: int = 0
    transition_count: int = 0
    average_duration: float = 0.0
    last_error: str | None = None
    tags: set[str] = field(default_factory=set)
    custom_data: dict[str, Any] = field(default_factory=dict)


class StateMetadataTracker:
    """Tracks metadata and statistics for states.

    Single responsibility: Maintain metadata, access counts, and tags for states.
    """

    def __init__(self) -> None:
        """Initialize the metadata tracker."""
        self._metadata: dict[str, StateMetadata] = {}
        self._lock = threading.RLock()

    def register_state(self, name: str) -> None:
        """Register a new state with initial metadata.

        Args:
            name: State name to register
        """
        with self._lock:
            if name not in self._metadata:
                self._metadata[name] = StateMetadata()
                logger.debug(f"Registered metadata for '{name}'")

    def unregister_state(self, name: str) -> None:
        """Unregister a state and remove its metadata.

        Args:
            name: State name to unregister
        """
        with self._lock:
            if name in self._metadata:
                del self._metadata[name]
                logger.debug(f"Unregistered metadata for '{name}'")

    def record_access(self, name: str) -> None:
        """Record an access to a state.

        Args:
            name: State name that was accessed
        """
        with self._lock:
            if name in self._metadata:
                self._metadata[name].last_accessed = utc_now()
                self._metadata[name].access_count += 1

    def record_transition(self, name: str, duration: float | None = None) -> None:
        """Record a transition for a state.

        Args:
            name: State name
            duration: Optional transition duration in seconds
        """
        with self._lock:
            if name in self._metadata:
                metadata = self._metadata[name]
                metadata.transition_count += 1

                if duration is not None:
                    # Update average duration using incremental formula
                    n = metadata.transition_count
                    metadata.average_duration = (metadata.average_duration * (n - 1) + duration) / n

    def record_error(self, name: str, error: str) -> None:
        """Record an error for a state.

        Args:
            name: State name
            error: Error message
        """
        with self._lock:
            if name in self._metadata:
                self._metadata[name].last_error = error
                logger.debug(f"Recorded error for '{name}': {error}")

    def get_metadata(self, name: str) -> StateMetadata | None:
        """Get metadata for a state.

        Args:
            name: State name

        Returns:
            StateMetadata or None
        """
        with self._lock:
            return self._metadata.get(name)

    def get_access_count(self, name: str) -> int:
        """Get access count for a state.

        Args:
            name: State name

        Returns:
            Access count or 0 if not found
        """
        with self._lock:
            metadata = self._metadata.get(name)
            return metadata.access_count if metadata else 0

    def get_transition_count(self, name: str) -> int:
        """Get transition count for a state.

        Args:
            name: State name

        Returns:
            Transition count or 0 if not found
        """
        with self._lock:
            metadata = self._metadata.get(name)
            return metadata.transition_count if metadata else 0

    def add_tag(self, name: str, tag: str) -> bool:
        """Add a tag to a state.

        Args:
            name: State name
            tag: Tag to add

        Returns:
            True if added successfully
        """
        with self._lock:
            if name in self._metadata:
                self._metadata[name].tags.add(tag)
                logger.debug(f"Added tag '{tag}' to '{name}'")
                return True
            return False

    def remove_tag(self, name: str, tag: str) -> bool:
        """Remove a tag from a state.

        Args:
            name: State name
            tag: Tag to remove

        Returns:
            True if removed successfully
        """
        with self._lock:
            if name in self._metadata:
                self._metadata[name].tags.discard(tag)
                logger.debug(f"Removed tag '{tag}' from '{name}'")
                return True
            return False

    def find_by_tag(self, tag: str) -> list[str]:
        """Find state names by tag.

        Args:
            tag: Tag to search for

        Returns:
            List of state names with the tag
        """
        with self._lock:
            results = []
            for name, metadata in self._metadata.items():
                if tag in metadata.tags:
                    results.append(name)
            return results

    def get_custom_data(self, name: str, key: str) -> Any:
        """Get custom data for a state.

        Args:
            name: State name
            key: Data key

        Returns:
            Custom data value or None
        """
        with self._lock:
            metadata = self._metadata.get(name)
            if metadata:
                return metadata.custom_data.get(key)
            return None

    def set_custom_data(self, name: str, key: str, value: Any) -> bool:
        """Set custom data for a state.

        Args:
            name: State name
            key: Data key
            value: Data value

        Returns:
            True if set successfully
        """
        with self._lock:
            if name in self._metadata:
                self._metadata[name].custom_data[key] = value
                return True
            return False

    def get_most_accessed(self, limit: int = 10) -> list[tuple[str, int]]:
        """Get the most accessed states.

        Args:
            limit: Maximum number of results

        Returns:
            List of (state_name, access_count) tuples
        """
        with self._lock:
            items = [(name, metadata.access_count) for name, metadata in self._metadata.items()]
            items.sort(key=lambda x: x[1], reverse=True)
            return items[:limit]

    def clear(self) -> None:
        """Clear all metadata."""
        with self._lock:
            self._metadata.clear()
            logger.debug("Metadata tracker cleared")
