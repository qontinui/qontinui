"""Screenshot-state association for integration testing visualization.

Associates screenshots with sets of active states to enable exact screenshot
matching during mock execution visualization.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class StateScreenshot:
    """Screenshot associated with a specific set of active states.

    During integration testing visualization, we need to display the exact
    screenshot that corresponds to the current set of active states. This
    class associates a screenshot file with its active state context.

    Attributes:
        screenshot_path: Path to screenshot file (relative to snapshot directory)
        active_states: Set of active state names when screenshot was captured
        timestamp: When screenshot was captured
        width: Screenshot width in pixels
        height: Screenshot height in pixels
        state_hash: Hash of sorted active states for quick lookup
        metadata: Additional screenshot metadata
    """

    screenshot_path: str
    active_states: set[str]
    timestamp: datetime
    width: int
    height: int
    state_hash: str = field(init=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate state hash after initialization."""
        self.state_hash = self._calculate_state_hash(self.active_states)

    @staticmethod
    def _calculate_state_hash(active_states: set[str]) -> str:
        """Calculate hash of active states for quick lookup.

        Args:
            active_states: Set of active state names

        Returns:
            Hash string
        """
        # Sort states for consistent hashing
        sorted_states = sorted(active_states)
        state_str = "|".join(sorted_states)
        return hashlib.md5(state_str.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "screenshot_path": self.screenshot_path,
            "active_states": sorted(self.active_states),
            "timestamp": self.timestamp.isoformat(),
            "width": self.width,
            "height": self.height,
            "state_hash": self.state_hash,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "StateScreenshot":
        """Reconstruct from dictionary."""
        return StateScreenshot(
            screenshot_path=data["screenshot_path"],
            active_states=set(data["active_states"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            width=data["width"],
            height=data["height"],
            metadata=data.get("metadata", {}),
        )


class StateScreenshotRegistry:
    """Registry of screenshots associated with active states.

    Maintains a mapping of active state combinations to screenshots,
    enabling quick lookup of the appropriate screenshot for visualization
    during integration testing.

    Matching Strategy:
    1. Exact match: Find screenshot with exactly matching states
    2. Subset match: Find screenshot with states as subset of current
    3. Overlap match: Find screenshot with highest state overlap
    4. Fallback: Most recent screenshot
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self.screenshots: list[StateScreenshot] = []
        self._hash_index: dict[str, list[StateScreenshot]] = {}

    def register_screenshot(
        self,
        screenshot_path: str,
        active_states: set[str],
        timestamp: datetime,
        width: int,
        height: int,
        metadata: dict[str, Any] | None = None,
    ) -> StateScreenshot:
        """Register a new screenshot with its active states.

        Args:
            screenshot_path: Path to screenshot file
            active_states: Set of active state names
            timestamp: When screenshot was captured
            width: Screenshot width
            height: Screenshot height
            metadata: Optional metadata

        Returns:
            Created StateScreenshot instance
        """
        state_screenshot = StateScreenshot(
            screenshot_path=screenshot_path,
            active_states=active_states,
            timestamp=timestamp,
            width=width,
            height=height,
            metadata=metadata or {},
        )

        self.screenshots.append(state_screenshot)

        # Add to hash index for fast exact lookup
        if state_screenshot.state_hash not in self._hash_index:
            self._hash_index[state_screenshot.state_hash] = []
        self._hash_index[state_screenshot.state_hash].append(state_screenshot)

        return state_screenshot

    def find_screenshot(self, active_states: set[str]) -> StateScreenshot | None:
        """Find best matching screenshot for active states.

        Uses 4-tier matching strategy:
        1. Exact match: States exactly match
        2. Subset match: Screenshot states are subset of current
        3. Overlap match: Maximum state overlap
        4. Fallback: Most recent screenshot

        Args:
            active_states: Current active states

        Returns:
            Best matching StateScreenshot or None
        """
        if not self.screenshots:
            return None

        # Strategy 1: Exact match using hash index
        state_hash = StateScreenshot._calculate_state_hash(active_states)
        if state_hash in self._hash_index:
            # Return most recent exact match
            exact_matches = self._hash_index[state_hash]
            return max(exact_matches, key=lambda s: s.timestamp)

        # Strategy 2: Subset match (screenshot states ⊆ current states)
        subset_matches = [
            s for s in self.screenshots if s.active_states.issubset(active_states)
        ]
        if subset_matches:
            # Prefer screenshot with most states, then most recent
            return max(
                subset_matches, key=lambda s: (len(s.active_states), s.timestamp)
            )

        # Strategy 3: Overlap match (maximize intersection)
        overlap_matches = [
            (s, len(s.active_states.intersection(active_states)))
            for s in self.screenshots
            if s.active_states.intersection(active_states)
        ]
        if overlap_matches:
            # Choose screenshot with highest overlap, then most recent
            screenshot, _ = max(overlap_matches, key=lambda x: (x[1], x[0].timestamp))
            return screenshot

        # Strategy 4: Fallback to most recent
        return max(self.screenshots, key=lambda s: s.timestamp)

    def find_all_for_states(self, active_states: set[str]) -> list[StateScreenshot]:
        """Find all screenshots matching or overlapping with active states.

        Args:
            active_states: Active states to match

        Returns:
            List of matching screenshots, sorted by relevance
        """
        matches = []

        for screenshot in self.screenshots:
            if screenshot.active_states == active_states:
                # Exact match - highest priority
                matches.append((screenshot, 3, len(screenshot.active_states)))
            elif screenshot.active_states.issubset(active_states):
                # Subset match
                matches.append((screenshot, 2, len(screenshot.active_states)))
            elif screenshot.active_states.intersection(active_states):
                # Overlap match
                overlap = len(screenshot.active_states.intersection(active_states))
                matches.append((screenshot, 1, overlap))

        # Sort by priority, then state count, then timestamp
        matches.sort(key=lambda x: (x[1], x[2], x[0].timestamp), reverse=True)

        return [s for s, _, _ in matches]

    def get_unique_state_combinations(self) -> list[set[str]]:
        """Get all unique state combinations that have screenshots.

        Returns:
            List of unique active state sets
        """
        unique_states = {}
        for screenshot in self.screenshots:
            state_hash = screenshot.state_hash
            if state_hash not in unique_states:
                unique_states[state_hash] = screenshot.active_states
        return list(unique_states.values())

    def to_dict(self) -> dict[str, Any]:
        """Convert registry to dictionary for JSON serialization."""
        return {
            "screenshots": [s.to_dict() for s in self.screenshots],
            "count": len(self.screenshots),
            "unique_states": len(self._hash_index),
        }

    def save_to_file(self, filepath: str):
        """Save registry to JSON file.

        Args:
            filepath: Path to save file
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load_from_file(filepath: str) -> "StateScreenshotRegistry":
        """Load registry from JSON file.

        Args:
            filepath: Path to load from

        Returns:
            Loaded StateScreenshotRegistry
        """
        with open(filepath) as f:
            data = json.load(f)

        registry = StateScreenshotRegistry()
        for screenshot_data in data.get("screenshots", []):
            screenshot = StateScreenshot.from_dict(screenshot_data)
            registry.screenshots.append(screenshot)

            # Rebuild hash index
            if screenshot.state_hash not in registry._hash_index:
                registry._hash_index[screenshot.state_hash] = []
            registry._hash_index[screenshot.state_hash].append(screenshot)

        return registry


@dataclass
class ActionVisualization:
    """Visualization data for a single action in integration testing.

    Contains all information needed to visualize an action during mock
    execution playback, including screenshot, action location, and results.

    Attributes:
        action_type: Type of action ("FIND", "CLICK", "TYPE", etc.)
        screenshot_path: Path to screenshot showing action context
        action_location: (x, y) coordinates of action (for click, type, etc.)
        action_region: Bounding box of action target (for find actions)
        success: Whether action succeeded
        matches: List of match regions for find actions
        text: Text entered (for type actions) or extracted (OCR)
        active_states: Active states during action
        timestamp: When action occurred
        duration_ms: Action duration in milliseconds
    """

    action_type: str
    screenshot_path: str
    action_location: tuple[int, int] | None = None
    action_region: dict[str, int] | None = None  # {x, y, w, h}
    success: bool = True
    matches: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""
    active_states: set[str] = field(default_factory=set)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "action_type": self.action_type,
            "screenshot_path": self.screenshot_path,
            "action_location": self.action_location,
            "action_region": self.action_region,
            "success": self.success,
            "matches": self.matches,
            "text": self.text,
            "active_states": sorted(self.active_states),
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ActionVisualization":
        """Reconstruct from dictionary."""
        return ActionVisualization(
            action_type=data["action_type"],
            screenshot_path=data["screenshot_path"],
            action_location=(
                tuple(data["action_location"]) if data.get("action_location") else None
            ),
            action_region=data.get("action_region"),
            success=data.get("success", True),
            matches=data.get("matches", []),
            text=data.get("text", ""),
            active_states=set(data.get("active_states", [])),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            duration_ms=data.get("duration_ms", 0.0),
        )
