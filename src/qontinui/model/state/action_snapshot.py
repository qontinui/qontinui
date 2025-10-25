"""ActionSnapshot - Record of an action taken in a specific state.

Part of Qontinui's integration testing framework.
Stores detailed information about actions performed, their results,
and state transitions for replay and validation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..element.region import Region


class ActionType(Enum):
    """Types of actions that can be recorded."""

    FIND = "FIND"
    CLICK = "CLICK"
    TYPE = "TYPE"
    DRAG = "DRAG"
    SCROLL = "SCROLL"
    WAIT = "WAIT"
    KEY = "KEY"
    HOVER = "HOVER"


@dataclass
class MatchResult:
    """Result of a pattern match operation."""

    region: Region
    score: float
    state_image_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "region": {
                "x": self.region.x,
                "y": self.region.y,
                "width": self.region.width,
                "height": self.region.height,
            },
            "score": self.score,
            "state_image_id": self.state_image_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchResult":
        """Create from dictionary."""
        region_data = data["region"]
        region = Region(
            region_data["x"], region_data["y"], region_data["width"], region_data["height"]
        )
        return cls(region=region, score=data["score"], state_image_id=data.get("state_image_id"))


@dataclass
class ActionSnapshot:
    """Snapshot of an action execution in a specific state.

    Records all details about an action including:
    - The action type and configuration
    - The state context when executed
    - Match results from pattern matching
    - Success/failure status
    - State transitions (screenshot changes)
    - Timing information

    This is the core data structure for Qontinui's integration testing,
    allowing actions to be recorded and replayed deterministically.
    """

    # Identity
    id: str
    timestamp: datetime

    # Action details
    action_type: ActionType
    action_config: dict[str, Any]  # Configuration used for the action

    # Match results
    matches: list[MatchResult] = field(default_factory=list)

    # State context
    state_name: str = ""
    state_id: str = ""
    active_states: list[str] = field(default_factory=list)  # All states active at this moment

    # Success indicators
    action_success: bool = False  # Did the action execute successfully?
    result_success: bool = False  # Did it achieve the desired result?

    # Screenshot management (for web testing)
    screenshot_id: str = ""  # Current screenshot when action was taken
    next_screenshot_id: str | None = None  # Screenshot to transition to after action

    # Timing
    duration: int = 0  # Duration in milliseconds

    # Text results (for TYPE actions)
    text: str | None = None

    # Additional metadata
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "action_type": self.action_type.value,
            "action_config": self.action_config,
            "matches": [m.to_dict() for m in self.matches],
            "state_name": self.state_name,
            "state_id": self.state_id,
            "active_states": self.active_states,
            "action_success": self.action_success,
            "result_success": self.result_success,
            "screenshot_id": self.screenshot_id,
            "next_screenshot_id": self.next_screenshot_id,
            "duration": self.duration,
            "text": self.text,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionSnapshot":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            action_type=ActionType(data["action_type"]),
            action_config=data["action_config"],
            matches=[MatchResult.from_dict(m) for m in data.get("matches", [])],
            state_name=data.get("state_name", ""),
            state_id=data.get("state_id", ""),
            active_states=data.get("active_states", []),
            action_success=data.get("action_success", False),
            result_success=data.get("result_success", False),
            screenshot_id=data.get("screenshot_id", ""),
            next_screenshot_id=data.get("next_screenshot_id"),
            duration=data.get("duration", 0),
            text=data.get("text"),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )

    def is_successful(self) -> bool:
        """Check if the action was successful."""
        return self.action_success and self.result_success

    def has_matches(self) -> bool:
        """Check if the action found any matches."""
        return len(self.matches) > 0

    def get_best_match(self) -> MatchResult | None:
        """Get the match with the highest score."""
        if not self.matches:
            return None
        return max(self.matches, key=lambda m: m.score)

    def get_transition_screenshot(self) -> str | None:
        """Get the screenshot ID to transition to."""
        return self.next_screenshot_id

    def matches_state(self, state_id: str) -> bool:
        """Check if this snapshot was taken in the given state."""
        return self.state_id == state_id or state_id in self.active_states

    def matches_action_type(self, action_type: ActionType) -> bool:
        """Check if this snapshot matches the given action type."""
        return self.action_type == action_type

    def __str__(self) -> str:
        """String representation."""
        status = "✓" if self.is_successful() else "✗"
        return (
            f"ActionSnapshot[{status}]({self.action_type.value} in {self.state_name}, "
            f"{len(self.matches)} matches, {self.duration}ms)"
        )
