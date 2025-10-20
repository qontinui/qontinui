"""Mock snapshot data structures for historical playback.

Provides data structures for recording and playing back automation actions
in mock mode, following the Brobot ActionRecord pattern.

Key Insight: Brobot uses the SAME Match class for both mock and real automation.
There is NO separate "MatchSnapshot" class. ActionRecord stores actual Match objects.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..model.match.match import Match

logger = logging.getLogger(__name__)


@dataclass
class ActionRecord:
    """Historical record of an action execution (Brobot ActionRecord pattern).

    Records all details of an action's execution including the action type,
    success/failure, all matches found, duration, and active states. This
    provides complete historical data for mock playback.

    IMPORTANT: Stores actual Match objects, NOT a separate MatchSnapshot type.
    This follows Brobot's design where the same Match class is used for both
    mock and real automation.

    Action Snapshot Structure (AS):
        AS = (o_a^h, S_Ξ^h, r_a^h) where:
        - o_a^h: action parameters (stored in action_type and metadata)
        - S_Ξ^h: active states during execution (stored in active_states)
        - r_a^h: action results (stored in action_success, match_list, text)

    Attributes:
        action_type: Type of action ("FIND", "CLICK", "TYPE", etc.) - part of o_a^h
        action_success: Whether the action succeeded - part of r_a^h
        match_list: List of Match objects found - part of r_a^h
        text: Extracted text from OCR (for text actions) - part of r_a^h
        duration: Execution duration in seconds
        timestamp: When this action was executed
        active_states: Set of active state names during execution (S_Ξ^h)
        metadata: Additional action-specific data - part of o_a^h
    """

    action_type: str
    action_success: bool
    match_list: list["Match"] = field(default_factory=list)
    text: str = ""
    duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    active_states: set[str] = field(default_factory=set)  # S_Ξ^h - Set of active states
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation

        Note: Match objects are serialized to their dict representation.
        On deserialization, they'll need to be reconstructed as Match objects.
        """
        return {
            "action_type": self.action_type,
            "action_success": self.action_success,
            "match_list": [self._match_to_dict(m) for m in self.match_list],
            "text": self.text,
            "duration": self.duration,
            "timestamp": self.timestamp.isoformat(),
            "active_states": sorted(self.active_states),  # Convert set to sorted list for JSON
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize to JSON string.

        Returns:
            JSON string representation

        Example:
            record = ActionRecord(...)
            json_str = record.to_json()
            # Save to file or database
        """
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def _match_to_dict(match: "Match") -> dict[str, Any]:
        """Convert Match object to dictionary for JSON serialization.

        Args:
            match: Match object to serialize

        Returns:
            Dictionary with Match data
        """

        # Extract region from match target location
        region = match.target.region if match.target and match.target.region else None

        return {
            "score": match.score,
            "text": getattr(match, "text", ""),
            "name": getattr(match, "name", ""),
            "region": (
                {
                    "x": region.x if region else 0,
                    "y": region.y if region else 0,
                    "w": region.w if region else 0,
                    "h": region.h if region else 0,
                }
                if region
                else None
            ),
            "timestamp": (
                match.timestamp.isoformat()
                if hasattr(match, "timestamp") and match.timestamp
                else datetime.now().isoformat()
            ),
        }

    @staticmethod
    def _dict_to_match(data: dict[str, Any]) -> "Match":
        """Reconstruct Match object from dictionary.

        Args:
            data: Dictionary from JSON deserialization

        Returns:
            Reconstructed Match object
        """
        from ..model.element.location import Location
        from ..model.element.region import Region
        from ..model.match.match import Match

        # Reconstruct region
        region = None
        if data.get("region"):
            r = data["region"]
            region = Region(r["x"], r["y"], r["w"], r["h"])

        # Reconstruct location
        location = Location(region=region) if region else Location()

        # Create Match object
        match = Match(
            target=location,
            score=data.get("score", 0.0),
        )

        # Set optional fields
        if "text" in data:
            match.text = data["text"]
        if "name" in data:
            match.name = data["name"]
        if "timestamp" in data:
            match.timestamp = datetime.fromisoformat(data["timestamp"])

        return match

    @staticmethod
    def from_json(json_str: str) -> "ActionRecord":
        """Deserialize from JSON string.

        Args:
            json_str: JSON string to parse

        Returns:
            ActionRecord instance

        Example:
            json_str = load_from_file("record.json")
            record = ActionRecord.from_json(json_str)
        """
        data = json.loads(json_str)
        return ActionRecord.from_dict(data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ActionRecord":
        """Reconstruct ActionRecord from dictionary.

        Args:
            data: Dictionary from JSON

        Returns:
            ActionRecord instance
        """
        return ActionRecord(
            action_type=data["action_type"],
            action_success=data["action_success"],
            match_list=[ActionRecord._dict_to_match(m) for m in data.get("match_list", [])],
            text=data.get("text", ""),
            duration=data.get("duration", 0.0),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            active_states=set(data.get("active_states", [])),
            metadata=data.get("metadata", {}),
        )


class ActionHistory:
    """Collection of historical action records for a pattern (Brobot ActionHistory pattern).

    Stores multiple ActionRecords for a pattern, allowing retrieval
    of appropriate historical data based on current state context.
    Follows the Brobot pattern of pattern.getMatchHistory().

    IMPORTANT: Returns List[Match] directly (extracted from ActionRecord.match_list),
    NOT ActionRecord objects. This matches Brobot's MockFind behavior.

    Example:
        # Create history
        history = ActionHistory()

        # Add records from real executions
        record = ActionRecord(
            action_type="FIND",
            action_success=True,
            match_list=[Match(...), Match(...)],
            duration=0.25,
            timestamp=datetime.now(),
            state_name="login_screen"
        )
        history.add_record(record)

        # Later, in mock mode, retrieve matches for current state
        state_name = "login_screen"
        matches = history.get_matches_for_state(state_name)
        # Returns List[Match] directly (not ActionRecord)
    """

    def __init__(self):
        """Initialize empty action history."""
        self.snapshots: list[ActionRecord] = []
        self.times_searched: int = 0
        self.times_found: int = 0
        logger.debug("ActionHistory initialized")

    def add_record(self, record: ActionRecord):
        """Add a new action record to history.

        Args:
            record: ActionRecord to add

        Example:
            history = ActionHistory()
            record = ActionRecord(
                action_type="FIND",
                action_success=True,
                match_list=[Match(...)],
                duration=0.25,
                state_name="login_screen"
            )
            history.add_record(record)
        """
        self.snapshots.append(record)
        self.times_searched += 1
        if record.action_success:
            self.times_found += 1
        logger.debug(f"Added record: {record.action_type} at {record.timestamp}")

    def get_matches_for_states(self, active_states: set[str]) -> list["Match"]:
        """Get matches for current active states using state overlap matching.

        Implements the matching function from section 11.1 that compares:
        - Current active states (S_Ξ) with
        - Historical active states (S_Ξ^h) from each snapshot

        Matching strategy:
        1. Exact match: S_Ξ == S_Ξ^h (highest priority)
        2. Subset match: S_Ξ^h ⊆ S_Ξ (snapshot states are subset of current)
        3. Overlap match: S_Ξ ∩ S_Ξ^h ≠ ∅ (at least one common state)
        4. Fallback: Any successful record

        Args:
            active_states: Current set of active state names (S_Ξ)

        Returns:
            List of Match objects from best matching record

        Example:
            current_states = {"login_screen", "english_language"}
            matches = history.get_matches_for_states(current_states)
        """
        if not self.snapshots:
            logger.warning("No records in history")
            return []

        successful = [r for r in self.snapshots if r.action_success]
        if not successful:
            logger.warning("No successful records in history")
            return []

        # Strategy 1: Exact state match (S_Ξ == S_Ξ^h)
        exact_matches = [r for r in successful if r.active_states == active_states]
        if exact_matches:
            record = max(exact_matches, key=lambda r: r.timestamp)
            logger.debug(
                f"Exact state match: {record.active_states} -> {len(record.match_list)} matches"
            )
            return record.match_list

        # Strategy 2: Subset match (S_Ξ^h ⊆ S_Ξ) - snapshot states are subset of current
        subset_matches = [r for r in successful if r.active_states.issubset(active_states)]
        if subset_matches:
            # Choose snapshot with most overlapping states
            record = max(subset_matches, key=lambda r: (len(r.active_states), r.timestamp))
            logger.debug(
                f"Subset match: {record.active_states} ⊆ {active_states} -> {len(record.match_list)} matches"
            )
            return record.match_list

        # Strategy 3: Overlap match (S_Ξ ∩ S_Ξ^h ≠ ∅) - at least one common state
        overlap_matches = [
            (r, len(r.active_states.intersection(active_states)))
            for r in successful
            if r.active_states.intersection(active_states)
        ]
        if overlap_matches:
            # Choose snapshot with highest overlap, then most recent
            record, overlap_count = max(overlap_matches, key=lambda x: (x[1], x[0].timestamp))
            logger.debug(
                f"Overlap match: {record.active_states} ∩ {active_states} ({overlap_count} common) -> {len(record.match_list)} matches"
            )
            return record.match_list

        # Strategy 4: Fallback - any successful record with matches
        with_matches = [r for r in successful if r.match_list]
        if with_matches:
            record = max(with_matches, key=lambda r: r.timestamp)
            logger.debug(
                f"Fallback match (no state overlap): {record.active_states} -> {len(record.match_list)} matches"
            )
            return record.match_list

        logger.warning("No successful records with matches in history")
        return []

    def get_random_record(self, active_states: set[str] | None = None) -> ActionRecord | None:
        """Get a random record matching the active states.

        Args:
            active_states: Set of active states to match (None = any record)

        Returns:
            Random ActionRecord or None
        """
        import random

        if not self.snapshots:
            return None

        if active_states:
            # Find records with overlapping states
            candidates = [r for r in self.snapshots if r.active_states.intersection(active_states)]
        else:
            candidates = self.snapshots

        return random.choice(candidates) if candidates else None

    def get_all_records(self) -> list[ActionRecord]:
        """Get all records in chronological order.

        Returns:
            List of all ActionRecords, oldest first
        """
        return sorted(self.snapshots, key=lambda r: r.timestamp)

    def is_empty(self) -> bool:
        """Check if history is empty.

        Returns:
            True if no records exist
        """
        return len(self.snapshots) == 0

    def clear(self):
        """Clear all records from history."""
        count = len(self.snapshots)
        self.snapshots = []
        self.times_searched = 0
        self.times_found = 0
        logger.debug(f"Cleared {count} records from history")

    def save_to_file(self, filepath: str):
        """Save records to JSON file.

        Args:
            filepath: Path to save file

        Example:
            history = ActionHistory()
            # ... add records ...
            history.save_to_file("pattern_history.json")
        """
        data = {
            "snapshots": [r.to_dict() for r in self.snapshots],
            "times_searched": self.times_searched,
            "times_found": self.times_found,
            "count": len(self.snapshots),
            "created": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.snapshots)} records to {filepath}")

    @staticmethod
    def load_from_file(filepath: str) -> "ActionHistory":
        """Load records from JSON file.

        Args:
            filepath: Path to load from

        Returns:
            ActionHistory instance with loaded records

        Example:
            history = ActionHistory.load_from_file("pattern_history.json")
            print(f"Loaded {len(history.snapshots)} records")
        """
        with open(filepath) as f:
            data = json.load(f)

        history = ActionHistory()
        for record_dict in data.get("snapshots", []):
            record = ActionRecord.from_dict(record_dict)
            history.snapshots.append(record)

        history.times_searched = data.get("times_searched", 0)
        history.times_found = data.get("times_found", 0)

        logger.info(f"Loaded {len(history.snapshots)} records from {filepath}")
        return history
