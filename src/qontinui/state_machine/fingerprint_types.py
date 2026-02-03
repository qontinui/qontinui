"""Fingerprint data types for enhanced state discovery.

This module defines data structures for element fingerprints computed by the
UI Bridge. Fingerprints enable cross-page element matching, position-aware
grouping, and semantic identification.

Key concepts:
- ElementFingerprint: Browser-computed fingerprint for stable element ID
- RepeatPattern: Information about list/grid/table repeating elements
- CaptureRecord: A single page capture with visible fingerprints
- TransitionRecord: A state change triggered by an action
- CooccurrenceExport: Full export from UI Bridge capture session
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RepeatPattern:
    """Information about repeating element patterns.

    When an element is part of a list, grid, or table, this provides
    details about the container and the element's position within it.

    Attributes:
        type: Container type ("list", "grid", "table")
        container_selector: CSS selector for the container element
        item_selector: CSS selector for individual items
        index: Zero-based index of this element in the list
        total_count: Total number of items in the container
    """

    type: str  # "list", "grid", "table"
    container_selector: str
    item_selector: str
    index: int
    total_count: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepeatPattern:
        """Create from dictionary (JSON deserialization)."""
        return cls(
            type=data.get("type", "list"),
            container_selector=data.get("containerSelector", ""),
            item_selector=data.get("itemSelector", ""),
            index=data.get("index", 0),
            total_count=data.get("totalCount", 1),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (JSON serialization)."""
        return {
            "type": self.type,
            "containerSelector": self.container_selector,
            "itemSelector": self.item_selector,
            "index": self.index,
            "totalCount": self.total_count,
        }


@dataclass
class ElementFingerprint:
    """Browser-computed fingerprint for stable element identification.

    Fingerprints capture structural and semantic properties of elements
    that remain stable across page loads and navigation, enabling
    cross-page element matching.

    Attributes:
        hash: Stable hash of fingerprint properties
        structural_path: Tag-only DOM path (e.g., "nav > ul > li > a")
        position_zone: Element location (header, footer, sidebar-left,
                       sidebar-right, main, modal)
        landmark_context: Nearest ARIA landmark (navigation, main, etc.)
        landmark_label: Optional label for the landmark
        role: ARIA role or implicit role
        tag_name: HTML tag name
        accessible_name: Normalized accessible name (dynamic patterns replaced)
        size_category: Visual size (icon, button, small, medium, large,
                       fullwidth, panel)
        relative_position: Viewport-relative position {top: float, left: float}
        is_repeating: Whether element is part of a list/grid/table
        repeat_pattern: Details about the repeat container if applicable
    """

    hash: str
    structural_path: str
    position_zone: str  # header, footer, sidebar-left, sidebar-right, main, modal
    landmark_context: str
    landmark_label: str | None = None
    role: str = ""
    tag_name: str = ""
    accessible_name: str | None = None
    size_category: str = ""  # icon, button, small, medium, large, fullwidth, panel
    relative_position: dict[str, float] = field(default_factory=dict)
    is_repeating: bool = False
    repeat_pattern: RepeatPattern | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ElementFingerprint:
        """Create from dictionary (JSON deserialization)."""
        repeat_pattern = None
        if data.get("repeatPattern"):
            repeat_pattern = RepeatPattern.from_dict(data["repeatPattern"])

        return cls(
            hash=data.get("hash", ""),
            structural_path=data.get("structuralPath", ""),
            position_zone=data.get("positionZone", "main"),
            landmark_context=data.get("landmarkContext", ""),
            landmark_label=data.get("landmarkLabel"),
            role=data.get("role", ""),
            tag_name=data.get("tagName", ""),
            accessible_name=data.get("accessibleName"),
            size_category=data.get("sizeCategory", ""),
            relative_position=data.get("relativePosition", {}),
            is_repeating=data.get("isRepeating", False),
            repeat_pattern=repeat_pattern,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (JSON serialization)."""
        result: dict[str, Any] = {
            "hash": self.hash,
            "structuralPath": self.structural_path,
            "positionZone": self.position_zone,
            "landmarkContext": self.landmark_context,
            "role": self.role,
            "tagName": self.tag_name,
            "sizeCategory": self.size_category,
            "relativePosition": self.relative_position,
            "isRepeating": self.is_repeating,
        }

        if self.landmark_label:
            result["landmarkLabel"] = self.landmark_label
        if self.accessible_name:
            result["accessibleName"] = self.accessible_name
        if self.repeat_pattern:
            result["repeatPattern"] = self.repeat_pattern.to_dict()

        return result


@dataclass
class CaptureRecord:
    """Record of a single page capture.

    Represents a snapshot of visible elements at a point in time,
    with their fingerprint hashes.

    Attributes:
        capture_id: Unique identifier for this capture
        url: Page URL at time of capture
        title: Page title at time of capture
        timestamp: Unix timestamp in milliseconds
        fingerprint_hashes: List of fingerprint hashes visible in this capture
        triggered_by: What triggered this capture ("manual", "action", "navigation")
    """

    capture_id: str
    url: str
    title: str
    timestamp: int
    fingerprint_hashes: list[str]
    triggered_by: str | None = None  # "manual", "action", "navigation"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaptureRecord:
        """Create from dictionary (JSON deserialization)."""
        return cls(
            capture_id=data.get("captureId", ""),
            url=data.get("url", ""),
            title=data.get("title", ""),
            timestamp=data.get("timestamp", 0),
            fingerprint_hashes=data.get("fingerprintHashes", []),
            triggered_by=data.get("triggeredBy"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (JSON serialization)."""
        result: dict[str, Any] = {
            "captureId": self.capture_id,
            "url": self.url,
            "title": self.title,
            "timestamp": self.timestamp,
            "fingerprintHashes": self.fingerprint_hashes,
        }
        if self.triggered_by:
            result["triggeredBy"] = self.triggered_by
        return result


@dataclass
class TransitionRecord:
    """Record of a state transition triggered by an action.

    Captures the before/after state of a user action, including
    which fingerprints appeared or disappeared.

    Attributes:
        action_id: Unique identifier for the action
        action_type: Type of action (click, type, navigate, etc.)
        target_fingerprint: Fingerprint of the action target element
        before_capture_id: Capture ID before the action
        after_capture_id: Capture ID after the action
        appeared_fingerprints: Fingerprints that appeared after the action
        disappeared_fingerprints: Fingerprints that disappeared after the action
        timestamp: Unix timestamp in milliseconds
    """

    action_id: str
    action_type: str
    target_fingerprint: str | None
    before_capture_id: str
    after_capture_id: str
    appeared_fingerprints: list[str]
    disappeared_fingerprints: list[str]
    timestamp: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransitionRecord:
        """Create from dictionary (JSON deserialization)."""
        return cls(
            action_id=data.get("actionId", ""),
            action_type=data.get("actionType", ""),
            target_fingerprint=data.get("targetFingerprint"),
            before_capture_id=data.get("beforeCaptureId", ""),
            after_capture_id=data.get("afterCaptureId", ""),
            appeared_fingerprints=data.get("appearedFingerprints", []),
            disappeared_fingerprints=data.get("disappearedFingerprints", []),
            timestamp=data.get("timestamp", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (JSON serialization)."""
        return {
            "actionId": self.action_id,
            "actionType": self.action_type,
            "targetFingerprint": self.target_fingerprint,
            "beforeCaptureId": self.before_capture_id,
            "afterCaptureId": self.after_capture_id,
            "appearedFingerprints": self.appeared_fingerprints,
            "disappearedFingerprints": self.disappeared_fingerprints,
            "timestamp": self.timestamp,
        }


@dataclass
class FingerprintStats:
    """Statistics for a fingerprint across captures.

    Attributes:
        total_appearances: Number of captures containing this fingerprint
        capture_ids: List of capture IDs where this fingerprint appeared
        first_seen: Timestamp of first appearance
        last_seen: Timestamp of last appearance
    """

    total_appearances: int
    capture_ids: list[str]
    first_seen: int = 0
    last_seen: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FingerprintStats:
        """Create from dictionary (JSON deserialization)."""
        return cls(
            total_appearances=data.get("totalAppearances", 0),
            capture_ids=data.get("captureIds", []),
            first_seen=data.get("firstSeen", 0),
            last_seen=data.get("lastSeen", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (JSON serialization)."""
        return {
            "totalAppearances": self.total_appearances,
            "captureIds": self.capture_ids,
            "firstSeen": self.first_seen,
            "lastSeen": self.last_seen,
        }


@dataclass
class PresenceMatrixEntry:
    """Entry in the presence matrix (which fingerprints appeared in which capture).

    Attributes:
        capture_id: ID of the capture
        url: URL of the page
        fingerprints: List of fingerprint hashes present in this capture
    """

    capture_id: str
    url: str
    fingerprints: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PresenceMatrixEntry:
        """Create from dictionary (JSON deserialization)."""
        return cls(
            capture_id=data.get("captureId", ""),
            url=data.get("url", ""),
            fingerprints=data.get("fingerprints", []),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (JSON serialization)."""
        return {
            "captureId": self.capture_id,
            "url": self.url,
            "fingerprints": self.fingerprints,
        }


@dataclass
class StateCandidate:
    """A candidate state from pre-computed co-occurrence analysis.

    Represents a group of fingerprints that always appear together
    (100% co-occurrence rate).

    Attributes:
        fingerprints: List of fingerprint hashes in this state
        cooccurrence_rate: Rate at which these fingerprints co-occur (typically 1.0)
        position_zone: Dominant position zone for this state
        landmark_context: Dominant landmark for this state
    """

    fingerprints: list[str]
    cooccurrence_rate: float
    position_zone: str | None = None
    landmark_context: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateCandidate:
        """Create from dictionary (JSON deserialization)."""
        return cls(
            fingerprints=data.get("fingerprints", []),
            cooccurrence_rate=data.get("cooccurrenceRate", 1.0),
            position_zone=data.get("positionZone"),
            landmark_context=data.get("landmarkContext"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (JSON serialization)."""
        result: dict[str, Any] = {
            "fingerprints": self.fingerprints,
            "cooccurrenceRate": self.cooccurrence_rate,
        }
        if self.position_zone:
            result["positionZone"] = self.position_zone
        if self.landmark_context:
            result["landmarkContext"] = self.landmark_context
        return result


@dataclass
class CooccurrenceExport:
    """Full export from UI Bridge capture session.

    Contains all fingerprints observed, their co-occurrence data,
    pre-computed state candidates, and recorded transitions.

    Attributes:
        session_id: Unique identifier for the capture session
        exported_at: Unix timestamp of export
        all_fingerprints: List of all fingerprint hashes observed
        fingerprint_details: Map of hash to ElementFingerprint
        presence_matrix: Which fingerprints appeared in which captures
        cooccurrence_counts: Pairwise co-occurrence counts between fingerprints
        fingerprint_stats: Statistics for each fingerprint
        transitions: Recorded state transitions
        state_candidates: Pre-computed state candidates (100% co-occurrence groups)
    """

    session_id: str
    exported_at: int
    all_fingerprints: list[str]
    fingerprint_details: dict[str, ElementFingerprint]
    presence_matrix: list[PresenceMatrixEntry]
    cooccurrence_counts: dict[str, dict[str, int]]
    fingerprint_stats: dict[str, FingerprintStats]
    transitions: list[TransitionRecord]
    state_candidates: list[StateCandidate]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CooccurrenceExport:
        """Create from dictionary (JSON deserialization).

        This is the main entry point for consuming UI Bridge exports.
        """
        # Parse fingerprint details
        fingerprint_details: dict[str, ElementFingerprint] = {}
        for fp_hash, fp_data in data.get("fingerprintDetails", {}).items():
            fingerprint_details[fp_hash] = ElementFingerprint.from_dict(fp_data)

        # Parse presence matrix
        presence_matrix = [
            PresenceMatrixEntry.from_dict(entry) for entry in data.get("presenceMatrix", [])
        ]

        # Parse fingerprint stats
        fingerprint_stats: dict[str, FingerprintStats] = {}
        for fp_hash, stats_data in data.get("fingerprintStats", {}).items():
            fingerprint_stats[fp_hash] = FingerprintStats.from_dict(stats_data)

        # Parse transitions
        transitions = [TransitionRecord.from_dict(t) for t in data.get("transitions", [])]

        # Parse state candidates
        state_candidates = [StateCandidate.from_dict(sc) for sc in data.get("stateCandidates", [])]

        return cls(
            session_id=data.get("sessionId", ""),
            exported_at=data.get("exportedAt", 0),
            all_fingerprints=data.get("allFingerprints", []),
            fingerprint_details=fingerprint_details,
            presence_matrix=presence_matrix,
            cooccurrence_counts=data.get("cooccurrenceCounts", {}),
            fingerprint_stats=fingerprint_stats,
            transitions=transitions,
            state_candidates=state_candidates,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (JSON serialization)."""
        return {
            "sessionId": self.session_id,
            "exportedAt": self.exported_at,
            "allFingerprints": self.all_fingerprints,
            "fingerprintDetails": {k: v.to_dict() for k, v in self.fingerprint_details.items()},
            "presenceMatrix": [entry.to_dict() for entry in self.presence_matrix],
            "cooccurrenceCounts": self.cooccurrence_counts,
            "fingerprintStats": {k: v.to_dict() for k, v in self.fingerprint_stats.items()},
            "transitions": [t.to_dict() for t in self.transitions],
            "stateCandidates": [sc.to_dict() for sc in self.state_candidates],
        }


# Position zone constants for validation and grouping
POSITION_ZONES = frozenset(["header", "footer", "sidebar-left", "sidebar-right", "main", "modal"])

# Global position zones (elements that appear across all states)
GLOBAL_POSITION_ZONES = frozenset(["header", "footer"])

# Blocking position zones (elements that block interaction with other states)
BLOCKING_POSITION_ZONES = frozenset(["modal"])

# Size categories for weighting
SIZE_CATEGORIES = frozenset(["icon", "button", "small", "medium", "large", "fullwidth", "panel"])

# ARIA landmarks
ARIA_LANDMARKS = frozenset(
    [
        "banner",
        "complementary",
        "contentinfo",
        "form",
        "main",
        "navigation",
        "region",
        "search",
    ]
)
