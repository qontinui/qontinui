"""Base classes and types for unified state discovery.

This module defines the strategy interface and common types used by all
state discovery implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class DiscoveryStrategyType(Enum):
    """Available state discovery strategy types."""

    LEGACY = "legacy"  # ID-based co-occurrence (original)
    FINGERPRINT = "fingerprint"  # Enhanced with element fingerprints
    AUTO = "auto"  # Auto-detect based on available data


@dataclass
class DiscoveredElement:
    """An element discovered during state analysis.

    This is the unified element type used across all discovery strategies.
    """

    id: str  # Unique element identifier
    name: str  # Human-readable name
    element_type: str  # Type of ID: 'ui-id', 'testid', 'html-id', 'fingerprint'
    render_ids: list[str] = field(default_factory=list)  # Renders where element appears

    # Optional fingerprint data (populated when using fingerprint strategy)
    fingerprint_hash: str | None = None
    position_zone: str | None = None  # header, footer, sidebar, main, modal
    landmark_context: str | None = None
    role: str | None = None
    accessible_name: str | None = None
    size_category: str | None = None
    is_repeating: bool = False

    # Optional metadata
    tag_name: str | None = None
    text_content: str | None = None
    component_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "type": self.element_type,
            "renderIds": self.render_ids,
        }

        # Include optional fingerprint data if present
        if self.fingerprint_hash:
            result["fingerprintHash"] = self.fingerprint_hash
        if self.position_zone:
            result["positionZone"] = self.position_zone
        if self.landmark_context:
            result["landmarkContext"] = self.landmark_context
        if self.role:
            result["role"] = self.role
        if self.accessible_name:
            result["accessibleName"] = self.accessible_name
        if self.size_category:
            result["sizeCategory"] = self.size_category
        if self.is_repeating:
            result["isRepeating"] = self.is_repeating

        # Include optional metadata
        if self.tag_name:
            result["tagName"] = self.tag_name
        if self.text_content:
            result["textContent"] = self.text_content
        if self.component_name:
            result["componentName"] = self.component_name

        return result


@dataclass
class DiscoveredState:
    """A discovered application state.

    This is the unified state type used across all discovery strategies.
    """

    id: str
    name: str
    element_ids: list[str]  # Element IDs that comprise this state
    render_ids: list[str]  # Renders where this state is active
    confidence: float = 0.0

    # Position and scope (from fingerprint strategy)
    position_zone: str | None = None  # Dominant position zone
    landmark_context: str | None = None  # Dominant landmark
    is_global: bool = False  # True if header/footer state
    is_modal: bool = False  # True if modal/blocking state

    # Metadata
    observation_count: int = 1
    first_observed: datetime = field(default_factory=datetime.utcnow)
    last_observed: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "stateImageIds": self.element_ids,  # Match existing API format
            "screenshotIds": self.render_ids,  # Match existing API format
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

        if self.position_zone:
            result["positionZone"] = self.position_zone
        if self.landmark_context:
            result["landmarkContext"] = self.landmark_context
        if self.is_global:
            result["isGlobal"] = self.is_global
        if self.is_modal:
            result["isModal"] = self.is_modal

        return result


@dataclass
class DiscoveredTransition:
    """A discovered transition between states."""

    id: str
    name: str
    action_type: str  # click, type, navigate, etc.
    from_state_ids: list[str]
    to_state_ids: list[str]
    trigger_element_id: str | None = None
    confidence: float = 0.0
    timestamp: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "actionType": self.action_type,
            "fromStateIds": self.from_state_ids,
            "toStateIds": self.to_state_ids,
            "triggerElementId": self.trigger_element_id,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


@dataclass
class StateDiscoveryResult:
    """Result of state discovery.

    This is the unified result type returned by all discovery strategies.
    Compatible with the existing UIBridgeStateDiscoveryResult for backward compatibility.
    """

    states: list[DiscoveredState]
    elements: list[DiscoveredElement]
    transitions: list[DiscoveredTransition] = field(default_factory=list)
    element_to_renders: dict[str, list[str]] = field(default_factory=dict)
    render_count: int = 0
    unique_element_count: int = 0

    # Strategy metadata
    strategy_used: DiscoveryStrategyType = DiscoveryStrategyType.LEGACY
    strategy_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Maintains backward compatibility with UIBridgeStateDiscoveryResult format.
        """
        return {
            "states": [s.to_dict() for s in self.states],
            "elements": [e.to_dict() for e in self.elements],
            "transitions": [t.to_dict() for t in self.transitions],
            "elementToRenders": self.element_to_renders,
            "renderCount": self.render_count,
            "uniqueElementCount": self.unique_element_count,
            "strategyUsed": self.strategy_used.value,
            "strategyMetadata": self.strategy_metadata,
        }


@dataclass
class StateDiscoveryInput:
    """Input data for state discovery.

    Supports multiple input formats:
    - renders: Raw render log entries (legacy format)
    - cooccurrence_export: Pre-computed co-occurrence data with fingerprints
    """

    # Legacy input: raw render logs
    renders: list[dict[str, Any]] = field(default_factory=list)
    include_html_ids: bool = False

    # Fingerprint input: pre-computed co-occurrence export
    cooccurrence_export: dict[str, Any] | None = None

    def has_fingerprint_data(self) -> bool:
        """Check if fingerprint data is available."""
        if not self.cooccurrence_export:
            return False

        # Check for fingerprint-specific fields
        return bool(
            self.cooccurrence_export.get("fingerprintDetails")
            or self.cooccurrence_export.get("stateCandidates")
        )

    def has_render_data(self) -> bool:
        """Check if render data is available."""
        return bool(self.renders)


class StateDiscoveryStrategy(ABC):
    """Abstract base class for state discovery strategies.

    All discovery implementations must inherit from this class and
    implement the discover() method.
    """

    @property
    @abstractmethod
    def strategy_type(self) -> DiscoveryStrategyType:
        """Return the strategy type identifier."""
        ...

    @abstractmethod
    def can_process(self, input_data: StateDiscoveryInput) -> bool:
        """Check if this strategy can process the given input.

        Args:
            input_data: The input data to check

        Returns:
            True if this strategy can process the input
        """
        ...

    @abstractmethod
    def discover(self, input_data: StateDiscoveryInput) -> StateDiscoveryResult:
        """Discover states from the input data.

        Args:
            input_data: Input data for discovery

        Returns:
            Discovery result containing states, elements, and transitions
        """
        ...

    def get_statistics(self) -> dict[str, Any]:
        """Get strategy-specific statistics.

        Override in subclasses to provide additional statistics.
        """
        return {}
