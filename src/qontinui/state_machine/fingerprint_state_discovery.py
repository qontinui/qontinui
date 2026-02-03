"""Fingerprint-enhanced state discovery.

This module provides state discovery using element fingerprints from the UI Bridge,
enabling cross-page element matching, position-aware grouping, and smarter
state identification.

Key enhancements over basic state discovery:
- Cross-page element matching via structural fingerprints
- Position zone classification (global vs state-specific elements)
- Repeat pattern deduplication (list items treated as one)
- Size-weighted co-occurrence scoring
- Semantic name matching

Example:
    from qontinui.state_machine import FingerprintStateDiscovery

    discovery = FingerprintStateDiscovery()

    # Load co-occurrence export from UI Bridge
    discovery.load_cooccurrence_export(cooccurrence_data)

    # Get discovered states
    states = discovery.get_discovered_states()

    # Or process incrementally
    discovery.process_capture(capture_record, fingerprints)
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .fingerprint_types import (
    BLOCKING_POSITION_ZONES,
    GLOBAL_POSITION_ZONES,
    CaptureRecord,
    CooccurrenceExport,
    ElementFingerprint,
    StateCandidate,
    TransitionRecord,
)

logger = logging.getLogger(__name__)


# Default size weights for element importance
DEFAULT_SIZE_WEIGHTS: dict[str, float] = {
    "icon": 0.1,
    "button": 0.3,
    "small": 0.5,
    "medium": 0.7,
    "large": 0.9,
    "fullwidth": 1.0,
    "panel": 1.0,
}


@dataclass
class FingerprintStateDiscoveryConfig:
    """Configuration for fingerprint-enhanced state discovery.

    Attributes:
        min_cooccurrence_rate: Minimum co-occurrence rate for grouping (0.0-1.0)
        use_fingerprint_matching: Enable cross-page fingerprint matching
        fingerprint_match_threshold: Threshold for fuzzy fingerprint matching
        treat_header_footer_as_global: Treat header/footer elements as global
        auto_detect_modal_states: Automatically detect modal states from position
        dedupe_repeating_elements: Deduplicate repeating elements (lists, grids)
        max_repeat_representatives: Maximum representatives to keep per repeat pattern
        use_size_weighting: Weight elements by visual size
        size_weights: Size category weights for importance scoring
        enable_semantic_matching: Enable matching by accessible name + role
        semantic_match_roles: Roles to consider for semantic matching
        min_state_elements: Minimum elements for a valid state
        max_state_elements: Maximum elements per state
    """

    # Co-occurrence threshold
    min_cooccurrence_rate: float = 0.95

    # Fingerprint matching
    use_fingerprint_matching: bool = True
    fingerprint_match_threshold: float = 0.8

    # Position zone handling
    treat_header_footer_as_global: bool = True
    auto_detect_modal_states: bool = True

    # Repeat pattern handling
    dedupe_repeating_elements: bool = True
    max_repeat_representatives: int = 3

    # Size weighting
    use_size_weighting: bool = True
    size_weights: dict[str, float] = field(default_factory=lambda: DEFAULT_SIZE_WEIGHTS.copy())

    # Semantic matching
    enable_semantic_matching: bool = True
    semantic_match_roles: list[str] = field(
        default_factory=lambda: ["button", "link", "textbox", "checkbox", "menuitem"]
    )

    # State size limits
    min_state_elements: int = 1
    max_state_elements: int = 100


@dataclass
class DiscoveredFingerprintState:
    """A discovered UI state with fingerprint-derived properties.

    Attributes:
        state_id: Unique identifier for this state
        name: Human-readable name for the state
        fingerprint_hashes: List of fingerprint hashes in this state
        element_ids: Original element IDs if available
        position_zone: Dominant position zone for this state
        landmark_context: Dominant ARIA landmark
        is_global: True if this is a global state (header/footer)
        is_modal: True if this is a modal/blocking state
        repeat_pattern_count: Number of repeating elements (before deduplication)
        confidence: Confidence score (0.0-1.0)
        observation_count: Number of times this state was observed
        first_observed: Timestamp of first observation
        last_observed: Timestamp of last observation
        metadata: Additional metadata
    """

    state_id: str
    name: str
    fingerprint_hashes: list[str]
    element_ids: list[str] = field(default_factory=list)
    position_zone: str = "main"
    landmark_context: str = ""
    is_global: bool = False
    is_modal: bool = False
    repeat_pattern_count: int = 0
    confidence: float = 0.0
    observation_count: int = 1
    first_observed: datetime = field(default_factory=datetime.utcnow)
    last_observed: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stateId": self.state_id,
            "name": self.name,
            "fingerprintHashes": self.fingerprint_hashes,
            "elementIds": self.element_ids,
            "positionZone": self.position_zone,
            "landmarkContext": self.landmark_context,
            "isGlobal": self.is_global,
            "isModal": self.is_modal,
            "repeatPatternCount": self.repeat_pattern_count,
            "confidence": self.confidence,
            "observationCount": self.observation_count,
            "firstObserved": self.first_observed.isoformat(),
            "lastObserved": self.last_observed.isoformat(),
            "metadata": self.metadata,
        }


class FingerprintStateDiscovery:
    """Fingerprint-enhanced state discovery algorithm.

    This class discovers UI states from element fingerprints, using
    co-occurrence analysis enhanced with position zones, semantic
    matching, and repeat pattern handling.
    """

    def __init__(self, config: FingerprintStateDiscoveryConfig | None = None) -> None:
        """Initialize fingerprint state discovery.

        Args:
            config: Discovery configuration
        """
        self.config = config or FingerprintStateDiscoveryConfig()

        # Fingerprint registry
        self._fingerprints: dict[str, ElementFingerprint] = {}

        # Capture tracking
        self._captures: list[CaptureRecord] = []
        self._capture_fingerprints: dict[str, set[str]] = {}  # capture_id -> fingerprints

        # Transition tracking
        self._transitions: list[TransitionRecord] = []

        # Co-occurrence data
        self._cooccurrence_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._fingerprint_appearance_count: dict[str, int] = defaultdict(int)

        # State candidates (from UI Bridge or computed)
        self._state_candidates: list[StateCandidate] = []

        # Discovered states
        self._discovered_states: list[DiscoveredFingerprintState] = []

        # Global fingerprints (appear across all states)
        self._global_fingerprints: set[str] = set()

    def load_cooccurrence_export(self, export_data: dict[str, Any]) -> None:
        """Load co-occurrence data exported from UI Bridge.

        This is the primary method for consuming UI Bridge exports.
        It parses the export data and uses pre-computed state candidates
        as the starting point for state discovery.

        Args:
            export_data: Dictionary from CooccurrenceExport.to_dict() or
                         raw JSON from UI Bridge
        """
        export = CooccurrenceExport.from_dict(export_data)

        # Store fingerprint details
        self._fingerprints = export.fingerprint_details.copy()

        # Store co-occurrence counts
        self._cooccurrence_counts = defaultdict(
            lambda: defaultdict(int),
            {k: defaultdict(int, v) for k, v in export.cooccurrence_counts.items()},
        )

        # Calculate appearance counts from stats
        for fp_hash, stats in export.fingerprint_stats.items():
            self._fingerprint_appearance_count[fp_hash] = stats.total_appearances

        # Store transitions
        self._transitions = export.transitions.copy()

        # Parse presence matrix into captures
        for entry in export.presence_matrix:
            capture = CaptureRecord(
                capture_id=entry.capture_id,
                url=entry.url,
                title="",
                timestamp=0,
                fingerprint_hashes=entry.fingerprints,
            )
            self._captures.append(capture)
            self._capture_fingerprints[entry.capture_id] = set(entry.fingerprints)

        # Use pre-computed state candidates
        self._state_candidates = export.state_candidates.copy()

        # Identify global fingerprints (header/footer elements)
        if self.config.treat_header_footer_as_global:
            self._identify_global_fingerprints()

        logger.info(
            f"Loaded co-occurrence export: {len(self._fingerprints)} fingerprints, "
            f"{len(self._captures)} captures, {len(self._state_candidates)} candidates"
        )

    def process_capture(
        self, capture: CaptureRecord, fingerprints: list[ElementFingerprint]
    ) -> None:
        """Process a single capture incrementally.

        Use this method for real-time state discovery as captures
        are recorded, rather than batch processing.

        Args:
            capture: Capture metadata
            fingerprints: List of fingerprints visible in this capture
        """
        # Store capture
        self._captures.append(capture)
        fp_hashes = set(capture.fingerprint_hashes)
        self._capture_fingerprints[capture.capture_id] = fp_hashes

        # Update fingerprint registry
        for fp in fingerprints:
            if fp.hash not in self._fingerprints:
                self._fingerprints[fp.hash] = fp

        # Update appearance counts
        for fp_hash in fp_hashes:
            self._fingerprint_appearance_count[fp_hash] += 1

        # Update co-occurrence counts
        fp_list = list(fp_hashes)
        for i, fp1 in enumerate(fp_list):
            for fp2 in fp_list[i + 1 :]:
                self._cooccurrence_counts[fp1][fp2] += 1
                self._cooccurrence_counts[fp2][fp1] += 1

        # Re-identify global fingerprints
        if self.config.treat_header_footer_as_global:
            self._identify_global_fingerprints()

        logger.debug(f"Processed capture {capture.capture_id}: {len(fp_hashes)} fingerprints")

    def discover_states(self) -> list[DiscoveredFingerprintState]:
        """Run state discovery and return discovered states.

        Returns:
            List of discovered states
        """
        if not self._fingerprints:
            logger.warning("No fingerprints to discover states from")
            return []

        # Start with pre-computed state candidates if available
        if self._state_candidates:
            self._process_state_candidates()
        else:
            self._compute_state_candidates()

        # Filter and refine states
        self._refine_states()

        logger.info(f"Discovered {len(self._discovered_states)} states")
        return self._discovered_states

    def get_discovered_states(self) -> list[DiscoveredFingerprintState]:
        """Get discovered states (runs discovery if needed).

        Returns:
            List of discovered states
        """
        if not self._discovered_states and self._fingerprints:
            return self.discover_states()
        return self._discovered_states

    def get_fingerprint(self, fp_hash: str) -> ElementFingerprint | None:
        """Get fingerprint details by hash.

        Args:
            fp_hash: Fingerprint hash

        Returns:
            ElementFingerprint or None
        """
        return self._fingerprints.get(fp_hash)

    def get_transitions(self) -> list[TransitionRecord]:
        """Get recorded transitions.

        Returns:
            List of transition records
        """
        return self._transitions

    def get_global_fingerprints(self) -> set[str]:
        """Get fingerprints identified as global (header/footer).

        Returns:
            Set of global fingerprint hashes
        """
        return self._global_fingerprints.copy()

    def get_statistics(self) -> dict[str, Any]:
        """Get discovery statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_fingerprints": len(self._fingerprints),
            "total_captures": len(self._captures),
            "total_transitions": len(self._transitions),
            "state_candidates": len(self._state_candidates),
            "discovered_states": len(self._discovered_states),
            "global_fingerprints": len(self._global_fingerprints),
            "modal_states": sum(1 for s in self._discovered_states if s.is_modal),
            "global_states": sum(1 for s in self._discovered_states if s.is_global),
        }

    def reset(self) -> None:
        """Reset all discovery state."""
        self._fingerprints.clear()
        self._captures.clear()
        self._capture_fingerprints.clear()
        self._transitions.clear()
        self._cooccurrence_counts.clear()
        self._fingerprint_appearance_count.clear()
        self._state_candidates.clear()
        self._discovered_states.clear()
        self._global_fingerprints.clear()

    # =========================================================================
    # Position Zone Handling
    # =========================================================================

    def _identify_global_fingerprints(self) -> None:
        """Identify global fingerprints (header/footer elements)."""
        self._global_fingerprints.clear()

        for fp_hash, fp in self._fingerprints.items():
            if self._classify_element_scope(fp) == "global":
                self._global_fingerprints.add(fp_hash)

    def _classify_element_scope(self, fingerprint: ElementFingerprint) -> str:
        """Classify element as global, blocking, or state-specific.

        Args:
            fingerprint: Element fingerprint

        Returns:
            Scope classification: "global", "blocking", or "state-specific"
        """
        if fingerprint.position_zone in GLOBAL_POSITION_ZONES:
            return "global"
        elif fingerprint.position_zone in BLOCKING_POSITION_ZONES:
            return "blocking"
        else:
            return "state-specific"

    def _filter_global_elements(self, fingerprint_hashes: list[str]) -> tuple[list[str], list[str]]:
        """Separate global elements from state-specific elements.

        Args:
            fingerprint_hashes: List of fingerprint hashes

        Returns:
            Tuple of (global_fingerprints, state_specific_fingerprints)
        """
        global_fps: list[str] = []
        specific_fps: list[str] = []

        for fp_hash in fingerprint_hashes:
            if fp_hash in self._global_fingerprints:
                global_fps.append(fp_hash)
            else:
                specific_fps.append(fp_hash)

        return global_fps, specific_fps

    def _get_dominant_position_zone(self, fingerprint_hashes: list[str]) -> str:
        """Get the dominant position zone for a set of fingerprints.

        Args:
            fingerprint_hashes: List of fingerprint hashes

        Returns:
            Dominant position zone
        """
        zone_counts: dict[str, int] = defaultdict(int)

        for fp_hash in fingerprint_hashes:
            fp = self._fingerprints.get(fp_hash)
            if fp:
                zone_counts[fp.position_zone] += 1

        if not zone_counts:
            return "main"

        return max(zone_counts, key=zone_counts.get)  # type: ignore[arg-type]

    def _get_dominant_landmark(self, fingerprint_hashes: list[str]) -> str:
        """Get the dominant landmark context for a set of fingerprints.

        Args:
            fingerprint_hashes: List of fingerprint hashes

        Returns:
            Dominant landmark context
        """
        landmark_counts: dict[str, int] = defaultdict(int)

        for fp_hash in fingerprint_hashes:
            fp = self._fingerprints.get(fp_hash)
            if fp and fp.landmark_context:
                landmark_counts[fp.landmark_context] += 1

        if not landmark_counts:
            return ""

        return max(landmark_counts, key=landmark_counts.get)  # type: ignore[arg-type]

    # =========================================================================
    # Repeat Pattern Handling
    # =========================================================================

    def _dedupe_repeating_elements(self, fingerprint_hashes: list[str]) -> list[str]:
        """Deduplicate repeating elements, keeping representatives.

        For list items, grid items, and table rows, we keep only
        a few representatives to avoid noisy state candidates.

        Args:
            fingerprint_hashes: List of fingerprint hashes

        Returns:
            Deduplicated list of fingerprint hashes
        """
        if not self.config.dedupe_repeating_elements:
            return fingerprint_hashes

        seen_patterns: dict[str, list[str]] = {}
        non_repeating: list[str] = []

        for fp_hash in fingerprint_hashes:
            fp = self._fingerprints.get(fp_hash)
            if not fp:
                continue

            if fp.is_repeating and fp.repeat_pattern:
                pattern_key = fp.repeat_pattern.item_selector
                if pattern_key not in seen_patterns:
                    seen_patterns[pattern_key] = []
                seen_patterns[pattern_key].append(fp_hash)
            else:
                non_repeating.append(fp_hash)

        # Keep up to max_repeat_representatives from each pattern
        result = non_repeating.copy()
        for pattern_fps in seen_patterns.values():
            result.extend(pattern_fps[: self.config.max_repeat_representatives])

        return result

    def _count_repeat_patterns(self, fingerprint_hashes: list[str]) -> int:
        """Count the number of repeating elements in a fingerprint set.

        Args:
            fingerprint_hashes: List of fingerprint hashes

        Returns:
            Count of repeating elements
        """
        count = 0
        for fp_hash in fingerprint_hashes:
            fp = self._fingerprints.get(fp_hash)
            if fp and fp.is_repeating:
                count += 1
        return count

    # =========================================================================
    # Semantic Matching
    # =========================================================================

    def _elements_match(self, fp1: ElementFingerprint, fp2: ElementFingerprint) -> bool:
        """Check if two fingerprints represent the same logical element.

        Uses multiple matching strategies:
        1. Exact hash match
        2. Structural path + position + role match
        3. Semantic name + role + size match (if enabled)

        Args:
            fp1: First fingerprint
            fp2: Second fingerprint

        Returns:
            True if elements match
        """
        # Exact hash match
        if fp1.hash == fp2.hash:
            return True

        # Structural match
        if (
            fp1.structural_path == fp2.structural_path
            and fp1.position_zone == fp2.position_zone
            and fp1.role == fp2.role
        ):
            return True

        # Semantic match (for enabled roles)
        if self.config.enable_semantic_matching:
            if (
                fp1.role in self.config.semantic_match_roles
                and fp1.role == fp2.role
                and fp1.accessible_name
                and fp1.accessible_name == fp2.accessible_name
                and fp1.size_category == fp2.size_category
            ):
                return True

        return False

    def _find_matching_fingerprint(
        self, target_fp: ElementFingerprint, candidates: list[str]
    ) -> str | None:
        """Find a matching fingerprint from a list of candidates.

        Args:
            target_fp: Fingerprint to match
            candidates: List of candidate fingerprint hashes

        Returns:
            Matching fingerprint hash or None
        """
        for fp_hash in candidates:
            fp = self._fingerprints.get(fp_hash)
            if fp and self._elements_match(target_fp, fp):
                return fp_hash
        return None

    # =========================================================================
    # Size-Weighted Co-occurrence
    # =========================================================================

    def _get_element_weight(self, fingerprint: ElementFingerprint) -> float:
        """Get importance weight for an element based on size.

        Larger elements (panels, fullwidth) are more significant for
        state identity than smaller elements (icons, buttons).

        Args:
            fingerprint: Element fingerprint

        Returns:
            Weight value (0.0-1.0)
        """
        if not self.config.use_size_weighting:
            return 1.0

        return self.config.size_weights.get(fingerprint.size_category, 0.5)

    def _calculate_weighted_cooccurrence(self, group1: list[str], group2: list[str]) -> float:
        """Calculate size-weighted co-occurrence between two groups.

        Elements are weighted by their visual size, so larger elements
        contribute more to the co-occurrence score.

        Args:
            group1: First fingerprint group
            group2: Second fingerprint group

        Returns:
            Weighted co-occurrence score (0.0-1.0)
        """
        if not group1 or not group2:
            return 0.0

        # Get captures for each group
        captures1 = self._get_group_captures(group1)
        captures2 = self._get_group_captures(group2)

        if not captures1 or not captures2:
            return 0.0

        # Calculate weighted intersection
        intersection = captures1 & captures2
        union = captures1 | captures2

        if not union:
            return 0.0

        # Weight the result by average element weights
        weights1 = [
            self._get_element_weight(self._fingerprints[fp])
            for fp in group1
            if fp in self._fingerprints
        ]
        weights2 = [
            self._get_element_weight(self._fingerprints[fp])
            for fp in group2
            if fp in self._fingerprints
        ]

        avg_weight = (
            (sum(weights1) + sum(weights2)) / (len(weights1) + len(weights2))
            if weights1 or weights2
            else 1.0
        )

        base_score = len(intersection) / len(union)
        return base_score * avg_weight

    def _get_group_captures(self, fingerprint_hashes: list[str]) -> set[str]:
        """Get captures where all fingerprints in the group appear.

        Args:
            fingerprint_hashes: List of fingerprint hashes

        Returns:
            Set of capture IDs
        """
        if not fingerprint_hashes:
            return set()

        # Start with all captures
        result: set[str] | None = None

        for fp_hash in fingerprint_hashes:
            fp_captures = {
                capture_id
                for capture_id, fps in self._capture_fingerprints.items()
                if fp_hash in fps
            }

            if result is None:
                result = fp_captures
            else:
                result &= fp_captures

        return result or set()

    # =========================================================================
    # State Candidate Discovery
    # =========================================================================

    def _process_state_candidates(self) -> None:
        """Process pre-computed state candidates from UI Bridge."""
        for candidate in self._state_candidates:
            # Skip empty candidates
            if not candidate.fingerprints:
                continue

            # Apply repeat pattern deduplication
            deduped_fingerprints = self._dedupe_repeating_elements(candidate.fingerprints)

            # Skip candidates with too few elements
            if len(deduped_fingerprints) < self.config.min_state_elements:
                continue

            # Skip candidates with too many elements
            if len(deduped_fingerprints) > self.config.max_state_elements:
                continue

            # Separate global and state-specific fingerprints
            global_fps, specific_fps = self._filter_global_elements(deduped_fingerprints)

            # Create states based on scope
            if self.config.treat_header_footer_as_global and global_fps:
                # Create a global state from global fingerprints
                self._create_state_from_fingerprints(global_fps, is_global=True, is_modal=False)

            if specific_fps:
                # Determine if this is a modal state
                is_modal = (
                    self.config.auto_detect_modal_states
                    and self._get_dominant_position_zone(specific_fps) in BLOCKING_POSITION_ZONES
                )

                # Create state from state-specific fingerprints
                self._create_state_from_fingerprints(
                    specific_fps, is_global=False, is_modal=is_modal
                )

    def _compute_state_candidates(self) -> None:
        """Compute state candidates from co-occurrence data.

        This is used when no pre-computed candidates are available.
        """
        if not self._captures:
            return

        # Group fingerprints by exact co-occurrence signature
        signature_groups: dict[frozenset[str], set[str]] = defaultdict(set)

        for fp_hash in self._fingerprints:
            # Get signature: set of captures where this fingerprint appears
            signature = frozenset(
                capture_id
                for capture_id, fps in self._capture_fingerprints.items()
                if fp_hash in fps
            )

            if signature:  # Only include fingerprints that appear in captures
                signature_groups[signature].add(fp_hash)

        # Convert to state candidates
        for _signature, fp_group in signature_groups.items():
            if len(fp_group) >= self.config.min_state_elements:
                candidate = StateCandidate(
                    fingerprints=list(fp_group),
                    cooccurrence_rate=1.0,  # 100% co-occurrence by construction
                )
                self._state_candidates.append(candidate)

        # Process the computed candidates
        self._process_state_candidates()

    def _create_state_from_fingerprints(
        self,
        fingerprint_hashes: list[str],
        is_global: bool = False,
        is_modal: bool = False,
    ) -> DiscoveredFingerprintState:
        """Create a discovered state from a list of fingerprints.

        Args:
            fingerprint_hashes: List of fingerprint hashes
            is_global: Whether this is a global state
            is_modal: Whether this is a modal state

        Returns:
            Created state
        """
        # Generate state ID
        state_id = self._generate_state_id(fingerprint_hashes)

        # Check if state already exists
        for existing in self._discovered_states:
            if existing.state_id == state_id:
                existing.observation_count += 1
                existing.last_observed = datetime.utcnow()
                return existing

        # Generate name
        name = self._generate_state_name(fingerprint_hashes, is_global, is_modal)

        # Get properties
        position_zone = self._get_dominant_position_zone(fingerprint_hashes)
        landmark_context = self._get_dominant_landmark(fingerprint_hashes)
        repeat_count = self._count_repeat_patterns(fingerprint_hashes)

        # Calculate confidence
        confidence = self._calculate_state_confidence(fingerprint_hashes)

        # Get element IDs from fingerprints (if available in metadata)
        element_ids: list[str] = []
        for fp_hash in fingerprint_hashes:
            fp = self._fingerprints.get(fp_hash)
            if fp and fp.accessible_name:
                element_ids.append(fp.accessible_name)

        state = DiscoveredFingerprintState(
            state_id=state_id,
            name=name,
            fingerprint_hashes=sorted(fingerprint_hashes),
            element_ids=element_ids,
            position_zone=position_zone,
            landmark_context=landmark_context,
            is_global=is_global,
            is_modal=is_modal,
            repeat_pattern_count=repeat_count,
            confidence=confidence,
        )

        self._discovered_states.append(state)
        logger.debug(f"Created state: {name} ({len(fingerprint_hashes)} fingerprints)")

        return state

    def _generate_state_id(self, fingerprint_hashes: list[str]) -> str:
        """Generate deterministic state ID from fingerprints.

        Args:
            fingerprint_hashes: List of fingerprint hashes

        Returns:
            State ID
        """
        sorted_hashes = sorted(fingerprint_hashes)
        hash_input = "|".join(sorted_hashes)
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        return f"fp_state_{hash_value}"

    def _generate_state_name(
        self,
        fingerprint_hashes: list[str],
        is_global: bool = False,
        is_modal: bool = False,
    ) -> str:
        """Generate human-readable state name.

        Args:
            fingerprint_hashes: List of fingerprint hashes
            is_global: Whether this is a global state
            is_modal: Whether this is a modal state

        Returns:
            State name
        """
        # Get position zone and landmark for naming
        position_zone = self._get_dominant_position_zone(fingerprint_hashes)
        landmark = self._get_dominant_landmark(fingerprint_hashes)

        # Try to get a meaningful name from accessible names
        accessible_names: list[str] = []
        for fp_hash in fingerprint_hashes[:3]:  # Only check first few
            fp = self._fingerprints.get(fp_hash)
            if fp and fp.accessible_name:
                accessible_names.append(fp.accessible_name)

        # Build name parts
        parts: list[str] = []

        if is_modal:
            parts.append("Modal")
        elif is_global:
            parts.append("Global")

        if landmark:
            parts.append(landmark.replace("-", " ").title())
        elif position_zone != "main":
            parts.append(position_zone.replace("-", " ").title())

        if accessible_names:
            first_name = accessible_names[0][:30]  # Truncate long names
            parts.append(first_name)

        if not parts:
            parts.append("State")

        name = " ".join(parts)

        if len(fingerprint_hashes) > 1:
            name += f" ({len(fingerprint_hashes)} elements)"

        return name

    def _calculate_state_confidence(self, fingerprint_hashes: list[str]) -> float:
        """Calculate confidence score for a state.

        Based on:
        - Number of observations
        - Size of the fingerprint group
        - Co-occurrence consistency

        Args:
            fingerprint_hashes: List of fingerprint hashes

        Returns:
            Confidence score (0.0-1.0)
        """
        if not fingerprint_hashes:
            return 0.0

        # Get observation count (minimum across all fingerprints)
        observation_counts = [
            self._fingerprint_appearance_count.get(fp, 0) for fp in fingerprint_hashes
        ]
        min_observations = min(observation_counts) if observation_counts else 0

        # Observation score (logarithmic, caps at 10 observations)
        import math

        observation_score = min(1.0, math.log10(min_observations + 1) / 1.0)

        # Size score (more elements = higher confidence, caps at 10)
        size_score = min(1.0, len(fingerprint_hashes) / 10.0)

        # Weight score (average weight of elements)
        weights = [
            self._get_element_weight(self._fingerprints[fp])
            for fp in fingerprint_hashes
            if fp in self._fingerprints
        ]
        weight_score = sum(weights) / len(weights) if weights else 0.5

        # Combine scores
        return observation_score * 0.5 + size_score * 0.3 + weight_score * 0.2

    # =========================================================================
    # State Refinement
    # =========================================================================

    def _refine_states(self) -> None:
        """Refine discovered states by merging similar ones."""
        # Merge states with high overlap
        self._merge_similar_states()

        # Sort by confidence
        self._discovered_states.sort(key=lambda s: s.confidence, reverse=True)

    def _merge_similar_states(self) -> None:
        """Merge states with high fingerprint overlap."""
        if len(self._discovered_states) <= 1:
            return

        threshold = self.config.min_cooccurrence_rate
        merged = True

        while merged:
            merged = False
            new_states: list[DiscoveredFingerprintState] = []

            for state in self._discovered_states:
                merged_with_existing = False
                state_fps = set(state.fingerprint_hashes)

                for existing in new_states:
                    existing_fps = set(existing.fingerprint_hashes)

                    # Calculate Jaccard similarity
                    intersection = len(state_fps & existing_fps)
                    union = len(state_fps | existing_fps)
                    similarity = intersection / union if union > 0 else 0.0

                    if similarity >= threshold:
                        # Merge into existing state
                        existing.fingerprint_hashes = sorted(state_fps | existing_fps)
                        existing.observation_count += state.observation_count
                        existing.confidence = max(existing.confidence, state.confidence)
                        merged_with_existing = True
                        merged = True
                        break

                if not merged_with_existing:
                    new_states.append(state)

            self._discovered_states = new_states

    # =========================================================================
    # Transition Building
    # =========================================================================

    def build_transitions_from_records(
        self,
    ) -> list[dict[str, Any]]:
        """Build transition definitions from recorded transitions.

        Returns:
            List of transition dictionaries for state machine
        """
        transitions: list[dict[str, Any]] = []

        for record in self._transitions:
            # Find states affected by this transition
            before_state = self._find_state_for_fingerprints(record.disappeared_fingerprints)
            after_state = self._find_state_for_fingerprints(record.appeared_fingerprints)

            if before_state and after_state and before_state != after_state:
                transition = {
                    "id": record.action_id,
                    "name": f"{record.action_type} transition",
                    "actionType": record.action_type,
                    "fromStates": [before_state.state_id],
                    "activateStates": [after_state.state_id],
                    "triggerElement": record.target_fingerprint,
                    "timestamp": record.timestamp,
                }
                transitions.append(transition)

        return transitions

    def _find_state_for_fingerprints(
        self, fingerprint_hashes: list[str]
    ) -> DiscoveredFingerprintState | None:
        """Find the state that best matches a set of fingerprints.

        Args:
            fingerprint_hashes: List of fingerprint hashes

        Returns:
            Best matching state or None
        """
        if not fingerprint_hashes:
            return None

        target_set = set(fingerprint_hashes)
        best_match: DiscoveredFingerprintState | None = None
        best_overlap = 0.0

        for state in self._discovered_states:
            state_set = set(state.fingerprint_hashes)
            intersection = len(target_set & state_set)
            union = len(target_set | state_set)
            overlap = intersection / union if union > 0 else 0.0

            if overlap > best_overlap:
                best_overlap = overlap
                best_match = state

        return best_match if best_overlap > 0.5 else None
