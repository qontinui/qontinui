"""
State Identifier for algorithmic state detection.

Identifies states from element compositions without semantic analysis.
States are defined by which elements are present on a page - two pages
with the same element composition are the same state.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from .element_comparator import CrossPageMatch, CrossPageMatcher, ElementComparator
from .models import (
    ElementFingerprint,
    ExtractedPageV2,
    IdentifiedState,
)

logger = logging.getLogger(__name__)


@dataclass
class StateSignature:
    """
    Signature representing a state's element composition.

    Used to identify pages that represent the same state.
    """

    # Hash of the element composition
    signature_hash: str

    # Element fingerprint hashes that make up this signature
    element_hashes: frozenset[str]

    # Number of elements
    element_count: int

    # Optional: additional metadata for debugging
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageStateMapping:
    """Mapping of a page to its identified state."""

    page_url: str
    screenshot_id: str
    state_id: str
    signature: StateSignature
    element_ids: list[str]


class StateIdentifier:
    """
    Identifies states algorithmically from element compositions.

    Process:
    1. For each page, build element signature from fingerprints
    2. Compare signatures across pages
    3. Pages with matching signatures = same state
    4. Group elements by cross-page matches (same visual element)
    5. Use matched elements as the "canonical" elements for states

    This approach:
    - Requires no semantic detection
    - Works for any UI (web, desktop, mobile)
    - Handles dynamic content (elements matched visually, not by selector)
    """

    def __init__(
        self,
        comparator: ElementComparator | None = None,
        similarity_threshold: float = 0.90,
        min_matching_elements: float = 0.7,
    ) -> None:
        """
        Initialize the state identifier.

        Args:
            comparator: Element comparator for cross-page matching.
            similarity_threshold: Threshold for element similarity.
            min_matching_elements: Minimum fraction of elements that must
                match for two pages to be considered the same state.
        """
        self.matcher = CrossPageMatcher(comparator, threshold=similarity_threshold)
        self.similarity_threshold = similarity_threshold
        self.min_matching_elements = min_matching_elements

    def identify_states(
        self,
        pages: list[ExtractedPageV2],
    ) -> tuple[list[IdentifiedState], list[PageStateMapping]]:
        """
        Identify states from extracted pages.

        Args:
            pages: List of extracted pages with elements and fingerprints.

        Returns:
            Tuple of (identified_states, page_state_mappings).
        """
        if not pages:
            return [], []

        logger.info(f"Identifying states from {len(pages)} pages...")

        # Step 1: Build element signatures for each page
        signatures = self._build_page_signatures(pages)

        # Step 2: Find cross-page element matches
        all_fingerprints = []
        for page in pages:
            all_fingerprints.extend(page.fingerprints)

        element_matches = self.matcher.find_matches(all_fingerprints)

        # Step 3: Build canonical element mapping
        # Maps element_id -> canonical_element_id (representative element)
        canonical_map = self._build_canonical_map(element_matches)

        # Step 4: Rebuild signatures using canonical elements
        canonical_signatures = self._canonicalize_signatures(signatures, canonical_map)

        # Step 5: Group pages by signature similarity
        state_groups = self._group_pages_by_state(canonical_signatures, pages)

        # Step 6: Create IdentifiedState objects
        states = []
        page_mappings = []

        for state_id, group in state_groups.items():
            # Get all elements for this state
            state_element_ids = set()
            page_urls = []
            screenshot_ids = []

            for page_url, screenshot_id, _sig in group:
                page_urls.append(page_url)
                screenshot_ids.append(screenshot_id)

                # Find page to get element IDs
                for page in pages:
                    if page.screenshot_id == screenshot_id:
                        state_element_ids.update(e.id for e in page.elements)
                        break

            # Generate state name from semantic hints
            state_name = self._generate_state_name(state_id, state_element_ids, pages)

            state = IdentifiedState(
                id=state_id,
                name=state_name,
                element_ids=list(state_element_ids),
                page_urls=page_urls,
                screenshot_ids=screenshot_ids,
                element_fingerprint_hashes=group[0][2].element_hashes if group else frozenset(),
            )
            states.append(state)

            # Create page mappings
            for page_url, screenshot_id, sig in group:
                page_mappings.append(
                    PageStateMapping(
                        page_url=page_url,
                        screenshot_id=screenshot_id,
                        state_id=state_id,
                        signature=sig,
                        element_ids=[
                            e.id
                            for page in pages
                            if page.screenshot_id == screenshot_id
                            for e in page.elements
                        ],
                    )
                )

        logger.info(f"Identified {len(states)} unique states")
        return states, page_mappings

    def _build_page_signatures(
        self,
        pages: list[ExtractedPageV2],
    ) -> dict[str, StateSignature]:
        """Build element signatures for each page."""
        signatures: dict[str, StateSignature] = {}

        for page in pages:
            if not page.fingerprints:
                logger.warning(f"No fingerprints for page {page.url}")
                continue

            # Create signature from fingerprint hashes
            element_hashes = frozenset(self._fingerprint_hash(fp) for fp in page.fingerprints)

            # Create stable signature hash
            sorted_hashes = sorted(element_hashes)
            signature_hash = hashlib.sha256("|".join(sorted_hashes).encode()).hexdigest()[:16]

            signatures[page.screenshot_id] = StateSignature(
                signature_hash=signature_hash,
                element_hashes=element_hashes,
                element_count=len(page.fingerprints),
                metadata={
                    "url": page.url,
                    "title": page.title,
                },
            )

        return signatures

    def _fingerprint_hash(self, fp: ElementFingerprint) -> str:
        """
        Create a hash representing an element fingerprint.

        Uses visual and content properties for matching.
        """
        # Combine key features
        parts = [
            fp.size_class.value,
            str(fp.width),
            str(fp.height),
            fp.visual_hash or "",
            fp.content_hash,
        ]

        return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]

    def _build_canonical_map(
        self,
        matches: list[CrossPageMatch],
    ) -> dict[str, str]:
        """
        Build mapping from element IDs to canonical (representative) element IDs.

        Elements that match across pages get mapped to the same canonical ID.
        """
        canonical: dict[str, str] = {}

        for match in matches:
            # Use the first element as canonical
            canonical_id = match.element_id
            canonical[match.element_id] = canonical_id

            for elem_id, _, _ in match.matching_elements:
                canonical[elem_id] = canonical_id

        return canonical

    def _canonicalize_signatures(
        self,
        signatures: dict[str, StateSignature],
        canonical_map: dict[str, str],
    ) -> dict[str, StateSignature]:
        """
        Rebuild signatures using canonical element IDs.

        This allows signatures from different pages with matching
        elements to be compared more accurately.
        """
        # Note: For now, we keep original signatures since fingerprint hashes
        # already handle visual matching. In future, could rebuild with
        # canonical element hashes.
        return signatures

    def _group_pages_by_state(
        self,
        signatures: dict[str, StateSignature],
        pages: list[ExtractedPageV2],
    ) -> dict[str, list[tuple[str, str, StateSignature]]]:
        """
        Group pages by state similarity.

        Pages with similar element compositions are grouped together.
        """
        groups: dict[str, list[tuple[str, str, StateSignature]]] = {}
        processed: set[str] = set()

        # Get page info
        page_info = {page.screenshot_id: (page.url, page.screenshot_id) for page in pages}

        # First pass: group by exact signature match
        for screenshot_id, sig in signatures.items():
            if screenshot_id in processed:
                continue

            state_id = sig.signature_hash
            page_url, _ = page_info.get(screenshot_id, ("", screenshot_id))

            if state_id not in groups:
                groups[state_id] = []

            groups[state_id].append((page_url, screenshot_id, sig))
            processed.add(screenshot_id)

        # Second pass: merge similar states
        # (Pages that share significant element overlap)
        merged_groups = self._merge_similar_states(groups, signatures)

        return merged_groups

    def _merge_similar_states(
        self,
        groups: dict[str, list[tuple[str, str, StateSignature]]],
        signatures: dict[str, StateSignature],
    ) -> dict[str, list[tuple[str, str, StateSignature]]]:
        """
        Merge state groups that have significant element overlap.

        Two states are merged if they share >= min_matching_elements
        of their elements.
        """
        state_ids = list(groups.keys())
        merged: dict[str, str] = {sid: sid for sid in state_ids}

        for i, sid_a in enumerate(state_ids):
            for sid_b in state_ids[i + 1 :]:
                if merged[sid_a] != merged[merged[sid_a]] or merged[sid_b] != merged[merged[sid_b]]:
                    continue  # Already merged

                # Get representative signatures
                sig_a = signatures.get(groups[sid_a][0][1]) if groups[sid_a] else None
                sig_b = signatures.get(groups[sid_b][0][1]) if groups[sid_b] else None

                if sig_a and sig_b:
                    overlap = self._compute_signature_overlap(sig_a, sig_b)

                    if overlap >= self.min_matching_elements:
                        # Merge sid_b into sid_a
                        merged[sid_b] = sid_a
                        logger.debug(
                            f"Merging state {sid_b} into {sid_a} " f"(overlap: {overlap:.2%})"
                        )

        # Rebuild groups with merges
        final_groups: dict[str, list[tuple[str, str, StateSignature]]] = {}

        for sid, pages in groups.items():
            target_id = merged[sid]
            # Follow merge chain
            while merged[target_id] != target_id:
                target_id = merged[target_id]

            if target_id not in final_groups:
                final_groups[target_id] = []
            final_groups[target_id].extend(pages)

        return final_groups

    def _compute_signature_overlap(
        self,
        sig_a: StateSignature,
        sig_b: StateSignature,
    ) -> float:
        """
        Compute overlap between two state signatures.

        Returns fraction of elements that match.
        """
        if not sig_a.element_hashes or not sig_b.element_hashes:
            return 0.0

        intersection = sig_a.element_hashes & sig_b.element_hashes
        union = sig_a.element_hashes | sig_b.element_hashes

        if not union:
            return 0.0

        # Jaccard similarity
        return len(intersection) / len(union)

    def _generate_state_name(
        self,
        state_id: str,
        element_ids: set[str],
        pages: list[ExtractedPageV2],
    ) -> str:
        """
        Generate a human-readable name for the state.

        Uses semantic hints from elements:
        - ARIA labels
        - Text content
        - Semantic tags (nav, header, etc.)
        - Page title
        """
        # Collect semantic hints
        hints: list[str] = []

        for page in pages:
            # Use page title
            if page.title and page.title not in hints:
                hints.append(page.title)

            # Check elements for semantic hints
            for element in page.elements:
                if element.id not in element_ids:
                    continue

                # ARIA label
                if element.aria_label:
                    hints.append(element.aria_label)

                # Semantic tag
                if element.tag_semantic:
                    hints.append(element.tag_semantic)

                # Semantic role
                if element.semantic_role:
                    hints.append(element.semantic_role)

        # Pick best hint
        if hints:
            # Prefer page title
            for hint in hints:
                if len(hint) > 3 and len(hint) < 50:
                    return hint[:30]

        # Fallback to state ID prefix
        return f"State_{state_id[:8]}"

    def compute_state_similarity(
        self,
        state_a: IdentifiedState,
        state_b: IdentifiedState,
    ) -> float:
        """
        Compute similarity between two states.

        Based on element fingerprint overlap.
        """
        if not state_a.element_fingerprint_hashes or not state_b.element_fingerprint_hashes:
            return 0.0

        intersection = state_a.element_fingerprint_hashes & state_b.element_fingerprint_hashes
        union = state_a.element_fingerprint_hashes | state_b.element_fingerprint_hashes

        if not union:
            return 0.0

        return len(intersection) / len(union)


def identify_states_from_pages(
    pages: list[ExtractedPageV2],
    comparator: ElementComparator | None = None,
    threshold: float = 0.90,
) -> tuple[list[IdentifiedState], list[PageStateMapping]]:
    """
    Convenience function to identify states from pages.

    Args:
        pages: Extracted pages with elements and fingerprints.
        comparator: Optional element comparator.
        threshold: Similarity threshold.

    Returns:
        Tuple of (states, page_mappings).
    """
    identifier = StateIdentifier(comparator, threshold)
    return identifier.identify_states(pages)
