"""
Element Comparator for cross-page element matching.

Provides modular comparison strategies for finding duplicate/similar
elements across pages. Uses Strategy Pattern to allow swapping algorithms.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np

from .element_fingerprint import (
    compute_hash_similarity,
    compute_histogram_similarity,
    compute_position_similarity,
    compute_size_similarity,
)
from .models import ElementFingerprint

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of comparing two elements."""

    element_a_id: str
    element_b_id: str
    similarity: float
    is_match: bool
    component_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossPageMatch:
    """A match between elements on different pages."""

    element_id: str
    screenshot_id: str
    matching_elements: list[tuple[str, str, float]]  # (element_id, screenshot_id, similarity)


@runtime_checkable
class ElementComparator(Protocol):
    """
    Protocol for element comparison strategies.

    Implementations can use different algorithms:
    - Fingerprint-based (size, color, visual hash)
    - Template matching on screenshots
    - Embedding-based similarity search
    """

    threshold: float

    def compare(
        self,
        fp_a: ElementFingerprint,
        fp_b: ElementFingerprint,
        image_a: np.ndarray | None = None,
        image_b: np.ndarray | None = None,
    ) -> MatchResult:
        """
        Compare two elements and return similarity.

        Args:
            fp_a: First element fingerprint.
            fp_b: Second element fingerprint.
            image_a: Optional cropped image of first element.
            image_b: Optional cropped image of second element.

        Returns:
            MatchResult with similarity score and component breakdown.
        """
        ...


class DefaultComparator:
    """
    Default comparator combining multiple fingerprint components.

    Weights:
    - Size: 0.15 (must match size class)
    - Color histogram: 0.25
    - Visual hash: 0.40
    - Content hash: 0.10
    - Position: 0.10
    """

    def __init__(
        self,
        threshold: float = 0.90,
        size_weight: float = 0.15,
        color_weight: float = 0.25,
        visual_weight: float = 0.40,
        content_weight: float = 0.10,
        position_weight: float = 0.10,
    ) -> None:
        """
        Initialize the default comparator.

        Args:
            threshold: Minimum similarity to consider a match.
            size_weight: Weight for size similarity.
            color_weight: Weight for color histogram similarity.
            visual_weight: Weight for visual hash similarity.
            content_weight: Weight for content hash match.
            position_weight: Weight for position similarity.
        """
        self.threshold = threshold
        self.size_weight = size_weight
        self.color_weight = color_weight
        self.visual_weight = visual_weight
        self.content_weight = content_weight
        self.position_weight = position_weight

    def compare(
        self,
        fp_a: ElementFingerprint,
        fp_b: ElementFingerprint,
        image_a: np.ndarray | None = None,
        image_b: np.ndarray | None = None,
    ) -> MatchResult:
        """Compare two element fingerprints."""
        # Quick rejection: size class must match
        if not fp_a.quick_match(fp_b):
            return MatchResult(
                element_a_id=fp_a.element_id,
                element_b_id=fp_b.element_id,
                similarity=0.0,
                is_match=False,
                component_scores={"size": 0.0},
                metadata={"rejection": "size_class_mismatch"},
            )

        # Compute component similarities
        scores: dict[str, float] = {}

        # Size similarity
        scores["size"] = compute_size_similarity(fp_a, fp_b)

        # Color histogram similarity
        if fp_a.color_histogram and fp_b.color_histogram:
            scores["color"] = compute_histogram_similarity(
                fp_a.color_histogram, fp_b.color_histogram
            )
        else:
            scores["color"] = 0.5  # Neutral if no histogram

        # Visual hash similarity
        if fp_a.visual_hash and fp_b.visual_hash:
            scores["visual"] = compute_hash_similarity(fp_a.visual_hash, fp_b.visual_hash)
        else:
            scores["visual"] = 0.5  # Neutral if no hash

        # Content hash similarity (exact match or 0)
        scores["content"] = 1.0 if fp_a.content_hash == fp_b.content_hash else 0.0

        # Position similarity
        scores["position"] = compute_position_similarity(fp_a, fp_b)

        # Compute weighted similarity
        similarity = (
            scores["size"] * self.size_weight
            + scores["color"] * self.color_weight
            + scores["visual"] * self.visual_weight
            + scores["content"] * self.content_weight
            + scores["position"] * self.position_weight
        )

        return MatchResult(
            element_a_id=fp_a.element_id,
            element_b_id=fp_b.element_id,
            similarity=similarity,
            is_match=similarity >= self.threshold,
            component_scores=scores,
        )


class TemplateMatchComparator:
    """
    Comparator using OpenCV template matching on element screenshots.

    More accurate but requires element images. Falls back to
    fingerprint comparison if images not available.
    """

    def __init__(
        self,
        threshold: float = 0.90,
        method: int = cv2.TM_CCOEFF_NORMED,
        fallback_comparator: ElementComparator | None = None,
    ) -> None:
        """
        Initialize template match comparator.

        Args:
            threshold: Minimum similarity to consider a match.
            method: OpenCV template matching method.
            fallback_comparator: Comparator to use if images unavailable.
        """
        self.threshold = threshold
        self.method = method
        self.fallback = fallback_comparator or DefaultComparator(threshold)

    def compare(
        self,
        fp_a: ElementFingerprint,
        fp_b: ElementFingerprint,
        image_a: np.ndarray | None = None,
        image_b: np.ndarray | None = None,
    ) -> MatchResult:
        """Compare using template matching."""
        # Quick rejection: size class must match
        if not fp_a.quick_match(fp_b):
            return MatchResult(
                element_a_id=fp_a.element_id,
                element_b_id=fp_b.element_id,
                similarity=0.0,
                is_match=False,
                component_scores={"size": 0.0},
                metadata={"rejection": "size_class_mismatch"},
            )

        # Fall back if no images
        if image_a is None or image_b is None:
            result = self.fallback.compare(fp_a, fp_b)
            result.metadata["fallback"] = True
            return result

        try:
            # Ensure same size for comparison
            h_a, w_a = image_a.shape[:2]
            h_b, w_b = image_b.shape[:2]

            # Resize smaller to larger dimensions
            target_h = max(h_a, h_b)
            target_w = max(w_a, w_b)

            img_a = cv2.resize(image_a, (target_w, target_h))
            img_b = cv2.resize(image_b, (target_w, target_h))

            # Convert to grayscale for matching
            if len(img_a.shape) == 3:
                gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
            else:
                gray_a = img_a

            if len(img_b.shape) == 3:
                gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
            else:
                gray_b = img_b

            # Template match (use smaller as template)
            if gray_a.size <= gray_b.size:
                template, search = gray_a, gray_b
            else:
                template, search = gray_b, gray_a

            # Pad search image if template is larger
            if template.shape[0] > search.shape[0] or template.shape[1] > search.shape[1]:
                # Swap
                template, search = search, template

            match_result = cv2.matchTemplate(search, template, self.method)
            _, max_val, _, _ = cv2.minMaxLoc(match_result)

            # Normalize similarity to 0-1 range
            similarity = float(max_val)
            if self.method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
                similarity = 1.0 - similarity

            return MatchResult(
                element_a_id=fp_a.element_id,
                element_b_id=fp_b.element_id,
                similarity=similarity,
                is_match=similarity >= self.threshold,
                component_scores={"template_match": similarity},
                metadata={"method": "template_match"},
            )

        except Exception as e:
            logger.warning(f"Template matching failed: {e}")
            fallback_result = self.fallback.compare(fp_a, fp_b)
            fallback_result.metadata["fallback"] = True
            fallback_result.metadata["error"] = str(e)
            return fallback_result


class CompositeComparator:
    """
    Composite comparator that combines multiple strategies.

    Useful for multi-stage matching:
    1. Quick fingerprint check
    2. Template matching confirmation
    """

    def __init__(
        self,
        comparators: list[tuple[ElementComparator, float]],
        threshold: float = 0.90,
    ) -> None:
        """
        Initialize composite comparator.

        Args:
            comparators: List of (comparator, weight) tuples.
            threshold: Overall threshold for match.
        """
        self.comparators = comparators
        self.threshold = threshold

        # Normalize weights
        total_weight = sum(w for _, w in comparators)
        self.weights = [(c, w / total_weight) for c, w in comparators]

    def compare(
        self,
        fp_a: ElementFingerprint,
        fp_b: ElementFingerprint,
        image_a: np.ndarray | None = None,
        image_b: np.ndarray | None = None,
    ) -> MatchResult:
        """Compare using all comparators."""
        # Quick rejection first
        if not fp_a.quick_match(fp_b):
            return MatchResult(
                element_a_id=fp_a.element_id,
                element_b_id=fp_b.element_id,
                similarity=0.0,
                is_match=False,
                component_scores={"size": 0.0},
                metadata={"rejection": "size_class_mismatch"},
            )

        total_similarity = 0.0
        all_scores: dict[str, float] = {}

        for comparator, weight in self.weights:
            result = comparator.compare(fp_a, fp_b, image_a, image_b)
            total_similarity += result.similarity * weight

            # Merge component scores
            for key, value in result.component_scores.items():
                all_scores[f"{type(comparator).__name__}_{key}"] = value

        return MatchResult(
            element_a_id=fp_a.element_id,
            element_b_id=fp_b.element_id,
            similarity=total_similarity,
            is_match=total_similarity >= self.threshold,
            component_scores=all_scores,
            metadata={"method": "composite"},
        )


class CrossPageMatcher:
    """
    Finds matching elements across multiple pages.

    Optimized for comparing elements from many pages:
    1. Groups elements by size class (reduces O(n²))
    2. Uses fingerprint pre-filtering
    3. Confirms with chosen comparator
    """

    def __init__(
        self,
        comparator: ElementComparator | None = None,
        threshold: float = 0.90,
    ) -> None:
        """
        Initialize the cross-page matcher.

        Args:
            comparator: Comparator strategy to use.
            threshold: Minimum similarity for a match.
        """
        self.comparator = comparator or DefaultComparator(threshold)
        self.threshold = threshold

    def find_matches(
        self,
        fingerprints: list[ElementFingerprint],
        element_images: dict[str, np.ndarray] | None = None,
    ) -> list[CrossPageMatch]:
        """
        Find matching elements across all pages.

        Args:
            fingerprints: All element fingerprints from all pages.
            element_images: Optional dict of element_id -> cropped image.

        Returns:
            List of CrossPageMatch objects grouping similar elements.
        """
        if not fingerprints:
            return []

        logger.info(f"Finding matches among {len(fingerprints)} elements...")

        # Group by size class for efficiency
        by_size: dict[str, list[ElementFingerprint]] = {}
        for fp in fingerprints:
            key = fp.size_class.value
            if key not in by_size:
                by_size[key] = []
            by_size[key].append(fp)

        # Find matches within each size group
        matches: dict[str, CrossPageMatch] = {}
        processed: set[str] = set()

        for size_class, group in by_size.items():
            logger.debug(f"Processing size class {size_class}: {len(group)} elements")

            for i, fp_a in enumerate(group):
                if fp_a.element_id in processed:
                    continue

                match_info = CrossPageMatch(
                    element_id=fp_a.element_id,
                    screenshot_id=fp_a.screenshot_id,
                    matching_elements=[],
                )

                for fp_b in group[i + 1 :]:
                    if fp_b.element_id in processed:
                        continue

                    # Skip same-page comparisons
                    if fp_a.screenshot_id == fp_b.screenshot_id:
                        continue

                    # Get images if available
                    img_a = element_images.get(fp_a.element_id) if element_images else None
                    img_b = element_images.get(fp_b.element_id) if element_images else None

                    result = self.comparator.compare(fp_a, fp_b, img_a, img_b)

                    if result.is_match:
                        match_info.matching_elements.append(
                            (fp_b.element_id, fp_b.screenshot_id, result.similarity)
                        )
                        processed.add(fp_b.element_id)

                if match_info.matching_elements:
                    matches[fp_a.element_id] = match_info
                    processed.add(fp_a.element_id)

        logger.info(f"Found {len(matches)} element groups with cross-page matches")
        return list(matches.values())

    def find_element_matches(
        self,
        element_fp: ElementFingerprint,
        candidate_fps: list[ElementFingerprint],
        element_images: dict[str, np.ndarray] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Find matches for a single element among candidates.

        Args:
            element_fp: Element to find matches for.
            candidate_fps: Candidate fingerprints to compare against.
            element_images: Optional element images.

        Returns:
            List of (element_id, similarity) tuples for matches.
        """
        matches = []
        img_a = element_images.get(element_fp.element_id) if element_images else None

        for fp_b in candidate_fps:
            # Skip same element
            if fp_b.element_id == element_fp.element_id:
                continue

            # Quick rejection
            if not element_fp.quick_match(fp_b):
                continue

            img_b = element_images.get(fp_b.element_id) if element_images else None
            result = self.comparator.compare(element_fp, fp_b, img_a, img_b)

            if result.is_match:
                matches.append((fp_b.element_id, result.similarity))

        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches


def create_comparator(
    strategy: str = "default",
    threshold: float = 0.90,
    **kwargs,
) -> ElementComparator:
    """
    Factory function to create a comparator by strategy name.

    Args:
        strategy: Strategy name ("default", "template", "composite").
        threshold: Similarity threshold.
        **kwargs: Strategy-specific options.

    Returns:
        Configured ElementComparator instance.
    """
    if strategy == "default":
        return DefaultComparator(threshold=threshold, **kwargs)
    elif strategy == "template":
        return TemplateMatchComparator(threshold=threshold, **kwargs)
    elif strategy == "composite":
        default = DefaultComparator(threshold=threshold)
        template = TemplateMatchComparator(threshold=threshold)
        return CompositeComparator(
            comparators=[(default, 0.4), (template, 0.6)],
            threshold=threshold,
        )
    else:
        logger.warning(f"Unknown strategy '{strategy}', using default")
        return DefaultComparator(threshold=threshold)
