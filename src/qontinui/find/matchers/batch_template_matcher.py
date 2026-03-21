"""Batch multi-template matching using Multi-Template-Matching library.

Matches multiple templates against a single screenshot in one pass,
with cross-template Non-Maximum Suppression (NMS) to deduplicate
overlapping detections from different templates.

This is more efficient than sequential single-template matching when
searching for multiple patterns: one screenshot capture, one NMS pass,
and parallelized template search via MTM's ThreadPoolExecutor.
"""

import logging
from typing import Any

import cv2
import numpy as np

from ...exceptions import ImageProcessingError
from ...model.element import Location, Pattern, Region
from ...model.match import Match as MatchObject
from ..match import Match

logger = logging.getLogger(__name__)

# Maximum templates per batch to avoid memory spikes
# (each template creates a full-size score map)
MAX_BATCH_SIZE = 20


class BatchTemplateMatcher:
    """Batch multi-template matcher using MTM library.

    Searches for multiple template patterns in a single screenshot
    simultaneously, returning deduplicated results with NMS applied
    across all templates.

    Args:
        method: OpenCV matching method (1-5; method 0/TM_SQDIFF not supported by MTM).
        nms_overlap_threshold: IoU threshold for cross-template NMS (0.0-1.0).
    """

    # MTM method mapping (MTM uses integer method IDs, not TM_SQDIFF=0)
    _METHOD_MAP = {
        "TM_CCOEFF": cv2.TM_CCOEFF,
        "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
        "TM_CCORR": cv2.TM_CCORR,
        "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
        "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
    }

    def __init__(
        self,
        method: str = "TM_CCOEFF_NORMED",
        nms_overlap_threshold: float = 0.3,
    ) -> None:
        if method == "TM_SQDIFF":
            raise ValueError(
                "TM_SQDIFF (method 0) is not supported by MTM. "
                "Use TM_CCOEFF_NORMED or another normalized method."
            )
        if method not in self._METHOD_MAP:
            raise ValueError(
                f"Unknown method: {method}. Available: {list(self._METHOD_MAP.keys())}"
            )
        self.method = method
        self.nms_overlap_threshold = nms_overlap_threshold

    def find_all_patterns(
        self,
        screenshot: Any,
        patterns: list[Pattern],
        similarity: float = 0.8,
        search_region: tuple[int, int, int, int] | None = None,
    ) -> dict[str, list[Match]]:
        """Find all patterns in a single screenshot using batch matching.

        Args:
            screenshot: Screenshot image (PIL Image, numpy array, or OpenCV mat).
            patterns: List of patterns to search for.
            similarity: Minimum similarity threshold (0.0-1.0).
            search_region: Optional (x, y, width, height) to restrict search area.

        Returns:
            Dict mapping pattern.name to list of Match objects.
            Patterns with no matches map to empty lists.

        Raises:
            ImageProcessingError: If image conversion or matching fails.
        """
        try:
            import MTM
        except ImportError as e:
            raise ImageProcessingError(
                "Multi-Template-Matching not installed. "
                "Install with: pip install Multi-Template-Matching"
            ) from e

        if not patterns:
            return {}

        # Convert screenshot to OpenCV BGR
        screenshot_bgr = self._convert_to_opencv(screenshot)

        # Apply search region
        search_img, offset_x, offset_y = self._apply_search_region(screenshot_bgr, search_region)

        # Separate masked vs unmasked patterns
        # MTM has limited mask support, so masked patterns fall back to sequential
        unmasked_patterns: list[Pattern] = []
        masked_patterns: list[Pattern] = []

        for p in patterns:
            if self._has_active_mask(p):
                masked_patterns.append(p)
            else:
                unmasked_patterns.append(p)

        results: dict[str, list[Match]] = {p.name: [] for p in patterns}

        # Batch match unmasked patterns via MTM
        if unmasked_patterns:
            batch_results = self._batch_match(
                search_img, unmasked_patterns, similarity, offset_x, offset_y
            )
            results.update(batch_results)

        # Sequential fallback for masked patterns
        if masked_patterns:
            from .template_matcher import TemplateMatcher

            seq_matcher = TemplateMatcher()
            for p in masked_patterns:
                matches = seq_matcher.find_matches(
                    screenshot=screenshot_bgr,
                    pattern=p,
                    find_all=True,
                    similarity=similarity,
                    search_region=search_region,
                )
                results[p.name] = matches

        return results

    def _batch_match(
        self,
        search_img: np.ndarray,
        patterns: list[Pattern],
        similarity: float,
        offset_x: int,
        offset_y: int,
    ) -> dict[str, list[Match]]:
        """Run MTM batch matching on unmasked patterns.

        Chunks patterns into batches of MAX_BATCH_SIZE to limit memory.
        """
        import MTM

        all_results: dict[str, list[Match]] = {p.name: [] for p in patterns}

        # Chunk patterns to limit memory usage
        for chunk_start in range(0, len(patterns), MAX_BATCH_SIZE):
            chunk = patterns[chunk_start : chunk_start + MAX_BATCH_SIZE]

            # Build MTM template list: [(label, template_array), ...]
            list_templates = []
            for p in chunk:
                template_bgr = self._get_template_bgr(p)
                list_templates.append((p.name, template_bgr))

            method_id = self._METHOD_MAP[self.method]

            # MTM.matchTemplates returns list of (label, bbox, score) tuples
            hits = MTM.matchTemplates(
                list_templates,
                search_img,
                score_threshold=similarity,
                method=method_id,
                maxOverlap=self.nms_overlap_threshold,
            )

            # Convert MTM hits to Match objects, grouped by pattern name
            for hit in hits:
                label, bbox, score = hit
                x, y, w, h = bbox

                # Apply search region offset
                abs_x = int(x) + offset_x
                abs_y = int(y) + offset_y

                center_x = abs_x + int(w) // 2
                center_y = abs_y + int(h) // 2

                match_obj = MatchObject(
                    target=Location(
                        x=center_x,
                        y=center_y,
                        region=Region(abs_x, abs_y, int(w), int(h)),
                    ),
                    score=float(score),
                    name=label,
                )
                all_results[label].append(Match(match_obj))

        # Sort each pattern's matches by score descending
        for name in all_results:
            all_results[name].sort(key=lambda m: m.similarity, reverse=True)

        return all_results

    def _has_active_mask(self, pattern: Pattern) -> bool:
        """Check if pattern has an active mask (not all-opaque)."""
        if pattern.mask is not None:
            if not (np.all(pattern.mask == 1.0) or np.all(pattern.mask == 255)):
                return True

        if pattern.pixel_data is not None:
            if len(pattern.pixel_data.shape) == 3 and pattern.pixel_data.shape[2] == 4:
                alpha = pattern.pixel_data[:, :, 3]
                if not np.all(alpha == 255):
                    return True

        return False

    def _get_template_bgr(self, pattern: Pattern) -> np.ndarray:
        """Extract BGR template from pattern (stripping alpha if present)."""
        if pattern.pixel_data is None:
            raise ImageProcessingError(f"Pattern '{pattern.name}' has no pixel data")

        template = pattern.pixel_data
        if len(template.shape) == 3 and template.shape[2] == 4:
            return template[:, :, :3]
        return template

    def _convert_to_opencv(self, image: Any) -> np.ndarray:
        """Convert various image formats to OpenCV BGR."""
        if hasattr(image, "get_mat_bgr"):
            result = image.get_mat_bgr()
            if result is None:
                raise ImageProcessingError("get_mat_bgr() returned None")
            return result

        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 4:
                return image[:, :, :3]
            return image

        try:
            from PIL import Image as PILImage

            if isinstance(image, PILImage.Image):
                rgb_array = np.array(image)
                return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        except ImportError as e:
            raise ImageProcessingError("PIL not available for image conversion") from e

        raise ImageProcessingError(
            f"Unsupported image type: {type(image)}. "
            "Expected PIL Image, numpy array, or object with get_mat_bgr()"
        )

    def _apply_search_region(
        self,
        screenshot: np.ndarray,
        search_region: tuple[int, int, int, int] | None,
    ) -> tuple[np.ndarray, int, int]:
        """Apply search region cropping."""
        if search_region is None:
            return screenshot, 0, 0

        x, y, width, height = search_region
        x = max(0, min(x, screenshot.shape[1]))
        y = max(0, min(y, screenshot.shape[0]))
        width = min(width, screenshot.shape[1] - x)
        height = min(height, screenshot.shape[0] - y)

        return screenshot[y : y + height, x : x + width], x, y
