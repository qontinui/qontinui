"""Batch multi-template matching using Multi-Template-Matching library.

Matches multiple templates against a single screenshot in one pass,
with cross-template Non-Maximum Suppression (NMS) to deduplicate
overlapping detections from different templates.

This is more efficient than sequential single-template matching when
searching for multiple patterns: one screenshot capture, one NMS pass,
and parallelized template search via MTM's ThreadPoolExecutor.

Multi-scale support: ``find_all_patterns_multiscale`` generates scaled
variants of each template and feeds them to MTM in a single batch,
then deduplicates cross-scale results via IoU-based NMS.
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
            import MTM  # noqa: F401 — availability check
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

    # ------------------------------------------------------------------
    # Multi-scale batch matching
    # ------------------------------------------------------------------

    def find_all_patterns_multiscale(
        self,
        screenshot: Any,
        patterns: list[Pattern],
        similarity: float = 0.8,
        search_region: tuple[int, int, int, int] | None = None,
        scales: list[float] | None = None,
    ) -> dict[str, list[Match]]:
        """Find all patterns at multiple scales in a single screenshot.

        Generates scaled variants of each template, feeds the full set to
        MTM in one batch, then merges cross-scale results back to the
        original pattern names with IoU-based NMS deduplication.

        Args:
            screenshot: Screenshot image (PIL Image, numpy array, or OpenCV mat).
            patterns: List of patterns to search for.
            similarity: Minimum similarity threshold (0.0-1.0).
            search_region: Optional (x, y, width, height) to restrict search area.
            scales: Scale factors to try. If None, uses DPI-aware defaults
                    from ``OpenCVMatcher.dpi_aware_scales()``.

        Returns:
            Dict mapping pattern.name to list of Match objects (best first).
        """
        if scales is None:
            scales = self._get_dpi_aware_scales()

        if not patterns:
            return {}

        try:
            import MTM
        except ImportError as e:
            raise ImageProcessingError(
                "Multi-Template-Matching not installed. "
                "Install with: pip install Multi-Template-Matching"
            ) from e

        screenshot_bgr = self._convert_to_opencv(screenshot)
        search_img, offset_x, offset_y = self._apply_search_region(screenshot_bgr, search_region)
        img_h, img_w = search_img.shape[:2]

        # Separate masked vs unmasked
        unmasked_patterns: list[Pattern] = []
        masked_patterns: list[Pattern] = []
        for p in patterns:
            if self._has_active_mask(p):
                masked_patterns.append(p)
            else:
                unmasked_patterns.append(p)

        results: dict[str, list[Match]] = {p.name: [] for p in patterns}

        # Build scaled template list for all unmasked patterns
        if unmasked_patterns:
            # Label format: "patternName@scale" to trace back after matching
            list_templates: list[tuple[str, np.ndarray]] = []
            for p in unmasked_patterns:
                template_bgr = self._get_template_bgr(p)
                for scale in scales:
                    scaled = self._resize_template(template_bgr, scale)
                    if scaled is None:
                        continue  # Too small or too large
                    # Skip if scaled template is bigger than the search image
                    sh, sw = scaled.shape[:2]
                    if sh > img_h or sw > img_w:
                        continue
                    label = f"{p.name}@{scale}"
                    list_templates.append((label, scaled))

            if list_templates:
                method_id = self._METHOD_MAP[self.method]

                # Run MTM on the full set of scaled templates
                # Use a generous maxOverlap here — we do our own cross-scale NMS
                hits = MTM.matchTemplates(
                    list_templates,
                    search_img,
                    score_threshold=similarity,
                    method=method_id,
                    maxOverlap=0.8,  # Loose — tighten in cross-scale NMS
                )

                # Group hits by original pattern name
                for hit in hits:
                    label, bbox, score = hit
                    # Parse "patternName@scale"
                    original_name = label.rsplit("@", 1)[0]
                    x, y, w, h = bbox

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
                        name=original_name,
                    )
                    results[original_name].append(Match(match_obj))

                # Cross-scale NMS per pattern
                for name in list(results.keys()):
                    if len(results[name]) > 1:
                        results[name] = self._nms_matches(results[name], self.nms_overlap_threshold)

        # Sequential fallback for masked patterns (no multi-scale via MTM)
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

        # Sort each pattern's matches by score descending
        for name in results:
            results[name].sort(key=lambda m: m.similarity, reverse=True)

        return results

    @staticmethod
    def _resize_template(
        template: np.ndarray,
        scale: float,
        min_size: int = 10,
    ) -> np.ndarray | None:
        """Resize a template by a scale factor.

        Args:
            template: BGR template image.
            scale: Scale factor (1.0 = original size).
            min_size: Minimum dimension in pixels. Returns None if too small.

        Returns:
            Resized template, or None if the result would be too small.
        """
        if scale == 1.0:
            return template

        h, w = template.shape[:2]
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        if new_w < min_size or new_h < min_size:
            return None

        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(template, (new_w, new_h), interpolation=interpolation)

    @staticmethod
    def _get_dpi_aware_scales() -> list[float]:
        """Get DPI-aware scale list, reusing HAL logic when available."""
        try:
            from ...hal.implementations.opencv_matcher import OpenCVMatcher

            return OpenCVMatcher.dpi_aware_scales()
        except ImportError:
            pass

        # Fallback: try basic Windows DPI detection
        import sys

        if sys.platform == "win32":
            try:
                import ctypes

                ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
                dpi = ctypes.windll.user32.GetDpiForSystem()  # type: ignore[attr-defined]
                dpi_scale = round(dpi / 96.0, 3)
                scales = {1.0}
                if dpi_scale != 1.0:
                    scales.add(round(1.0 / dpi_scale, 3))
                    scales.add(dpi_scale)
                return sorted(scales)
            except Exception:
                pass

        return [1.0]  # No DPI info — native scale only

    @staticmethod
    def _nms_matches(matches: list[Match], overlap_threshold: float) -> list[Match]:
        """IoU-based Non-Maximum Suppression across matches.

        Keeps the highest-confidence match and removes overlapping lower-
        confidence matches that exceed the IoU threshold.
        """
        if len(matches) <= 1:
            return matches

        sorted_matches = sorted(matches, key=lambda m: m.similarity, reverse=True)
        kept: list[Match] = []

        for match in sorted_matches:
            region = match.get_region()
            if region is None:
                continue

            overlaps = False
            for k in kept:
                kr = k.get_region()
                if kr is None:
                    continue

                # Calculate IoU
                ix1 = max(region.x, kr.x)
                iy1 = max(region.y, kr.y)
                ix2 = min(region.x + region.width, kr.x + kr.width)
                iy2 = min(region.y + region.height, kr.y + kr.height)

                if ix2 > ix1 and iy2 > iy1:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    area1 = region.width * region.height
                    area2 = kr.width * kr.height
                    iou = intersection / (area1 + area2 - intersection)
                    if iou > overlap_threshold:
                        overlaps = True
                        break

            if not overlaps:
                kept.append(match)

        return kept
