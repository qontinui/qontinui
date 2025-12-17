"""OpenCV template matching implementation.

Provides template-based image matching using various OpenCV algorithms.
Supports both masked and unmasked template matching with automatic
method selection based on mask presence.
"""

import math
from typing import Any

import cv2
import numpy as np

from ...exceptions import ImageProcessingError
from ...model.element import Location, Pattern, Region
from ...model.match import Match as MatchObject
from ..match import Match
from .image_matcher import ImageMatcher


class TemplateMatcher(ImageMatcher):
    """OpenCV template matching implementation.

    Uses OpenCV's matchTemplate function with support for:
    - Multiple matching methods (TM_CCOEFF_NORMED, TM_SQDIFF, etc.)
    - Masked template matching (transparent regions)
    - Multi-scale matching (optional)
    - Non-maximum suppression for multiple matches

    Attributes:
        method: OpenCV matching method name (e.g., "TM_CCOEFF_NORMED")
        nms_overlap_threshold: IoU threshold for non-maximum suppression (0.0-1.0)
    """

    # Available OpenCV template matching methods
    METHODS = {
        "TM_CCOEFF": cv2.TM_CCOEFF,
        "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
        "TM_CCORR": cv2.TM_CCORR,
        "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
        "TM_SQDIFF": cv2.TM_SQDIFF,
        "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
    }

    def __init__(
        self,
        method: str = "TM_CCOEFF_NORMED",
        nms_overlap_threshold: float = 0.3,
    ) -> None:
        """Initialize template matcher.

        Args:
            method: OpenCV matching method name. Default is TM_CCOEFF_NORMED (best quality).
                   For masked matching, TM_SQDIFF will be used automatically.
            nms_overlap_threshold: Overlap threshold for non-maximum suppression.
                                  Higher values keep more overlapping matches.
                                  Range: 0.0 (no overlap) to 1.0 (full overlap).

        Raises:
            ValueError: If method name is not recognized
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.METHODS.keys())}")
        self.method = method
        self.nms_overlap_threshold = nms_overlap_threshold

    def find_matches(
        self,
        screenshot: Any,
        pattern: Pattern,
        find_all: bool = False,
        similarity: float = 0.8,
        search_region: tuple[int, int, int, int] | None = None,
    ) -> list[Match]:
        """Find template matches using OpenCV matchTemplate.

        Automatically selects the best matching method:
        - Masked templates: Uses TM_SQDIFF (only non-normalized method that works)
        - Unmasked templates: Uses configured method (default TM_CCOEFF_NORMED)

        Args:
            screenshot: Screenshot image (PIL Image, numpy array, or OpenCV mat)
            pattern: Pattern to search for with pixel data and optional mask
            find_all: If True, find all matches above threshold with NMS
            similarity: Minimum similarity threshold (0.0 to 1.0)
            search_region: Optional region to search in as (x, y, width, height)

        Returns:
            List of Match objects sorted by similarity (highest first)

        Raises:
            ImageProcessingError: If image conversion or matching fails
        """
        matches, _ = self.find_matches_with_debug(
            screenshot=screenshot,
            pattern=pattern,
            find_all=find_all,
            similarity=similarity,
            search_region=search_region,
            collect_debug=False,
        )
        return matches

    def find_matches_with_debug(
        self,
        screenshot: Any,
        pattern: Pattern,
        find_all: bool = False,
        similarity: float = 0.8,
        search_region: tuple[int, int, int, int] | None = None,
        collect_debug: bool = False,
    ) -> tuple[list[Match], dict[str, Any] | None]:
        """Find template matches with optional debug data collection.

        Automatically selects the best matching method:
        - Masked templates: Uses TM_SQDIFF (only non-normalized method that works)
        - Unmasked templates: Uses configured method (default TM_CCOEFF_NORMED)

        Args:
            screenshot: Screenshot image (PIL Image, numpy array, or OpenCV mat)
            pattern: Pattern to search for with pixel data and optional mask
            find_all: If True, find all matches above threshold with NMS
            similarity: Minimum similarity threshold (0.0 to 1.0)
            search_region: Optional region to search in as (x, y, width, height)
            collect_debug: If True, collect and return debug data

        Returns:
            Tuple of (matches, debug_data):
            - matches: List of Match objects sorted by similarity (highest first)
            - debug_data: Debug information dict or None if collect_debug=False

        Raises:
            ImageProcessingError: If image conversion or matching fails
        """
        try:
            # Convert screenshot to OpenCV format
            screenshot_bgr = self._convert_to_opencv(screenshot)

            # Extract template and mask from pattern
            template_bgr, mask = self._extract_template_and_mask(pattern)

            # Apply search region if specified
            search_img, offset_x, offset_y = self._apply_search_region(
                screenshot_bgr, search_region
            )

            # Perform template matching
            result = self._match_template(search_img, template_bgr, mask)

            # Determine which method was used
            has_mask = mask is not None
            method_used = cv2.TM_SQDIFF if has_mask else self.METHODS[self.method]

            # Collect debug data if requested
            debug_data = None
            if collect_debug:
                debug_data = self._collect_debug_data(
                    result=result,
                    threshold=similarity,
                    method=method_used,
                    template_shape=template_bgr.shape,
                    has_mask=has_mask,
                    top_n=5,
                )

            # Extract matches from result
            matches = self._extract_matches(
                result=result,
                template_shape=template_bgr.shape,  # type: ignore[arg-type]
                pattern=pattern,
                offset=(offset_x, offset_y),
                find_all=find_all,
                similarity=similarity,
            )

            # DEBUG: Log how many matches were extracted
            import sys

            print(
                f"[MATCH_DEBUG] Extracted {len(matches)} matches for pattern {pattern.name}",
                file=sys.stderr,
                flush=True,
            )
            if len(matches) == 0:
                print(
                    "[MATCH_DEBUG] WARNING: No matches extracted but debug_data might have matches",
                    file=sys.stderr,
                    flush=True,
                )
                if debug_data and "top_matches" in debug_data:
                    print(
                        f"[MATCH_DEBUG] debug_data has {len(debug_data['top_matches'])} top_matches",
                        file=sys.stderr,
                        flush=True,
                    )

            return matches, debug_data

        except cv2.error as e:
            raise ImageProcessingError(f"OpenCV error during matching: {e}") from e
        except (ValueError, TypeError) as e:
            raise ImageProcessingError(f"Image format error: {e}") from e

    def _convert_to_opencv(self, image: Any) -> np.ndarray:
        """Convert various image formats to OpenCV BGR format.

        Args:
            image: PIL Image, numpy array, or object with get_mat_bgr() method

        Returns:
            OpenCV BGR image as numpy array

        Raises:
            ImageProcessingError: If conversion fails or format is unsupported
        """
        # Already OpenCV format with get_mat_bgr method
        if hasattr(image, "get_mat_bgr"):
            result = image.get_mat_bgr()
            if result is None:
                raise ImageProcessingError("get_mat_bgr() returned None")
            return result  # type: ignore[no-any-return]

        # Already numpy array
        if isinstance(image, np.ndarray):
            # Ensure it's BGR (remove alpha channel if present)
            if len(image.shape) == 3 and image.shape[2] == 4:
                return image[:, :, :3]
            return image

        # Try PIL Image conversion
        try:
            from PIL import Image as PILImage

            if isinstance(image, PILImage.Image):
                # Convert PIL Image to numpy array (RGB)
                rgb_array = np.array(image)
                # Convert RGB to BGR for OpenCV
                return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        except ImportError as e:
            raise ImageProcessingError("PIL not available for image conversion") from e
        except (ValueError, TypeError) as e:
            raise ImageProcessingError(f"Failed to convert PIL Image: {e}") from e

        raise ImageProcessingError(
            f"Unsupported image type: {type(image)}. "
            "Expected PIL Image, numpy array, or object with get_mat_bgr()"
        )

    def _extract_template_and_mask(self, pattern: Pattern) -> tuple[np.ndarray, np.ndarray | None]:
        """Extract template image and mask from pattern.

        Args:
            pattern: Pattern object with pixel_data and optional mask

        Returns:
            Tuple of (template_bgr, mask) where:
            - template_bgr is BGR image without alpha channel
            - mask is uint8 mask (0-255) or None if no mask is active

        Raises:
            ImageProcessingError: If pattern has no pixel data
        """
        if pattern.pixel_data is None:
            raise ImageProcessingError("Pattern has no pixel data")

        template = pattern.pixel_data

        # Extract mask (priority: pattern mask > alpha channel > None)
        mask = None
        pattern_mask = pattern.mask

        if pattern_mask is not None:
            # Check if mask is all ones (equivalent to no mask)
            # Using a mask of all ones with normalized methods can cause NaN values
            if not (np.all(pattern_mask == 1.0) or np.all(pattern_mask == 255)):
                # Convert mask to uint8 format (0-255) if it's in float format (0.0-1.0)
                if pattern_mask.dtype in (np.float32, np.float64):
                    mask = (pattern_mask * 255).astype(np.uint8)
                else:
                    mask = pattern_mask.astype(np.uint8)

        # Fall back to alpha channel if present and no pattern mask
        if mask is None and len(template.shape) == 3 and template.shape[2] == 4:
            alpha = template[:, :, 3]
            # Only use alpha if it's not all opaque
            if not np.all(alpha == 255):
                mask = alpha.copy()

        # Prepare template (remove alpha channel if present)
        if len(template.shape) == 3 and template.shape[2] == 4:
            template_bgr = template[:, :, :3]
        else:
            template_bgr = template

        return template_bgr, mask

    def _apply_search_region(
        self,
        screenshot: np.ndarray,
        search_region: tuple[int, int, int, int] | None,
    ) -> tuple[np.ndarray, int, int]:
        """Apply search region to screenshot if specified.

        Args:
            screenshot: Full screenshot image
            search_region: Optional (x, y, width, height) tuple

        Returns:
            Tuple of (cropped_image, offset_x, offset_y)
            If no region specified, returns full image with (0, 0) offset
        """
        if search_region is None:
            return screenshot, 0, 0

        x, y, width, height = search_region

        # Ensure region is within bounds
        x = max(0, min(x, screenshot.shape[1]))
        y = max(0, min(y, screenshot.shape[0]))
        width = min(width, screenshot.shape[1] - x)
        height = min(height, screenshot.shape[0] - y)

        # Crop image to region
        cropped = screenshot[y : y + height, x : x + width]

        return cropped, x, y

    def _match_template(
        self,
        search_img: np.ndarray,
        template: np.ndarray,
        mask: np.ndarray | None,
    ) -> np.ndarray:
        """Perform OpenCV template matching.

        Automatically selects matching method based on mask presence:
        - With mask: Uses TM_SQDIFF (only non-normalized method that works reliably)
        - Without mask: Uses configured method (default TM_CCOEFF_NORMED)

        Args:
            search_img: Image to search in (BGR, no alpha)
            template: Template image (BGR, no alpha)
            mask: Optional mask (uint8, 0-255)

        Returns:
            Match result array where higher values indicate better matches
            (normalized to 0-1 range where 1.0 is perfect match)

        Raises:
            cv2.error: If OpenCV matching fails
        """
        if mask is not None:
            # Use squared difference for masked matching (only reliable method)
            # NOTE: Normalized methods (TM_*_NORMED) produce NaN with masks in OpenCV 4.x
            result = cv2.matchTemplate(search_img, template, cv2.TM_SQDIFF, mask=mask)

            # Convert TM_SQDIFF to similarity (lower diff = higher similarity)
            # Normalize: similarity = 1 / (1 + sqdiff / max_possible_diff)
            # This maps: sqdiff=0 -> 1.0, sqdiff=infinity -> 0.0
            max_diff = 255.0 * 255.0 * template.shape[0] * template.shape[1] * 3
            result = 1.0 / (1.0 + result / max_diff)
        else:
            # Use configured method for unmasked matching (default TM_CCOEFF_NORMED)
            method_flag = self.METHODS[self.method]
            result = cv2.matchTemplate(search_img, template, method_flag)

            # Normalize result to 0-1 range if using non-normalized method
            if not self.method.endswith("_NORMED"):
                min_val, max_val = result.min(), result.max()
                if max_val > min_val:
                    result = (result - min_val) / (max_val - min_val)

        return result

    def _extract_matches(
        self,
        result: np.ndarray,
        template_shape: tuple[int, int, int],
        pattern: Pattern,
        offset: tuple[int, int],
        find_all: bool,
        similarity: float,
    ) -> list[Match]:
        """Extract match objects from OpenCV result array.

        Args:
            result: OpenCV matchTemplate result array (0-1 range, higher is better)
            template_shape: Shape of template (height, width, channels)
            pattern: Original pattern object
            offset: (x, y) offset to add to match coordinates
            find_all: If True, find all matches above threshold
            similarity: Minimum similarity threshold

        Returns:
            List of Match objects sorted by similarity (highest first)
        """
        template_height, template_width = template_shape[:2]
        offset_x, offset_y = offset

        matches_list: list[Match] = []

        if find_all:
            # Find all matches above threshold
            locations = np.where(result >= similarity)

            for pt in zip(*locations[::-1], strict=False):  # Switch x and y
                score = result[pt[1], pt[0]]

                # Skip non-finite scores (NaN, inf)
                if not math.isfinite(float(score)):
                    continue

                x = int(pt[0]) + offset_x
                y = int(pt[1]) + offset_y

                # Calculate center of the match region for target location
                center_x = x + template_width // 2
                center_y = y + template_height // 2

                match_obj = MatchObject(
                    target=Location(
                        x=center_x,
                        y=center_y,
                        region=Region(x, y, template_width, template_height),
                    ),
                    score=float(score),
                    name=pattern.name,
                )
                matches_list.append(Match(match_obj))

            # Apply Non-Maximum Suppression to remove overlapping matches
            if len(matches_list) > 1:
                matches_list = self._apply_nms(matches_list, self.nms_overlap_threshold)

            # Final filter to ensure all matches meet threshold
            matches_list = [m for m in matches_list if m.similarity >= similarity]

        else:
            # Find only best match
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # DEBUG: Log what we found
            import logging

            logger = logging.getLogger(__name__)
            logger.info(
                f"[MATCH_DEBUG] find_all=False: max_val={max_val}, similarity={similarity}, passes={max_val >= similarity}"
            )
            logger.info(
                f"[MATCH_DEBUG] Location: max_loc={max_loc}, offset=({offset_x}, {offset_y})"
            )

            import sys

            print(
                f"[MATCH_DEBUG] find_all=False: max_val={max_val}, similarity={similarity}, passes={max_val >= similarity}",
                file=sys.stderr,
                flush=True,
            )

            if max_val >= similarity:
                x = int(max_loc[0]) + offset_x
                y = int(max_loc[1]) + offset_y

                # Calculate center of the match region for target location
                center_x = x + template_width // 2
                center_y = y + template_height // 2

                match_obj = MatchObject(
                    target=Location(
                        x=center_x,
                        y=center_y,
                        region=Region(x, y, template_width, template_height),
                    ),
                    score=float(max_val),
                    name=pattern.name,
                )
                matches_list.append(Match(match_obj))
                print(
                    f"[MATCH_DEBUG] Created match: region_top_left=({x}, {y}), center=({center_x}, {center_y}), score={float(max_val)}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"[MATCH_DEBUG] NO MATCH CREATED - max_val ({max_val}) < similarity ({similarity})",
                    file=sys.stderr,
                    flush=True,
                )

        # Sort by similarity (highest first)
        matches_list.sort(key=lambda m: m.similarity, reverse=True)

        return matches_list

    def _apply_nms(self, matches: list[Match], overlap_threshold: float) -> list[Match]:
        """Apply Non-Maximum Suppression to remove overlapping matches.

        Uses IoU (Intersection over Union) to determine overlap.
        Keeps matches with highest similarity scores.

        Args:
            matches: List of matches to filter
            overlap_threshold: IoU threshold for considering matches as overlapping
                             Range: 0.0 (no overlap) to 1.0 (full overlap)

        Returns:
            Filtered list of non-overlapping matches
        """
        if not matches:
            return matches

        # Sort by score descending
        sorted_matches = sorted(matches, key=lambda m: m.similarity, reverse=True)

        kept_matches: list[Match] = []
        for match in sorted_matches:
            # Skip matches without regions
            if match.region is None:
                continue

            # Check if this match overlaps with any kept match
            should_keep = True
            for kept in kept_matches:
                # Skip kept matches without regions
                if kept.region is None:
                    continue

                # Calculate IoU (Intersection over Union)
                iou = self._calculate_iou(match.region, kept.region)

                if iou > overlap_threshold:
                    should_keep = False
                    break

            if should_keep:
                kept_matches.append(match)

        return kept_matches

    def _calculate_iou(self, region1: Region, region2: Region) -> float:
        """Calculate Intersection over Union (IoU) between two regions.

        Args:
            region1: First region
            region2: Second region

        Returns:
            IoU value between 0.0 (no overlap) and 1.0 (identical)
        """
        # Calculate intersection rectangle
        x1 = max(region1.x, region2.x)
        y1 = max(region1.y, region2.y)
        x2 = min(region1.x + region1.width, region2.x + region2.width)
        y2 = min(region1.y + region1.height, region2.y + region2.height)

        # Check if there's an intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0

        # Calculate areas
        intersection = (x2 - x1) * (y2 - y1)
        area1 = region1.width * region1.height
        area2 = region2.width * region2.height
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _collect_debug_data(
        self,
        result: np.ndarray,
        threshold: float,
        method: int,
        template_shape: tuple,
        has_mask: bool,
        top_n: int = 5,
    ) -> dict[str, Any]:
        """Collect debug data including top N matches.

        Args:
            result: OpenCV matchTemplate result array (0-1 range, higher is better)
            threshold: Similarity threshold used for matching
            method: OpenCV matching method (e.g., cv2.TM_CCOEFF_NORMED)
            template_shape: Shape of template (height, width, channels)
            has_mask: Whether a mask was used for matching
            top_n: Number of top matches to collect (default 5)

        Returns:
            Dictionary containing debug data with structure:
            {
                "threshold": float,
                "top_matches": [{"confidence": float, "location": {"x": int, "y": int}, "rank": int}, ...],
                "match_method": str,
                "template_size": {"width": int, "height": int},
                "has_mask": bool,
                "why_failed": str | None
            }
        """
        template_height, template_width = template_shape[:2]

        # Determine method name
        method_name = "UNKNOWN"
        for name, value in self.METHODS.items():
            if value == method:
                method_name = name
                break
        if has_mask:
            method_name = "TM_SQDIFF (masked)"

        # Flatten the result array and get indices sorted by confidence (descending)
        flat_result = result.flatten()
        # Get indices of top N values
        top_indices = np.argpartition(flat_result, -min(top_n, len(flat_result)))[-top_n:]
        # Sort these indices by their actual values (descending)
        top_indices = top_indices[np.argsort(-flat_result[top_indices])]

        # Convert flat indices to 2D coordinates and collect match info
        top_matches = []
        for rank, flat_idx in enumerate(top_indices, start=1):
            # Convert flat index to 2D coordinates
            y_coord = int(flat_idx // result.shape[1])
            x_coord = int(flat_idx % result.shape[1])
            confidence = float(result[y_coord, x_coord])

            # Only include finite confidence values
            if math.isfinite(confidence):
                top_matches.append(
                    {
                        "confidence": confidence,
                        "location": {"x": x_coord, "y": y_coord},
                        "rank": rank,
                    }
                )

        # Determine why matching failed (if best match is below threshold)
        why_failed = None
        if top_matches and float(top_matches[0]["confidence"]) < threshold:  # type: ignore[arg-type]
            best_confidence = float(top_matches[0]["confidence"])  # type: ignore[arg-type]
            gap = threshold - best_confidence
            why_failed = f"Best match confidence {best_confidence:.4f} is below threshold {threshold:.4f} (gap: {gap:.4f})"

        return {
            "threshold": threshold,
            "top_matches": top_matches,
            "match_method": method_name,
            "template_size": {"width": template_width, "height": template_height},
            "has_mask": has_mask,
            "why_failed": why_failed,
        }
