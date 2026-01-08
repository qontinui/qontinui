"""Layout Analyzer for GUI Environment Discovery.

Extracts layout information from screenshots using:
- Canny edge detection for region boundaries
- Hough transform for alignment guides
- Flood-fill for contiguous color regions
- Content clustering for logical regions
- Grid detection from element alignment patterns
"""

import logging
import uuid
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.environment import (
    AlignmentGuide,
    BoundingBox,
    GridConfiguration,
    Layout,
    LayoutRegion,
    RegionCharacteristics,
    SemanticRegionType,
)

from qontinui.vision.environment.analyzers.base import BaseAnalyzer

logger = logging.getLogger(__name__)


class LayoutAnalyzer(BaseAnalyzer[Layout]):
    """Analyzes screenshots to extract layout information.

    Uses edge detection and content analysis to identify distinct
    regions, grid systems, and alignment guides.
    """

    def __init__(
        self,
        edge_threshold_low: int = 50,
        edge_threshold_high: int = 150,
        min_region_area: int = 5000,
        alignment_threshold: int = 5,
    ) -> None:
        """Initialize the layout analyzer.

        Args:
            edge_threshold_low: Low threshold for Canny edge detection.
            edge_threshold_high: High threshold for Canny edge detection.
            min_region_area: Minimum region area in pixels to consider.
            alignment_threshold: Pixel tolerance for alignment detection.
        """
        super().__init__("LayoutAnalyzer")
        self.edge_threshold_low = edge_threshold_low
        self.edge_threshold_high = edge_threshold_high
        self.min_region_area = min_region_area
        self.alignment_threshold = alignment_threshold

    async def analyze(
        self,
        screenshots: list[NDArray[np.uint8]],
        ocr_results: list[list[dict[str, Any]]] | None = None,
        **kwargs: Any,
    ) -> Layout:
        """Analyze screenshots to extract layout information.

        Args:
            screenshots: List of screenshots as numpy arrays (BGR format).
            ocr_results: Optional OCR results for semantic labeling.

        Returns:
            Layout with extracted information.
        """
        self.reset()

        if not screenshots:
            return Layout(confidence=0.0)

        self._log_progress(f"Analyzing {len(screenshots)} screenshots")

        # Get screen resolution from first screenshot
        first_screenshot = self._ensure_bgr(screenshots[0])
        h, w = first_screenshot.shape[:2]
        screen_resolution = (w, h)

        # Detect regions across all screenshots
        all_regions = []
        for screenshot in screenshots:
            screenshot = self._ensure_bgr(screenshot)
            regions = self._detect_regions(screenshot)
            all_regions.append(regions)

        # Merge and score regions by stability
        merged_regions = self._merge_regions(all_regions, len(screenshots))

        # Add semantic labels
        if ocr_results:
            merged_regions = self._add_semantic_labels(
                merged_regions, ocr_results, screen_resolution
            )

        # Detect grid system
        grid = self._detect_grid(merged_regions, screen_resolution)

        # Detect alignment guides
        alignment_guides = self._detect_alignment_guides(screenshots)

        self._screenshots_analyzed = len(screenshots)
        self.confidence = self._calculate_confidence(
            len(screenshots),
            min_samples=2,
            optimal_samples=10,
        )

        return Layout(
            regions={r.id: r for r in merged_regions},
            grid=grid,
            alignment_guides=alignment_guides,
            screen_resolution=screen_resolution,
            screenshots_analyzed=len(screenshots),
            confidence=self.confidence,
        )

    def _detect_regions(
        self,
        screenshot: NDArray[np.uint8],
    ) -> list[dict[str, Any]]:
        """Detect distinct regions in a screenshot.

        Args:
            screenshot: BGR screenshot.

        Returns:
            List of detected region dictionaries.
        """
        regions = []

        try:
            import cv2

            h, w = screenshot.shape[:2]

            # Convert to grayscale
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

            # Edge detection
            edges = cv2.Canny(gray, self.edge_threshold_low, self.edge_threshold_high)

            # Dilate edges to close gaps
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_region_area:
                    continue

                # Get bounding rectangle
                x, y, rw, rh = cv2.boundingRect(contour)

                # Extract region
                region_img = screenshot[y : y + rh, x : x + rw]

                # Get characteristics
                characteristics = self._analyze_region_characteristics(
                    region_img, screenshot, (x, y, rw, rh)
                )

                regions.append(
                    {
                        "bounds": (x, y, rw, rh),
                        "area": area,
                        "characteristics": characteristics,
                    }
                )

            # Also detect large color regions
            color_regions = self._detect_color_regions(screenshot)
            regions.extend(color_regions)

        except ImportError:
            logger.warning("OpenCV not available, using simple region detection")
            regions = self._simple_region_detection(screenshot)

        return regions

    def _detect_color_regions(
        self,
        screenshot: NDArray[np.uint8],
    ) -> list[dict[str, Any]]:
        """Detect large contiguous color regions.

        Args:
            screenshot: BGR screenshot.

        Returns:
            List of color region dictionaries.
        """
        regions = []

        try:
            import cv2

            h, w = screenshot.shape[:2]

            # Quantize colors to reduce noise
            quantized = (screenshot // 32) * 32

            # Find unique colors
            flat = quantized.reshape(-1, 3)
            unique_colors = np.unique(flat, axis=0)

            for color in unique_colors:
                # Create mask for this color
                mask = np.all(quantized == color, axis=2).astype(np.uint8) * 255

                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < self.min_region_area * 2:  # Higher threshold for color regions
                        continue

                    x, y, rw, rh = cv2.boundingRect(contour)

                    # Skip very small regions
                    if rw < 50 or rh < 50:
                        continue

                    b, g, r = color
                    bg_color = self._rgb_to_hex(int(r), int(g), int(b))

                    regions.append(
                        {
                            "bounds": (x, y, rw, rh),
                            "area": area,
                            "characteristics": {
                                "background_color": bg_color,
                                "is_color_region": True,
                            },
                        }
                    )

        except ImportError:
            pass

        return regions

    def _simple_region_detection(
        self,
        screenshot: NDArray[np.uint8],
    ) -> list[dict[str, Any]]:
        """Simple region detection without OpenCV.

        Args:
            screenshot: BGR screenshot.

        Returns:
            List of basic region dictionaries.
        """
        h, w = screenshot.shape[:2]

        # Simple approach: divide into standard regions
        regions = [
            {"bounds": (0, 0, w, int(h * 0.08)), "area": w * int(h * 0.08)},  # Header
            {"bounds": (0, 0, int(w * 0.2), h), "area": int(w * 0.2) * h},  # Sidebar
            {
                "bounds": (int(w * 0.2), int(h * 0.08), int(w * 0.8), int(h * 0.84)),
                "area": int(w * 0.8) * int(h * 0.84),
            },  # Main
        ]

        for region in regions:
            x, y, rw, rh = region["bounds"]
            region_img = screenshot[y : y + rh, x : x + rw]
            region["characteristics"] = self._analyze_region_characteristics(
                region_img, screenshot, region["bounds"]
            )

        return regions

    def _analyze_region_characteristics(
        self,
        region_img: NDArray[np.uint8],
        full_screenshot: NDArray[np.uint8],
        bounds: tuple[int, int, int, int],
    ) -> dict[str, Any]:
        """Analyze visual characteristics of a region.

        Args:
            region_img: Cropped region image.
            full_screenshot: Full screenshot for context.
            bounds: Region bounds (x, y, width, height).

        Returns:
            Dictionary of characteristics.
        """
        characteristics: dict[str, Any] = {}

        if region_img.size == 0:
            return characteristics

        # Get dominant background color
        flat = region_img.reshape(-1, 3)
        # Simple mode approximation
        quantized = (flat // 32) * 32
        from collections import Counter

        counts = Counter(map(tuple, quantized))
        if counts:
            most_common = counts.most_common(1)[0][0]
            b, g, r = most_common
            characteristics["background_color"] = self._rgb_to_hex(int(r), int(g), int(b))

        # Check for vertical/horizontal lists (simplified)
        # This would need proper element detection for accuracy
        characteristics["has_vertical_list"] = False
        characteristics["has_horizontal_list"] = False

        return characteristics

    def _merge_regions(
        self,
        all_regions: list[list[dict[str, Any]]],
        n_screenshots: int,
    ) -> list[LayoutRegion]:
        """Merge regions from multiple screenshots and calculate stability.

        Args:
            all_regions: Regions detected in each screenshot.
            n_screenshots: Total number of screenshots.

        Returns:
            List of merged LayoutRegion objects.
        """
        # Simple approach: use regions from first screenshot
        # and check stability across others
        if not all_regions or not all_regions[0]:
            return []

        merged = []

        for region_dict in all_regions[0]:
            bounds = region_dict["bounds"]
            x, y, w, h = bounds

            # Count how many screenshots have a similar region
            match_count = 1  # First screenshot
            for other_regions in all_regions[1:]:
                for other in other_regions:
                    ox, oy, ow, oh = other["bounds"]
                    # Check for overlap (simplified IoU)
                    if self._regions_similar(bounds, other["bounds"]):
                        match_count += 1
                        break

            stability = match_count / n_screenshots

            characteristics = region_dict.get("characteristics", {})

            merged.append(
                LayoutRegion(
                    id=f"region_{uuid.uuid4().hex[:8]}",
                    bounds=BoundingBox(x=x, y=y, width=w, height=h),
                    characteristics=RegionCharacteristics(
                        background_color=characteristics.get("background_color"),
                        has_vertical_list=characteristics.get("has_vertical_list", False),
                        has_horizontal_list=characteristics.get("has_horizontal_list", False),
                        content_varies=stability < 0.5,
                    ),
                    semantic_label=SemanticRegionType.UNKNOWN,
                    stability=stability,
                )
            )

        return merged

    def _regions_similar(
        self,
        bounds1: tuple[int, int, int, int],
        bounds2: tuple[int, int, int, int],
        tolerance: float = 0.2,
    ) -> bool:
        """Check if two regions are similar (high IoU).

        Args:
            bounds1: First region (x, y, width, height).
            bounds2: Second region (x, y, width, height).
            tolerance: Position tolerance as fraction of size.

        Returns:
            True if regions are similar.
        """
        x1, y1, w1, h1 = bounds1
        x2, y2, w2, h2 = bounds2

        # Check position similarity
        pos_tol = max(w1, h1) * tolerance
        if abs(x1 - x2) > pos_tol or abs(y1 - y2) > pos_tol:
            return False

        # Check size similarity
        if abs(w1 - w2) / max(w1, w2) > tolerance:
            return False
        if abs(h1 - h2) / max(h1, h2) > tolerance:
            return False

        return True

    def _add_semantic_labels(
        self,
        regions: list[LayoutRegion],
        ocr_results: list[list[dict[str, Any]]],
        screen_resolution: tuple[int, int],
    ) -> list[LayoutRegion]:
        """Add semantic labels to regions based on position and content.

        Args:
            regions: Layout regions.
            ocr_results: OCR results for content analysis.
            screen_resolution: (width, height).

        Returns:
            Regions with semantic labels.
        """
        w, h = screen_resolution

        for region in regions:
            bounds = region.bounds
            rx, ry = bounds.x, bounds.y
            rw, rh = bounds.width, bounds.height

            # Position-based heuristics
            if ry < h * 0.1 and rw > w * 0.5:
                region.semantic_label = SemanticRegionType.HEADER
            elif ry > h * 0.9:
                region.semantic_label = SemanticRegionType.FOOTER
            elif rx < w * 0.25 and rh > h * 0.5:
                region.semantic_label = SemanticRegionType.SIDEBAR
            elif rx > w * 0.2 and ry > h * 0.1 and rh > h * 0.5:
                region.semantic_label = SemanticRegionType.MAIN_CONTENT

            # Check OCR content for navigation keywords
            if ocr_results:
                text_in_region = self._get_text_in_region(
                    bounds, ocr_results[0] if ocr_results else []
                )
                nav_keywords = ["home", "menu", "settings", "profile", "dashboard"]
                if any(kw in text_in_region.lower() for kw in nav_keywords):
                    if region.semantic_label == SemanticRegionType.UNKNOWN:
                        region.semantic_label = SemanticRegionType.NAVIGATION

        return regions

    def _get_text_in_region(
        self,
        bounds: BoundingBox,
        ocr_result: list[dict[str, Any]],
    ) -> str:
        """Get text content within a region.

        Args:
            bounds: Region bounds.
            ocr_result: OCR results.

        Returns:
            Concatenated text in region.
        """
        texts = []

        for item in ocr_result:
            bbox = item.get("bbox")
            if not bbox or not item.get("text"):
                continue

            # Get text position
            if len(bbox) == 4 and not isinstance(bbox[0], (list, tuple)):
                tx, ty, tw, th = bbox
            else:
                continue

            # Check if text is inside region
            if (
                bounds.x <= tx < bounds.x + bounds.width
                and bounds.y <= ty < bounds.y + bounds.height
            ):
                texts.append(item["text"])

        return " ".join(texts)

    def _detect_grid(
        self,
        regions: list[LayoutRegion],
        screen_resolution: tuple[int, int],
    ) -> GridConfiguration:
        """Detect grid system from region alignments.

        Args:
            regions: Detected layout regions.
            screen_resolution: (width, height).

        Returns:
            GridConfiguration.
        """
        w, h = screen_resolution

        # Collect x positions of region edges
        x_positions = []
        for region in regions:
            x_positions.append(region.bounds.x)
            x_positions.append(region.bounds.x + region.bounds.width)

        if len(x_positions) < 4:
            return GridConfiguration(detected=False)

        # Find common gutters (gaps between positions)
        x_positions = sorted(set(x_positions))
        gutters = [x_positions[i + 1] - x_positions[i] for i in range(len(x_positions) - 1)]

        if not gutters:
            return GridConfiguration(detected=False)

        # Check if gutters are consistent (indicating a grid)
        from collections import Counter

        gutter_counts = Counter(round(g / 4) * 4 for g in gutters if g > 5)  # Round to 4px

        if not gutter_counts:
            return GridConfiguration(detected=False)

        most_common_gutter = gutter_counts.most_common(1)[0][0]

        # Estimate number of columns
        content_width = w - (x_positions[0] + (w - x_positions[-1]))
        if most_common_gutter > 0:
            estimated_columns = max(1, int(content_width / (content_width / 12)))
        else:
            estimated_columns = 12

        return GridConfiguration(
            detected=True,
            columns=min(24, max(1, estimated_columns)),
            gutter=most_common_gutter if most_common_gutter > 0 else None,
            margin=x_positions[0] if x_positions else None,
        )

    def _detect_alignment_guides(
        self,
        screenshots: list[NDArray[np.uint8]],
    ) -> list[AlignmentGuide]:
        """Detect alignment guides using Hough transform.

        Args:
            screenshots: List of screenshots.

        Returns:
            List of AlignmentGuide objects.
        """
        guides = []

        try:
            import cv2

            # Use first screenshot
            screenshot = self._ensure_bgr(screenshots[0])
            h, w = screenshot.shape[:2]

            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Detect lines with Hough transform
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=100,
                minLineLength=min(w, h) * 0.3,
                maxLineGap=10,
            )

            if lines is None:
                return guides

            vertical_lines = []
            horizontal_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Check if vertical
                if abs(x1 - x2) < self.alignment_threshold:
                    vertical_lines.append((x1 + x2) // 2)
                # Check if horizontal
                elif abs(y1 - y2) < self.alignment_threshold:
                    horizontal_lines.append((y1 + y2) // 2)

            # Cluster and deduplicate lines
            from collections import Counter

            v_counts = Counter(round(x / 10) * 10 for x in vertical_lines)
            h_counts = Counter(round(y / 10) * 10 for y in horizontal_lines)

            # Add top vertical guides
            for x, count in v_counts.most_common(5):
                confidence = min(1.0, count / 10)
                if confidence > 0.3:
                    guides.append(
                        AlignmentGuide(
                            type="vertical",
                            position=x,
                            confidence=confidence,
                        )
                    )

            # Add top horizontal guides
            for y, count in h_counts.most_common(5):
                confidence = min(1.0, count / 10)
                if confidence > 0.3:
                    guides.append(
                        AlignmentGuide(
                            type="horizontal",
                            position=y,
                            confidence=confidence,
                        )
                    )

        except ImportError:
            logger.debug("OpenCV not available for Hough transform")

        return guides
