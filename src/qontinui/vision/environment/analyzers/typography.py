"""Typography Analyzer for GUI Environment Discovery.

Extracts typography information from screenshots using:
- OCR with bounding box extraction for text size clustering
- Character shape analysis for font family classification
- Line spacing measurement from vertically adjacent text
- Semantic size mapping (heading, body, small)
"""

import logging
from collections import defaultdict
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.environment import (
    BoundingBox,
    DetectedFont,
    FontFamily,
    FontWeight,
    TextRegion,
    TextSizes,
    Typography,
)

from qontinui.vision.environment.analyzers.base import BaseAnalyzer

logger = logging.getLogger(__name__)


class TypographyAnalyzer(BaseAnalyzer[Typography]):
    """Analyzes screenshots to extract typography information.

    Uses OCR to detect text, then clusters by size and analyzes
    character shapes for font classification.
    """

    def __init__(
        self,
        min_text_height: int = 8,
        max_text_height: int = 100,
        size_cluster_tolerance: int = 2,
    ) -> None:
        """Initialize the typography analyzer.

        Args:
            min_text_height: Minimum text height to consider (pixels).
            max_text_height: Maximum text height to consider (pixels).
            size_cluster_tolerance: Tolerance for clustering text sizes (pixels).
        """
        super().__init__("TypographyAnalyzer")
        self.min_text_height = min_text_height
        self.max_text_height = max_text_height
        self.size_cluster_tolerance = size_cluster_tolerance

    async def analyze(
        self,
        screenshots: list[NDArray[np.uint8]],
        ocr_results: list[list[dict[str, Any]]] | None = None,
        **kwargs: Any,
    ) -> Typography:
        """Analyze screenshots to extract typography information.

        Args:
            screenshots: List of screenshots as numpy arrays (BGR format).
            ocr_results: Optional pre-computed OCR results. If not provided,
                OCR will be performed on each screenshot.

        Returns:
            Typography with extracted information.
        """
        self.reset()

        if not screenshots:
            return Typography(confidence=0.0)

        self._log_progress(f"Analyzing {len(screenshots)} screenshots")

        # Perform OCR if results not provided
        if ocr_results is None:
            ocr_results = await self._perform_ocr(screenshots)

        # Collect all text detections
        all_detections = self._collect_detections(screenshots, ocr_results)

        if not all_detections:
            return Typography(
                screenshots_analyzed=len(screenshots),
                confidence=0.0,
            )

        # Cluster text by size
        size_clusters = self._cluster_by_size(all_detections)

        # Map sizes to semantic names
        text_sizes = self._map_semantic_sizes(size_clusters)

        # Create detected font samples
        detected_fonts = self._create_font_samples(all_detections, screenshots)

        # Identify common text regions
        shape = screenshots[0].shape[:2]
        text_regions = self._identify_text_regions(
            all_detections, (int(shape[0]), int(shape[1]))
        )

        # Detect languages
        languages = self._detect_languages(all_detections)

        self._screenshots_analyzed = len(screenshots)
        self.confidence = self._calculate_confidence(
            len(all_detections),
            min_samples=10,
            optimal_samples=100,
        )

        return Typography(
            detected_fonts=detected_fonts,
            text_sizes=text_sizes,
            languages_detected=languages,
            common_text_regions=text_regions,
            screenshots_analyzed=len(screenshots),
            confidence=self.confidence,
        )

    async def _perform_ocr(
        self,
        screenshots: list[NDArray[np.uint8]],
    ) -> list[list[dict[str, Any]]]:
        """Perform OCR on screenshots.

        Args:
            screenshots: List of BGR screenshots.

        Returns:
            List of OCR results for each screenshot.
        """
        results = []

        try:
            # Try EasyOCR first
            import easyocr

            reader = easyocr.Reader(["en"], gpu=False, verbose=False)

            for screenshot in screenshots:
                screenshot = self._ensure_bgr(screenshot)
                rgb = self._bgr_to_rgb(screenshot)

                ocr_output = reader.readtext(rgb)
                detections = []

                for bbox, text, conf in ocr_output:
                    # EasyOCR bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                    x1 = min(p[0] for p in bbox)
                    y1 = min(p[1] for p in bbox)
                    x2 = max(p[0] for p in bbox)
                    y2 = max(p[1] for p in bbox)

                    detections.append(
                        {
                            "text": text,
                            "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                            "confidence": conf,
                        }
                    )

                results.append(detections)

        except ImportError:
            logger.warning("EasyOCR not available, trying pytesseract")

            try:
                import pytesseract
                from PIL import Image

                for screenshot in screenshots:
                    screenshot = self._ensure_bgr(screenshot)
                    rgb = self._bgr_to_rgb(screenshot)
                    pil_image = Image.fromarray(rgb)

                    # Get detailed output with bounding boxes
                    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)

                    detections = []
                    for i, text in enumerate(data["text"]):
                        if text.strip() and data["conf"][i] > 0:
                            detections.append(
                                {
                                    "text": text,
                                    "bbox": (
                                        data["left"][i],
                                        data["top"][i],
                                        data["width"][i],
                                        data["height"][i],
                                    ),
                                    "confidence": data["conf"][i] / 100.0,
                                }
                            )

                    results.append(detections)

            except ImportError:
                logger.error("No OCR engine available (easyocr or pytesseract)")
                return [[] for _ in screenshots]

        return results

    def _collect_detections(
        self,
        screenshots: list[NDArray[np.uint8]],
        ocr_results: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Collect and filter all text detections.

        Args:
            screenshots: List of screenshots.
            ocr_results: OCR results for each screenshot.

        Returns:
            List of filtered detections with screenshot index.
        """
        detections = []

        for idx, (_screenshot, ocr_result) in enumerate(zip(screenshots, ocr_results, strict=False)):
            for item in ocr_result:
                bbox = item.get("bbox")
                if not bbox:
                    continue

                # Handle different bbox formats
                if len(bbox) == 4:
                    if isinstance(bbox[0], (list, tuple)):
                        # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] format
                        x1 = min(p[0] for p in bbox)
                        y1 = min(p[1] for p in bbox)
                        x2 = max(p[0] for p in bbox)
                        y2 = max(p[1] for p in bbox)
                        height = y2 - y1
                        width = x2 - x1
                    else:
                        # (x, y, width, height) format
                        x, y, width, height = bbox
                else:
                    continue

                # Filter by size
                if height < self.min_text_height or height > self.max_text_height:
                    continue

                detections.append(
                    {
                        "text": item.get("text", ""),
                        "bbox": bbox,
                        "height": int(height),
                        "width": int(width),
                        "confidence": item.get("confidence", 0.0),
                        "screenshot_idx": idx,
                    }
                )

        return detections

    def _cluster_by_size(
        self,
        detections: list[dict[str, Any]],
    ) -> dict[int, list[dict[str, Any]]]:
        """Cluster text detections by height.

        Args:
            detections: List of text detections.

        Returns:
            Dictionary mapping cluster center height to detections.
        """
        if not detections:
            return {}

        # Simple clustering by rounding to tolerance
        clusters: dict[int, list[dict[str, Any]]] = defaultdict(list)

        for detection in detections:
            height = detection["height"]
            # Round to nearest cluster center
            cluster_center = (
                round(height / self.size_cluster_tolerance) * self.size_cluster_tolerance
            )
            clusters[cluster_center].append(detection)

        return dict(clusters)

    def _map_semantic_sizes(
        self,
        clusters: dict[int, list[dict[str, Any]]],
    ) -> TextSizes:
        """Map size clusters to semantic names.

        Args:
            clusters: Height clusters from _cluster_by_size.

        Returns:
            TextSizes with semantic mappings.
        """
        if not clusters:
            return TextSizes()

        # Sort clusters by count (most common = body text)
        sorted_sizes = sorted(
            clusters.keys(),
            key=lambda h: len(clusters[h]),
            reverse=True,
        )

        # Also sort by size for heading detection
        sizes_by_height: list[int] = sorted(clusters.keys(), reverse=True)

        text_sizes = TextSizes()

        # Most common size is likely body text
        if sorted_sizes:
            text_sizes.body = sorted_sizes[0]

        # Largest sizes are headings
        for size in sizes_by_height:
            if text_sizes.body and size > text_sizes.body * 1.5:
                if text_sizes.heading_large is None:
                    text_sizes.heading_large = size
                elif text_sizes.heading is None and size < text_sizes.heading_large:
                    text_sizes.heading = size
                elif (
                    text_sizes.heading_small is None
                    and text_sizes.heading
                    and size < text_sizes.heading
                ):
                    text_sizes.heading_small = size

        # Smaller sizes
        if text_sizes.body:
            for size in sorted(clusters.keys()):
                if size < text_sizes.body * 0.9:
                    if text_sizes.small is None:
                        text_sizes.small = size
                    elif text_sizes.tiny is None and size < text_sizes.small:
                        text_sizes.tiny = size

        return text_sizes

    def _create_font_samples(
        self,
        detections: list[dict[str, Any]],
        screenshots: list[NDArray[np.uint8]],
    ) -> list[DetectedFont]:
        """Create font samples from detections.

        Args:
            detections: Text detections.
            screenshots: Original screenshots.

        Returns:
            List of DetectedFont samples.
        """
        samples = []
        seen_heights: set[int] = set()

        # Get one sample per unique height
        for detection in detections:
            height = detection["height"]
            if height in seen_heights:
                continue

            seen_heights.add(height)

            bbox = detection["bbox"]
            if len(bbox) == 4 and not isinstance(bbox[0], (list, tuple)):
                x, y, w, h = bbox
            else:
                continue

            # Estimate font family from text characteristics
            font_family = self._classify_font_family(
                detection["text"],
                screenshots[detection["screenshot_idx"]],
                bbox,
            )

            # Estimate weight
            weight = self._estimate_font_weight(
                screenshots[detection["screenshot_idx"]],
                bbox,
            )

            samples.append(
                DetectedFont(
                    sample_region=BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h)),
                    estimated_family=font_family,
                    size_px=height,
                    weight=weight,
                    sample_text=detection["text"][:50] if detection["text"] else None,
                )
            )

            if len(samples) >= 10:  # Limit samples
                break

        return samples

    def _classify_font_family(
        self,
        text: str,
        screenshot: NDArray[np.uint8],
        bbox: tuple[int, int, int, int],
    ) -> FontFamily:
        """Classify font family based on character shapes.

        Args:
            text: The detected text.
            screenshot: Source screenshot.
            bbox: Bounding box (x, y, width, height).

        Returns:
            FontFamily classification.
        """
        # Simple heuristics based on character shapes
        # In a full implementation, this would use ML or detailed stroke analysis

        x, y, w, h = bbox

        try:
            # Extract region
            screenshot = self._ensure_bgr(screenshot)
            region = screenshot[y : y + h, x : x + w]

            if region.size == 0:
                return FontFamily.UNKNOWN

            # Check for monospace characteristics
            if self._looks_monospace(text, w, h):
                return FontFamily.MONOSPACE

            # Check for serif characteristics (simplified)
            if self._has_serifs(region):
                return FontFamily.SERIF

            # Default to sans-serif (most common in UIs)
            return FontFamily.SANS_SERIF

        except Exception as e:
            logger.debug(f"Font classification error: {e}")
            return FontFamily.UNKNOWN

    def _looks_monospace(self, text: str, width: int, height: int) -> bool:
        """Check if text appears monospace based on dimensions.

        Args:
            text: Detected text.
            width: Text region width.
            height: Text region height.

        Returns:
            True if text appears monospace.
        """
        if not text or len(text) < 3:
            return False

        # Monospace: character width roughly constant
        expected_width = len(text) * height * 0.6  # Typical mono aspect ratio
        actual_width = width

        ratio = actual_width / expected_width if expected_width > 0 else 0
        return 0.8 < ratio < 1.2

    def _has_serifs(self, region: NDArray[np.uint8]) -> bool:
        """Check if text region appears to have serifs.

        Args:
            region: Text region image.

        Returns:
            True if serifs detected.
        """
        # Simplified serif detection using edge analysis
        # In a full implementation, this would be more sophisticated
        try:
            import cv2

            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Serifs create more horizontal edges at top/bottom
            h = region.shape[0]
            top_edges = edges[: h // 4].sum()
            bottom_edges = edges[-h // 4 :].sum()
            middle_edges = edges[h // 4 : -h // 4].sum()

            # Serif fonts have more edge detail at extremes
            if middle_edges > 0:
                ratio = (top_edges + bottom_edges) / middle_edges
                return bool(ratio > 1.5)

        except ImportError:
            pass

        return False

    def _estimate_font_weight(
        self,
        screenshot: NDArray[np.uint8],
        bbox: tuple[int, int, int, int],
    ) -> FontWeight:
        """Estimate font weight from stroke analysis.

        Args:
            screenshot: Source screenshot.
            bbox: Text bounding box.

        Returns:
            FontWeight estimation.
        """
        try:
            x, y, w, h = bbox
            screenshot = self._ensure_bgr(screenshot)
            region = screenshot[y : y + h, x : x + w]

            if region.size == 0:
                return FontWeight.NORMAL

            import cv2

            # Convert to grayscale
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

            # Threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Calculate stroke width approximation
            # Count pixels in the foreground relative to total
            foreground_ratio = (binary < 128).sum() / binary.size

            # Thicker strokes = more foreground
            if foreground_ratio > 0.5:
                return FontWeight.BOLD
            elif foreground_ratio > 0.4:
                return FontWeight.SEMIBOLD
            elif foreground_ratio > 0.3:
                return FontWeight.MEDIUM
            elif foreground_ratio < 0.15:
                return FontWeight.LIGHT
            else:
                return FontWeight.NORMAL

        except Exception:
            return FontWeight.NORMAL

    def _identify_text_regions(
        self,
        detections: list[dict[str, Any]],
        screen_size: tuple[int, int],
    ) -> list[TextRegion]:
        """Identify common text regions.

        Args:
            detections: All text detections.
            screen_size: (height, width) of screen.

        Returns:
            List of identified text regions.
        """
        if not detections:
            return []

        h, w = screen_size
        regions = []

        # Define potential region boundaries
        region_defs = [
            ("header", 0, 0, w, int(h * 0.1)),
            ("top_bar", 0, 0, w, int(h * 0.05)),
            ("sidebar", 0, 0, int(w * 0.2), h),
            ("main_content", int(w * 0.2), int(h * 0.1), int(w * 0.8), int(h * 0.8)),
            ("footer", 0, int(h * 0.9), w, int(h * 0.1)),
        ]

        for name, rx, ry, rw, rh in region_defs:
            # Count detections in this region
            region_detections = []
            for d in detections:
                bbox = d["bbox"]
                if len(bbox) == 4 and not isinstance(bbox[0], (list, tuple)):
                    dx, dy, dw, dh = bbox
                    # Check if detection is mostly inside region
                    if rx <= dx < rx + rw and ry <= dy < ry + rh:
                        region_detections.append(d)

            if len(region_detections) >= 2:
                # Calculate average text size in region
                avg_size = sum(d["height"] for d in region_detections) / len(region_detections)

                # Get sample content
                sample_content = [d["text"] for d in region_detections[:5] if d["text"]]

                regions.append(
                    TextRegion(
                        name=name,
                        bounds=BoundingBox(x=rx, y=ry, width=rw, height=rh),
                        avg_size=int(avg_size),
                        typical_content=sample_content if sample_content else None,
                    )
                )

        return regions

    def _detect_languages(
        self,
        detections: list[dict[str, Any]],
    ) -> list[str]:
        """Detect languages from text content.

        Args:
            detections: Text detections.

        Returns:
            List of ISO 639-1 language codes.
        """
        try:
            from langdetect import detect_langs

            # Combine all text
            all_text = " ".join(d["text"] for d in detections if d.get("text"))

            if len(all_text) < 20:
                return ["en"]  # Default to English for short text

            detected = detect_langs(all_text)
            return [lang.lang for lang in detected[:3]]  # Top 3 languages

        except ImportError:
            logger.debug("langdetect not available, defaulting to English")
            return ["en"]
        except Exception:
            return ["en"]
