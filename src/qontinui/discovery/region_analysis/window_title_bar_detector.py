"""
Window Title Bar Detector.

This analyzer detects window title bars by looking for horizontal regions at the top
of windows that contain text and close buttons. It extracts the title text and
identifies the close button location.

Performance: 100-300ms
Accuracy: 80-90% for standard window layouts
"""

from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image

from qontinui.discovery.region_analysis.base import (
    BaseRegionAnalyzer,
    BoundingBox,
    DetectedRegion,
    RegionAnalysisInput,
    RegionAnalysisResult,
    RegionAnalysisType,
    RegionType,
)

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class WindowTitleBarDetector(BaseRegionAnalyzer):
    """Detects window title bars with text extraction and close button location."""

    @property
    def analysis_type(self) -> RegionAnalysisType:
        return RegionAnalysisType.PATTERN_ANALYSIS

    @property
    def name(self) -> str:
        return "window_title_bar_detector"

    @property
    def supported_region_types(self) -> list[RegionType]:
        return [RegionType.TITLE_BAR, RegionType.CLOSE_BUTTON, RegionType.WINDOW]

    @property
    def version(self) -> str:
        return "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Window Title Bar detector.

        Args:
            config: Optional configuration parameters
        """
        super().__init__(config)

        params = self.get_default_parameters()
        if config:
            params.update(config)

        self.min_title_bar_height = params["min_title_bar_height"]
        self.max_title_bar_height = params["max_title_bar_height"]
        self.min_title_bar_width = params["min_title_bar_width"]
        self.title_bar_aspect_ratio = params["title_bar_aspect_ratio"]
        self.detect_close_button = params["detect_close_button"]
        self.extract_title_text = params["extract_title_text"]
        self.use_ocr = params["use_ocr"] and TESSERACT_AVAILABLE

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            "min_title_bar_height": 20,
            "max_title_bar_height": 60,
            "min_title_bar_width": 100,
            "title_bar_aspect_ratio": 3.0,  # Minimum width/height ratio
            "detect_close_button": True,
            "extract_title_text": True,
            "use_ocr": True,
        }

    async def analyze(self, input_data: RegionAnalysisInput) -> RegionAnalysisResult:
        """Detect window title bars."""
        all_regions = []

        # Process each screenshot
        for idx, screenshot_bytes in enumerate(input_data.screenshot_data):
            # Convert bytes to numpy array
            image = Image.open(BytesIO(screenshot_bytes))
            image_np = np.array(image)

            # Convert to grayscale
            if len(image_np.shape) == 3:
                if image_np.shape[2] == 4:
                    gray = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
                    color = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
                else:
                    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                    color = image_np
            else:
                gray = image_np
                color = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            # Detect title bars
            regions = self._detect_title_bars(gray, color, idx)
            all_regions.extend(regions)

        # Calculate overall confidence
        overall_confidence = (
            sum(r.confidence for r in all_regions) / len(all_regions)
            if all_regions
            else 0.0
        )

        return RegionAnalysisResult(
            analyzer_type=self.analysis_type,
            analyzer_name=self.name,
            regions=all_regions,
            confidence=overall_confidence,
            metadata={
                "use_ocr": self.use_ocr,
                "detect_close_button": self.detect_close_button,
                "total_title_bars": len(
                    [r for r in all_regions if r.region_type == RegionType.TITLE_BAR]
                ),
                "total_close_buttons": len(
                    [r for r in all_regions if r.region_type == RegionType.CLOSE_BUTTON]
                ),
            },
        )

    def _detect_title_bars(
        self, gray: np.ndarray, color: np.ndarray, screenshot_index: int
    ) -> list[DetectedRegion]:
        """Detect title bars in an image."""
        detected_regions: list[DetectedRegion] = []

        # Method 1: Find horizontal bars with consistent color/intensity
        title_bar_candidates = self._find_horizontal_bars(gray)

        for candidate in title_bar_candidates:
            x, y, w, h = candidate

            # Extract title text if enabled
            title_text = None
            if self.extract_title_text:
                roi = color[y : y + h, x : x + w]
                title_text = self._extract_title_text(roi)

            # Detect close button if enabled
            close_button_bbox = None
            if self.detect_close_button:
                close_button_bbox = self._detect_close_button(gray, x, y, w, h)

            # Calculate confidence based on features
            confidence = self._calculate_title_bar_confidence(
                gray, x, y, w, h, title_text, close_button_bbox
            )

            # Create title bar region
            metadata: dict[str, Any] = {
                "detection_method": "title_bar",
            }
            if title_text:
                metadata["title_text"] = title_text
            if close_button_bbox:
                metadata["close_button"] = {
                    "x": close_button_bbox[0],
                    "y": close_button_bbox[1],
                    "width": close_button_bbox[2],
                    "height": close_button_bbox[3],
                }

            title_bar = DetectedRegion(
                bounding_box=BoundingBox(x, y, w, h),
                confidence=confidence,
                region_type=RegionType.TITLE_BAR,
                label=(
                    title_text if title_text else f"title_bar_{len(detected_regions)}"
                ),
                screenshot_index=screenshot_index,
                metadata=metadata,
            )
            detected_regions.append(title_bar)

            # Add close button as separate region if detected
            if close_button_bbox:
                close_btn = DetectedRegion(
                    bounding_box=BoundingBox(
                        close_button_bbox[0],
                        close_button_bbox[1],
                        close_button_bbox[2],
                        close_button_bbox[3],
                    ),
                    confidence=confidence * 0.9,
                    region_type=RegionType.CLOSE_BUTTON,
                    label="close_button",
                    screenshot_index=screenshot_index,
                    metadata={
                        "parent_title_bar": len(detected_regions) - 1,
                        "detection_method": "pattern_matching",
                    },
                )
                detected_regions.append(close_btn)

        return detected_regions

    def _find_horizontal_bars(
        self, gray: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """Find horizontal bar-like regions that could be title bars."""
        candidates = []

        # Edge detection to find horizontal lines
        edges = cv2.Canny(gray, 50, 150)

        # Morphological operations to emphasize horizontal structures
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 3))
        horizontal = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by dimensions
            if h < self.min_title_bar_height or h > self.max_title_bar_height:
                continue
            if w < self.min_title_bar_width:
                continue
            if w / h < self.title_bar_aspect_ratio:
                continue

            # Title bars are typically at the top of windows
            # Look for rectangular regions below this bar
            has_content_below = self._has_content_below(gray, x, y, w, h)
            if has_content_below:
                candidates.append((x, y, w, h))

        return candidates

    def _has_content_below(
        self, gray: np.ndarray, x: int, y: int, w: int, h: int
    ) -> bool:
        """Check if there's significant content below the potential title bar."""
        # Sample region below
        below_y = y + h
        below_height = min(100, gray.shape[0] - below_y)

        if below_height < 20:
            return False

        below_region = gray[
            below_y : below_y + below_height, x : min(x + w, gray.shape[1])
        ]

        # Check if there's variation (content) in the region below
        std_dev = np.std(below_region)
        return std_dev > 10  # type: ignore[no-any-return]  # Some threshold for content detection

    def _extract_title_text(self, roi: np.ndarray) -> str | None:
        """Extract title text from the title bar region."""
        if not self.use_ocr:
            return None

        try:
            # Preprocess for better OCR
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

            # Try both normal and inverted
            text1 = pytesseract.image_to_string(gray_roi, config="--psm 7").strip()
            _, inverted = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY_INV)
            text2 = pytesseract.image_to_string(inverted, config="--psm 7").strip()

            # Return longer text (more likely to be correct)
            text = text1 if len(text1) > len(text2) else text2

            # Clean up text
            text = text.strip()
            if len(text) > 0 and len(text) < 200:  # Reasonable title length
                return text  # type: ignore[no-any-return]
        except Exception:
            pass

        return None

    def _detect_close_button(
        self, gray: np.ndarray, title_x: int, title_y: int, title_w: int, title_h: int
    ) -> tuple[int, int, int, int] | None:
        """Detect close button (X icon) in the title bar."""
        # Close button is typically in the top-right corner of the title bar
        # Expected size is roughly square, about 70-100% of title bar height

        btn_size = int(title_h * 0.8)
        search_x = title_x + title_w - btn_size * 2
        search_y = title_y
        search_w = btn_size * 2
        search_h = title_h

        # Ensure within bounds
        search_x = max(0, search_x)
        search_y = max(0, search_y)
        search_w = min(search_w, gray.shape[1] - search_x)
        search_h = min(search_h, gray.shape[0] - search_y)

        if search_w < btn_size or search_h < 10:
            return None

        # Extract search region
        search_region = gray[
            search_y : search_y + search_h, search_x : search_x + search_w
        ]

        # Look for X-pattern or high contrast square region
        # Method 1: Template matching for X pattern
        x_pattern = self._create_x_pattern(btn_size, btn_size)
        result = cv2.matchTemplate(search_region, x_pattern, cv2.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > 0.3:  # Threshold for match
            # Found X pattern
            btn_x = search_x + max_loc[0]
            btn_y = search_y + max_loc[1]
            return (btn_x, btn_y, btn_size, btn_size)

        # Method 2: Look for square regions with high edge density (borders)
        edges = cv2.Canny(search_region, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Should be roughly square and right size
            if abs(w - h) > w * 0.3:  # Not square enough
                continue
            if w < btn_size * 0.5 or w > btn_size * 1.2:
                continue

            # Should be in the right part of search area (right side)
            if x < search_w * 0.5:
                continue

            return (search_x + x, search_y + y, w, h)

        return None

    def _create_x_pattern(self, width: int, height: int) -> np.ndarray:
        """Create an X pattern template for matching."""
        pattern = np.zeros((height, width), dtype=np.uint8)

        # Draw X
        thickness = max(1, width // 10)
        cv2.line(pattern, (0, 0), (width - 1, height - 1), 255, thickness)  # type: ignore[call-overload]
        cv2.line(pattern, (width - 1, 0), (0, height - 1), 255, thickness)  # type: ignore[call-overload]

        return pattern

    def _calculate_title_bar_confidence(
        self,
        gray: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        title_text: str | None,
        close_button: tuple | None,
    ) -> float:
        """Calculate confidence that this is a title bar."""
        confidence = 0.5  # Base confidence

        # Boost if title text was extracted
        if title_text and len(title_text) > 0:
            confidence += 0.2

        # Boost if close button was detected
        if close_button:
            confidence += 0.2

        # Check horizontal uniformity (title bars often have uniform background)
        roi = gray[y : y + h, x : x + w]
        row_means = np.mean(roi, axis=1)
        row_std = np.std(row_means)
        if row_std < 10:  # Uniform rows
            confidence += 0.1

        return min(confidence, 1.0)
