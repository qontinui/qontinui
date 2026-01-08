"""Text metrics analysis module.

Provides analysis of text visual properties including:
- Font size estimation
- Baseline detection
- Line spacing measurement
- Character spacing analysis
- Text alignment detection
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.assertions import BoundingBox

if TYPE_CHECKING:
    from qontinui_schemas.testing.environment import GUIEnvironment, Typography

    from qontinui.vision.verification.config import VisionConfig
    from qontinui.vision.verification.detection.ocr import OCRResult

logger = logging.getLogger(__name__)


class TextAlignment(str, Enum):
    """Text alignment types."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFIED = "justified"


@dataclass
class TextLine:
    """Represents a line of text with metrics."""

    text: str
    bounds: BoundingBox
    baseline_y: int
    cap_height: int
    x_height: int
    descender_depth: int
    ascender_height: int
    font_size_estimate: int
    words: list["TextWord"] = field(default_factory=list)

    @property
    def line_height(self) -> int:
        """Get total line height."""
        return self.bounds.height

    @property
    def center_y(self) -> float:
        """Get vertical center of text."""
        return self.bounds.y + self.bounds.height / 2


@dataclass
class TextWord:
    """Represents a word with metrics."""

    text: str
    bounds: BoundingBox
    confidence: float
    char_spacing_estimate: float = 0.0


@dataclass
class TextMetrics:
    """Aggregated text metrics for a region."""

    lines: list[TextLine]
    average_font_size: float
    average_line_spacing: float
    average_char_spacing: float
    alignment: TextAlignment
    dominant_baseline: int | None
    text_bounds: BoundingBox | None

    @property
    def line_count(self) -> int:
        """Get number of lines."""
        return len(self.lines)

    @property
    def total_text(self) -> str:
        """Get all text combined."""
        return "\n".join(line.text for line in self.lines)


class TextMetricsAnalyzer:
    """Analyzes text visual metrics.

    Uses OCR results and image analysis to determine:
    - Font sizes and their distribution
    - Baseline positions for alignment
    - Line and character spacing
    - Text alignment patterns

    Usage:
        analyzer = TextMetricsAnalyzer(config, environment)

        # Analyze text in region
        metrics = await analyzer.analyze_region(screenshot, region)

        # Get font size estimate
        font_size = analyzer.estimate_font_size(text_bounds, screenshot)

        # Check text alignment
        alignment = analyzer.detect_alignment(lines)
    """

    def __init__(
        self,
        config: "VisionConfig | None" = None,
        environment: "GUIEnvironment | None" = None,
    ) -> None:
        """Initialize text metrics analyzer.

        Args:
            config: Vision configuration.
            environment: GUI environment with typography hints.
        """
        self._config = config
        self._environment = environment
        self._typography: Typography | None = None

        if environment is not None:
            self._typography = environment.typography

    def _get_known_font_sizes(self) -> list[int]:
        """Get known font sizes from environment.

        Returns:
            List of known font sizes.
        """
        if self._typography is None:
            return []

        sizes = self._typography.text_sizes
        return [
            s
            for s in [
                sizes.heading_large,
                sizes.heading,
                sizes.heading_small,
                sizes.body,
                sizes.small,
                sizes.tiny,
            ]
            if s is not None
        ]

    def estimate_font_size(
        self,
        text_height: int,
        snap_to_known: bool = True,
    ) -> int:
        """Estimate font size from text height.

        Args:
            text_height: Height of text bounding box in pixels.
            snap_to_known: Snap to nearest known font size if available.

        Returns:
            Estimated font size in pixels.
        """
        # Rough conversion: font size ≈ height * 0.75
        # This accounts for line height being larger than font size
        base_estimate = int(text_height * 0.75)

        if snap_to_known:
            known_sizes = self._get_known_font_sizes()
            if known_sizes:
                # Find closest known size
                closest = min(known_sizes, key=lambda s: abs(s - base_estimate))
                # Only snap if reasonably close (within 20%)
                if abs(closest - base_estimate) <= base_estimate * 0.2:
                    return closest

        return base_estimate

    def detect_baseline(
        self,
        region: NDArray[np.uint8],
        bounds: BoundingBox,
    ) -> int:
        """Detect text baseline position.

        Args:
            region: Image region containing text.
            bounds: Bounding box of text.

        Returns:
            Y coordinate of baseline relative to region top.
        """
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region

        # Apply threshold to isolate text
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find horizontal profile (sum of pixels per row)
        h_profile = np.sum(binary, axis=1)

        # Baseline is typically at the bottom of the main text body
        # Find the lowest row with significant content
        threshold = np.max(h_profile) * 0.1
        content_rows = np.where(h_profile > threshold)[0]

        if len(content_rows) == 0:
            # Default to 80% down
            return int(bounds.height * 0.8)

        # Baseline is near the bottom of content
        # Account for descenders by looking at content distribution
        bottom_content = content_rows[-1]

        # Check if there are descenders (content below main body)
        main_content_end = int(np.percentile(content_rows, 85))

        if bottom_content > main_content_end + 3:
            # Has descenders, baseline is at main_content_end
            return main_content_end
        else:
            return bottom_content

    def detect_cap_height(
        self,
        region: NDArray[np.uint8],
    ) -> int:
        """Detect cap height (height of capital letters).

        Args:
            region: Image region containing text.

        Returns:
            Cap height in pixels.
        """
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        h_profile = np.sum(binary, axis=1)
        threshold = np.max(h_profile) * 0.1
        content_rows = np.where(h_profile > threshold)[0]

        if len(content_rows) == 0:
            return region.shape[0]

        # Cap height is from top of content to baseline
        top = content_rows[0]
        bottom = int(np.percentile(content_rows, 85))

        return bottom - top

    def detect_x_height(
        self,
        region: NDArray[np.uint8],
    ) -> int:
        """Detect x-height (height of lowercase letters without ascenders).

        Args:
            region: Image region containing text.

        Returns:
            X-height in pixels (estimate).
        """
        # X-height is typically about 70% of cap height
        cap_height = self.detect_cap_height(region)
        return int(cap_height * 0.7)

    def measure_line_spacing(
        self,
        lines: list[TextLine],
    ) -> float:
        """Measure average line spacing.

        Args:
            lines: List of text lines.

        Returns:
            Average line spacing in pixels.
        """
        if len(lines) < 2:
            return 0.0

        # Sort by vertical position
        sorted_lines = sorted(lines, key=lambda line: line.bounds.y)

        spacings = []
        for i in range(1, len(sorted_lines)):
            prev = sorted_lines[i - 1]
            curr = sorted_lines[i]

            # Spacing from bottom of prev to top of curr
            spacing = curr.bounds.y - (prev.bounds.y + prev.bounds.height)
            spacings.append(spacing)

        return sum(spacings) / len(spacings) if spacings else 0.0

    def measure_char_spacing(
        self,
        word: str,
        word_width: int,
        font_size: int,
    ) -> float:
        """Estimate character spacing.

        Args:
            word: Text content.
            word_width: Width of word in pixels.
            font_size: Estimated font size.

        Returns:
            Average character spacing in pixels.
        """
        if len(word) <= 1:
            return 0.0

        # Estimate average character width (roughly 0.5 * font size for proportional)
        avg_char_width = font_size * 0.5
        expected_width = avg_char_width * len(word)

        # Extra space is distributed as character spacing
        extra_space = word_width - expected_width
        char_spacing = extra_space / (len(word) - 1) if len(word) > 1 else 0

        return max(0, char_spacing)

    def detect_alignment(
        self,
        lines: list[TextLine],
        region_width: int,
        tolerance: int = 10,
    ) -> TextAlignment:
        """Detect text alignment pattern.

        Args:
            lines: List of text lines.
            region_width: Width of containing region.
            tolerance: Alignment tolerance in pixels.

        Returns:
            Detected text alignment.
        """
        if not lines:
            return TextAlignment.LEFT

        left_edges = [line.bounds.x for line in lines]
        right_edges = [line.bounds.x + line.bounds.width for line in lines]
        centers = [line.bounds.x + line.bounds.width / 2 for line in lines]

        # Check left alignment
        left_variance = np.std(left_edges)
        if left_variance <= tolerance:
            # Check if also right-aligned (justified)
            right_variance = np.std(right_edges)
            if right_variance <= tolerance and len(lines) > 1:
                return TextAlignment.JUSTIFIED
            return TextAlignment.LEFT

        # Check right alignment
        right_variance = np.std(right_edges)
        if right_variance <= tolerance:
            return TextAlignment.RIGHT

        # Check center alignment
        center_variance = np.std(centers)
        expected_center = region_width / 2
        center_offset = abs(np.mean(centers) - expected_center)

        if center_variance <= tolerance and center_offset <= tolerance * 2:
            return TextAlignment.CENTER

        # Default to left
        return TextAlignment.LEFT

    async def analyze_region(
        self,
        screenshot: NDArray[np.uint8],
        region: BoundingBox,
        ocr_results: list["OCRResult"] | None = None,
    ) -> TextMetrics:
        """Analyze text metrics in a region.

        Args:
            screenshot: Screenshot to analyze.
            region: Region to analyze.
            ocr_results: Optional pre-computed OCR results.

        Returns:
            Text metrics for the region.
        """
        # Get OCR results if not provided
        if ocr_results is None:
            from qontinui.vision.verification.detection.ocr import get_ocr_engine

            ocr = get_ocr_engine(self._config, self._environment)
            ocr_results = await ocr.detect_text(screenshot, region)

        if not ocr_results:
            return TextMetrics(
                lines=[],
                average_font_size=0,
                average_line_spacing=0,
                average_char_spacing=0,
                alignment=TextAlignment.LEFT,
                dominant_baseline=None,
                text_bounds=None,
            )

        # Extract region image
        img_region = screenshot[
            region.y : region.y + region.height,
            region.x : region.x + region.width,
        ]

        # Group OCR results into lines
        lines = self._group_into_lines(ocr_results, img_region)

        # Calculate metrics
        font_sizes = [line.font_size_estimate for line in lines if line.font_size_estimate > 0]
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0

        avg_line_spacing = self.measure_line_spacing(lines)

        # Average char spacing from words
        char_spacings = []
        for line in lines:
            for word in line.words:
                if word.char_spacing_estimate > 0:
                    char_spacings.append(word.char_spacing_estimate)
        avg_char_spacing = sum(char_spacings) / len(char_spacings) if char_spacings else 0

        alignment = self.detect_alignment(lines, region.width)

        # Find dominant baseline
        baselines = [line.baseline_y for line in lines]
        dominant_baseline = int(np.median(baselines)) if baselines else None

        # Calculate overall text bounds
        if lines:
            min_x = min(line.bounds.x for line in lines)
            min_y = min(line.bounds.y for line in lines)
            max_x = max(line.bounds.x + line.bounds.width for line in lines)
            max_y = max(line.bounds.y + line.bounds.height for line in lines)
            text_bounds = BoundingBox(
                x=min_x,
                y=min_y,
                width=max_x - min_x,
                height=max_y - min_y,
            )
        else:
            text_bounds = None

        return TextMetrics(
            lines=lines,
            average_font_size=avg_font_size,
            average_line_spacing=avg_line_spacing,
            average_char_spacing=avg_char_spacing,
            alignment=alignment,
            dominant_baseline=dominant_baseline,
            text_bounds=text_bounds,
        )

    def _group_into_lines(
        self,
        ocr_results: list["OCRResult"],
        region: NDArray[np.uint8],
    ) -> list[TextLine]:
        """Group OCR results into text lines.

        Args:
            ocr_results: OCR detection results.
            region: Image region.

        Returns:
            List of text lines.
        """
        if not ocr_results:
            return []

        # Sort by vertical position
        sorted_results = sorted(ocr_results, key=lambda r: (r.bounds.y, r.bounds.x))

        lines: list[TextLine] = []
        current_line_results: list[OCRResult] = []
        current_line_y = sorted_results[0].bounds.y

        # Group into lines (results within similar Y position)
        line_threshold = 10  # Pixels

        for result in sorted_results:
            if abs(result.bounds.y - current_line_y) <= line_threshold:
                current_line_results.append(result)
            else:
                # Save current line
                if current_line_results:
                    line = self._create_line(current_line_results, region)
                    lines.append(line)

                # Start new line
                current_line_results = [result]
                current_line_y = result.bounds.y

        # Don't forget last line
        if current_line_results:
            line = self._create_line(current_line_results, region)
            lines.append(line)

        return lines

    def _create_line(
        self,
        results: list["OCRResult"],
        region: NDArray[np.uint8],
    ) -> TextLine:
        """Create a TextLine from OCR results.

        Args:
            results: OCR results for this line.
            region: Image region.

        Returns:
            TextLine instance.
        """
        # Sort by X position
        results = sorted(results, key=lambda r: r.bounds.x)

        # Combine text
        text = " ".join(r.text for r in results)

        # Calculate bounds
        min_x = min(r.bounds.x for r in results)
        min_y = min(r.bounds.y for r in results)
        max_x = max(r.bounds.x + r.bounds.width for r in results)
        max_y = max(r.bounds.y + r.bounds.height for r in results)

        bounds = BoundingBox(
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
        )

        # Extract line region for analysis
        line_region = region[
            max(0, bounds.y) : min(region.shape[0], bounds.y + bounds.height),
            max(0, bounds.x) : min(region.shape[1], bounds.x + bounds.width),
        ]

        # Detect metrics
        baseline_y = (
            self.detect_baseline(line_region, bounds) if line_region.size > 0 else bounds.height
        )
        cap_height = self.detect_cap_height(line_region) if line_region.size > 0 else bounds.height
        x_height = (
            self.detect_x_height(line_region) if line_region.size > 0 else int(bounds.height * 0.7)
        )

        font_size = self.estimate_font_size(bounds.height)

        # Ascender and descender
        ascender_height = max(0, cap_height - x_height)
        descender_depth = max(0, bounds.height - baseline_y)

        # Create words
        words = []
        for result in results:
            char_spacing = self.measure_char_spacing(result.text, result.bounds.width, font_size)
            words.append(
                TextWord(
                    text=result.text,
                    bounds=result.bounds,
                    confidence=result.confidence,
                    char_spacing_estimate=char_spacing,
                )
            )

        return TextLine(
            text=text,
            bounds=bounds,
            baseline_y=baseline_y + bounds.y,  # Absolute Y
            cap_height=cap_height,
            x_height=x_height,
            descender_depth=descender_depth,
            ascender_height=ascender_height,
            font_size_estimate=font_size,
            words=words,
        )


__all__ = [
    "TextAlignment",
    "TextLine",
    "TextMetrics",
    "TextMetricsAnalyzer",
    "TextWord",
]
