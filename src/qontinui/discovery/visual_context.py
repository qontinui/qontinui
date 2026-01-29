"""Visual Context Generator for AI consumption.

This module generates annotated screenshots, visual diffs, and interaction heatmaps
designed for AI systems to understand GUI states. The visual annotations include:
- Element IDs and bounding boxes
- Color-coded element types
- Visual diffs between states
- Clickable region heatmaps

Example:
    >>> from qontinui.discovery.visual_context import VisualContextGenerator
    >>> generator = VisualContextGenerator()
    >>> snapshot = generator.generate_annotated_snapshot(screenshot)
    >>> print(f"Found {len(snapshot.elements)} elements")
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import cv2
import numpy as np

if TYPE_CHECKING:
    from .element_detection import DetectedElement


class ElementColorScheme:
    """Color scheme for element types in BGR format (OpenCV standard).

    Colors are chosen for maximum visual distinction and accessibility.
    Each element type has a unique color to help AI systems quickly
    identify element categories in annotated screenshots.
    """

    # Primary interactive elements
    BUTTON = (255, 165, 0)  # Orange
    INPUT = (0, 255, 0)  # Green
    LINK = (255, 255, 0)  # Cyan
    CHECKBOX = (255, 0, 255)  # Magenta
    RADIO = (255, 0, 128)  # Pink

    # Navigation elements
    MENU = (128, 0, 255)  # Purple
    DROPDOWN = (0, 128, 255)  # Orange-yellow
    TAB = (128, 128, 255)  # Light red

    # Content elements
    TEXT = (128, 128, 128)  # Gray
    IMAGE = (0, 255, 255)  # Yellow
    ICON = (255, 128, 0)  # Blue-cyan

    # Layout elements
    MODAL = (0, 0, 255)  # Red
    SIDEBAR = (128, 255, 0)  # Green-cyan
    HEADER = (255, 128, 128)  # Light blue

    # Default for unknown types
    UNKNOWN = (200, 200, 200)  # Light gray

    @classmethod
    def get_color(cls, element_type: str | None) -> tuple[int, int, int]:
        """Get the color for an element type.

        Args:
            element_type: Type of the element (button, input, link, etc.)

        Returns:
            BGR color tuple for the element type
        """
        if element_type is None:
            return cls.UNKNOWN

        type_upper = element_type.upper()
        return getattr(cls, type_upper, cls.UNKNOWN)


@dataclass
class AnnotatedSnapshot:
    """Result of generating an annotated screenshot.

    Attributes:
        image: The annotated image as a numpy array (BGR format)
        base64_png: Base64-encoded PNG string for easy transmission
        elements: List of element metadata with IDs and bounds
        detection_time_ms: Time taken to detect elements in milliseconds
        annotation_time_ms: Time taken to annotate the image in milliseconds
    """

    image: np.ndarray
    base64_png: str
    elements: list[dict[str, Any]]
    detection_time_ms: float
    annotation_time_ms: float

    def save(self, file_path: str) -> None:
        """Save the annotated image to a file.

        Args:
            file_path: Path to save the image (should end in .png)
        """
        cv2.imwrite(file_path, self.image)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with base64 image and element metadata
        """
        return {
            "base64_png": self.base64_png,
            "elements": self.elements,
            "detection_time_ms": self.detection_time_ms,
            "annotation_time_ms": self.annotation_time_ms,
            "element_count": len(self.elements),
        }


@dataclass
class VisualDiff:
    """Result of comparing two screenshots.

    Attributes:
        diff_image: The diff image as a numpy array (BGR format)
        base64_png: Base64-encoded PNG string
        appeared_regions: List of bounding boxes for new regions
        disappeared_regions: List of bounding boxes for removed regions
        total_change_percentage: Percentage of pixels that changed
        computation_time_ms: Time taken to compute the diff
    """

    diff_image: np.ndarray
    base64_png: str
    appeared_regions: list[tuple[int, int, int, int]]  # (x, y, w, h)
    disappeared_regions: list[tuple[int, int, int, int]]  # (x, y, w, h)
    total_change_percentage: float
    computation_time_ms: float = 0.0

    def save(self, file_path: str) -> None:
        """Save the diff image to a file.

        Args:
            file_path: Path to save the image (should end in .png)
        """
        cv2.imwrite(file_path, self.diff_image)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with diff metadata
        """
        return {
            "base64_png": self.base64_png,
            "appeared_regions": [
                {"x": r[0], "y": r[1], "width": r[2], "height": r[3]} for r in self.appeared_regions
            ],
            "disappeared_regions": [
                {"x": r[0], "y": r[1], "width": r[2], "height": r[3]}
                for r in self.disappeared_regions
            ],
            "total_change_percentage": self.total_change_percentage,
            "computation_time_ms": self.computation_time_ms,
            "appeared_count": len(self.appeared_regions),
            "disappeared_count": len(self.disappeared_regions),
        }


@dataclass
class InteractionHeatmap:
    """Result of generating an interaction heatmap.

    Attributes:
        heatmap_image: The heatmap image as a numpy array (BGR format)
        base64_png: Base64-encoded PNG string
        clickable_regions: List of clickable region metadata
        computation_time_ms: Time taken to generate the heatmap
    """

    heatmap_image: np.ndarray
    base64_png: str
    clickable_regions: list[dict[str, Any]]
    computation_time_ms: float = 0.0

    def save(self, file_path: str) -> None:
        """Save the heatmap image to a file.

        Args:
            file_path: Path to save the image (should end in .png)
        """
        cv2.imwrite(file_path, self.heatmap_image)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with heatmap metadata
        """
        return {
            "base64_png": self.base64_png,
            "clickable_regions": self.clickable_regions,
            "computation_time_ms": self.computation_time_ms,
            "clickable_count": len(self.clickable_regions),
        }


class VisualContextGenerator:
    """Generates visual context for AI consumption.

    This class creates annotated screenshots and visual diffs that help
    AI systems understand the current state of a GUI. The annotations
    are designed to be easily parsed and understood by language models.

    Example:
        >>> generator = VisualContextGenerator()
        >>> # Generate annotated snapshot with auto-detection
        >>> snapshot = generator.generate_annotated_snapshot(screenshot)
        >>> # Generate visual diff between two states
        >>> diff = generator.generate_visual_diff(before_img, after_img)
        >>> # Generate interaction heatmap
        >>> heatmap = generator.generate_interaction_heatmap(screenshot)
    """

    def __init__(
        self,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.5,
        font_thickness: int = 1,
        box_thickness: int = 2,
        text_padding: int = 5,
    ) -> None:
        """Initialize the visual context generator.

        Args:
            font: OpenCV font to use for text
            font_scale: Scale factor for text size
            font_thickness: Thickness of text
            box_thickness: Thickness of bounding box lines
            text_padding: Padding around text labels
        """
        self.font = font
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.box_thickness = box_thickness
        self.text_padding = text_padding

        # Colors for annotations
        self.color_text_bg = (0, 0, 0)  # Black background for text
        self.color_text = (255, 255, 255)  # White text

        # Diff colors
        self.color_appeared = (0, 255, 0)  # Green for new regions
        self.color_disappeared = (0, 0, 255)  # Red for removed regions
        self.color_unchanged = (128, 128, 128)  # Gray for unchanged

        # Heatmap colors
        self.heatmap_color_high = (0, 0, 255)  # Red for high interactivity
        self.heatmap_color_low = (255, 255, 0)  # Cyan for low interactivity

    def generate_annotated_snapshot(
        self,
        screenshot: np.ndarray,
        elements: list[DetectedElement] | None = None,
        auto_detect: bool = True,
        include_legend: bool = True,
    ) -> AnnotatedSnapshot:
        """Generate a screenshot annotated with element IDs and bounding boxes.

        Args:
            screenshot: The screenshot to annotate (BGR format numpy array)
            elements: Optional list of pre-detected elements
            auto_detect: If True and no elements provided, auto-detect elements
            include_legend: If True, include a color legend at the top

        Returns:
            AnnotatedSnapshot with the annotated image and element metadata
        """
        start_time = time.perf_counter()
        detection_time_ms = 0.0

        # Convert PIL Image to numpy if needed
        screenshot = self._ensure_numpy(screenshot)

        # Detect elements if not provided
        if elements is None and auto_detect:
            detect_start = time.perf_counter()
            elements = self._auto_detect_elements(screenshot)
            detection_time_ms = (time.perf_counter() - detect_start) * 1000

        # Create a copy for annotation
        annotated = screenshot.copy()

        # Build element metadata list
        element_metadata: list[dict[str, Any]] = []

        if elements:
            # Sort elements by size (larger first) so smaller elements are drawn on top
            sorted_elements = sorted(
                elements,
                key=lambda e: e.bounding_box.width * e.bounding_box.height,
                reverse=True,
            )

            for idx, element in enumerate(sorted_elements):
                element_id = f"E{idx:03d}"
                color = ElementColorScheme.get_color(element.element_type)

                # Draw bounding box
                x, y = element.bounding_box.x, element.bounding_box.y
                w, h = element.bounding_box.width, element.bounding_box.height

                cv2.rectangle(
                    annotated,
                    (x, y),
                    (x + w, y + h),
                    color,
                    self.box_thickness,
                )

                # Draw element ID label
                label = f"{element_id}"
                if element.element_type:
                    label = f"{element_id}:{element.element_type[:3].upper()}"

                self._draw_label(annotated, x, y - 5, label, color)

                # Store element metadata
                element_metadata.append(
                    {
                        "id": element_id,
                        "type": element.element_type,
                        "label": element.label,
                        "bounds": {
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                        },
                        "confidence": element.confidence,
                        "center": {"x": x + w // 2, "y": y + h // 2},
                    }
                )

        # Add legend if requested
        if include_legend:
            self._draw_legend(annotated)

        # Add element count info at top
        self._draw_info_bar(annotated, f"Elements: {len(element_metadata)}")

        # Encode to base64
        base64_png = self._encode_to_base64(annotated)

        annotation_time_ms = (time.perf_counter() - start_time) * 1000 - detection_time_ms

        return AnnotatedSnapshot(
            image=annotated,
            base64_png=base64_png,
            elements=element_metadata,
            detection_time_ms=detection_time_ms,
            annotation_time_ms=annotation_time_ms,
        )

    def generate_visual_diff(
        self,
        before: np.ndarray,
        after: np.ndarray,
        threshold: float = 30.0,
        min_region_area: int = 100,
    ) -> VisualDiff:
        """Generate a visual diff highlighting changes between two screenshots.

        Args:
            before: Screenshot before the change (BGR format)
            after: Screenshot after the change (BGR format)
            threshold: Pixel difference threshold for considering a change
            min_region_area: Minimum area in pixels to consider a changed region

        Returns:
            VisualDiff with the diff image and change metadata
        """
        start_time = time.perf_counter()

        # Ensure numpy arrays
        before = self._ensure_numpy(before)
        after = self._ensure_numpy(after)

        # Ensure same dimensions
        if before.shape != after.shape:
            # Resize after to match before
            after = cv2.resize(after, (before.shape[1], before.shape[0]))

        # Convert to grayscale for comparison
        before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

        # Compute absolute difference
        diff = cv2.absdiff(before_gray, after_gray)

        # Threshold to get binary mask
        _, binary_mask = cv2.threshold(diff, int(threshold), 255, cv2.THRESH_BINARY)

        # Apply morphological operations to clean up noise
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Calculate total change percentage
        total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
        changed_pixels = np.sum(binary_mask > 0)
        change_percentage = (changed_pixels / total_pixels) * 100

        # Find contours for changed regions
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create diff visualization image
        # Start with a blend of before and after
        diff_image = cv2.addWeighted(before, 0.5, after, 0.5, 0)

        appeared_regions: list[tuple[int, int, int, int]] = []
        disappeared_regions: list[tuple[int, int, int, int]] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_region_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Determine if region appeared or disappeared
            # by comparing average intensity in before vs after
            before_roi = before_gray[y : y + h, x : x + w]
            after_roi = after_gray[y : y + h, x : x + w]

            before_mean = np.mean(before_roi)
            after_mean = np.mean(after_roi)

            if after_mean > before_mean:
                # Region appeared (brighter in after)
                color = self.color_appeared
                appeared_regions.append((x, y, w, h))
                label = "NEW"
            else:
                # Region disappeared (brighter in before)
                color = self.color_disappeared
                disappeared_regions.append((x, y, w, h))
                label = "GONE"

            # Draw rectangle
            cv2.rectangle(diff_image, (x, y), (x + w, y + h), color, 2)

            # Draw label
            self._draw_label(diff_image, x, y - 5, label, color)

        # Add summary info at top
        summary = f"Changed: {change_percentage:.1f}% | +{len(appeared_regions)} -{len(disappeared_regions)}"
        self._draw_info_bar(diff_image, summary)

        # Encode to base64
        base64_png = self._encode_to_base64(diff_image)

        computation_time_ms = (time.perf_counter() - start_time) * 1000

        return VisualDiff(
            diff_image=diff_image,
            base64_png=base64_png,
            appeared_regions=appeared_regions,
            disappeared_regions=disappeared_regions,
            total_change_percentage=float(change_percentage),
            computation_time_ms=computation_time_ms,
        )

    def generate_interaction_heatmap(
        self,
        screenshot: np.ndarray,
        elements: list[DetectedElement] | None = None,
        auto_detect: bool = True,
        alpha: float = 0.6,
    ) -> InteractionHeatmap:
        """Generate a heatmap showing clickable/interactive regions.

        Regions with higher interactivity potential are shown in warmer colors.
        This helps AI systems identify where actions can be taken.

        Args:
            screenshot: The screenshot to analyze (BGR format)
            elements: Optional list of pre-detected elements
            auto_detect: If True and no elements provided, auto-detect elements
            alpha: Transparency of the heatmap overlay (0-1)

        Returns:
            InteractionHeatmap with the heatmap image and region metadata
        """
        start_time = time.perf_counter()

        # Ensure numpy array
        screenshot = self._ensure_numpy(screenshot)

        # Detect elements if not provided
        if elements is None and auto_detect:
            elements = self._auto_detect_elements(screenshot)

        # Create a base heatmap (grayscale)
        height, width = screenshot.shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)

        # Clickable element types (higher weight = more interactive)
        interactivity_weights: dict[str | None, float] = {
            "button": 1.0,
            "link": 0.9,
            "input": 0.8,
            "checkbox": 0.8,
            "radio": 0.8,
            "dropdown": 0.7,
            "tab": 0.6,
            "menu": 0.6,
            "icon": 0.5,
            None: 0.3,  # Unknown types get lower weight
        }

        clickable_regions: list[dict[str, Any]] = []

        if elements:
            for element in elements:
                x, y = element.bounding_box.x, element.bounding_box.y
                w, h = element.bounding_box.width, element.bounding_box.height

                # Clamp to image bounds
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(width, x + w), min(height, y + h)

                # Get interactivity weight
                weight = interactivity_weights.get(
                    element.element_type,
                    interactivity_weights[None],
                )

                # Add to heatmap (higher weight = higher value)
                heatmap[y1:y2, x1:x2] = np.maximum(
                    heatmap[y1:y2, x1:x2], weight * element.confidence
                )

                # Track clickable regions
                if weight >= 0.5:  # Consider interactive if weight >= 0.5
                    clickable_regions.append(
                        {
                            "type": element.element_type,
                            "bounds": {"x": x, "y": y, "width": w, "height": h},
                            "center": {"x": x + w // 2, "y": y + h // 2},
                            "interactivity_score": weight * element.confidence,
                            "label": element.label,
                        }
                    )

        # Normalize heatmap to 0-255
        if np.max(heatmap) > 0:
            heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
        else:
            heatmap = heatmap.astype(np.uint8)

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Blend with original screenshot
        heatmap_image = cv2.addWeighted(screenshot, 1 - alpha, heatmap_colored, alpha, 0)

        # Add info bar
        self._draw_info_bar(heatmap_image, f"Clickable regions: {len(clickable_regions)}")

        # Encode to base64
        base64_png = self._encode_to_base64(heatmap_image)

        computation_time_ms = (time.perf_counter() - start_time) * 1000

        return InteractionHeatmap(
            heatmap_image=heatmap_image,
            base64_png=base64_png,
            clickable_regions=clickable_regions,
            computation_time_ms=computation_time_ms,
        )

    def _ensure_numpy(self, image: Any) -> np.ndarray:
        """Ensure the image is a numpy array in BGR format.

        Args:
            image: Image as numpy array or PIL Image

        Returns:
            Numpy array in BGR format
        """
        # Handle PIL Image
        try:
            from PIL import Image

            if isinstance(image, Image.Image):
                image = np.array(image)
                # PIL uses RGB, OpenCV uses BGR
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except ImportError:
            pass

        return cast(np.ndarray, image)

    def _auto_detect_elements(self, screenshot: np.ndarray) -> list[DetectedElement]:
        """Auto-detect UI elements in a screenshot.

        Uses contour detection and heuristics to identify potential UI elements.
        This is a simplified version - for production use, integrate with
        proper element detectors from the element_detection module.

        Args:
            screenshot: Screenshot to analyze

        Returns:
            List of detected elements
        """
        from .element_detection import BoundingBox, DetectedElement

        # Convert to grayscale
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        elements: list[DetectedElement] = []
        min_area = 100  # Minimum area to consider
        max_area = screenshot.shape[0] * screenshot.shape[1] * 0.5  # Max 50% of image

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Simple heuristics to guess element type
            aspect_ratio = w / h if h > 0 else 0
            element_type: str | None = None

            if 2.0 < aspect_ratio < 6.0 and h < 50:
                element_type = "button"
            elif 3.0 < aspect_ratio < 10.0 and h < 40:
                element_type = "input"
            elif 0.8 < aspect_ratio < 1.2 and w < 50:
                element_type = "icon"
            elif aspect_ratio > 6.0:
                element_type = "text"

            # Calculate confidence based on regularity of shape
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            confidence = min(1.0, solidity + 0.3)

            elements.append(
                DetectedElement(
                    bounding_box=BoundingBox(x=x, y=y, width=w, height=h),
                    confidence=confidence,
                    element_type=element_type,
                )
            )

        return elements

    def _draw_label(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        text: str,
        color: tuple[int, int, int],
    ) -> None:
        """Draw a text label with background.

        Args:
            image: Image to draw on
            x: X position
            y: Y position
            text: Text to draw
            color: Color for the label (BGR)
        """
        # Ensure coordinates are integers
        x = int(round(x))
        y = int(round(y))

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.font_thickness
        )

        # Ensure label stays within image bounds
        if y - text_height - self.text_padding < 0:
            y = text_height + self.text_padding + 5  # Move below instead

        # Draw background rectangle
        cv2.rectangle(
            image,
            (x, y - text_height - self.text_padding),
            (x + text_width + self.text_padding, y),
            self.color_text_bg,
            -1,  # Filled
        )

        # Draw text
        cv2.putText(
            image,
            text,
            (x + 2, y - 2),
            self.font,
            self.font_scale,
            self.color_text,
            self.font_thickness,
            cv2.LINE_AA,
        )

    def _draw_info_bar(self, image: np.ndarray, text: str) -> None:
        """Draw an info bar at the top of the image.

        Args:
            image: Image to draw on
            text: Text to display in the info bar
        """
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.font_thickness
        )

        # Calculate position (centered at top)
        x = int((image.shape[1] - text_width) // 2)
        y = int(text_height + self.text_padding)

        # Draw background
        cv2.rectangle(
            image,
            (x - self.text_padding, 0),
            (x + text_width + self.text_padding, y + self.text_padding),
            self.color_text_bg,
            -1,
        )

        # Draw text
        cv2.putText(
            image,
            text,
            (x, y),
            self.font,
            self.font_scale,
            self.color_text,
            self.font_thickness,
            cv2.LINE_AA,
        )

    def _draw_legend(self, image: np.ndarray) -> None:
        """Draw a color legend in the top-right corner.

        Args:
            image: Image to draw on
        """
        legend_items = [
            ("BTN", ElementColorScheme.BUTTON),
            ("INP", ElementColorScheme.INPUT),
            ("LNK", ElementColorScheme.LINK),
            ("ICO", ElementColorScheme.ICON),
            ("TXT", ElementColorScheme.TEXT),
        ]

        # Calculate legend position (top-right corner)
        item_height = 15
        item_width = 40
        padding = 5
        start_x = image.shape[1] - (item_width + padding * 2)
        start_y = padding + 25  # Below info bar

        # Draw legend background
        legend_height = len(legend_items) * item_height + padding * 2
        cv2.rectangle(
            image,
            (start_x - padding, start_y - padding),
            (image.shape[1] - padding, start_y + legend_height),
            (40, 40, 40),
            -1,
        )

        # Draw legend items
        for i, (label, color) in enumerate(legend_items):
            y = start_y + i * item_height + item_height // 2

            # Draw color swatch
            cv2.rectangle(
                image,
                (start_x, y - 5),
                (start_x + 10, y + 5),
                color,
                -1,
            )

            # Draw label
            cv2.putText(
                image,
                label,
                (start_x + 15, y + 4),
                self.font,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _encode_to_base64(self, image: np.ndarray) -> str:
        """Encode an image to base64 PNG string.

        Args:
            image: Image to encode (BGR format)

        Returns:
            Base64-encoded PNG string
        """
        success, buffer = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("Failed to encode image as PNG")

        png_bytes = buffer.tobytes()
        return base64.b64encode(png_bytes).decode("utf-8")
