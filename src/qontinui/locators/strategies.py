"""Locator strategies for finding UI elements.

Provides multiple strategies for locating elements on screen:
- VisualPatternStrategy: Template matching (existing)
- SemanticTextStrategy: Text content matching
- RelativePositionStrategy: Relative to anchor elements
- ColorRegionStrategy: Color pattern matching
- StructuralStrategy: Visual structure detection
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..model.element import Location, Pattern, Region

logger = logging.getLogger(__name__)


@dataclass
class ScreenContext:
    """Context information about the screen being searched.

    Attributes:
        screenshot: Screenshot image (numpy array)
        timestamp: When screenshot was taken
        monitor_info: Optional monitor information
        metadata: Additional context metadata
    """

    screenshot: np.ndarray
    timestamp: float = 0.0
    monitor_info: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchResult:
    """Result from a locator strategy.

    Attributes:
        region: Matched region
        confidence: Match confidence (0.0-1.0)
        strategy_name: Name of strategy that found this match
        metadata: Additional match metadata
    """

    region: Region
    confidence: float
    strategy_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_location(self) -> Location:
        """Convert to Location at region center.

        Returns:
            Location at center of matched region
        """
        center_x = self.region.x + self.region.width // 2
        center_y = self.region.y + self.region.height // 2
        return Location(x=center_x, y=center_y, region=self.region)


class LocatorStrategy(ABC):
    """Base class for locator strategies.

    Each strategy implements a different method of finding UI elements.
    Strategies are tried in sequence until one succeeds.
    """

    @abstractmethod
    def find(
        self,
        target: Pattern | dict[str, Any],
        context: ScreenContext,
        min_confidence: float = 0.7,
    ) -> MatchResult | None:
        """Find element using this strategy.

        Args:
            target: Pattern or configuration dict describing what to find
            context: Screen context with screenshot and metadata
            min_confidence: Minimum confidence threshold (0.0-1.0)

        Returns:
            MatchResult if found with sufficient confidence, None otherwise
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name for logging and reporting.

        Returns:
            Human-readable strategy name
        """
        pass

    @abstractmethod
    def can_handle(self, target: Pattern | dict[str, Any]) -> bool:
        """Check if this strategy can handle the given target.

        Args:
            target: Target to check

        Returns:
            True if strategy can handle this target type
        """
        pass


class VisualPatternStrategy(LocatorStrategy):
    """Template matching strategy using existing vision system.

    Uses OpenCV template matching with mask support.
    This is the primary/default strategy.
    """

    def __init__(self) -> None:
        """Initialize visual pattern strategy."""
        # Import here to avoid circular dependency
        from ..find.matchers import TemplateMatcher

        self.matcher = TemplateMatcher(method="TM_CCOEFF_NORMED", nms_overlap_threshold=0.3)

    def find(
        self,
        target: Pattern | dict[str, Any],
        context: ScreenContext,
        min_confidence: float = 0.7,
    ) -> MatchResult | None:
        """Find using template matching.

        Args:
            target: Pattern with pixel_data and mask
            context: Screen context
            min_confidence: Minimum confidence threshold

        Returns:
            MatchResult if match found above threshold
        """
        if not isinstance(target, Pattern):
            logger.warning(f"VisualPatternStrategy requires Pattern, got {type(target)}")
            return None

        try:
            matches = self.matcher.find_matches(
                screenshot=context.screenshot,
                pattern=target,
                find_all=False,
                similarity=min_confidence,
            )

            if matches and len(matches) > 0:
                best_match = matches[0]
                if best_match.region is None:
                    return None

                return MatchResult(
                    region=best_match.region,
                    confidence=best_match.similarity,
                    strategy_name=self.get_name(),
                    metadata={
                        "method": "template_matching",
                        "pattern_name": target.name,
                    },
                )

            return None

        except Exception as e:
            logger.error(f"VisualPatternStrategy error: {e}", exc_info=True)
            return None

    def get_name(self) -> str:
        """Get strategy name.

        Returns:
            Strategy name
        """
        return "VisualPattern"

    def can_handle(self, target: Pattern | dict[str, Any]) -> bool:
        """Check if target is a Pattern with pixel data.

        Args:
            target: Target to check

        Returns:
            True if target is Pattern with pixel_data
        """
        return isinstance(target, Pattern) and target.pixel_data is not None


class SemanticTextStrategy(LocatorStrategy):
    """Find elements by text content using OCR.

    Uses optical character recognition to find elements containing
    specific text. Useful when visual appearance changes but text remains same.
    """

    def __init__(self) -> None:
        """Initialize semantic text strategy."""
        self.use_ocr = True

    def find(
        self,
        target: Pattern | dict[str, Any],
        context: ScreenContext,
        min_confidence: float = 0.7,
    ) -> MatchResult | None:
        """Find element by text content.

        Args:
            target: Dict with 'text' field or Pattern with text metadata
            context: Screen context
            min_confidence: Minimum confidence threshold

        Returns:
            MatchResult if text found above threshold
        """
        # Extract target text
        target_text = None
        if isinstance(target, dict):
            target_text = target.get("text")
        elif isinstance(target, Pattern):
            target_text = target.metadata.get("text") if hasattr(target, "metadata") else None  # type: ignore[attr-defined]

        if not target_text:
            return None

        try:
            # Try using pytesseract if available
            try:
                import pytesseract

                # Perform OCR on screenshot
                screenshot_rgb = cv2.cvtColor(context.screenshot, cv2.COLOR_BGR2RGB)
                ocr_data = pytesseract.image_to_data(
                    screenshot_rgb, output_type=pytesseract.Output.DICT
                )

                # Search for matching text
                for i, text in enumerate(ocr_data["text"]):
                    if not text:
                        continue

                    # Simple text matching (could be enhanced with fuzzy matching)
                    if target_text.lower() in text.lower():
                        confidence = float(ocr_data["conf"][i]) / 100.0
                        if confidence >= min_confidence:
                            x = int(ocr_data["left"][i])
                            y = int(ocr_data["top"][i])
                            w = int(ocr_data["width"][i])
                            h = int(ocr_data["height"][i])

                            return MatchResult(
                                region=Region(x=x, y=y, width=w, height=h),
                                confidence=confidence,
                                strategy_name=self.get_name(),
                                metadata={"text": text, "method": "ocr"},
                            )

            except ImportError:
                logger.debug("pytesseract not available for SemanticTextStrategy")
                return None

            return None

        except Exception as e:
            logger.error(f"SemanticTextStrategy error: {e}", exc_info=True)
            return None

    def get_name(self) -> str:
        """Get strategy name.

        Returns:
            Strategy name
        """
        return "SemanticText"

    def can_handle(self, target: Pattern | dict[str, Any]) -> bool:
        """Check if target has text to search for.

        Args:
            target: Target to check

        Returns:
            True if target has text field
        """
        if isinstance(target, dict):
            return "text" in target
        elif isinstance(target, Pattern):
            return hasattr(target, "metadata") and "text" in target.metadata  # type: ignore[attr-defined]
        return False


class RelativePositionStrategy(LocatorStrategy):
    """Find elements relative to anchor elements.

    Finds target by its spatial relationship to a stable anchor element.
    Example: "Find submit button to the right of cancel button"
    """

    def __init__(self) -> None:
        """Initialize relative position strategy."""
        self.visual_strategy = VisualPatternStrategy()

    def find(
        self,
        target: Pattern | dict[str, Any],
        context: ScreenContext,
        min_confidence: float = 0.7,
    ) -> MatchResult | None:
        """Find element relative to anchor.

        Args:
            target: Dict with 'anchor_pattern' and 'offset' fields
            context: Screen context
            min_confidence: Minimum confidence threshold

        Returns:
            MatchResult if anchor found and target exists at offset
        """
        if not isinstance(target, dict):
            return None

        anchor_pattern = target.get("anchor_pattern")
        offset_x = target.get("offset_x", 0)
        offset_y = target.get("offset_y", 0)
        target_size = target.get("target_size")  # (width, height)

        if not anchor_pattern or not isinstance(anchor_pattern, Pattern):
            return None

        try:
            # Find anchor using visual pattern matching
            anchor_result = self.visual_strategy.find(anchor_pattern, context, min_confidence)
            if not anchor_result:
                return None

            # Calculate target region relative to anchor
            anchor_center_x = anchor_result.region.x + anchor_result.region.width // 2
            anchor_center_y = anchor_result.region.y + anchor_result.region.height // 2

            target_x = anchor_center_x + offset_x
            target_y = anchor_center_y + offset_y

            # Use target size if provided, otherwise use anchor size
            if target_size:
                target_w, target_h = target_size
            else:
                target_w = anchor_result.region.width
                target_h = anchor_result.region.height

            # Adjust to top-left corner
            target_x -= target_w // 2
            target_y -= target_h // 2

            # Create result with reduced confidence (since we're inferring position)
            relative_confidence = anchor_result.confidence * 0.9

            return MatchResult(
                region=Region(x=target_x, y=target_y, width=target_w, height=target_h),
                confidence=relative_confidence,
                strategy_name=self.get_name(),
                metadata={
                    "anchor_region": str(anchor_result.region),
                    "offset": (offset_x, offset_y),
                },
            )

        except Exception as e:
            logger.error(f"RelativePositionStrategy error: {e}", exc_info=True)
            return None

    def get_name(self) -> str:
        """Get strategy name.

        Returns:
            Strategy name
        """
        return "RelativePosition"

    def can_handle(self, target: Pattern | dict[str, Any]) -> bool:
        """Check if target has anchor and offset.

        Args:
            target: Target to check

        Returns:
            True if target has anchor_pattern
        """
        return isinstance(target, dict) and "anchor_pattern" in target


class ColorRegionStrategy(LocatorStrategy):
    """Find elements by color patterns.

    Finds regions with specific color characteristics.
    Useful for finding highlighted elements, color-coded status indicators, etc.
    """

    def __init__(self) -> None:
        """Initialize color region strategy."""
        pass

    def find(
        self,
        target: Pattern | dict[str, Any],
        context: ScreenContext,
        min_confidence: float = 0.7,
    ) -> MatchResult | None:
        """Find region by color pattern.

        Args:
            target: Dict with 'color_range' (HSV min/max) and optional 'min_area'
            context: Screen context
            min_confidence: Minimum confidence threshold

        Returns:
            MatchResult if color region found
        """
        if not isinstance(target, dict):
            return None

        color_range = target.get("color_range")  # ((h_min, s_min, v_min), (h_max, s_max, v_max))
        min_area = target.get("min_area", 100)

        if not color_range:
            return None

        try:
            # Convert screenshot to HSV
            screenshot_hsv = cv2.cvtColor(context.screenshot, cv2.COLOR_BGR2HSV)

            # Create mask for color range
            lower_bound = np.array(color_range[0], dtype=np.uint8)
            upper_bound = np.array(color_range[1], dtype=np.uint8)
            mask = cv2.inRange(screenshot_hsv, lower_bound, upper_bound)

            # Find contours in mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            # Find largest contour above minimum area
            largest_contour = None
            largest_area = 0.0

            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_area and area > largest_area:
                    largest_contour = contour
                    largest_area = area

            if largest_contour is None:
                return None

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Calculate confidence based on color match percentage
            roi_mask = mask[y : y + h, x : x + w]
            match_pixels = np.sum(roi_mask > 0)
            total_pixels = w * h
            confidence = float(match_pixels) / float(total_pixels) if total_pixels > 0 else 0.0

            if confidence < min_confidence:
                return None

            return MatchResult(
                region=Region(x=x, y=y, width=w, height=h),
                confidence=confidence,
                strategy_name=self.get_name(),
                metadata={"color_range": color_range, "area": largest_area},
            )

        except Exception as e:
            logger.error(f"ColorRegionStrategy error: {e}", exc_info=True)
            return None

    def get_name(self) -> str:
        """Get strategy name.

        Returns:
            Strategy name
        """
        return "ColorRegion"

    def can_handle(self, target: Pattern | dict[str, Any]) -> bool:
        """Check if target has color range.

        Args:
            target: Target to check

        Returns:
            True if target has color_range
        """
        return isinstance(target, dict) and "color_range" in target


class StructuralStrategy(LocatorStrategy):
    """Find elements by visual structure (buttons, inputs, etc.).

    Uses edge detection and contour analysis to identify UI element types
    based on their visual structure rather than exact appearance.
    """

    def __init__(self) -> None:
        """Initialize structural strategy."""
        pass

    def find(
        self,
        target: Pattern | dict[str, Any],
        context: ScreenContext,
        min_confidence: float = 0.7,
    ) -> MatchResult | None:
        """Find element by structural characteristics.

        Args:
            target: Dict with 'element_type' (button, input, etc.) and size constraints
            context: Screen context
            min_confidence: Minimum confidence threshold

        Returns:
            MatchResult if structural match found
        """
        if not isinstance(target, dict):
            return None

        element_type = target.get("element_type")  # 'button', 'input', 'checkbox', etc.
        min_width = target.get("min_width", 50)
        max_width = target.get("max_width", 500)
        min_height = target.get("min_height", 20)
        max_height = target.get("max_height", 100)

        if not element_type:
            return None

        try:
            # Convert to grayscale
            screenshot_gray = cv2.cvtColor(context.screenshot, cv2.COLOR_BGR2GRAY)

            # Apply edge detection
            edges = cv2.Canny(screenshot_gray, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            candidates = []

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Filter by size constraints
                if not (min_width <= w <= max_width and min_height <= h <= max_height):
                    continue

                # Calculate aspect ratio
                aspect_ratio = float(w) / float(h) if h > 0 else 0

                # Score based on element type expectations
                confidence = 0.0

                if element_type == "button":
                    # Buttons typically have 1.5:1 to 4:1 aspect ratio
                    if 1.5 <= aspect_ratio <= 4.0:
                        confidence = 0.8
                elif element_type == "input":
                    # Input fields typically have wider aspect ratio
                    if 3.0 <= aspect_ratio <= 10.0:
                        confidence = 0.8
                elif element_type == "checkbox":
                    # Checkboxes are roughly square
                    if 0.8 <= aspect_ratio <= 1.2:
                        confidence = 0.8

                if confidence >= min_confidence:
                    candidates.append(
                        (
                            Region(x=x, y=y, width=w, height=h),
                            confidence,
                            {"aspect_ratio": aspect_ratio},
                        )
                    )

            if not candidates:
                return None

            # Return first (highest confidence) candidate
            # Could be enhanced to prefer candidates in certain screen regions
            region, confidence, metadata = candidates[0]

            return MatchResult(
                region=region,
                confidence=confidence,
                strategy_name=self.get_name(),
                metadata={**metadata, "element_type": element_type},
            )

        except Exception as e:
            logger.error(f"StructuralStrategy error: {e}", exc_info=True)
            return None

    def get_name(self) -> str:
        """Get strategy name.

        Returns:
            Strategy name
        """
        return "Structural"

    def can_handle(self, target: Pattern | dict[str, Any]) -> bool:
        """Check if target has element type.

        Args:
            target: Target to check

        Returns:
            True if target has element_type
        """
        return isinstance(target, dict) and "element_type" in target
