"""
Vision-based extraction using Computer Vision and Machine Learning.

Analyzes screenshots to detect UI elements using:
- Classical CV (OpenCV): edges, contours, templates
- ML Models: object detection, icon recognition
- OCR: text detection and recognition
- Segmentation: SAM for precise boundaries
"""

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..abstract_extractor import (
    AbstractExtractor,
    ExtractedElement,
    ExtractedState,
    ExtractionContext,
    ExtractionResult,
)

if TYPE_CHECKING:
    from ..extractor_config import ExtractorConfig, VisionConfig

logger = logging.getLogger(__name__)


class VisionExtractor(AbstractExtractor):
    """
    Vision-based extraction using Computer Vision and ML.

    Analyzes screenshots to detect UI elements. Works with ANY GUI
    since it only requires screenshots as input.

    This extractor combines multiple detection methods:
    1. Classical CV: Edge detection, contour analysis
    2. ML Detection: Object detection models (optional)
    3. OCR: Text detection and recognition
    4. Segmentation: SAM/segment-anything (optional)

    Example:
        >>> extractor = VisionExtractor()
        >>> context = ExtractionContext(screenshot_path=Path("screenshot.png"))
        >>> config = ExtractorConfig()
        >>> result = await extractor.extract(context, config)
    """

    def __init__(self) -> None:
        """Initialize the vision extractor."""
        self._ocr_engine = None  # Lazy loaded
        self._ml_detector = None  # Lazy loaded
        self._segmenter = None  # Lazy loaded
        self._element_counter = 0

    async def extract(
        self,
        context: ExtractionContext,
        config: "ExtractorConfig",
    ) -> ExtractionResult:
        """
        Perform vision-based extraction.

        Args:
            context: Extraction context with screenshot path
            config: Extractor configuration

        Returns:
            ExtractionResult with detected elements and states
        """
        await self.validate_context(context)

        extraction_id = str(uuid.uuid4())
        result = ExtractionResult(
            extraction_id=extraction_id,
            extraction_method="vision",
            context=context,
        )

        try:
            # Load screenshot
            screenshot = self._load_screenshot(context)
            if screenshot is None:
                result.add_error("Failed to load screenshot")
                result.complete()
                return result

            # Extract elements using configured methods
            elements = await self.extract_elements(context, config)
            result.elements = self.filter_elements(elements, config)

            # Detect states if enabled
            if config.detect_states:
                states = await self.extract_states(context, config, result.elements)
                result.states = states

            # Capture screenshot reference
            if context.screenshot_path:
                result.screenshots = [str(context.screenshot_path)]
                result.screenshots_dir = context.screenshot_path.parent

            result.complete()
            logger.info(
                f"Vision extraction complete: {len(result.elements)} elements, "
                f"{len(result.states)} states"
            )

        except Exception as e:
            logger.error(f"Vision extraction failed: {e}", exc_info=True)
            result.add_error(str(e))
            result.complete()

        return result

    async def extract_elements(
        self,
        context: ExtractionContext,
        config: "ExtractorConfig",
    ) -> list[ExtractedElement]:
        """
        Extract elements using vision techniques.

        Combines results from:
        - Classical CV detection
        - ML-based detection (if enabled)
        - OCR text detection (if enabled)
        """
        screenshot = self._load_screenshot(context)
        if screenshot is None:
            return []

        vision_config = config.vision
        all_elements: list[ExtractedElement] = []

        # Classical CV detection
        if vision_config.use_classical_cv:
            cv_elements = await self._detect_with_classical_cv(screenshot, vision_config)
            all_elements.extend(cv_elements)

        # ML-based detection
        if vision_config.use_ml_detection and vision_config.detection_model:
            ml_elements = await self._detect_with_ml(screenshot, vision_config)
            all_elements.extend(ml_elements)

        # OCR text detection
        if vision_config.use_ocr:
            ocr_elements = await self._detect_with_ocr(screenshot, vision_config)
            all_elements.extend(ocr_elements)

        # Deduplicate overlapping elements
        deduplicated = self._deduplicate_elements(all_elements, vision_config.iou_threshold)

        return deduplicated

    async def extract_states(
        self,
        context: ExtractionContext,
        config: "ExtractorConfig",
        elements: list[ExtractedElement] | None = None,
    ) -> list[ExtractedState]:
        """
        Detect UI states/regions from visual analysis.

        Uses element clustering and region detection to identify
        logical UI groupings.
        """
        screenshot = self._load_screenshot(context)
        if screenshot is None:
            return []

        if elements is None:
            elements = await self.extract_elements(context, config)

        # For now, create a single page state containing all elements
        # TODO: Implement more sophisticated state detection
        height, width = screenshot.shape[:2]

        page_state = ExtractedState(
            id="state_page_001",
            name="Page",
            state_type="page",
            bbox=(0, 0, width, height),
            element_ids=[e.id for e in elements],
            confidence=1.0,
            extraction_method="vision",
            detection_method="full_page",
        )

        return [page_state]

    async def capture_screenshot(
        self,
        context: ExtractionContext,
        region: tuple[int, int, int, int] | None = None,
    ) -> Path:
        """
        Capture or copy screenshot.

        For vision extraction, this typically copies the source screenshot
        or captures from screen if no screenshot is provided.
        """
        if context.screenshot_path and context.screenshot_path.exists():
            return context.screenshot_path

        # TODO: Implement screen capture via HAL
        raise NotImplementedError("Screen capture not implemented for VisionExtractor")

    @classmethod
    def supports_target(cls, context: ExtractionContext) -> bool:
        """Check if vision extraction can handle this target."""
        # Vision extraction works with any target that has a screenshot
        # or any desktop target (we can capture screenshot)
        if context.screenshot_path is not None:
            return True
        if context.platform in ("win32", "darwin", "linux"):
            return True
        return False

    @classmethod
    def get_name(cls) -> str:
        """Return extractor name."""
        return "vision"

    @classmethod
    def get_priority(cls) -> int:
        """Return priority (lower than DOM/Accessibility)."""
        return 10  # Fallback priority

    def _load_screenshot(self, context: ExtractionContext) -> np.ndarray | None:
        """Load screenshot from path."""
        if context.screenshot_path is None:
            return None
        if not context.screenshot_path.exists():
            logger.error(f"Screenshot not found: {context.screenshot_path}")
            return None

        screenshot = cv2.imread(str(context.screenshot_path))
        if screenshot is None:
            logger.error(f"Failed to load screenshot: {context.screenshot_path}")
            return None

        return screenshot

    async def _detect_with_classical_cv(
        self,
        screenshot: np.ndarray,
        config: "VisionConfig",
    ) -> list[ExtractedElement]:
        """
        Detect elements using classical computer vision.

        Uses edge detection and contour analysis to find UI elements.
        """
        elements: list[ExtractedElement] = []

        # Convert to grayscale
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, config.edge_detection_low, config.edge_detection_high)

        # Morphological operations to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            # Filter by area
            if area < config.contour_min_area:
                continue

            # Approximate to polygon
            epsilon = config.contour_approximation_epsilon * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Determine element type based on shape
            element_type = self._classify_shape(approx, w, h, config)

            self._element_counter += 1
            element = ExtractedElement(
                id=f"cv_element_{self._element_counter:04d}",
                element_type=element_type,
                bbox=(x, y, w, h),
                confidence=0.7,  # Base confidence for CV detection
                is_interactive=(element_type in ("button", "input")),
                extraction_method="vision",
                source_backend="classical_cv",
            )
            elements.append(element)

        logger.debug(f"Classical CV detected {len(elements)} elements")
        return elements

    def _classify_shape(
        self,
        approx: np.ndarray,
        width: int,
        height: int,
        config: "VisionConfig",
    ) -> str:
        """Classify detected shape as element type."""
        # Check if rectangular (likely button or input)
        if len(approx) == 4:
            aspect_ratio = width / height if height > 0 else 0

            # Check if button-like dimensions
            if (
                config.button_aspect_ratio_min < aspect_ratio < config.button_aspect_ratio_max
                and config.button_width_min < width < config.button_width_max
                and config.button_height_min < height < config.button_height_max
            ):
                return "button"

            # Could be input field
            if aspect_ratio > 3 and height < 50:
                return "input"

        return "container"

    async def _detect_with_ml(
        self,
        screenshot: np.ndarray,
        config: "VisionConfig",
    ) -> list[ExtractedElement]:
        """
        Detect elements using ML object detection model.

        TODO: Implement ML detection
        """
        # Placeholder for ML detection
        # Would use YOLO, DETR, or custom trained model
        logger.debug("ML detection not yet implemented")
        return []

    async def _detect_with_ocr(
        self,
        screenshot: np.ndarray,
        config: "VisionConfig",
    ) -> list[ExtractedElement]:
        """
        Detect text elements using OCR.

        TODO: Integrate with existing OCR from HAL
        """
        elements: list[ExtractedElement] = []

        try:
            # Lazy load OCR engine
            if self._ocr_engine is None:
                if config.ocr_engine == "easyocr":
                    import easyocr

                    self._ocr_engine = easyocr.Reader(["en"])
                else:
                    logger.warning(f"OCR engine {config.ocr_engine} not supported")
                    return []

            # Convert BGR to RGB for OCR
            rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)

            # Run OCR
            results = self._ocr_engine.readtext(rgb)

            for bbox, text, confidence in results:
                if confidence < config.text_confidence_threshold:
                    continue

                # Convert bbox format
                x_coords = [int(p[0]) for p in bbox]
                y_coords = [int(p[1]) for p in bbox]
                x = min(x_coords)
                y = min(y_coords)
                w = max(x_coords) - x
                h = max(y_coords) - y

                self._element_counter += 1
                element = ExtractedElement(
                    id=f"ocr_element_{self._element_counter:04d}",
                    element_type="text",
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    text=text,
                    is_interactive=False,
                    extraction_method="vision",
                    source_backend="ocr",
                )
                elements.append(element)

            logger.debug(f"OCR detected {len(elements)} text elements")

        except ImportError:
            logger.warning("EasyOCR not installed, skipping OCR detection")
        except Exception as e:
            logger.error(f"OCR detection failed: {e}")

        return elements

    def _deduplicate_elements(
        self,
        elements: list[ExtractedElement],
        iou_threshold: float,
    ) -> list[ExtractedElement]:
        """
        Remove duplicate/overlapping elements.

        Keeps the element with higher confidence when IoU exceeds threshold.
        """
        if not elements:
            return []

        # Sort by confidence (descending)
        sorted_elements = sorted(elements, key=lambda e: e.confidence, reverse=True)

        kept: list[ExtractedElement] = []

        for element in sorted_elements:
            # Check if this element overlaps with any kept element
            is_duplicate = False

            for kept_element in kept:
                if element.iou(kept_element) > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(element)

        logger.debug(
            f"Deduplicated {len(elements)} -> {len(kept)} elements "
            f"(IoU threshold: {iou_threshold})"
        )
        return kept
