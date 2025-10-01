"""OCR-based semantic processor for text detection."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

try:
    import pytesseract

    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import easyocr

    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

from ..core import PixelLocation, SemanticObject, SemanticScene
from ..core.semantic_object import ObjectType
from .base import SemanticProcessor


class OCRProcessor(SemanticProcessor):
    """Semantic processor focused on text detection and recognition.

    Can use either Tesseract or EasyOCR depending on availability.
    """

    def __init__(self, engine: str = "auto"):
        """Initialize OCR processor.

        Args:
            engine: OCR engine to use ('tesseract', 'easyocr', or 'auto')
        """
        super().__init__()

        self.engine = self._select_engine(engine)
        self.reader = None

        if self.engine == "easyocr" and HAS_EASYOCR:
            # Initialize EasyOCR reader (lazy loading)
            self.reader = easyocr.Reader(["en"])

    def _select_engine(self, requested: str) -> str:
        """Select OCR engine based on availability.

        Args:
            requested: Requested engine

        Returns:
            Selected engine name
        """
        if requested == "auto":
            if HAS_EASYOCR:
                return "easyocr"
            elif HAS_TESSERACT:
                return "tesseract"
            else:
                return "none"

        if requested == "tesseract" and HAS_TESSERACT:
            return "tesseract"
        elif requested == "easyocr" and HAS_EASYOCR:
            return "easyocr"

        # Fallback
        if HAS_EASYOCR:
            return "easyocr"
        elif HAS_TESSERACT:
            return "tesseract"
        else:
            return "none"

    def process(self, screenshot: np.ndarray[Any, Any]) -> SemanticScene:
        """Process screenshot for text detection.

        Args:
            screenshot: Screenshot as numpy array

        Returns:
            SemanticScene with detected text objects
        """
        start_time = time.time()
        scene = SemanticScene(source_image=screenshot)

        if self.engine == "none":
            # No OCR engine available
            return scene

        # Convert to grayscale if needed
        if len(screenshot.shape) == 3:
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        else:
            gray = screenshot

        if self.engine == "tesseract":
            self._process_with_tesseract(gray, scene)
        elif self.engine == "easyocr":
            self._process_with_easyocr(screenshot, scene)  # EasyOCR works better with color

        self._record_processing_time(start_time)
        return scene

    def _process_with_tesseract(self, image: np.ndarray[Any, Any], scene: SemanticScene) -> None:
        """Process image using Tesseract OCR.

        Args:
            image: Grayscale image
            scene: Scene to add objects to
        """
        if not HAS_TESSERACT:
            return

        # Get detailed data including bounding boxes
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            text = data["text"][i].strip()
            if not text:
                continue

            # Get confidence (Tesseract uses 0-100 scale)
            confidence = float(data["conf"][i]) / 100.0

            if confidence < self._config.min_confidence:
                continue

            # Get bounding box
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            # Create pixel location
            location = PixelLocation.from_rectangle(x, y, w, h)

            # Create semantic object
            obj = SemanticObject(
                location=location,
                description=f"Text: {text}",
                confidence=confidence,
                object_type=ObjectType.TEXT,
            )
            obj.set_text(text)

            # Detect if it's a heading (larger text)
            if h > 30:  # Arbitrary threshold for heading
                obj.set_object_type("heading")

            scene.add_object(obj)

    def _process_with_easyocr(self, image: np.ndarray[Any, Any], scene: SemanticScene) -> None:
        """Process image using EasyOCR.

        Args:
            image: Color image
            scene: Scene to add objects to
        """
        if not HAS_EASYOCR or self.reader is None:
            return

        # Run OCR
        results = self.reader.readtext(image)

        for detection in results:
            bbox, text, confidence = detection

            if confidence < self._config.min_confidence:
                continue

            # EasyOCR returns polygon points
            # Convert to bounding rectangle
            points = np.array(bbox)
            x_min = int(np.min(points[:, 0]))
            y_min = int(np.min(points[:, 1]))
            x_max = int(np.max(points[:, 0]))
            y_max = int(np.max(points[:, 1]))

            # Create pixel location from polygon or rectangle
            if len(bbox) == 4:
                # Use polygon if 4 points provided
                location = PixelLocation.from_polygon([(int(p[0]), int(p[1])) for p in bbox])
            else:
                # Fall back to rectangle
                location = PixelLocation.from_rectangle(x_min, y_min, x_max - x_min, y_max - y_min)

            # Create semantic object
            obj = SemanticObject(
                location=location,
                description=f"Text: {text}",
                confidence=confidence,
                object_type=ObjectType.TEXT,
            )
            obj.set_text(text)

            # Try to classify text type
            self._classify_text_object(obj, text, y_max - y_min)

            scene.add_object(obj)

    def _classify_text_object(self, obj: SemanticObject, text: str, height: int) -> None:
        """Classify text object based on content and appearance.

        Args:
            obj: Semantic object to classify
            text: Text content
            height: Text height in pixels
        """
        text_lower = text.lower()

        # Check for button-like text
        button_keywords = [
            "ok",
            "cancel",
            "submit",
            "apply",
            "close",
            "save",
            "open",
            "delete",
            "add",
            "remove",
            "yes",
            "no",
        ]
        if any(keyword in text_lower for keyword in button_keywords):
            obj.object_type = ObjectType.BUTTON
            obj.set_interactable(True)
            return

        # Check for link patterns
        if text.startswith("http") or text.startswith("www."):
            obj.object_type = ObjectType.LINK
            obj.set_interactable(True)
            return

        # Check for heading (large text)
        if height > 30:
            obj.object_type = ObjectType.HEADING
            return

        # Check for paragraph (multi-word text)
        if len(text.split()) > 5:
            obj.object_type = ObjectType.PARAGRAPH
            return

    def get_supported_object_types(self) -> set[str]:
        """Get supported object types.

        Returns:
            Set of object type names this processor can detect
        """
        return {
            ObjectType.TEXT.value,
            ObjectType.HEADING.value,
            ObjectType.PARAGRAPH.value,
            ObjectType.BUTTON.value,
            ObjectType.LINK.value,
        }

    def supports_incremental_processing(self) -> bool:
        """Check if incremental processing is supported.

        Returns:
            False - OCR doesn't benefit much from incremental processing
        """
        return False
