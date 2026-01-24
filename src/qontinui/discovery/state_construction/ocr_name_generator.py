"""OCR-based Name Generator.

Generates meaningful names for states, images, and regions using OCR
and semantic analysis.

This module implements Phase 3.2 of the DETECTION_MIGRATION_PLAN, providing
intelligent naming strategies for automatically detected GUI elements.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    pass

try:
    import pytesseract

    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import easyocr

    HAS_EASYOCR = True
except (ImportError, OSError):
    # OSError can happen when PyTorch DLLs fail to load on Windows
    HAS_EASYOCR = False


class OCRNameGenerator:
    """Generates meaningful names using OCR text extraction.

    Uses a priority-based naming strategy:
    1. OCR text (window titles, button labels, headings)
    2. Semantic descriptions (title_bar, button, panel)
    3. Position-based fallbacks (element_120_450)

    Supports both pytesseract and easyocr engines with automatic fallback.

    Example:
        >>> generator = OCRNameGenerator(engine='auto')
        >>> name = generator.generate_name_from_image(button_image, context='button')
        >>> print(name)
        'save_button'

        >>> state_name = generator.generate_state_name(screenshot)
        >>> print(state_name)
        'inventory_screen'
    """

    def __init__(self, engine: str = "auto") -> None:
        """Initialize OCR name generator.

        Args:
            engine: OCR engine to use - 'tesseract', 'easyocr', or 'auto'.
                   'auto' tries easyocr first, then tesseract.

        Raises:
            ValueError: If no OCR engine is available.
        """
        self.engine = self._select_engine(engine)
        self.reader: easyocr.Reader | None = None

        if self.engine == "easyocr" and HAS_EASYOCR:
            # Lazy initialization of EasyOCR reader
            self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def _select_engine(self, requested: str) -> str:
        """Select OCR engine based on availability.

        Args:
            requested: Requested engine name

        Returns:
            Selected engine name

        Raises:
            ValueError: If no OCR engine is available
        """
        if requested == "auto":
            if HAS_EASYOCR:
                return "easyocr"
            elif HAS_TESSERACT:
                return "tesseract"
            else:
                raise ValueError(
                    "No OCR engine available. Install either 'pytesseract' or 'easyocr'.\n"
                    "  pip install pytesseract  # Requires tesseract binary\n"
                    "  pip install easyocr      # Pure Python, includes models"
                )

        if requested == "tesseract":
            if not HAS_TESSERACT:
                raise ValueError("Tesseract not available. Install: pip install pytesseract")
            return "tesseract"

        if requested == "easyocr":
            if not HAS_EASYOCR:
                raise ValueError("EasyOCR not available. Install: pip install easyocr")
            return "easyocr"

        # Fallback with auto-selection
        return self._select_engine("auto")

    def generate_name_from_image(self, image: np.ndarray, context: str = "generic") -> str:
        """Generate a name from an image using OCR.

        Extracts text from the image and converts it to a valid identifier.
        Falls back to context-based or position-based naming if OCR fails.

        Args:
            image: Image as numpy array (BGR or grayscale)
            context: Context hint for semantic naming - 'title_bar', 'button',
                    'panel', 'icon', 'header', 'element', etc.

        Returns:
            Generated name as valid Python identifier (lowercase, underscores)

        Example:
            >>> image = load_button_image()  # Contains text "Save File"
            >>> name = generator.generate_name_from_image(image, 'button')
            >>> print(name)
            'save_file_button'
        """
        # Extract text from image
        text = self._extract_text(image)

        if text:
            # Clean and sanitize text
            name = self._sanitize_text(text)

            # Add context suffix if name is too short
            if len(name) < 3:
                name = f"{context}_{name}" if name else context
            elif context != "generic" and not name.endswith(context):
                # Add context suffix for clarity
                name = f"{name}_{context}"

            return name

        # Fallback: no text found, use context-based naming
        return self._generate_fallback_name(image, context)

    def generate_state_name(
        self,
        screenshot: np.ndarray,
        detected_text_regions: list[dict] | None = None,
    ) -> str:
        """Generate a name for a state from its screenshot.

        Uses multiple strategies to find meaningful names:
        1. Title bar text (top 10% of screen)
        2. Prominent headings (large text near top)
        3. Pre-detected text regions (if provided)
        4. Fallback to hash-based identifier

        Args:
            screenshot: Full screenshot as numpy array
            detected_text_regions: Optional list of pre-detected text regions.
                                   Each dict should have keys: 'text', 'x', 'y',
                                   'width', 'height', 'area'

        Returns:
            State name as valid identifier

        Example:
            >>> screenshot = capture_screen()
            >>> name = generator.generate_state_name(screenshot)
            >>> print(name)
            'main_menu'
        """
        if screenshot is None or screenshot.size == 0:
            return "empty_state"

        h, w = screenshot.shape[:2]

        # Strategy 1: Check title bar region (top 10% of screen)
        title_bar_height = max(int(h * 0.1), 30)  # At least 30 pixels
        title_bar = screenshot[:title_bar_height, :]
        title_text = self._extract_text(title_bar)

        if title_text and len(title_text) > 3:
            sanitized = self._sanitize_text(title_text)
            if len(sanitized) >= 3:
                return sanitized

        # Strategy 2: Look for large/prominent text near top
        top_third_height = int(h * 0.33)
        top_third = screenshot[:top_third_height, :]
        prominent_text = self._extract_prominent_text(top_third)

        if prominent_text:
            sanitized = self._sanitize_text(prominent_text)
            if len(sanitized) >= 3:
                return sanitized

        # Strategy 3: Use detected text regions if provided
        if detected_text_regions:
            # Find largest text region in upper half
            top_regions = [r for r in detected_text_regions if r.get("y", 0) < h * 0.5]

            if top_regions:
                # Sort by area (largest first)
                largest = max(top_regions, key=lambda r: r.get("area", 0))
                text = largest.get("text", "")
                if text:
                    sanitized = self._sanitize_text(text)
                    if len(sanitized) >= 3:
                        return sanitized

        # Fallback: Generate name from screenshot hash
        return self._generate_hash_based_name(screenshot)

    def _extract_text(self, image: np.ndarray) -> str:
        """Extract all text from image using configured OCR engine.

        Args:
            image: Image as numpy array

        Returns:
            Extracted text as string (may be empty)
        """
        if image is None or image.size == 0:
            return ""

        try:
            if self.engine == "easyocr":
                return self._extract_text_easyocr(image)
            elif self.engine == "tesseract":
                return self._extract_text_tesseract(image)
        except Exception:
            # Silently fail on OCR errors (common with empty regions)
            # Could log this in production
            pass

        return ""

    def _extract_text_easyocr(self, image: np.ndarray) -> str:
        """Extract text using EasyOCR.

        Args:
            image: Image as numpy array

        Returns:
            Extracted text
        """
        if self.reader is None:
            return ""

        # EasyOCR expects RGB or grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.reader.readtext(image, detail=0)  # detail=0 returns only text
        text = " ".join(results)
        return text.strip()

    def _extract_text_tesseract(self, image: np.ndarray) -> str:
        """Extract text using Tesseract OCR.

        Args:
            image: Image as numpy array

        Returns:
            Extracted text
        """
        if not HAS_TESSERACT:
            return ""

        # Tesseract works better with grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        text = pytesseract.image_to_string(gray)
        return text.strip()  # type: ignore[no-any-return]

    def _extract_prominent_text(self, image: np.ndarray) -> str:
        """Extract the most prominent (largest) text from image.

        Identifies headings, titles, and other large text elements.

        Args:
            image: Image as numpy array

        Returns:
            Most prominent text found, or empty string
        """
        if image is None or image.size == 0:
            return ""

        try:
            if self.engine == "easyocr":
                return self._extract_prominent_easyocr(image)
            elif self.engine == "tesseract":
                return self._extract_prominent_tesseract(image)
        except Exception:
            pass

        return ""

    def _extract_prominent_easyocr(self, image: np.ndarray) -> str:
        """Extract prominent text using EasyOCR's bounding box sizes.

        Args:
            image: Image as numpy array

        Returns:
            Most prominent text
        """
        if self.reader is None:
            return ""

        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.reader.readtext(image)

        if not results:
            return ""

        # Score by bounding box area (larger = more prominent)
        def score_result(result) -> float:
            bbox = result[0]
            # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            points = np.array(bbox)
            width = np.max(points[:, 0]) - np.min(points[:, 0])
            height = np.max(points[:, 1]) - np.min(points[:, 1])
            return width * height  # type: ignore[no-any-return]

        best = max(results, key=score_result)
        return best[1].strip()  # type: ignore[no-any-return]

    def _extract_prominent_tesseract(self, image: np.ndarray) -> str:
        """Extract prominent text using Tesseract's font size detection.

        Args:
            image: Image as numpy array

        Returns:
            Most prominent text
        """
        if not HAS_TESSERACT:
            return ""

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Get detailed data including bounding boxes
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

        if not data or "text" not in data:
            return ""

        # Find text with largest height (font size indicator)
        best_text = ""
        max_height = 0

        for i, text in enumerate(data["text"]):
            if not text.strip():
                continue

            height = data["height"][i]
            confidence = data["conf"][i]

            # Only consider high-confidence detections
            if confidence > 60 and height > max_height:
                max_height = height
                best_text = text

        return best_text.strip()

    def _sanitize_text(self, text: str) -> str:
        """Sanitize text to valid Python identifier.

        Converts text to lowercase, replaces spaces/separators with underscores,
        removes special characters, and truncates to reasonable length.

        Args:
            text: Raw text to sanitize

        Returns:
            Valid identifier string

        Example:
            >>> generator._sanitize_text("Save File!")
            'save_file'
            >>> generator._sanitize_text("Player-1 Inventory")
            'player_1_inventory'
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Replace common separators with underscores
        text = re.sub(r"[\s\-./\\:]+", "_", text)

        # Remove special characters (keep only alphanumeric and underscores)
        text = re.sub(r"[^a-z0-9_]", "", text)

        # Remove consecutive underscores
        text = re.sub(r"_+", "_", text)

        # Strip leading/trailing underscores
        text = text.strip("_")

        # Ensure doesn't start with number (Python identifier rule)
        if text and text[0].isdigit():
            text = f"n_{text}"

        # Truncate to 50 characters at word boundary
        if len(text) > 50:
            text = text[:50]
            # Try to break at last underscore
            if "_" in text:
                text = text.rsplit("_", 1)[0]

        return text or "unnamed"

    def _generate_fallback_name(self, image: np.ndarray, context: str) -> str:
        """Generate fallback name when OCR fails.

        Uses context and image properties to create a meaningful name.

        Args:
            image: Image that failed OCR
            context: Element context

        Returns:
            Fallback identifier
        """
        # Use image dimensions in name for uniqueness
        h, w = image.shape[:2]

        # Create hash for uniqueness
        img_hash = hash(image.tobytes()) % 1000

        return f"{context}_{w}x{h}_{img_hash}"

    def _generate_hash_based_name(self, screenshot: np.ndarray) -> str:
        """Generate name based on screenshot content hash.

        Creates a deterministic name that's consistent for the same screenshot.

        Args:
            screenshot: Screenshot array

        Returns:
            Hash-based state name
        """
        # Use image properties for more readable hash
        h, w = screenshot.shape[:2]

        # Create stable hash
        content_hash = hash(screenshot.tobytes()) % 10000

        return f"state_{w}x{h}_{content_hash}"


class NameValidator:
    """Validates and refines generated names.

    Provides utilities for checking name validity, detecting conflicts,
    and suggesting alternatives.
    """

    @staticmethod
    def is_valid_identifier(name: str) -> bool:
        """Check if name is a valid Python identifier.

        Args:
            name: Name to validate

        Returns:
            True if valid identifier
        """
        if not name:
            return False

        # Check Python identifier rules
        if not name.isidentifier():
            return False

        # Check not a Python keyword
        import keyword

        if keyword.iskeyword(name):
            return False

        return True

    @staticmethod
    def is_meaningful(name: str, min_length: int = 3) -> bool:
        """Check if name is meaningful (not just hash/position).

        Args:
            name: Name to check
            min_length: Minimum length for meaningful name

        Returns:
            True if name appears meaningful
        """
        if len(name) < min_length:
            return False

        # Check if mostly numeric/hash-like
        numeric_ratio = sum(c.isdigit() for c in name) / len(name)
        if numeric_ratio > 0.5:
            return False

        # Check for common fallback patterns
        fallback_patterns = [
            r"^element_\d+",
            r"^state_\d+",
            r"^unnamed",
            r"^\d+x\d+",
        ]

        for pattern in fallback_patterns:
            if re.match(pattern, name):
                return False

        return True

    @staticmethod
    def suggest_alternative(name: str, existing_names: set[str]) -> str:
        """Suggest alternative name to avoid conflicts.

        Args:
            name: Original name
            existing_names: Set of existing names to avoid

        Returns:
            Alternative name (may append number suffix)
        """
        if name not in existing_names:
            return name

        # Try numbered suffixes
        counter = 2
        while f"{name}_{counter}" in existing_names:
            counter += 1

        return f"{name}_{counter}"


# Convenience functions for quick usage


def generate_element_name(image: np.ndarray, element_type: str = "element") -> str:
    """Generate name for a UI element image.

    Convenience function for one-off name generation without creating a generator.

    Args:
        image: Element image
        element_type: Type of element (button, icon, panel, etc.)

    Returns:
        Generated name
    """
    generator = OCRNameGenerator()
    return generator.generate_name_from_image(image, context=element_type)


def generate_state_name_from_screenshot(screenshot: np.ndarray) -> str:
    """Generate name for a state from screenshot.

    Convenience function for one-off state name generation.

    Args:
        screenshot: State screenshot

    Returns:
        Generated state name
    """
    generator = OCRNameGenerator()
    return generator.generate_state_name(screenshot)
