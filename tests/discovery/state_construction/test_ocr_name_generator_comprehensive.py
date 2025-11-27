"""Comprehensive tests for OCRNameGenerator with mocked OCR engines.

This test suite uses mocking to avoid external OCR dependencies (tesseract, easyocr)
while thoroughly testing the OCR name generator functionality.

Key test areas:
- OCR engine selection and initialization
- Text extraction with both engines (mocked)
- Text sanitization and validation
- Fallback naming strategies
- State name generation
- Empty/corrupt image handling
- Integration with mocked OCR
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from qontinui.src.qontinui.discovery.state_construction.ocr_name_generator import (
    NameValidator,
    OCRNameGenerator,
    generate_element_name,
    generate_state_name_from_screenshot,
)
from tests.fixtures.screenshot_fixtures import (
    ElementSpec,
    SyntheticScreenshotGenerator,
    create_button_screenshot,
)


class TestOCREngineSelection:
    """Test OCR engine selection and initialization."""

    @patch(
        "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_EASYOCR", True
    )
    @patch(
        "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_TESSERACT", True
    )
    def test_auto_prefers_easyocr(self):
        """Test that 'auto' mode prefers easyocr when available.

        Verifies:
            - EasyOCR is selected over tesseract when both available
        """
        with patch(
            "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.easyocr.Reader"
        ):
            generator = OCRNameGenerator(engine="auto")
            assert generator.engine == "easyocr"

    @patch(
        "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_EASYOCR", False
    )
    @patch(
        "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_TESSERACT", True
    )
    def test_auto_fallback_to_tesseract(self):
        """Test that 'auto' mode falls back to tesseract.

        Verifies:
            - Tesseract is used when easyocr unavailable
        """
        generator = OCRNameGenerator(engine="auto")
        assert generator.engine == "tesseract"

    @patch(
        "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_EASYOCR", False
    )
    @patch(
        "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_TESSERACT", False
    )
    def test_auto_raises_when_none_available(self):
        """Test error when no OCR engine available.

        Verifies:
            - ValueError raised with helpful message
        """
        with pytest.raises(ValueError) as exc_info:
            OCRNameGenerator(engine="auto")

        assert "no ocr engine available" in str(exc_info.value).lower()
        assert "pytesseract" in str(exc_info.value).lower()
        assert "easyocr" in str(exc_info.value).lower()

    @patch(
        "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_TESSERACT", False
    )
    def test_explicit_tesseract_unavailable(self):
        """Test error when requesting unavailable tesseract.

        Verifies:
            - ValueError raised when tesseract explicitly requested but unavailable
        """
        with pytest.raises(ValueError) as exc_info:
            OCRNameGenerator(engine="tesseract")

        assert "tesseract not available" in str(exc_info.value).lower()

    @patch(
        "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_EASYOCR", False
    )
    def test_explicit_easyocr_unavailable(self):
        """Test error when requesting unavailable easyocr.

        Verifies:
            - ValueError raised when easyocr explicitly requested but unavailable
        """
        with pytest.raises(ValueError) as exc_info:
            OCRNameGenerator(engine="easyocr")

        assert "easyocr not available" in str(exc_info.value).lower()

    @patch(
        "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_EASYOCR", True
    )
    @patch("qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.easyocr.Reader")
    def test_easyocr_reader_initialization(self, mock_reader_class):
        """Test that easyocr reader is initialized correctly.

        Verifies:
            - Reader is created with correct parameters
            - English language is specified
            - GPU is disabled for testing
        """
        generator = OCRNameGenerator(engine="easyocr")

        mock_reader_class.assert_called_once_with(["en"], gpu=False, verbose=False)
        assert generator.reader is not None


class TestTextExtractionMocked:
    """Test text extraction with mocked OCR engines."""

    @pytest.fixture
    def mock_generator_easyocr(self):
        """Create generator with mocked easyocr."""
        with patch(
            "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_EASYOCR",
            True,
        ):
            with patch(
                "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.easyocr.Reader"
            ) as mock_reader_class:
                mock_reader = Mock()
                mock_reader_class.return_value = mock_reader
                generator = OCRNameGenerator(engine="easyocr")
                generator.reader = mock_reader
                return generator

    @pytest.fixture
    def mock_generator_tesseract(self):
        """Create generator with mocked tesseract."""
        with patch(
            "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_TESSERACT",
            True,
        ):
            generator = OCRNameGenerator(engine="tesseract")
            return generator

    def test_extract_text_easyocr_success(self, mock_generator_easyocr):
        """Test successful text extraction with easyocr.

        Verifies:
            - Extracts text correctly
            - Joins multiple text regions
        """
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        # Mock easyocr results
        mock_generator_easyocr.reader.readtext.return_value = ["Save", "File", "As"]

        text = mock_generator_easyocr._extract_text(image)

        assert text == "Save File As"

    def test_extract_text_easyocr_empty(self, mock_generator_easyocr):
        """Test easyocr with no text found.

        Verifies:
            - Returns empty string when no text detected
        """
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        mock_generator_easyocr.reader.readtext.return_value = []

        text = mock_generator_easyocr._extract_text(image)

        assert text == ""

    @patch("qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.pytesseract")
    def test_extract_text_tesseract_success(self, mock_pytesseract, mock_generator_tesseract):
        """Test successful text extraction with tesseract.

        Verifies:
            - Extracts text correctly
            - Strips whitespace
        """
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        mock_pytesseract.image_to_string.return_value = "  Main Menu  \n"

        text = mock_generator_tesseract._extract_text(image)

        assert text == "Main Menu"

    @patch("qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.pytesseract")
    def test_extract_text_tesseract_empty(self, mock_pytesseract, mock_generator_tesseract):
        """Test tesseract with no text found.

        Verifies:
            - Returns empty string
        """
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        mock_pytesseract.image_to_string.return_value = ""

        text = mock_generator_tesseract._extract_text(image)

        assert text == ""

    def test_extract_text_empty_image(self, mock_generator_easyocr):
        """Test with empty/None image.

        Verifies:
            - Handles empty images gracefully
        """
        text = mock_generator_easyocr._extract_text(np.array([]))
        assert text == ""

        text = mock_generator_easyocr._extract_text(None)
        assert text == ""

    def test_extract_text_exception_handling(self, mock_generator_easyocr):
        """Test that OCR exceptions are caught.

        Verifies:
            - Returns empty string on OCR failure
            - No exception propagates
        """
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        mock_generator_easyocr.reader.readtext.side_effect = RuntimeError("OCR failed")

        text = mock_generator_easyocr._extract_text(image)

        assert text == ""


class TestProminentTextExtraction:
    """Test extraction of prominent text (headings, titles)."""

    @pytest.fixture
    def mock_generator_easyocr(self):
        """Create generator with mocked easyocr."""
        with patch(
            "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_EASYOCR",
            True,
        ):
            with patch(
                "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.easyocr.Reader"
            ) as mock_reader_class:
                mock_reader = Mock()
                mock_reader_class.return_value = mock_reader
                generator = OCRNameGenerator(engine="easyocr")
                generator.reader = mock_reader
                return generator

    def test_extract_prominent_easyocr_largest_bbox(self, mock_generator_easyocr):
        """Test that largest text region is selected.

        Verifies:
            - Text with largest bounding box is returned as prominent
        """
        image = np.zeros((200, 300, 3), dtype=np.uint8)

        # Mock results with different bbox sizes
        mock_generator_easyocr.reader.readtext.return_value = [
            ([[10, 10], [60, 10], [60, 30], [10, 30]], "Small", 0.9),  # 50x20 = 1000
            ([[10, 50], [200, 50], [200, 90], [10, 90]], "Large Heading", 0.95),  # 190x40 = 7600
            ([[10, 100], [80, 100], [80, 120], [10, 120]], "Medium", 0.85),  # 70x20 = 1400
        ]

        text = mock_generator_easyocr._extract_prominent_text(image)

        assert text == "Large Heading"

    def test_extract_prominent_easyocr_empty(self, mock_generator_easyocr):
        """Test with no text found.

        Verifies:
            - Returns empty string when no text detected
        """
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        mock_generator_easyocr.reader.readtext.return_value = []

        text = mock_generator_easyocr._extract_prominent_text(image)

        assert text == ""

    @patch(
        "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_TESSERACT", True
    )
    @patch("qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.pytesseract")
    def test_extract_prominent_tesseract_largest_height(self, mock_pytesseract):
        """Test tesseract prominence based on font height.

        Verifies:
            - Text with largest height (font size) is selected
        """
        generator = OCRNameGenerator(engine="tesseract")
        image = np.zeros((200, 300, 3), dtype=np.uint8)

        # Mock tesseract data
        mock_pytesseract.image_to_data.return_value = {
            "text": ["", "Small", "Large", "Medium", ""],
            "height": [0, 12, 36, 18, 0],
            "conf": [0, 85, 92, 88, 0],
        }

        text = generator._extract_prominent_text(image)

        assert text == "Large"

    @patch(
        "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_TESSERACT", True
    )
    @patch("qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.pytesseract")
    def test_extract_prominent_tesseract_confidence_filter(self, mock_pytesseract):
        """Test that low-confidence text is filtered out.

        Verifies:
            - Only high-confidence text is considered
        """
        generator = OCRNameGenerator(engine="tesseract")
        image = np.zeros((200, 300, 3), dtype=np.uint8)

        mock_pytesseract.image_to_data.return_value = {
            "text": ["LowConf", "HighConf"],
            "height": [50, 30],  # LowConf is larger
            "conf": [40, 85],  # But has low confidence
        }

        text = generator._extract_prominent_text(image)

        # Should pick HighConf despite smaller size
        assert text == "HighConf"


class TestGenerateNameFromImage:
    """Test generate_name_from_image with mocked OCR."""

    @pytest.fixture
    def mock_generator(self):
        """Create generator with mocked extraction."""
        with patch(
            "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_EASYOCR",
            True,
        ):
            with patch(
                "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.easyocr.Reader"
            ):
                generator = OCRNameGenerator(engine="easyocr")
                return generator

    def test_generate_with_extracted_text(self, mock_generator):
        """Test name generation when text is extracted.

        Verifies:
            - Extracted text is sanitized and used
            - Context suffix is added
        """
        image = np.zeros((40, 120, 3), dtype=np.uint8)

        # Mock text extraction
        with patch.object(mock_generator, "_extract_text", return_value="Save File"):
            name = mock_generator.generate_name_from_image(image, context="button")

        assert name == "save_file_button"

    def test_generate_short_text_adds_context(self, mock_generator):
        """Test that short text gets context prefix.

        Verifies:
            - Very short text gets context added
        """
        image = np.zeros((40, 120, 3), dtype=np.uint8)

        with patch.object(mock_generator, "_extract_text", return_value="OK"):
            name = mock_generator.generate_name_from_image(image, context="button")

        # Short text should get context
        assert "button" in name
        assert "ok" in name

    def test_generate_no_text_fallback(self, mock_generator):
        """Test fallback when no text extracted.

        Verifies:
            - Uses fallback naming strategy
            - Includes context and dimensions
        """
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        with patch.object(mock_generator, "_extract_text", return_value=""):
            name = mock_generator.generate_name_from_image(image, context="icon")

        assert "icon" in name
        # Should include dimensions or hash
        assert name.replace("_", "").replace("x", "").replace("icon", "").isalnum()

    def test_generate_with_generic_context(self, mock_generator):
        """Test with generic context.

        Verifies:
            - Generic context is handled correctly
        """
        image = np.zeros((50, 100, 3), dtype=np.uint8)

        with patch.object(mock_generator, "_extract_text", return_value="Label"):
            name = mock_generator.generate_name_from_image(image, context="generic")

        assert name == "label"


class TestGenerateStateName:
    """Test state name generation with mocked OCR."""

    @pytest.fixture
    def mock_generator(self):
        """Create generator with mocked extraction."""
        with patch(
            "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_EASYOCR",
            True,
        ):
            with patch(
                "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.easyocr.Reader"
            ):
                generator = OCRNameGenerator(engine="easyocr")
                return generator

    def test_generate_from_title_bar(self, mock_generator):
        """Test name generation from title bar text.

        Verifies:
            - Title bar region is checked first
            - Text is sanitized
        """
        screenshot = np.zeros((600, 800, 3), dtype=np.uint8)

        with patch.object(mock_generator, "_extract_text") as mock_extract:
            # First call: title bar has text
            mock_extract.return_value = "Settings Menu"

            name = mock_generator.generate_state_name(screenshot)

        assert name == "settings_menu"

    def test_generate_from_prominent_text(self, mock_generator):
        """Test fallback to prominent text when no title bar.

        Verifies:
            - Prominent text is used when title bar empty
        """
        screenshot = np.zeros((600, 800, 3), dtype=np.uint8)

        with patch.object(mock_generator, "_extract_text", return_value=""):
            with patch.object(
                mock_generator, "_extract_prominent_text", return_value="Main Dashboard"
            ):
                name = mock_generator.generate_state_name(screenshot)

        assert name == "main_dashboard"

    def test_generate_from_detected_regions(self, mock_generator):
        """Test using pre-detected text regions.

        Verifies:
            - Largest region in upper half is used
        """
        screenshot = np.zeros((600, 800, 3), dtype=np.uint8)

        regions = [
            {"text": "Footer Text", "x": 100, "y": 500, "width": 200, "height": 30, "area": 6000},
            {"text": "Inventory", "x": 100, "y": 50, "width": 300, "height": 40, "area": 12000},
            {"text": "Small", "x": 400, "y": 100, "width": 50, "height": 20, "area": 1000},
        ]

        with patch.object(mock_generator, "_extract_text", return_value=""):
            with patch.object(mock_generator, "_extract_prominent_text", return_value=""):
                name = mock_generator.generate_state_name(screenshot, detected_text_regions=regions)

        assert name == "inventory"

    def test_generate_empty_screenshot(self, mock_generator):
        """Test with empty screenshot.

        Verifies:
            - Returns 'empty_state' for None/empty array
        """
        assert mock_generator.generate_state_name(None) == "empty_state"
        assert mock_generator.generate_state_name(np.array([])) == "empty_state"

    def test_generate_fallback_to_hash(self, mock_generator):
        """Test fallback to hash-based name.

        Verifies:
            - Uses hash when all text extraction fails
        """
        screenshot = np.zeros((600, 800, 3), dtype=np.uint8)

        with patch.object(mock_generator, "_extract_text", return_value=""):
            with patch.object(mock_generator, "_extract_prominent_text", return_value=""):
                name = mock_generator.generate_state_name(screenshot)

        assert name.startswith("state_")
        assert "800x600" in name or "600x800" in name


class TestTextSanitizationComprehensive:
    """Comprehensive tests for text sanitization."""

    @pytest.fixture
    def generator(self):
        """Create generator (engine doesn't matter for sanitization)."""
        with patch(
            "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_EASYOCR",
            True,
        ):
            with patch(
                "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.easyocr.Reader"
            ):
                return OCRNameGenerator(engine="easyocr")

    def test_sanitize_special_chars_comprehensive(self, generator):
        """Test sanitization of various special characters.

        Verifies:
            - All special chars are removed or replaced
        """
        test_cases = [
            ("Hello@World", "hello_world"),
            ("Test#123", "test_123"),
            ("Price$99", "price_99"),
            ("50%Off", "50_off"),
            ("C++Code", "c_code"),
            ("Email: test@example.com", "email_test_example_com"),
        ]

        for input_text, expected in test_cases:
            result = generator._sanitize_text(input_text)
            assert result == expected, f"Failed for: {input_text}"

    def test_sanitize_unicode_characters(self, generator):
        """Test handling of unicode characters.

        Verifies:
            - Unicode chars are removed (not ASCII)
        """
        result = generator._sanitize_text("Café Menu")
        # Non-ASCII chars should be removed
        assert result.isascii()
        assert result.isidentifier()

    def test_sanitize_path_separators(self, generator):
        """Test handling of path-like strings.

        Verifies:
            - Path separators become underscores
        """
        assert generator._sanitize_text("File/Path/Here") == "file_path_here"
        assert generator._sanitize_text("C:\\Windows\\System") == "c_windows_system"

    def test_sanitize_multiple_spaces(self, generator):
        """Test collapsing multiple spaces.

        Verifies:
            - Multiple spaces become single underscore
        """
        assert generator._sanitize_text("Too    Many     Spaces") == "too_many_spaces"

    def test_sanitize_mixed_separators(self, generator):
        """Test mixed separator characters.

        Verifies:
            - All separators normalized to underscore
        """
        result = generator._sanitize_text("Mixed-Text_With.Separators/Here")
        assert result == "mixed_text_with_separators_here"

    def test_sanitize_leading_trailing(self, generator):
        """Test removal of leading/trailing special chars.

        Verifies:
            - Leading/trailing underscores removed
        """
        assert generator._sanitize_text("_Leading") == "leading"
        assert generator._sanitize_text("Trailing_") == "trailing"
        assert generator._sanitize_text("__Both__") == "both"

    def test_sanitize_numeric_start_prefix(self, generator):
        """Test prefixing numbers at start.

        Verifies:
            - Numbers at start get 'n_' prefix
        """
        result = generator._sanitize_text("123Test")
        assert result == "n_123test"
        assert result.isidentifier()

    def test_sanitize_truncation_at_underscore(self, generator):
        """Test that truncation respects word boundaries.

        Verifies:
            - Long text truncated at underscore
        """
        long_text = "this_is_a_very_long_text_that_needs_to_be_truncated_properly"
        result = generator._sanitize_text(long_text)

        assert len(result) <= 50
        # Should end at underscore boundary
        assert not result.endswith("_")

    def test_sanitize_all_special_chars_becomes_unnamed(self, generator):
        """Test that pure special chars become 'unnamed'.

        Verifies:
            - Returns 'unnamed' when no valid chars remain
        """
        assert generator._sanitize_text("!@#$%^&*()") == "unnamed"
        assert generator._sanitize_text("     ") == "unnamed"
        assert generator._sanitize_text("___") == "unnamed"


class TestNameValidatorComprehensive:
    """Comprehensive tests for NameValidator."""

    def test_is_valid_identifier_edge_cases(self):
        """Test edge cases for identifier validation.

        Verifies:
            - Various edge cases are handled correctly
        """
        # Valid cases
        assert NameValidator.is_valid_identifier("_private")
        assert NameValidator.is_valid_identifier("__dunder__")
        assert NameValidator.is_valid_identifier("name123")
        assert NameValidator.is_valid_identifier("CamelCase")
        assert NameValidator.is_valid_identifier("snake_case")

        # Invalid cases
        assert not NameValidator.is_valid_identifier("123start")
        assert not NameValidator.is_valid_identifier("has-hyphen")
        assert not NameValidator.is_valid_identifier("has space")
        assert not NameValidator.is_valid_identifier("has.dot")
        assert not NameValidator.is_valid_identifier("class")  # keyword
        assert not NameValidator.is_valid_identifier("")

    def test_is_meaningful_comprehensive(self):
        """Test comprehensive meaningful name detection.

        Verifies:
            - Various patterns are correctly classified
        """
        # Meaningful names
        meaningful = [
            "save_button",
            "main_menu",
            "inventory_screen",
            "player_health",
            "settings_dialog",
        ]
        for name in meaningful:
            assert NameValidator.is_meaningful(name), f"Should be meaningful: {name}"

        # Not meaningful (fallback patterns)
        not_meaningful = [
            "element_12345",
            "state_789",
            "unnamed",
            "600x800",
            "element_456",
            "ab",  # Too short
            "123456789",  # Mostly numeric
        ]
        for name in not_meaningful:
            assert not NameValidator.is_meaningful(name), f"Should not be meaningful: {name}"

    def test_suggest_alternative_sequential(self):
        """Test sequential alternative suggestions.

        Verifies:
            - Correctly handles multiple conflicts
        """
        existing = {"button", "button_2", "button_3", "button_4"}

        result = NameValidator.suggest_alternative("button", existing)
        assert result == "button_5"

    def test_suggest_alternative_gaps(self):
        """Test alternative with gaps in numbering.

        Verifies:
            - Fills gaps in numbering sequence
        """
        existing = {"button", "button_2", "button_4"}  # button_3 missing

        result = NameValidator.suggest_alternative("button", existing)
        # Should be button_3 (fills gap) or button_5 (next available)
        # Actually just increments from base, so button_3
        assert result == "button_3"


class TestEmptyAndCorruptImages:
    """Test handling of empty and corrupt images."""

    @pytest.fixture
    def mock_generator(self):
        """Create generator with mocked extraction."""
        with patch(
            "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_EASYOCR",
            True,
        ):
            with patch(
                "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.easyocr.Reader"
            ):
                return OCRNameGenerator(engine="easyocr")

    def test_none_image(self, mock_generator):
        """Test with None image.

        Verifies:
            - Returns empty string, doesn't crash
        """
        text = mock_generator._extract_text(None)
        assert text == ""

    def test_empty_array(self, mock_generator):
        """Test with empty numpy array.

        Verifies:
            - Handles empty array gracefully
        """
        text = mock_generator._extract_text(np.array([]))
        assert text == ""

    def test_single_pixel(self, mock_generator):
        """Test with single pixel image.

        Verifies:
            - Handles minimal image
        """
        with patch.object(mock_generator, "_extract_text_easyocr", return_value=""):
            tiny = np.zeros((1, 1, 3), dtype=np.uint8)
            name = mock_generator.generate_name_from_image(tiny, "element")

        assert name.isidentifier()

    def test_wrong_dtype(self, mock_generator):
        """Test with unexpected dtype.

        Verifies:
            - Handles different dtypes gracefully
        """
        with patch.object(mock_generator, "_extract_text_easyocr", return_value=""):
            float_img = np.random.random((100, 100, 3)).astype(np.float64)
            # Should handle without crashing
            name = mock_generator.generate_name_from_image(float_img, "element")

        assert isinstance(name, str)

    def test_wrong_dimensions(self, mock_generator):
        """Test with 1D or 4D array.

        Verifies:
            - Handles unusual dimensions
        """
        with patch.object(mock_generator, "_extract_text_easyocr", return_value=""):
            # 1D array
            arr_1d = np.zeros(100, dtype=np.uint8)
            name = mock_generator.generate_name_from_image(arr_1d, "element")
            assert isinstance(name, str)


class TestConvenienceFunctionsWithMocking:
    """Test convenience functions with mocked OCR."""

    @patch(
        "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_EASYOCR", True
    )
    @patch("qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.easyocr.Reader")
    def test_generate_element_name_convenience(self, mock_reader_class):
        """Test convenience function for element names.

        Verifies:
            - Function creates generator and generates name
        """
        mock_reader = Mock()
        mock_reader.readtext.return_value = []
        mock_reader_class.return_value = mock_reader

        image = np.zeros((50, 100, 3), dtype=np.uint8)
        name = generate_element_name(image, "button")

        assert isinstance(name, str)
        assert name.isidentifier()

    @patch(
        "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_EASYOCR", True
    )
    @patch("qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.easyocr.Reader")
    def test_generate_state_name_convenience(self, mock_reader_class):
        """Test convenience function for state names.

        Verifies:
            - Function creates generator and generates name
        """
        mock_reader = Mock()
        mock_reader.readtext.return_value = []
        mock_reader_class.return_value = mock_reader

        screenshot = np.zeros((600, 800, 3), dtype=np.uint8)
        name = generate_state_name_from_screenshot(screenshot)

        assert isinstance(name, str)
        assert name.isidentifier()


class TestIntegrationWithSyntheticScreenshots:
    """Integration tests using synthetic screenshots."""

    @pytest.fixture
    def mock_generator(self):
        """Create generator with mocked extraction."""
        with patch(
            "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_EASYOCR",
            True,
        ):
            with patch(
                "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.easyocr.Reader"
            ):
                return OCRNameGenerator(engine="easyocr")

    def test_button_screenshot_naming(self, mock_generator):
        """Test naming from synthetic button screenshot.

        Verifies:
            - Generates valid names for button images
        """
        screenshot, buttons = create_button_screenshot(num_buttons=3)

        with patch.object(mock_generator, "_extract_text", return_value="Main Menu"):
            name = mock_generator.generate_state_name(screenshot)

        assert name == "main_menu"

    def test_multiple_elements_consistent_naming(self, mock_generator):
        """Test that same screenshot produces same name.

        Verifies:
            - Naming is deterministic
        """
        generator = SyntheticScreenshotGenerator()
        elements = [ElementSpec("button", x=100, y=100, width=120, height=40, text="Submit")]
        screenshot = generator.generate(width=600, height=400, elements=elements)

        with patch.object(mock_generator, "_extract_text", return_value=""):
            name1 = mock_generator.generate_state_name(screenshot)
            name2 = mock_generator.generate_state_name(screenshot)

        assert name1 == name2


class TestErrorRecovery:
    """Test error recovery and robustness."""

    @pytest.fixture
    def mock_generator(self):
        """Create generator with mocked extraction."""
        with patch(
            "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.HAS_EASYOCR",
            True,
        ):
            with patch(
                "qontinui.src.qontinui.discovery.state_construction.ocr_name_generator.easyocr.Reader"
            ):
                return OCRNameGenerator(engine="easyocr")

    def test_ocr_exception_recovery(self, mock_generator):
        """Test recovery from OCR exceptions.

        Verifies:
            - Falls back gracefully when OCR throws exception
        """
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        with patch.object(
            mock_generator, "_extract_text_easyocr", side_effect=RuntimeError("OCR failed")
        ):
            # Should not crash, should use fallback
            name = mock_generator.generate_name_from_image(image, "button")

        assert isinstance(name, str)
        assert name.isidentifier()
        assert "button" in name

    def test_cv2_exception_recovery(self, mock_generator):
        """Test recovery from cv2 exceptions.

        Verifies:
            - Handles image processing errors
        """
        # Corrupt image data
        corrupt = np.array([[1, 2], [3, 4]], dtype=np.uint8)  # Too small for most operations

        with patch.object(mock_generator, "_extract_text", return_value=""):
            # Should not crash
            name = mock_generator.generate_name_from_image(corrupt, "element")

        assert isinstance(name, str)
        assert name.isidentifier()
