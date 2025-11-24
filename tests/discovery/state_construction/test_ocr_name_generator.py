"""Tests for OCR Name Generator.

Tests naming strategies, text extraction, sanitization, and fallback behavior.
"""

import numpy as np
import pytest

from qontinui.discovery.state_construction.ocr_name_generator import (
    NameValidator,
    OCRNameGenerator,
    generate_element_name,
    generate_state_name_from_screenshot,
)


class TestOCRNameGenerator:
    """Test suite for OCR name generation."""

    @pytest.fixture
    def generator(self):
        """Create OCR name generator instance."""
        try:
            return OCRNameGenerator(engine="auto")
        except ValueError:
            pytest.skip("No OCR engine available (tesseract or easyocr)")

    def test_initialization_auto(self):
        """Test automatic engine selection."""
        try:
            generator = OCRNameGenerator(engine="auto")
            assert generator.engine in ["tesseract", "easyocr"]
        except ValueError:
            pytest.skip("No OCR engine available")

    def test_initialization_invalid_engine(self):
        """Test error handling for invalid engine."""
        with pytest.raises(ValueError):
            OCRNameGenerator(engine="invalid_engine")

    def test_sanitize_text_basic(self, generator):
        """Test basic text sanitization."""
        assert generator._sanitize_text("Hello World") == "hello_world"
        assert generator._sanitize_text("Save File") == "save_file"
        assert generator._sanitize_text("Player-1") == "player_1"

    def test_sanitize_text_special_characters(self, generator):
        """Test sanitization with special characters."""
        assert generator._sanitize_text("Click Me!") == "click_me"
        assert generator._sanitize_text("File/Path") == "file_path"
        assert generator._sanitize_text("Test:Value") == "test_value"
        assert generator._sanitize_text("Some-Text.Here") == "some_text_here"

    def test_sanitize_text_consecutive_separators(self, generator):
        """Test handling of consecutive separators."""
        assert generator._sanitize_text("Hello   World") == "hello_world"
        assert generator._sanitize_text("Test---Name") == "test_name"
        assert generator._sanitize_text("_Leading_Trailing_") == "leading_trailing"

    def test_sanitize_text_numeric_start(self, generator):
        """Test handling of names starting with numbers."""
        result = generator._sanitize_text("123 Main")
        assert result == "n_123_main"
        assert result.isidentifier()

    def test_sanitize_text_truncation(self, generator):
        """Test long text truncation."""
        long_text = "this is a very long text that should be truncated to fifty characters"
        result = generator._sanitize_text(long_text)
        assert len(result) <= 50
        assert result.isidentifier()

    def test_sanitize_text_empty(self, generator):
        """Test empty text handling."""
        assert generator._sanitize_text("") == "unnamed"
        assert generator._sanitize_text("   ") == "unnamed"
        assert generator._sanitize_text("!!!") == "unnamed"

    def test_generate_fallback_name(self, generator):
        """Test fallback name generation."""
        image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        name = generator._generate_fallback_name(image, "button")

        assert "button" in name
        assert "200x100" in name or "100x200" in name  # width x height or height x width
        assert name.replace("_", "").replace("x", "").replace("button", "").isdigit()

    def test_generate_hash_based_name(self, generator):
        """Test hash-based state name generation."""
        screenshot = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        name = generator._generate_hash_based_name(screenshot)

        assert name.startswith("state_")
        assert "800x600" in name or "600x800" in name
        assert name.replace("_", "").replace("x", "").replace("state", "").isdigit()

    def test_generate_hash_based_name_consistency(self, generator):
        """Test that same screenshot produces same name."""
        screenshot = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        name1 = generator._generate_hash_based_name(screenshot)
        name2 = generator._generate_hash_based_name(screenshot)

        assert name1 == name2

    def test_generate_name_from_image_fallback(self, generator):
        """Test name generation with fallback for blank image."""
        # Create blank image (OCR will fail)
        blank_image = np.zeros((100, 200, 3), dtype=np.uint8)
        name = generator.generate_name_from_image(blank_image, context="button")

        # Should use fallback strategy
        assert "button" in name
        assert name.isidentifier()

    def test_generate_state_name_empty(self, generator):
        """Test state name generation for empty screenshot."""
        name = generator.generate_state_name(None)
        assert name == "empty_state"

        empty = np.array([])
        name = generator.generate_state_name(empty)
        assert name == "empty_state"

    def test_generate_state_name_blank(self, generator):
        """Test state name generation for blank screenshot."""
        blank = np.zeros((600, 800, 3), dtype=np.uint8)
        name = generator.generate_state_name(blank)

        # Should generate hash-based name
        assert name.startswith("state_")
        assert name.isidentifier()

    def test_generate_state_name_with_regions(self, generator):
        """Test state name generation with pre-detected text regions."""
        screenshot = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

        # Mock detected text regions
        regions = [
            {"text": "Main Menu", "x": 100, "y": 50, "width": 200, "height": 40, "area": 8000},
            {"text": "Footer", "x": 100, "y": 550, "width": 150, "height": 20, "area": 3000},
        ]

        name = generator.generate_state_name(screenshot, detected_text_regions=regions)

        # Should use the prominent text region
        assert name == "main_menu"

    def test_extract_text_empty_image(self, generator):
        """Test text extraction from empty image."""
        empty = np.array([])
        text = generator._extract_text(empty)
        assert text == ""

    def test_extract_prominent_text_empty(self, generator):
        """Test prominent text extraction from empty image."""
        empty = np.array([])
        text = generator._extract_prominent_text(empty)
        assert text == ""


class TestNameValidator:
    """Test suite for name validation utilities."""

    def test_is_valid_identifier_valid(self):
        """Test validation of valid identifiers."""
        assert NameValidator.is_valid_identifier("valid_name")
        assert NameValidator.is_valid_identifier("button_123")
        assert NameValidator.is_valid_identifier("_private")
        assert NameValidator.is_valid_identifier("camelCase")

    def test_is_valid_identifier_invalid(self):
        """Test validation of invalid identifiers."""
        assert not NameValidator.is_valid_identifier("")
        assert not NameValidator.is_valid_identifier("123invalid")
        assert not NameValidator.is_valid_identifier("has-dash")
        assert not NameValidator.is_valid_identifier("has space")
        assert not NameValidator.is_valid_identifier("has.dot")

    def test_is_valid_identifier_keywords(self):
        """Test that Python keywords are rejected."""
        assert not NameValidator.is_valid_identifier("class")
        assert not NameValidator.is_valid_identifier("def")
        assert not NameValidator.is_valid_identifier("return")
        assert not NameValidator.is_valid_identifier("if")

    def test_is_meaningful_basic(self):
        """Test basic meaningful name detection."""
        assert NameValidator.is_meaningful("save_button")
        assert NameValidator.is_meaningful("main_menu")
        assert NameValidator.is_meaningful("inventory")

    def test_is_meaningful_too_short(self):
        """Test rejection of too-short names."""
        assert not NameValidator.is_meaningful("ab")
        assert not NameValidator.is_meaningful("x")
        assert NameValidator.is_meaningful("abc")

    def test_is_meaningful_numeric(self):
        """Test rejection of mostly-numeric names."""
        assert not NameValidator.is_meaningful("element_123456")
        assert not NameValidator.is_meaningful("12345")
        assert NameValidator.is_meaningful("button_1")  # Less than 50% numeric

    def test_is_meaningful_fallback_patterns(self):
        """Test rejection of fallback patterns."""
        assert not NameValidator.is_meaningful("element_123")
        assert not NameValidator.is_meaningful("state_456")
        assert not NameValidator.is_meaningful("unnamed")
        assert not NameValidator.is_meaningful("800x600")

    def test_suggest_alternative_no_conflict(self):
        """Test alternative suggestion when no conflict exists."""
        existing = {"other_name", "different_name"}
        result = NameValidator.suggest_alternative("new_name", existing)
        assert result == "new_name"

    def test_suggest_alternative_with_conflict(self):
        """Test alternative suggestion when conflict exists."""
        existing = {"button"}
        result = NameValidator.suggest_alternative("button", existing)
        assert result == "button_2"

    def test_suggest_alternative_multiple_conflicts(self):
        """Test alternative suggestion with multiple conflicts."""
        existing = {"button", "button_2", "button_3"}
        result = NameValidator.suggest_alternative("button", existing)
        assert result == "button_4"


class TestConvenienceFunctions:
    """Test convenience functions for quick usage."""

    def test_generate_element_name(self):
        """Test quick element name generation."""
        try:
            image = np.zeros((50, 100, 3), dtype=np.uint8)
            name = generate_element_name(image, "button")
            assert isinstance(name, str)
            assert name.isidentifier()
        except ValueError:
            pytest.skip("No OCR engine available")

    def test_generate_state_name_from_screenshot(self):
        """Test quick state name generation."""
        try:
            screenshot = np.zeros((600, 800, 3), dtype=np.uint8)
            name = generate_state_name_from_screenshot(screenshot)
            assert isinstance(name, str)
            assert name.isidentifier()
        except ValueError:
            pytest.skip("No OCR engine available")


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.fixture
    def generator(self):
        """Create generator or skip if unavailable."""
        try:
            return OCRNameGenerator(engine="auto")
        except ValueError:
            pytest.skip("No OCR engine available")

    def test_complete_workflow_button(self, generator):
        """Test complete workflow: image -> name -> validation."""
        # Create synthetic button image (blank for now)
        button_image = np.zeros((40, 120, 3), dtype=np.uint8)

        # Generate name
        name = generator.generate_name_from_image(button_image, context="button")

        # Validate
        assert NameValidator.is_valid_identifier(name)
        assert len(name) > 0

    def test_complete_workflow_state(self, generator):
        """Test complete workflow: screenshot -> state name -> validation."""
        # Create synthetic screenshot
        screenshot = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

        # Generate name
        name = generator.generate_state_name(screenshot)

        # Validate
        assert NameValidator.is_valid_identifier(name)
        assert len(name) > 0

    def test_name_uniqueness_handling(self, generator):
        """Test handling of duplicate names."""
        existing_names = set()

        # Generate multiple names
        for i in range(5):
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            name = generator.generate_name_from_image(image, context="element")

            # Ensure uniqueness
            unique_name = NameValidator.suggest_alternative(name, existing_names)
            existing_names.add(unique_name)

        # Should have 5 unique names
        assert len(existing_names) == 5


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def generator(self):
        """Create generator or skip if unavailable."""
        try:
            return OCRNameGenerator(engine="auto")
        except ValueError:
            pytest.skip("No OCR engine available")

    def test_very_small_image(self, generator):
        """Test with very small image."""
        tiny = np.zeros((5, 5, 3), dtype=np.uint8)
        name = generator.generate_name_from_image(tiny, "icon")
        assert name.isidentifier()

    def test_very_large_image(self, generator):
        """Test with very large image."""
        large = np.zeros((4000, 3000, 3), dtype=np.uint8)
        name = generator.generate_state_name(large)
        assert name.isidentifier()

    def test_grayscale_image(self, generator):
        """Test with grayscale image."""
        gray = np.zeros((100, 200), dtype=np.uint8)
        name = generator.generate_name_from_image(gray, "element")
        assert name.isidentifier()

    def test_unicode_text_sanitization(self, generator):
        """Test sanitization of unicode text."""
        # Simulate OCR result with unicode
        result = generator._sanitize_text("Café Menu")
        assert result.isascii()
        assert result.isidentifier()

    def test_all_special_characters(self, generator):
        """Test text that's entirely special characters."""
        result = generator._sanitize_text("!@#$%^&*()")
        assert result == "unnamed"

    def test_consistency_across_calls(self, generator):
        """Test that same input produces same output."""
        image = np.random.RandomState(42).randint(0, 255, (100, 200, 3), dtype=np.uint8)

        name1 = generator.generate_name_from_image(image.copy(), "button")
        name2 = generator.generate_name_from_image(image.copy(), "button")

        assert name1 == name2
