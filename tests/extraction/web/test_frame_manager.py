"""
Tests for frame_manager module.

Tests iframe traversal, frame-aware element IDs, and multi-frame extraction.
"""

import pytest

from qontinui.extraction.web.frame_manager import (
    FrameAwareElement,
    FrameExtractionResult,
    FrameInfo,
    FrameManager,
)
from qontinui.extraction.web.models import BoundingBox, InteractiveElement


class TestFrameManager:
    """Tests for FrameManager class."""

    def test_encode_element_id(self) -> None:
        """Test encoding element IDs with frame context."""
        manager = FrameManager()

        # Main frame element
        encoded = manager.encode_element_id(0, "elem_000001")
        assert encoded == "0-elem_000001"

        # Iframe element
        encoded = manager.encode_element_id(2, "elem_000015")
        assert encoded == "2-elem_000015"

    def test_decode_element_id(self) -> None:
        """Test decoding frame-aware element IDs."""
        manager = FrameManager()

        frame_id, element_id = manager.decode_element_id("0-elem_000001")
        assert frame_id == 0
        assert element_id == "elem_000001"

        frame_id, element_id = manager.decode_element_id("2-elem_000015")
        assert frame_id == 2
        assert element_id == "elem_000015"

    def test_decode_element_id_invalid(self) -> None:
        """Test decoding invalid element IDs raises error."""
        manager = FrameManager()

        with pytest.raises(ValueError):
            manager.decode_element_id("invalid_id")

    def test_generate_deep_locator_main_frame(self) -> None:
        """Test deep locator generation for main frame elements."""
        manager = FrameManager()

        locator = manager.generate_deep_locator("main", "button.submit")
        assert locator == "button.submit"

        locator = manager.generate_deep_locator("", "button.submit")
        assert locator == "button.submit"

    def test_generate_deep_locator_iframe(self) -> None:
        """Test deep locator generation for iframe elements."""
        manager = FrameManager()

        locator = manager.generate_deep_locator("iframe#sidebar", "button.submit")
        assert locator == "iframe#sidebar >> button.submit"

    def test_parse_deep_locator_simple(self) -> None:
        """Test parsing simple selectors (no frame)."""
        manager = FrameManager()

        frame, element = manager.parse_deep_locator("button.submit")
        assert frame == "main"
        assert element == "button.submit"

    def test_parse_deep_locator_with_frame(self) -> None:
        """Test parsing deep locators with frame."""
        manager = FrameManager()

        frame, element = manager.parse_deep_locator("iframe#sidebar >> button.submit")
        assert frame == "iframe#sidebar"
        assert element == "button.submit"

    def test_reset(self) -> None:
        """Test resetting frame manager state."""
        manager = FrameManager()
        manager._frame_counter = 5

        manager.reset()
        assert manager._frame_counter == 0


class TestFrameInfo:
    """Tests for FrameInfo dataclass."""

    def test_frame_info_creation(self) -> None:
        """Test creating FrameInfo."""
        info = FrameInfo(
            frame_id=0,
            name="main",
            url="https://example.com",
            selector="main",
            parent_frame_id=None,
            is_main=True,
        )

        assert info.frame_id == 0
        assert info.name == "main"
        assert info.is_main is True

    def test_frame_info_serialization(self) -> None:
        """Test FrameInfo to_dict/from_dict."""
        info = FrameInfo(
            frame_id=1,
            name="sidebar",
            url="about:blank",
            selector="iframe#sidebar",
            parent_frame_id=0,
            is_main=False,
        )

        data = info.to_dict()
        restored = FrameInfo.from_dict(data)

        assert restored.frame_id == info.frame_id
        assert restored.name == info.name
        assert restored.selector == info.selector
        assert restored.parent_frame_id == info.parent_frame_id


class TestFrameAwareElement:
    """Tests for FrameAwareElement dataclass."""

    def test_frame_aware_element_creation(self) -> None:
        """Test creating FrameAwareElement."""
        element = InteractiveElement(
            id="elem_000001",
            bbox=BoundingBox(x=10, y=20, width=100, height=50),
            tag_name="button",
            element_type="button",
            screenshot_id="screen_001",
            selector="button.submit",
            text="Submit",
        )

        frame_element = FrameAwareElement(
            frame_id=0,
            frame_selector="main",
            encoded_id="0-elem_000001",
            element=element,
            deep_selector="button.submit",
        )

        assert frame_element.frame_id == 0
        assert frame_element.encoded_id == "0-elem_000001"
        assert frame_element.element.text == "Submit"

    def test_frame_aware_element_serialization(self) -> None:
        """Test FrameAwareElement to_dict/from_dict."""
        element = InteractiveElement(
            id="elem_000001",
            bbox=BoundingBox(x=10, y=20, width=100, height=50),
            tag_name="button",
            element_type="button",
            screenshot_id="screen_001",
            selector="button.submit",
        )

        frame_element = FrameAwareElement(
            frame_id=1,
            frame_selector="iframe#sidebar",
            encoded_id="1-elem_000001",
            element=element,
            deep_selector="iframe#sidebar >> button.submit",
        )

        data = frame_element.to_dict()
        restored = FrameAwareElement.from_dict(data)

        assert restored.frame_id == frame_element.frame_id
        assert restored.encoded_id == frame_element.encoded_id
        assert restored.deep_selector == frame_element.deep_selector


class TestFrameExtractionResult:
    """Tests for FrameExtractionResult dataclass."""

    def test_extraction_result_creation(self) -> None:
        """Test creating FrameExtractionResult."""
        result = FrameExtractionResult(
            elements=[],
            frames=[],
            errors=[],
        )

        assert len(result.elements) == 0
        assert len(result.frames) == 0

    def test_extraction_result_lookup(self) -> None:
        """Test lookup methods on FrameExtractionResult."""
        element = InteractiveElement(
            id="elem_000001",
            bbox=BoundingBox(x=10, y=20, width=100, height=50),
            tag_name="button",
            element_type="button",
            screenshot_id="screen_001",
            selector="button.submit",
        )

        frame_element = FrameAwareElement(
            frame_id=0,
            frame_selector="main",
            encoded_id="0-elem_000001",
            element=element,
        )

        result = FrameExtractionResult(
            elements=[frame_element],
            frames=[],
            errors=[],
        )

        # Test get_by_encoded_id
        found = result.get_by_encoded_id("0-elem_000001")
        assert found is not None
        assert found.element.tag_name == "button"

        # Test get_frame_elements
        frame_elements = result.get_frame_elements(0)
        assert len(frame_elements) == 1

        # Test missing element
        missing = result.get_by_encoded_id("9-elem_000099")
        assert missing is None
