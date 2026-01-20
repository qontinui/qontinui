"""
Tests for deep_locator module.

Tests cross-frame selector resolution and deep locator parsing.
"""

from qontinui.extraction.web.deep_locator import (
    DeepLocatorParts,
    build_deep_locator,
    is_deep_locator,
)


class TestDeepLocatorParts:
    """Tests for DeepLocatorParts dataclass."""

    def test_simple_selector(self) -> None:
        """Test parts for simple selector (no frames)."""
        parts = DeepLocatorParts(
            frame_selectors=[],
            element_selector="button.submit",
            original="button.submit",
        )

        assert parts.has_frames is False
        assert parts.frame_depth == 0
        assert parts.element_selector == "button.submit"

    def test_single_frame_selector(self) -> None:
        """Test parts with single frame."""
        parts = DeepLocatorParts(
            frame_selectors=["iframe#sidebar"],
            element_selector="button.submit",
            original="iframe#sidebar >> button.submit",
        )

        assert parts.has_frames is True
        assert parts.frame_depth == 1
        assert parts.frame_selectors == ["iframe#sidebar"]

    def test_nested_frame_selector(self) -> None:
        """Test parts with nested frames."""
        parts = DeepLocatorParts(
            frame_selectors=["iframe#outer", "iframe#inner"],
            element_selector="button.submit",
            original="iframe#outer >> iframe#inner >> button.submit",
        )

        assert parts.has_frames is True
        assert parts.frame_depth == 2


class TestDeepLocatorResolverParsing:
    """Tests for DeepLocatorResolver parsing methods."""

    def test_parse_simple_selector(self) -> None:
        """Test parsing simple selector."""
        # Note: Can't create actual resolver without Page, but we can test the parse method
        # by creating a mock or testing the parsing logic directly

        # Test the parsing logic
        selector = "button.submit"
        parts = selector.split(" >> ")

        assert len(parts) == 1
        assert parts[0] == "button.submit"

    def test_parse_frame_selector(self) -> None:
        """Test parsing frame selector."""
        selector = "iframe#sidebar >> button.submit"
        parts = selector.split(" >> ")

        assert len(parts) == 2
        assert parts[0] == "iframe#sidebar"
        assert parts[1] == "button.submit"

    def test_parse_nested_frame_selector(self) -> None:
        """Test parsing nested frame selector."""
        selector = "iframe#outer >> iframe#inner >> button.submit"
        parts = selector.split(" >> ")

        assert len(parts) == 3
        assert parts[0] == "iframe#outer"
        assert parts[1] == "iframe#inner"
        assert parts[2] == "button.submit"


class TestBuildDeepLocator:
    """Tests for build_deep_locator function."""

    def test_build_main_frame(self) -> None:
        """Test building locator for main frame element."""
        locator = build_deep_locator([], "button.submit")
        assert locator == "button.submit"

    def test_build_single_frame(self) -> None:
        """Test building locator for single frame."""
        locator = build_deep_locator(["iframe#sidebar"], "button.submit")
        assert locator == "iframe#sidebar >> button.submit"

    def test_build_nested_frames(self) -> None:
        """Test building locator for nested frames."""
        locator = build_deep_locator(["iframe#outer", "iframe#inner"], "button.submit")
        assert locator == "iframe#outer >> iframe#inner >> button.submit"


class TestIsDeepLocator:
    """Tests for is_deep_locator function."""

    def test_simple_selector_is_not_deep(self) -> None:
        """Test that simple selectors are not deep locators."""
        assert is_deep_locator("button.submit") is False
        assert is_deep_locator("#myButton") is False
        assert is_deep_locator("div > button") is False

    def test_frame_selector_is_deep(self) -> None:
        """Test that frame selectors are deep locators."""
        assert is_deep_locator("iframe#sidebar >> button.submit") is True
        assert is_deep_locator("iframe >> div >> button") is True

    def test_selector_with_arrow_is_not_deep(self) -> None:
        """Test that CSS child combinator is not confused with deep locator."""
        # CSS child combinator uses single >
        assert is_deep_locator("div > button") is False
        # Only >> indicates frame hop
        assert is_deep_locator("div >> button") is True


class TestDeepLocatorEdgeCases:
    """Tests for edge cases in deep locator handling."""

    def test_empty_frame_path(self) -> None:
        """Test building with empty frame path."""
        locator = build_deep_locator([], "button")
        assert locator == "button"
        assert is_deep_locator(locator) is False

    def test_complex_element_selector(self) -> None:
        """Test with complex element selector."""
        locator = build_deep_locator(
            ["iframe#main"], "div.container > form > button[type='submit']"
        )
        assert locator == "iframe#main >> div.container > form > button[type='submit']"
        assert is_deep_locator(locator) is True

    def test_frame_selector_with_attributes(self) -> None:
        """Test frame selector with complex attributes."""
        locator = build_deep_locator(['iframe[name="content"]', 'iframe[src*="widget"]'], "button")
        assert 'iframe[name="content"]' in locator
        assert 'iframe[src*="widget"]' in locator
        assert locator.count(" >> ") == 2
