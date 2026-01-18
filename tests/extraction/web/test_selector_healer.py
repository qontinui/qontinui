"""
Tests for selector_healer module.

Tests automatic selector repair when DOM changes.
"""

import pytest

from qontinui.extraction.web.selector_healer import (
    HealingAttempt,
    HealingResult,
    SelectorHealer,
)
from qontinui.extraction.web.models import BoundingBox, InteractiveElement


class TestHealingAttempt:
    """Tests for HealingAttempt dataclass."""

    def test_create_attempt(self) -> None:
        """Test creating a healing attempt record."""
        attempt = HealingAttempt(
            strategy="selector_variation",
            selector_tried="button.submit",
            success=True,
        )

        assert attempt.strategy == "selector_variation"
        assert attempt.success is True
        assert attempt.error is None

    def test_attempt_with_error(self) -> None:
        """Test attempt with error."""
        attempt = HealingAttempt(
            strategy="text_match",
            selector_tried="//*[text()='Submit']",
            success=False,
            error="Element not found",
        )

        assert attempt.success is False
        assert attempt.error == "Element not found"

    def test_to_dict(self) -> None:
        """Test serialization."""
        attempt = HealingAttempt(
            strategy="position_match",
            selector_tried="button:nth-child(2)",
            success=True,
        )

        data = attempt.to_dict()

        assert data["strategy"] == "position_match"
        assert data["success"] is True


class TestHealingResult:
    """Tests for HealingResult dataclass."""

    def test_successful_result(self) -> None:
        """Test successful healing result."""
        result = HealingResult(
            success=True,
            original_selector="button.old-class",
            healed_selector="button.new-class",
            element=None,  # Would be actual ElementHandle
            confidence=0.9,
            strategy_used="selector_variation",
        )

        assert result.success is True
        assert result.healed_selector == "button.new-class"
        assert result.confidence == 0.9

    def test_failed_result(self) -> None:
        """Test failed healing result."""
        result = HealingResult(
            success=False,
            original_selector="button.deleted",
            healed_selector=None,
            element=None,
            confidence=0.0,
            strategy_used="none",
        )

        assert result.success is False
        assert result.healed_selector is None

    def test_result_with_attempts(self) -> None:
        """Test result with recorded attempts."""
        attempts = [
            HealingAttempt(
                strategy="selector_variation",
                selector_tried="button.var1",
                success=False,
            ),
            HealingAttempt(
                strategy="text_match",
                selector_tried="//*[text()='Submit']",
                success=True,
            ),
        ]

        result = HealingResult(
            success=True,
            original_selector="button.old",
            healed_selector="//*[text()='Submit']",
            element=None,
            confidence=0.85,
            strategy_used="text_match",
            attempts=attempts,
        )

        assert len(result.attempts) == 2
        assert result.attempts[1].success is True

    def test_to_dict(self) -> None:
        """Test serialization."""
        result = HealingResult(
            success=True,
            original_selector="button.old",
            healed_selector="button.new",
            element=None,
            confidence=0.9,
            strategy_used="selector_variation",
            attempts=[
                HealingAttempt(
                    strategy="selector_variation",
                    selector_tried="button.new",
                    success=True,
                )
            ],
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["original_selector"] == "button.old"
        assert data["healed_selector"] == "button.new"
        assert len(data["attempts"]) == 1


class TestSelectorHealerVariations:
    """Tests for selector variation generation."""

    def test_generate_variations_with_nth_child(self) -> None:
        """Test removing nth-child from selector."""
        healer = SelectorHealer()

        variations = healer._generate_selector_variations(
            "div.container > button:nth-child(2)"
        )

        # Should include version without nth-child
        assert "div.container > button" in variations

    def test_generate_variations_with_child_combinator(self) -> None:
        """Test variations with child combinator."""
        healer = SelectorHealer()

        variations = healer._generate_selector_variations("nav > ul > li > a.link")

        # Should include parent only
        assert "nav > ul > li" in variations
        # Should include descendant version
        assert "nav > ul > li a.link" in variations

    def test_generate_variations_with_classes(self) -> None:
        """Test variations removing classes."""
        healer = SelectorHealer()

        variations = healer._generate_selector_variations("button.primary.large")

        # Should include version without last class
        assert "button.primary" in variations
        # Should include tag only
        assert "button" in variations

    def test_generate_variations_with_id(self) -> None:
        """Test variations extracting ID."""
        healer = SelectorHealer()

        variations = healer._generate_selector_variations("div.container #submit-btn")

        # Should include ID only
        assert "#submit-btn" in variations

    def test_generate_variations_unique(self) -> None:
        """Test that variations are unique."""
        healer = SelectorHealer()

        variations = healer._generate_selector_variations("button.btn.btn")

        # Should not have duplicates
        assert len(variations) == len(set(variations))


class TestSelectorHealerElementDescription:
    """Tests for element description building."""

    def test_build_description_button(self) -> None:
        """Test building description for button."""
        healer = SelectorHealer()
        elem = InteractiveElement(
            id="btn1",
            bbox=BoundingBox(x=0, y=0, width=100, height=50),
            tag_name="button",
            element_type="button",
            screenshot_id="test",
            selector="button",
            text="Submit",
        )

        desc = healer._build_element_description(elem)

        assert "button" in desc
        assert "Submit" in desc

    def test_build_description_link(self) -> None:
        """Test building description for link."""
        healer = SelectorHealer()
        elem = InteractiveElement(
            id="link1",
            bbox=BoundingBox(x=0, y=0, width=100, height=50),
            tag_name="a",
            element_type="link",
            screenshot_id="test",
            selector="a",
            text="Home",
            href="/home",
        )

        desc = healer._build_element_description(elem)

        assert "link" in desc
        assert "Home" in desc
        assert "/home" in desc

    def test_build_description_with_aria(self) -> None:
        """Test building description with aria-label."""
        healer = SelectorHealer()
        elem = InteractiveElement(
            id="btn1",
            bbox=BoundingBox(x=0, y=0, width=100, height=50),
            tag_name="button",
            element_type="button",
            screenshot_id="test",
            selector="button",
            text="X",
            aria_label="Close dialog",
        )

        desc = healer._build_element_description(elem)

        assert "Close dialog" in desc
