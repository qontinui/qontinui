"""Tests for VisualValidator."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from qontinui.validation.validation_types import ChangeType, ExpectedChange
from qontinui.validation.visual_validator import VisualValidator


class TestVisualValidator:
    """Tests for VisualValidator class."""

    def test_validate_any_change_no_change(self):
        """Test validation when ANY_CHANGE expected but nothing changed."""
        validator = VisualValidator()

        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = np.zeros((100, 100, 3), dtype=np.uint8)

        expected = ExpectedChange(type=ChangeType.ANY_CHANGE)
        result = validator.validate(before, after, expected)

        # No change when ANY_CHANGE expected - should fail
        assert not result.success

    def test_validate_any_change_with_change(self):
        """Test validation when ANY_CHANGE expected and change happened."""
        validator = VisualValidator()

        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = np.zeros((100, 100, 3), dtype=np.uint8)
        after[40:60, 40:60] = 255  # Add white square

        expected = ExpectedChange(type=ChangeType.ANY_CHANGE)
        result = validator.validate(before, after, expected)

        # Change detected - should pass
        assert result.success

    def test_validate_no_change_success(self):
        """Test validation when NO_CHANGE expected and nothing changed."""
        validator = VisualValidator()

        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = np.zeros((100, 100, 3), dtype=np.uint8)

        expected = ExpectedChange(type=ChangeType.NO_CHANGE)
        result = validator.validate(before, after, expected)

        # No change as expected - should pass
        assert result.success

    def test_validate_no_change_failure(self):
        """Test validation when NO_CHANGE expected but change happened."""
        validator = VisualValidator()

        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = np.zeros((100, 100, 3), dtype=np.uint8)
        after[40:60, 40:60] = 255  # Unexpected change

        expected = ExpectedChange(type=ChangeType.NO_CHANGE)
        result = validator.validate(before, after, expected)

        # Change when none expected - should fail
        assert not result.success

    def test_validate_region_changes(self):
        """Test validation with REGION_CHANGES type."""
        validator = VisualValidator()

        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = np.zeros((100, 100, 3), dtype=np.uint8)
        # Change in specified region
        after[20:40, 20:40] = 255

        expected = ExpectedChange(
            type=ChangeType.REGION_CHANGES,
            region=(10, 10, 50, 50),  # Region covering the change
            min_change_threshold=1.0,
        )
        result = validator.validate(before, after, expected)

        # Change in specified region - should pass
        assert result.success

    def test_validate_region_no_change_in_region(self):
        """Test validation when region doesn't change."""
        validator = VisualValidator()

        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = np.zeros((100, 100, 3), dtype=np.uint8)
        # Change outside specified region
        after[80:100, 80:100] = 255

        expected = ExpectedChange(
            type=ChangeType.REGION_CHANGES,
            region=(10, 10, 30, 30),  # Region NOT covering the change
            min_change_threshold=1.0,
        )
        result = validator.validate(before, after, expected)

        # No change in specified region - should fail
        assert not result.success

    def test_validate_default_any_change(self):
        """Test that default expectation is ANY_CHANGE."""
        validator = VisualValidator()

        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = np.zeros((100, 100, 3), dtype=np.uint8)
        after[40:60, 40:60] = 255

        # No expected specified - defaults to ANY_CHANGE
        result = validator.validate(before, after, expected=None)

        assert result.success

    def test_validation_records_time(self):
        """Test that validation records time spent."""
        validator = VisualValidator()

        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = np.zeros((100, 100, 3), dtype=np.uint8)

        result = validator.validate(before, after)

        assert result.validation_time_ms >= 0

    def test_validation_includes_diff(self):
        """Test that validation result includes visual diff."""
        validator = VisualValidator()

        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = np.full((100, 100, 3), 255, dtype=np.uint8)

        result = validator.validate(before, after)

        assert result.diff is not None
        assert result.diff.change_percentage > 0


class TestExpectedChange:
    """Tests for ExpectedChange class."""

    def test_expected_change_creation(self):
        """Test creating expected change."""
        change = ExpectedChange(
            type=ChangeType.ELEMENT_APPEARS,
            description="Button appears",
            region=(10, 20, 100, 50),
        )

        assert change.type == ChangeType.ELEMENT_APPEARS
        assert change.description == "Button appears"
        assert change.region == (10, 20, 100, 50)

    def test_expected_change_defaults(self):
        """Test expected change default values."""
        change = ExpectedChange(type=ChangeType.ANY_CHANGE)

        assert change.description == ""
        assert change.region is None
        assert change.pattern is None


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_change_types_exist(self):
        """Test that all change types exist."""
        assert ChangeType.ANY_CHANGE
        assert ChangeType.NO_CHANGE
        assert ChangeType.ELEMENT_APPEARS
        assert ChangeType.ELEMENT_DISAPPEARS
        assert ChangeType.REGION_CHANGES
