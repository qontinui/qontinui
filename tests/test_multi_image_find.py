"""
Tests for multi-image FIND actions with search strategies.

This module tests the new multi-image support for FIND actions:
- Schema validation for image_ids (list)
- Multiple pattern loading
- Strategy application (FIRST, BEST, ALL, EACH)
"""

import pytest
from pydantic import ValidationError

from qontinui.config.models.find_actions import FindStateImageActionConfig
from qontinui.config.models.targets import ImageTarget


class TestImageTargetSchema:
    """Test ImageTarget schema with multiple image IDs."""

    def test_single_image_id_list(self):
        """Test that single image works with list format."""
        target = ImageTarget(imageIds=["image-123"])
        assert target.image_ids == ["image-123"]
        assert target.type == "image"

    def test_multiple_image_ids(self):
        """Test that multiple images work."""
        target = ImageTarget(imageIds=["image-1", "image-2", "image-3"])
        assert len(target.image_ids) == 3
        assert "image-1" in target.image_ids
        assert "image-2" in target.image_ids
        assert "image-3" in target.image_ids

    def test_empty_image_ids_fails(self):
        """Test that empty image_ids list is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ImageTarget(imageIds=[])

        errors = exc_info.value.errors()
        assert any("min_length" in str(e).lower() for e in errors)

    def test_camel_case_alias(self):
        """Test that camelCase alias works (imageIds)."""
        # Frontend sends camelCase
        data = {"type": "image", "imageIds": ["img-1", "img-2"]}
        target = ImageTarget(**data)
        assert target.image_ids == ["img-1", "img-2"]


class TestFindStateImageActionConfigSchema:
    """Test FindStateImageActionConfig schema with multiple image IDs."""

    def test_single_image_id(self):
        """Test single image ID in list format."""
        config = FindStateImageActionConfig(stateId="state-1", imageIds=["stateimage-123"])
        assert config.image_ids == ["stateimage-123"]
        assert config.state_id == "state-1"

    def test_multiple_image_ids(self):
        """Test multiple image IDs."""
        config = FindStateImageActionConfig(
            stateId="state-1", imageIds=["stateimage-1", "stateimage-2", "stateimage-3"]
        )
        assert len(config.image_ids) == 3

    def test_camel_case_alias(self):
        """Test camelCase aliases work."""
        data = {"stateId": "state-1", "imageIds": ["img-1", "img-2"]}
        config = FindStateImageActionConfig(**data)
        assert config.state_id == "state-1"
        assert config.image_ids == ["img-1", "img-2"]

    def test_empty_image_ids_fails(self):
        """Test that empty image_ids fails validation."""
        with pytest.raises(ValidationError):
            FindStateImageActionConfig(stateId="state-1", imageIds=[])


class TestSearchStrategyIntegration:
    """Integration tests for search strategy behavior.

    Note: These are schema/config tests. Full execution tests would require
    actual image assets and screen capture capabilities.
    """

    def test_first_strategy_config(self):
        """Test FIRST strategy configuration."""
        target = ImageTarget(
            imageIds=["btn-ok", "btn-accept", "btn-continue"],
            searchOptions={"similarity": 0.8, "searchStrategy": "FIRST"},
        )
        assert target.search_options.search_strategy == "FIRST"
        assert len(target.image_ids) == 3

    def test_best_strategy_config(self):
        """Test BEST strategy configuration."""
        target = ImageTarget(
            imageIds=["logo-v1", "logo-v2", "logo-v3"],
            searchOptions={"similarity": 0.9, "searchStrategy": "BEST"},
        )
        assert target.search_options.search_strategy == "BEST"

    def test_all_strategy_config(self):
        """Test ALL strategy configuration."""
        target = ImageTarget(
            imageIds=["btn-1", "btn-2", "btn-3"], searchOptions={"searchStrategy": "ALL"}
        )
        assert target.search_options.search_strategy == "ALL"

    def test_each_strategy_config(self):
        """Test EACH strategy configuration."""
        target = ImageTarget(
            imageIds=["indicator-1", "indicator-2"], searchOptions={"searchStrategy": "EACH"}
        )
        assert target.search_options.search_strategy == "EACH"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
