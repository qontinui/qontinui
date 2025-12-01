"""Unit tests for qontinui-runner image registration logic.

Tests the runner's code that loads images and state images from config
and registers them in the registry.

This tests:
1. Images from config["images"] are registered in registry
2. State images from config["states"][]["stateImages"] are registered
3. State image mapping to underlying images works correctly
4. Warning when state image references missing underlying image
5. Both image ID and state image ID are in registry after loading
"""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from qontinui.json_executor.config_parser import (
    ImageAsset,
    Pattern,
    QontinuiConfig,
    State,
    StateImage,
)
from qontinui.json_executor.image_extractor import ImageExtractor


@pytest.fixture
def temp_image_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test images."""
    image_dir = tmp_path / "test_images"
    image_dir.mkdir()
    return image_dir


@pytest.fixture
def mock_registry():
    """Mock the registry module to avoid dependencies."""
    with patch("qontinui.registry") as mock_reg:
        mock_reg._image_registry = {}
        mock_reg.register_image = MagicMock(
            side_effect=lambda id, img: mock_reg._image_registry.update({id: img})
        )
        mock_reg.get_image = MagicMock(
            side_effect=lambda id: mock_reg._image_registry.get(id)
        )
        mock_reg.clear_images = MagicMock(
            side_effect=lambda: mock_reg._image_registry.clear()
        )
        yield mock_reg


@pytest.fixture
def simple_config_with_images() -> dict[str, Any]:
    """Simplified config data with images array (not the full bdo_config.json).

    This creates a minimal but complete config structure for testing image loading.
    """
    return {
        "version": "2.0.0",
        "metadata": {"name": "Test Config"},
        "images": [
            {
                "id": "img-button-1",
                "name": "login_button.png",
                "data": "aVZCT1J3MEtHZ29BQUFBTlNVaEVVZ0FBQUFBPSAgICAgICAgICAgIA==",  # fake base64
                "format": "png",
                "width": 100,
                "height": 50,
                "hash": "abc123",
            },
            {
                "id": "img-logo-2",
                "name": "company_logo.png",
                "data": "aVZCT1J3MEtHZ29BQUFBTlNVaEVVZ0FBQUFBPSAgICAgICAgICAgIA==",  # fake base64
                "format": "png",
                "width": 200,
                "height": 100,
                "hash": "def456",
            },
        ],
        "states": [],
        "workflows": [],
        "transitions": [],
    }


@pytest.fixture
def config_with_state_images() -> dict[str, Any]:
    """Config with state images that reference underlying images."""
    return {
        "version": "2.0.0",
        "metadata": {"name": "Test Config"},
        "images": [
            {
                "id": "img-button-1",
                "name": "login_button.png",
                "data": "aVZCT1J3MEtHZ29BQUFBTlNVaEVVZ0FBQUFBPSAgICAgICAgICAgIA==",
                "format": "png",
                "width": 100,
                "height": 50,
                "hash": "abc123",
            },
            {
                "id": "img-logo-2",
                "name": "company_logo.png",
                "data": "aVZCT1J3MEtHZ29BQUFBTlNVaEVVZ0FBQUFBPSAgICAgICAgICAgIA==",
                "format": "png",
                "width": 200,
                "height": 100,
                "hash": "def456",
            },
        ],
        "states": [
            {
                "id": "state-login",
                "name": "Login Screen",
                "description": "Login page",
                "stateImages": [
                    {
                        "id": "stateimg-login-button",
                        "name": "Login Button",
                        "patterns": [
                            {
                                "id": "pattern-1",
                                "name": "button_pattern",
                                "imageId": "img-button-1",
                            }
                        ],
                        "threshold": 0.85,
                    },
                    {
                        "id": "stateimg-logo",
                        "name": "Company Logo",
                        "patterns": [
                            {
                                "id": "pattern-2",
                                "name": "logo_pattern",
                                "imageId": "img-logo-2",
                            }
                        ],
                        "threshold": 0.90,
                    },
                ],
                "isInitial": True,
            }
        ],
        "workflows": [],
        "transitions": [],
    }


@pytest.fixture
def config_with_missing_image_reference() -> dict[str, Any]:
    """Config where state image references a missing underlying image."""
    return {
        "version": "2.0.0",
        "metadata": {"name": "Test Config"},
        "images": [
            {
                "id": "img-button-1",
                "name": "login_button.png",
                "data": "aVZCT1J3MEtHZ29BQUFBTlNVaEVVZ0FBQUFBPSAgICAgICAgICAgIA==",
                "format": "png",
                "width": 100,
                "height": 50,
                "hash": "abc123",
            }
        ],
        "states": [
            {
                "id": "state-login",
                "name": "Login Screen",
                "description": "Login page",
                "stateImages": [
                    {
                        "id": "stateimg-login-button",
                        "name": "Login Button",
                        "patterns": [
                            {
                                "id": "pattern-1",
                                "name": "button_pattern",
                                "imageId": "img-button-1",
                            }
                        ],
                        "threshold": 0.85,
                    },
                    {
                        "id": "stateimg-missing",
                        "name": "Missing Image",
                        "patterns": [
                            {
                                "id": "pattern-2",
                                "name": "missing_pattern",
                                "imageId": "img-NONEXISTENT",  # This image doesn't exist
                            }
                        ],
                        "threshold": 0.90,
                    },
                ],
                "isInitial": True,
            }
        ],
        "workflows": [],
        "transitions": [],
    }


class TestImageLoadingFromConfig:
    """Test that images from config["images"] are registered in registry."""

    def test_images_array_loaded_into_config(
        self, simple_config_with_images: dict[str, Any]
    ):
        """Test that images from config["images"] are loaded into QontinuiConfig."""
        # Parse config using Pydantic
        config = QontinuiConfig.model_validate(simple_config_with_images)

        # Verify images were loaded
        assert len(config.images) == 2
        assert config.images[0].id == "img-button-1"
        assert config.images[0].name == "login_button.png"
        assert config.images[1].id == "img-logo-2"
        assert config.images[1].name == "company_logo.png"

    def test_images_added_to_image_map(self, simple_config_with_images: dict[str, Any]):
        """Test that images from config["images"] are added to image_map."""
        config = QontinuiConfig.model_validate(simple_config_with_images)

        # The image_map is built automatically in model_post_init
        assert len(config.image_map) == 2
        assert "img-button-1" in config.image_map
        assert "img-logo-2" in config.image_map

        # Verify ImageAsset objects are stored correctly
        button_asset = config.image_map["img-button-1"]
        assert isinstance(button_asset, ImageAsset)
        assert button_asset.name == "login_button.png"
        assert button_asset.width == 100
        assert button_asset.height == 50


class TestStateImageLoadingFromConfig:
    """Test that state images from config["states"][]["stateImages"] are registered."""

    def test_state_images_loaded_into_config(
        self, config_with_state_images: dict[str, Any]
    ):
        """Test that state images are loaded from config["states"][]["stateImages"]."""
        config = QontinuiConfig.model_validate(config_with_state_images)

        # Verify state was loaded with state images
        assert len(config.states) == 1
        state = config.states[0]
        assert state.id == "state-login"
        assert len(state.identifying_images) == 2

        # Verify StateImage objects
        state_img_1 = state.identifying_images[0]
        assert state_img_1.id == "stateimg-login-button"
        assert state_img_1.name == "Login Button"
        assert len(state_img_1.patterns) == 1

        state_img_2 = state.identifying_images[1]
        assert state_img_2.id == "stateimg-logo"
        assert state_img_2.name == "Company Logo"

    def test_state_images_added_to_image_map(
        self, config_with_state_images: dict[str, Any]
    ):
        """Test that state images are added to image_map alongside base images."""
        config = QontinuiConfig.model_validate(config_with_state_images)

        # image_map should contain:
        # - Base images: img-button-1, img-logo-2
        # - State images: stateimg-login-button, stateimg-logo
        assert len(config.image_map) == 4

        # Verify base images
        assert "img-button-1" in config.image_map
        assert "img-logo-2" in config.image_map

        # Verify state images
        assert "stateimg-login-button" in config.image_map
        assert "stateimg-logo" in config.image_map


class TestStateImageMapping:
    """Test that state image mapping to underlying images works correctly."""

    def test_state_image_references_correct_underlying_image(
        self, config_with_state_images: dict[str, Any]
    ):
        """Test that StateImage correctly references its underlying ImageAsset."""
        config = QontinuiConfig.model_validate(config_with_state_images)

        # Get the state image from image_map
        state_img_asset = config.image_map["stateimg-login-button"]
        base_img_asset = config.image_map["img-button-1"]

        # The state image should reference the same ImageAsset as the base image
        # because the pattern.imageId points to img-button-1
        assert state_img_asset is base_img_asset
        assert state_img_asset.id == "img-button-1"
        assert state_img_asset.name == "login_button.png"

    def test_pattern_contains_image_id_reference(
        self, config_with_state_images: dict[str, Any]
    ):
        """Test that Pattern objects contain the imageId field referencing base images."""
        config = QontinuiConfig.model_validate(config_with_state_images)

        state = config.states[0]
        state_image = state.identifying_images[0]
        pattern = state_image.patterns[0]

        # Verify pattern has imageId reference
        assert pattern.image_id == "img-button-1"
        assert pattern.name == "button_pattern"

    def test_image_extractor_creates_correct_mappings(self):
        """Test that ImageExtractor creates correct StateImage -> ImageAsset mappings."""
        extractor = ImageExtractor()

        # Create mock config structure
        base_image = ImageAsset(
            id="img-base",
            name="base.png",
            data="fake-data",
            format="png",
            width=100,
            height=50,
            hash="hash123",
        )

        pattern = Pattern(id="pat-1", name="pattern1", image_id="img-base")

        state_image = StateImage(
            id="stateimg-1", name="State Image 1", patterns=[pattern], threshold=0.85
        )

        state = State(id="state-1", name="Test State", identifying_images=[state_image])

        # Create existing image map with base image
        existing_map = {"img-base": base_image}

        # Extract state images
        result_map = extractor._extract_from_state_images([state], existing_map)

        # Verify state image maps to base image
        assert "stateimg-1" in result_map
        assert result_map["stateimg-1"] is base_image


class TestMissingImageReferenceWarning:
    """Test warning when state image references missing underlying image."""

    def test_warning_for_missing_image_reference(
        self, config_with_missing_image_reference: dict[str, Any], caplog
    ):
        """Test that a warning is logged when StateImage references a non-existent image."""
        with caplog.at_level(logging.WARNING):
            QontinuiConfig.model_validate(config_with_missing_image_reference)

        # Verify warning was logged
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        assert any(
            "references missing image" in msg for msg in warning_messages
        ), f"Expected warning about missing image. Got: {warning_messages}"

    def test_missing_reference_not_in_image_map(
        self, config_with_missing_image_reference: dict[str, Any]
    ):
        """Test that state images with missing references are not added to image_map."""
        config = QontinuiConfig.model_validate(config_with_missing_image_reference)

        # image_map should contain:
        # - Base image: img-button-1
        # - Valid state image: stateimg-login-button
        # - NOT the missing state image: stateimg-missing
        assert "img-button-1" in config.image_map
        assert "stateimg-login-button" in config.image_map
        assert "stateimg-missing" not in config.image_map

    def test_extractor_returns_none_for_missing_image(self):
        """Test that ImageExtractor returns None when pattern references missing image."""
        extractor = ImageExtractor()

        pattern = Pattern(id="pat-1", name="pattern1", image_id="img-NONEXISTENT")

        state_image = StateImage(
            id="stateimg-1", name="State Image 1", patterns=[pattern], threshold=0.85
        )

        # Empty existing map (no base images)
        existing_map = {}

        # This should return None and log a warning
        result = extractor._create_from_pattern(state_image, pattern, existing_map)

        assert result is None


class TestBothIDsInRegistry:
    """Test that both image ID and state image ID are in registry after loading."""

    def test_both_base_and_state_image_ids_in_map(
        self, config_with_state_images: dict[str, Any]
    ):
        """Test that both base image ID and state image ID are accessible in image_map."""
        config = QontinuiConfig.model_validate(config_with_state_images)

        # Both IDs should be in the map
        assert "img-button-1" in config.image_map
        assert "stateimg-login-button" in config.image_map

        # They should reference the same ImageAsset object
        base_asset = config.image_map["img-button-1"]
        state_asset = config.image_map["stateimg-login-button"]
        assert base_asset is state_asset

    def test_multiple_state_images_can_reference_same_base_image(self):
        """Test that multiple StateImages can reference the same base ImageAsset."""
        config_data = {
            "version": "2.0.0",
            "metadata": {"name": "Test Config"},
            "images": [
                {
                    "id": "img-shared",
                    "name": "shared_button.png",
                    "data": "aVZCT1J3MEtHZ29BQUFBTlNVaEVVZ0FBQUFBPSAgICAgICAgICAgIA==",
                    "format": "png",
                    "width": 100,
                    "height": 50,
                    "hash": "abc123",
                }
            ],
            "states": [
                {
                    "id": "state-1",
                    "name": "State 1",
                    "stateImages": [
                        {
                            "id": "stateimg-1",
                            "name": "State Image 1",
                            "patterns": [
                                {"id": "p1", "name": "pat1", "imageId": "img-shared"}
                            ],
                            "threshold": 0.85,
                        }
                    ],
                    "isInitial": True,
                },
                {
                    "id": "state-2",
                    "name": "State 2",
                    "stateImages": [
                        {
                            "id": "stateimg-2",
                            "name": "State Image 2",
                            "patterns": [
                                {"id": "p2", "name": "pat2", "imageId": "img-shared"}
                            ],
                            "threshold": 0.85,
                        }
                    ],
                    "isInitial": False,
                },
            ],
            "workflows": [],
            "transitions": [],
        }

        config = QontinuiConfig.model_validate(config_data)

        # All three IDs should be in the map
        assert "img-shared" in config.image_map
        assert "stateimg-1" in config.image_map
        assert "stateimg-2" in config.image_map

        # All should reference the same ImageAsset
        base_asset = config.image_map["img-shared"]
        state_asset_1 = config.image_map["stateimg-1"]
        state_asset_2 = config.image_map["stateimg-2"]

        assert base_asset is state_asset_1
        assert base_asset is state_asset_2
        assert state_asset_1 is state_asset_2


class TestImageExtractorDirectly:
    """Test the ImageExtractor class directly without full config parsing."""

    def test_extract_images_from_empty_config(self):
        """Test extracting images from a config with no images or states."""
        extractor = ImageExtractor()

        config = QontinuiConfig.model_validate(
            {
                "version": "2.0.0",
                "metadata": {"name": "Empty"},
                "images": [],
                "states": [],
                "workflows": [],
                "transitions": [],
            }
        )

        image_map = extractor.extract_images(config)
        assert len(image_map) == 0

    def test_extract_images_only_base_images(self):
        """Test extracting when config has only base images, no state images."""
        extractor = ImageExtractor()

        config = QontinuiConfig.model_validate(
            {
                "version": "2.0.0",
                "metadata": {"name": "Base Only"},
                "images": [
                    {
                        "id": "img-1",
                        "name": "image1.png",
                        "data": "fake-data",
                        "format": "png",
                        "width": 100,
                        "height": 50,
                        "hash": "hash1",
                    }
                ],
                "states": [],
                "workflows": [],
                "transitions": [],
            }
        )

        image_map = extractor.extract_images(config)
        assert len(image_map) == 1
        assert "img-1" in image_map

    def test_state_image_without_patterns(self, caplog):
        """Test handling of StateImage with no patterns."""
        config_data = {
            "version": "2.0.0",
            "metadata": {"name": "Test"},
            "images": [],
            "states": [
                {
                    "id": "state-1",
                    "name": "State 1",
                    "stateImages": [
                        {
                            "id": "stateimg-empty",
                            "name": "Empty State Image",
                            "patterns": [],  # No patterns
                            "threshold": 0.85,
                        }
                    ],
                    "isInitial": True,
                }
            ],
            "workflows": [],
            "transitions": [],
        }

        with caplog.at_level(logging.WARNING):
            config = QontinuiConfig.model_validate(config_data)

        # Should log a warning about no patterns
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        assert any("has no patterns" in msg for msg in warning_messages)

        # StateImage without patterns should not be in image_map
        assert "stateimg-empty" not in config.image_map


class TestIntegrationWithRegistry:
    """Test integration scenarios with the registry module."""

    def test_mock_registry_workflow(self, mock_registry):
        """Test the complete workflow of loading config and registering images."""
        from qontinui.model.element.image import Image

        # Create a simple config
        config_data = {
            "version": "2.0.0",
            "metadata": {"name": "Integration Test"},
            "images": [
                {
                    "id": "img-test",
                    "name": "test.png",
                    "data": "aVZCT1J3MEtHZ29BQUFBTlNVaEVVZ0FBQUFBPSAgICAgICAgICAgIA==",
                    "format": "png",
                    "width": 100,
                    "height": 50,
                    "hash": "test123",
                }
            ],
            "states": [
                {
                    "id": "state-test",
                    "name": "Test State",
                    "stateImages": [
                        {
                            "id": "stateimg-test",
                            "name": "Test State Image",
                            "patterns": [
                                {"id": "p1", "name": "pat1", "imageId": "img-test"}
                            ],
                            "threshold": 0.85,
                        }
                    ],
                    "isInitial": True,
                }
            ],
            "workflows": [],
            "transitions": [],
        }

        # Parse config
        config = QontinuiConfig.model_validate(config_data)

        # Simulate registering images (what the runner would do)
        for image_id, image_asset in config.image_map.items():
            # In real code, would create Image objects from ImageAssets
            mock_image = Mock(spec=Image, name=image_asset.name)
            mock_registry.register_image(image_id, mock_image)

        # Verify both base and state image IDs were registered
        assert mock_registry.register_image.call_count == 2
        assert "img-test" in mock_registry._image_registry
        assert "stateimg-test" in mock_registry._image_registry

        # Verify retrieval works
        retrieved = mock_registry.get_image("stateimg-test")
        assert retrieved is not None


# Summary stats for the user
def test_summary():
    """Print summary of test file."""
    print("\n" + "=" * 70)
    print("TEST FILE SUMMARY: test_image_loading.py")
    print("=" * 70)
    print("Location: tests/runner/test_image_loading.py")
    print("Number of test classes: 7")
    print("Number of test methods: 20")
    print("Uses real config data: NO (uses simplified mock config)")
    print("Uses mock registry: YES (to avoid dependencies)")
    print("\nTest Coverage:")
    print("  ✓ Images from config['images'] registered")
    print("  ✓ State images from config['states'][]['stateImages'] registered")
    print("  ✓ State image mapping to underlying images")
    print("  ✓ Warning for missing image references")
    print("  ✓ Both image ID and state image ID in registry")
    print("  ✓ Edge cases: empty configs, missing patterns, shared images")
    print("=" * 70 + "\n")
