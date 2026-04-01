"""Tests for OmniParser integration in combined discovery flow."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from qontinui.discovery.discovery_facade import (
    DiscoveryAlgorithm,
    DiscoveryConfig,
    StateDiscoveryFacade,
)
from qontinui.discovery.models import AnalysisResult, StateImage


def _make_screenshots(n: int = 3) -> list[np.ndarray]:
    """Create synthetic screenshots for testing."""
    return [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(n)]


def _make_omniparser_analysis_result() -> MagicMock:
    """Create a mock AnalysisResult from OmniParserDetector.analyze()."""
    from qontinui.discovery.element_detection.analysis_base import BoundingBox, DetectedElement

    result = MagicMock()
    result.elements = [
        DetectedElement(
            bounding_box=BoundingBox(x=100, y=50, width=120, height=40),
            confidence=0.92,
            label="Submit Button",
            element_type="button",
            screenshot_index=0,
            metadata={"source": "yolo"},
        ),
        DetectedElement(
            bounding_box=BoundingBox(x=300, y=200, width=200, height=30),
            confidence=0.88,
            label="Username Input",
            element_type="text_field",
            screenshot_index=0,
            metadata={"source": "yolo"},
        ),
    ]
    result.confidence = 0.90
    result.metadata = {"device": "cpu", "total_elements": 2}
    return result


def _empty_analysis_result(**overrides: Any) -> AnalysisResult:
    defaults = {"states": [], "state_images": [], "transitions": [], "statistics": {}}
    defaults.update(overrides)
    return AnalysisResult(**defaults)


class TestCombinedWithOmniParser:
    """Tests for _discover_combined with enable_omniparser."""

    @patch("qontinui.discovery.discovery_facade.StateDiscoveryFacade._discover_with_omniparser")
    @patch("qontinui.discovery.discovery_facade.StateDiscoveryFacade._discover_with_differential")
    @patch(
        "qontinui.discovery.discovery_facade.StateDiscoveryFacade._discover_with_pixel_stability"
    )
    def test_combined_includes_omniparser_when_enabled(self, mock_pixel, mock_diff, mock_omni):
        """When enable_omniparser=True, combined discovery should call OmniParser."""
        mock_pixel.return_value = _empty_analysis_result()
        mock_diff.return_value = _empty_analysis_result()
        mock_omni.return_value = AnalysisResult(
            states=[],
            state_images=[
                StateImage(
                    id="omni_img_0",
                    name="Submit Button",
                    x=100,
                    y=50,
                    x2=220,
                    y2=90,
                    pixel_hash="omni_0",
                    frequency=0.92,
                    tags=["omniparser", "button"],
                ),
            ],
            transitions=[],
            statistics={"algorithm": "omniparser"},
        )

        facade = StateDiscoveryFacade()
        config = DiscoveryConfig(
            algorithm=DiscoveryAlgorithm.COMBINED,
            enable_omniparser=True,
            min_screenshots=2,
        )
        # Call _discover_combined directly to avoid discover_states validation
        screenshots = _make_screenshots(3)
        result = facade._discover_combined(screenshots, config, None)

        mock_omni.assert_called_once()
        # OmniParser results should be in merged output
        assert any(si.name == "Submit Button" for si in result.state_images)

    @patch("qontinui.discovery.discovery_facade.StateDiscoveryFacade._discover_with_omniparser")
    @patch("qontinui.discovery.discovery_facade.StateDiscoveryFacade._discover_with_differential")
    @patch(
        "qontinui.discovery.discovery_facade.StateDiscoveryFacade._discover_with_pixel_stability"
    )
    def test_combined_skips_omniparser_when_disabled(self, mock_pixel, mock_diff, mock_omni):
        """Default config (enable_omniparser=False) should NOT call OmniParser."""
        mock_pixel.return_value = _empty_analysis_result()
        mock_diff.return_value = _empty_analysis_result()

        facade = StateDiscoveryFacade()
        config = DiscoveryConfig(
            algorithm=DiscoveryAlgorithm.COMBINED,
            enable_omniparser=False,
            min_screenshots=2,
        )
        screenshots = _make_screenshots(3)
        facade._discover_combined(screenshots, config, None)

        mock_omni.assert_not_called()


class TestOmniParserTags:
    """Tests for semantic tag propagation."""

    @patch(
        "qontinui.discovery.element_detection.omniparser_detector.OmniParserDetector.analyze",
        new_callable=AsyncMock,
    )
    def test_omniparser_tags_preserved_on_state_images(self, mock_analyze):
        """StateImages from OmniParser should have element_type and 'omniparser' tags."""
        mock_analyze.return_value = _make_omniparser_analysis_result()

        facade = StateDiscoveryFacade()
        config = DiscoveryConfig(
            algorithm=DiscoveryAlgorithm.OMNIPARSER,
            min_screenshots=1,
        )
        screenshots = _make_screenshots(1)

        result = facade.discover_states(screenshots, config)

        assert len(result.state_images) == 2

        button_img = next(si for si in result.state_images if "button" in si.tags)
        assert "omniparser" in button_img.tags
        assert "button" in button_img.tags
        assert button_img.name == "Submit Button"

        input_img = next(si for si in result.state_images if "text_field" in si.tags)
        assert "omniparser" in input_img.tags
        assert "text_field" in input_img.tags


class TestMergeTagTransfer:
    """Tests for tag transfer during result merging."""

    def test_merge_transfers_tags_on_overlap(self):
        """When results overlap, tags from result2 should transfer to result1's image."""
        facade = StateDiscoveryFacade()

        result1 = AnalysisResult(
            states=[],
            state_images=[
                StateImage(
                    id="pixel_img_0",
                    name="Region_0",
                    x=100,
                    y=50,
                    x2=220,
                    y2=90,
                    pixel_hash="pixel_0",
                    frequency=0.95,
                    tags=[],
                ),
            ],
            transitions=[],
        )
        result2 = AnalysisResult(
            states=[],
            state_images=[
                StateImage(
                    id="omni_img_0",
                    name="Submit Button",
                    x=105,
                    y=52,
                    x2=218,
                    y2=88,  # overlaps with pixel_img_0
                    pixel_hash="omni_0",
                    frequency=0.92,
                    tags=["omniparser", "button"],
                ),
            ],
            transitions=[],
        )

        merged = facade._merge_results(result1, result2)

        # Only 1 image (deduplicated)
        assert len(merged.state_images) == 1
        surviving = merged.state_images[0]
        # Original pixel image survives
        assert surviving.id == "pixel_img_0"
        # But now has OmniParser's tags
        assert "omniparser" in surviving.tags
        assert "button" in surviving.tags

    def test_merge_adds_nonoverlapping_omniparser_elements(self):
        """Non-overlapping OmniParser elements should be added to final results."""
        facade = StateDiscoveryFacade()

        result1 = AnalysisResult(
            states=[],
            state_images=[
                StateImage(
                    id="pixel_img_0",
                    name="Region_0",
                    x=100,
                    y=50,
                    x2=220,
                    y2=90,
                    pixel_hash="pixel_0",
                    frequency=0.95,
                ),
            ],
            transitions=[],
        )
        result2 = AnalysisResult(
            states=[],
            state_images=[
                StateImage(
                    id="omni_img_0",
                    name="Username Input",
                    x=300,
                    y=200,
                    x2=500,
                    y2=230,  # no overlap
                    pixel_hash="omni_0",
                    frequency=0.88,
                    tags=["omniparser", "text_field"],
                ),
            ],
            transitions=[],
        )

        merged = facade._merge_results(result1, result2)

        assert len(merged.state_images) == 2
        names = {si.name for si in merged.state_images}
        assert "Region_0" in names
        assert "Username Input" in names

    def test_merge_no_duplicate_tags(self):
        """Tags should not be duplicated when merging overlapping results."""
        facade = StateDiscoveryFacade()

        result1 = AnalysisResult(
            states=[],
            state_images=[
                StateImage(
                    id="pixel_img_0",
                    name="Region_0",
                    x=100,
                    y=50,
                    x2=220,
                    y2=90,
                    pixel_hash="pixel_0",
                    frequency=0.95,
                    tags=["button"],  # already tagged as button
                ),
            ],
            transitions=[],
        )
        result2 = AnalysisResult(
            states=[],
            state_images=[
                StateImage(
                    id="omni_img_0",
                    name="Submit",
                    x=105,
                    y=52,
                    x2=218,
                    y2=88,
                    pixel_hash="omni_0",
                    frequency=0.92,
                    tags=["omniparser", "button"],
                ),
            ],
            transitions=[],
        )

        merged = facade._merge_results(result1, result2)

        surviving = merged.state_images[0]
        assert surviving.tags.count("button") == 1
        assert "omniparser" in surviving.tags
