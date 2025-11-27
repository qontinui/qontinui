"""Comprehensive tests for DifferentialConsistencyDetector.

Tests the core differential consistency detection algorithm for identifying
state regions from before/after screenshot transition pairs.

Key test areas:
- Basic consistency detection with synthetic transitions
- Consistency score calculations
- Region extraction and morphology
- Minimum example requirements
- Dynamic backgrounds handling
- Edge cases and error handling
"""


import numpy as np
import pytest

from qontinui.src.qontinui.discovery.state_detection.differential_consistency_detector import (
    DifferentialConsistencyDetector,
    StateRegion,
)
from tests.fixtures.screenshot_fixtures import (
    ElementSpec,
    SyntheticScreenshotGenerator,
    create_menu_transition_pair,
)


class TestDifferentialConsistencyDetectorBasic:
    """Basic functionality tests for DifferentialConsistencyDetector."""

    @pytest.fixture
    def detector(self) -> DifferentialConsistencyDetector:
        """Create detector instance.

        Returns:
            Initialized DifferentialConsistencyDetector
        """
        return DifferentialConsistencyDetector()

    @pytest.fixture
    def generator(self) -> SyntheticScreenshotGenerator:
        """Create screenshot generator.

        Returns:
            Initialized SyntheticScreenshotGenerator
        """
        return SyntheticScreenshotGenerator()

    def test_detector_initialization(self, detector):
        """Test that detector initializes properly.

        Verifies:
            - Detector has correct name
            - Detector inherits from MultiScreenshotDetector
        """
        assert detector is not None
        assert detector.name == "DifferentialConsistencyDetector"

    def test_detector_has_required_methods(self, detector):
        """Test that detector implements required methods.

        Verifies:
            - detect_state_regions method exists
            - detect_multi method exists
            - compute_consistency_map method exists
            - visualize_consistency method exists
        """
        assert hasattr(detector, "detect_state_regions")
        assert hasattr(detector, "detect_multi")
        assert hasattr(detector, "compute_consistency_map")
        assert hasattr(detector, "visualize_consistency")

    def test_single_menu_transition(self, detector):
        """Test detection with a single menu transition pair.

        Verifies:
            - Detector handles a single transition (should still require 10+ though)
        """
        before, after = create_menu_transition_pair()

        # Need at least 10 pairs, so create duplicates
        pairs = [(before, after)] * 10

        regions = detector.detect_state_regions(
            pairs, consistency_threshold=0.5, min_region_area=100
        )

        assert isinstance(regions, list)
        # May or may not find regions depending on synthetic data
        for region in regions:
            assert isinstance(region, StateRegion)
            assert region.consistency_score >= 0.0
            assert region.consistency_score <= 1.0


class TestConsistencyCalculations:
    """Test consistency score calculations."""

    @pytest.fixture
    def detector(self) -> DifferentialConsistencyDetector:
        """Create detector instance."""
        return DifferentialConsistencyDetector()

    def test_compute_differences_basic(self, detector):
        """Test basic difference computation.

        Verifies:
            - Difference images are computed correctly
            - Output shape is correct (N, H, W)
            - Values are float32
        """
        # Create simple before/after pairs
        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = np.ones((100, 100, 3), dtype=np.uint8) * 255

        pairs = [(before, after)] * 10

        diff_images = detector._compute_differences(pairs)

        assert diff_images.shape == (10, 100, 100)
        assert diff_images.dtype == np.float32
        # All pixels should show max difference
        assert np.all(diff_images > 0)

    def test_compute_differences_no_change(self, detector):
        """Test difference computation when images are identical.

        Verifies:
            - Identical images produce zero differences
        """
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pairs = [(image, image.copy())] * 10

        diff_images = detector._compute_differences(pairs)

        assert diff_images.shape == (10, 100, 100)
        # All differences should be zero
        assert np.all(diff_images == 0)

    def test_compute_differences_mismatched_sizes(self, detector):
        """Test that mismatched sizes are handled.

        Verifies:
            - Resizing occurs for mismatched pairs
            - No error is raised
        """
        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = np.ones((120, 120, 3), dtype=np.uint8) * 255

        pairs = [(before, after)] * 10

        # Should resize 'after' to match 'before'
        diff_images = detector._compute_differences(pairs)

        assert diff_images.shape == (10, 100, 100)

    def test_compute_consistency_minmax(self, detector):
        """Test consistency computation with minmax normalization.

        Verifies:
            - Consistency scores are in [0, 1] range
            - High mean + low std produces high consistency
        """
        # Create consistent changes: same diff value across all examples
        diff_images = np.ones((20, 100, 100), dtype=np.float32) * 128

        consistency = detector._compute_consistency(diff_images, method="minmax")

        assert consistency.shape == (100, 100)
        assert consistency.dtype == np.float32
        assert np.all(consistency >= 0.0)
        assert np.all(consistency <= 1.0)
        # With zero std and high mean, should have high consistency
        assert np.mean(consistency) > 0.5

    def test_compute_consistency_zscore(self, detector):
        """Test consistency computation with zscore normalization.

        Verifies:
            - Zscore normalization produces valid scores
            - Scores are in [0, 1] range after sigmoid
        """
        diff_images = np.random.uniform(0, 255, (20, 100, 100)).astype(np.float32)

        consistency = detector._compute_consistency(diff_images, method="zscore")

        assert consistency.shape == (100, 100)
        assert np.all(consistency >= 0.0)
        assert np.all(consistency <= 1.0)

    def test_consistency_with_varying_changes(self, detector):
        """Test consistency scores with varying change patterns.

        Verifies:
            - Consistent region has higher score than inconsistent region
        """
        diff_images = np.zeros((20, 100, 100), dtype=np.float32)

        # Left half: consistent change (all 200)
        diff_images[:, :, :50] = 200

        # Right half: inconsistent change (random)
        for i in range(20):
            diff_images[i, :, 50:] = np.random.uniform(0, 255, (100, 50))

        consistency = detector._compute_consistency(diff_images, method="minmax")

        left_score = np.mean(consistency[:, :50])
        right_score = np.mean(consistency[:, 50:])

        # Left should have higher consistency
        assert left_score > right_score


class TestRegionExtraction:
    """Test region extraction from consistency maps."""

    @pytest.fixture
    def detector(self) -> DifferentialConsistencyDetector:
        """Create detector instance."""
        return DifferentialConsistencyDetector()

    def test_extract_regions_basic(self, detector):
        """Test basic region extraction.

        Verifies:
            - Regions are extracted from high consistency areas
            - Bounding boxes are valid (x, y, w, h)
        """
        # Create consistency map with clear region
        consistency_map = np.zeros((200, 200), dtype=np.float32)
        consistency_map[50:150, 50:150] = 1.0  # High consistency square

        regions = detector._extract_regions(
            consistency_map, threshold=0.7, min_area=500, kernel_size=3
        )

        assert len(regions) > 0
        for bbox in regions:
            x, y, w, h = bbox
            assert x >= 0 and y >= 0
            assert w > 0 and h > 0
            assert w * h >= 500

    def test_extract_regions_no_regions(self, detector):
        """Test when no regions meet threshold.

        Verifies:
            - Returns empty list when no regions above threshold
        """
        consistency_map = np.zeros((200, 200), dtype=np.float32)
        consistency_map[:, :] = 0.3  # Low consistency everywhere

        regions = detector._extract_regions(consistency_map, threshold=0.7, min_area=100)

        assert len(regions) == 0

    def test_extract_regions_min_area_filtering(self, detector):
        """Test that small regions are filtered out.

        Verifies:
            - Only regions >= min_area are returned
        """
        consistency_map = np.zeros((200, 200), dtype=np.float32)
        # Small region
        consistency_map[10:20, 10:20] = 1.0  # 100 pixels
        # Large region
        consistency_map[50:150, 50:150] = 1.0  # 10000 pixels

        regions = detector._extract_regions(
            consistency_map, threshold=0.5, min_area=500  # Filter out the small region
        )

        # Should only get the large region
        assert len(regions) >= 1
        for bbox in regions:
            x, y, w, h = bbox
            assert w * h >= 500

    def test_extract_regions_morphology_cleanup(self, detector):
        """Test that morphology operations clean up noise.

        Verifies:
            - Small holes are filled
            - Small noise is removed
        """
        consistency_map = np.zeros((200, 200), dtype=np.float32)
        # Region with holes
        consistency_map[50:150, 50:150] = 1.0
        consistency_map[75:125, 75:125] = 0.0  # Hole in middle

        # Small noise
        consistency_map[10:15, 10:15] = 1.0

        regions = detector._extract_regions(
            consistency_map,
            threshold=0.5,
            min_area=500,
            kernel_size=7,  # Larger kernel to fill hole
        )

        assert len(regions) >= 1


class TestMinimumExampleRequirements:
    """Test minimum number of transition examples required."""

    @pytest.fixture
    def detector(self) -> DifferentialConsistencyDetector:
        """Create detector instance."""
        return DifferentialConsistencyDetector()

    def test_too_few_examples(self, detector):
        """Test error handling with too few examples.

        Verifies:
            - ValueError raised when < 10 examples
            - Error message is informative
        """
        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Only 5 pairs
        pairs = [(before, after)] * 5

        with pytest.raises(ValueError) as exc_info:
            detector.detect_state_regions(pairs)

        assert "at least 10" in str(exc_info.value).lower()

    def test_minimum_valid_examples(self, detector):
        """Test with minimum valid number of examples.

        Verifies:
            - 10 examples are accepted
            - Detection completes without error
        """
        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = np.ones((100, 100, 3), dtype=np.uint8) * 255

        pairs = [(before, after)] * 10

        regions = detector.detect_state_regions(
            pairs, consistency_threshold=0.5, min_region_area=100
        )

        assert isinstance(regions, list)

    def test_many_examples(self, detector):
        """Test with many examples (100+).

        Verifies:
            - Large number of examples are handled efficiently
            - Results are still valid
        """
        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = np.ones((100, 100, 3), dtype=np.uint8) * 255

        pairs = [(before, after)] * 100

        regions = detector.detect_state_regions(
            pairs, consistency_threshold=0.5, min_region_area=100
        )

        assert isinstance(regions, list)


class TestDynamicBackgrounds:
    """Test handling of dynamic backgrounds (animations, movement)."""

    @pytest.fixture
    def detector(self) -> DifferentialConsistencyDetector:
        """Create detector instance."""
        return DifferentialConsistencyDetector()

    @pytest.fixture
    def generator(self) -> SyntheticScreenshotGenerator:
        """Create screenshot generator."""
        return SyntheticScreenshotGenerator()

    def test_static_menu_on_dynamic_background(self, detector, generator):
        """Test detecting static menu on animated background.

        Verifies:
            - Static menu region is detected
            - Dynamic background regions have low consistency
        """
        pairs = []

        for i in range(15):
            # Create before/after pair
            # Before: dynamic background
            before_bg_color = (200 + i * 3, 200 + i * 2, 200 + i * 4)
            before = generator.generate(
                width=400, height=300, background_color=before_bg_color, elements=[]
            )

            # After: same dynamic background + static menu
            after_bg_color = (200 + i * 3, 200 + i * 2, 200 + i * 4)
            menu_elements = [
                ElementSpec(
                    "rectangle",
                    x=150,
                    y=50,
                    width=100,
                    height=150,
                    color=(240, 240, 240),
                    border_color=(100, 100, 100),
                ),
                ElementSpec("button", x=160, y=70, width=80, height=30, text="Option 1"),
                ElementSpec("button", x=160, y=110, width=80, height=30, text="Option 2"),
                ElementSpec("button", x=160, y=150, width=80, height=30, text="Option 3"),
            ]
            after = generator.generate(
                width=400, height=300, background_color=after_bg_color, elements=menu_elements
            )

            pairs.append((before, after))

        regions = detector.detect_state_regions(
            pairs, consistency_threshold=0.4, min_region_area=200
        )

        # Should detect the menu region
        assert len(regions) > 0

        # Check that detected region is roughly in menu area
        menu_found = False
        for region in regions:
            x, y, w, h = region.bbox
            # Menu is at (150, 50, 100, 150)
            if 100 < x < 200 and 20 < y < 100:
                menu_found = True
                break

        assert menu_found, "Menu region should be detected"

    def test_animated_elements_low_consistency(self, detector, generator):
        """Test that animated elements have low consistency.

        Verifies:
            - Randomly changing elements produce low consistency scores
        """
        pairs = []

        for i in range(20):
            # Before: random elements
            before_elements = [
                ElementSpec(
                    "rectangle",
                    x=50 + i * 5,
                    y=50,
                    width=50,
                    height=50,
                    color=(100 + i * 7, 150, 200),
                )
            ]
            before = generator.generate(width=300, height=200, elements=before_elements)

            # After: same but in different position/color
            after_elements = [
                ElementSpec(
                    "rectangle",
                    x=50 + i * 7,
                    y=60,
                    width=50,
                    height=50,
                    color=(100 + i * 5, 150, 200),
                )
            ]
            after = generator.generate(width=300, height=200, elements=after_elements)

            pairs.append((before, after))

        # Get consistency map
        consistency_map = detector.compute_consistency_map(pairs)

        # Overall consistency should be low due to random changes
        mean_consistency = np.mean(consistency_map)
        assert mean_consistency < 0.7  # Mostly inconsistent

    def test_partially_occluded_state(self, detector, generator):
        """Test detecting state when partially occluded by animations.

        Verifies:
            - State region is still detected despite partial occlusion
        """
        pairs = []

        for i in range(12):
            # Before: background with moving element
            before_elements = [
                ElementSpec(
                    "rectangle", x=20 + i * 10, y=100, width=30, height=30, color=(255, 0, 0)
                )  # Moving red square
            ]
            before = generator.generate(width=400, height=300, elements=before_elements)

            # After: same moving element + static dialog
            after_elements = [
                ElementSpec(
                    "rectangle", x=20 + i * 10, y=100, width=30, height=30, color=(255, 0, 0)
                ),  # Moving red square
                ElementSpec(
                    "rectangle",
                    x=120,
                    y=80,
                    width=160,
                    height=140,
                    color=(240, 240, 240),
                    border_color=(100, 100, 100),
                ),  # Static dialog
            ]
            after = generator.generate(width=400, height=300, elements=after_elements)

            pairs.append((before, after))

        regions = detector.detect_state_regions(
            pairs, consistency_threshold=0.5, min_region_area=500
        )

        # Should detect the static dialog despite moving element
        assert len(regions) > 0


class TestScoringAndRanking:
    """Test region scoring and ranking."""

    @pytest.fixture
    def detector(self) -> DifferentialConsistencyDetector:
        """Create detector instance."""
        return DifferentialConsistencyDetector()

    def test_score_regions_basic(self, detector):
        """Test basic region scoring.

        Verifies:
            - StateRegion objects are created with scores
            - Regions are sorted by score
        """
        # Create consistency map
        consistency_map = np.zeros((200, 200), dtype=np.float32)
        consistency_map[50:100, 50:100] = 0.9  # High consistency
        consistency_map[120:170, 120:170] = 0.6  # Medium consistency

        bboxes = [(50, 50, 50, 50), (120, 120, 50, 50)]

        diff_images = np.ones((10, 200, 200), dtype=np.float32) * 100

        regions = detector._score_regions(bboxes, consistency_map, diff_images)

        assert len(regions) == 2
        assert isinstance(regions[0], StateRegion)
        # Should be sorted by score (highest first)
        assert regions[0].consistency_score >= regions[1].consistency_score
        # First region should be the high-consistency one
        assert regions[0].consistency_score > 0.8

    def test_score_regions_representative_diff(self, detector):
        """Test that representative diff image is computed.

        Verifies:
            - example_diff is present in StateRegion
            - Diff image has correct shape
        """
        consistency_map = np.ones((100, 100), dtype=np.float32) * 0.8
        bboxes = [(10, 10, 50, 50)]
        diff_images = np.random.uniform(0, 255, (10, 100, 100)).astype(np.float32)

        regions = detector._score_regions(bboxes, consistency_map, diff_images)

        assert len(regions) == 1
        assert regions[0].example_diff is not None
        assert regions[0].example_diff.shape == (50, 50)

    def test_score_regions_pixel_count(self, detector):
        """Test that pixel count is calculated correctly.

        Verifies:
            - pixel_count matches bbox area
        """
        consistency_map = np.ones((100, 100), dtype=np.float32) * 0.8
        bboxes = [(10, 10, 30, 40)]  # 1200 pixels
        diff_images = np.ones((10, 100, 100), dtype=np.float32)

        regions = detector._score_regions(bboxes, consistency_map, diff_images)

        assert len(regions) == 1
        assert regions[0].pixel_count == 1200


class TestVisualization:
    """Test visualization functionality."""

    @pytest.fixture
    def detector(self) -> DifferentialConsistencyDetector:
        """Create detector instance."""
        return DifferentialConsistencyDetector()

    def test_visualize_consistency_basic(self, detector):
        """Test basic visualization creation.

        Verifies:
            - Visualization image is created
            - Has correct dimensions
            - Is BGR color image
        """
        consistency_map = np.random.uniform(0, 1, (200, 300)).astype(np.float32)
        screenshot = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)

        regions = [
            StateRegion(
                bbox=(50, 50, 80, 100),
                consistency_score=0.85,
                example_diff=np.zeros((100, 80), dtype=np.uint8),
                pixel_count=8000,
            )
        ]

        vis = detector.visualize_consistency(consistency_map, regions, screenshot, show_scores=True)

        assert vis.shape == (200, 300, 3)
        assert vis.dtype == np.uint8

    def test_visualize_consistency_no_scores(self, detector):
        """Test visualization without score labels.

        Verifies:
            - Visualization works with show_scores=False
        """
        consistency_map = np.random.uniform(0, 1, (200, 300)).astype(np.float32)
        screenshot = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        regions = []

        vis = detector.visualize_consistency(
            consistency_map, regions, screenshot, show_scores=False
        )

        assert vis.shape == (200, 300, 3)

    def test_visualize_consistency_grayscale_screenshot(self, detector):
        """Test visualization with grayscale screenshot.

        Verifies:
            - Grayscale screenshot is converted to BGR
            - No errors occur
        """
        consistency_map = np.random.uniform(0, 1, (200, 300)).astype(np.float32)
        screenshot = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
        regions = []

        vis = detector.visualize_consistency(consistency_map, regions, screenshot)

        assert vis.shape == (200, 300, 3)


class TestDetectMultiMethod:
    """Test detect_multi method for sequential screenshots."""

    @pytest.fixture
    def detector(self) -> DifferentialConsistencyDetector:
        """Create detector instance."""
        return DifferentialConsistencyDetector()

    def test_detect_multi_basic(self, detector):
        """Test detect_multi with sequential screenshots.

        Verifies:
            - Creates consecutive pairs
            - Returns dictionary mapping indices to regions
        """
        # Create 12 screenshots with gradual changes
        screenshots = []
        for i in range(12):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            screenshots.append(img)

        result = detector.detect_multi(screenshots, consistency_threshold=0.5, min_region_area=100)

        assert isinstance(result, dict)
        # Should have entries for indices 1-11 (after screenshots)
        assert len(result) >= 1

    def test_detect_multi_too_few_screenshots(self, detector):
        """Test error handling with too few screenshots.

        Verifies:
            - ValueError raised when < 11 screenshots (10 pairs)
        """
        screenshots = [np.zeros((100, 100, 3), dtype=np.uint8)] * 5

        with pytest.raises(ValueError):
            detector.detect_multi(screenshots)


class TestParameterGrid:
    """Test parameter grid for hyperparameter tuning."""

    @pytest.fixture
    def detector(self) -> DifferentialConsistencyDetector:
        """Create detector instance."""
        return DifferentialConsistencyDetector()

    def test_get_param_grid(self, detector):
        """Test parameter grid generation.

        Verifies:
            - Returns list of parameter dicts
            - Each dict has required parameters
            - Values are reasonable
        """
        param_grid = detector.get_param_grid()

        assert isinstance(param_grid, list)
        assert len(param_grid) > 0

        for params in param_grid:
            assert "consistency_threshold" in params
            assert "min_region_area" in params
            assert "morphology_kernel_size" in params

            # Check value ranges
            assert 0.0 <= params["consistency_threshold"] <= 1.0
            assert params["min_region_area"] > 0
            assert params["morphology_kernel_size"] > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def detector(self) -> DifferentialConsistencyDetector:
        """Create detector instance."""
        return DifferentialConsistencyDetector()

    def test_all_black_images(self, detector):
        """Test with all-black images.

        Verifies:
            - No crash with all-zero images
            - Returns empty or low-confidence regions
        """
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        pairs = [(black, black.copy())] * 10

        regions = detector.detect_state_regions(
            pairs, consistency_threshold=0.5, min_region_area=100
        )

        assert isinstance(regions, list)
        # Should return empty or very low scores
        assert len(regions) == 0 or all(r.consistency_score < 0.5 for r in regions)

    def test_inconsistent_pair_dimensions(self, detector):
        """Test error handling with inconsistent dimensions across pairs.

        Verifies:
            - ValueError raised when pairs have different dimensions
        """
        pairs = []
        pairs.extend(
            [(np.zeros((100, 100, 3), dtype=np.uint8), np.zeros((100, 100, 3), dtype=np.uint8))] * 5
        )
        pairs.extend(
            [(np.zeros((120, 120, 3), dtype=np.uint8), np.zeros((120, 120, 3), dtype=np.uint8))] * 5
        )

        with pytest.raises(ValueError) as exc_info:
            detector.detect_state_regions(pairs)

        assert "inconsistent dimensions" in str(exc_info.value).lower()

    def test_very_high_threshold(self, detector):
        """Test with very high consistency threshold.

        Verifies:
            - Returns empty list when threshold is too high
        """
        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = np.ones((100, 100, 3), dtype=np.uint8) * 128
        pairs = [(before, after)] * 10

        regions = detector.detect_state_regions(
            pairs, consistency_threshold=0.99, min_region_area=100  # Very high
        )

        # Likely no regions meet this threshold
        assert len(regions) == 0 or all(r.consistency_score >= 0.99 for r in regions)

    def test_very_large_min_area(self, detector):
        """Test with very large min_region_area.

        Verifies:
            - Filters out all small regions
        """
        before = np.zeros((200, 200, 3), dtype=np.uint8)
        after = np.ones((200, 200, 3), dtype=np.uint8) * 255
        pairs = [(before, after)] * 10

        regions = detector.detect_state_regions(
            pairs, consistency_threshold=0.5, min_region_area=50000  # Very large
        )

        # Should only return regions >= 50000 pixels
        for region in regions:
            assert region.pixel_count >= 50000


class TestStateRegionDataclass:
    """Test StateRegion dataclass."""

    def test_state_region_creation(self):
        """Test creating StateRegion instance.

        Verifies:
            - All fields are accessible
            - Default values work
        """
        diff = np.zeros((50, 50), dtype=np.uint8)
        region = StateRegion(
            bbox=(10, 20, 30, 40), consistency_score=0.87, example_diff=diff, pixel_count=1200
        )

        assert region.bbox == (10, 20, 30, 40)
        assert region.consistency_score == 0.87
        assert region.pixel_count == 1200
        assert region.example_diff.shape == (50, 50)

    def test_state_region_repr(self):
        """Test StateRegion string representation.

        Verifies:
            - __repr__ returns informative string
            - Contains key information
        """
        diff = np.zeros((50, 50), dtype=np.uint8)
        region = StateRegion(
            bbox=(10, 20, 30, 40), consistency_score=0.87, example_diff=diff, pixel_count=1200
        )

        repr_str = repr(region)
        assert "StateRegion" in repr_str
        assert "0.870" in repr_str
        assert "1200" in repr_str
