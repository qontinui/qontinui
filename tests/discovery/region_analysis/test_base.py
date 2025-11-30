"""
Base test cases for region analysis components.

This module provides base test classes and utilities for testing region analysis
functionality including region detection, classification, and relationship analysis.

Example usage:
    >>> import pytest
    >>> from tests.discovery.region_analysis.test_base import BaseRegionAnalyzerTest
    >>>
    >>> class TestMyRegionAnalyzer(BaseRegionAnalyzerTest):
    ...     @pytest.fixture
    ...     def analyzer(self):
    ...         return MyRegionAnalyzer()
    ...
    ...     def test_dialog_detection(self, analyzer, synthetic_screenshot):
    ...         # Test implementation
    ...         pass
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pytest

from tests.fixtures.screenshot_fixtures import ElementSpec, SyntheticScreenshotGenerator


class BaseRegionAnalyzerTest(ABC):
    """
    Abstract base class for region analyzer test cases.

    Provides common test methods and utilities for testing region analyzers.
    Subclass this to test specific region analyzer implementations.

    Example:
        >>> class TestDialogAnalyzer(BaseRegionAnalyzerTest):
        ...     @pytest.fixture
        ...     def analyzer(self):
        ...         return DialogAnalyzer()
        ...
        ...     def test_dialog_detection(self, analyzer, synthetic_screenshot):
        ...         screenshot = synthetic_screenshot(
        ...             elements=[
        ...                 ElementSpec("rectangle", x=200, y=150, width=400, height=300)
        ...             ]
        ...         )
        ...         regions = analyzer.analyze(screenshot)
        ...         assert len(regions) >= 1
    """

    @pytest.fixture
    @abstractmethod
    def analyzer(self):
        """
        Fixture that provides the analyzer instance to test.

        Must be implemented by subclasses.

        Returns:
            Region analyzer instance to test
        """
        pass

    def test_analyzer_initialization(self, analyzer):
        """
        Test that analyzer initializes correctly.

        Args:
            analyzer: Region analyzer instance from fixture
        """
        assert analyzer is not None
        assert hasattr(analyzer, "analyze") or hasattr(analyzer, "detect_regions")

    def test_empty_screenshot(self, analyzer):
        """
        Test analyzer behavior on empty screenshot.

        Args:
            analyzer: Region analyzer instance from fixture
        """
        generator = SyntheticScreenshotGenerator()
        empty_screen = generator.generate(width=800, height=600, elements=[])

        if hasattr(analyzer, "analyze"):
            regions = analyzer.analyze(empty_screen)
        else:
            regions = analyzer.detect_regions(empty_screen)

        # Should return empty list or list with low-confidence regions
        assert isinstance(regions, list)

    def test_single_region_detection(self, analyzer):
        """
        Test detection of a single region.

        Args:
            analyzer: Region analyzer instance from fixture
        """
        generator = SyntheticScreenshotGenerator()
        screenshot = generator.generate(
            width=800,
            height=600,
            elements=[
                # Create a dialog-like region
                ElementSpec(
                    "rectangle",
                    x=200,
                    y=150,
                    width=400,
                    height=300,
                    color=(240, 240, 240),
                    border_color=(100, 100, 100),
                )
            ],
        )

        if hasattr(analyzer, "analyze"):
            regions = analyzer.analyze(screenshot)
        else:
            regions = analyzer.detect_regions(screenshot)

        assert isinstance(regions, list)
        # Should detect at least one region
        assert len(regions) >= 1

    def test_multiple_regions_detection(self, analyzer):
        """
        Test detection of multiple regions.

        Args:
            analyzer: Region analyzer instance from fixture
        """
        generator = SyntheticScreenshotGenerator()
        screenshot = generator.generate(
            width=1024,
            height=768,
            elements=[
                # Toolbar region
                ElementSpec(
                    "rectangle",
                    x=0,
                    y=0,
                    width=1024,
                    height=60,
                    color=(220, 220, 220),
                    border_color=(150, 150, 150),
                ),
                # Content region
                ElementSpec(
                    "rectangle",
                    x=100,
                    y=100,
                    width=600,
                    height=500,
                    color=(255, 255, 255),
                    border_color=(200, 200, 200),
                ),
                # Sidebar region
                ElementSpec(
                    "rectangle",
                    x=750,
                    y=100,
                    width=250,
                    height=500,
                    color=(240, 240, 240),
                    border_color=(180, 180, 180),
                ),
            ],
        )

        if hasattr(analyzer, "analyze"):
            regions = analyzer.analyze(screenshot)
        else:
            regions = analyzer.detect_regions(screenshot)

        assert isinstance(regions, list)
        # Should detect multiple regions
        assert len(regions) >= 2

    def test_region_properties(self, analyzer):
        """
        Test that detected regions have expected properties.

        Args:
            analyzer: Region analyzer instance from fixture
        """
        generator = SyntheticScreenshotGenerator()
        screenshot = generator.generate(
            width=800,
            height=600,
            elements=[
                ElementSpec("rectangle", x=100, y=100, width=200, height=200, color=(240, 240, 240))
            ],
        )

        if hasattr(analyzer, "analyze"):
            regions = analyzer.analyze(screenshot)
        else:
            regions = analyzer.detect_regions(screenshot)

        if len(regions) > 0:
            region = regions[0]
            # Check required properties
            assert hasattr(region, "x") or hasattr(region, "bbox")
            assert hasattr(region, "y") or hasattr(region, "bbox")
            assert hasattr(region, "width") or hasattr(region, "bbox")
            assert hasattr(region, "height") or hasattr(region, "bbox")

    def test_region_hierarchy(self, analyzer):
        """
        Test detection of nested/hierarchical regions.

        Args:
            analyzer: Region analyzer instance from fixture
        """
        generator = SyntheticScreenshotGenerator()
        screenshot = generator.generate(
            width=800,
            height=600,
            elements=[
                # Parent region (dialog)
                ElementSpec(
                    "rectangle",
                    x=150,
                    y=100,
                    width=500,
                    height=400,
                    color=(240, 240, 240),
                    border_color=(100, 100, 100),
                ),
                # Child region (content area)
                ElementSpec(
                    "rectangle",
                    x=170,
                    y=130,
                    width=460,
                    height=340,
                    color=(255, 255, 255),
                    border_color=(150, 150, 150),
                ),
            ],
        )

        if hasattr(analyzer, "analyze"):
            regions = analyzer.analyze(screenshot)
        else:
            regions = analyzer.detect_regions(screenshot)

        # Should detect regions (may or may not detect hierarchy)
        assert isinstance(regions, list)
        assert len(regions) >= 1


class BaseRegionTypeTest:
    """
    Base test class for testing detection of specific region types.

    Provides test methods focused on detecting and validating specific
    region types (dialogs, toolbars, sidebars, content areas, etc.).

    Example:
        >>> class TestDialogRegionDetection(BaseRegionTypeTest):
        ...     region_type = "dialog"
        ...
        ...     @pytest.fixture
        ...     def analyzer(self):
        ...         return DialogRegionAnalyzer()
    """

    region_type: str = "unknown"

    @pytest.fixture
    @abstractmethod
    def analyzer(self):
        """Fixture providing the analyzer instance."""
        pass

    def test_detect_single_region(self, analyzer):
        """Test detection of single region of specified type."""
        generator = SyntheticScreenshotGenerator()

        # Create region based on type
        if self.region_type == "dialog":
            elements = [
                ElementSpec(
                    "rectangle",
                    x=200,
                    y=150,
                    width=400,
                    height=300,
                    color=(240, 240, 240),
                    border_color=(100, 100, 100),
                )
            ]
        elif self.region_type == "toolbar":
            elements = [
                ElementSpec(
                    "rectangle",
                    x=0,
                    y=0,
                    width=800,
                    height=50,
                    color=(220, 220, 220),
                    border_color=(150, 150, 150),
                )
            ]
        elif self.region_type == "sidebar":
            elements = [
                ElementSpec(
                    "rectangle",
                    x=0,
                    y=0,
                    width=200,
                    height=600,
                    color=(240, 240, 240),
                    border_color=(180, 180, 180),
                )
            ]
        else:
            elements = [
                ElementSpec(
                    "rectangle",
                    x=100,
                    y=100,
                    width=300,
                    height=300,
                    color=(255, 255, 255),
                    border_color=(200, 200, 200),
                )
            ]

        screenshot = generator.generate(width=800, height=600, elements=elements)

        if hasattr(analyzer, "analyze"):
            regions = analyzer.analyze(screenshot)
        else:
            regions = analyzer.detect_regions(screenshot)

        # Should detect the region
        assert len(regions) >= 1

    def test_region_size_validation(self, analyzer):
        """Test that detected regions have reasonable sizes."""
        generator = SyntheticScreenshotGenerator()
        screenshot = generator.generate(
            width=800,
            height=600,
            elements=[
                ElementSpec("rectangle", x=100, y=100, width=200, height=200, color=(240, 240, 240))
            ],
        )

        if hasattr(analyzer, "analyze"):
            regions = analyzer.analyze(screenshot)
        else:
            regions = analyzer.detect_regions(screenshot)

        for region in regions:
            if hasattr(region, "width") and hasattr(region, "height"):
                # Regions should have positive dimensions
                assert region.width > 0
                assert region.height > 0
                # Regions should fit within screenshot
                assert region.width <= 800
                assert region.height <= 600


class RegionRelationshipTest:
    """
    Base class for testing relationships between regions.

    Provides utilities for testing spatial relationships, containment,
    overlap, and other region interactions.

    Example:
        >>> class TestRegionHierarchy(RegionRelationshipTest):
        ...     @pytest.fixture
        ...     def analyzer(self):
        ...         return HierarchicalRegionAnalyzer()
        ...
        ...     def test_nested_regions(self, analyzer):
        ...         # Test implementation
        ...         pass
    """

    @pytest.fixture
    @abstractmethod
    def analyzer(self):
        """Fixture providing the analyzer instance."""
        pass

    def test_overlapping_regions(self, analyzer):
        """Test handling of overlapping regions."""
        generator = SyntheticScreenshotGenerator()
        screenshot = generator.generate(
            width=800,
            height=600,
            elements=[
                ElementSpec(
                    "rectangle",
                    x=100,
                    y=100,
                    width=300,
                    height=300,
                    color=(240, 240, 240),
                    border_color=(100, 100, 100),
                ),
                ElementSpec(
                    "rectangle",
                    x=200,
                    y=200,
                    width=300,
                    height=300,
                    color=(220, 220, 220),
                    border_color=(120, 120, 120),
                ),
            ],
        )

        if hasattr(analyzer, "analyze"):
            regions = analyzer.analyze(screenshot)
        else:
            regions = analyzer.detect_regions(screenshot)

        # Should handle overlapping regions (may merge or keep separate)
        assert isinstance(regions, list)

    def test_adjacent_regions(self, analyzer):
        """Test detection of adjacent regions."""
        generator = SyntheticScreenshotGenerator()
        screenshot = generator.generate(
            width=800,
            height=600,
            elements=[
                # Left region
                ElementSpec(
                    "rectangle",
                    x=50,
                    y=100,
                    width=300,
                    height=400,
                    color=(240, 240, 240),
                    border_color=(100, 100, 100),
                ),
                # Right region (adjacent)
                ElementSpec(
                    "rectangle",
                    x=350,
                    y=100,
                    width=400,
                    height=400,
                    color=(220, 220, 220),
                    border_color=(120, 120, 120),
                ),
            ],
        )

        if hasattr(analyzer, "analyze"):
            regions = analyzer.analyze(screenshot)
        else:
            regions = analyzer.detect_regions(screenshot)

        # Should detect separate adjacent regions
        assert len(regions) >= 1

    def test_contained_regions(self, analyzer):
        """Test detection of contained/nested regions."""
        generator = SyntheticScreenshotGenerator()
        screenshot = generator.generate(
            width=800,
            height=600,
            elements=[
                # Parent region
                ElementSpec(
                    "rectangle",
                    x=100,
                    y=100,
                    width=600,
                    height=400,
                    color=(240, 240, 240),
                    border_color=(100, 100, 100),
                ),
                # Child region (inside parent)
                ElementSpec(
                    "rectangle",
                    x=150,
                    y=150,
                    width=500,
                    height=300,
                    color=(255, 255, 255),
                    border_color=(150, 150, 150),
                ),
            ],
        )

        if hasattr(analyzer, "analyze"):
            regions = analyzer.analyze(screenshot)
        else:
            regions = analyzer.detect_regions(screenshot)

        # Should detect regions (hierarchy detection is optional)
        assert isinstance(regions, list)
        assert len(regions) >= 1


class RegionPerformanceTest:
    """
    Base class for performance testing of region analyzers.

    Provides utilities for measuring analysis speed and efficiency.

    Example:
        >>> class TestAnalyzerPerformance(RegionPerformanceTest):
        ...     @pytest.fixture
        ...     def analyzer(self):
        ...         return MyRegionAnalyzer()
        ...
        ...     def test_analysis_speed(self, analyzer):
        ...         self.measure_analysis_time(analyzer, num_runs=10)
    """

    @pytest.fixture
    @abstractmethod
    def analyzer(self):
        """Fixture providing the analyzer instance."""
        pass

    def measure_analysis_time(
        self, analyzer, screenshot: np.ndarray | None = None, num_runs: int = 10
    ) -> float:
        """
        Measure average region analysis time.

        Args:
            analyzer: Region analyzer instance
            screenshot: Screenshot to test with (generates one if None)
            num_runs: Number of analysis runs to average

        Returns:
            Average analysis time in seconds
        """
        import time

        if screenshot is None:
            generator = SyntheticScreenshotGenerator()
            screenshot = generator.generate(
                width=1024,
                height=768,
                elements=[ElementSpec("rectangle", x=100, y=100, width=400, height=300)],
            )

        times = []
        for _ in range(num_runs):
            start = time.time()
            if hasattr(analyzer, "analyze"):
                analyzer.analyze(screenshot)
            else:
                analyzer.detect_regions(screenshot)
            times.append(time.time() - start)

        return sum(times) / len(times)

    def test_analysis_speed_reasonable(self, analyzer):
        """
        Test that region analysis completes in reasonable time.

        Args:
            analyzer: Region analyzer instance from fixture
        """
        avg_time = self.measure_analysis_time(analyzer, num_runs=5)
        # Should complete in less than 10 seconds per analysis
        assert avg_time < 10.0, f"Region analysis too slow: {avg_time:.2f}s"


# Utility functions for region testing


def calculate_region_overlap(
    region1: tuple[int, int, int, int], region2: tuple[int, int, int, int]
) -> float:
    """
    Calculate overlap ratio between two regions.

    Args:
        region1: First region as (x, y, width, height)
        region2: Second region as (x, y, width, height)

    Returns:
        Overlap ratio (intersection area / smaller region area)

    Example:
        >>> region1 = (0, 0, 100, 100)
        >>> region2 = (50, 50, 100, 100)
        >>> overlap = calculate_region_overlap(region1, region2)
        >>> assert 0 < overlap < 1
    """
    x1, y1, w1, h1 = region1
    x2, y2, w2, h2 = region2

    # Calculate intersection
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection = x_overlap * y_overlap

    if intersection == 0:
        return 0.0

    # Use smaller region as denominator
    area1 = w1 * h1
    area2 = w2 * h2
    min_area = min(area1, area2)

    return intersection / min_area if min_area > 0 else 0.0


def is_region_contained(
    inner: tuple[int, int, int, int], outer: tuple[int, int, int, int], tolerance: int = 5
) -> bool:
    """
    Check if one region is contained within another.

    Args:
        inner: Inner region as (x, y, width, height)
        outer: Outer region as (x, y, width, height)
        tolerance: Pixel tolerance for containment check

    Returns:
        True if inner is contained in outer

    Example:
        >>> inner = (100, 100, 50, 50)
        >>> outer = (50, 50, 200, 200)
        >>> assert is_region_contained(inner, outer)
    """
    x1, y1, w1, h1 = inner
    x2, y2, w2, h2 = outer

    return (
        x1 >= x2 - tolerance
        and y1 >= y2 - tolerance
        and x1 + w1 <= x2 + w2 + tolerance
        and y1 + h1 <= y2 + h2 + tolerance
    )


def are_regions_adjacent(
    region1: tuple[int, int, int, int], region2: tuple[int, int, int, int], tolerance: int = 10
) -> bool:
    """
    Check if two regions are adjacent (touching or very close).

    Args:
        region1: First region as (x, y, width, height)
        region2: Second region as (x, y, width, height)
        tolerance: Maximum pixel distance to consider adjacent

    Returns:
        True if regions are adjacent

    Example:
        >>> region1 = (0, 0, 100, 100)
        >>> region2 = (100, 0, 100, 100)
        >>> assert are_regions_adjacent(region1, region2)
    """
    x1, y1, w1, h1 = region1
    x2, y2, w2, h2 = region2

    # Check horizontal adjacency
    if abs(x1 + w1 - x2) <= tolerance or abs(x2 + w2 - x1) <= tolerance:
        # Check vertical overlap
        if not (y1 + h1 < y2 or y2 + h2 < y1):
            return True

    # Check vertical adjacency
    if abs(y1 + h1 - y2) <= tolerance or abs(y2 + h2 - y1) <= tolerance:
        # Check horizontal overlap
        if not (x1 + w1 < x2 or x2 + w2 < x1):
            return True

    return False


def assert_region_valid(region: tuple[int, int, int, int], max_width: int, max_height: int):
    """
    Assert that a region is valid.

    Args:
        region: Region as (x, y, width, height)
        max_width: Maximum valid width
        max_height: Maximum valid height

    Raises:
        AssertionError: If region is invalid
    """
    x, y, w, h = region
    assert x >= 0, f"Negative x coordinate: {x}"
    assert y >= 0, f"Negative y coordinate: {y}"
    assert w > 0, f"Non-positive width: {w}"
    assert h > 0, f"Non-positive height: {h}"
    assert x + w <= max_width, f"Region extends beyond width: {x + w} > {max_width}"
    assert y + h <= max_height, f"Region extends beyond height: {y + h} > {max_height}"


def find_regions_by_type(regions: list[Any], region_type: str) -> list[Any]:
    """
    Filter regions by type.

    Args:
        regions: List of region objects
        region_type: Type to filter by

    Returns:
        List of regions matching the specified type

    Example:
        >>> regions = [
        ...     MockRegion(0, 0, 100, 100, "dialog"),
        ...     MockRegion(0, 0, 800, 50, "toolbar"),
        ... ]
        >>> dialogs = find_regions_by_type(regions, "dialog")
        >>> assert len(dialogs) == 1
    """
    result = []
    for region in regions:
        if hasattr(region, "region_type") and region.region_type == region_type:
            result.append(region)
        elif hasattr(region, "type") and region.type == region_type:
            result.append(region)

    return result
