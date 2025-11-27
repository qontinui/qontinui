"""
Detector fixtures for testing element detection components.

This module provides mock detectors, test data, and common fixtures
for testing element detection, region analysis, and state detection.

Example usage:
    >>> import pytest
    >>> from tests.fixtures.detector_fixtures import mock_element_detector
    >>>
    >>> def test_detection(mock_element_detector):
    ...     detector = mock_element_detector()
    ...     elements = detector.detect(screenshot)
    ...     assert len(elements) > 0
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest


@dataclass
class MockDetectionResult:
    """Mock detection result for testing."""

    element_type: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x, y, width, height)
    mask: np.ndarray | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class MockRegion:
    """Mock region for testing region analysis."""

    x: int
    y: int
    width: int
    height: int
    region_type: str = "unknown"
    elements: list[MockDetectionResult] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is within region."""
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height


@dataclass
class MockState:
    """Mock state for testing state detection."""

    state_id: str
    name: str
    regions: list[MockRegion] = field(default_factory=list)
    elements: list[MockDetectionResult] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_element_count(self) -> int:
        """Get total element count."""
        return len(self.elements) + sum(len(r.elements) for r in self.regions)


class MockElementDetector:
    """
    Mock element detector for testing.

    This provides a configurable mock detector that can return predetermined
    detection results for testing detection pipelines.

    Example:
        >>> detector = MockElementDetector(
        ...     detection_results=[
        ...         MockDetectionResult("button", 0.95, (100, 100, 80, 40))
        ...     ]
        ... )
        >>> results = detector.detect(screenshot)
        >>> assert len(results) == 1
        >>> assert results[0].element_type == "button"
    """

    def __init__(
        self,
        detection_results: list[MockDetectionResult] | None = None,
        detection_delay: float = 0.0,
        should_fail: bool = False,
    ):
        """
        Initialize mock detector.

        Args:
            detection_results: List of results to return from detect()
            detection_delay: Simulated detection delay in seconds
            should_fail: If True, detect() will raise an exception
        """
        self.detection_results = detection_results or []
        self.detection_delay = detection_delay
        self.should_fail = should_fail
        self.call_count = 0
        self.last_screenshot = None

    def detect(self, screenshot: np.ndarray) -> list[MockDetectionResult]:
        """
        Mock detection method.

        Args:
            screenshot: Input screenshot

        Returns:
            List of mock detection results

        Raises:
            RuntimeError: If should_fail is True
        """
        self.call_count += 1
        self.last_screenshot = screenshot

        if self.should_fail:
            raise RuntimeError("Mock detector configured to fail")

        import time

        if self.detection_delay > 0:
            time.sleep(self.detection_delay)

        return self.detection_results.copy()

    def reset(self):
        """Reset detector state."""
        self.call_count = 0
        self.last_screenshot = None


class MockRegionAnalyzer:
    """
    Mock region analyzer for testing.

    Example:
        >>> analyzer = MockRegionAnalyzer(
        ...     regions=[MockRegion(0, 0, 100, 100, "dialog")]
        ... )
        >>> regions = analyzer.analyze(screenshot)
        >>> assert len(regions) == 1
        >>> assert regions[0].region_type == "dialog"
    """

    def __init__(
        self,
        regions: list[MockRegion] | None = None,
        should_fail: bool = False,
    ):
        """
        Initialize mock region analyzer.

        Args:
            regions: List of regions to return from analyze()
            should_fail: If True, analyze() will raise an exception
        """
        self.regions = regions or []
        self.should_fail = should_fail
        self.call_count = 0

    def analyze(self, screenshot: np.ndarray) -> list[MockRegion]:
        """
        Mock region analysis method.

        Args:
            screenshot: Input screenshot

        Returns:
            List of mock regions

        Raises:
            RuntimeError: If should_fail is True
        """
        self.call_count += 1

        if self.should_fail:
            raise RuntimeError("Mock analyzer configured to fail")

        return self.regions.copy()

    def reset(self):
        """Reset analyzer state."""
        self.call_count = 0


class MockStateDetector:
    """
    Mock state detector for testing.

    Example:
        >>> detector = MockStateDetector(
        ...     states=[MockState("state1", "Login Screen")]
        ... )
        >>> states = detector.detect_states(screenshot)
        >>> assert len(states) == 1
        >>> assert states[0].name == "Login Screen"
    """

    def __init__(
        self,
        states: list[MockState] | None = None,
        should_fail: bool = False,
    ):
        """
        Initialize mock state detector.

        Args:
            states: List of states to return from detect_states()
            should_fail: If True, detect_states() will raise an exception
        """
        self.states = states or []
        self.should_fail = should_fail
        self.call_count = 0

    def detect_states(self, screenshot: np.ndarray) -> list[MockState]:
        """
        Mock state detection method.

        Args:
            screenshot: Input screenshot

        Returns:
            List of mock states

        Raises:
            RuntimeError: If should_fail is True
        """
        self.call_count += 1

        if self.should_fail:
            raise RuntimeError("Mock state detector configured to fail")

        return self.states.copy()

    def reset(self):
        """Reset detector state."""
        self.call_count = 0


# Pytest Fixtures


@pytest.fixture
def mock_detection_result() -> MockDetectionResult:
    """
    Fixture providing a sample mock detection result.

    Example:
        >>> def test_result_processing(mock_detection_result):
        ...     assert mock_detection_result.element_type == "button"
        ...     assert mock_detection_result.confidence > 0.9
    """
    return MockDetectionResult(
        element_type="button",
        confidence=0.95,
        bbox=(100, 100, 80, 40),
        attributes={"text": "Submit", "enabled": True},
    )


@pytest.fixture
def mock_detection_results() -> list[MockDetectionResult]:
    """
    Fixture providing multiple mock detection results.

    Example:
        >>> def test_multiple_detections(mock_detection_results):
        ...     assert len(mock_detection_results) == 3
        ...     types = {r.element_type for r in mock_detection_results}
        ...     assert "button" in types
    """
    return [
        MockDetectionResult("button", 0.95, (100, 100, 80, 40)),
        MockDetectionResult("input", 0.92, (100, 200, 200, 35)),
        MockDetectionResult("icon", 0.88, (300, 150, 50, 50)),
    ]


@pytest.fixture
def mock_element_detector() -> MockElementDetector:
    """
    Fixture providing a mock element detector with sample results.

    Example:
        >>> def test_detector(mock_element_detector):
        ...     results = mock_element_detector.detect(screenshot)
        ...     assert mock_element_detector.call_count == 1
    """
    return MockElementDetector(
        detection_results=[
            MockDetectionResult("button", 0.95, (100, 100, 80, 40)),
            MockDetectionResult("text", 0.90, (100, 50, 200, 30)),
        ]
    )


@pytest.fixture
def mock_region() -> MockRegion:
    """
    Fixture providing a sample mock region.

    Example:
        >>> def test_region_analysis(mock_region):
        ...     assert mock_region.region_type == "dialog"
        ...     assert mock_region.area == 40000
    """
    return MockRegion(
        x=100,
        y=100,
        width=200,
        height=200,
        region_type="dialog",
        confidence=0.93,
    )


@pytest.fixture
def mock_regions() -> list[MockRegion]:
    """
    Fixture providing multiple mock regions.

    Example:
        >>> def test_multiple_regions(mock_regions):
        ...     assert len(mock_regions) == 3
        ...     types = {r.region_type for r in mock_regions}
        ...     assert "dialog" in types
    """
    return [
        MockRegion(100, 100, 200, 200, "dialog"),
        MockRegion(400, 100, 300, 400, "content"),
        MockRegion(50, 500, 700, 100, "toolbar"),
    ]


@pytest.fixture
def mock_region_analyzer() -> MockRegionAnalyzer:
    """
    Fixture providing a mock region analyzer.

    Example:
        >>> def test_analyzer(mock_region_analyzer):
        ...     regions = mock_region_analyzer.analyze(screenshot)
        ...     assert mock_region_analyzer.call_count == 1
    """
    return MockRegionAnalyzer(
        regions=[
            MockRegion(100, 100, 200, 200, "dialog"),
            MockRegion(400, 100, 300, 400, "content"),
        ]
    )


@pytest.fixture
def mock_state() -> MockState:
    """
    Fixture providing a sample mock state.

    Example:
        >>> def test_state_processing(mock_state):
        ...     assert mock_state.name == "Login Screen"
        ...     assert mock_state.confidence > 0.9
    """
    return MockState(
        state_id="login_state",
        name="Login Screen",
        regions=[
            MockRegion(200, 150, 400, 300, "login_form"),
        ],
        elements=[
            MockDetectionResult("button", 0.95, (350, 380, 100, 40)),
        ],
        confidence=0.94,
    )


@pytest.fixture
def mock_states() -> list[MockState]:
    """
    Fixture providing multiple mock states.

    Example:
        >>> def test_multiple_states(mock_states):
        ...     assert len(mock_states) == 2
        ...     names = {s.name for s in mock_states}
        ...     assert "Login Screen" in names
    """
    return [
        MockState(
            state_id="login_state",
            name="Login Screen",
            confidence=0.94,
        ),
        MockState(
            state_id="main_state",
            name="Main Dashboard",
            confidence=0.91,
        ),
    ]


@pytest.fixture
def mock_state_detector() -> MockStateDetector:
    """
    Fixture providing a mock state detector.

    Example:
        >>> def test_state_detection(mock_state_detector):
        ...     states = mock_state_detector.detect_states(screenshot)
        ...     assert mock_state_detector.call_count == 1
    """
    return MockStateDetector(
        states=[
            MockState("login_state", "Login Screen", confidence=0.94),
        ]
    )


@pytest.fixture
def detection_config() -> dict[str, Any]:
    """
    Fixture providing sample detection configuration.

    Example:
        >>> def test_with_config(detection_config):
        ...     assert detection_config["confidence_threshold"] == 0.8
        ...     assert "element_types" in detection_config
    """
    return {
        "confidence_threshold": 0.8,
        "element_types": ["button", "input", "text", "icon"],
        "max_detections": 100,
        "nms_threshold": 0.5,
        "enable_masking": True,
    }


@pytest.fixture
def region_analysis_config() -> dict[str, Any]:
    """
    Fixture providing sample region analysis configuration.

    Example:
        >>> def test_region_config(region_analysis_config):
        ...     assert region_analysis_config["min_region_size"] == 1000
    """
    return {
        "min_region_size": 1000,
        "max_region_overlap": 0.3,
        "region_types": ["dialog", "content", "toolbar", "sidebar"],
        "merge_similar_regions": True,
    }


@pytest.fixture
def state_detection_config() -> dict[str, Any]:
    """
    Fixture providing sample state detection configuration.

    Example:
        >>> def test_state_config(state_detection_config):
        ...     assert state_detection_config["min_state_confidence"] == 0.7
    """
    return {
        "min_state_confidence": 0.7,
        "max_states_per_screenshot": 3,
        "require_unique_elements": True,
        "enable_state_transitions": True,
        "state_similarity_threshold": 0.85,
    }


@pytest.fixture
def sample_mask() -> np.ndarray:
    """
    Fixture providing a sample binary mask.

    Example:
        >>> def test_mask_processing(sample_mask):
        ...     assert sample_mask.dtype == np.uint8
        ...     assert np.all((sample_mask == 0) | (sample_mask == 255))
    """
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    return mask


@pytest.fixture
def sample_screenshot() -> np.ndarray:
    """
    Fixture providing a sample screenshot for testing.

    Example:
        >>> def test_with_screenshot(sample_screenshot):
        ...     assert sample_screenshot.shape == (600, 800, 3)
        ...     assert sample_screenshot.dtype == np.uint8
    """
    return np.full((600, 800, 3), 240, dtype=np.uint8)
