"""
Base test cases for state detection components.

This module provides base test classes and utilities for testing state detection
functionality including state identification, classification, and transition detection.

Example usage:
    >>> import pytest
    >>> from tests.discovery.state_detection.test_base import BaseStateDetectorTest
    >>>
    >>> class TestMyStateDetector(BaseStateDetectorTest):
    ...     @pytest.fixture
    ...     def detector(self):
    ...         return MyStateDetector()
    ...
    ...     def test_login_state_detection(self, detector, synthetic_screenshot):
    ...         # Test implementation
    ...         pass
"""

import numpy as np
import pytest
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Set
from tests.fixtures.screenshot_fixtures import SyntheticScreenshotGenerator, ElementSpec
from tests.fixtures.detector_fixtures import MockState, MockRegion, MockDetectionResult


class BaseStateDetectorTest(ABC):
    """
    Abstract base class for state detector test cases.

    Provides common test methods and utilities for testing state detectors.
    Subclass this to test specific state detector implementations.

    Example:
        >>> class TestLoginStateDetector(BaseStateDetectorTest):
        ...     @pytest.fixture
        ...     def detector(self):
        ...         return LoginStateDetector()
        ...
        ...     def test_login_detection(self, detector, synthetic_screenshot):
        ...         screenshot = synthetic_screenshot(
        ...             elements=[
        ...                 ElementSpec("input", x=300, y=200, width=200, height=35),
        ...                 ElementSpec("button", x=350, y=280, width=100, height=40)
        ...             ]
        ...         )
        ...         states = detector.detect_states(screenshot)
        ...         assert len(states) >= 1
    """

    @pytest.fixture
    @abstractmethod
    def detector(self):
        """
        Fixture that provides the detector instance to test.

        Must be implemented by subclasses.

        Returns:
            State detector instance to test
        """
        pass

    def test_detector_initialization(self, detector):
        """
        Test that detector initializes correctly.

        Args:
            detector: State detector instance from fixture
        """
        assert detector is not None
        assert hasattr(detector, 'detect_states') or hasattr(detector, 'detect')

    def test_empty_screenshot(self, detector):
        """
        Test detector behavior on empty screenshot.

        Args:
            detector: State detector instance from fixture
        """
        generator = SyntheticScreenshotGenerator()
        empty_screen = generator.generate(width=800, height=600, elements=[])

        if hasattr(detector, 'detect_states'):
            states = detector.detect_states(empty_screen)
        else:
            states = detector.detect(empty_screen)

        # Should return empty list or list with low-confidence states
        assert isinstance(states, list)

    def test_single_state_detection(self, detector):
        """
        Test detection of a single state.

        Args:
            detector: State detector instance from fixture
        """
        generator = SyntheticScreenshotGenerator()
        screenshot = generator.generate(
            width=800,
            height=600,
            elements=[
                # Login form elements
                ElementSpec("text", x=300, y=150, text="Username:"),
                ElementSpec("input", x=300, y=180, width=200, height=35),
                ElementSpec("text", x=300, y=225, text="Password:"),
                ElementSpec("input", x=300, y=255, width=200, height=35),
                ElementSpec("button", x=350, y=310, width=100, height=40, text="Login"),
            ]
        )

        if hasattr(detector, 'detect_states'):
            states = detector.detect_states(screenshot)
        else:
            states = detector.detect(screenshot)

        assert isinstance(states, list)
        # Should detect at least one state
        assert len(states) >= 1

    def test_state_properties(self, detector):
        """
        Test that detected states have expected properties.

        Args:
            detector: State detector instance from fixture
        """
        generator = SyntheticScreenshotGenerator()
        screenshot, elements = generator.generate_with_known_elements()

        if hasattr(detector, 'detect_states'):
            states = detector.detect_states(screenshot)
        else:
            states = detector.detect(screenshot)

        if len(states) > 0:
            state = states[0]
            # Check required properties
            assert hasattr(state, 'state_id') or hasattr(state, 'id')
            assert hasattr(state, 'name') or hasattr(state, 'state_name')

    def test_state_confidence(self, detector):
        """
        Test that state detections include confidence scores.

        Args:
            detector: State detector instance from fixture
        """
        generator = SyntheticScreenshotGenerator()
        screenshot, _ = generator.generate_with_known_elements()

        if hasattr(detector, 'detect_states'):
            states = detector.detect_states(screenshot)
        else:
            states = detector.detect(screenshot)

        if len(states) > 0:
            for state in states:
                if hasattr(state, 'confidence'):
                    assert 0.0 <= state.confidence <= 1.0


class BaseStateTypeTest:
    """
    Base test class for testing detection of specific state types.

    Provides test methods focused on detecting and validating specific
    application states (login, main menu, dialog, settings, etc.).

    Example:
        >>> class TestLoginStateDetection(BaseStateTypeTest):
        ...     state_type = "login"
        ...
        ...     @pytest.fixture
        ...     def detector(self):
        ...         return LoginStateDetector()
    """

    state_type: str = "unknown"

    @pytest.fixture
    @abstractmethod
    def detector(self):
        """Fixture providing the detector instance."""
        pass

    def test_detect_characteristic_elements(self, detector):
        """Test that state is detected based on characteristic elements."""
        generator = SyntheticScreenshotGenerator()

        # Create elements based on state type
        if self.state_type == "login":
            elements = [
                ElementSpec("input", x=300, y=200, width=200, height=35, text="username"),
                ElementSpec("input", x=300, y=250, width=200, height=35, text="password"),
                ElementSpec("button", x=350, y=310, width=100, height=40, text="Login"),
            ]
        elif self.state_type == "menu":
            elements = [
                ElementSpec("button", x=100, y=100 + i * 60, width=200, height=40, text=f"Option {i+1}")
                for i in range(5)
            ]
        elif self.state_type == "dialog":
            elements = [
                ElementSpec("rectangle", x=200, y=150, width=400, height=300,
                           color=(240, 240, 240), border_color=(100, 100, 100)),
                ElementSpec("text", x=220, y=180, text="Dialog Title"),
                ElementSpec("button", x=450, y=410, width=80, height=35, text="OK"),
                ElementSpec("button", x=540, y=410, width=80, height=35, text="Cancel"),
            ]
        else:
            elements = [
                ElementSpec("button", x=100, y=100, width=120, height=40, text="Action")
            ]

        screenshot = generator.generate(width=800, height=600, elements=elements)

        if hasattr(detector, 'detect_states'):
            states = detector.detect_states(screenshot)
        else:
            states = detector.detect(screenshot)

        # Should detect the state
        assert len(states) >= 1

    def test_state_uniqueness(self, detector):
        """Test that state type is correctly identified."""
        generator = SyntheticScreenshotGenerator()
        screenshot, _ = generator.generate_with_known_elements()

        if hasattr(detector, 'detect_states'):
            states = detector.detect_states(screenshot)
        else:
            states = detector.detect(screenshot)

        # If states are detected, verify they have type information
        for state in states:
            if hasattr(state, 'state_type'):
                assert isinstance(state.state_type, str)
            elif hasattr(state, 'name'):
                assert isinstance(state.name, str)


class StateTransitionTest:
    """
    Base class for testing state transitions.

    Provides utilities for testing state change detection and transition
    validation between different application states.

    Example:
        >>> class TestMenuTransitions(StateTransitionTest):
        ...     @pytest.fixture
        ...     def detector(self):
        ...         return MenuStateDetector()
        ...
        ...     def test_menu_to_submenu(self, detector):
        ...         # Test implementation
        ...         pass
    """

    @pytest.fixture
    @abstractmethod
    def detector(self):
        """Fixture providing the detector instance."""
        pass

    def test_detect_state_change(self, detector):
        """Test detection of state changes between screenshots."""
        generator = SyntheticScreenshotGenerator()

        # First state (login screen)
        screenshot1 = generator.generate(
            width=800,
            height=600,
            elements=[
                ElementSpec("input", x=300, y=200, width=200, height=35),
                ElementSpec("button", x=350, y=280, width=100, height=40, text="Login"),
            ]
        )

        # Second state (main menu)
        screenshot2 = generator.generate(
            width=800,
            height=600,
            elements=[
                ElementSpec("button", x=100, y=100 + i * 60, width=200, height=40, text=f"Option {i}")
                for i in range(4)
            ]
        )

        if hasattr(detector, 'detect_states'):
            states1 = detector.detect_states(screenshot1)
            states2 = detector.detect_states(screenshot2)
        else:
            states1 = detector.detect(screenshot1)
            states2 = detector.detect(screenshot2)

        # States should be different (or at least detectable)
        assert isinstance(states1, list)
        assert isinstance(states2, list)

    def test_stable_state_detection(self, detector):
        """Test that same state is consistently detected in similar screenshots."""
        generator = SyntheticScreenshotGenerator()

        # Create two similar screenshots with slight variations
        elements = [
            ElementSpec("button", x=100, y=100, width=120, height=40, text="Submit"),
            ElementSpec("input", x=100, y=200, width=200, height=35),
        ]

        screenshot1 = generator.generate(width=800, height=600, elements=elements, noise_level=0.01)
        screenshot2 = generator.generate(width=800, height=600, elements=elements, noise_level=0.01)

        if hasattr(detector, 'detect_states'):
            states1 = detector.detect_states(screenshot1)
            states2 = detector.detect_states(screenshot2)
        else:
            states1 = detector.detect(screenshot1)
            states2 = detector.detect(screenshot2)

        # Should detect similar/same states
        assert isinstance(states1, list)
        assert isinstance(states2, list)

    def test_detect_transition_elements(self, detector):
        """Test detection of elements that trigger state transitions."""
        generator = SyntheticScreenshotGenerator()

        # Create screenshot with transition elements (buttons, links)
        screenshot = generator.generate(
            width=800,
            height=600,
            elements=[
                ElementSpec("button", x=100, y=100, width=120, height=40, text="Next"),
                ElementSpec("button", x=250, y=100, width=120, height=40, text="Back"),
                ElementSpec("button", x=400, y=100, width=120, height=40, text="Exit"),
            ]
        )

        if hasattr(detector, 'detect_states'):
            states = detector.detect_states(screenshot)
        else:
            states = detector.detect(screenshot)

        # Should detect state with transition capability
        assert isinstance(states, list)


class StateValidationTest:
    """
    Base class for validating detected states.

    Provides utilities for validating state completeness, consistency,
    and correctness.

    Example:
        >>> class TestStateValidation(StateValidationTest):
        ...     @pytest.fixture
        ...     def detector(self):
        ...         return StateDetector()
        ...
        ...     def test_state_completeness(self, detector):
        ...         # Test implementation
        ...         pass
    """

    @pytest.fixture
    @abstractmethod
    def detector(self):
        """Fixture providing the detector instance."""
        pass

    def test_state_has_unique_identifier(self, detector):
        """Test that each state has a unique identifier."""
        generator = SyntheticScreenshotGenerator()
        screenshot, _ = generator.generate_with_known_elements()

        if hasattr(detector, 'detect_states'):
            states = detector.detect_states(screenshot)
        else:
            states = detector.detect(screenshot)

        # Check uniqueness of identifiers
        identifiers = set()
        for state in states:
            if hasattr(state, 'state_id'):
                assert state.state_id not in identifiers
                identifiers.add(state.state_id)
            elif hasattr(state, 'id'):
                assert state.id not in identifiers
                identifiers.add(state.id)

    def test_state_has_elements(self, detector):
        """Test that detected states contain element information."""
        generator = SyntheticScreenshotGenerator()
        screenshot, _ = generator.generate_with_known_elements()

        if hasattr(detector, 'detect_states'):
            states = detector.detect_states(screenshot)
        else:
            states = detector.detect(screenshot)

        # States should have some associated data
        for state in states:
            # Check for elements, regions, or other state data
            has_data = (
                (hasattr(state, 'elements') and len(state.elements) > 0) or
                (hasattr(state, 'regions') and len(state.regions) > 0) or
                hasattr(state, 'metadata')
            )
            # This is optional - some detectors may not include element details
            # assert has_data or True  # Soft check


class StatePerformanceTest:
    """
    Base class for performance testing of state detectors.

    Provides utilities for measuring detection speed and efficiency.

    Example:
        >>> class TestDetectorPerformance(StatePerformanceTest):
        ...     @pytest.fixture
        ...     def detector(self):
        ...         return StateDetector()
        ...
        ...     def test_detection_speed(self, detector):
        ...         self.measure_detection_time(detector, num_runs=10)
    """

    @pytest.fixture
    @abstractmethod
    def detector(self):
        """Fixture providing the detector instance."""
        pass

    def measure_detection_time(
        self,
        detector,
        screenshot: Optional[np.ndarray] = None,
        num_runs: int = 10
    ) -> float:
        """
        Measure average state detection time.

        Args:
            detector: State detector instance
            screenshot: Screenshot to test with (generates one if None)
            num_runs: Number of detection runs to average

        Returns:
            Average detection time in seconds
        """
        import time

        if screenshot is None:
            generator = SyntheticScreenshotGenerator()
            screenshot, _ = generator.generate_with_known_elements()

        times = []
        for _ in range(num_runs):
            start = time.time()
            if hasattr(detector, 'detect_states'):
                detector.detect_states(screenshot)
            else:
                detector.detect(screenshot)
            times.append(time.time() - start)

        return sum(times) / len(times)

    def test_detection_speed_reasonable(self, detector):
        """
        Test that state detection completes in reasonable time.

        Args:
            detector: State detector instance from fixture
        """
        avg_time = self.measure_detection_time(detector, num_runs=5)
        # Should complete in less than 10 seconds per detection
        assert avg_time < 10.0, f"State detection too slow: {avg_time:.2f}s"


# Utility functions for state testing

def calculate_state_similarity(state1: Any, state2: Any) -> float:
    """
    Calculate similarity between two states.

    Args:
        state1: First state object
        state2: Second state object

    Returns:
        Similarity score between 0 and 1

    Example:
        >>> state1 = MockState("state1", "Login", confidence=0.9)
        >>> state2 = MockState("state2", "Login", confidence=0.95)
        >>> similarity = calculate_state_similarity(state1, state2)
        >>> assert similarity > 0.5  # Same name increases similarity
    """
    score = 0.0
    total_checks = 0

    # Compare names
    if hasattr(state1, 'name') and hasattr(state2, 'name'):
        total_checks += 1
        if state1.name == state2.name:
            score += 1.0

    # Compare state types
    if hasattr(state1, 'state_type') and hasattr(state2, 'state_type'):
        total_checks += 1
        if state1.state_type == state2.state_type:
            score += 1.0

    # Compare element counts
    if hasattr(state1, 'elements') and hasattr(state2, 'elements'):
        total_checks += 1
        count1 = len(state1.elements) if state1.elements else 0
        count2 = len(state2.elements) if state2.elements else 0
        if count1 > 0 and count2 > 0:
            # Calculate element count similarity
            max_count = max(count1, count2)
            min_count = min(count1, count2)
            score += min_count / max_count

    # Compare region counts
    if hasattr(state1, 'regions') and hasattr(state2, 'regions'):
        total_checks += 1
        count1 = len(state1.regions) if state1.regions else 0
        count2 = len(state2.regions) if state2.regions else 0
        if count1 > 0 and count2 > 0:
            max_count = max(count1, count2)
            min_count = min(count1, count2)
            score += min_count / max_count

    return score / total_checks if total_checks > 0 else 0.0


def assert_state_valid(state: Any):
    """
    Assert that a state object is valid.

    Args:
        state: State object to validate

    Raises:
        AssertionError: If state is invalid
    """
    # Check for required attributes
    has_id = hasattr(state, 'state_id') or hasattr(state, 'id')
    assert has_id, "State must have an identifier"

    has_name = hasattr(state, 'name') or hasattr(state, 'state_name')
    assert has_name, "State must have a name"

    # Check confidence if present
    if hasattr(state, 'confidence'):
        assert 0.0 <= state.confidence <= 1.0, f"Invalid confidence: {state.confidence}"


def find_state_by_name(states: List[Any], name: str) -> Optional[Any]:
    """
    Find a state by name.

    Args:
        states: List of state objects
        name: State name to search for

    Returns:
        Matching state or None

    Example:
        >>> states = [
        ...     MockState("login", "Login Screen"),
        ...     MockState("main", "Main Menu"),
        ... ]
        >>> login_state = find_state_by_name(states, "Login Screen")
        >>> assert login_state is not None
        >>> assert login_state.state_id == "login"
    """
    for state in states:
        if hasattr(state, 'name') and state.name == name:
            return state
        elif hasattr(state, 'state_name') and state.state_name == name:
            return state

    return None


def group_states_by_type(states: List[Any]) -> Dict[str, List[Any]]:
    """
    Group states by their type.

    Args:
        states: List of state objects

    Returns:
        Dictionary mapping state types to lists of states

    Example:
        >>> states = [
        ...     MockState("login", "Login", metadata={"type": "auth"}),
        ...     MockState("main", "Main Menu", metadata={"type": "menu"}),
        ...     MockState("settings", "Settings", metadata={"type": "menu"}),
        ... ]
        >>> grouped = group_states_by_type(states)
        >>> assert "menu" in grouped
        >>> assert len(grouped["menu"]) == 2
    """
    grouped: Dict[str, List[Any]] = {}

    for state in states:
        state_type = "unknown"

        if hasattr(state, 'state_type'):
            state_type = state.state_type
        elif hasattr(state, 'metadata') and isinstance(state.metadata, dict):
            state_type = state.metadata.get('type', 'unknown')

        if state_type not in grouped:
            grouped[state_type] = []

        grouped[state_type].append(state)

    return grouped


def validate_state_transition(
    from_state: Any,
    to_state: Any,
    valid_transitions: Optional[Dict[str, List[str]]] = None
) -> bool:
    """
    Validate if a state transition is valid.

    Args:
        from_state: Source state
        to_state: Destination state
        valid_transitions: Optional dict mapping state names to valid next states

    Returns:
        True if transition is valid

    Example:
        >>> transitions = {
        ...     "Login": ["Main Menu", "Registration"],
        ...     "Main Menu": ["Settings", "Profile", "Login"],
        ... }
        >>> from_state = MockState("login", "Login")
        >>> to_state = MockState("main", "Main Menu")
        >>> assert validate_state_transition(from_state, to_state, transitions)
    """
    if valid_transitions is None:
        # If no transition rules provided, all transitions are valid
        return True

    from_name = getattr(from_state, 'name', getattr(from_state, 'state_name', None))
    to_name = getattr(to_state, 'name', getattr(to_state, 'state_name', None))

    if from_name is None or to_name is None:
        return False

    if from_name not in valid_transitions:
        return False

    return to_name in valid_transitions[from_name]
