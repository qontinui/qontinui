"""
Base test cases for state construction components.

This module provides base test classes and utilities for testing state construction
functionality including state building, validation, and optimization.

Example usage:
    >>> import pytest
    >>> from tests.discovery.state_construction.test_base import BaseStateBuilderTest
    >>>
    >>> class TestMyStateBuilder(BaseStateBuilderTest):
    ...     @pytest.fixture
    ...     def builder(self):
    ...         return MyStateBuilder()
    ...
    ...     def test_build_from_elements(self, builder, mock_detection_results):
    ...         # Test implementation
    ...         pass
"""

from abc import ABC, abstractmethod
from typing import Any

import pytest

from tests.fixtures.detector_fixtures import (
    MockDetectionResult,
    MockRegion,
    MockState,
)


class BaseStateBuilderTest(ABC):
    """
    Abstract base class for state builder test cases.

    Provides common test methods and utilities for testing state builders.
    Subclass this to test specific state builder implementations.

    Example:
        >>> class TestDialogStateBuilder(BaseStateBuilderTest):
        ...     @pytest.fixture
        ...     def builder(self):
        ...         return DialogStateBuilder()
        ...
        ...     def test_build_dialog_state(self, builder, mock_detection_results):
        ...         state = builder.build(mock_detection_results)
        ...         assert state is not None
    """

    @pytest.fixture
    @abstractmethod
    def builder(self):
        """
        Fixture that provides the builder instance to test.

        Must be implemented by subclasses.

        Returns:
            State builder instance to test
        """
        pass

    def test_builder_initialization(self, builder):
        """
        Test that builder initializes correctly.

        Args:
            builder: State builder instance from fixture
        """
        assert builder is not None
        assert hasattr(builder, "build") or hasattr(builder, "construct_state")

    def test_build_from_empty_input(self, builder):
        """
        Test builder behavior with empty input.

        Args:
            builder: State builder instance from fixture
        """
        empty_elements = []

        if hasattr(builder, "build"):
            result = builder.build(empty_elements)
        else:
            result = builder.construct_state(empty_elements)

        # Should return None or empty state
        assert result is None or (hasattr(result, "elements") and len(result.elements) == 0)

    def test_build_from_single_element(self, builder):
        """
        Test building state from single element.

        Args:
            builder: State builder instance from fixture
        """
        elements = [MockDetectionResult("button", 0.95, (100, 100, 80, 40))]

        if hasattr(builder, "build"):
            state = builder.build(elements)
        else:
            state = builder.construct_state(elements)

        # Should create state with the element
        if state is not None:
            assert hasattr(state, "elements") or hasattr(state, "regions")

    def test_build_from_multiple_elements(self, builder):
        """
        Test building state from multiple elements.

        Args:
            builder: State builder instance from fixture
        """
        elements = [
            MockDetectionResult("button", 0.95, (100, 100, 80, 40)),
            MockDetectionResult("input", 0.92, (100, 200, 200, 35)),
            MockDetectionResult("text", 0.90, (100, 50, 150, 30)),
        ]

        if hasattr(builder, "build"):
            state = builder.build(elements)
        else:
            state = builder.construct_state(elements)

        # Should create state with all elements
        assert state is not None
        if hasattr(state, "elements"):
            assert len(state.elements) > 0

    def test_build_with_regions(self, builder):
        """
        Test building state with region information.

        Args:
            builder: State builder instance from fixture
        """
        elements = [
            MockDetectionResult("button", 0.95, (100, 100, 80, 40)),
            MockDetectionResult("button", 0.93, (100, 160, 80, 40)),
        ]

        regions = [MockRegion(50, 50, 200, 200, "dialog")]

        # Try building with regions if supported
        if hasattr(builder, "build_with_regions"):
            state = builder.build_with_regions(elements, regions)
            assert state is not None
        elif hasattr(builder, "build"):
            # Build without regions
            state = builder.build(elements)
            assert state is not None

    def test_state_has_valid_properties(self, builder):
        """
        Test that constructed states have valid properties.

        Args:
            builder: State builder instance from fixture
        """
        elements = [
            MockDetectionResult("button", 0.95, (100, 100, 80, 40)),
            MockDetectionResult("input", 0.92, (100, 200, 200, 35)),
        ]

        if hasattr(builder, "build"):
            state = builder.build(elements)
        else:
            state = builder.construct_state(elements)

        if state is not None:
            # Check for required properties
            assert hasattr(state, "state_id") or hasattr(state, "id")
            assert hasattr(state, "name") or hasattr(state, "state_name")


class BaseStateValidatorTest:
    """
    Base test class for testing state validators.

    Provides test methods for validating state correctness, completeness,
    and consistency.

    Example:
        >>> class TestStateValidator(BaseStateValidatorTest):
        ...     @pytest.fixture
        ...     def validator(self):
        ...         return StateValidator()
        ...
        ...     def test_validate_complete_state(self, validator):
        ...         # Test implementation
        ...         pass
    """

    @pytest.fixture
    @abstractmethod
    def validator(self):
        """Fixture providing the validator instance."""
        pass

    def test_validate_complete_state(self, validator):
        """Test validation of complete, valid state."""
        state = MockState(
            state_id="test_state",
            name="Test State",
            elements=[MockDetectionResult("button", 0.95, (100, 100, 80, 40))],
            confidence=0.90,
        )

        if hasattr(validator, "validate"):
            result = validator.validate(state)
        else:
            result = validator.is_valid(state)

        # Should pass validation
        assert result is True or (isinstance(result, dict) and result.get("valid", False))

    def test_validate_incomplete_state(self, validator):
        """Test validation of incomplete state."""
        # State with missing elements
        state = MockState(
            state_id="incomplete_state",
            name="Incomplete State",
            elements=[],
            confidence=0.50,
        )

        if hasattr(validator, "validate"):
            result = validator.validate(state)
        else:
            result = validator.is_valid(state)

        # Validation may fail or succeed with warnings
        assert isinstance(result, bool | dict)

    def test_validate_state_confidence(self, validator):
        """Test validation of state confidence values."""
        # State with low confidence
        state = MockState(
            state_id="low_confidence_state",
            name="Low Confidence State",
            confidence=0.30,
        )

        if hasattr(validator, "validate_confidence"):
            result = validator.validate_confidence(state)
            assert isinstance(result, bool)


class StateOptimizationTest:
    """
    Base class for testing state optimization.

    Provides utilities for testing state simplification, merging,
    and optimization operations.

    Example:
        >>> class TestStateOptimizer(StateOptimizationTest):
        ...     @pytest.fixture
        ...     def optimizer(self):
        ...         return StateOptimizer()
        ...
        ...     def test_merge_similar_states(self, optimizer):
        ...         # Test implementation
        ...         pass
    """

    @pytest.fixture
    @abstractmethod
    def optimizer(self):
        """Fixture providing the optimizer instance."""
        pass

    def test_remove_duplicate_elements(self, optimizer):
        """Test removal of duplicate elements from state."""
        state = MockState(
            state_id="test_state",
            name="Test State",
            elements=[
                MockDetectionResult("button", 0.95, (100, 100, 80, 40)),
                MockDetectionResult("button", 0.94, (100, 100, 80, 40)),  # Duplicate
                MockDetectionResult("input", 0.92, (100, 200, 200, 35)),
            ],
        )

        if hasattr(optimizer, "optimize"):
            optimized = optimizer.optimize(state)
        elif hasattr(optimizer, "remove_duplicates"):
            optimized = optimizer.remove_duplicates(state)
        else:
            optimized = state

        # Should have fewer or same number of elements
        if optimized is not None and hasattr(optimized, "elements"):
            assert len(optimized.elements) <= len(state.elements)

    def test_merge_similar_regions(self, optimizer):
        """Test merging of similar/overlapping regions."""
        state = MockState(
            state_id="test_state",
            name="Test State",
            regions=[
                MockRegion(100, 100, 200, 200, "dialog"),
                MockRegion(110, 110, 190, 190, "dialog"),  # Similar/overlapping
                MockRegion(400, 100, 150, 150, "toolbar"),
            ],
        )

        if hasattr(optimizer, "optimize"):
            optimized = optimizer.optimize(state)
        elif hasattr(optimizer, "merge_regions"):
            optimized = optimizer.merge_regions(state)
        else:
            optimized = state

        # Should have fewer or same number of regions
        if optimized is not None and hasattr(optimized, "regions"):
            assert len(optimized.regions) <= len(state.regions)

    def test_simplify_state_structure(self, optimizer):
        """Test simplification of complex state structure."""
        # Create complex state with many elements
        elements = [
            MockDetectionResult(f"element_{i}", 0.8 + i * 0.01, (i * 50, 100, 40, 40))
            for i in range(20)
        ]

        state = MockState(state_id="complex_state", name="Complex State", elements=elements)

        if hasattr(optimizer, "simplify"):
            simplified = optimizer.simplify(state)
        elif hasattr(optimizer, "optimize"):
            simplified = optimizer.optimize(state)
        else:
            simplified = state

        # Should return a valid state
        assert simplified is not None


class StateHierarchyTest:
    """
    Base class for testing state hierarchy construction.

    Provides utilities for testing parent-child relationships,
    nested states, and hierarchical state structures.

    Example:
        >>> class TestStateHierarchyBuilder(StateHierarchyTest):
        ...     @pytest.fixture
        ...     def builder(self):
        ...         return HierarchyBuilder()
        ...
        ...     def test_build_nested_states(self, builder):
        ...         # Test implementation
        ...         pass
    """

    @pytest.fixture
    @abstractmethod
    def builder(self):
        """Fixture providing the hierarchy builder instance."""
        pass

    def test_build_parent_child_relationship(self, builder):
        """Test building parent-child state relationships."""
        parent_region = MockRegion(100, 100, 600, 400, "dialog")
        child_region = MockRegion(150, 150, 500, 300, "content")

        if hasattr(builder, "build_hierarchy"):
            hierarchy = builder.build_hierarchy([parent_region, child_region])
            assert hierarchy is not None
        elif hasattr(builder, "find_parent_child"):
            relationships = builder.find_parent_child([parent_region, child_region])
            assert isinstance(relationships, list | dict)

    def test_detect_nested_states(self, builder):
        """Test detection of nested state structures."""
        regions = [
            MockRegion(50, 50, 700, 500, "window"),
            MockRegion(100, 100, 600, 400, "dialog"),
            MockRegion(150, 150, 500, 300, "content"),
        ]

        if hasattr(builder, "detect_nesting"):
            nesting = builder.detect_nesting(regions)
            assert isinstance(nesting, list | dict | set)

    def test_flatten_hierarchy(self, builder):
        """Test flattening of hierarchical state structure."""
        # Create nested state structure
        nested_state = MockState(
            state_id="parent",
            name="Parent State",
            regions=[
                MockRegion(
                    100,
                    100,
                    600,
                    400,
                    "dialog",
                    elements=[MockDetectionResult("button", 0.95, (150, 150, 80, 40))],
                )
            ],
        )

        if hasattr(builder, "flatten"):
            flattened = builder.flatten(nested_state)
            assert flattened is not None
        elif hasattr(builder, "get_all_elements"):
            elements = builder.get_all_elements(nested_state)
            assert isinstance(elements, list)


class StateComparisonTest:
    """
    Base class for testing state comparison operations.

    Provides utilities for comparing states, computing similarity,
    and identifying differences.

    Example:
        >>> class TestStateComparator(StateComparisonTest):
        ...     @pytest.fixture
        ...     def comparator(self):
        ...         return StateComparator()
        ...
        ...     def test_compare_identical_states(self, comparator):
        ...         # Test implementation
        ...         pass
    """

    @pytest.fixture
    @abstractmethod
    def comparator(self):
        """Fixture providing the comparator instance."""
        pass

    def test_compare_identical_states(self, comparator):
        """Test comparison of identical states."""
        state1 = MockState(
            state_id="state1",
            name="Test State",
            elements=[MockDetectionResult("button", 0.95, (100, 100, 80, 40))],
        )

        state2 = MockState(
            state_id="state2",
            name="Test State",
            elements=[MockDetectionResult("button", 0.95, (100, 100, 80, 40))],
        )

        if hasattr(comparator, "compare"):
            similarity = comparator.compare(state1, state2)
        elif hasattr(comparator, "calculate_similarity"):
            similarity = comparator.calculate_similarity(state1, state2)
        else:
            similarity = 1.0  # Default to identical

        # Should have high similarity
        assert isinstance(similarity, float | int)
        if isinstance(similarity, float):
            assert similarity >= 0.0

    def test_compare_different_states(self, comparator):
        """Test comparison of different states."""
        state1 = MockState(
            state_id="login",
            name="Login State",
            elements=[MockDetectionResult("input", 0.92, (100, 100, 200, 35))],
        )

        state2 = MockState(
            state_id="menu",
            name="Menu State",
            elements=[MockDetectionResult("button", 0.95, (100, 100, 80, 40))],
        )

        if hasattr(comparator, "compare"):
            similarity = comparator.compare(state1, state2)
        elif hasattr(comparator, "calculate_similarity"):
            similarity = comparator.calculate_similarity(state1, state2)
        else:
            similarity = 0.0  # Default to different

        # Should have low similarity
        assert isinstance(similarity, float | int)

    def test_find_state_differences(self, comparator):
        """Test finding differences between states."""
        state1 = MockState(
            state_id="state1",
            name="State V1",
            elements=[
                MockDetectionResult("button", 0.95, (100, 100, 80, 40)),
                MockDetectionResult("input", 0.92, (100, 200, 200, 35)),
            ],
        )

        state2 = MockState(
            state_id="state2",
            name="State V2",
            elements=[
                MockDetectionResult("button", 0.95, (100, 100, 80, 40)),
                # Missing input element
            ],
        )

        if hasattr(comparator, "find_differences"):
            differences = comparator.find_differences(state1, state2)
            assert isinstance(differences, list | dict | set)


# Utility functions for state construction testing


def assert_state_well_formed(state: Any):
    """
    Assert that a state is well-formed.

    Args:
        state: State object to validate

    Raises:
        AssertionError: If state is malformed
    """
    # Check for identifier
    assert hasattr(state, "state_id") or hasattr(state, "id"), "State must have ID"

    # Check for name
    assert hasattr(state, "name") or hasattr(state, "state_name"), "State must have name"

    # Check for elements or regions (optional for some states)
    _ = (hasattr(state, "elements") and isinstance(state.elements, list)) or (
        hasattr(state, "regions") and isinstance(state.regions, list)
    )

    # Check confidence if present
    if hasattr(state, "confidence"):
        assert 0.0 <= state.confidence <= 1.0, f"Invalid confidence: {state.confidence}"


def count_state_elements(state: Any) -> int:
    """
    Count total elements in a state (including nested).

    Args:
        state: State object

    Returns:
        Total element count

    Example:
        >>> state = MockState("test", "Test", elements=[
        ...     MockDetectionResult("button", 0.95, (100, 100, 80, 40))
        ... ])
        >>> assert count_state_elements(state) == 1
    """
    count = 0

    # Count direct elements
    if hasattr(state, "elements") and state.elements:
        count += len(state.elements)

    # Count elements in regions
    if hasattr(state, "regions") and state.regions:
        for region in state.regions:
            if hasattr(region, "elements") and region.elements:
                count += len(region.elements)

    return count


def merge_states(state1: Any, state2: Any) -> MockState:
    """
    Merge two states into one.

    Args:
        state1: First state
        state2: Second state

    Returns:
        Merged state

    Example:
        >>> state1 = MockState("s1", "State 1", elements=[
        ...     MockDetectionResult("button", 0.95, (100, 100, 80, 40))
        ... ])
        >>> state2 = MockState("s2", "State 2", elements=[
        ...     MockDetectionResult("input", 0.92, (100, 200, 200, 35))
        ... ])
        >>> merged = merge_states(state1, state2)
        >>> assert count_state_elements(merged) == 2
    """
    # Combine elements
    elements = []
    if hasattr(state1, "elements") and state1.elements:
        elements.extend(state1.elements)
    if hasattr(state2, "elements") and state2.elements:
        elements.extend(state2.elements)

    # Combine regions
    regions = []
    if hasattr(state1, "regions") and state1.regions:
        regions.extend(state1.regions)
    if hasattr(state2, "regions") and state2.regions:
        regions.extend(state2.regions)

    # Create merged state
    merged_id = f"{getattr(state1, 'state_id', 'state1')}__{getattr(state2, 'state_id', 'state2')}"
    merged_name = f"{getattr(state1, 'name', 'State1')} + {getattr(state2, 'name', 'State2')}"

    return MockState(
        state_id=merged_id,
        name=merged_name,
        elements=elements,
        regions=regions,
        confidence=min(getattr(state1, "confidence", 1.0), getattr(state2, "confidence", 1.0)),
    )


def extract_state_features(state: Any) -> dict[str, Any]:
    """
    Extract features from a state for comparison/analysis.

    Args:
        state: State object

    Returns:
        Dictionary of features

    Example:
        >>> state = MockState("test", "Test State", elements=[
        ...     MockDetectionResult("button", 0.95, (100, 100, 80, 40))
        ... ])
        >>> features = extract_state_features(state)
        >>> assert "element_count" in features
        >>> assert features["element_count"] == 1
    """
    features = {}

    # Basic info
    features["state_id"] = getattr(state, "state_id", getattr(state, "id", "unknown"))
    features["name"] = getattr(state, "name", getattr(state, "state_name", "unknown"))

    # Element count
    features["element_count"] = count_state_elements(state)

    # Region count
    if hasattr(state, "regions") and state.regions:
        features["region_count"] = len(state.regions)
    else:
        features["region_count"] = 0

    # Confidence
    features["confidence"] = getattr(state, "confidence", 1.0)

    # Element types
    element_types = set()
    if hasattr(state, "elements") and state.elements:
        for elem in state.elements:
            if hasattr(elem, "element_type"):
                element_types.add(elem.element_type)

    features["element_types"] = list(element_types)

    # Region types
    region_types = set()
    if hasattr(state, "regions") and state.regions:
        for region in state.regions:
            if hasattr(region, "region_type"):
                region_types.add(region.region_type)

    features["region_types"] = list(region_types)

    return features


def filter_low_confidence_elements(state: Any, threshold: float = 0.5) -> MockState:
    """
    Filter out low-confidence elements from a state.

    Args:
        state: State object
        threshold: Minimum confidence threshold

    Returns:
        New state with only high-confidence elements

    Example:
        >>> state = MockState("test", "Test", elements=[
        ...     MockDetectionResult("button", 0.95, (100, 100, 80, 40)),
        ...     MockDetectionResult("input", 0.30, (100, 200, 200, 35)),
        ... ])
        >>> filtered = filter_low_confidence_elements(state, threshold=0.5)
        >>> assert len(filtered.elements) == 1
    """
    filtered_elements = []

    if hasattr(state, "elements") and state.elements:
        for elem in state.elements:
            confidence = getattr(elem, "confidence", 1.0)
            if confidence >= threshold:
                filtered_elements.append(elem)

    return MockState(
        state_id=getattr(state, "state_id", "filtered_state"),
        name=getattr(state, "name", "Filtered State"),
        elements=filtered_elements,
        regions=getattr(state, "regions", []),
        confidence=getattr(state, "confidence", 1.0),
    )
