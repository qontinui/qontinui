"""Tests for state manager."""

from unittest.mock import Mock

import numpy as np
import pytest

from qontinui.state_management import Element, QontinuiStateManager, State, Transition
from qontinui.state_management.models import ElementType, TransitionType


class TestQontinuiStateManager:
    """Test QontinuiStateManager class."""

    @pytest.fixture
    def manager(self):
        """Create QontinuiStateManager instance."""
        return QontinuiStateManager(use_hierarchical=False)

    @pytest.fixture
    def sample_state(self):
        """Create a sample state."""
        elements = [
            Element(
                id="btn_login",
                bbox=(100, 200, 150, 50),
                description="Login button",
                element_type=ElementType.BUTTON,
            ),
            Element(
                id="input_username",
                bbox=(100, 100, 200, 40),
                description="Username input",
                element_type=ElementType.INPUT,
            ),
        ]

        return State(name="login", elements=elements, min_elements=2)

    @pytest.fixture
    def sample_transition(self):
        """Create a sample transition."""
        return Transition(
            from_state="login",
            to_state="home",
            action_type=TransitionType.CLICK,
            trigger_element="btn_login",
            probability=0.95,
        )

    def test_init(self):
        """Test manager initialization."""
        manager = QontinuiStateManager(use_hierarchical=True)
        assert manager.use_hierarchical
        assert len(manager.active_states) == 0
        assert manager.activation_threshold == 0.75
        assert manager.deactivation_threshold == 0.3
        assert manager.evidence_decay == 0.95

    def test_add_state(self, manager, sample_state):
        """Test adding a state."""
        manager.add_state(sample_state)

        assert sample_state.name in manager.state_graph.states
        assert manager.state_graph.get_state(sample_state.name) == sample_state

    def test_add_transition(self, manager, sample_state, sample_transition):
        """Test adding a transition."""
        # Add states first
        manager.add_state(sample_state)
        home_state = State(name="home", elements=[], min_elements=0)
        manager.add_state(home_state)

        # Add transition
        manager.add_transition(sample_transition)

        # Check transition was added to state graph
        state_in_graph = manager.state_graph.get_state("login")
        assert len(state_in_graph.transitions) == 1
        assert state_in_graph.transitions[0].to_state == "home"

    def test_activate_state(self, manager, sample_state):
        """Test state activation."""
        manager.add_state(sample_state)

        # Low evidence - should not activate
        manager.activate_state("login", 0.5)
        assert "login" not in manager.active_states

        # High evidence - should activate
        manager.activate_state("login", 0.9)
        assert "login" in manager.active_states

        # Check activation history (state_name, evidence_score, timestamp)
        assert len(manager.activation_history) == 1
        assert manager.activation_history[0][0] == "login"
        assert manager.activation_history[0][1] == 0.9
        # Third element is datetime - just check it exists
        assert len(manager.activation_history[0]) == 3

    def test_deactivate_state(self, manager, sample_state):
        """Test state deactivation."""
        manager.add_state(sample_state)
        manager.activate_state("login", 0.9)

        assert "login" in manager.active_states

        manager.deactivate_state("login")
        assert "login" not in manager.active_states
        assert manager.state_evidence.get("login", 0) == 0.0

    def test_update_evidence(self, manager, sample_state):
        """Test evidence update with current elements."""
        manager.add_state(sample_state)

        # Create matching elements
        current_elements = [
            Element(
                id="detected_btn",
                bbox=(105, 205, 145, 45),  # Slightly offset but overlapping
                description="Detected button",
            ),
            Element(
                id="detected_input",
                bbox=(100, 100, 200, 40),  # Exact match
                description="Detected input",
            ),
        ]

        manager.update_evidence(current_elements)

        # Evidence should be calculated based on matching elements
        assert "login" in manager.state_evidence

    def test_calculate_element_similarity_with_embeddings(self, manager):
        """Test element similarity calculation with embeddings."""
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([1.0, 0.0, 0.0])  # Same embedding
        embedding3 = np.array([0.0, 1.0, 0.0])  # Different embedding

        elem1 = Element(id="e1", bbox=(0, 0, 10, 10), embedding=embedding1)
        elem2 = Element(id="e2", bbox=(0, 0, 10, 10), embedding=embedding2)
        elem3 = Element(id="e3", bbox=(0, 0, 10, 10), embedding=embedding3)

        # Same embeddings should have high similarity
        similarity = manager._calculate_element_similarity(elem1, elem2)
        assert similarity > 0.99

        # Different embeddings should have low similarity
        similarity = manager._calculate_element_similarity(elem1, elem3)
        assert similarity < 0.1

    def test_calculate_element_similarity_with_bbox(self, manager):
        """Test element similarity calculation with bounding boxes."""
        elem1 = Element(id="e1", bbox=(0, 0, 100, 100))
        elem2 = Element(id="e2", bbox=(50, 50, 100, 100))  # Overlapping
        elem3 = Element(id="e3", bbox=(200, 200, 100, 100))  # Not overlapping

        # Overlapping elements should have positive similarity
        similarity = manager._calculate_element_similarity(elem1, elem2)
        assert similarity > 0

        # Non-overlapping elements should have zero similarity
        similarity = manager._calculate_element_similarity(elem1, elem3)
        assert similarity == 0.0

    def test_get_current_states(self, manager, sample_state):
        """Test getting current active states."""
        manager.add_state(sample_state)
        manager.activate_state("login", 0.9)

        current_states = manager.get_current_states()
        assert "login" in current_states
        assert isinstance(current_states, set)

    def test_get_possible_transitions(self, manager, sample_state, sample_transition):
        """Test getting possible transitions from current states."""
        manager.add_state(sample_state)
        home_state = State(name="home", elements=[], min_elements=0)
        manager.add_state(home_state)

        # Add transition to state graph
        manager.state_graph.add_transition(sample_transition)

        # Activate login state
        manager.activate_state("login", 0.9)

        transitions = manager.get_possible_transitions()
        assert len(transitions) == 1
        assert transitions[0].to_state == "home"

    def test_register_callbacks(self, manager, sample_state):
        """Test callback registration and execution."""
        manager.add_state(sample_state)

        # Create mock callbacks
        enter_callback = Mock()
        exit_callback = Mock()

        manager.register_enter_callback("login", enter_callback)
        manager.register_exit_callback("login", exit_callback)

        # Activate state - should trigger enter callback
        manager.activate_state("login", 0.9)
        enter_callback.assert_called_once_with("login")

        # Deactivate state - should trigger exit callback
        manager.deactivate_state("login")
        exit_callback.assert_called_once_with("login")

    def test_reset(self, manager, sample_state):
        """Test manager reset."""
        manager.add_state(sample_state)
        manager.activate_state("login", 0.9)

        assert len(manager.active_states) > 0
        assert len(manager.state_evidence) > 0
        assert len(manager.activation_history) > 0

        manager.reset()

        assert len(manager.active_states) == 0
        assert len(manager.state_evidence) == 0
        assert len(manager.activation_history) == 0

    def test_get_state_graph_visualization(self, manager, sample_state, sample_transition):
        """Test state graph visualization."""
        manager.add_state(sample_state)
        home_state = State(name="home", elements=[], min_elements=0)
        manager.add_state(home_state)
        manager.state_graph.add_transition(sample_transition)

        manager.activate_state("login", 0.9)

        visualization = manager.get_state_graph_visualization()

        assert isinstance(visualization, str)
        assert "login" in visualization
        assert "home" in visualization
        assert "✓" in visualization  # Active state marker
        assert "click" in visualization  # Transition type

    def test_hierarchical_state_machine(self):
        """Test hierarchical state machine features."""
        manager = QontinuiStateManager(use_hierarchical=True)

        # Create parent and child states
        parent_state = State(name="main", elements=[], min_elements=0)
        child_state = State(name="main_login", elements=[], min_elements=0, parent_state="main")

        manager.add_state(parent_state)
        manager.add_state(child_state, parent="main")

        assert "main" in manager.state_graph.states
        assert "main_login" in manager.state_graph.states
