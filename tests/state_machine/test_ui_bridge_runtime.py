"""Tests for UI Bridge Runtime state machine integration.

These tests verify the state machine integration works correctly
with mock UI Bridge client.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

# Import the runtime components
from qontinui.state_machine import (
    StateDiscoveryConfig,
    TransitionDetector,
    TransitionDetectorConfig,
    UIBridgeRuntime,
    UIBridgeRuntimeConfig,
    UIBridgeState,
    UIBridgeStateDiscovery,
    UIBridgeTransition,
    generate_state_id,
)


class MockUIBridgeClient:
    """Mock UI Bridge client for testing."""

    def __init__(self) -> None:
        self._visible_elements: set[str] = set()
        self._click_results: dict[str, bool] = {}
        self._type_results: dict[str, bool] = {}

    def set_visible_elements(self, elements: list[str]) -> None:
        """Set which elements are visible."""
        self._visible_elements = set(elements)

    def find(
        self,
        *,
        interactive_only: bool = False,
        include_hidden: bool = False,
    ) -> Any:
        """Mock find method."""

        class Element:
            def __init__(self, elem_id: str):
                self.id = elem_id

        class Response:
            def __init__(self, elements: set[str]):
                self.elements = [Element(e) for e in elements]

        return Response(self._visible_elements)

    def get_active_states(self) -> list[str]:
        """Mock get_active_states - returns empty to force element detection."""
        return []

    def execute_transition(self, transition_id: str) -> Any:
        """Mock execute_transition."""
        return MagicMock(success=True)

    def navigate_to(self, target_states: list[str]) -> Any:
        """Mock navigate_to - raises to fall back to pathfinding."""
        raise NotImplementedError("Use pathfinding")

    def click(self, element_id: str, **kwargs: Any) -> Any:
        """Mock click action."""
        success = self._click_results.get(element_id, True)
        return MagicMock(success=success)

    def type(self, element_id: str, text: str, **kwargs: Any) -> Any:
        """Mock type action."""
        success = self._type_results.get(element_id, True)
        return MagicMock(success=success)


class TestUIBridgeState:
    """Tests for UIBridgeState dataclass."""

    def test_state_creation(self) -> None:
        """Test creating a basic state."""
        state = UIBridgeState(
            id="dashboard",
            name="Dashboard View",
            element_ids=["nav-menu", "sidebar", "main-content"],
        )

        assert state.id == "dashboard"
        assert state.name == "Dashboard View"
        assert len(state.element_ids) == 3
        assert state.blocking is False
        assert state.path_cost == 1.0

    def test_state_with_blocking(self) -> None:
        """Test creating a blocking (modal) state."""
        state = UIBridgeState(
            id="confirm_dialog",
            name="Confirmation Dialog",
            element_ids=["dialog-overlay", "dialog-content", "confirm-btn"],
            blocking=True,
            blocks=["dashboard", "sidebar"],
        )

        assert state.blocking is True
        assert "dashboard" in state.blocks

    def test_to_multistate_conversion(self) -> None:
        """Test converting to multistate format."""
        state = UIBridgeState(
            id="test_state",
            name="Test State",
            element_ids=["elem1", "elem2"],
            blocking=True,
            blocks=["other_state"],
            group="modals",
        )

        ms_params = state.to_multistate()

        assert ms_params["id"] == "test_state"
        assert ms_params["name"] == "Test State"
        assert ms_params["blocking"] is True
        assert "elem1" in ms_params["elements"]
        assert ms_params["group"] == "modals"


class TestUIBridgeTransition:
    """Tests for UIBridgeTransition dataclass."""

    def test_transition_creation(self) -> None:
        """Test creating a basic transition."""
        transition = UIBridgeTransition(
            id="open_settings",
            name="Open Settings Panel",
            from_states=["dashboard"],
            activate_states=["settings_panel"],
            exit_states=[],
            actions=[{"type": "click", "elementId": "settings-btn"}],
        )

        assert transition.id == "open_settings"
        assert "dashboard" in transition.from_states
        assert "settings_panel" in transition.activate_states
        assert len(transition.actions) == 1


class TestUIBridgeRuntime:
    """Tests for UIBridgeRuntime."""

    def test_runtime_creation(self) -> None:
        """Test creating runtime with mock client."""
        client = MockUIBridgeClient()
        runtime = UIBridgeRuntime(client)

        assert runtime.client == client
        assert runtime.manager is not None

    def test_register_state(self) -> None:
        """Test registering a state."""
        client = MockUIBridgeClient()
        runtime = UIBridgeRuntime(client)

        state = UIBridgeState(
            id="test_state",
            name="Test State",
            element_ids=["elem1", "elem2"],
        )

        runtime.register_state(state)

        assert "test_state" in runtime._ui_states
        assert "elem1" in runtime._element_to_states
        assert "test_state" in runtime._element_to_states["elem1"]

    def test_register_duplicate_state_fails(self) -> None:
        """Test that registering duplicate state fails."""
        client = MockUIBridgeClient()
        runtime = UIBridgeRuntime(client)

        state = UIBridgeState(id="test", name="Test", element_ids=["e1"])
        runtime.register_state(state)

        with pytest.raises(ValueError, match="already registered"):
            runtime.register_state(state)

    def test_register_transition(self) -> None:
        """Test registering a transition."""
        client = MockUIBridgeClient()
        runtime = UIBridgeRuntime(client)

        # Register states first
        runtime.register_state(UIBridgeState(id="state_a", name="State A", element_ids=["a1"]))
        runtime.register_state(UIBridgeState(id="state_b", name="State B", element_ids=["b1"]))

        # Register transition
        transition = UIBridgeTransition(
            id="a_to_b",
            name="A to B",
            from_states=["state_a"],
            activate_states=["state_b"],
            exit_states=["state_a"],
            actions=[{"type": "click", "elementId": "btn"}],
        )

        runtime.register_transition(transition)

        assert "a_to_b" in runtime._ui_transitions

    def test_detect_states_from_elements(self) -> None:
        """Test state detection from visible elements."""
        client = MockUIBridgeClient()
        runtime = UIBridgeRuntime(client)

        # Register states
        runtime.register_state(
            UIBridgeState(id="dashboard", name="Dashboard", element_ids=["nav", "sidebar"])
        )
        runtime.register_state(
            UIBridgeState(id="settings", name="Settings", element_ids=["settings-panel"])
        )

        # Set visible elements to match dashboard
        client.set_visible_elements(["nav", "sidebar", "extra-elem"])

        active = runtime.get_active_states()

        assert "dashboard" in active
        assert "settings" not in active

    def test_execute_transition_success(self) -> None:
        """Test successful transition execution."""
        client = MockUIBridgeClient()
        config = UIBridgeRuntimeConfig(state_activation_delay_ms=0)
        runtime = UIBridgeRuntime(client, config)

        # Register states and transition
        runtime.register_state(UIBridgeState(id="login", name="Login", element_ids=["login-form"]))
        runtime.register_state(
            UIBridgeState(id="dashboard", name="Dashboard", element_ids=["dashboard"])
        )

        runtime.register_transition(
            UIBridgeTransition(
                id="do_login",
                name="Login",
                from_states=["login"],
                activate_states=["dashboard"],
                exit_states=["login"],
                actions=[{"type": "click", "elementId": "submit-btn"}],
            )
        )

        result = runtime.execute_transition("do_login")

        assert result.success is True
        assert "dashboard" in result.activated_states
        assert "login" in result.deactivated_states

    def test_execute_transition_not_found(self) -> None:
        """Test transition execution with unknown transition."""
        client = MockUIBridgeClient()
        runtime = UIBridgeRuntime(client)

        result = runtime.execute_transition("unknown_transition")

        assert result.success is False
        assert "not registered" in result.error


class TestStateDiscovery:
    """Tests for UIBridgeStateDiscovery."""

    def test_process_single_render(self) -> None:
        """Test processing a single render."""
        discovery = UIBridgeStateDiscovery()

        render = {
            "id": "render_1",
            "type": "dom_snapshot",
            "snapshot": {
                "root": {
                    "attributes": {"data-ui-id": "nav-menu"},
                    "children": [
                        {"attributes": {"data-testid": "sidebar"}},
                    ],
                },
            },
        }

        elements = discovery.process_render(render)

        assert "ui:nav-menu" in elements
        assert "testid:sidebar" in elements

    def test_discover_states_from_renders(self) -> None:
        """Test state discovery from multiple renders."""
        config = StateDiscoveryConfig(
            min_element_occurrences=1,
            cooccurrence_threshold=0.9,
        )
        discovery = UIBridgeStateDiscovery(config)

        # Create renders where elements co-occur
        for i in range(3):
            discovery.process_render(
                {
                    "id": f"render_{i}",
                    "elements": [
                        {"id": "nav"},
                        {"id": "sidebar"},
                    ],
                }
            )

        states = discovery.discover_states()

        # Should find at least one state grouping nav and sidebar
        assert len(states) >= 1


class TestTransitionDetector:
    """Tests for TransitionDetector."""

    def test_record_action(self) -> None:
        """Test recording an action with state change."""
        config = TransitionDetectorConfig(min_observation_count=1)
        detector = TransitionDetector(config)

        # Record an action
        transition = detector.record_action(
            action={"type": "click", "elementId": "settings-btn"},
            before_states=["dashboard"],
            after_states=["dashboard", "settings_panel"],
        )

        # First observation won't create confirmed transition
        assert transition is None

        # Second observation confirms the pattern
        transition = detector.record_action(
            action={"type": "click", "elementId": "settings-btn"},
            before_states=["dashboard"],
            after_states=["dashboard", "settings_panel"],
        )

        assert transition is not None
        assert "settings_panel" in transition.activate_states

    def test_get_transitions_from_state(self) -> None:
        """Test getting transitions available from a state."""
        config = TransitionDetectorConfig(min_observation_count=1)
        detector = TransitionDetector(config)

        # Record multiple observations to confirm transition
        for _ in range(3):
            detector.record_action(
                action={"type": "click", "elementId": "btn"},
                before_states=["state_a"],
                after_states=["state_a", "state_b"],
            )

        transitions = detector.get_transitions_from_state("state_a")

        assert len(transitions) >= 1


class TestGenerateStateId:
    """Tests for state ID generation."""

    def test_deterministic_id(self) -> None:
        """Test that same elements produce same ID."""
        elements1 = ["elem_a", "elem_b", "elem_c"]
        elements2 = ["elem_c", "elem_a", "elem_b"]  # Different order

        id1 = generate_state_id(elements1)
        id2 = generate_state_id(elements2)

        assert id1 == id2  # Order shouldn't matter

    def test_different_elements_different_id(self) -> None:
        """Test that different elements produce different IDs."""
        id1 = generate_state_id(["a", "b"])
        id2 = generate_state_id(["a", "c"])

        assert id1 != id2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
