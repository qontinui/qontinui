"""UI Bridge Runtime for MultiState Integration.

This module implements the multistate StateSpaceRuntime protocol for UI Bridge,
enabling model-based GUI automation with pathfinding support.

The runtime:
- Queries UI Bridge for current visible elements
- Maps elements to states via co-occurrence analysis
- Executes transitions via UI Bridge actions
- Integrates with multistate pathfinding

Example:
    from qontinui.state_machine import UIBridgeRuntime
    from ui_bridge import UIBridgeClient

    client = UIBridgeClient("http://localhost:9876")
    runtime = UIBridgeRuntime(client)

    # Get active states based on visible elements
    active = runtime.get_active_states()

    # Navigate to target states
    result = runtime.navigate_to(["checkout_page", "cart_visible"])

    # Execute a specific transition
    result = runtime.execute_transition("open_settings")
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

# Import multistate components
try:
    from multistate import State, StateManager, StateManagerConfig, Transition
    from multistate.pathfinding.multi_target import Path, SearchStrategy
    from multistate.transitions.reliability import ReliabilityTracker

    MULTISTATE_AVAILABLE = True
except ImportError:
    # Multistate not available - provide stub types
    MULTISTATE_AVAILABLE = False

    class State:  # type: ignore[no-redef]
        """Stub State class when multistate is not available."""

        def __init__(self, **kwargs: Any) -> None:
            self.id = kwargs.get("id", "")
            self.name = kwargs.get("name", "")

    class Transition:  # type: ignore[no-redef]
        """Stub Transition class when multistate is not available."""

        def __init__(self, **kwargs: Any) -> None:
            self.id = kwargs.get("id", "")

    class StateManager:  # type: ignore[no-redef]
        """Stub StateManager when multistate is not available."""

        def __init__(self, config: Any = None) -> None:
            self._states: dict[str, Any] = {}
            self._transitions: dict[str, Any] = {}
            self._active: set[str] = set()

        def add_state(self, **kwargs: Any) -> State:
            state = State(**kwargs)
            self._states[kwargs.get("id", "")] = state
            return state

        def add_transition(self, **kwargs: Any) -> Transition:
            trans = Transition(**kwargs)
            self._transitions[kwargs.get("id", "")] = trans
            return trans

        def find_path_to(self, targets: list[str], **kwargs: Any) -> Any:
            return None

        def can_execute(self, transition_id: str) -> bool:
            return transition_id in self._transitions

        def execute_transition(self, transition_id: str) -> None:
            pass

        def activate_states(self, state_ids: set[str]) -> None:
            self._active = state_ids

        def get_available_transitions(self) -> list[Any]:
            return list(self._transitions.values())

        def analyze_complexity(self) -> dict[str, Any]:
            return {
                "num_states": len(self._states),
                "num_transitions": len(self._transitions),
                "active_states": len(self._active),
                "reachable_states": len(self._states),
                "num_groups": 0,
            }

    class StateManagerConfig:  # type: ignore[no-redef]
        """Stub StateManagerConfig when multistate is not available."""

        def __init__(self, **kwargs: Any) -> None:
            pass

    class Path:  # type: ignore[no-redef]
        """Stub Path when multistate is not available."""

        def __init__(self) -> None:
            self.transitions_sequence: list[Any] = []
            self.targets: set[Any] = set()

    class SearchStrategy:  # type: ignore[no-redef]
        """Stub SearchStrategy enum."""

        DIJKSTRA = "dijkstra"
        BFS = "bfs"
        A_STAR = "astar"

    class ReliabilityTracker:  # type: ignore[no-redef]
        """Stub ReliabilityTracker when multistate is not available."""

        def __init__(self, **kwargs: Any) -> None:
            self._stats: dict[str, Any] = {}

        def record_success(self, transition_id: str, duration: float = 0.0) -> None:
            pass

        def record_failure(self, transition_id: str, duration: float = 0.0) -> None:
            pass

        def get_stats(self, transition_id: str) -> Any:
            return type("Stats", (), {"success_rate": 1.0})()

        def get_dynamic_cost(self, transition_id: str, base_cost: float) -> float:
            return base_cost

        def get_summary(self) -> dict[str, Any]:
            return {}


logger = logging.getLogger(__name__)


class UIBridgeClientProtocol(Protocol):
    """Protocol for UI Bridge client to allow dependency injection."""

    def find(
        self,
        *,
        interactive_only: bool = False,
        include_hidden: bool = False,
    ) -> Any:
        """Find elements in the UI."""
        ...

    def get_active_states(self) -> list[str]:
        """Get currently active state IDs from UI Bridge."""
        ...

    def execute_transition(self, transition_id: str) -> Any:
        """Execute a transition by ID."""
        ...

    def navigate_to(self, target_states: list[str]) -> Any:
        """Navigate to target states."""
        ...

    def click(self, element_id: str, **kwargs: Any) -> Any:
        """Click an element."""
        ...

    def type(self, element_id: str, text: str, **kwargs: Any) -> Any:
        """Type into an element."""
        ...


@dataclass
class UIBridgeState:
    """Represents a UI state in the UI Bridge context.

    States are defined by collections of elements that appear together.
    """

    id: str
    name: str
    element_ids: list[str]
    blocking: bool = False
    blocks: list[str] = field(default_factory=list)
    group: str | None = None
    path_cost: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_multistate(self) -> dict[str, Any]:
        """Convert to multistate State parameters."""
        return {
            "id": self.id,
            "name": self.name,
            "elements": set(self.element_ids),
            "blocking": self.blocking,
            "blocks": set(self.blocks),
            "group": self.group,
        }


@dataclass
class UIBridgeTransition:
    """Represents a transition between UI states.

    Transitions define how to move from one state configuration to another,
    including the actions to execute.
    """

    id: str
    name: str
    from_states: list[str]
    activate_states: list[str]
    exit_states: list[str]
    actions: list[dict[str, Any]] = field(default_factory=list)
    activate_groups: list[str] = field(default_factory=list)
    exit_groups: list[str] = field(default_factory=list)
    path_cost: float = 1.0
    stays_visible: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionExecutionResult:
    """Result of executing a transition."""

    success: bool
    transition_id: str
    activated_states: list[str]
    deactivated_states: list[str]
    error: str | None = None
    duration_ms: float = 0.0
    action_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class UIBridgeRuntimeConfig:
    """Configuration for UI Bridge Runtime."""

    # State detection settings
    element_visibility_threshold: float = 0.5
    state_activation_delay_ms: int = 100

    # Pathfinding settings
    search_strategy: SearchStrategy = SearchStrategy.DIJKSTRA
    max_path_depth: int = 50

    # Reliability tracking
    enable_reliability_tracking: bool = True
    reliability_cost_multiplier: float = 2.0

    # Logging
    log_level: str = "INFO"
    log_transitions: bool = True


class UIBridgeRuntime:
    """Runtime implementation for UI Bridge + multistate integration.

    This class bridges UI Bridge's element-based API with multistate's
    state machine framework, enabling:

    1. **State Detection**: Query UI Bridge for visible elements and map
       to application states
    2. **Transition Execution**: Execute transitions via UI Bridge actions
    3. **Pathfinding**: Use multistate's multi-target pathfinding to navigate
    4. **Reliability Tracking**: Track transition success/failure for path cost

    The runtime follows the StateSpaceRuntime protocol from multistate,
    making it compatible with all multistate pathfinding algorithms.
    """

    def __init__(
        self,
        client: UIBridgeClientProtocol,
        config: UIBridgeRuntimeConfig | None = None,
    ) -> None:
        """Initialize UI Bridge Runtime.

        Args:
            client: UI Bridge client for element/action access
            config: Runtime configuration
        """
        self.client = client
        self.config = config or UIBridgeRuntimeConfig()

        # Initialize multistate manager
        manager_config = StateManagerConfig(
            default_search_strategy=self.config.search_strategy,
            max_path_depth=self.config.max_path_depth,
            log_transitions=self.config.log_transitions,
            enable_metrics=True,
        )
        self.manager = StateManager(manager_config)

        # State and transition registries
        self._ui_states: dict[str, UIBridgeState] = {}
        self._ui_transitions: dict[str, UIBridgeTransition] = {}

        # Element to state mapping for fast lookup
        self._element_to_states: dict[str, set[str]] = {}

        # Reliability tracking
        self._reliability: ReliabilityTracker | None = None
        if self.config.enable_reliability_tracking:
            self._reliability = ReliabilityTracker(
                cost_multiplier_on_failure=self.config.reliability_cost_multiplier
            )

        # Action handlers for different action types
        self._action_handlers: dict[str, Callable[..., bool]] = {
            "click": self._handle_click_action,
            "type": self._handle_type_action,
            "select": self._handle_select_action,
            "wait": self._handle_wait_action,
            "navigate": self._handle_navigate_action,
        }

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(getattr(logging, self.config.log_level))

    # =========================================================================
    # State Registration
    # =========================================================================

    def register_state(self, state: UIBridgeState) -> State:
        """Register a UI state with the runtime.

        Args:
            state: UI Bridge state definition

        Returns:
            multistate State object
        """
        if state.id in self._ui_states:
            raise ValueError(f"State '{state.id}' already registered")

        # Store UI Bridge state
        self._ui_states[state.id] = state

        # Build element -> state mapping
        for elem_id in state.element_ids:
            if elem_id not in self._element_to_states:
                self._element_to_states[elem_id] = set()
            self._element_to_states[elem_id].add(state.id)

        # Register with multistate manager
        ms_state = self.manager.add_state(
            id=state.id,
            name=state.name,
            elements=set(state.element_ids),
            blocking=state.blocking,
            blocks=set(state.blocks),
            group=state.group,
        )

        self.logger.info(f"Registered state: {state.id} with {len(state.element_ids)} elements")
        return ms_state

    def register_states(self, states: list[UIBridgeState]) -> list[State]:
        """Register multiple states.

        Args:
            states: List of UI Bridge state definitions

        Returns:
            List of multistate State objects
        """
        return [self.register_state(s) for s in states]

    def register_transition(self, transition: UIBridgeTransition) -> Transition:
        """Register a transition with the runtime.

        Args:
            transition: UI Bridge transition definition

        Returns:
            multistate Transition object
        """
        if transition.id in self._ui_transitions:
            raise ValueError(f"Transition '{transition.id}' already registered")

        # Validate states exist
        for state_id in (
            transition.from_states + transition.activate_states + transition.exit_states
        ):
            if state_id not in self._ui_states:
                raise ValueError(f"State '{state_id}' not registered")

        # Store UI Bridge transition
        self._ui_transitions[transition.id] = transition

        # Register with multistate manager
        ms_transition = self.manager.add_transition(
            id=transition.id,
            name=transition.name,
            from_states=transition.from_states,
            activate_states=transition.activate_states,
            exit_states=transition.exit_states,
            activate_groups=transition.activate_groups,
            exit_groups=transition.exit_groups,
            path_cost=transition.path_cost,
        )

        self.logger.info(
            f"Registered transition: {transition.id} "
            f"({transition.from_states} -> {transition.activate_states})"
        )
        return ms_transition

    def register_transitions(self, transitions: list[UIBridgeTransition]) -> list[Transition]:
        """Register multiple transitions.

        Args:
            transitions: List of UI Bridge transition definitions

        Returns:
            List of multistate Transition objects
        """
        return [self.register_transition(t) for t in transitions]

    # =========================================================================
    # State Detection (StateSpaceRuntime protocol)
    # =========================================================================

    def get_active_states(self) -> set[str]:
        """Get currently active states based on visible UI elements.

        This queries UI Bridge for visible elements and maps them to states.

        Returns:
            Set of active state IDs
        """
        try:
            # Try to get states directly from UI Bridge first
            ui_bridge_states = self.client.get_active_states()
            if ui_bridge_states:
                return set(ui_bridge_states)
        except Exception:
            # Fall back to element-based detection
            pass

        return self._detect_states_from_elements()

    def _detect_states_from_elements(self) -> set[str]:
        """Detect active states by checking visible elements.

        Returns:
            Set of active state IDs
        """
        try:
            # Get all visible elements
            response = self.client.find(interactive_only=False, include_hidden=False)
            visible_elements = {elem.id for elem in response.elements}
        except Exception as e:
            self.logger.warning(f"Failed to find elements: {e}")
            return set()

        # Check which states have all their elements visible
        active_states: set[str] = set()

        for state_id, state in self._ui_states.items():
            state_elements = set(state.element_ids)

            # A state is active if all its elements are visible
            if state_elements and state_elements.issubset(visible_elements):
                active_states.add(state_id)

        return active_states

    def is_state_active(self, state_id: str) -> bool:
        """Check if a specific state is currently active.

        Args:
            state_id: State ID to check

        Returns:
            True if state is active
        """
        if state_id not in self._ui_states:
            raise ValueError(f"State '{state_id}' not registered")

        active = self.get_active_states()
        return state_id in active

    def sync_active_states(self) -> set[str]:
        """Sync runtime's active states with actual UI state.

        Returns:
            Current active state IDs
        """
        active = self.get_active_states()

        # Update multistate manager
        self.manager.activate_states(active)

        self.logger.debug(f"Synced {len(active)} active states")
        return active

    # =========================================================================
    # Transition Execution
    # =========================================================================

    def can_execute_transition(self, transition_id: str) -> bool:
        """Check if a transition can be executed from current state.

        Args:
            transition_id: Transition to check

        Returns:
            True if transition can execute
        """
        if transition_id not in self._ui_transitions:
            return False

        # Sync active states first
        self.sync_active_states()

        return bool(self.manager.can_execute(transition_id))

    def execute_transition(self, transition_id: str) -> TransitionExecutionResult:
        """Execute a transition via UI Bridge actions.

        Args:
            transition_id: Transition to execute

        Returns:
            TransitionExecutionResult with execution details
        """
        if transition_id not in self._ui_transitions:
            return TransitionExecutionResult(
                success=False,
                transition_id=transition_id,
                activated_states=[],
                deactivated_states=[],
                error=f"Transition '{transition_id}' not registered",
            )

        transition = self._ui_transitions[transition_id]
        start_time = time.time()
        action_results: list[dict[str, Any]] = []

        try:
            # Execute each action in the transition
            for action in transition.actions:
                action_type = action.get("type", "click")
                handler = self._action_handlers.get(action_type)

                if not handler:
                    raise ValueError(f"Unknown action type: {action_type}")

                success = handler(action)
                action_results.append({"action": action, "success": success})

                if not success:
                    raise RuntimeError(f"Action failed: {action}")

            # Small delay for UI to settle
            if self.config.state_activation_delay_ms > 0:
                time.sleep(self.config.state_activation_delay_ms / 1000)

            # Update multistate manager
            self.manager.execute_transition(transition_id)

            duration_ms = (time.time() - start_time) * 1000

            # Record reliability
            if self._reliability:
                self._reliability.record_success(transition_id, duration_ms / 1000)

            self.logger.info(f"Executed transition '{transition_id}' successfully")

            return TransitionExecutionResult(
                success=True,
                transition_id=transition_id,
                activated_states=transition.activate_states,
                deactivated_states=transition.exit_states,
                duration_ms=duration_ms,
                action_results=action_results,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            # Record reliability
            if self._reliability:
                self._reliability.record_failure(transition_id, duration_ms / 1000)

            self.logger.error(f"Transition '{transition_id}' failed: {e}")

            return TransitionExecutionResult(
                success=False,
                transition_id=transition_id,
                activated_states=[],
                deactivated_states=[],
                error=str(e),
                duration_ms=duration_ms,
                action_results=action_results,
            )

    # =========================================================================
    # Pathfinding & Navigation
    # =========================================================================

    def find_path(
        self,
        target_states: list[str],
        from_states: set[str] | None = None,
    ) -> Path | None:
        """Find path to reach ALL target states.

        Args:
            target_states: States that must ALL be reached
            from_states: Starting states (uses current if None)

        Returns:
            Path to reach all targets, or None if impossible
        """
        # Sync current states
        if from_states is None:
            from_states = self.sync_active_states()

        return self.manager.find_path_to(target_states, from_states=from_states)

    def navigate_to(
        self,
        target_states: list[str],
        strategy: SearchStrategy | None = None,
    ) -> TransitionExecutionResult:
        """Navigate to target states using pathfinding.

        This finds the optimal path and executes all transitions.

        Args:
            target_states: States to reach
            strategy: Search strategy (uses config default if None)

        Returns:
            Aggregated execution result
        """
        # Try UI Bridge's navigate_to first
        try:
            result = self.client.navigate_to(target_states)
            if result.success:
                return TransitionExecutionResult(
                    success=True,
                    transition_id="ui_bridge_navigate",
                    activated_states=result.final_active_states,
                    deactivated_states=[],
                    duration_ms=result.duration_ms,
                )
        except Exception:
            pass

        # Fall back to multistate pathfinding
        path = self.find_path(target_states)

        if not path:
            return TransitionExecutionResult(
                success=False,
                transition_id="navigate",
                activated_states=[],
                deactivated_states=[],
                error=f"No path found to states: {target_states}",
            )

        # Execute path
        return self._execute_path(path)

    def _execute_path(self, path: Path) -> TransitionExecutionResult:
        """Execute a navigation path.

        Args:
            path: Path to execute

        Returns:
            Aggregated execution result
        """
        all_activated: list[str] = []
        all_deactivated: list[str] = []
        all_action_results: list[dict[str, Any]] = []
        total_duration = 0.0

        for transition in path.transitions_sequence:
            result = self.execute_transition(transition.id)

            total_duration += result.duration_ms
            all_action_results.extend(result.action_results)

            if result.success:
                all_activated.extend(result.activated_states)
                all_deactivated.extend(result.deactivated_states)
            else:
                # Path execution failed
                return TransitionExecutionResult(
                    success=False,
                    transition_id=f"path_to_{path.targets}",
                    activated_states=all_activated,
                    deactivated_states=all_deactivated,
                    error=f"Failed at transition '{transition.id}': {result.error}",
                    duration_ms=total_duration,
                    action_results=all_action_results,
                )

        return TransitionExecutionResult(
            success=True,
            transition_id=f"path_to_{[s.id for s in path.targets]}",
            activated_states=all_activated,
            deactivated_states=all_deactivated,
            duration_ms=total_duration,
            action_results=all_action_results,
        )

    # =========================================================================
    # Action Handlers
    # =========================================================================

    def _handle_click_action(self, action: dict[str, Any]) -> bool:
        """Handle click action."""
        element_id = action.get("elementId") or action.get("target")
        if not element_id:
            raise ValueError("Click action requires elementId or target")

        try:
            result = self.client.click(element_id)
            return bool(result.success)
        except Exception as e:
            self.logger.error(f"Click action failed: {e}")
            return False

    def _handle_type_action(self, action: dict[str, Any]) -> bool:
        """Handle type action."""
        element_id = action.get("elementId") or action.get("target")
        text = action.get("text", "")
        clear = action.get("clear", False)

        if not element_id:
            raise ValueError("Type action requires elementId or target")

        try:
            result = self.client.type(element_id, text, clear=clear)
            return bool(result.success)
        except Exception as e:
            self.logger.error(f"Type action failed: {e}")
            return False

    def _handle_select_action(self, action: dict[str, Any]) -> bool:
        """Handle select action."""
        element_id = action.get("elementId") or action.get("target")
        value = action.get("value")

        if not element_id or not value:
            raise ValueError("Select action requires elementId and value")

        try:
            # Assume client has select method
            result = getattr(self.client, "select", lambda *a, **k: None)(element_id, value)
            return result.success if result else False
        except Exception as e:
            self.logger.error(f"Select action failed: {e}")
            return False

    def _handle_wait_action(self, action: dict[str, Any]) -> bool:
        """Handle wait action."""
        delay_ms = action.get("delayMs", action.get("delay", 1000))
        time.sleep(delay_ms / 1000)
        return True

    def _handle_navigate_action(self, action: dict[str, Any]) -> bool:
        """Handle navigate action (URL navigation)."""
        url = action.get("url")
        if not url:
            raise ValueError("Navigate action requires url")

        try:
            # Assume client has navigate method
            result = getattr(self.client, "navigate", lambda *a, **k: None)(url)
            return result.success if result else True  # Allow missing navigate
        except Exception as e:
            self.logger.error(f"Navigate action failed: {e}")
            return False

    # =========================================================================
    # Reliability & Statistics
    # =========================================================================

    def get_transition_reliability(self, transition_id: str) -> float:
        """Get reliability score for a transition.

        Args:
            transition_id: Transition to check

        Returns:
            Reliability score (0.0 to 1.0)
        """
        if not self._reliability:
            return 1.0

        stats = self._reliability.get_stats(transition_id)
        return float(stats.success_rate)

    def get_dynamic_path_cost(self, transition_id: str) -> float:
        """Get dynamic path cost adjusted for reliability.

        Args:
            transition_id: Transition to check

        Returns:
            Adjusted path cost
        """
        if transition_id not in self._ui_transitions:
            return float("inf")

        base_cost = self._ui_transitions[transition_id].path_cost

        if self._reliability:
            return self._reliability.get_dynamic_cost(transition_id, base_cost)

        return base_cost

    def get_statistics(self) -> dict[str, Any]:
        """Get runtime statistics.

        Returns:
            Dictionary with runtime statistics
        """
        complexity = self.manager.analyze_complexity()

        stats: dict[str, Any] = {
            "states": {
                "registered": len(self._ui_states),
                "active": len(self.get_active_states()),
                "multistate_states": complexity["num_states"],
                "reachable": complexity["reachable_states"],
            },
            "transitions": {
                "registered": len(self._ui_transitions),
                "available": len(self.manager.get_available_transitions()),
                "multistate_transitions": complexity["num_transitions"],
            },
            "groups": complexity["num_groups"],
            "complexity": complexity,
        }

        if self._reliability:
            stats["reliability"] = self._reliability.get_summary()

        return stats

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Serialize runtime state to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "states": {sid: vars(s) for sid, s in self._ui_states.items()},
            "transitions": {tid: vars(t) for tid, t in self._ui_transitions.items()},
            "active_states": list(self.get_active_states()),
            "config": vars(self.config),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        client: UIBridgeClientProtocol,
    ) -> UIBridgeRuntime:
        """Create runtime from dictionary.

        Args:
            data: Serialized runtime data
            client: UI Bridge client

        Returns:
            UIBridgeRuntime instance
        """
        config_data = data.get("config", {})
        config = UIBridgeRuntimeConfig(**config_data)

        runtime = cls(client, config)

        # Register states
        for state_data in data.get("states", {}).values():
            state = UIBridgeState(**state_data)
            runtime.register_state(state)

        # Register transitions
        for trans_data in data.get("transitions", {}).values():
            trans = UIBridgeTransition(**trans_data)
            runtime.register_transition(trans)

        return runtime


def generate_state_id(element_ids: list[str]) -> str:
    """Generate a deterministic state ID from element IDs.

    This creates a stable ID based on the sorted element combination.

    Args:
        element_ids: Elements that define the state

    Returns:
        Deterministic state ID
    """
    sorted_ids = sorted(element_ids)
    hash_input = "|".join(sorted_ids)
    hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
    return f"state_{hash_value}"
