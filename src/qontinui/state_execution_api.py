"""StateExecutionAPI - Clean facade for state management.

This module provides the ONLY interface that API and Runner should use for
state management. All state operations are delegated to focused executor
components.

Architecture:
    - StateExecutionAPI: Facade/coordinator that delegates to executors
    - TransitionExecutor: Handles transition execution
    - StateNavigationExecutor: Handles navigation and pathfinding
    - StateValidator: Handles validation
    - StateEventEmitter: Handles event emission
    - StateQuery: Handles state queries and statistics
    - All state logic delegated to multistate components

Key Features:
    1. Transition execution via TransitionExecutor
    2. Multi-target pathfinding via StateNavigationExecutor
    3. State queries via StateQuery
    4. Rich result objects with detailed information
    5. Optional event emission callbacks for API/Runner integration

Example:
    >>> api = StateExecutionAPI(config, hal)
    >>> result = api.execute_transition("login_transition")
    >>> if result.success:
    ...     print(f"Activated states: {result.activated_states}")
    >>>
    >>> nav_result = api.navigate_to_states([1, 2, 3], execute=True)
    >>> if nav_result.success:
    ...     print(f"Path: {[t.name for t in nav_result.path]}")
"""

import logging
from collections.abc import Callable
from typing import Any

from qontinui.action_executors import DelegatingActionExecutor
from qontinui.json_executor.config_parser import QontinuiConfig
from qontinui.multistate_integration import (
    EnhancedStateMemory,
    EnhancedTransitionExecutor,
    PathfindingNavigator,
)
from qontinui.state_management.event_emitter import StateEventEmitter
from qontinui.state_management.state_navigation_executor import (
    NavigationResult,
    StateNavigationExecutor,
)
from qontinui.state_management.state_query import StateQuery
from qontinui.state_management.state_validator import StateValidator
from qontinui.state_management.transition_executor import (
    TransitionExecutionResult,
    TransitionExecutor,
)

logger = logging.getLogger(__name__)


class StateExecutionAPI:
    """Facade for state execution and navigation.

    This is the ONLY interface that API and Runner should use for state
    management. It delegates all operations to focused executor components:
    - TransitionExecutor: Transition execution
    - StateNavigationExecutor: Navigation and pathfinding
    - StateValidator: Validation
    - StateEventEmitter: Event emission
    - StateQuery: State queries and statistics

    The API provides:
    1. Transition execution with workflow execution
    2. Multi-target pathfinding and navigation
    3. State queries (active states, available transitions)
    4. Event emission integration via callbacks
    5. Rich result objects with detailed information

    Example:
        >>> api = StateExecutionAPI(config, hal)
        >>>
        >>> # Execute transition
        >>> result = api.execute_transition("login_transition")
        >>> if result.success:
        ...     print(f"Now in states: {result.active_states_after}")
        >>>
        >>> # Navigate to multiple states
        >>> nav = api.navigate_to_states([1, 2, 3], execute=True)
        >>> if nav.success:
        ...     print(f"Reached: {nav.targets_reached}")
    """

    def __init__(self, config: QontinuiConfig, hal: Any) -> None:
        """Initialize StateExecutionAPI.

        Args:
            config: Qontinui configuration with states and transitions
            hal: Hardware abstraction layer for device control

        Raises:
            ValueError: If config is invalid or missing required data
        """
        self.config = config
        self.hal = hal

        logger.info("Initializing StateExecutionAPI")

        # Initialize components
        try:
            self._initialize_state_management()
            self._initialize_executors()
            self._register_states_and_transitions()
            logger.info("StateExecutionAPI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize StateExecutionAPI: {e}")
            raise

    def _initialize_state_management(self) -> None:
        """Initialize multistate management components.

        Creates:
        - EnhancedStateMemory: For state tracking
        - DelegatingActionExecutor: For workflow execution
        - EnhancedTransitionExecutor: For transition execution
        - PathfindingNavigator: For pathfinding

        Raises:
            ValueError: If initialization fails
        """
        logger.debug("Initializing state management components")

        # Create state memory
        self.state_memory = EnhancedStateMemory(state_service=None)
        logger.debug("Created EnhancedStateMemory")

        # Create action executor for workflow execution
        self.action_executor = DelegatingActionExecutor(config=self.config, hal=self.hal)
        logger.debug("Created DelegatingActionExecutor")

        # Create transition executor with workflow executor
        self.transition_executor = EnhancedTransitionExecutor(
            state_memory=self.state_memory,
            multistate_adapter=self.state_memory.multistate_adapter,
            workflow_executor=self.action_executor,
        )
        logger.debug("Created EnhancedTransitionExecutor")

        # Create pathfinding navigator
        self.navigator = PathfindingNavigator(
            state_memory=self.state_memory,
            transition_executor=self.transition_executor,
            multistate_adapter=self.state_memory.multistate_adapter,
            workflow_executor=self.action_executor,
        )
        logger.debug("Created PathfindingNavigator")

    def _initialize_executors(self) -> None:
        """Initialize focused executor components.

        Creates:
        - StateValidator: For validation
        - StateEventEmitter: For event emission
        - TransitionExecutor: For transition execution
        - StateNavigationExecutor: For navigation
        - StateQuery: For state queries
        """
        logger.debug("Initializing executor components")

        # Create validator
        self.validator = StateValidator(config=self.config)
        logger.debug("Created StateValidator")

        # Create event emitter
        self.event_emitter = StateEventEmitter()
        logger.debug("Created StateEventEmitter")

        # Create transition executor
        self.transition_executor_facade = TransitionExecutor(
            validator=self.validator,
            state_memory=self.state_memory,
            transition_executor=self.transition_executor,
            event_emitter=self.event_emitter,
        )
        logger.debug("Created TransitionExecutor")

        # Create navigation executor
        self.navigation_executor = StateNavigationExecutor(
            navigator=self.navigator,
            event_emitter=self.event_emitter,
        )
        logger.debug("Created StateNavigationExecutor")

        # Create state query
        self.query = StateQuery(
            state_memory=self.state_memory,
            validator=self.validator,
            config=self.config,
            transition_executor=self.transition_executor,
            navigator=self.navigator,
        )
        logger.debug("Created StateQuery")

    def _register_states_and_transitions(self) -> None:
        """Register states and transitions from config.

        Processes the config to register all states and transitions with
        the multistate adapter for pathfinding and execution.

        Raises:
            ValueError: If registration fails
        """
        logger.debug("Registering states and transitions from config")

        # Register states with multistate adapter
        for state in self.config.states:
            try:
                self.state_memory.multistate_adapter.register_qontinui_state(state)
                logger.debug(f"Registered state: {state.id}")
            except Exception as e:
                logger.warning(f"Failed to register state {state.id}: {e}")

        logger.info(
            f"Registered {len(self.config.states)} states and "
            f"{len(self.config.transitions)} transitions"
        )

    def execute_transition(
        self, transition_id: str, emit_event_callback: Callable[[str, dict], None] | None = None
    ) -> TransitionExecutionResult:
        """Execute a transition by ID.

        Delegates to TransitionExecutor for execution with phased workflow.

        Args:
            transition_id: ID of transition to execute
            emit_event_callback: Optional callback for emitting events

        Returns:
            TransitionExecutionResult with execution details

        Example:
            >>> result = api.execute_transition("login_transition")
            >>> if result.success:
            ...     print(f"Activated: {result.activated_states}")
        """
        return self.transition_executor_facade.execute_transition(
            transition_id, emit_event_callback
        )

    def navigate_to_states(
        self,
        target_state_ids: list[int],
        execute: bool = True,
        emit_event_callback: Callable[[str, dict], None] | None = None,
    ) -> NavigationResult:
        """Navigate to reach ALL specified target states.

        Delegates to StateNavigationExecutor for pathfinding and execution.

        Args:
            target_state_ids: IDs of ALL states to reach
            execute: If True, execute the path; if False, just compute it
            emit_event_callback: Optional callback for emitting events

        Returns:
            NavigationResult with path and execution details

        Example:
            >>> result = api.navigate_to_states([1, 2, 3], execute=True)
            >>> if result.success:
            ...     print(f"Reached all targets")
        """
        return self.navigation_executor.navigate_to_states(
            target_state_ids, execute, emit_event_callback
        )

    def get_active_states(self) -> set[int]:
        """Get currently active state IDs.

        Delegates to StateQuery for state information.

        Returns:
            Set of active state IDs

        Example:
            >>> active = api.get_active_states()
            >>> print(f"Active states: {active}")
        """
        return self.query.get_active_states()

    def get_available_transitions(self) -> list[Any]:
        """Get transitions available from current states.

        Delegates to StateQuery for available transitions.

        Returns:
            List of available transition objects

        Example:
            >>> transitions = api.get_available_transitions()
            >>> for t in transitions:
            ...     print(f"Available: {t.name}")
        """
        return self.query.get_available_transitions()

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about state execution.

        Delegates to StateQuery for statistics.

        Returns:
            Dictionary with execution statistics

        Example:
            >>> stats = api.get_statistics()
            >>> print(f"Total transitions: {stats['total_transitions']}")
        """
        return self.query.get_statistics()
