"""Delegating action executor that routes to specialized executors.

This module provides the main ActionExecutor that delegates execution to specialized
command pattern executors via the registry pattern.
"""

import logging
from types import SimpleNamespace
from typing import Any, Optional

from pydantic import ValidationError

from ..config import Action, get_typed_config
from ..wrappers import Keyboard, Mouse, Screen, TimeWrapper
from .base import ExecutionContext
from .registry import create_executor, get_registered_action_types

logger = logging.getLogger(__name__)


class DelegatingActionExecutor:
    """Main action executor that delegates to specialized executors.

    Key responsibilities:
    - Action validation via get_typed_config
    - Retry logic with configurable attempts
    - Pre/post action pauses
    - Event emission for monitoring
    - Cross-cutting concerns

    Delegated responsibilities:
    - Actual action execution (to specialized executors)
    - Action-specific logic (click, type, find, etc.)

    Attributes:
        config: Parsed automation configuration containing states, workflows, images.
        state_executor: Reference to StateExecutor for GO_TO_STATE navigation.
        defaults: Default action configuration from system settings.
        last_find_location: Coordinates (x, y) of most recent FIND result.
        context: Shared execution context passed to all specialized executors.

    Example:
        >>> config = ConfigParser().parse_file("automation.json")
        >>> executor = DelegatingActionExecutor(config)
        >>> action = Action(type="CLICK", config={"x": 100, "y": 200})
        >>> success = executor.execute_action(action)
    """

    def __init__(
        self,
        config: Any,  # QontinuiConfig
        state_executor: Optional[Any] = None,
        use_graph_execution: bool = False,
        workflow_executor: Optional[Any] = None,
    ) -> None:
        """Initialize DelegatingActionExecutor.

        Args:
            config: Parsed automation configuration
            state_executor: Optional state executor for GO_TO_STATE navigation
            use_graph_execution: Whether to use graph-based execution
            workflow_executor: Optional reference to workflow executor for RUN_WORKFLOW
        """
        self.config = config
        self.state_executor = state_executor
        self.use_graph_execution = use_graph_execution
        self.workflow_executor = workflow_executor

        # Track last find result for "Last Find Result" targeting
        self.last_find_location: Optional[tuple[int, int]] = None

        # Get action defaults configuration
        self.defaults = self._create_defaults()

        # Initialize wrappers (HAL components)
        self.time_wrapper = TimeWrapper()
        self.mouse_wrapper = Mouse()
        self.keyboard_wrapper = Keyboard()
        self.screen_wrapper = Screen()

        # Initialize variable context and executors for control flow and data operations
        from ..actions.control_flow import ControlFlowExecutor
        from ..actions.data_operations import DataOperationsExecutor, VariableContext

        self.variable_context = VariableContext()
        self.data_operations_executor = DataOperationsExecutor(self.variable_context)

        # Create action executor callback for control flow
        def action_executor_callback(action_id: str, variables: dict) -> dict:
            """Execute an action by ID with given variable context."""
            # Find the action in config
            for workflow in self.config.workflows:
                for action in workflow.actions:
                    if action.id == action_id:
                        # Update variable context before execution
                        for key, value in variables.items():
                            self.variable_context.set(key, value, "local")

                        # Execute the action
                        success = self.execute_action(action)
                        return {"success": success}

            logger.warning(f"Action ID '{action_id}' not found in config")
            return {"success": False, "error": f"Action ID '{action_id}' not found"}

        self.control_flow_executor = ControlFlowExecutor(
            action_executor=action_executor_callback,
            variables=self.variable_context.get_all_variables(),
        )

        # Create execution context for specialized executors
        self.context = ExecutionContext(
            # Configuration access
            config=self.config,
            defaults=self.defaults,
            # HAL components (wrappers)
            mouse=self.mouse_wrapper,
            keyboard=self.keyboard_wrapper,
            screen=self.screen_wrapper,
            time=self.time_wrapper,
            # Shared state
            last_find_location=self.last_find_location,
            variable_context=self.variable_context,
            state_executor=self.state_executor,
            # Sub-executors for control flow and data operations
            control_flow_executor=self.control_flow_executor,
            data_operations_executor=self.data_operations_executor,
            # Workflow execution
            workflow_executor=self.workflow_executor,
            execute_action=self.execute_action,
            # Event emission functions
            emit_event=self._emit_event,
            emit_action_event=self._emit_action_event,
            emit_image_recognition_event=self._emit_image_recognition_event,
        )

        logger.info(
            f"DelegatingActionExecutor initialized with {len(get_registered_action_types())} "
            f"registered action types (graph_execution={'enabled' if use_graph_execution else 'disabled'})"
        )

    def _create_defaults(self) -> SimpleNamespace:
        """Create default configuration for actions.

        Returns:
            SimpleNamespace with mouse and keyboard defaults
        """
        return SimpleNamespace(
            mouse=SimpleNamespace(
                click_hold_duration=50,  # ms
                click_release_delay=100,  # ms
                click_safety_release=True,
                safety_release_delay=50,  # ms
                drag_default_duration=1.0,  # seconds
                drag_start_delay=0.1,  # seconds
                drag_end_delay=0.1,  # seconds
            ),
            keyboard=SimpleNamespace(
                key_press_duration=50,  # ms
            ),
        )

    def execute_action(self, action: Action) -> bool:
        """Execute a single automation action with retry logic.

        Handles the complete lifecycle of action execution including:
        - Action validation via get_typed_config
        - Pre-action pauses
        - Retry logic with configurable attempts
        - Delegation to specialized executors
        - Post-action pauses
        - Event emission for real-time monitoring

        Args:
            action: Action object containing type, configuration, and execution parameters

        Returns:
            bool: True if action executed successfully within retry attempts,
                False if all attempts failed

        Example:
            >>> action = Action(
            ...     id="click_1",
            ...     type="CLICK",
            ...     config={"target": {"type": "coordinates", "coordinates": {"x": 100, "y": 200}}},
            ...     execution={"retryCount": 3, "continueOnError": False}
            ... )
            >>> success = executor.execute_action(action)
        """
        logger.info(f"Executing action: {action.type} (ID: {action.id})")

        # Validate action configuration using Pydantic schemas
        typed_config = None
        try:
            typed_config = get_typed_config(action)
            logger.debug(f"Action config validated: {type(typed_config).__name__}")
        except ValidationError as e:
            logger.error(f"Action config validation failed: {e}")
            self._emit_action_event(
                action.type,
                action.id,
                False,
                {"error": f"Validation failed: {str(e)}"},
            )
            return False
        except ValueError as e:
            logger.warning(f"Unknown action type {action.type}, continuing without validation: {e}")
            typed_config = None

        # Extract action details for logging
        action_details = {"config": action.config}

        # Get pause settings from base config
        pause_before = 0
        pause_after = 0
        if action.base:
            pause_before = action.base.pause_before_begin or 0
            pause_after = action.base.pause_after_end or 0

        # Pause before action if specified
        if pause_before > 0:
            logger.debug(f"Waiting {pause_before}ms before action")
            self.time_wrapper.wait(pause_before / 1000.0)

        # Get retry count and continue_on_error from execution settings
        retry_count = 0
        continue_on_error = False
        if action.execution:
            retry_count = action.execution.retry_count or 0
            continue_on_error = action.execution.continue_on_error or False

        logger.debug(f"Retry config: retry_count={retry_count}, continue_on_error={continue_on_error}")

        # Retry logic: initial attempt + retry_count additional attempts on failure
        total_attempts = 1 + retry_count
        logger.debug(f"Starting retry loop: total_attempts={total_attempts}")

        for attempt in range(total_attempts):
            logger.debug(f"Retry attempt {attempt + 1}/{total_attempts}")
            try:
                # Delegate to specialized executor via registry
                result = self._delegate_to_executor(action, typed_config)
                logger.debug(f"Executor returned: {result}")

                if result:
                    # Add execution details if available
                    if isinstance(result, dict):
                        action_details.update(result)
                        logger.debug(f"Action returned details: {result}")
                        result = True

                    # Emit success event
                    event_data = {**action_details, "attempts": attempt + 1}
                    logger.debug(f"Emitting success event with data: {event_data}")
                    self._emit_action_event(
                        action.type, action.id, True, event_data
                    )

                    # Pause after action if specified
                    if pause_after > 0:
                        logger.debug(f"Waiting {pause_after}ms after action")
                        self.time_wrapper.wait(pause_after / 1000.0)

                    return True

                # Action failed, retry if attempts remain
                if attempt < total_attempts - 1:
                    logger.info(f"Action failed, retrying... (attempt {attempt + 2}/{total_attempts})")
                    self.time_wrapper.wait(1)

            except Exception as e:
                # Sanitize error message to remove unicode characters
                error_msg = str(e).encode("ascii", "replace").decode("ascii")
                logger.error(f"Error executing action: {error_msg}", exc_info=True)

                # Emit error event
                self._emit_action_event(
                    action.type,
                    action.id,
                    False,
                    {"error": error_msg, "attempts": attempt + 1},
                )

                # If this was the last attempt or we shouldn't continue on error, fail
                if attempt >= total_attempts - 1:
                    logger.error(f"All {total_attempts} attempts failed for action {action.id}")
                    return False

                # Otherwise, retry
                logger.info(f"Retrying after error... (attempt {attempt + 2}/{total_attempts})")
                self.time_wrapper.wait(1)

        # All attempts exhausted
        logger.error(f"Action {action.id} failed after {total_attempts} attempts")
        self._emit_action_event(
            action.type,
            action.id,
            False,
            {"error": "All retry attempts exhausted", "attempts": total_attempts},
        )
        return False

    def _delegate_to_executor(self, action: Action, typed_config: Any) -> bool:
        """Delegate action execution to the appropriate specialized executor.

        This is the core delegation method that:
        1. Looks up the executor for the action type via registry
        2. Syncs shared state to context
        3. Calls executor.execute()
        4. Syncs shared state back from context

        Args:
            action: Action to execute
            typed_config: Validated configuration object for the action

        Returns:
            bool: True if action succeeded, False otherwise

        Raises:
            ActionExecutionError: If no executor is registered for the action type
        """
        # Sync shared state to context before delegation
        self.context.last_find_location = self.last_find_location

        # Get executor for action type via registry
        logger.debug(f"Looking up executor for action type: {action.type}")
        executor = create_executor(action.type, self.context)

        # Delegate execution to specialized executor
        logger.debug(f"Delegating to {executor.__class__.__name__}")
        result = executor.execute(action, typed_config)

        # Sync shared state back from context
        self.last_find_location = self.context.last_find_location

        return result

    # Event emission methods (cross-cutting concern kept in delegating executor)

    def _emit_event(self, event_name: str, data: dict) -> None:
        """Emit event as JSON to stdout for Tauri to parse.

        Args:
            event_name: Name of the event
            data: Event data dictionary
        """
        import json

        event = {
            "type": "event",
            "event": event_name,
            "timestamp": self.time_wrapper.now().timestamp(),
            "sequence": 0,  # Can be managed by caller if needed
            "data": data,
        }
        print(json.dumps(event), flush=True)

    def _emit_image_recognition_event(self, data: dict) -> None:
        """Emit image recognition event.

        Args:
            data: Event data dictionary
        """
        self._emit_event("image_recognition", data)

    def _emit_action_event(
        self, action_type: str, action_id: str, success: bool, details: dict | None = None
    ) -> None:
        """Emit action execution event.

        Args:
            action_type: Type of action (e.g., "CLICK")
            action_id: Unique identifier for the action
            success: Whether the action succeeded
            details: Optional additional event details
        """
        data = {"action_type": action_type, "action_id": action_id, "success": success}
        if details:
            data.update(details)
        self._emit_event("action_execution", data)

    # Workflow execution methods (delegated from old executor)

    def execute_workflow(self, workflow_id: str, initial_context: dict | None = None) -> dict:
        """Execute a workflow by ID, using graph or sequential execution.

        Args:
            workflow_id: ID of the workflow to execute
            initial_context: Optional initial context variables

        Returns:
            Dictionary with execution results

        Raises:
            ValueError: If workflow not found or has invalid structure
        """
        workflow = self.config.workflow_map.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow '{workflow_id}' not found")

        logger.info(f"Executing workflow '{workflow.name}' (id={workflow_id})")

        # Check if workflow has connections (graph format)
        has_connections = hasattr(workflow, "connections") and workflow.connections

        if self.use_graph_execution and has_connections:
            # Use graph execution
            logger.info(f"Using GRAPH EXECUTION for workflow '{workflow.name}'")
            return self._execute_workflow_graph(workflow, initial_context)
        else:
            # Use sequential execution
            if self.use_graph_execution and not has_connections:
                logger.warning(
                    f"Graph execution requested but workflow '{workflow.name}' has no connections. "
                    f"Falling back to sequential execution."
                )
            logger.info(f"Using SEQUENTIAL EXECUTION for workflow '{workflow.name}'")
            return self._execute_workflow_sequential(workflow)

    def _execute_workflow_graph(
        self, workflow: Any, initial_context: dict | None = None
    ) -> dict:
        """Execute workflow using graph-based execution.

        Args:
            workflow: Workflow object with connections
            initial_context: Optional initial context variables

        Returns:
            Dictionary with execution results
        """
        from ..execution.graph_executor import GraphExecutor

        logger.info(f"Initializing GraphExecutor for workflow '{workflow.name}'")

        # Create graph executor
        graph_executor = GraphExecutor(workflow, self)

        # Execute with initial context
        result = graph_executor.execute(initial_context)

        logger.info(
            f"Graph execution completed for '{workflow.name}': "
            f"success={result.get('success')}, "
            f"actions_executed={len(result.get('summary', {}).get('execution_order', []))}"
        )

        return result

    def _execute_workflow_sequential(self, workflow: Any) -> dict:
        """Execute workflow using sequential execution.

        Args:
            workflow: Workflow object

        Returns:
            Dictionary with execution results
        """
        logger.info(f"Executing workflow '{workflow.name}' sequentially")

        results = {"success": True, "actions_executed": 0, "actions_failed": 0, "errors": []}

        for action in workflow.actions:
            try:
                success = self.execute_action(action)
                if success:
                    results["actions_executed"] += 1
                else:
                    results["actions_failed"] += 1
                    results["success"] = False

                    # Check if we should stop on failure
                    continue_on_error = (
                        action.execution.continue_on_error
                        if action.execution
                        else False
                    )
                    if not continue_on_error:
                        logger.error(
                            f"Stopping sequential execution due to failure in action '{action.id}'"
                        )
                        break
            except Exception as e:
                results["actions_failed"] += 1
                results["success"] = False
                results["errors"].append({"action_id": action.id, "error": str(e)})

                # Check if we should stop on error
                continue_on_error = (
                    action.execution.continue_on_error
                    if action.execution
                    else False
                )
                if not continue_on_error:
                    logger.error(
                        f"Stopping sequential execution due to error in action '{action.id}'"
                    )
                    break

        logger.info(
            f"Sequential execution completed for '{workflow.name}': "
            f"success={results['success']}, "
            f"executed={results['actions_executed']}, "
            f"failed={results['actions_failed']}"
        )

        return results
