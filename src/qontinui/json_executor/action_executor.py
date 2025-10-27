"""Executor for individual actions in the automation."""

import logging
import sys
from typing import Any

import cv2
import numpy as np
from pydantic import ValidationError

from ..actions.control_flow import ControlFlowExecutor
from ..actions.data_operations import DataOperationsExecutor, VariableContext
from ..config import (
    Action,
    ClickActionConfig,
    DragActionConfig,
    ExistsActionConfig,
    FindActionConfig,
    GoToStateActionConfig,
    KeyDownActionConfig,
    KeyPressActionConfig,
    KeyUpActionConfig,
    MouseDownActionConfig,
    MouseMoveActionConfig,
    MouseUpActionConfig,
    RunWorkflowActionConfig,
    ScreenshotActionConfig,
    ScrollActionConfig,
    TypeActionConfig,
    VanishActionConfig,
    WaitActionConfig,
    get_typed_config,
)
from ..wrappers import Keyboard, Mouse, Screen, TimeWrapper
from .config_parser import QontinuiConfig, Workflow
from .constants import DEFAULT_SIMILARITY_THRESHOLD

# Set up logger
logger = logging.getLogger(__name__)


class ActionExecutor:
    """Executes individual automation actions from JSON configuration.

    ActionExecutor translates high-level action definitions (CLICK, TYPE, FIND, etc.)
    into actual GUI interactions using the Hardware Abstraction Layer (HAL). It handles
    image recognition, coordinate calculation, retry logic, and event emission for monitoring.

    Supported action types:
        Mouse actions:
            - CLICK, DOUBLE_CLICK, RIGHT_CLICK - Click at coordinates or image
            - DRAG - Drag from one location to another
            - MOUSE_MOVE, MOVE - Move mouse to coordinates
            - MOUSE_DOWN, MOUSE_UP - Press/release mouse button
            - SCROLL, MOUSE_SCROLL - Scroll wheel

        Keyboard actions:
            - TYPE - Type text string
            - KEY_DOWN, KEY_UP - Press/release key
            - KEY_PRESS - Press and release key

        Vision actions:
            - FIND - Locate image on screen
            - EXISTS - Check if image exists
            - VANISH - Wait for image to disappear

        Navigation actions:
            - GO_TO_STATE - Navigate to target state via state machine
            - RUN_WORKFLOW - Execute a workflow by ID

        Utility actions:
            - WAIT - Pause execution
            - SCREENSHOT - Capture screen

    Attributes:
        config: Parsed automation configuration containing states, workflows, images.
        state_executor: Reference to StateExecutor for GO_TO_STATE navigation.
        defaults: Default action configuration from system settings.
        last_find_location: Coordinates (x, y) of most recent FIND result.

    Example:
        >>> config = ConfigParser().parse_file("automation.json")
        >>> executor = ActionExecutor(config)
        >>> action = Action(type="CLICK", config={"x": 100, "y": 200})
        >>> success = executor.execute_action(action)

    Note:
        This executor performs REAL GUI automation only. It requires an active
        display and performs actual mouse/keyboard actions.
    """

    def __init__(
        self,
        config: QontinuiConfig,
        state_executor=None,
        use_graph_execution: bool = False,
        hal=None,
    ):
        """Initialize ActionExecutor.

        Args:
            config: Parsed automation configuration
            state_executor: Optional state executor for GO_TO_STATE navigation
            use_graph_execution: Whether to use graph-based execution
            hal: Optional HAL container for dependency injection. If None,
                wrappers will use HALFactory (deprecated pattern).

        Example:
            # New pattern with dependency injection (recommended)
            >>> from qontinui.hal import initialize_hal
            >>> hal = initialize_hal()
            >>> executor = ActionExecutor(config, hal=hal)

            # Legacy pattern (deprecated)
            >>> executor = ActionExecutor(config)
        """
        self.config = config
        self.state_executor = state_executor
        self.use_graph_execution = use_graph_execution
        self.hal = hal

        # Track last find result for "Last Find Result" targeting
        self.last_find_location: tuple[int, int] | None = None
        # Get action defaults configuration
        self.defaults = self._create_defaults()

        # Initialize wrappers
        # Note: Wrappers currently use class methods with HALFactory for backward
        # compatibility. In the future, these should be instance-based with HAL container.
        self.time_wrapper = TimeWrapper()
        self.mouse_wrapper = Mouse()
        self.keyboard_wrapper = Keyboard()
        self.screen_wrapper = Screen()

        # Initialize variable context and executors for control flow and data operations
        self.variable_context = VariableContext()
        self.data_operations_executor = DataOperationsExecutor(self.variable_context)

        # Create action executor callback for control flow
        # This allows control flow actions to execute nested actions
        def action_executor_callback(action_id: str, variables: dict) -> dict:
            """Execute an action by ID with given variable context."""
            # Find the action in config (v2.0.0: workflows, v1.0.0: processes)
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

        logger.info(
            f"ActionExecutor initialized with control flow and data operations support "
            f"(graph_execution={'enabled' if use_graph_execution else 'disabled'})"
        )

    def execute_workflow(self, workflow_id: str, initial_context: dict | None = None) -> dict:
        """Execute a workflow by ID, using graph or sequential execution based on configuration.

        Supports Workflow format from schema.py.

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
        self, workflow: Workflow, initial_context: dict | None = None
    ) -> dict:
        """Execute workflow using graph-based execution.

        Supports both v2.0.0 Workflow and v1.0.0 Process formats.

        Args:
            workflow: Workflow/Process object with connections
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

    def _execute_workflow_sequential(self, workflow: Workflow) -> dict:
        """Execute workflow using sequential execution.

        Args:
            workflow: Workflow/Process object

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

    def _create_defaults(self):
        """Create default configuration for actions.

        Returns a simple namespace with mouse and keyboard defaults.
        """
        from types import SimpleNamespace

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

        Handles the complete lifecycle of action execution including pre-action pauses,
        the main action execution, retries on failure, post-action pauses, and event
        emission for real-time monitoring.

        The new Action model from schema.py includes an optional position field for graph-based
        workflow visualization. This field is not used during execution and is safely ignored.

        Args:
            action: Action object containing type, configuration, and execution parameters.
                Can be either config format or new Pydantic format.

        Returns:
            bool: True if action executed successfully within retry attempts,
                False if all attempts failed.

        Raises:
            ValidationError: If action configuration is invalid
            TimeoutError: If action exceeds configured timeout limit

        Example:
            >>> action = Action(
            ...     id="click_1",
            ...     type="CLICK",
            ...     config={"target": {"type": "coordinates", "coordinates": {"x": 100, "y": 200}}},
            ...     execution={"retryCount": 3, "continueOnError": False}
            ... )
            >>> success = executor.execute_action(action)
        """
        # MARKER: This is the execute_action method - EDITED VERSION 2025-01-26
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        print(f"[DEBUG] ========== execute_action CALLED ==========", file=sys.stderr, flush=True)
        print(f"[DEBUG] Action: {action.type} (ID: {action.id})", file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        # CRITICAL DEBUG: Log immediately after first debug print
        print(f"[DEBUG-LINE-331] After first debug print, about to call logger.info", file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        # Wrap logger.info in try-except to catch any silent failures
        try:
            print(f"[DEBUG-LINE-332] BEFORE logger.info call", file=sys.stderr, flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            logger.info(f"Executing action: {action.type} (ID: {action.id})")
            print(f"[DEBUG-LINE-332] AFTER logger.info call", file=sys.stderr, flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception as e:
            print(f"[DEBUG-ERROR] logger.info FAILED: {e}", file=sys.stderr, flush=True)
            sys.stdout.flush()
            sys.stderr.flush()

        # Validate action configuration using Pydantic schemas
        print(f"[DEBUG-LINE-335] About to initialize typed_config variable", file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        typed_config = None

        print(f"[DEBUG-LINE-336] typed_config initialized to None", file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        try:
            print(f"[DEBUG-LINE-337] BEFORE get_typed_config call", file=sys.stderr, flush=True)
            print(f"[DEBUG-LINE-337] action.type={action.type}, action.config={action.config}", file=sys.stderr, flush=True)
            sys.stdout.flush()
            sys.stderr.flush()

            typed_config = get_typed_config(action)

            print(f"[DEBUG-LINE-337] AFTER get_typed_config call", file=sys.stderr, flush=True)
            print(f"[DEBUG-LINE-337] typed_config type: {type(typed_config).__name__}", file=sys.stderr, flush=True)
            sys.stdout.flush()
            sys.stderr.flush()

            print(f"[DEBUG-LINE-338] BEFORE logger.debug call", file=sys.stderr, flush=True)
            sys.stdout.flush()
            sys.stderr.flush()

            logger.debug(f"Action config validated: {type(typed_config).__name__}")

            print(f"[DEBUG-LINE-338] AFTER logger.debug call", file=sys.stderr, flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
        except ValidationError as e:
            print(f"[DEBUG-LINE-340] ValidationError caught: {e}", file=sys.stderr, flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            logger.error(f"Action config validation failed: {e}")
            self._emit_action_event(
                action.type,
                action.id,
                False,
                {"error": f"Validation failed: {str(e)}"},
            )
            return False
        except ValueError as e:
            print(f"[DEBUG-LINE-349] ValueError caught: {e}", file=sys.stderr, flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            logger.warning(f"Unknown action type {action.type}, continuing without validation: {e}")
            typed_config = None

        # Extract action details for logging
        print(f"[DEBUG-LINE-353] BEFORE creating action_details dict", file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        try:
            print(f"[DEBUG-LINE-353] Accessing action.config property", file=sys.stderr, flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            action_details = {"config": action.config}
            print(f"[DEBUG-LINE-353] action_details created successfully", file=sys.stderr, flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception as e:
            print(f"[DEBUG-ERROR] Failed to create action_details: {e}", file=sys.stderr, flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            action_details = {"config": {}}

        # Get pause settings from base config
        print(f"[DEBUG-LINE-356] Getting pause settings from base config", file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        pause_before = 0
        pause_after = 0
        try:
            print(f"[DEBUG-LINE-358] Accessing action.base property", file=sys.stderr, flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            if action.base:
                print(f"[DEBUG-LINE-359] action.base exists, accessing pause_before_begin", file=sys.stderr, flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
                pause_before = action.base.pause_before_begin or 0
                print(f"[DEBUG-LINE-360] pause_before={pause_before}, accessing pause_after_end", file=sys.stderr, flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
                pause_after = action.base.pause_after_end or 0
                print(f"[DEBUG-LINE-360] pause_after={pause_after}", file=sys.stderr, flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
            else:
                print(f"[DEBUG-LINE-358] action.base is None", file=sys.stderr, flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
        except Exception as e:
            print(f"[DEBUG-ERROR] Failed to get pause settings: {e}", file=sys.stderr, flush=True)
            sys.stdout.flush()
            sys.stderr.flush()

        # Pause before action if specified
        print(f"[DEBUG-LINE-363] Checking if pause_before > 0", file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        if pause_before > 0:
            print(f"[PAUSE] Waiting {pause_before}ms before action")
            try:
                print(f"[DEBUG-LINE-365] Calling time_wrapper.wait({pause_before / 1000.0})", file=sys.stderr, flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
                self.time_wrapper.wait(pause_before / 1000.0)
                print(f"[DEBUG-LINE-365] time_wrapper.wait completed", file=sys.stderr, flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception as e:
                print(f"[DEBUG-ERROR] time_wrapper.wait failed: {e}", file=sys.stderr, flush=True)
                sys.stdout.flush()
                sys.stderr.flush()

        # Get retry count and continue_on_error from execution settings
        print(f"[DEBUG-LINE-368] Getting retry settings", file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        retry_count = 0
        continue_on_error = False
        try:
            print(f"[DEBUG-LINE-370] Accessing action.execution property", file=sys.stderr, flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            if action.execution:
                print(f"[DEBUG-LINE-371] action.execution exists, accessing retry_count", file=sys.stderr, flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
                retry_count = action.execution.retry_count or 0
                print(f"[DEBUG-LINE-371] retry_count={retry_count}, accessing continue_on_error", file=sys.stderr, flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
                continue_on_error = action.execution.continue_on_error or False
                print(f"[DEBUG-LINE-372] continue_on_error={continue_on_error}", file=sys.stderr, flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
            else:
                print(f"[DEBUG-LINE-370] action.execution is None", file=sys.stderr, flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
        except Exception as e:
            print(f"[DEBUG-ERROR] Failed to get retry settings: {e}", file=sys.stderr, flush=True)
            sys.stdout.flush()
            sys.stderr.flush()

        print(f"[DEBUG-LINE-374] BEFORE logger.debug call", file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        print(f"[DEBUG-LINE-550] About to call logger.debug with retry_count={retry_count}, continue_on_error={continue_on_error}", file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        logger.debug(f"Retry config: retry_count={retry_count}, continue_on_error={continue_on_error}")
        print(f"[DEBUG-LINE-550] logger.debug completed", file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        print(f"[DEBUG-LINE-374] AFTER logger.debug call", file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        # Retry logic: initial attempt + retry_count additional attempts on failure
        print(f"[DEBUG-LINE-557] BEFORE total_attempts calculation, retry_count={retry_count}", file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        total_attempts = 1 + retry_count
        print(f"[DEBUG-LINE-558] AFTER total_attempts calculation, total_attempts={total_attempts}", file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        print(f"[DEBUG] Starting retry loop: total_attempts={total_attempts}")

        for attempt in range(total_attempts):
            print(f"[DEBUG] Retry attempt {attempt + 1}/{total_attempts}")
            try:
                print(f"[DEBUG] About to call _execute_action_type for {action.type}")
                result = self._execute_action_type(action, typed_config)
                print(f"[DEBUG] _execute_action_type returned: {result}")
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
                        print(f"[PAUSE] Waiting {pause_after}ms after action")
                        self.time_wrapper.wait(pause_after / 1000.0)
                        print(f"[PAUSE] Completed waiting {pause_after}ms")

                    return True

                if attempt < total_attempts - 1:
                    print(f"Action failed, retrying... (attempt {attempt + 2}/{total_attempts})")
                    self.time_wrapper.wait(1)

            except Exception as e:
                # Sanitize error message to remove unicode characters
                error_msg = str(e).encode("ascii", "replace").decode("ascii")
                logger.error(f"Error executing action: {error_msg}", exc_info=True)
                print(f"Error executing action: {error_msg}")
                # Emit error event
                self._emit_action_event(
                    action.type,
                    action.id,
                    False,
                    {**action_details, "error": error_msg, "attempts": attempt + 1},
                )
                if not continue_on_error and attempt == total_attempts - 1:
                    raise

        # Emit failure event after all retries
        logger.warning(f"Action {action.type} failed after {total_attempts} attempts")
        self._emit_action_event(
            action.type,
            action.id,
            False,
            {**action_details, "attempts": total_attempts, "reason": "All retries failed"},
        )
        # Always return False for failed actions - continue_on_error only controls exception raising
        return False

    def _execute_action_type(self, action: Action, typed_config: Any = None) -> bool:
        """Execute specific action type with validated config.

        Args:
            action: Pydantic Action model
            typed_config: Type-specific config (e.g., ClickActionConfig) or None

        Returns:
            bool: True if action succeeded, False otherwise
        """
        print(f"[DEBUG] _execute_action_type called with action.type={action.type}")
        logger.debug(f"Executing action type: {action.type}")

        action_map = {
            # Pure mouse actions
            "MOUSE_MOVE": self._execute_mouse_move,
            "MOUSE_DOWN": self._execute_mouse_down,
            "MOUSE_UP": self._execute_mouse_up,
            "MOUSE_SCROLL": self._execute_scroll,
            # Pure keyboard actions
            "KEY_DOWN": self._execute_key_down,
            "KEY_UP": self._execute_key_up,
            "KEY_PRESS": self._execute_key_press,
            # Combined mouse actions
            "MOVE": self._execute_mouse_move,  # Alias for MOUSE_MOVE
            "CLICK": self._execute_click,
            "DOUBLE_CLICK": self._execute_double_click,
            "RIGHT_CLICK": self._execute_right_click,
            "DRAG": self._execute_drag,
            "SCROLL": self._execute_scroll,
            # Combined keyboard actions
            "TYPE": self._execute_type,
            # Other actions
            "FIND": self._execute_find,
            "WAIT": self._execute_wait,
            "VANISH": self._execute_vanish,
            "EXISTS": self._execute_exists,
            "SCREENSHOT": self._execute_screenshot,
            "GO_TO_STATE": self._execute_go_to_state,
            "RUN_WORKFLOW": self._execute_run_workflow,
            # Control flow actions
            "LOOP": self._execute_loop,
            "IF": self._execute_if,
            "BREAK": self._execute_break,
            "CONTINUE": self._execute_continue,
            # Data operation actions
            "SET_VARIABLE": self._execute_set_variable,
            "GET_VARIABLE": self._execute_get_variable,
            "MAP": self._execute_map,
            "REDUCE": self._execute_reduce,
            "SORT": self._execute_sort,
            "FILTER": self._execute_filter,
            "STRING_OPERATION": self._execute_string_operation,
            "MATH_OPERATION": self._execute_math_operation,
        }

        print(f"[DEBUG] Looking up handler for action type: {action.type}")
        handler = action_map.get(action.type)
        print(f"[DEBUG] Handler found: {handler is not None}")

        if handler:
            print(f"[DEBUG] Calling handler for {action.type}...", file=sys.stderr, flush=True)
            # Pass typed_config if available
            if typed_config is not None:
                print(f"[DEBUG-HANDLER-698] About to call handler with typed_config", file=sys.stderr, flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
                result = handler(action, typed_config)
                print(f"[DEBUG-HANDLER-700] Handler returned: {result}", file=sys.stderr, flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
                return result
            else:
                print(f"[DEBUG] Calling handler without typed_config", file=sys.stderr, flush=True)
                result = handler(action)
                print(f"[DEBUG] Handler returned: {result}", file=sys.stderr, flush=True)
                return result
        else:
            logger.error(f"Unknown action type: {action.type}")
            print(f"Unknown action type: {action.type}")
            return False

    def _get_target_location_from_typed(self, target_config: Any) -> tuple[int, int] | None:
        """Get target location from typed TargetConfig (Pydantic model).

        Args:
            target_config: TargetConfig union type (ImageTarget, RegionTarget, CoordinatesTarget, etc.)

        Returns:
            Tuple of (x, y) coordinates or None if not found
        """
        from ..config import CoordinatesTarget, ImageTarget, RegionTarget

        logger.debug(f"Getting location from typed target: {type(target_config).__name__}")

        # Handle different target types
        if isinstance(target_config, ImageTarget):
            # Find image on screen
            image_id = target_config.image_id
            similarity = DEFAULT_SIMILARITY_THRESHOLD

            # Get similarity from search options if available
            if target_config.search_options and target_config.search_options.similarity is not None:
                similarity = target_config.search_options.similarity

            logger.debug(f"Finding image {image_id} with similarity {similarity}")

            if image_id:
                image = self.config.image_map.get(image_id)
                if image and image.file_path:
                    return self._find_image_on_screen(image.file_path, similarity)
                else:
                    if not image:
                        logger.error(f"Image ID not found in image_map: {image_id}")
                    else:
                        logger.error(f"Image file_path is None for image: {image_id}")

        elif isinstance(target_config, CoordinatesTarget):
            coords = target_config.coordinates
            logger.debug(f"Using coordinates: ({coords.x}, {coords.y})")
            return (coords.x, coords.y)

        elif isinstance(target_config, RegionTarget):
            region = target_config.region
            # Return center of region
            x = region.x + region.width // 2
            y = region.y + region.height // 2
            logger.debug(f"Using region center: ({x}, {y})")
            return (x, y)

        elif isinstance(target_config, str) and target_config == "Last Find Result":
            # Handle string target for backward compatibility
            if self.last_find_location:
                logger.debug(f"Using Last Find Result: {self.last_find_location}")
                return self.last_find_location
            else:
                logger.error("Last Find Result requested but no previous find result available")

        logger.error(f"Unsupported target type: {type(target_config)}")
        return None

    def _get_target_location(self, config: dict[str, Any]) -> tuple[int, int] | None:
        """DEPRECATED: Get target location from legacy action config dict.

        This method exists only for backward compatibility and should not be used.
        Use _get_target_location_from_typed() with typed configs instead.
        """
        raise NotImplementedError(
            "Legacy dict-based config is no longer supported. "
            "All actions must use typed Pydantic configs via get_typed_config()."
        )

    def _get_target_location_legacy(self, config: dict[str, Any]) -> tuple[int, int] | None:
        """Legacy implementation - kept for reference only, not to be used."""
        target = config.get("target", {})

        # Handle "Last Find Result" string target
        if isinstance(target, str) and target == "Last Find Result":
            if self.last_find_location:
                print(f"[TARGET] Using Last Find Result: {self.last_find_location}")
                return self.last_find_location
            else:
                print("[ERROR] Last Find Result requested but no previous find result available")
                return None

        if target.get("type") == "image":
            # Find image on screen
            image_id = target.get("imageId")

            # Similarity precedence (lowest to highest):
            # 1. Global default (from constants.py)
            # 2. Target threshold (from StateImage/Pattern)
            # 3. Action similarity (from action options)
            threshold = DEFAULT_SIMILARITY_THRESHOLD
            if target.get("threshold") is not None:
                threshold = target.get("threshold")  # StateImage/Pattern similarity
            if config.get("similarity") is not None:
                threshold = config.get("similarity")  # Action options similarity (highest priority)

            if image_id:
                image = self.config.image_map.get(image_id)
                if image and image.file_path:
                    return self._find_image_on_screen(image.file_path, threshold)
                else:
                    if not image:
                        print(f"[ERROR] Image ID not found in image_map: {image_id}")
                        print(f"   Available images: {list(self.config.image_map.keys())}")
                    else:
                        print(f"[ERROR] Image file_path is None for image: {image_id}")

        elif target.get("type") == "coordinates":
            coords = target.get("coordinates", {})
            return (coords.get("x", 0), coords.get("y", 0))

        elif target.get("type") == "region":
            region = target.get("region", {})
            # Return center of region
            x = region.get("x", 0) + region.get("width", 0) // 2
            y = region.get("y", 0) + region.get("height", 0) // 2
            return (x, y)

        return None

    def _emit_event(self, event_name: str, data: dict):
        """Emit event as JSON to stdout for Tauri to parse."""
        import json

        event = {
            "type": "event",
            "event": event_name,
            "timestamp": self.time_wrapper.now().timestamp(),
            "sequence": 0,  # Can be managed by caller if needed
            "data": data,
        }
        print(json.dumps(event), flush=True)

    def _emit_image_recognition_event(self, data: dict):
        """Emit image recognition event."""
        self._emit_event("image_recognition", data)

    def _emit_action_event(
        self, action_type: str, action_id: str, success: bool, details: dict | None = None
    ):
        """Emit action execution event."""
        data = {"action_type": action_type, "action_id": action_id, "success": success}
        if details:
            data.update(details)
        self._emit_event("action_execution", data)

    def _find_image_on_screen(
        self, image_path: str, threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> tuple[int, int] | None:
        """Find image on screen using template matching."""
        try:
            # Take screenshot using wrapper
            screenshot = Screen.capture()
            screenshot_np = np.array(screenshot)
            screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)

            # Load template image
            template = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                error_msg = f"Failed to load template image: {image_path}"
                print(f"[ERROR] {error_msg}")
                self._emit_image_recognition_event(
                    {
                        "image_path": image_path,
                        "template_size": "unknown",
                        "screenshot_size": f"{screenshot_gray.shape[1]}x{screenshot_gray.shape[0]}",
                        "threshold": threshold,
                        "confidence": 0,
                        "found": False,
                        "error": error_msg,
                    }
                )
                return None

            template_size = f"{template.shape[1]}x{template.shape[0]}"
            screenshot_size = f"{screenshot_gray.shape[1]}x{screenshot_gray.shape[0]}"

            # Debug info
            print("[IMG] Image Recognition Debug:")
            print(f"   Template: {image_path}")
            print(f"   Template size: {template_size}")
            print(f"   Screenshot size: {screenshot_size}")
            print(f"   Threshold: {threshold:.2f}")

            # Template matching
            result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            print(f"   Confidence: {max_val:.4f} (need >= {threshold:.2f})")

            if max_val >= threshold:
                # Return center of found image
                h, w = template.shape
                center_x = max_loc[0] + w // 2
                center_y = max_loc[1] + h // 2
                location_str = f"({center_x}, {center_y})"
                print(f"   [OK] Found at {location_str}")

                # Store as last find location for "Last Find Result" targeting
                self.last_find_location = (center_x, center_y)

                # Emit success event
                self._emit_image_recognition_event(
                    {
                        "image_path": image_path,
                        "template_size": template_size,
                        "screenshot_size": screenshot_size,
                        "threshold": threshold,
                        "confidence": max_val,
                        "found": True,
                        "location": location_str,
                    }
                )
                return (center_x, center_y)
            else:
                # Calculate how close we were
                gap = threshold - max_val
                percent_off = (gap / threshold) * 100
                best_match_str = f"({max_loc[0]}, {max_loc[1]})"
                print(f"   [FAIL] Not found (missed by {gap:.4f} / {percent_off:.1f}%)")
                print(f"   Best match location: {best_match_str}")

                # Emit failure event
                self._emit_image_recognition_event(
                    {
                        "image_path": image_path,
                        "template_size": template_size,
                        "screenshot_size": screenshot_size,
                        "threshold": threshold,
                        "confidence": max_val,
                        "found": False,
                        "gap": gap,
                        "percent_off": percent_off,
                        "best_match_location": best_match_str,
                    }
                )
                return None

        except Exception as e:
            error_msg = f"Error finding image: {e}"
            print(f"[ERROR] {error_msg}")
            # Skip traceback printing to avoid unicode encoding issues
            # import traceback
            # traceback.print_exc()

            # Emit error event
            self._emit_image_recognition_event(
                {
                    "image_path": image_path,
                    "template_size": "unknown",
                    "screenshot_size": "unknown",
                    "threshold": threshold,
                    "confidence": 0,
                    "found": False,
                    "error": error_msg,
                }
            )
            return None

    def _execute_find(self, action: Action, typed_config: FindActionConfig) -> bool:
        """Execute FIND action with type-safe config.

        Args:
            action: Pydantic Action model
            typed_config: Pre-validated FindActionConfig (required)

        Returns:
            bool: True if target was found
        """
        logger.debug("Executing FIND action")
        location = self._get_target_location_from_typed(typed_config.target)

        if location:
            logger.info(f"FIND action succeeded: found at {location}")
            return True
        else:
            logger.warning("FIND action failed: target not found")
            return False

    def _execute_click(self, action: Action, typed_config: ClickActionConfig) -> bool:
        """Execute CLICK action with type-safe config.

        This is a COMBINED action that orchestrates pure actions.
        Uses Pydantic ClickActionConfig for type-safe access to configuration.

        Args:
            action: Pydantic Action model
            typed_config: Pre-validated ClickActionConfig (required)

        Returns:
            bool: True if click succeeded
        """
        from ..hal.interfaces import MouseButton

        logger.debug("Executing CLICK action")

        # Check if target is None or currentPosition (pure action)
        location = None
        if typed_config.target and typed_config.target.type != "currentPosition":
            location = self._get_target_location_from_typed(typed_config.target)
            if not location:
                logger.error("Failed to get target location from typed config")
                return False
        else:
            logger.debug("No target specified - clicking at current position (pure action)")

        # Get button from typed config
        button = MouseButton.LEFT
        button_name = "LEFT"
        if typed_config.mouse_button:
            button_map = {
                "LEFT": MouseButton.LEFT,
                "RIGHT": MouseButton.RIGHT,
                "MIDDLE": MouseButton.MIDDLE,
            }
            button = button_map.get(typed_config.mouse_button.value, MouseButton.LEFT)
            button_name = typed_config.mouse_button.value

        # Get click count
        click_count = typed_config.number_of_clicks or 1

        # Get timing values
        hold_duration = (
            typed_config.press_duration or self.defaults.mouse.click_hold_duration
        ) / 1000.0
        release_delay = (
            typed_config.pause_after_release or self.defaults.mouse.click_release_delay
        )

        # Safety settings (not yet in typed config, use defaults)
        safety_release = self.defaults.mouse.click_safety_release
        safety_delay = self.defaults.mouse.safety_release_delay

        # Get current mouse position for debugging
        current_pos = Mouse.position()
        print(f"[COMBINED ACTION: CLICK] Current position: ({current_pos.x}, {current_pos.y})")
        print(
            f"[COMBINED ACTION: CLICK] Target: {location}, Button: {button_name}, Count: {click_count}"
        )
        print(f"[COMBINED ACTION: CLICK] Timing: hold={hold_duration}s, release={release_delay}s")

        # Step 1: Safety - Release all buttons first (if enabled)
        if safety_release:
            print("[COMBINED ACTION: CLICK] Step 1: Release all mouse buttons")
            Mouse.up(button=MouseButton.LEFT)
            Mouse.up(button=MouseButton.RIGHT)
            Mouse.up(button=MouseButton.MIDDLE)
            self.time_wrapper.wait(safety_delay)

        # Step 2: Perform clicks at current position (mouse should already be positioned by MOVE)
        print(
            f"[COMBINED ACTION: CLICK] Step 2: Perform {click_count} click(s) at current position"
        )
        for i in range(click_count):
            print(f"[COMBINED ACTION: CLICK] Click {i+1}/{click_count}")

            # Sub-action: MOUSE_DOWN
            Mouse.down(button=button)
            self.time_wrapper.wait(hold_duration)

            # Sub-action: MOUSE_UP
            Mouse.up(button=button)

            if i < click_count - 1:
                self.time_wrapper.wait(release_delay)

        # Step 3: Final safety release
        self.time_wrapper.wait(release_delay)
        Mouse.up(button=button)
        print("[COMBINED ACTION: CLICK] Step 3: Final safety release")

        print(f"[COMBINED ACTION: CLICK] Completed {click_count} {button_name}-click(s)")
        return True

    def _execute_double_click(
        self, action: Action, typed_config: ClickActionConfig | None = None
    ) -> bool:
        """Execute DOUBLE_CLICK action - convenience wrapper for CLICK with clickCount=2.

        This is a convenience action that delegates to CLICK with clickCount=2.
        It provides a simpler, more explicit action type for the common case of double-clicking.

        For advanced scenarios (e.g., triple-click, right double-click), use CLICK with clickCount
        and optionally clickType parameters.
        """
        print("[COMBINED ACTION: DOUBLE_CLICK] Delegating to CLICK with clickCount=2")

        # Create new config with clickCount=2, preserving all other settings
        # Note: double_click_interval maps to click_release_delay in CLICK
        double_click_config = {**action.config, "clickCount": 2}

        # If user specified double_click_interval, map it to click_release_delay
        if "double_click_interval" in action.config:
            double_click_config["click_release_delay"] = action.config["double_click_interval"]

        # Create new action with CLICK type but preserve all timing overrides
        click_action = Action(id=action.id, type="CLICK", config=double_click_config)

        # Delegate to CLICK implementation
        return self._execute_click(click_action)

    def _execute_right_click(
        self, action: Action, typed_config: ClickActionConfig | None = None
    ) -> bool:
        """Execute RIGHT_CLICK action - convenience wrapper for CLICK with clickType=right.

        This is a convenience action that delegates to CLICK with clickType="right".
        It provides a simpler, more explicit action type for the common case of right-clicking.

        For advanced scenarios (e.g., right double-click), use CLICK with both clickType and clickCount.
        """
        print("[COMBINED ACTION: RIGHT_CLICK] Delegating to CLICK with clickType=right")

        # Create new config with clickType=right, preserving all other settings
        right_click_config = {**action.config, "clickType": "right"}

        # Create new action with CLICK type but preserve all timing overrides
        click_action = Action(id=action.id, type="CLICK", config=right_click_config)

        # Delegate to CLICK implementation
        return self._execute_click(click_action)

    def _execute_type(self, action: Action, typed_config: TypeActionConfig = None) -> bool:
        """Execute TYPE action with type-safe config.

        Args:
            action: Pydantic Action model
            typed_config: Pre-validated TypeActionConfig

        Returns:
            bool: True if text was typed successfully
        """
        print(f"[DEBUG-TYPE-1171] Executing TYPE action", file=sys.stderr, flush=True)
        logger.debug("Executing TYPE action")

        print(f"[DEBUG-TYPE-1173] Checking if typed_config exists", file=sys.stderr, flush=True)
        if not typed_config:
            logger.error("TYPE action requires valid TypeActionConfig")
            return False

        print(f"[DEBUG-TYPE-1177] Initializing text variable", file=sys.stderr, flush=True)
        text = ""

        # Get text directly or from text_source
        print(f"[DEBUG-TYPE-1180] Checking typed_config.text", file=sys.stderr, flush=True)
        if typed_config.text:
            text = typed_config.text
            logger.debug(f"Using direct text: '{text}'")
        elif typed_config.text_source:
            print(f"[DEBUG-TYPE-1191] Processing text_source", file=sys.stderr, flush=True)
            # Get text from state string
            state_id = typed_config.text_source.state_id
            string_ids = typed_config.text_source.string_ids

            print(f"[DEBUG-TYPE-1195] state_id={state_id}, string_ids={string_ids}", file=sys.stderr, flush=True)
            logger.debug(f"Looking for state string: state_id={state_id}, string_ids={string_ids}")

            print(f"[DEBUG-TYPE-1198] Checking if state exists in state_map", file=sys.stderr, flush=True)
            if state_id and string_ids and state_id in self.config.state_map:
                print(f"[DEBUG-TYPE-1199] Getting state from state_map", file=sys.stderr, flush=True)
                state = self.config.state_map[state_id]
                print(f"[DEBUG-TYPE-1200] Accessing state.state_strings", file=sys.stderr, flush=True)
                state_strings = state.state_strings
                print(f"[DEBUG-TYPE-1201] Got state_strings, count={len(state_strings)}", file=sys.stderr, flush=True)
                logger.debug(
                    f"State strings in '{state_id}': {[(s.id, s.value) for s in state_strings]}"
                )

                # Find the string in the state
                print(f"[DEBUG-TYPE-1206] Iterating through state_strings", file=sys.stderr, flush=True)
                for state_string in state_strings:
                    if state_string.id in string_ids:
                        text = state_string.value
                        logger.debug(f"Found matching string: '{text}'")
                        break

                if not text:
                    logger.error(f"No matching string found for IDs: {string_ids}")
            else:
                logger.error(f"State '{state_id}' not found or no string IDs provided")

        # Type the text if we have it
        print(f"[DEBUG-TYPE-1218] Checking if text exists, text='{text}'", file=sys.stderr, flush=True)
        if text:
            Keyboard.type(text)
            logger.info(f"Successfully typed: '{text}'")
            print(f"[TYPE] Successfully typed: '{text}'")
            return True

        logger.error("TYPE action failed - no text to type")
        print("[ERROR] TYPE action failed - no text to type")
        return False

    def _execute_key_down(self, action: Action, typed_config: KeyDownActionConfig = None) -> bool:
        """Execute KEY_DOWN action (pure) - press and hold key.

        This is a PURE action that only presses the key down.
        The key remains pressed until KEY_UP is called.
        """
        key = action.config.get("key")
        if not key:
            print("[ERROR] KEY_DOWN requires 'key' parameter")
            return False

        Keyboard.down(key)
        print(f"[PURE ACTION] Key '{key}' pressed down")
        return True

    def _execute_key_up(self, action: Action, typed_config: KeyUpActionConfig = None) -> bool:
        """Execute KEY_UP action (pure) - release key.

        This is a PURE action that only releases the key.
        """
        key = action.config.get("key")
        if not key:
            print("[ERROR] KEY_UP requires 'key' parameter")
            return False

        Keyboard.up(key)
        print(f"[PURE ACTION] Key '{key}' released")
        return True

    def _execute_key_press(self, action: Action, typed_config: KeyPressActionConfig = None) -> bool:
        """Execute KEY_PRESS action (pure) - press and release key.

        This is a PURE action that presses and immediately releases a key.
        Equivalent to KEY_DOWN + KEY_UP.
        """
        keys = action.config.get("keys", [])
        if not keys and "key" in action.config:
            keys = [action.config["key"]]

        for key in keys:
            Keyboard.press(key)
            print(f"[PURE ACTION] Key '{key}' pressed and released")
        return True

    def _execute_drag(self, action: Action, typed_config: DragActionConfig) -> bool:
        """Execute DRAG action (combined) - MOUSE_MOVE + MOUSE_DOWN + MOUSE_MOVE + MOUSE_UP.

        This is a COMBINED action using pure actions:
        1. Move to start position
        2. Press button
        3. Move to end position (with button held)
        4. Release button

        Timing values can be overridden in action config or set via defaults.
        """
        from ..hal.interfaces import MouseButton

        start = self._get_target_location_from_typed(typed_config.source)
        if not start:
            logger.error("Failed to get start location for DRAG")
            return False

        # Get destination - can be TargetConfig, Coordinates, or Region
        end = None
        if isinstance(typed_config.destination, dict):
            # It's a Coordinates or Region
            end_x = typed_config.destination.get("x", 0)
            end_y = typed_config.destination.get("y", 0)
            end = (end_x, end_y)
        else:
            # It's a TargetConfig - get location
            end = self._get_target_location_from_typed(typed_config.destination)

        if not end:
            logger.error("Failed to get destination location for DRAG")
            return False

        # Get timing values from typed config or defaults
        duration = (
            typed_config.drag_duration or self.defaults.mouse.drag_default_duration * 1000
        ) / 1000.0
        start_delay = typed_config.delay_before_move or self.defaults.mouse.drag_start_delay
        end_delay = typed_config.delay_after_drag or self.defaults.mouse.drag_end_delay

        print(f"[COMBINED ACTION: DRAG] From {start} to {end}")
        print(
            f"[COMBINED ACTION: DRAG] Timing: duration={duration}s, start_delay={start_delay}s, end_delay={end_delay}s"
        )

        # Step 1: Move to start position
        print("[COMBINED ACTION: DRAG] Step 1: Move to start")
        Mouse.move(start[0], start[1], 0)

        # Step 2: Press button
        self.time_wrapper.wait(start_delay)
        print("[COMBINED ACTION: DRAG] Step 2: Press left button")
        Mouse.down(button=MouseButton.LEFT)

        # Step 3: Move to end position (dragging)
        self.time_wrapper.wait(start_delay)
        print("[COMBINED ACTION: DRAG] Step 3: Move to end (dragging)")
        Mouse.move(end[0], end[1], duration)

        # Step 4: Release button
        self.time_wrapper.wait(end_delay)
        print("[COMBINED ACTION: DRAG] Step 4: Release left button")
        Mouse.up(button=MouseButton.LEFT)

        print("[COMBINED ACTION: DRAG] Completed")
        return True

    def _execute_scroll(self, action: Action, typed_config: ScrollActionConfig = None) -> bool:
        """Execute SCROLL action."""
        direction = action.config.get("direction", "down")
        distance = action.config.get("distance", 3)

        # Scroll amount (positive for up, negative for down)
        scroll_amount = distance if direction == "up" else -distance

        location = self._get_target_location(action.config)
        if location:
            Mouse.move(location[0], location[1])

        Mouse.scroll(scroll_amount)
        print(f"Scrolled {direction} by {distance}")
        return True

    def _execute_wait(self, action: Action, typed_config: WaitActionConfig = None) -> bool:
        """Execute WAIT action."""
        duration = action.config.get("duration", 1000)
        self.time_wrapper.wait(duration / 1000.0)
        print(f"Waited {duration}ms")
        return True

    def _execute_vanish(self, action: Action, typed_config: VanishActionConfig = None) -> bool:
        """Execute VANISH action - wait for element to disappear."""
        timeout = action.timeout / 1000.0
        start_time = self.time_wrapper.now()

        elapsed = 0.0
        while elapsed < timeout:
            location = self._get_target_location(action.config)
            if location is None:
                print("Element vanished")
                return True
            self.time_wrapper.wait(0.5)
            elapsed = (self.time_wrapper.now() - start_time).total_seconds()

        print("Element did not vanish within timeout")
        return False

    def _execute_exists(self, action: Action, typed_config: ExistsActionConfig = None) -> bool:
        """Execute EXISTS action - check if element exists."""
        location = self._get_target_location(action.config)
        exists = location is not None
        print(f"Element exists: {exists}")
        return exists

    def _execute_mouse_move(
        self, action: Action, typed_config: MouseMoveActionConfig = None
    ) -> bool:
        """Execute MOUSE_MOVE action (pure) - move mouse to position.

        This is a PURE action that only moves the mouse cursor.
        """
        location = self._get_target_location(action.config)
        if location:
            # Get duration from config (in milliseconds, convert to seconds)
            duration_ms = action.config.get("duration", 0)
            duration_seconds = duration_ms / 1000.0 if duration_ms else 0.0

            Mouse.move(location[0], location[1], duration_seconds)
            print(f"[PURE ACTION] Mouse moved to {location} (duration: {duration_ms}ms)")
            return True
        return False

    def _execute_mouse_down(
        self, action: Action, typed_config: MouseDownActionConfig = None
    ) -> bool:
        """Execute MOUSE_DOWN action (pure) - press and hold mouse button.

        This is a PURE action that only presses the mouse button.
        The button remains pressed until MOUSE_UP is called.
        """
        from ..hal.interfaces import MouseButton

        # Get button type from config
        button_type = action.config.get("button", "left").lower()
        button_map = {
            "left": MouseButton.LEFT,
            "right": MouseButton.RIGHT,
            "middle": MouseButton.MIDDLE,
        }
        button = button_map.get(button_type, MouseButton.LEFT)

        # Get optional position
        location = self._get_target_location(action.config) if "target" in action.config else None

        if location:
            Mouse.down(location[0], location[1], button)
            print(f"[PURE ACTION] Mouse {button_type} button pressed at {location}")
        else:
            Mouse.down(button=button)
            print(f"[PURE ACTION] Mouse {button_type} button pressed at current position")

        return True

    def _execute_mouse_up(self, action: Action, typed_config: MouseUpActionConfig = None) -> bool:
        """Execute MOUSE_UP action (pure) - release mouse button.

        This is a PURE action that only releases the mouse button.
        """
        from ..hal.interfaces import MouseButton

        # Get button type from config
        button_type = action.config.get("button", "left").lower()
        button_map = {
            "left": MouseButton.LEFT,
            "right": MouseButton.RIGHT,
            "middle": MouseButton.MIDDLE,
        }
        button = button_map.get(button_type, MouseButton.LEFT)

        # Get optional position
        location = self._get_target_location(action.config) if "target" in action.config else None

        if location:
            Mouse.up(location[0], location[1], button)
            print(f"[PURE ACTION] Mouse {button_type} button released at {location}")
        else:
            Mouse.up(button=button)
            print(f"[PURE ACTION] Mouse {button_type} button released at current position")

        return True

    def _execute_screenshot(
        self, action: Action, typed_config: ScreenshotActionConfig = None
    ) -> bool:
        """Execute SCREENSHOT action."""
        region = action.config.get("region")
        timestamp = int(self.time_wrapper.now().timestamp())
        filename = action.config.get("filename", f"screenshot_{timestamp}.png")

        if region:
            Screen.save(
                filename, region=(region["x"], region["y"], region["width"], region["height"])
            )
        else:
            Screen.save(filename)

        print(f"Screenshot saved to {filename}")
        return True

    def _execute_go_to_state(
        self, action: Action, typed_config: GoToStateActionConfig = None
    ) -> bool:
        """Execute GO_TO_STATE action.

        Navigates to one or more target states using the qontinui library's
        pathfinding (which uses the multistate library for multi-target pathfinding).

        The multistate library will find the optimal path to reach ALL specified states.
        Note: Transitions may activate additional states beyond the targets. For example,
        if there's a transition A -> {B,C} and you request GO_TO_STATE([B]), the
        transition will be executed, activating both B and C.
        """
        print(f"[DEBUG] ===== _execute_go_to_state CALLED =====")
        # Get target state IDs from config
        print(f"[DEBUG] Getting state IDs from config...")
        state_ids = typed_config.state_ids if typed_config else action.config.get("stateIds", [])
        print(f"[DEBUG] State IDs: {state_ids}")

        if not state_ids:
            print("GO_TO_STATE action missing 'stateIds' config")
            return False

        if not self.state_executor:
            print(f"GO_TO_STATE: {state_ids} (no state executor available)")
            return False

        # Validate all target states exist
        target_states = []
        for state_id in state_ids:
            if state_id not in self.config.state_map:
                print(f"GO_TO_STATE: State '{state_id}' not found")
                return False
            target_states.append(self.config.state_map[state_id])

        print(f"[DEBUG] Getting current state from state_executor...")
        current_state_id = self.state_executor.current_state
        print(f"[DEBUG] Current state: {current_state_id}")

        # Check if already at all target states
        print(f"[DEBUG] Checking if already at target states...")
        if all(current_state_id == sid for sid in state_ids):
            target_names = [self.config.state_map[sid].name for sid in state_ids]
            print(f"GO_TO_STATE: Already at state(s) {', '.join(target_names)}")
            return True

        print(f"[DEBUG] Not at target states, need to navigate")
        # Delegate to qontinui library's pathfinding (which uses multistate)
        # This handles multi-target pathfinding automatically
        print(f"[DEBUG] Importing navigation_api...")
        from .. import navigation_api

        print(f"[DEBUG] Setting workflow executor...")
        # Set workflow executor so transitions can execute workflows
        navigation_api.set_workflow_executor(self)

        print(f"[DEBUG] Converting state IDs to names...")
        # Convert state IDs to state names for the navigation API
        target_names = [st.name for st in target_states]
        print(f"GO_TO_STATE: Navigating to {len(state_ids)} state(s): {', '.join(target_names)}")

        # Call navigation_api.open_states with state names
        success = navigation_api.open_states(target_names)

        if success:
            print(f"GO_TO_STATE: Successfully navigated to {', '.join(target_names)}")
        else:
            print(f"GO_TO_STATE: Failed to navigate to {', '.join(target_names)}")

        return success

    def _build_state_graph(self):
        """Build a StateGraph from the JSON config for use with StateTraversal."""
        from ..state_management.models import (
            State as SMState,
        )
        from ..state_management.models import (
            StateGraph,
            TransitionType,
        )
        from ..state_management.models import (
            Transition as SMTransition,
        )

        # Create StateGraph
        state_graph = StateGraph()

        # Add states (State objects, not Elements)
        for state in self.config.states:
            sm_state = SMState(
                name=state.name,
                elements=[],  # StateGraph doesn't need the actual elements for traversal
                transitions=[],
            )
            state_graph.add_state(sm_state)

        # Add transitions - iterate through states and their outgoing transitions
        for state in self.config.states:
            for trans in state.outgoing_transitions:
                # Collect all target states (to_state + activate_states)
                target_state_ids = []

                # Add to_state if present
                if trans.to_state:
                    target_state_ids.append(trans.to_state)

                # Add activate_states if present
                if trans.activate_states:
                    target_state_ids.extend(trans.activate_states)

                # Create edges for ALL target states
                for state_id in target_state_ids:
                    to_state = self.config.state_map.get(state_id)
                    if to_state:
                        sm_transition = SMTransition(
                            from_state=state.name,
                            to_state=to_state.name,
                            action_type=TransitionType.CUSTOM,
                            probability=1.0,
                            metadata={"config_transition_id": trans.id},
                        )
                        state_graph.add_transition(sm_transition)

            # IncomingTransitions don't create edges - they represent
            # processes that verify you've reached a state, so we skip them

        return state_graph

    def _map_traversal_to_config_transitions(self, sm_transitions):
        """Map StateTraversal transitions back to config transitions.

        Args:
            sm_transitions: List of state_management.Transition objects

        Returns:
            List of config_parser.Transition objects (deduplicated)
        """
        config_transitions = []
        seen_ids = set()

        for sm_trans in sm_transitions:
            # Get the config transition ID from metadata
            trans_id = sm_trans.metadata.get("config_transition_id")

            # Only add each config transition once
            if trans_id and trans_id not in seen_ids:
                # Find the config transition by ID across all states
                found = False
                for state in self.config.states:
                    # Check outgoing transitions
                    for config_trans in state.outgoing_transitions:
                        if config_trans.id == trans_id:
                            config_transitions.append(config_trans)
                            seen_ids.add(trans_id)
                            found = True
                            break
                    if found:
                        break

        return config_transitions

    def _execute_run_workflow(
        self, action: Action, typed_config: RunWorkflowActionConfig = None
    ) -> bool:
        """Execute RUN_WORKFLOW action - runs a nested workflow with optional repetition."""
        workflow_id = action.config.get("workflowId")
        if not workflow_id:
            print("RUN_WORKFLOW action missing 'workflowId' config")
            return False

        workflow = self.config.workflow_map.get(workflow_id)
        if not workflow:
            print(f"RUN_WORKFLOW: Workflow '{workflow_id}' not found in config")
            return False

        # Get repetition configuration
        repetition_config = action.config.get("workflowRepetition", {})
        repetition_enabled = repetition_config.get("enabled", False)

        if not repetition_enabled:
            # No repetition - execute once
            return self._execute_workflow_once(workflow, workflow_id, 1, 1)

        # Repetition enabled
        max_repeats = repetition_config.get("maxRepeats", 10)
        delay_ms = repetition_config.get("delay", 0)
        until_success = repetition_config.get("untilSuccess", False)
        delay_seconds = delay_ms / 1000.0
        total_runs = max_repeats + 1

        print(f"RUN_WORKFLOW: Workflow '{workflow.name}' with repetition:")
        print(f"   Max repeats: {max_repeats}")
        print(f"   Delay: {delay_ms}ms")
        print(f"   Until success: {until_success}")
        print(f"   Total runs: {total_runs}")

        if until_success:
            # Mode: Repeat until success or max repeats
            for run_num in range(1, total_runs + 1):
                success = self._execute_workflow_once(workflow, workflow_id, run_num, total_runs)

                if success:
                    print(
                        f"RUN_WORKFLOW: Workflow succeeded on run {run_num}/{total_runs}, stopping early"
                    )
                    return True

                # Delay before next attempt (if not the last run)
                if run_num < total_runs and delay_seconds > 0:
                    print(f"RUN_WORKFLOW: Waiting {delay_ms}ms before next attempt")
                    self.time_wrapper.wait(delay_seconds)

            # Reached max repeats without success
            print(f"RUN_WORKFLOW: Workflow failed after {total_runs} attempts")
            return False
        else:
            # Mode: Run fixed count, aggregate results
            results = []
            for run_num in range(1, total_runs + 1):
                success = self._execute_workflow_once(workflow, workflow_id, run_num, total_runs)
                results.append(success)

                # Delay before next run (if not the last run)
                if run_num < total_runs and delay_seconds > 0:
                    print(f"RUN_WORKFLOW: Waiting {delay_ms}ms before next run")
                    self.time_wrapper.wait(delay_seconds)

            # Success if at least one run succeeded
            success_count = sum(1 for r in results if r)
            overall_success = success_count > 0
            print(f"RUN_WORKFLOW: Completed {total_runs} runs, {success_count} succeeded")
            return overall_success

    def _execute_workflow_once(
        self, workflow: Workflow, workflow_id: str, run_num: int, total_runs: int
    ) -> bool:
        """Execute a workflow once and emit events.

        Supports both v2.0.0 Workflow and v1.0.0 Process formats.

        Args:
            workflow: Workflow/Process object to execute
            workflow_id: ID of the workflow
            run_num: Current run number
            total_runs: Total number of runs

        Returns:
            bool: True if workflow execution succeeded
        """
        print(f"RUN_WORKFLOW: Executing workflow '{workflow.name}' (run {run_num}/{total_runs})")

        # Emit workflow started event
        self._emit_event(
            "workflow_started",
            {
                "workflow_id": workflow_id,
                "workflow_name": workflow.name,
                "process_type": workflow.type,
                "action_count": len(workflow.actions),
                "run_number": run_num,
                "total_runs": total_runs,
            },
        )

        success = True
        # Execute the nested workflow actions
        if workflow.type == "sequence":
            for nested_action in workflow.actions:
                if not self.execute_action(nested_action):
                    print(f"RUN_WORKFLOW: Nested action failed in workflow '{workflow.name}'")
                    success = False
                    break
        elif workflow.type == "parallel":
            # For now, execute sequentially (parallel execution would need threading)
            for nested_action in workflow.actions:
                self.execute_action(nested_action)

        # Emit process completed event (event name kept for backward compatibility)
        self._emit_event(
            "workflow_completed",
            {
                "workflow_id": workflow_id,
                "workflow_name": workflow.name,
                "success": success,
                "run_number": run_num,
                "total_runs": total_runs,
            },
        )

        print(
            f"RUN_WORKFLOW: Completed workflow '{workflow.name}' (run {run_num}/{total_runs}): {'SUCCESS' if success else 'FAILED'}"
        )
        return success

    # ========================================================================
    # Control Flow Actions
    # ========================================================================

    def _execute_loop(self, action: Action, typed_config: Any = None) -> bool:
        """Execute LOOP action (FOR, WHILE, FOREACH).

        Delegates to ControlFlowExecutor for loop execution.
        """
        logger.info(f"Executing LOOP action: {action.id}")

        # Sync variables from control flow executor to our context
        self.control_flow_executor.variables = self.variable_context.get_all_variables()

        try:
            result = self.control_flow_executor.execute_loop(action)

            # Sync variables back from control flow executor
            for key, value in self.control_flow_executor.variables.items():
                self.variable_context.set(key, value, "local")

            success = result.get("success", False)
            logger.info(
                f"LOOP completed: {result.get('iterations_completed', 0)} iterations, success={success}"
            )
            return success

        except Exception as e:
            logger.error(f"LOOP action failed: {e}", exc_info=True)
            return False

    def _execute_if(self, action: Action, typed_config: Any = None) -> bool:
        """Execute IF action (conditional branching).

        Delegates to ControlFlowExecutor for condition evaluation and branching.
        """
        logger.info(f"Executing IF action: {action.id}")

        # Sync variables from control flow executor to our context
        self.control_flow_executor.variables = self.variable_context.get_all_variables()

        try:
            result = self.control_flow_executor.execute_if(action)

            # Sync variables back from control flow executor
            for key, value in self.control_flow_executor.variables.items():
                self.variable_context.set(key, value, "local")

            success = result.get("success", False)
            logger.info(f"IF completed: branch={result.get('branch_taken')}, success={success}")
            return success

        except Exception as e:
            logger.error(f"IF action failed: {e}", exc_info=True)
            return False

    def _execute_break(self, action: Action, typed_config: Any = None) -> bool:
        """Execute BREAK action (exit loop).

        Delegates to ControlFlowExecutor which raises BreakLoop exception.
        """
        logger.info(f"Executing BREAK action: {action.id}")

        try:
            self.control_flow_executor.execute_break(action)
            # If we reach here, condition was not met
            return True

        except Exception as e:
            # BreakLoop exception should propagate up
            logger.debug(f"BREAK raised exception (expected): {e}")
            raise

    def _execute_continue(self, action: Action, typed_config: Any = None) -> bool:
        """Execute CONTINUE action (skip to next iteration).

        Delegates to ControlFlowExecutor which raises ContinueLoop exception.
        """
        logger.info(f"Executing CONTINUE action: {action.id}")

        try:
            self.control_flow_executor.execute_continue(action)
            # If we reach here, condition was not met
            return True

        except Exception as e:
            # ContinueLoop exception should propagate up
            logger.debug(f"CONTINUE raised exception (expected): {e}")
            raise

    # ========================================================================
    # Data Operation Actions
    # ========================================================================

    def _execute_set_variable(self, action: Action, typed_config: Any = None) -> bool:
        """Execute SET_VARIABLE action.

        Delegates to DataOperationsExecutor for variable management.
        """
        logger.info(f"Executing SET_VARIABLE action: {action.id}")

        try:
            # Create execution context with current variables
            context = self.variable_context.get_all_variables()

            result = self.data_operations_executor.execute_set_variable(action, context)

            success = result.get("success", False)
            if success:
                var_name = result.get("variable_name")
                logger.info(f"SET_VARIABLE completed: {var_name} = {result.get('value')}")
            else:
                logger.error(f"SET_VARIABLE failed: {result.get('error')}")

            return success

        except Exception as e:
            logger.error(f"SET_VARIABLE action failed: {e}", exc_info=True)
            return False

    def _execute_get_variable(self, action: Action, typed_config: Any = None) -> bool:
        """Execute GET_VARIABLE action.

        Delegates to DataOperationsExecutor for variable retrieval.
        """
        logger.info(f"Executing GET_VARIABLE action: {action.id}")

        try:
            # Create execution context with current variables
            context = self.variable_context.get_all_variables()

            result = self.data_operations_executor.execute_get_variable(action, context)

            success = result.get("success", False)
            if success:
                var_name = result.get("variable_name")
                logger.info(f"GET_VARIABLE completed: {var_name} = {result.get('value')}")
            else:
                logger.error(f"GET_VARIABLE failed: {result.get('error')}")

            return success

        except Exception as e:
            logger.error(f"GET_VARIABLE action failed: {e}", exc_info=True)
            return False

    def _execute_map(self, action: Action, typed_config: Any = None) -> bool:
        """Execute MAP action.

        Delegates to DataOperationsExecutor for collection transformation.
        """
        logger.info(f"Executing MAP action: {action.id}")

        try:
            context = self.variable_context.get_all_variables()
            result = self.data_operations_executor.execute_map(action, context)

            success = result.get("success", False)
            if success:
                logger.info(f"MAP completed: {result.get('item_count')} items transformed")
            else:
                logger.error(f"MAP failed: {result.get('error')}")

            return success

        except Exception as e:
            logger.error(f"MAP action failed: {e}", exc_info=True)
            return False

    def _execute_reduce(self, action: Action, typed_config: Any = None) -> bool:
        """Execute REDUCE action.

        Delegates to DataOperationsExecutor for collection reduction.
        """
        logger.info(f"Executing REDUCE action: {action.id}")

        try:
            context = self.variable_context.get_all_variables()
            result = self.data_operations_executor.execute_reduce(action, context)

            success = result.get("success", False)
            if success:
                logger.info(f"REDUCE completed: result = {result.get('reduced_value')}")
            else:
                logger.error(f"REDUCE failed: {result.get('error')}")

            return success

        except Exception as e:
            logger.error(f"REDUCE action failed: {e}", exc_info=True)
            return False

    def _execute_sort(self, action: Action, typed_config: Any = None) -> bool:
        """Execute SORT action.

        Delegates to DataOperationsExecutor for collection sorting.
        """
        logger.info(f"Executing SORT action: {action.id}")

        try:
            context = self.variable_context.get_all_variables()
            result = self.data_operations_executor.execute_sort(action, context)

            success = result.get("success", False)
            if success:
                logger.info(f"SORT completed: {result.get('item_count')} items sorted")
            else:
                logger.error(f"SORT failed: {result.get('error')}")

            return success

        except Exception as e:
            logger.error(f"SORT action failed: {e}", exc_info=True)
            return False

    def _execute_filter(self, action: Action, typed_config: Any = None) -> bool:
        """Execute FILTER action.

        Delegates to DataOperationsExecutor for collection filtering.
        """
        logger.info(f"Executing FILTER action: {action.id}")

        try:
            context = self.variable_context.get_all_variables()
            result = self.data_operations_executor.execute_filter(action, context)

            success = result.get("success", False)
            if success:
                original = result.get("original_count", 0)
                filtered = result.get("filtered_count", 0)
                logger.info(f"FILTER completed: {original} -> {filtered} items")
            else:
                logger.error(f"FILTER failed: {result.get('error')}")

            return success

        except Exception as e:
            logger.error(f"FILTER action failed: {e}", exc_info=True)
            return False

    def _execute_string_operation(self, action: Action, typed_config: Any = None) -> bool:
        """Execute STRING_OPERATION action.

        Delegates to DataOperationsExecutor for string manipulation.
        """
        logger.info(f"Executing STRING_OPERATION action: {action.id}")

        try:
            context = self.variable_context.get_all_variables()
            result = self.data_operations_executor.execute_string_operation(action, context)

            success = result.get("success", False)
            if success:
                logger.info(f"STRING_OPERATION completed: {result.get('operation')}")
            else:
                logger.error(f"STRING_OPERATION failed: {result.get('error')}")

            return success

        except Exception as e:
            logger.error(f"STRING_OPERATION action failed: {e}", exc_info=True)
            return False

    def _execute_math_operation(self, action: Action, typed_config: Any = None) -> bool:
        """Execute MATH_OPERATION action.

        Delegates to DataOperationsExecutor for mathematical operations.
        """
        logger.info(f"Executing MATH_OPERATION action: {action.id}")

        try:
            context = self.variable_context.get_all_variables()
            result = self.data_operations_executor.execute_math_operation(action, context)

            success = result.get("success", False)
            if success:
                logger.info(
                    f"MATH_OPERATION completed: {result.get('operation')} = {result.get('result')}"
                )
            else:
                logger.error(f"MATH_OPERATION failed: {result.get('error')}")

            return success

        except Exception as e:
            logger.error(f"MATH_OPERATION action failed: {e}", exc_info=True)
            return False
