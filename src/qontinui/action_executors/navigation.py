"""Navigation action executors for state transitions and workflow execution.

This module provides specialized executors for navigation-related actions that
change the application state or execute nested workflows.
"""

import logging
from typing import Any

from ..config.schema import Action, GoToStateActionConfig, RunWorkflowActionConfig
from ..exceptions import ActionExecutionError, StateNotFoundException
from .base import ActionExecutorBase
from .registry import register_executor

logger = logging.getLogger(__name__)


@register_executor
class NavigationActionExecutor(ActionExecutorBase):
    """Executor for navigation actions: GO_TO_STATE and RUN_WORKFLOW.

    This executor handles state machine navigation and nested workflow execution.
    It coordinates with the state executor for pathfinding and with the workflow
    executor for running nested workflows.

    Supported actions:
        - GO_TO_STATE: Navigate to target states using pathfinding
        - RUN_WORKFLOW: Execute nested workflow with optional repetition

    Example:
        >>> context = ExecutionContext(...)
        >>> executor = NavigationActionExecutor(context)
        >>>
        >>> # Execute GO_TO_STATE action
        >>> goto_action = Action(type="GO_TO_STATE", config={"stateIds": ["state1"]})
        >>> goto_config = GoToStateActionConfig(state_ids=["state1"])
        >>> executor.execute(goto_action, goto_config)
        True
        >>>
        >>> # Execute RUN_WORKFLOW action
        >>> workflow_action = Action(type="RUN_WORKFLOW", config={"workflowId": "wf1"})
        >>> workflow_config = RunWorkflowActionConfig(workflow_id="wf1")
        >>> executor.execute(workflow_action, workflow_config)
        True
    """

    def get_supported_action_types(self) -> list[str]:
        """Get list of action types handled by this executor.

        Returns:
            List containing "GO_TO_STATE" and "RUN_WORKFLOW"
        """
        return ["GO_TO_STATE", "RUN_WORKFLOW"]

    def execute(self, action: Action, typed_config: Any) -> bool:
        """Execute navigation action with validated configuration.

        Args:
            action: Pydantic Action model with type, id, config
            typed_config: Type-specific validated configuration object
                - GoToStateActionConfig for GO_TO_STATE actions
                - RunWorkflowActionConfig for RUN_WORKFLOW actions

        Returns:
            bool: True if action succeeded, False otherwise

        Raises:
            ActionExecutionError: If action execution fails critically
            StateNotFoundException: If target state not found (GO_TO_STATE)
        """
        logger.debug(f"Executing navigation action: {action.type}")

        if action.type == "GO_TO_STATE":
            return self._execute_go_to_state(action, typed_config)
        elif action.type == "RUN_WORKFLOW":
            return self._execute_run_workflow(action, typed_config)
        else:
            logger.error(f"Unknown navigation action type: {action.type}")
            raise ActionExecutionError(
                action_type=action.type,
                reason=f"Navigation executor does not handle {action.type}",
                action_id=action.id,
            )

    def _execute_go_to_state(
        self, action: Action, typed_config: GoToStateActionConfig
    ) -> bool:
        """Execute GO_TO_STATE action - navigate to target states.

        Navigates to one or more target states using the qontinui library's
        pathfinding (which uses the multistate library for multi-target pathfinding).

        The multistate library will find the optimal path to reach ALL specified states.
        Note: Transitions may activate additional states beyond the targets. For example,
        if there's a transition A -> {B,C} and you request GO_TO_STATE([B]), the
        transition will be executed, activating both B and C.

        Args:
            action: Pydantic Action model
            typed_config: Pre-validated GoToStateActionConfig

        Returns:
            bool: True if navigation succeeded, False otherwise

        Raises:
            StateNotFoundException: If any target state doesn't exist
            ActionExecutionError: If navigation fails
        """
        logger.debug("Executing GO_TO_STATE action")

        # Get target state IDs from config
        state_ids = (
            typed_config.state_ids
            if typed_config
            else action.config.get("stateIds", [])
        )
        logger.debug(f"Target state IDs: {state_ids}")

        if not state_ids:
            raise ActionExecutionError(
                action_type="GO_TO_STATE",
                reason="stateIds is required and must not be empty",
                action_id=action.id,
            )

        if not self.context.state_executor:
            logger.warning("GO_TO_STATE: No state executor available")
            raise ActionExecutionError(
                action_type="GO_TO_STATE",
                reason="No state executor available for navigation",
                action_id=action.id,
            )

        # Validate all target states exist
        target_states = []
        for state_id in state_ids:
            if state_id not in self.context.config.state_map:
                raise StateNotFoundException(state_id, action_id=action.id)
            target_states.append(self.context.config.state_map[state_id])

        # Check if already at all target states
        current_state_id = self.context.state_executor.current_state
        logger.debug(f"Current state: {current_state_id}")

        if all(current_state_id == sid for sid in state_ids):
            target_names = [
                self.context.config.state_map[sid].name for sid in state_ids
            ]
            logger.info(f"Already at target state(s): {', '.join(target_names)}")

            self._emit_action_success(
                action,
                {
                    "state_ids": state_ids,
                    "state_names": target_names,
                    "already_at_target": True,
                },
            )
            return True

        # Delegate to qontinui library's pathfinding (which uses multistate)
        logger.debug("Initiating navigation via navigation_api")
        from .. import navigation_api

        # REMOVED: Don't override workflow_executor - it's already set by the runner
        # The runner calls navigation_api.set_workflow_executor() immediately after
        # load_configuration() with the proper executor that can execute workflows.
        # Overriding it here with self.context.workflow_executor (which may be None)
        # breaks navigation transitions.
        # Old code: navigation_api.set_workflow_executor(self.context.workflow_executor)
        # Convert state IDs to state names for the navigation API
        target_names = [st.name for st in target_states]
        logger.info(
            f"Navigating to {len(state_ids)} state(s): {', '.join(target_names)}"
        )

        try:
            # Call navigation_api.open_states with state names
            success = navigation_api.open_states(target_names)

            if success:
                logger.info(f"Successfully navigated to: {', '.join(target_names)}")
                self._emit_action_success(
                    action,
                    {
                        "state_ids": state_ids,
                        "state_names": target_names,
                        "navigation_successful": True,
                    },
                )
            else:
                logger.warning(f"Failed to navigate to: {', '.join(target_names)}")
                self._emit_action_failure(
                    action,
                    "Navigation failed to reach target states",
                    {"state_ids": state_ids, "state_names": target_names},
                )

            return success

        except Exception as e:
            logger.error(f"Navigation failed with exception: {e}", exc_info=True)
            self._emit_action_failure(
                action,
                f"Navigation exception: {e}",
                {"state_ids": state_ids, "state_names": target_names},
            )
            raise ActionExecutionError(
                action_type="GO_TO_STATE",
                reason=f"Navigation failed: {e}",
                action_id=action.id,
            ) from e

    def _execute_run_workflow(
        self, action: Action, typed_config: RunWorkflowActionConfig
    ) -> bool:
        """Execute RUN_WORKFLOW action - run nested workflow with optional repetition.

        This method executes a nested workflow by looking it up in the config's
        workflow_map and then executing its actions. It supports:
        - Single execution (no repetition)
        - Fixed repetition count with delay
        - Retry until success with max attempts

        Args:
            action: Pydantic Action model
            typed_config: Pre-validated RunWorkflowActionConfig

        Returns:
            bool: True if workflow execution succeeded, False otherwise

        Raises:
            InvalidActionParametersException: If workflow_id is missing or invalid
            ActionExecutionError: If workflow execution fails
        """
        logger.debug("Executing RUN_WORKFLOW action")

        # Get workflow ID from config
        workflow_id = action.config.get("workflowId")
        if not workflow_id:
            raise ActionExecutionError(
                action_type="RUN_WORKFLOW",
                reason="workflowId is required",
                action_id=action.id,
            )

        # Look up workflow in config
        workflow = self.context.config.workflow_map.get(workflow_id)
        if not workflow:
            raise ActionExecutionError(
                action_type="RUN_WORKFLOW",
                reason=f"Workflow '{workflow_id}' not found in config",
                action_id=action.id,
            )

        logger.info(f"Found workflow: '{workflow.name}' (id: {workflow_id})")

        # Get repetition configuration
        repetition_config = action.config.get("workflowRepetition", {})
        repetition_enabled = repetition_config.get("enabled", False)

        if not repetition_enabled:
            # No repetition - execute once
            logger.debug("Executing workflow once (no repetition)")
            return self._execute_workflow_once(workflow, workflow_id, 1, 1)

        # Repetition enabled
        max_repeats = repetition_config.get("maxRepeats", 10)
        delay_ms = repetition_config.get("delay", 0)
        until_success = repetition_config.get("untilSuccess", False)
        delay_seconds = delay_ms / 1000.0
        total_runs = max_repeats + 1

        logger.info(
            f"Executing workflow with repetition: max_repeats={max_repeats}, "
            f"delay={delay_ms}ms, until_success={until_success}"
        )

        if until_success:
            # Mode: Repeat until success or max repeats
            for run_num in range(1, total_runs + 1):
                success = self._execute_workflow_once(
                    workflow, workflow_id, run_num, total_runs
                )

                if success:
                    logger.info(
                        f"Workflow succeeded on run {run_num}/{total_runs}, stopping early"
                    )
                    return True

                # Delay before next attempt (if not the last run)
                if run_num < total_runs and delay_seconds > 0:
                    logger.debug(f"Waiting {delay_ms}ms before next attempt")
                    self.context.time.wait(delay_seconds)

            # Reached max repeats without success
            logger.warning(f"Workflow failed after {total_runs} attempts")
            return False
        else:
            # Mode: Run fixed count, aggregate results
            results = []
            for run_num in range(1, total_runs + 1):
                success = self._execute_workflow_once(
                    workflow, workflow_id, run_num, total_runs
                )
                results.append(success)

                # Delay before next run (if not the last run)
                if run_num < total_runs and delay_seconds > 0:
                    logger.debug(f"Waiting {delay_ms}ms before next run")
                    self.context.time.wait(delay_seconds)

            # Success if at least one run succeeded
            success_count = sum(1 for r in results if r)
            overall_success = success_count > 0
            logger.info(f"Completed {total_runs} runs, {success_count} succeeded")
            return overall_success

    def _execute_workflow_once(
        self, workflow: Any, workflow_id: str, run_num: int, total_runs: int
    ) -> bool:
        """Execute a workflow once and emit events.

        This method handles the actual execution of a workflow's actions and
        emits start/completion events for monitoring.

        Args:
            workflow: Workflow object to execute
            workflow_id: ID of the workflow
            run_num: Current run number (1-indexed)
            total_runs: Total number of runs planned

        Returns:
            bool: True if workflow execution succeeded
        """
        logger.info(
            f"Executing workflow '{workflow.name}' (run {run_num}/{total_runs})"
        )

        # Emit workflow started event
        self.context.emit_event(
            "workflow_started",
            {
                "workflow_id": workflow_id,
                "workflow_name": workflow.name,
                "workflow_format": workflow.format,  # Always "graph" now
                "action_count": len(workflow.actions),
                "run_number": run_num,
                "total_runs": total_runs,
            },
        )

        success = True
        executed_count = 0

        try:
            # Execute the nested workflow using workflow_executor
            # All workflows are now graph-based, so we delegate to the executor
            if workflow.format == "graph":
                # Graph execution - use GraphTraverser to get proper execution order
                # The actions array order may not match the connection graph order
                from ..execution.graph_traverser import GraphTraverser

                traverser = GraphTraverser(workflow)
                action_map = {action.id: action for action in workflow.actions}

                # Find entry points and execute in graph order
                entry_points = traverser.find_entry_points()
                if not entry_points:
                    logger.warning(
                        f"No entry points found in workflow '{workflow.name}'"
                    )
                    # Fallback to array order if no entry points
                    for nested_action in workflow.actions:
                        action_success = self.context.execute_action(nested_action)
                        if not action_success:
                            logger.warning(
                                f"Nested action failed in workflow '{workflow.name}': "
                                f"{nested_action.type} (id: {nested_action.id}), continuing execution"
                            )
                            success = False
                        executed_count += 1
                else:
                    # Execute following the graph connections
                    executed_set: set[str] = set()
                    execution_queue = list(entry_points)

                    while execution_queue:
                        action_id = execution_queue.pop(0)
                        if action_id in executed_set:
                            continue

                        nested_action = action_map.get(action_id)
                        if not nested_action:
                            logger.warning(
                                f"Action '{action_id}' not found in workflow"
                            )
                            continue

                        # Execute the action
                        action_success = self.context.execute_action(nested_action)
                        executed_set.add(action_id)
                        executed_count += 1

                        if not action_success:
                            logger.warning(
                                f"Nested action failed in workflow '{workflow.name}': "
                                f"{nested_action.type} (id: {nested_action.id}), continuing execution"
                            )
                            success = False

                        # Get next actions from connections
                        next_actions = traverser.get_next_actions(action_id, "main")
                        for next_id, _ in next_actions:
                            if next_id not in executed_set:
                                execution_queue.append(next_id)

            elif workflow.type == "parallel":
                # For now, execute sequentially (parallel execution would need threading)
                # In future, this could use concurrent.futures or similar
                logger.debug(
                    "Executing 'parallel' workflow sequentially (threading not implemented)"
                )
                for nested_action in workflow.actions:
                    self.context.execute_action(nested_action)
                    executed_count += 1

            else:
                logger.warning(
                    f"Unknown workflow type: {workflow.type}, executing sequentially"
                )
                for nested_action in workflow.actions:
                    # Model-based GUI automation principle: always continue, never stop on failure
                    action_success = self.context.execute_action(nested_action)
                    if not action_success:
                        success = False
                    executed_count += 1

        except Exception as e:
            logger.error(f"Exception during workflow execution: {e}", exc_info=True)
            success = False

        # Emit workflow completed event
        self.context.emit_event(
            "workflow_completed",
            {
                "workflow_id": workflow_id,
                "workflow_name": workflow.name,
                "success": success,
                "run_number": run_num,
                "total_runs": total_runs,
                "actions_executed": executed_count,
                "total_actions": len(workflow.actions),
            },
        )

        status = "SUCCESS" if success else "FAILED"
        logger.info(
            f"Completed workflow '{workflow.name}' (run {run_num}/{total_runs}): {status} "
            f"({executed_count}/{len(workflow.actions)} actions)"
        )

        return success
