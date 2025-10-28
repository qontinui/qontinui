"""Conditional execution for IF/ELSE operations.

This module provides the ConditionalExecutor class which handles IF/ELSE
conditional branching logic for control flow operations.
"""

import logging
from typing import Any

from qontinui.config import Action, IfActionConfig
from qontinui.orchestration.execution_context import ExecutionContext

from .condition_evaluator import ConditionEvaluator

logger = logging.getLogger(__name__)


class ConditionalExecutor:
    """Executor for IF/ELSE conditional operations.

    This class handles conditional branching logic, evaluating conditions
    and executing the appropriate action sequences based on the result.
    It uses ConditionEvaluator for condition evaluation and delegates
    action execution back to the ExecutionContext.

    The executor maintains no state between executions - all state is
    managed by the ExecutionContext.

    Example:
        >>> context = ExecutionContext({"counter": 5})
        >>> executor = ConditionalExecutor(context)
        >>> if_action = Action(
        ...     id='if-1',
        ...     type='IF',
        ...     config={
        ...         'condition': {
        ...             'type': 'variable',
        ...             'variableName': 'counter',
        ...             'operator': '>',
        ...             'expectedValue': 3
        ...         },
        ...         'thenActions': ['action-1'],
        ...         'elseActions': ['action-2']
        ...     }
        ... )
        >>> result = executor.execute_if(if_action)
        >>> print(result['branch_taken'])  # 'then'
    """

    def __init__(self, context: ExecutionContext) -> None:
        """Initialize the conditional executor.

        Args:
            context: ExecutionContext providing variable access, action execution,
                    and state management
        """
        self.context = context
        self.condition_evaluator = ConditionEvaluator(context)
        logger.debug("ConditionalExecutor initialized with context")

    def execute_if(self, action: Action, config: IfActionConfig) -> dict[str, Any]:
        """Execute an IF action (conditional branching).

        Evaluates the condition and executes either the 'then' or 'else'
        action sequence based on the result. Returns detailed execution
        statistics including which branch was taken and how many actions
        were executed.

        Args:
            action: The IF action to execute
            config: Validated IF action configuration

        Returns:
            Dictionary containing:
                - success: Whether execution succeeded
                - condition_result: Boolean result of condition evaluation
                - branch_taken: 'then' or 'else'
                - actions_executed: Number of actions executed
                - errors: List of errors encountered (if any)
                - action_id: ID of the action executed

        Raises:
            ValueError: If condition evaluation fails
        """
        logger.info("Evaluating IF condition (action_id=%s)", action.id)

        result = {
            "success": True,
            "condition_result": False,
            "branch_taken": None,
            "actions_executed": 0,
            "errors": [],
            "action_id": action.id,
        }

        try:
            # Evaluate condition using ConditionEvaluator
            condition_result = self.condition_evaluator.evaluate_condition(config.condition)
            result["condition_result"] = condition_result

            logger.debug("Condition evaluated to: %s", condition_result)

            # Determine which branch to execute
            if condition_result:
                result["branch_taken"] = "then"
                actions_to_execute = config.then_actions
                logger.debug("Executing THEN branch with %d actions", len(actions_to_execute))
            else:
                result["branch_taken"] = "else"
                actions_to_execute = config.else_actions or []
                logger.debug("Executing ELSE branch with %d actions", len(actions_to_execute))

            # Execute the selected action sequence
            exec_result = self._execute_action_sequence(actions_to_execute)
            result["actions_executed"] = exec_result["actions_executed"]
            result["errors"].extend(exec_result["errors"])

            # Mark as failed if any errors occurred
            if exec_result["errors"]:
                result["success"] = False

        except ValueError as e:
            # Condition evaluation error
            logger.error("IF condition evaluation failed: %s", str(e), exc_info=True)
            result["success"] = False
            result["errors"].append({"type": "ConditionEvaluationError", "message": str(e)})

        except TypeError as e:
            # Type error during condition evaluation
            logger.error("IF condition type error: %s", str(e), exc_info=True)
            result["success"] = False
            result["errors"].append({"type": "ConditionTypeError", "message": str(e)})

        except Exception as e:
            # Unexpected error
            logger.error("IF action failed unexpectedly: %s", str(e), exc_info=True)
            result["success"] = False
            result["errors"].append({"type": type(e).__name__, "message": str(e)})

        return result

    def _execute_action_sequence(self, action_ids: list[str]) -> dict[str, Any]:
        """Execute a sequence of actions.

        Executes each action in the sequence using the ExecutionContext's
        execute_action callback. Collects execution statistics and errors
        for all actions in the sequence.

        Note: This method is shared conceptually with LoopExecutor but
        implemented separately to maintain executor independence.

        Args:
            action_ids: List of action IDs to execute in sequence

        Returns:
            Dictionary containing:
                - actions_executed: Number of actions executed
                - errors: List of errors encountered during execution

        Raises:
            Any exceptions from action execution are caught and logged
            as errors in the result dictionary.
        """
        result = {"actions_executed": 0, "errors": []}

        for action_id in action_ids:
            logger.debug("Executing action: %s", action_id)

            try:
                # Get action from config
                action = self._get_action_by_id(action_id)
                if not action:
                    logger.error("Action not found: %s", action_id)
                    result["errors"].append(
                        {
                            "action_id": action_id,
                            "type": "ActionNotFound",
                            "message": f"Action with ID '{action_id}' not found in workflow",
                        }
                    )
                    result["actions_executed"] += 1
                    continue

                # Execute action via context callback
                # The ExecutionContext.execute_action method handles all the
                # orchestration including error handling, event emission, etc.
                success = self.context.execute_action(action)

                # Track execution
                if not success:
                    logger.warning("Action %s failed", action_id)
                    result["errors"].append(
                        {
                            "action_id": action_id,
                            "type": "ActionExecutionFailed",
                            "message": "Action execution returned false",
                        }
                    )

                result["actions_executed"] += 1

            except Exception as e:
                # Unexpected exception during action execution
                logger.error("Action %s raised exception: %s", action_id, str(e), exc_info=True)
                result["errors"].append(
                    {"action_id": action_id, "type": type(e).__name__, "message": str(e)}
                )
                result["actions_executed"] += 1

        return result

    def _get_action_by_id(self, action_id: str) -> Action | None:
        """Get action from config by ID.

        Searches through the workflow configuration to find an action
        with the specified ID.

        Args:
            action_id: Action ID to search for

        Returns:
            Action if found, None otherwise
        """
        if not self.context.config or not hasattr(self.context.config, "workflow"):
            logger.warning("No workflow config available to find action: %s", action_id)
            return None

        workflow = self.context.config.workflow
        if not workflow or not hasattr(workflow, "actions"):
            logger.warning("Workflow has no actions list")
            return None

        # Search for action by ID
        for action in workflow.actions:
            if action.id == action_id:
                return action

        logger.debug("Action not found: %s", action_id)
        return None
