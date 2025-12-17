"""Switch execution for SWITCH/CASE operations.

This module provides the SwitchExecutor class which handles SWITCH/CASE
conditional branching logic for control flow operations.
"""

import logging
from typing import Any

from qontinui.config import Action, SwitchActionConfig
from qontinui.orchestration.execution_context import ExecutionContext

from .condition_evaluator import ConditionEvaluator

logger = logging.getLogger(__name__)


class SwitchExecutor:
    """Executor for SWITCH/CASE conditional operations.

    This class handles switch-case branching logic, evaluating an expression
    and executing the appropriate action sequence based on the result.
    It uses ConditionEvaluator for expression evaluation and delegates
    action execution back to the ExecutionContext.

    The executor maintains no state between executions - all state is
    managed by the ExecutionContext.

    Example:
        >>> context = ExecutionContext({"status": "ready"})
        >>> executor = SwitchExecutor(context)
        >>> switch_action = Action(
        ...     id='switch-1',
        ...     type='SWITCH',
        ...     config={
        ...         'expression': 'status',
        ...         'cases': [
        ...             {'value': 'ready', 'actions': ['action-1']},
        ...             {'value': 'error', 'actions': ['action-2']},
        ...             {'value': ['pending', 'waiting'], 'actions': ['action-3']}
        ...         ],
        ...         'defaultActions': ['action-4']
        ...     }
        ... )
        >>> result = executor.execute_switch(switch_action)
        >>> print(result['matched_case'])  # 'ready'
    """

    def __init__(self, context: ExecutionContext) -> None:
        """Initialize the switch executor.

        Args:
            context: ExecutionContext providing variable access, action execution,
                    and state management
        """
        self.context = context
        self.condition_evaluator = ConditionEvaluator(context)
        logger.debug("SwitchExecutor initialized with context")

    def execute_switch(
        self, action: Action, config: SwitchActionConfig
    ) -> dict[str, Any]:
        """Execute a SWITCH action (case-based branching).

        Evaluates the expression and executes the action sequence from the
        matching case, or the default actions if no case matches. Returns
        detailed execution statistics including which case matched and how
        many actions were executed.

        Args:
            action: The SWITCH action to execute
            config: Validated SWITCH action configuration

        Returns:
            Dictionary containing:
                - success: Whether execution succeeded
                - expression_value: Result of expression evaluation
                - matched_case: Value of the matched case (or 'default')
                - case_index: Index of matched case (-1 for default)
                - actions_executed: Number of actions executed
                - errors: List of errors encountered (if any)
                - action_id: ID of the action executed

        Raises:
            ValueError: If expression evaluation fails
        """
        logger.info("Evaluating SWITCH expression (action_id=%s)", action.id)

        result: dict[str, Any] = {
            "success": True,
            "expression_value": None,
            "matched_case": None,
            "case_index": -1,
            "actions_executed": 0,
            "errors": [],
            "action_id": action.id,
        }

        try:
            # Evaluate expression to get the value to match
            expression_value = self._evaluate_expression(config.expression)
            result["expression_value"] = expression_value

            logger.debug("Expression evaluated to: %s", expression_value)

            # Find matching case
            actions_to_execute: list[str] = []
            matched = False

            for case_index, case in enumerate(config.cases):
                if self._matches_case(expression_value, case.value):
                    result["matched_case"] = case.value
                    result["case_index"] = case_index
                    actions_to_execute = case.actions
                    matched = True
                    logger.debug(
                        "Matched case %d with value: %s (%d actions)",
                        case_index,
                        case.value,
                        len(actions_to_execute),
                    )
                    break

            # If no case matched, use default actions
            if not matched:
                result["matched_case"] = "default"
                result["case_index"] = -1
                actions_to_execute = config.default_actions or []
                logger.debug(
                    "No case matched, executing DEFAULT with %d actions",
                    len(actions_to_execute),
                )

            # Execute the selected action sequence
            exec_result = self._execute_action_sequence(actions_to_execute)
            result["actions_executed"] = exec_result["actions_executed"]
            if isinstance(result["errors"], list):
                result["errors"].extend(exec_result["errors"])

            # Mark as failed if any errors occurred
            if exec_result["errors"]:
                result["success"] = False

        except ValueError as e:
            # Expression evaluation error
            logger.error(
                "SWITCH expression evaluation failed: %s", str(e), exc_info=True
            )
            result["success"] = False
            if isinstance(result["errors"], list):
                result["errors"].append(
                    {"type": "ExpressionEvaluationError", "message": str(e)}
                )

        except TypeError as e:
            # Type error during expression evaluation
            logger.error("SWITCH expression type error: %s", str(e), exc_info=True)
            result["success"] = False
            if isinstance(result["errors"], list):
                result["errors"].append(
                    {"type": "ExpressionTypeError", "message": str(e)}
                )

        except Exception as e:
            # Unexpected error
            logger.error("SWITCH action failed unexpectedly: %s", str(e), exc_info=True)
            result["success"] = False
            if isinstance(result["errors"], list):
                result["errors"].append({"type": type(e).__name__, "message": str(e)})

        return result

    def _evaluate_expression(self, expression: str) -> Any:
        """Evaluate a Python expression and return the result.

        Uses the same security-restricted eval context as ConditionEvaluator
        to evaluate expressions safely with access to workflow variables.

        SECURITY WARNING:
        This method uses eval() to execute Python expressions. It is designed for
        TRUSTED INPUT ONLY (automation scripts written by developers).

        Args:
            expression: Python expression string to evaluate

        Returns:
            Result of expression evaluation

        Raises:
            ValueError: If expression evaluation fails
        """
        logger.debug("Evaluating SWITCH expression: %s", expression)

        try:
            # Create safe evaluation context with variables
            # Include both namespaced and direct access to variables
            variables = self.context.variables
            eval_context = {"variables": variables, **variables}

            # Evaluate with restricted builtins for safety
            result = eval(expression, {"__builtins__": {}}, eval_context)
            logger.debug(
                "Expression result: %s (type: %s)", result, type(result).__name__
            )
            return result

        except NameError as e:
            logger.error("Expression references undefined variable: %s", str(e))
            raise ValueError(f"Invalid expression '{expression}': {e}") from e

        except SyntaxError as e:
            logger.error("Expression has invalid syntax: %s", str(e))
            raise ValueError(f"Invalid expression '{expression}': {e}") from e

        except TypeError as e:
            logger.error("Expression type error: %s", str(e))
            raise ValueError(f"Invalid expression '{expression}': {e}") from e

        except ZeroDivisionError as e:
            logger.error("Expression division by zero: %s", str(e))
            raise ValueError(f"Invalid expression '{expression}': {e}") from e

    def _matches_case(self, expression_value: Any, case_value: Any | list[Any]) -> bool:
        """Check if expression value matches a case value.

        A case value can be either a single value or a list of values.
        The expression matches if it equals any of the values.

        Args:
            expression_value: The evaluated expression result
            case_value: Single value or list of values to match against

        Returns:
            True if expression_value matches any of the case values
        """
        # If case_value is a list, check if expression matches any value in the list
        if isinstance(case_value, list):
            return bool(expression_value in case_value)
        else:
            # Single value comparison
            return bool(expression_value == case_value)

    def _execute_action_sequence(self, action_ids: list[str]) -> dict[str, Any]:
        """Execute a sequence of actions.

        Executes each action in the sequence using the ExecutionContext's
        execute_action callback. Collects execution statistics and errors
        for all actions in the sequence.

        Note: This method is shared conceptually with ConditionalExecutor and
        LoopExecutor but implemented separately to maintain executor independence.

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
        errors_list: list[dict[str, Any]] = []
        result: dict[str, Any] = {"actions_executed": 0, "errors": errors_list}

        for action_id in action_ids:
            logger.debug("Executing action: %s", action_id)

            try:
                # Get action from config
                action = self._get_action_by_id(action_id)
                if not action:
                    logger.error("Action not found: %s", action_id)
                    errors_list.append(
                        {
                            "action_id": action_id,
                            "type": "ActionNotFound",
                            "message": f"Action with ID '{action_id}' not found in workflow",
                        }
                    )
                    result["actions_executed"] = result["actions_executed"] + 1  # type: ignore[assignment]
                    continue

                # Execute action via context callback
                # The ExecutionContext.execute_action method handles all the
                # orchestration including error handling, event emission, etc.
                # Note: execute_action is a method that may be added to ExecutionContext
                # If it doesn't exist, we'll catch the AttributeError below
                success = getattr(self.context, "execute_action", lambda x: False)(
                    action
                )

                # Track execution
                if not success:
                    logger.warning("Action %s failed", action_id)
                    errors_list.append(
                        {
                            "action_id": action_id,
                            "type": "ActionExecutionFailed",
                            "message": "Action execution returned false",
                        }
                    )

                result["actions_executed"] = result["actions_executed"] + 1  # type: ignore[assignment]

            except Exception as e:
                # Unexpected exception during action execution
                logger.error(
                    "Action %s raised exception: %s", action_id, str(e), exc_info=True
                )
                errors_list.append(
                    {
                        "action_id": action_id,
                        "type": type(e).__name__,
                        "message": str(e),
                    }
                )
                result["actions_executed"] = result["actions_executed"] + 1  # type: ignore[assignment]

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
        config = getattr(self.context, "config", None)
        if not config or not hasattr(config, "workflow"):
            logger.warning("No workflow config available to find action: %s", action_id)
            return None

        workflow = getattr(config, "workflow", None)
        if not workflow or not hasattr(workflow, "actions"):
            logger.warning("Workflow has no actions list")
            return None

        # Search for action by ID
        actions = getattr(workflow, "actions", [])
        for action in actions:
            if hasattr(action, "id") and action.id == action_id:
                return action  # type: ignore[no-any-return]

        logger.debug("Action not found: %s", action_id)
        return None
