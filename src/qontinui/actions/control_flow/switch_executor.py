"""Switch execution for SWITCH/CASE operations.

This module provides the SwitchExecutor class which handles SWITCH/CASE
conditional branching logic for control flow operations.
"""

import logging
from typing import Any

from qontinui.actions.data_operations.evaluator import SafeEvaluator
from qontinui.config import Action, SwitchActionConfig
from qontinui.orchestration.execution_context import ExecutionContext

from .condition_evaluator import ConditionEvaluator
from .exceptions import BreakLoop, ContinueLoop

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

    async def execute_switch(
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
            exec_result = await self._execute_action_sequence(actions_to_execute)
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

        Uses SafeEvaluator which performs AST-based whitelist validation before
        evaluation, blocking dangerous operations (imports, file I/O, exec, etc.).

        SECURITY NOTE:
        Designed for TRUSTED INPUT ONLY (automation scripts written by developers).
        Do not use with user-provided input from web forms, APIs, or CLI arguments.
        For untrusted scenarios, run Qontinui in isolated containers/VMs.

        Args:
            expression: Python expression string to evaluate

        Returns:
            Result of expression evaluation

        Raises:
            ValueError: If expression evaluation fails
        """
        logger.debug("Evaluating SWITCH expression: %s", expression)

        # Create evaluation context with both namespaced and direct variable access
        variables = self.context.variables
        eval_context = {"variables": variables, **variables}

        result = SafeEvaluator.safe_eval(expression, eval_context)
        logger.debug("Expression result: %s (type: %s)", result, type(result).__name__)
        return result

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

    async def _execute_action_sequence(self, action_ids: list[str]) -> dict[str, Any]:
        """Execute a sequence of actions by ID via the context callback.

        Passes the action_id STRING straight to the context's execute_action
        callback (the legacy ControlFlowExecutor contract). The callback —
        one layer up, supplied by the user — owns id→action resolution.

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

        execute_action = getattr(self.context, "execute_action", None)
        if execute_action is None:
            logger.warning(
                "No execute_action callback in context, skipping %d actions",
                len(action_ids),
            )
            return result

        for action_id in action_ids:
            logger.debug("Executing action: %s", action_id)
            try:
                action_result = await execute_action(action_id)
                success = action_result.get("success", True)
                if not success:
                    errors_list.append(
                        {
                            "action_id": action_id,
                            "message": action_result.get("error", "Action failed"),
                        }
                    )
                result["actions_executed"] = result["actions_executed"] + 1  # type: ignore[assignment]
            except (BreakLoop, ContinueLoop):
                result["actions_executed"] = result["actions_executed"] + 1  # type: ignore[assignment]
                raise
            except Exception as e:
                logger.error("Action %s raised: %s", action_id, e, exc_info=True)
                errors_list.append(
                    {
                        "action_id": action_id,
                        "type": type(e).__name__,
                        "message": str(e),
                    }
                )
                result["actions_executed"] = result["actions_executed"] + 1  # type: ignore[assignment]
        return result
