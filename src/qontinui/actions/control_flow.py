"""Control flow action executor for LOOP, IF, BREAK, and CONTINUE actions.

This module implements control flow operations for the qontinui action system,
supporting FOR, WHILE, and FOREACH loops with proper variable management,
condition evaluation, and error handling.
"""

import logging
from collections.abc import Callable
from typing import Any

from qontinui.config import (
    Action,
    BreakActionConfig,
    ConditionConfig,
    ContinueActionConfig,
    IfActionConfig,
    LoopActionConfig,
    get_typed_config,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================


class BreakLoop(Exception):
    """Exception raised to break out of a loop."""

    def __init__(self, message: str = "Loop break triggered"):
        self.message = message
        super().__init__(self.message)


class ContinueLoop(Exception):
    """Exception raised to continue to next loop iteration."""

    def __init__(self, message: str = "Loop continue triggered"):
        self.message = message
        super().__init__(self.message)


# ============================================================================
# Control Flow Executor
# ============================================================================


class ControlFlowExecutor:
    """Executor for control flow actions (LOOP, IF, BREAK, CONTINUE).

    This class manages execution of control flow actions including loops,
    conditionals, and flow control statements. It maintains execution context
    with variables and supports nested action execution.

    Example:
        >>> executor = ControlFlowExecutor()
        >>> loop_action = Action(
        ...     id='loop-1',
        ...     type='LOOP',
        ...     config={
        ...         'loopType': 'FOR',
        ...         'iterations': 10,
        ...         'iteratorVariable': 'i',
        ...         'actions': ['action-1', 'action-2']
        ...     }
        ... )
        >>> result = executor.execute_loop(loop_action)
    """

    def __init__(
        self,
        action_executor: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        variables: dict[str, Any] | None = None,
    ):
        """Initialize the control flow executor.

        Args:
            action_executor: Function to execute actions by ID. Should accept
                           (action_id, variables) and return execution result.
                           If None, actions will be logged but not executed.
            variables: Initial variable context. If None, starts with empty dict.
        """
        self.action_executor = action_executor
        self.variables = variables if variables is not None else {}
        logger.debug("ControlFlowExecutor initialized with %d variables", len(self.variables))

    # ========================================================================
    # Main Execution Methods
    # ========================================================================

    def execute_loop(self, action: Action) -> dict[str, Any]:
        """Execute a LOOP action (FOR, WHILE, or FOREACH).

        Args:
            action: The loop action to execute

        Returns:
            Dictionary containing:
                - success: Whether loop completed successfully
                - iterations_completed: Number of iterations executed
                - stopped_early: Whether loop was broken early
                - errors: List of errors encountered (if any)
                - duration_ms: Total execution time in milliseconds

        Raises:
            ValueError: If loop configuration is invalid
        """
        config: LoopActionConfig = get_typed_config(action)

        logger.info(
            "Starting %s loop (action_id=%s, max_iterations=%s)",
            config.loop_type,
            action.id,
            config.max_iterations,
        )

        result = {
            "success": True,
            "iterations_completed": 0,
            "stopped_early": False,
            "errors": [],
            "action_id": action.id,
            "loop_type": config.loop_type,
        }

        import time

        start_time = time.time()

        try:
            if config.loop_type == "FOR":
                result.update(self._execute_for_loop(config))
            elif config.loop_type == "WHILE":
                result.update(self._execute_while_loop(config))
            elif config.loop_type == "FOREACH":
                result.update(self._execute_foreach_loop(config))
            else:
                raise ValueError(f"Unknown loop type: {config.loop_type}")

        except BreakLoop as e:
            logger.info("Loop broken: %s", e.message)
            result["stopped_early"] = True
            result["break_message"] = e.message

        except Exception as e:
            logger.error("Loop execution failed: %s", str(e), exc_info=True)
            result["success"] = False
            result["errors"].append({"type": type(e).__name__, "message": str(e)})

        finally:
            end_time = time.time()
            result["duration_ms"] = (end_time - start_time) * 1000

            logger.info(
                "Loop completed: %d iterations, success=%s, stopped_early=%s",
                result["iterations_completed"],
                result["success"],
                result.get("stopped_early", False),
            )

        return result

    def execute_if(self, action: Action) -> dict[str, Any]:
        """Execute an IF action (conditional branching).

        Args:
            action: The IF action to execute

        Returns:
            Dictionary containing:
                - success: Whether execution succeeded
                - condition_result: Boolean result of condition evaluation
                - branch_taken: 'then' or 'else'
                - actions_executed: Number of actions executed
                - errors: List of errors encountered (if any)
        """
        config: IfActionConfig = get_typed_config(action)

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
            # Evaluate condition
            condition_result = self.evaluate_condition(config.condition)
            result["condition_result"] = condition_result

            logger.debug("Condition evaluated to: %s", condition_result)

            # Execute appropriate branch
            if condition_result:
                result["branch_taken"] = "then"
                actions_to_execute = config.then_actions
                logger.debug("Executing THEN branch with %d actions", len(actions_to_execute))
            else:
                result["branch_taken"] = "else"
                actions_to_execute = config.else_actions or []
                logger.debug("Executing ELSE branch with %d actions", len(actions_to_execute))

            # Execute actions
            exec_result = self._execute_action_sequence(actions_to_execute)
            result["actions_executed"] = exec_result["actions_executed"]
            result["errors"].extend(exec_result["errors"])

            if exec_result["errors"]:
                result["success"] = False

        except Exception as e:
            logger.error("IF action failed: %s", str(e), exc_info=True)
            result["success"] = False
            result["errors"].append({"type": type(e).__name__, "message": str(e)})

        return result

    def execute_break(self, action: Action) -> None:
        """Execute a BREAK action (exit loop).

        Args:
            action: The BREAK action to execute

        Raises:
            BreakLoop: Always raises to signal loop break
        """
        config: BreakActionConfig = get_typed_config(action)

        # Check condition if present
        if config.condition:
            should_break = self.evaluate_condition(config.condition)
            if not should_break:
                logger.debug("BREAK condition not met, continuing loop")
                return

        message = config.message or "Break triggered"
        logger.info("Executing BREAK: %s", message)
        raise BreakLoop(message)

    def execute_continue(self, action: Action) -> None:
        """Execute a CONTINUE action (skip to next iteration).

        Args:
            action: The CONTINUE action to execute

        Raises:
            ContinueLoop: Always raises to signal iteration skip
        """
        config: ContinueActionConfig = get_typed_config(action)

        # Check condition if present
        if config.condition:
            should_continue = self.evaluate_condition(config.condition)
            if not should_continue:
                logger.debug("CONTINUE condition not met, proceeding normally")
                return

        message = config.message or "Continue triggered"
        logger.info("Executing CONTINUE: %s", message)
        raise ContinueLoop(message)

    # ========================================================================
    # Loop Implementation Methods
    # ========================================================================

    def _execute_for_loop(self, config: LoopActionConfig) -> dict[str, Any]:
        """Execute a FOR loop (fixed iteration count).

        Args:
            config: Loop configuration

        Returns:
            Partial result dictionary with iterations and errors
        """
        if config.iterations is None:
            raise ValueError("FOR loop requires 'iterations' to be specified")

        iterations = config.iterations
        max_iterations = config.max_iterations or 10000  # Safety limit

        if iterations > max_iterations:
            logger.warning(
                "FOR loop iterations (%d) exceeds max_iterations (%d), capping",
                iterations,
                max_iterations,
            )
            iterations = max_iterations

        result = {"iterations_completed": 0, "errors": []}

        for i in range(iterations):
            logger.debug("FOR loop iteration %d/%d", i + 1, iterations)

            # Set iterator variable if specified
            if config.iterator_variable:
                self.variables[config.iterator_variable] = i
                logger.debug("Set %s = %d", config.iterator_variable, i)

            try:
                # Execute actions in this iteration
                exec_result = self._execute_action_sequence(config.actions)
                result["errors"].extend(exec_result["errors"])

                # Check if we should break on error
                if exec_result["errors"] and config.break_on_error:
                    logger.warning("Breaking loop due to error (break_on_error=True)")
                    break

            except ContinueLoop as e:
                logger.debug("Continue to next iteration: %s", e.message)
                result["iterations_completed"] += 1
                continue

            except BreakLoop:
                # Let BreakLoop propagate up
                result["iterations_completed"] += 1
                raise

            result["iterations_completed"] += 1

        return result

    def _execute_while_loop(self, config: LoopActionConfig) -> dict[str, Any]:
        """Execute a WHILE loop (condition-based).

        Args:
            config: Loop configuration

        Returns:
            Partial result dictionary with iterations and errors
        """
        if config.condition is None:
            raise ValueError("WHILE loop requires 'condition' to be specified")

        max_iterations = config.max_iterations or 10000  # Safety limit
        result = {"iterations_completed": 0, "errors": []}

        iteration = 0
        while iteration < max_iterations:
            # Evaluate condition
            try:
                condition_result = self.evaluate_condition(config.condition)
            except Exception as e:
                logger.error("Failed to evaluate WHILE condition: %s", str(e))
                result["errors"].append(
                    {"type": "ConditionEvaluationError", "message": str(e), "iteration": iteration}
                )
                break

            if not condition_result:
                logger.debug("WHILE condition false, exiting loop after %d iterations", iteration)
                break

            logger.debug("WHILE loop iteration %d", iteration)

            # Set iterator variable if specified
            if config.iterator_variable:
                self.variables[config.iterator_variable] = iteration

            try:
                # Execute actions in this iteration
                exec_result = self._execute_action_sequence(config.actions)
                result["errors"].extend(exec_result["errors"])

                # Check if we should break on error
                if exec_result["errors"] and config.break_on_error:
                    logger.warning("Breaking loop due to error (break_on_error=True)")
                    break

            except ContinueLoop as e:
                logger.debug("Continue to next iteration: %s", e.message)
                result["iterations_completed"] += 1
                iteration += 1
                continue

            except BreakLoop:
                # Let BreakLoop propagate up
                result["iterations_completed"] += 1
                raise

            result["iterations_completed"] += 1
            iteration += 1

        if iteration >= max_iterations:
            logger.warning("WHILE loop hit max_iterations (%d), stopping", max_iterations)
            result["errors"].append(
                {
                    "type": "MaxIterationsExceeded",
                    "message": f"Hit max_iterations: {max_iterations}",
                }
            )

        return result

    def _execute_foreach_loop(self, config: LoopActionConfig) -> dict[str, Any]:
        """Execute a FOREACH loop (iterate over collection).

        Args:
            config: Loop configuration

        Returns:
            Partial result dictionary with iterations and errors
        """
        if config.collection is None:
            raise ValueError("FOREACH loop requires 'collection' to be specified")

        # Get collection items
        try:
            items = self._get_collection(config.collection)
        except Exception as e:
            logger.error("Failed to get collection for FOREACH: %s", str(e))
            return {
                "iterations_completed": 0,
                "errors": [{"type": type(e).__name__, "message": str(e)}],
            }

        if not items:
            logger.info("FOREACH collection is empty, skipping loop")
            return {"iterations_completed": 0, "errors": []}

        max_iterations = config.max_iterations or len(items)
        if len(items) > max_iterations:
            logger.warning(
                "FOREACH collection size (%d) exceeds max_iterations (%d), truncating",
                len(items),
                max_iterations,
            )
            items = items[:max_iterations]

        result = {"iterations_completed": 0, "errors": []}

        for index, item in enumerate(items):
            logger.debug("FOREACH iteration %d/%d, item=%s", index + 1, len(items), item)

            # Set iterator variable if specified
            if config.iterator_variable:
                self.variables[config.iterator_variable] = item
                logger.debug("Set %s = %s", config.iterator_variable, item)

            try:
                # Execute actions in this iteration
                exec_result = self._execute_action_sequence(config.actions)
                result["errors"].extend(exec_result["errors"])

                # Check if we should break on error
                if exec_result["errors"] and config.break_on_error:
                    logger.warning("Breaking loop due to error (break_on_error=True)")
                    break

            except ContinueLoop as e:
                logger.debug("Continue to next iteration: %s", e.message)
                result["iterations_completed"] += 1
                continue

            except BreakLoop:
                # Let BreakLoop propagate up
                result["iterations_completed"] += 1
                raise

            result["iterations_completed"] += 1

        return result

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def evaluate_condition(self, condition: ConditionConfig) -> bool:
        """Evaluate a condition and return boolean result.

        Supports multiple condition types:
        - image_exists: Check if image is found on screen
        - image_vanished: Check if image is NOT found on screen
        - text_exists: Check if text is found on screen
        - variable: Compare variable value
        - expression: Evaluate Python expression

        Args:
            condition: Condition configuration

        Returns:
            True if condition is met, False otherwise

        Raises:
            ValueError: If condition type is unknown or configuration invalid
        """
        logger.debug("Evaluating condition: type=%s", condition.type)

        if condition.type == "variable":
            return self._evaluate_variable_condition(condition)

        elif condition.type == "expression":
            return self._evaluate_expression_condition(condition)

        elif condition.type == "image_exists":
            return self._evaluate_image_exists_condition(condition)

        elif condition.type == "image_vanished":
            return not self._evaluate_image_exists_condition(condition)

        elif condition.type == "text_exists":
            return self._evaluate_text_exists_condition(condition)

        else:
            raise ValueError(f"Unknown condition type: {condition.type}")

    def _evaluate_variable_condition(self, condition: ConditionConfig) -> bool:
        """Evaluate a variable-based condition.

        Args:
            condition: Condition configuration

        Returns:
            Boolean result of comparison
        """
        if not condition.variable_name:
            raise ValueError("Variable condition requires 'variable_name'")

        var_name = condition.variable_name
        if var_name not in self.variables:
            logger.warning("Variable '%s' not found in context, treating as None", var_name)
            var_value = None
        else:
            var_value = self.variables[var_name]

        expected = condition.expected_value
        operator = condition.operator or "=="

        logger.debug(
            "Variable condition: %s %s %s (actual=%s)", var_name, operator, expected, var_value
        )

        return self._compare_values(var_value, operator, expected)

    def _evaluate_expression_condition(self, condition: ConditionConfig) -> bool:
        """Evaluate a Python expression condition.

        Args:
            condition: Condition configuration

        Returns:
            Boolean result of expression evaluation

        Raises:
            ValueError: If expression is invalid
        """
        if not condition.expression:
            raise ValueError("Expression condition requires 'expression'")

        expression = condition.expression
        logger.debug("Evaluating expression: %s", expression)

        try:
            # Create safe evaluation context with variables
            eval_context = {"variables": self.variables, **self.variables}
            result = eval(expression, {"__builtins__": {}}, eval_context)
            logger.debug("Expression result: %s", result)
            return bool(result)

        except Exception as e:
            logger.error("Expression evaluation failed: %s", str(e))
            raise ValueError(f"Invalid expression '{expression}': {e}") from e

    def _evaluate_image_exists_condition(self, condition: ConditionConfig) -> bool:
        """Evaluate an image_exists condition.

        This is a placeholder implementation. In a full system, this would
        use the image finding subsystem to search for the image.

        Args:
            condition: Condition configuration

        Returns:
            True if image exists, False otherwise
        """
        if not condition.image_id:
            raise ValueError("Image condition requires 'image_id'")

        logger.debug("Image exists check: image_id=%s", condition.image_id)

        # TODO: Integrate with actual image finding system
        # For now, return False (image not found)
        logger.warning("Image finding not implemented, returning False")
        return False

    def _evaluate_text_exists_condition(self, condition: ConditionConfig) -> bool:
        """Evaluate a text_exists condition.

        This is a placeholder implementation. In a full system, this would
        use OCR to search for text on screen.

        Args:
            condition: Condition configuration

        Returns:
            True if text exists, False otherwise
        """
        if not condition.text:
            raise ValueError("Text condition requires 'text'")

        logger.debug("Text exists check: text=%s", condition.text)

        # TODO: Integrate with actual OCR system
        # For now, return False (text not found)
        logger.warning("Text finding not implemented, returning False")
        return False

    def _compare_values(self, actual: Any, operator: str, expected: Any) -> bool:
        """Compare two values using the specified operator.

        Args:
            actual: Actual value
            operator: Comparison operator (==, !=, >, <, >=, <=, contains, matches)
            expected: Expected value

        Returns:
            Result of comparison

        Raises:
            ValueError: If operator is unknown
        """
        if operator == "==":
            return actual == expected
        elif operator == "!=":
            return actual != expected
        elif operator == ">":
            return actual > expected
        elif operator == "<":
            return actual < expected
        elif operator == ">=":
            return actual >= expected
        elif operator == "<=":
            return actual <= expected
        elif operator == "contains":
            return expected in actual
        elif operator == "matches":
            import re

            return bool(re.match(expected, str(actual)))
        else:
            raise ValueError(f"Unknown operator: {operator}")

    def _get_collection(self, collection_config) -> list[Any]:
        """Get collection items for FOREACH loop.

        Args:
            collection_config: Collection configuration from LoopActionConfig

        Returns:
            List of items to iterate over

        Raises:
            ValueError: If collection configuration is invalid
        """
        logger.debug("Getting collection: type=%s", collection_config.type)

        if collection_config.type == "variable":
            if not collection_config.variable_name:
                raise ValueError("Variable collection requires 'variable_name'")

            var_name = collection_config.variable_name
            if var_name not in self.variables:
                raise ValueError(f"Collection variable '{var_name}' not found")

            items = self.variables[var_name]
            if not isinstance(items, list | tuple):
                raise ValueError(f"Collection variable '{var_name}' is not iterable")

            return list(items)

        elif collection_config.type == "range":
            start = collection_config.start or 0
            end = collection_config.end
            step = collection_config.step or 1

            if end is None:
                raise ValueError("Range collection requires 'end'")

            return list(range(start, end, step))

        elif collection_config.type == "matches":
            # TODO: Integrate with image finding system
            logger.warning("Match-based collections not implemented yet, returning empty list")
            return []

        else:
            raise ValueError(f"Unknown collection type: {collection_config.type}")

    def _execute_action_sequence(self, action_ids: list[str]) -> dict[str, Any]:
        """Execute a sequence of actions.

        Args:
            action_ids: List of action IDs to execute

        Returns:
            Dictionary containing:
                - actions_executed: Number of actions executed
                - errors: List of errors encountered
        """
        result = {"actions_executed": 0, "errors": []}

        if not self.action_executor:
            logger.warning("No action executor configured, skipping %d actions", len(action_ids))
            return result

        for action_id in action_ids:
            logger.debug("Executing action: %s", action_id)

            try:
                # Execute action with current variables
                action_result = self.action_executor(action_id, self.variables)

                # Check for success
                if not action_result.get("success", True):
                    logger.warning("Action %s failed", action_id)
                    result["errors"].append(
                        {
                            "action_id": action_id,
                            "message": action_result.get("error", "Action failed"),
                        }
                    )

                result["actions_executed"] += 1

            except (BreakLoop, ContinueLoop):
                # Let control flow exceptions propagate
                result["actions_executed"] += 1
                raise

            except Exception as e:
                logger.error("Action %s raised exception: %s", action_id, str(e), exc_info=True)
                result["errors"].append(
                    {"action_id": action_id, "type": type(e).__name__, "message": str(e)}
                )
                result["actions_executed"] += 1

        return result

    # ========================================================================
    # Variable Management
    # ========================================================================

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the execution context.

        Args:
            name: Variable name
            value: Variable value
        """
        logger.debug("Setting variable: %s = %s", name, value)
        self.variables[name] = value

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a variable from the execution context.

        Args:
            name: Variable name
            default: Default value if variable not found

        Returns:
            Variable value or default
        """
        return self.variables.get(name, default)

    def clear_variables(self) -> None:
        """Clear all variables from the execution context."""
        logger.debug("Clearing all variables")
        self.variables.clear()

    def get_all_variables(self) -> dict[str, Any]:
        """Get all variables in the execution context.

        Returns:
            Copy of variables dictionary
        """
        return self.variables.copy()
