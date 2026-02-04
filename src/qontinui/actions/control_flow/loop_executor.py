"""Loop executor for control flow operations.

This module provides the LoopExecutor class which handles execution of all loop
types (FOR, WHILE, FOREACH) with proper variable management, condition evaluation,
and error handling.
"""

import logging
import time
from typing import Any

from qontinui.config import LoopActionConfig, LoopCollection
from qontinui.orchestration.execution_context import ExecutionContext

from .condition_evaluator import ConditionEvaluator
from .exceptions import BreakLoop, ContinueLoop

logger = logging.getLogger(__name__)


class LoopExecutor:
    """Executor for loop operations (FOR, WHILE, FOREACH).

    This class manages execution of all loop types including:
    - FOR loops: Fixed iteration count
    - WHILE loops: Condition-based iteration
    - FOREACH loops: Iterate over collections

    The executor uses an ExecutionContext to manage variables and track execution
    state. It handles loop control flow exceptions (BreakLoop, ContinueLoop) and
    integrates with ConditionEvaluator for WHILE loop conditions.

    Example:
        >>> context = ExecutionContext()
        >>> executor = LoopExecutor(context)
        >>> config = LoopActionConfig(
        ...     loop_type="FOR",
        ...     iterations=10,
        ...     iterator_variable="i",
        ...     actions=["action-1", "action-2"]
        ... )
        >>> result = executor.execute_loop(config)
    """

    def __init__(self, context: ExecutionContext) -> None:
        """Initialize the loop executor.

        Args:
            context: ExecutionContext providing variable management and execution callback
        """
        self.context = context
        self.condition_evaluator = ConditionEvaluator(context)
        logger.debug("LoopExecutor initialized with context")

    def execute_loop(self, config: LoopActionConfig, action_id: str) -> dict[str, Any]:
        """Execute a LOOP action (FOR, WHILE, or FOREACH).

        Args:
            config: Loop configuration specifying type and parameters
            action_id: ID of the loop action (for logging and results)

        Returns:
            Dictionary containing:
                - success: Whether loop completed successfully
                - iterations_completed: Number of iterations executed
                - stopped_early: Whether loop was broken early
                - errors: List of errors encountered (if any)
                - duration_ms: Total execution time in milliseconds
                - action_id: The action ID
                - loop_type: The type of loop executed

        Raises:
            ValueError: If loop configuration is invalid
        """
        logger.info(
            "Starting %s loop (action_id=%s, max_iterations=%s)",
            config.loop_type,
            action_id,
            config.max_iterations,
        )

        errors_list: list[dict[str, Any]] = []
        result: dict[str, Any] = {
            "success": True,
            "iterations_completed": 0,
            "stopped_early": False,
            "errors": errors_list,
            "action_id": action_id,
            "loop_type": config.loop_type,
        }

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

        except ValueError as e:
            logger.error("Loop configuration error: %s", str(e))
            result["success"] = False
            if isinstance(result["errors"], list):
                errors_list.append({"type": "ValueError", "message": str(e)})

        except RuntimeError as e:
            logger.error("Loop execution error: %s", str(e))
            result["success"] = False
            if isinstance(result["errors"], list):
                errors_list.append({"type": "RuntimeError", "message": str(e)})

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

    def _execute_for_loop(self, config: LoopActionConfig) -> dict[str, Any]:
        """Execute a FOR loop (fixed iteration count).

        Args:
            config: Loop configuration with iterations count

        Returns:
            Partial result dictionary with iterations_completed and errors

        Raises:
            ValueError: If iterations is not specified
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

        errors_list: list[dict[str, Any]] = []
        result: dict[str, Any] = {"iterations_completed": 0, "errors": errors_list}

        for i in range(iterations):
            logger.debug("FOR loop iteration %d/%d", i + 1, iterations)

            # Set iterator variable if specified
            if config.iterator_variable:
                self.context.set_variable(config.iterator_variable, i)
                logger.debug("Set %s = %d", config.iterator_variable, i)

            try:
                # Execute actions in this iteration
                exec_result = self._execute_action_sequence(config.actions)
                errors_list.extend(exec_result["errors"])

                # Check if we should break on error
                if exec_result["errors"] and config.break_on_error:
                    logger.warning("Breaking loop due to error (break_on_error=True)")
                    break

            except ContinueLoop as e:
                logger.debug("Continue to next iteration: %s", e.message)
                result["iterations_completed"] = result["iterations_completed"] + 1  # type: ignore[assignment]
                continue

            except BreakLoop:
                # Let BreakLoop propagate up
                result["iterations_completed"] = result["iterations_completed"] + 1  # type: ignore[assignment]
                raise

            result["iterations_completed"] = result["iterations_completed"] + 1  # type: ignore[assignment]

        return result

    def _execute_while_loop(self, config: LoopActionConfig) -> dict[str, Any]:
        """Execute a WHILE loop (condition-based).

        Args:
            config: Loop configuration with condition

        Returns:
            Partial result dictionary with iterations_completed and errors

        Raises:
            ValueError: If condition is not specified
        """
        if config.condition is None:
            raise ValueError("WHILE loop requires 'condition' to be specified")

        max_iterations = config.max_iterations or 10000  # Safety limit
        errors_list: list[dict[str, Any]] = []
        result: dict[str, Any] = {"iterations_completed": 0, "errors": errors_list}

        iteration = 0
        while iteration < max_iterations:
            # Evaluate condition
            try:
                condition_result = self.condition_evaluator.evaluate_condition(config.condition)
            except ValueError as e:
                logger.error("Failed to evaluate WHILE condition: %s", str(e))
                errors_list.append(
                    {
                        "type": "ConditionEvaluationError",
                        "message": str(e),
                        "iteration": iteration,
                    }
                )
                break

            if not condition_result:
                logger.debug("WHILE condition false, exiting loop after %d iterations", iteration)
                break

            logger.debug("WHILE loop iteration %d", iteration)

            # Set iterator variable if specified
            if config.iterator_variable:
                self.context.set_variable(config.iterator_variable, iteration)

            try:
                # Execute actions in this iteration
                exec_result = self._execute_action_sequence(config.actions)
                errors_list.extend(exec_result["errors"])

                # Check if we should break on error
                if exec_result["errors"] and config.break_on_error:
                    logger.warning("Breaking loop due to error (break_on_error=True)")
                    break

            except ContinueLoop as e:
                logger.debug("Continue to next iteration: %s", e.message)
                result["iterations_completed"] = result["iterations_completed"] + 1  # type: ignore[assignment]
                iteration += 1
                continue

            except BreakLoop:
                # Let BreakLoop propagate up
                result["iterations_completed"] = result["iterations_completed"] + 1  # type: ignore[assignment]
                raise

            result["iterations_completed"] = result["iterations_completed"] + 1  # type: ignore[assignment]
            iteration += 1

        if iteration >= max_iterations:
            logger.warning("WHILE loop hit max_iterations (%d), stopping", max_iterations)
            errors_list.append(
                {
                    "type": "MaxIterationsExceeded",
                    "message": f"Hit max_iterations: {max_iterations}",
                }
            )

        return result

    def _execute_foreach_loop(self, config: LoopActionConfig) -> dict[str, Any]:
        """Execute a FOREACH loop (iterate over collection).

        Args:
            config: Loop configuration with collection

        Returns:
            Partial result dictionary with iterations_completed and errors

        Raises:
            ValueError: If collection is not specified
        """
        if config.collection is None:
            raise ValueError("FOREACH loop requires 'collection' to be specified")

        # Get collection items
        try:
            items = self._get_collection(config.collection)
        except ValueError as e:
            logger.error("Failed to get collection for FOREACH: %s", str(e))
            return {
                "iterations_completed": 0,
                "errors": [{"type": "ValueError", "message": str(e)}],
            }
        except KeyError as e:
            logger.error("Collection variable not found for FOREACH: %s", str(e))
            return {
                "iterations_completed": 0,
                "errors": [{"type": "KeyError", "message": str(e)}],
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

        errors_list: list[dict[str, Any]] = []
        result: dict[str, Any] = {"iterations_completed": 0, "errors": errors_list}

        for index, item in enumerate(items):
            logger.debug("FOREACH iteration %d/%d, item=%s", index + 1, len(items), item)

            # Set iterator variable if specified
            if config.iterator_variable:
                self.context.set_variable(config.iterator_variable, item)
                logger.debug("Set %s = %s", config.iterator_variable, item)

            try:
                # Execute actions in this iteration
                exec_result = self._execute_action_sequence(config.actions)
                errors_list.extend(exec_result["errors"])

                # Check if we should break on error
                if exec_result["errors"] and config.break_on_error:
                    logger.warning("Breaking loop due to error (break_on_error=True)")
                    break

            except ContinueLoop as e:
                logger.debug("Continue to next iteration: %s", e.message)
                result["iterations_completed"] = result["iterations_completed"] + 1  # type: ignore[assignment]
                continue

            except BreakLoop:
                # Let BreakLoop propagate up
                result["iterations_completed"] = result["iterations_completed"] + 1  # type: ignore[assignment]
                raise

            result["iterations_completed"] = result["iterations_completed"] + 1  # type: ignore[assignment]

        return result

    def _get_collection(self, collection_config: LoopCollection) -> list[Any]:
        """Get collection items for FOREACH loop.

        Supports three collection types:
        - variable: Read from a variable containing a list/tuple
        - range: Generate a range of numbers
        - matches: Get matches from image finding (not yet implemented)

        Args:
            collection_config: Collection configuration from LoopActionConfig

        Returns:
            List of items to iterate over

        Raises:
            ValueError: If collection configuration is invalid or variable is not iterable
            KeyError: If collection variable is not found
        """
        logger.debug("Getting collection: type=%s", collection_config.type)

        if collection_config.type == "variable":
            if not collection_config.variable_name:
                raise ValueError("Variable collection requires 'variable_name'")

            var_name = collection_config.variable_name
            if not self.context.has_variable(var_name):
                raise KeyError(f"Collection variable '{var_name}' not found")

            items = self.context.get_variable(var_name)
            if not isinstance(items, list | tuple):
                raise ValueError(
                    f"Collection variable '{var_name}' is not iterable (got {type(items).__name__})"
                )

            return list(items)

        elif collection_config.type == "range":
            start = collection_config.start or 0
            end = collection_config.end
            step = collection_config.step or 1

            if end is None:
                raise ValueError("Range collection requires 'end'")

            if step == 0:
                raise ValueError("Range collection 'step' cannot be zero")

            return list(range(start, end, step))

        elif collection_config.type == "matches":
            # Placeholder: Match-based collections require image finding integration
            # Integration point: Use Find action to get matches, then extract locations/regions
            # Example: find_result = find_action.perform(...); return [m.target for m in find_result.matches]
            logger.warning("Match-based collections not implemented yet, returning empty list")
            return []

        else:
            raise ValueError(f"Unknown collection type: {collection_config.type}")

    def _execute_action_sequence(self, action_ids: list[str]) -> dict[str, Any]:
        """Execute a sequence of actions in the current loop iteration.

        Uses the ExecutionContext's execute_action callback to execute each action.
        Handles control flow exceptions (BreakLoop, ContinueLoop) by propagating them
        up to the loop handler.

        Args:
            action_ids: List of action IDs to execute in sequence

        Returns:
            Dictionary containing:
                - actions_executed: Number of actions executed
                - errors: List of errors encountered during execution
        """
        errors_list: list[dict[str, Any]] = []
        result: dict[str, Any] = {"actions_executed": 0, "errors": errors_list}

        # Check if execute_action callback is available
        if not hasattr(self.context, "execute_action") or not callable(self.context.execute_action):
            logger.warning(
                "No execute_action callback in context, skipping %d actions",
                len(action_ids),
            )
            return result

        for action_id in action_ids:
            logger.debug("Executing action: %s", action_id)

            try:
                # Execute action using context callback
                action_result = self.context.execute_action(action_id)

                # Check for success
                if not action_result.get("success", True):
                    logger.warning("Action %s failed", action_id)
                    errors_list.append(
                        {
                            "action_id": action_id,
                            "message": action_result.get("error", "Action failed"),
                        }
                    )

                result["actions_executed"] = result["actions_executed"] + 1  # type: ignore[assignment]

            except (BreakLoop, ContinueLoop):
                # Let control flow exceptions propagate to loop handler
                result["actions_executed"] = result["actions_executed"] + 1  # type: ignore[assignment]
                raise

            except ValueError as e:
                logger.error("Action %s raised ValueError: %s", action_id, str(e))
                errors_list.append(
                    {"action_id": action_id, "type": "ValueError", "message": str(e)}
                )
                result["actions_executed"] = result["actions_executed"] + 1  # type: ignore[assignment]

            except RuntimeError as e:
                logger.error("Action %s raised RuntimeError: %s", action_id, str(e))
                errors_list.append(
                    {"action_id": action_id, "type": "RuntimeError", "message": str(e)}
                )
                result["actions_executed"] = result["actions_executed"] + 1  # type: ignore[assignment]

            except KeyError as e:
                logger.error("Action %s raised KeyError: %s", action_id, str(e))
                errors_list.append({"action_id": action_id, "type": "KeyError", "message": str(e)})
                result["actions_executed"] = result["actions_executed"] + 1  # type: ignore[assignment]

        return result
