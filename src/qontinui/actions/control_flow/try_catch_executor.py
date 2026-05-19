"""Try-catch execution for error handling.

This module provides the TryCatchExecutor class which handles TRY_CATCH
error handling operations with try, catch, and finally blocks.
"""

import logging
from typing import Any

from qontinui.config import Action, TryCatchActionConfig
from qontinui.orchestration.execution_context import ExecutionContext

from .exceptions import BreakLoop, ContinueLoop

logger = logging.getLogger(__name__)


class TryCatchExecutor:
    """Executor for TRY_CATCH error handling operations.

    This class handles try-catch-finally error handling logic, executing
    try actions and falling back to catch actions on error. Finally actions
    are always executed regardless of success or failure. Error information
    can be captured to a variable for inspection in catch blocks.

    The executor maintains no state between executions - all state is
    managed by the ExecutionContext.

    Example:
        >>> context = ExecutionContext({"data": None})
        >>> executor = TryCatchExecutor(context)
        >>> try_catch_action = Action(
        ...     id='try-catch-1',
        ...     type='TRY_CATCH',
        ...     config={
        ...         'tryActions': ['risky-action-1'],
        ...         'catchActions': ['error-handler-1'],
        ...         'finallyActions': ['cleanup-1'],
        ...         'errorVariable': 'error'
        ...     }
        ... )
        >>> result = executor.execute_try_catch(try_catch_action)
        >>> print(result['branch_taken'])  # 'try' or 'catch'
    """

    def __init__(self, context: ExecutionContext) -> None:
        """Initialize the try-catch executor.

        Args:
            context: ExecutionContext providing variable access, action execution,
                    and state management
        """
        self.context = context
        logger.debug("TryCatchExecutor initialized with context")

    async def execute_try_catch(
        self, action: Action, config: TryCatchActionConfig
    ) -> dict[str, Any]:
        """Execute a TRY_CATCH action (error handling).

        Executes the try actions and falls back to catch actions if an error
        occurs. Finally actions are executed regardless of success or failure.
        Returns detailed execution statistics including which branch was taken
        and how many actions were executed.

        Args:
            action: The TRY_CATCH action to execute
            config: Validated TRY_CATCH action configuration

        Returns:
            Dictionary containing:
                - success: Whether execution succeeded overall
                - branch_taken: 'try' or 'catch'
                - try_actions_executed: Number of try actions executed
                - catch_actions_executed: Number of catch actions executed
                - finally_actions_executed: Number of finally actions executed
                - error_caught: Error information if an error was caught
                - errors: List of errors encountered (if any)
                - action_id: ID of the action executed

        Raises:
            Exception: Exceptions are caught and handled, not raised
        """
        logger.info("Executing TRY_CATCH (action_id=%s)", action.id)

        result: dict[str, Any] = {
            "success": True,
            "branch_taken": None,
            "try_actions_executed": 0,
            "catch_actions_executed": 0,
            "finally_actions_executed": 0,
            "error_caught": None,
            "errors": [],
            "action_id": action.id,
        }

        error_caught = None

        # Execute try block
        try:
            logger.debug("Executing TRY block with %d actions", len(config.try_actions))
            try_result = await self._execute_action_sequence(config.try_actions)
            result["try_actions_executed"] = try_result["actions_executed"]

            # Check if try block had errors
            if try_result["errors"]:
                # Treat errors in try block as caught errors
                error_caught = {
                    "type": "TryBlockError",
                    "message": "One or more actions in try block failed",
                    "details": try_result["errors"],
                }
                result["branch_taken"] = "catch"
                logger.debug("TRY block had errors, executing CATCH block")
            else:
                result["branch_taken"] = "try"
                logger.debug("TRY block succeeded")

        except Exception as e:
            # Unexpected exception during try block execution
            logger.warning("TRY block raised exception: %s", str(e), exc_info=True)
            error_caught = {
                "type": type(e).__name__,
                "message": str(e),
            }
            result["branch_taken"] = "catch"

        # Execute catch block if an error was caught
        if error_caught and config.catch_actions:
            # Set error variable if specified
            if config.error_variable:
                self.context.set_variable(config.error_variable, error_caught)
                logger.debug(
                    "Set error variable '%s' = %s", config.error_variable, error_caught
                )

            try:
                logger.debug(
                    "Executing CATCH block with %d actions", len(config.catch_actions)
                )
                catch_result = await self._execute_action_sequence(config.catch_actions)
                result["catch_actions_executed"] = catch_result["actions_executed"]
                result["error_caught"] = error_caught

                # Propagate catch errors to main errors list
                if catch_result["errors"]:
                    if isinstance(result["errors"], list):
                        result["errors"].extend(catch_result["errors"])
                    result["success"] = False

            except Exception as e:
                # Unexpected exception during catch block execution
                logger.error("CATCH block raised exception: %s", str(e), exc_info=True)
                result["success"] = False
                if isinstance(result["errors"], list):
                    result["errors"].append(
                        {
                            "type": "CatchBlockError",
                            "message": f"Catch block failed: {str(e)}",
                        }
                    )

        elif error_caught:
            # Error occurred but no catch block defined
            logger.debug("Error occurred but no CATCH block defined")
            result["error_caught"] = error_caught
            result["success"] = False
            if isinstance(result["errors"], list):
                result["errors"].append(error_caught)

        # Execute finally block (always executed)
        if config.finally_actions:
            try:
                logger.debug(
                    "Executing FINALLY block with %d actions",
                    len(config.finally_actions),
                )
                finally_result = await self._execute_action_sequence(
                    config.finally_actions
                )
                result["finally_actions_executed"] = finally_result["actions_executed"]

                # Finally errors are added but don't override success if try succeeded
                if finally_result["errors"]:
                    if isinstance(result["errors"], list):
                        result["errors"].extend(finally_result["errors"])
                    # Mark as failed only if not already successful
                    if result["branch_taken"] == "catch" or error_caught:
                        result["success"] = False

            except Exception as e:
                # Unexpected exception during finally block execution
                logger.error(
                    "FINALLY block raised exception: %s", str(e), exc_info=True
                )
                if isinstance(result["errors"], list):
                    result["errors"].append(
                        {
                            "type": "FinallyBlockError",
                            "message": f"Finally block failed: {str(e)}",
                        }
                    )
                # Finally failures always mark as failed
                result["success"] = False

        return result

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
