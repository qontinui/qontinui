"""Control flow action executor adapter for the registry system.

This module provides an adapter that integrates the existing ControlFlowExecutor
from qontinui.actions.control_flow into the new registry-based action executor
architecture. It bridges the gap between the old monolithic design and the new
command pattern approach.

The adapter wraps the existing ControlFlowExecutor and translates between:
- New: ActionExecutorBase interface with ExecutionContext
- Old: ControlFlowExecutor with action_executor callback and variables dict
"""

import logging
from typing import Any

from ..actions.control_flow import BreakLoop, ContinueLoop
from ..actions.control_flow import ControlFlowExecutor as LegacyControlFlowExecutor
from ..config import (
    Action,
    BreakActionConfig,
    ContinueActionConfig,
    IfActionConfig,
    LoopActionConfig,
    SwitchActionConfig,
    TryCatchActionConfig,
)
from ..exceptions import ActionExecutionError
from .base import ActionExecutorBase, ExecutionContext
from .registry import register_executor

logger = logging.getLogger(__name__)


@register_executor
class ControlFlowExecutorAdapter(ActionExecutorBase):
    """Adapter for integrating ControlFlowExecutor into the registry system.

    This adapter wraps the existing ControlFlowExecutor and provides the
    ActionExecutorBase interface expected by the new registry system. It
    handles the translation of dependencies and method signatures between
    the old and new architectures.

    Architecture:
        - Uses adapter pattern to bridge old and new designs
        - Wraps existing ControlFlowExecutor without modifying it
        - Translates ExecutionContext to ControlFlowExecutor dependencies
        - Converts between new and old method signatures

    Supported Actions:
        - LOOP: FOR, WHILE, and FOREACH loops with break/continue support
        - IF: Conditional branching with condition evaluation
        - SWITCH: Case-based branching with multiple conditions
        - TRY_CATCH: Error handling with try, catch, and finally blocks
        - BREAK: Exit current loop (raises BreakLoop exception)
        - CONTINUE: Skip to next loop iteration (raises ContinueLoop exception)

    Example:
        context = ExecutionContext(...)
        adapter = ControlFlowExecutorAdapter(context)

        # Execute a loop
        loop_action = Action(type="LOOP", config={
            "loopType": "FOR",
            "iterations": 5,
            "iteratorVariable": "i",
            "actions": ["action-1", "action-2"]
        })
        success = adapter.execute(loop_action, LoopActionConfig(...))

        # Execute an IF condition
        if_action = Action(type="IF", config={
            "condition": {"type": "variable", "variableName": "count", "expectedValue": 0},
            "thenActions": ["action-1"],
            "elseActions": ["action-2"]
        })
        success = adapter.execute(if_action, IfActionConfig(...))
    """

    def __init__(self, context: ExecutionContext) -> None:
        """Initialize adapter with execution context.

        Creates a wrapped ControlFlowExecutor with translated dependencies.

        Args:
            context: Execution context containing all dependencies
        """
        super().__init__(context)

        # Create action executor callback that bridges to new system
        action_executor_callback = self._create_action_executor_callback()

        # Get initial variables from variable context
        initial_variables = {}
        if context.variable_context:
            initial_variables = context.variable_context.get_all_variables()

        # Create wrapped executor with translated dependencies
        self._wrapped_executor = LegacyControlFlowExecutor(
            action_executor=action_executor_callback, variables=initial_variables
        )

        logger.debug("ControlFlowExecutorAdapter initialized")

    def get_supported_action_types(self) -> list[str]:
        """Get list of control flow action types this executor handles.

        Returns:
            List containing: LOOP, IF, SWITCH, TRY_CATCH, BREAK, CONTINUE
        """
        return ["LOOP", "IF", "SWITCH", "TRY_CATCH", "BREAK", "CONTINUE"]

    async def execute(self, action: Action, typed_config: Any) -> bool:
        """Execute control flow action with validated configuration.

        Delegates to the wrapped ControlFlowExecutor after synchronizing
        variable state and translating exceptions.

        Args:
            action: Pydantic Action model with type, config, etc.
            typed_config: Type-specific validated configuration object

        Returns:
            True if action succeeded, False otherwise

        Raises:
            ActionExecutionError: If action execution fails critically
        """
        action_type = action.type
        logger.debug(f"Executing control flow action: {action_type}")

        try:
            # Sync variables before execution
            self._sync_variables_to_wrapped()

            # Route to appropriate handler
            if action_type == "LOOP":
                return self._execute_loop(action, typed_config)
            elif action_type == "IF":
                return self._execute_if(action, typed_config)
            elif action_type == "SWITCH":
                return self._execute_switch(action, typed_config)
            elif action_type == "TRY_CATCH":
                return self._execute_try_catch(action, typed_config)
            elif action_type == "BREAK":
                return self._execute_break(action, typed_config)
            elif action_type == "CONTINUE":
                return self._execute_continue(action, typed_config)
            else:
                raise ActionExecutionError(
                    action_type=action_type,
                    reason=f"Unsupported control flow action type: {action_type}",
                )

        except (BreakLoop, ContinueLoop):
            # Let control flow exceptions propagate
            # These are part of normal loop control flow
            raise

        except ActionExecutionError:
            # Re-raise our own exceptions
            raise

        except Exception as e:
            logger.error(f"Unexpected error executing {action_type}: {e}", exc_info=True)
            raise ActionExecutionError(
                action_type=action_type,
                reason=f"Control flow execution failed: {e}",
            ) from e

        finally:
            # Sync variables back after execution
            self._sync_variables_from_wrapped()

    # ========================================================================
    # Action Type Handlers
    # ========================================================================

    def _execute_loop(self, action: Action, typed_config: LoopActionConfig) -> bool:
        """Execute LOOP action via wrapped executor.

        Args:
            action: Action model
            typed_config: Validated LoopActionConfig

        Returns:
            True if loop completed successfully
        """
        logger.info(
            f"Executing LOOP action: type={typed_config.loop_type}, "
            f"max_iterations={typed_config.max_iterations}"
        )

        try:
            result = self._wrapped_executor.execute_loop(action)

            # Check success
            if result.get("success"):
                self._emit_action_success(
                    action,
                    {
                        "iterations_completed": result.get("iterations_completed", 0),
                        "stopped_early": result.get("stopped_early", False),
                        "loop_type": result.get("loop_type"),
                    },
                )
                return True
            else:
                self._emit_action_failure(
                    action,
                    "Loop execution failed",
                    {
                        "iterations_completed": result.get("iterations_completed", 0),
                        "errors": result.get("errors", []),
                    },
                )
                return False

        except BreakLoop:
            # BreakLoop is normal control flow, not an error
            logger.debug("Loop terminated with BREAK")
            self._emit_action_success(action, {"terminated_by": "break"})
            raise  # Re-raise to propagate break

        except Exception as e:
            logger.error(f"Loop execution failed: {e}")
            self._emit_action_failure(action, str(e))
            return False

    def _execute_if(self, action: Action, typed_config: IfActionConfig) -> bool:
        """Execute IF action via wrapped executor.

        Args:
            action: Action model
            typed_config: Validated IfActionConfig

        Returns:
            True if condition evaluation and branch execution succeeded
        """
        logger.info("Executing IF action")

        try:
            result = self._wrapped_executor.execute_if(action)

            # Check success
            if result.get("success"):
                self._emit_action_success(
                    action,
                    {
                        "condition_result": result.get("condition_result"),
                        "branch_taken": result.get("branch_taken"),
                        "actions_executed": result.get("actions_executed", 0),
                    },
                )
                return True
            else:
                self._emit_action_failure(
                    action,
                    "IF execution failed",
                    {
                        "condition_result": result.get("condition_result"),
                        "errors": result.get("errors", []),
                    },
                )
                return False

        except Exception as e:
            logger.error(f"IF execution failed: {e}")
            self._emit_action_failure(action, str(e))
            return False

    def _execute_switch(self, action: Action, typed_config: SwitchActionConfig) -> bool:
        """Execute SWITCH action via wrapped executor.

        Args:
            action: Action model
            typed_config: Validated SwitchActionConfig

        Returns:
            True if switch evaluation and branch execution succeeded
        """
        logger.info("Executing SWITCH action")

        try:
            result = self._wrapped_executor.execute_switch(action)

            # Check success
            if result.get("success"):
                self._emit_action_success(
                    action,
                    {
                        "expression_value": result.get("expression_value"),
                        "matched_case": result.get("matched_case"),
                        "case_index": result.get("case_index"),
                        "actions_executed": result.get("actions_executed", 0),
                    },
                )
                return True
            else:
                self._emit_action_failure(
                    action,
                    "SWITCH execution failed",
                    {
                        "expression_value": result.get("expression_value"),
                        "errors": result.get("errors", []),
                    },
                )
                return False

        except Exception as e:
            logger.error(f"SWITCH execution failed: {e}")
            self._emit_action_failure(action, str(e))
            return False

    def _execute_try_catch(self, action: Action, typed_config: TryCatchActionConfig) -> bool:
        """Execute TRY_CATCH action via wrapped executor.

        Args:
            action: Action model
            typed_config: Validated TryCatchActionConfig

        Returns:
            True if execution succeeded overall
        """
        logger.info("Executing TRY_CATCH action")

        try:
            result = self._wrapped_executor.execute_try_catch(action)

            # Check success
            if result.get("success"):
                self._emit_action_success(
                    action,
                    {
                        "branch_taken": result.get("branch_taken"),
                        "try_actions_executed": result.get("try_actions_executed", 0),
                        "catch_actions_executed": result.get("catch_actions_executed", 0),
                        "finally_actions_executed": result.get("finally_actions_executed", 0),
                        "error_caught": result.get("error_caught"),
                    },
                )
                return True
            else:
                self._emit_action_failure(
                    action,
                    "TRY_CATCH execution failed",
                    {
                        "branch_taken": result.get("branch_taken"),
                        "error_caught": result.get("error_caught"),
                        "errors": result.get("errors", []),
                    },
                )
                return False

        except Exception as e:
            logger.error(f"TRY_CATCH execution failed: {e}")
            self._emit_action_failure(action, str(e))
            return False

    def _execute_break(self, action: Action, typed_config: BreakActionConfig) -> bool:
        """Execute BREAK action via wrapped executor.

        Args:
            action: Action model
            typed_config: Validated BreakActionConfig

        Returns:
            Never returns normally - always raises BreakLoop

        Raises:
            BreakLoop: Always raises to signal loop break
        """
        logger.info("Executing BREAK action")

        try:
            # This will raise BreakLoop if condition is met (or no condition)
            self._wrapped_executor.execute_break(action)
            # If we get here, the break condition was not met
            logger.debug("BREAK condition not met, continuing")
            return True

        except BreakLoop:
            # Expected exception - re-raise to propagate
            self._emit_action_success(
                action,
                {"message": (typed_config.message if typed_config else "Break triggered")},
            )
            raise

    def _execute_continue(self, action: Action, typed_config: ContinueActionConfig) -> bool:
        """Execute CONTINUE action via wrapped executor.

        Args:
            action: Action model
            typed_config: Validated ContinueActionConfig

        Returns:
            Never returns normally - always raises ContinueLoop

        Raises:
            ContinueLoop: Always raises to signal iteration skip
        """
        logger.info("Executing CONTINUE action")

        try:
            # This will raise ContinueLoop if condition is met (or no condition)
            self._wrapped_executor.execute_continue(action)
            # If we get here, the continue condition was not met
            logger.debug("CONTINUE condition not met, continuing")
            return True

        except ContinueLoop:
            # Expected exception - re-raise to propagate
            self._emit_action_success(
                action,
                {"message": (typed_config.message if typed_config else "Continue triggered")},
            )
            raise

    # ========================================================================
    # Dependency Translation
    # ========================================================================

    def _create_action_executor_callback(self):
        """Create action executor callback that bridges to new system.

        The old ControlFlowExecutor expects a callback with signature:
            (action_id: str, variables: dict) -> dict

        This creates a wrapper that translates from that signature to the new
        execute_action callback in ExecutionContext.

        Returns:
            Callback function compatible with ControlFlowExecutor
        """

        def callback(action_id: str, variables: dict[str, Any]) -> dict[str, Any]:
            """Execute action by ID with variables.

            Args:
                action_id: Action ID to execute
                variables: Variable context for execution

            Returns:
                Result dictionary with at least {"success": bool}
            """
            logger.debug(f"Control flow executing nested action: {action_id}")

            try:
                # Update variable context before execution
                if self.context.variable_context:
                    for key, value in variables.items():
                        self.context.variable_context.set(key, value)

                # Get action from config
                action = self._get_action_by_id(action_id)
                if not action:
                    logger.error(f"Action not found: {action_id}")
                    return {"success": False, "error": f"Action not found: {action_id}"}

                # Execute via context callback
                success = self.context.execute_action(action)

                return {"success": success}

            except Exception as e:
                logger.error(f"Error executing action {action_id}: {e}")
                return {"success": False, "error": str(e)}

        return callback

    def _get_action_by_id(self, action_id: str) -> Action | None:
        """Get action from config by ID.

        Args:
            action_id: Action ID to find

        Returns:
            Action model or None if not found
        """
        # Access action_map through context.config
        if hasattr(self.context.config, "action_map"):
            action_map = self.context.config.action_map
            return action_map.get(action_id)  # type: ignore[no-any-return]
        return None

    def _sync_variables_to_wrapped(self) -> None:
        """Sync variables from ExecutionContext to wrapped executor.

        This ensures the wrapped executor has the latest variable state
        before executing an action.
        """
        if not self.context.variable_context:
            return

        try:
            # Get all variables from context
            variables = self.context.variable_context.get_all_variables()

            # Update wrapped executor's variables
            self._wrapped_executor.variables.update(variables)
            logger.debug(f"Synced {len(variables)} variables to wrapped executor")

        except Exception as e:
            logger.warning(f"Failed to sync variables to wrapped executor: {e}")

    def _sync_variables_from_wrapped(self) -> None:
        """Sync variables from wrapped executor back to ExecutionContext.

        This ensures any variables set during control flow execution
        are propagated back to the main execution context.
        """
        if not self.context.variable_context:
            return

        try:
            # Get all variables from wrapped executor
            variables = self._wrapped_executor.get_all_variables()

            # Update context variables
            for key, value in variables.items():
                self.context.variable_context.set(key, value)

            logger.debug(f"Synced {len(variables)} variables from wrapped executor")

        except Exception as e:
            logger.warning(f"Failed to sync variables from wrapped executor: {e}")
