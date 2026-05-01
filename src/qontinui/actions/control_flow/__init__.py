"""Control flow action executor - Re-export module.

This module provides backward-compatible access to control flow components
that have been refactored into a modular structure.

The original monolithic ControlFlowExecutor has been split into specialized
executors for better separation of concerns:
- LoopExecutor: Handles FOR, WHILE, and FOREACH loops
- ConditionalExecutor: Handles IF/ELSE conditional branching
- SwitchExecutor: Handles SWITCH case-based branching
- TryCatchExecutor: Handles TRY_CATCH error handling
- FlowControlExecutor: Handles BREAK and CONTINUE statements

This module exports both the specialized executors and a wrapper ControlFlowExecutor
that maintains the original API for backward compatibility.

Exports:
    - ControlFlowExecutor: Main executor class (wrapper for backward compatibility)
    - BreakLoop: Exception raised to break out of loops
    - ContinueLoop: Exception raised to skip to next loop iteration
    - LoopExecutor: Specialized executor for loop operations
    - ConditionalExecutor: Specialized executor for conditional branching
    - SwitchExecutor: Specialized executor for case-based branching
    - TryCatchExecutor: Specialized executor for error handling
    - FlowControlExecutor: Specialized executor for flow control statements
    - ConditionEvaluator: Condition evaluation utility
"""

import logging
from collections.abc import Callable
from typing import Any

from qontinui.config import Action, get_typed_config

from .condition_evaluator import ConditionEvaluator
from .conditional_executor import ConditionalExecutor
from .exceptions import BreakLoop, ContinueLoop
from .flow_control_executor import FlowControlExecutor
from .loop_executor import LoopExecutor
from .switch_executor import SwitchExecutor
from .try_catch_executor import TryCatchExecutor

logger = logging.getLogger(__name__)

__all__ = [
    "ControlFlowExecutor",
    "BreakLoop",
    "ContinueLoop",
    "LoopExecutor",
    "ConditionalExecutor",
    "FlowControlExecutor",
    "TryCatchExecutor",
    "ConditionEvaluator",
    "SwitchExecutor",
]


class ControlFlowExecutor:
    """Backward-compatible wrapper for control flow execution.

    This class maintains the original ControlFlowExecutor API while delegating
    to specialized executors internally. It provides a unified interface for
    executing loop, conditional, and flow control actions.

    The executor manages its own variable context and provides action execution
    through a callback function, matching the original API signature.

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

    Architecture:
        This wrapper creates a simple ExecutionContext internally to bridge
        between the old callback-based API and the new context-based executors.
        It maintains the same public interface as the original ControlFlowExecutor
        for seamless migration.
    """

    def __init__(
        self,
        action_executor: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        variables: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the control flow executor.

        Args:
            action_executor: Function to execute actions by ID. Should accept
                           (action_id, variables) and return execution result.
                           If None, actions will be logged but not executed.
            variables: Initial variable context. If None, starts with empty dict.
        """
        self.action_executor = action_executor
        self.variables = variables if variables is not None else {}

        # Create minimal execution context for specialized executors
        self._context = self._create_execution_context()

        # Initialize specialized executors
        self._loop_executor = LoopExecutor(self._context)
        self._conditional_executor = ConditionalExecutor(self._context)
        self._flow_control_executor = FlowControlExecutor(self._context)
        self._switch_executor = SwitchExecutor(self._context)
        self._try_catch_executor = TryCatchExecutor(self._context)

        logger.debug(
            "ControlFlowExecutor initialized with %d variables", len(self.variables)
        )

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
        config = get_typed_config(action)
        return self._loop_executor.execute_loop(config, action.id)  # type: ignore[arg-type]

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
        config = get_typed_config(action)
        return self._conditional_executor.execute_if(action, config)  # type: ignore[arg-type]

    def execute_break(self, action: Action) -> None:
        """Execute a BREAK action (exit loop).

        Args:
            action: The BREAK action to execute

        Raises:
            BreakLoop: Always raises to signal loop break
        """
        self._flow_control_executor.execute_break(action)

    def execute_continue(self, action: Action) -> None:
        """Execute a CONTINUE action (skip to next iteration).

        Args:
            action: The CONTINUE action to execute

        Raises:
            ContinueLoop: Always raises to signal iteration skip
        """
        self._flow_control_executor.execute_continue(action)

    def execute_switch(self, action: Action) -> dict[str, Any]:
        """Execute a SWITCH action (case-based branching).

        Args:
            action: The SWITCH action to execute

        Returns:
            Dictionary containing:
                - success: Whether execution succeeded
                - expression_value: Result of expression evaluation
                - matched_case: Value of the matched case (or 'default')
                - case_index: Index of matched case (-1 for default)
                - actions_executed: Number of actions executed
                - errors: List of errors encountered (if any)
        """
        config = get_typed_config(action)
        return self._switch_executor.execute_switch(action, config)  # type: ignore[arg-type]

    def execute_try_catch(self, action: Action) -> dict[str, Any]:
        """Execute a TRY_CATCH action (error handling).

        Args:
            action: The TRY_CATCH action to execute

        Returns:
            Dictionary containing:
                - success: Whether execution succeeded overall
                - branch_taken: 'try' or 'catch'
                - try_actions_executed: Number of try actions executed
                - catch_actions_executed: Number of catch actions executed
                - finally_actions_executed: Number of finally actions executed
                - error_caught: Error information if an error was caught
                - errors: List of errors encountered (if any)
        """
        config = get_typed_config(action)
        return self._try_catch_executor.execute_try_catch(action, config)  # type: ignore[arg-type]

    # ========================================================================
    # Variable Management (maintain API compatibility)
    # ========================================================================

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the execution context.

        Args:
            name: Variable name
            value: Variable value
        """
        logger.debug("Setting variable: %s = %s", name, value)
        self.variables[name] = value
        self._context.set_variable(name, value)

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
        self._context.clear_variables()

    def get_all_variables(self) -> dict[str, Any]:
        """Get all variables in the execution context.

        Returns:
            Copy of variables dictionary
        """
        return self.variables.copy()

    # ========================================================================
    # Internal Helper Methods
    # ========================================================================

    def _create_execution_context(self):
        """Create a minimal execution context for specialized executors.

        Returns:
            ExecutionContext-like object with necessary interfaces
        """
        # Import here to avoid circular dependencies
        from qontinui.orchestration.execution_context import ExecutionContext

        # Create context with initial variables
        context = ExecutionContext(initial_variables=self.variables)

        # Add execute_action callback
        def execute_action_callback(action_id: str) -> bool:
            """Execute action using the provided action_executor callback."""
            if not self.action_executor:
                logger.warning(
                    f"No action executor configured, skipping action {action_id}"
                )
                return True

            try:
                # Sync variables from context before execution
                self.variables.update(context.variables)

                # Execute via callback
                result = self.action_executor(action_id, self.variables)

                # Sync variables back to context after execution
                for key, value in self.variables.items():
                    context.set_variable(key, value)

                return result.get("success", True)  # type: ignore[no-any-return]

            except Exception as e:
                logger.error(f"Action execution failed: {e}")
                return False

        # Attach execute_action method to context
        context.execute_action = execute_action_callback

        return context
