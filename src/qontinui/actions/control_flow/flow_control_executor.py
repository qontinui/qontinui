"""Flow control executor for BREAK and CONTINUE operations.

This module provides the FlowControlExecutor class which handles execution of
BREAK and CONTINUE actions, supporting both conditional and unconditional
flow control operations within loops.
"""

import logging
from typing import Any

from qontinui.config import Action, BreakActionConfig, ContinueActionConfig, get_typed_config
from qontinui.orchestration.execution_context import ExecutionContext

from .condition_evaluator import ConditionEvaluator
from .exceptions import BreakLoop, ContinueLoop

logger = logging.getLogger(__name__)


class FlowControlExecutor:
    """Executor for flow control actions (BREAK and CONTINUE).

    This class manages execution of flow control statements within loops,
    supporting both conditional and unconditional break/continue operations.
    It uses ConditionEvaluator for condition evaluation and raises appropriate
    exceptions (BreakLoop, ContinueLoop) to signal flow control to loop executors.

    The executor is designed as a standalone component that integrates with
    the ExecutionContext for variable access and state management.

    Example:
        >>> context = ExecutionContext({"counter": 5})
        >>> executor = FlowControlExecutor(context)
        >>> break_action = Action(
        ...     id='break-1',
        ...     type='BREAK',
        ...     config={
        ...         'condition': {
        ...             'type': 'variable',
        ...             'variable_name': 'counter',
        ...             'operator': '>',
        ...             'expected_value': 3
        ...         },
        ...         'message': 'Counter exceeded threshold'
        ...     }
        ... )
        >>> try:
        ...     executor.execute_break(break_action)
        ... except BreakLoop as e:
        ...     print(f"Loop broken: {e.message}")
    """

    def __init__(self, context: ExecutionContext) -> None:
        """Initialize the flow control executor.

        Args:
            context: ExecutionContext providing variable access and state management.
                    This context is used by the ConditionEvaluator for condition
                    evaluation and variable substitution.
        """
        self.context = context
        self.condition_evaluator = ConditionEvaluator(context)
        logger.debug("FlowControlExecutor initialized with context")

    def execute_break(self, action: Action) -> None:
        """Execute a BREAK action to exit a loop.

        Evaluates an optional condition and raises BreakLoop exception when the
        break should occur. If no condition is specified, the break is unconditional.

        The raised BreakLoop exception should be caught by the loop executor to
        terminate the loop and return control to the code following the loop.

        Args:
            action: The BREAK action to execute, containing optional condition
                   and message configuration

        Raises:
            BreakLoop: When the break condition is met (or unconditionally if no
                      condition is specified). Contains a message describing the
                      break reason.

        Example:
            >>> # Unconditional break
            >>> action = Action(id='break-1', type='BREAK', config={})
            >>> executor.execute_break(action)  # Always raises BreakLoop

            >>> # Conditional break
            >>> action = Action(
            ...     id='break-2',
            ...     type='BREAK',
            ...     config={
            ...         'condition': {
            ...             'type': 'variable',
            ...             'variable_name': 'error_count',
            ...             'operator': '>=',
            ...             'expected_value': 3
            ...         }
            ...     }
            ... )
            >>> executor.execute_break(action)  # Raises only if condition is true
        """
        config: BreakActionConfig = get_typed_config(action)

        # Check condition if present
        if config.condition:
            logger.debug("Evaluating BREAK condition for action %s", action.id)
            should_break = self.condition_evaluator.evaluate_condition(config.condition)

            if not should_break:
                logger.debug(
                    "BREAK condition not met for action %s, continuing loop",
                    action.id
                )
                return

            logger.info(
                "BREAK condition met for action %s, breaking loop",
                action.id
            )
        else:
            logger.info("Executing unconditional BREAK for action %s", action.id)

        # Prepare break message
        message = config.message or "Break triggered"
        logger.info("Raising BreakLoop: %s", message)

        # Raise exception to signal loop break
        raise BreakLoop(message)

    def execute_continue(self, action: Action) -> None:
        """Execute a CONTINUE action to skip to the next loop iteration.

        Evaluates an optional condition and raises ContinueLoop exception when
        the continue should occur. If no condition is specified, the continue
        is unconditional.

        The raised ContinueLoop exception should be caught by the loop executor
        to skip the remaining actions in the current iteration and proceed to
        the next iteration.

        Args:
            action: The CONTINUE action to execute, containing optional condition
                   and message configuration

        Raises:
            ContinueLoop: When the continue condition is met (or unconditionally
                         if no condition is specified). Contains a message
                         describing the continue reason.

        Example:
            >>> # Unconditional continue
            >>> action = Action(id='continue-1', type='CONTINUE', config={})
            >>> executor.execute_continue(action)  # Always raises ContinueLoop

            >>> # Conditional continue
            >>> action = Action(
            ...     id='continue-2',
            ...     type='CONTINUE',
            ...     config={
            ...         'condition': {
            ...             'type': 'variable',
            ...             'variable_name': 'skip_item',
            ...             'operator': '==',
            ...             'expected_value': True
            ...         },
            ...         'message': 'Skipping invalid item'
            ...     }
            ... )
            >>> executor.execute_continue(action)  # Raises only if condition is true
        """
        config: ContinueActionConfig = get_typed_config(action)

        # Check condition if present
        if config.condition:
            logger.debug("Evaluating CONTINUE condition for action %s", action.id)
            should_continue = self.condition_evaluator.evaluate_condition(config.condition)

            if not should_continue:
                logger.debug(
                    "CONTINUE condition not met for action %s, proceeding normally",
                    action.id
                )
                return

            logger.info(
                "CONTINUE condition met for action %s, skipping to next iteration",
                action.id
            )
        else:
            logger.info("Executing unconditional CONTINUE for action %s", action.id)

        # Prepare continue message
        message = config.message or "Continue triggered"
        logger.info("Raising ContinueLoop: %s", message)

        # Raise exception to signal iteration skip
        raise ContinueLoop(message)
