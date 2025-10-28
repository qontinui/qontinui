"""Action chain executor - ported from Qontinui framework.

Executes a chain of actions according to a specified chaining strategy.
"""

from typing import Any, Optional

from ...action_chain_options import ActionChainOptions, ChainingStrategy
from ...action_config import ActionConfig
from ...action_interface import ActionInterface
from ...action_result import ActionResult
from ...object_collection import ObjectCollection
from ..service.action_service import ActionService


class ActionChainExecutor:
    """Executes a chain of actions according to a specified chaining strategy.

    Port of ActionChainExecutor from Qontinui framework.

    ActionChainExecutor is responsible for managing the execution of multiple actions
    in sequence, where the results of one action influence the execution of the next.
    It supports different chaining strategies (nested and confirm) that determine how
    results flow from one action to the next.

    This component follows the Single Responsibility Principle by focusing solely on
    chain execution logic, while individual action execution is delegated to the
    appropriate action implementations.
    """

    def __init__(
        self,
        action_execution: Optional["ActionExecution"] = None,
        action_service: ActionService | None = None,
    ) -> None:
        """Initialize ActionChainExecutor.

        Args:
            action_execution: Executes individual actions
            action_service: Resolves action implementations
        """
        self.action_execution = action_execution
        self.action_service = action_service

    def execute_chain(
        self,
        chain_options: ActionChainOptions,
        initial_result: ActionResult,
        *object_collections: ObjectCollection,
    ) -> ActionResult:
        """Execute a chain of actions according to the specified chaining strategy.

        The execution flow depends on the strategy:
        - NESTED: Each action searches within the results of the previous action
        - CONFIRM: Each action validates the results of the previous action

        Args:
            chain_options: The configuration for the action chain
            initial_result: The initial action result to start the chain
            object_collections: The object collections to use for the initial action

        Returns:
            The final ActionResult after all actions in the chain have executed

        Raises:
            ActionFailedException: If any action in the chain fails
        """
        # Create a new result that will accumulate all execution history
        final_result = ActionResult()

        # Execute the initial action
        current_result = self._execute_action(
            chain_options.get_initial_action(), initial_result, object_collections
        )

        # Store the initial action's result in history
        final_result.add_execution_record(self._create_action_record(current_result))  # type: ignore[arg-type]

        # If initial action failed, return immediately
        if not current_result.is_success:
            final_result.success = False
            return final_result

        # Execute subsequent actions based on strategy
        for next_action in chain_options.get_chained_actions():
            current_result = self._execute_next_in_chain(
                chain_options.get_strategy(), current_result, next_action, object_collections
            )

            # Store each action's result in history
            final_result.add_execution_record(self._create_action_record(current_result))  # type: ignore[arg-type]

            # If any action fails, stop the chain
            if not current_result.is_success:
                break

        # Copy final state to the result
        final_result.match_list = current_result.match_list
        final_result.success = current_result.is_success
        final_result.duration = current_result.duration
        final_result.text = current_result.text
        final_result.active_states = current_result.active_states

        # Copy movements if any
        for movement in current_result.get_movements():
            final_result.add_movement(movement)

        return final_result

    def _execute_next_in_chain(
        self,
        strategy: ChainingStrategy,
        previous_result: ActionResult,
        next_action: ActionConfig,
        original_collections: tuple[Any, ...],
    ) -> ActionResult:
        """Execute the next action in the chain based on the chaining strategy.

        Args:
            strategy: The chaining strategy
            previous_result: Result from the previous action
            next_action: Configuration for the next action
            original_collections: Original object collections

        Returns:
            Result of executing the next action
        """
        if strategy == ChainingStrategy.NESTED:
            return self._execute_nested_action(previous_result, next_action)
        elif strategy == ChainingStrategy.CONFIRM:
            return self._execute_confirming_action(
                previous_result, next_action, original_collections
            )
        else:
            raise ValueError(f"Unknown chaining strategy: {strategy}")

    def _execute_nested_action(
        self, previous_result: ActionResult, next_action: ActionConfig
    ) -> ActionResult:
        """Execute an action in NESTED mode where it searches within previous results.

        Args:
            previous_result: Result containing regions to search within
            next_action: Configuration for the next action

        Returns:
            Result of the nested action
        """
        # Create search regions from previous matches
        search_regions = []
        for match in previous_result.match_list:
            search_regions.append(match.get_region())

        if not search_regions:
            # No regions to search within
            empty_result = ActionResult()
            empty_result.success = False
            return empty_result

        # Update the action configuration to search within these regions
        # This is a simplified approach - in full implementation would modify search regions
        # For now, execute with the original configuration
        return self._execute_action(next_action, ActionResult(next_action))  # type: ignore[arg-type]

    def _execute_confirming_action(
        self,
        previous_result: ActionResult,
        next_action: ActionConfig,
        original_collections: tuple[Any, ...],
    ) -> ActionResult:
        """Execute an action in CONFIRM mode where it validates previous results.

        Args:
            previous_result: Result to confirm
            next_action: Configuration for the confirming action
            original_collections: Original object collections

        Returns:
            Result of the confirming action
        """
        # Execute the action with original collections
        # The action should confirm the previous results
        return self._execute_action(next_action, ActionResult(next_action), original_collections)  # type: ignore[arg-type]

    def _execute_action(
        self,
        action_config: ActionConfig,
        result: ActionResult,
        object_collections: tuple[Any, ...] = (),
    ) -> ActionResult:
        """Execute a single action.

        Args:
            action_config: Configuration for the action
            result: Result container
            object_collections: Object collections for the action

        Returns:
            Result of the action execution
        """
        if self.action_service:
            action = self.action_service.get_action(action_config)
            if action:
                if self.action_execution:
                    return self.action_execution.perform(
                        action, "", action_config, object_collections
                    )
                else:
                    # Direct execution
                    action.perform(result, *object_collections)
                    return result

        # No action found or service not available
        result.success = False
        return result

    def _create_action_record(self, action_result: ActionResult) -> dict[str, Any]:
        """Create an action record for history tracking.

        Args:
            action_result: The action result to record

        Returns:
            Dictionary containing action execution details
        """
        return {
            "success": action_result.is_success,
            "matches": len(action_result.match_list),
            "duration": action_result.duration,
            "text": action_result.text,
        }


class ActionExecution:
    """Placeholder for ActionExecution class.

    Handles action lifecycle and execution.
    """

    def perform(
        self,
        action: ActionInterface,
        description: str,
        config: ActionConfig,
        collections: tuple[Any, ...],
    ) -> ActionResult:
        """Execute an action with lifecycle management.

        Args:
            action: The action to execute
            description: Description of the action
            config: Action configuration
            collections: Object collections

        Returns:
            Result of the action
        """
        result = ActionResult(config)  # type: ignore[arg-type]
        result.action_description = description
        action.perform(result, *collections)
        return result
