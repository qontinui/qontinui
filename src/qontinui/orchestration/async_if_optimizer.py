"""Async optimizer for batching consecutive IF-image_exists actions.

This module provides optimization for workflows that contain multiple consecutive
IF actions checking for image existence. Instead of searching for images sequentially,
it batches them and searches in parallel for significant performance improvements.

Performance:
- Sequential: N images × 200ms = N/5 seconds
- Parallel (with this optimizer): ~200-400ms regardless of N

Usage:
    optimizer = AsyncIfOptimizer(context)
    result = await optimizer.execute_workflow_with_batching(actions)
"""

import logging
from typing import Any

from ..actions.control_flow.condition_evaluator import ConditionEvaluator
from ..config import Action
from .execution_context import ExecutionContext

logger = logging.getLogger(__name__)


class AsyncIfOptimizer:
    """Optimizes workflows by batching consecutive IF-image_exists actions.

    This class analyzes workflows to detect patterns of consecutive IF actions
    that check for image existence, then executes them in parallel rather than
    sequentially for improved performance.
    """

    def __init__(self, context: ExecutionContext) -> None:
        """Initialize optimizer with execution context.

        Args:
            context: Execution context containing configuration and state
        """
        self.context = context
        self.condition_evaluator = ConditionEvaluator(context)

    async def execute_workflow_with_batching(
        self,
        actions: list[Action],
    ) -> dict[str, Any]:
        """Execute workflow with IF-image_exists batching optimization.

        Analyzes the workflow and batches consecutive IF-image_exists actions
        for parallel execution.

        Args:
            actions: List of actions to execute

        Returns:
            Execution result dictionary with success status and details
        """
        logger.info(f"Optimizing workflow with {len(actions)} actions")

        # Detect batchable IF actions
        batches = self._detect_if_image_exists_batches(actions)

        if not batches:
            logger.debug("No batchable IF-image_exists actions found")
            return {"success": True, "batches_optimized": 0}

        logger.info(f"Found {len(batches)} batches to optimize")

        # Execute batches
        results = {}
        for batch_info in batches:
            batch_results = await self._execute_if_batch(batch_info)
            results.update(batch_results)

        return {
            "success": True,
            "batches_optimized": len(batches),
            "batch_results": results,
        }

    def _detect_if_image_exists_batches(
        self,
        actions: list[Action],
    ) -> list[dict[str, Any]]:
        """Detect consecutive IF-image_exists actions that can be batched.

        Args:
            actions: List of actions to analyze

        Returns:
            List of batch information dictionaries
        """
        batches = []
        current_batch = []

        for i, action in enumerate(actions):
            # Check if this is an IF action with image_exists condition
            if self._is_if_image_exists_action(action):
                current_batch.append(
                    {
                        "index": i,
                        "action": action,
                    }
                )
            else:
                # Non-IF or non-image_exists action breaks the batch
                if len(current_batch) >= 2:  # Only batch if 2+ actions
                    batches.append(
                        {
                            "actions": current_batch,
                            "start_index": current_batch[0]["index"],
                            "end_index": current_batch[-1]["index"],
                        }
                    )
                current_batch = []

        # Handle final batch
        if len(current_batch) >= 2:
            batches.append(
                {
                    "actions": current_batch,
                    "start_index": current_batch[0]["index"],
                    "end_index": current_batch[-1]["index"],
                }
            )

        return batches

    def _is_if_image_exists_action(self, action: Action) -> bool:
        """Check if action is an IF with image_exists condition.

        Args:
            action: Action to check

        Returns:
            True if this is an IF-image_exists action with empty branches
        """
        if action.type != "IF":
            return False

        # Check if it has image_exists condition
        if not hasattr(action.config, "condition"):
            return False

        condition = action.config.condition
        if not hasattr(condition, "type") or condition.type != "image_exists":
            return False

        # Only batch IFs with empty then/else branches (like state verification)
        if hasattr(action.config, "then_actions") and action.config.then_actions:
            return False
        if hasattr(action.config, "else_actions") and action.config.else_actions:
            return False

        return True

    async def _execute_if_batch(
        self,
        batch_info: dict[str, Any],
    ) -> dict[str, bool]:
        """Execute a batch of IF-image_exists actions in parallel.

        Args:
            batch_info: Batch information with actions to execute

        Returns:
            Dictionary mapping action IDs to condition results
        """
        actions_data = batch_info["actions"]
        logger.info(f"Executing batch of {len(actions_data)} IF-image_exists actions")

        # Build list of conditions to evaluate
        conditions = []
        for action_data in actions_data:
            action = action_data["action"]
            conditions.append((action.id, action.config.condition))

        # Execute all conditions in parallel
        results = await self.condition_evaluator.evaluate_multiple_image_exists_async(
            conditions
        )

        logger.info(
            f"Batch execution complete: {sum(results.values())} conditions true"
        )
        return results


async def execute_if_image_exists_batch(
    context: ExecutionContext,
    actions: list[Action],
) -> dict[str, bool]:
    """Convenience function to batch-execute IF-image_exists actions.

    This is a simplified API for the common use case of executing a list
    of IF-image_exists actions (like in state verification workflows).

    Args:
        context: Execution context
        actions: List of IF actions to execute

    Returns:
        Dictionary mapping action IDs to condition results (True/False)

    Example:
        actions = [if_action1, if_action2, if_action3]
        results = await execute_if_image_exists_batch(context, actions)
        # results = {"if-1": True, "if-2": False, "if-3": False}
        # Image for if-1 was found, others were not
    """
    optimizer = AsyncIfOptimizer(context)

    # Build conditions list
    conditions = []
    for action in actions:
        if hasattr(action.config, "condition"):
            conditions.append((action.id, action.config.condition))

    # Execute in parallel
    results = await optimizer.condition_evaluator.evaluate_multiple_image_exists_async(
        conditions
    )

    return results
