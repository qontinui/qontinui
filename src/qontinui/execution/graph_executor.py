"""Graph-based workflow execution engine.

This module provides the main GraphExecutor class that orchestrates execution
of graph-format workflows using the GraphTraverser, ConnectionRouter, and
MergeHandler components.
"""

import logging
import time
from collections import deque
from typing import Any

from ..config import Action, Workflow
from .connection_router import ConnectionRouter
from .graph_traverser import GraphTraverser, TraversalState
from .merge_handler import MergeHandler

logger = logging.getLogger(__name__)


class ExecutionState:
    """Tracks state during workflow execution."""

    def __init__(self, workflow: Workflow) -> None:
        """Initialize execution state.

        Args:
            workflow: The workflow being executed
        """
        self.workflow = workflow
        self.context: dict[str, Any] = {}
        self.action_states: dict[str, TraversalState] = {}
        self.action_results: dict[str, dict[str, Any]] = {}
        self.execution_order: list[str] = []
        self.errors: list[dict[str, Any]] = []
        self.start_time: float | None = None
        self.end_time: float | None = None

        # Initialize all actions as pending
        for action in workflow.actions:
            self.action_states[action.id] = TraversalState.PENDING

    def mark_executing(self, action_id: str):
        """Mark action as currently executing."""
        self.action_states[action_id] = TraversalState.EXECUTING
        logger.debug(f"Action '{action_id}' state: EXECUTING")

    def mark_completed(self, action_id: str, result: dict[str, Any]):
        """Mark action as completed."""
        self.action_states[action_id] = TraversalState.COMPLETED
        self.action_results[action_id] = result
        self.execution_order.append(action_id)
        logger.debug(
            f"Action '{action_id}' state: COMPLETED (success={result.get('success', False)})"
        )

    def mark_failed(self, action_id: str, error: Any):
        """Mark action as failed."""
        self.action_states[action_id] = TraversalState.FAILED
        self.errors.append({"action_id": action_id, "error": str(error), "timestamp": time.time()})
        logger.error(f"Action '{action_id}' state: FAILED - {error}")

    def mark_skipped(self, action_id: str, reason: str):
        """Mark action as skipped."""
        self.action_states[action_id] = TraversalState.SKIPPED
        logger.debug(f"Action '{action_id}' state: SKIPPED - {reason}")

    def is_pending(self, action_id: str) -> bool:
        """Check if action is pending."""
        return self.action_states.get(action_id) == TraversalState.PENDING

    def is_completed(self, action_id: str) -> bool:
        """Check if action is completed."""
        return self.action_states.get(action_id) == TraversalState.COMPLETED

    def get_elapsed_time(self) -> float:
        """Get elapsed execution time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def get_summary(self) -> dict[str, Any]:
        """Get execution summary."""
        completed = sum(1 for s in self.action_states.values() if s == TraversalState.COMPLETED)
        failed = sum(1 for s in self.action_states.values() if s == TraversalState.FAILED)
        skipped = sum(1 for s in self.action_states.values() if s == TraversalState.SKIPPED)
        pending = sum(1 for s in self.action_states.values() if s == TraversalState.PENDING)

        return {
            "workflow_id": self.workflow.id,
            "workflow_name": self.workflow.name,
            "total_actions": len(self.workflow.actions),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "pending": pending,
            "execution_order": self.execution_order,
            "elapsed_time": self.get_elapsed_time(),
            "errors": self.errors,
        }


class GraphExecutor:
    """Main executor for graph-based workflows.

    The GraphExecutor orchestrates the execution of workflows by:
    1. Validating workflow structure using GraphTraverser
    2. Finding entry points and initializing execution state
    3. Traversing the graph and executing actions
    4. Routing between actions using ConnectionRouter
    5. Handling merge points using MergeHandler
    6. Delegating individual action execution to ActionExecutor
    7. Managing execution hooks for monitoring and debugging

    Attributes:
        workflow: The workflow to execute
        action_executor: ActionExecutor instance for running individual actions
        traverser: GraphTraverser for graph navigation
        router: ConnectionRouter for determining next actions
        merger: MergeHandler for handling merge points
        hooks: List of execution hooks
        execution_state: Current execution state
    """

    def __init__(self, workflow: Workflow, action_executor: Any) -> None:
        """Initialize graph executor.

        Args:
            workflow: The workflow to execute
            action_executor: ActionExecutor instance for running actions
        """
        self.workflow = workflow
        self.action_executor = action_executor

        # Initialize graph components
        self.traverser = GraphTraverser(workflow)
        self.router = ConnectionRouter(workflow.connections, self.traverser.action_map)
        self.merger = MergeHandler(workflow.connections, self.traverser.action_map)

        # Execution hooks
        self.hooks: list[Any] = []

        # Execution state
        self.execution_state: ExecutionState | None = None

        logger.info(f"Initialized GraphExecutor for workflow '{workflow.name}'")

    def add_hook(self, hook: Any):
        """Add an execution hook.

        Args:
            hook: ExecutionHook instance
        """
        self.hooks.append(hook)
        logger.debug(f"Added execution hook: {type(hook).__name__}")

    def remove_hook(self, hook: Any):
        """Remove an execution hook.

        Args:
            hook: ExecutionHook instance to remove
        """
        if hook in self.hooks:
            self.hooks.remove(hook)
            logger.debug(f"Removed execution hook: {type(hook).__name__}")

    def execute(self, initial_context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute the workflow.

        This is the main entry point for workflow execution. It:
        1. Validates workflow structure
        2. Initializes execution state
        3. Finds entry points
        4. Executes actions in graph order
        5. Returns final execution context

        Args:
            initial_context: Initial context variables (optional)

        Returns:
            Dictionary with execution results and final context

        Raises:
            ValueError: If workflow validation fails
            RuntimeError: If execution encounters critical errors
        """
        logger.info(f"Starting execution of workflow '{self.workflow.name}'")

        # Step 1: Validate workflow structure
        is_valid, errors = self.traverser.validate_workflow()
        if not is_valid:
            error_msg = f"Workflow validation failed: {'; '.join(errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Step 2: Initialize execution state
        self.execution_state = ExecutionState(self.workflow)
        self.execution_state.start_time = time.time()

        # Initialize context from workflow variables and initial context
        self.execution_state.context = self._initialize_context(initial_context)

        # Step 3: Find entry points
        entry_points = self.traverser.find_entry_points()
        if not entry_points:
            raise ValueError("No entry points found in workflow")

        logger.info(f"Executing from {len(entry_points)} entry points: {entry_points}")

        # Step 4: Execute workflow
        try:
            self._execute_from_entry_points(entry_points)
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            self.execution_state.end_time = time.time()
            raise

        # Step 5: Finalize execution
        self.execution_state.end_time = time.time()

        logger.info(
            f"Workflow execution completed in {self.execution_state.get_elapsed_time():.2f}s - "
            f"{len(self.execution_state.execution_order)} actions executed"
        )

        # Return results
        return {
            "success": len(self.execution_state.errors) == 0,
            "context": self.execution_state.context,
            "summary": self.execution_state.get_summary(),
            "results": self.execution_state.action_results,
        }

    def _initialize_context(self, initial_context: dict[str, Any] | None) -> dict[str, Any]:
        """Initialize execution context from workflow variables and initial context.

        Args:
            initial_context: User-provided initial context

        Returns:
            Initialized context dictionary
        """
        context = {}

        # Add workflow variables
        if self.workflow.variables:
            if self.workflow.variables.local:
                context.update(self.workflow.variables.local)
            if self.workflow.variables.process:
                context.update(self.workflow.variables.process)
            if self.workflow.variables.global_vars:
                context.update(self.workflow.variables.global_vars)

        # Override with initial context
        if initial_context:
            context.update(initial_context)

        logger.debug(f"Initialized context with {len(context)} variables")
        return context

    def _execute_from_entry_points(self, entry_points: list[str]):
        """Execute workflow starting from entry points.

        Uses breadth-first traversal to execute actions. Actions are added
        to a queue and executed in order, respecting dependencies and merge points.

        Args:
            entry_points: List of entry point action IDs
        """
        # Queue of actions to execute: (action_id, from_action_id)
        # from_action_id is None for entry points
        execution_queue = deque([(entry_id, None) for entry_id in entry_points])

        # Track which actions are queued to avoid duplicates
        queued_actions: set[str] = set(entry_points)

        while execution_queue:
            action_id, from_action_id = execution_queue.popleft()

            # Skip if already executed
            if self.execution_state.is_completed(action_id):
                logger.debug(f"Skipping '{action_id}' - already executed")
                continue

            # Check if this is a merge point
            if self.merger.is_merge_point(action_id):
                # Register arrival at merge point
                if from_action_id:
                    from_result = self.execution_state.action_results.get(from_action_id, {})
                    is_ready = self.merger.register_arrival(action_id, from_action_id, from_result)

                    if not is_ready:
                        logger.debug(
                            f"Merge point '{action_id}' not ready - "
                            f"waiting for {len(self.merger.get_blocking_paths(action_id))} more paths"
                        )
                        continue

                    # Merge point is ready - merge contexts
                    merged_context = self.merger.get_merged_context(action_id)
                    self.execution_state.context.update(merged_context)

            # Get action
            action = self.traverser.action_map.get(action_id)
            if not action:
                logger.error(f"Action '{action_id}' not found in workflow")
                continue

            # Execute action
            try:
                result = self._execute_action(action)

                # Mark as completed
                self.execution_state.mark_completed(action_id, result)

                # Route to next actions
                next_actions = self._route_to_next_actions(action, result)

                # Add next actions to queue
                for next_action_id, _ in next_actions:
                    if next_action_id not in queued_actions:
                        execution_queue.append((next_action_id, action_id))
                        queued_actions.add(next_action_id)

            except Exception as e:
                # Mark as failed
                self.execution_state.mark_failed(action_id, e)

                # Check if we should continue or stop
                continue_on_error = (
                    action.execution.continue_on_error if action.execution else False
                )

                if not continue_on_error:
                    # Check for error path
                    error_next = self.router._get_connections(action_id, "error")
                    if error_next:
                        # Follow error path
                        for next_action_id, _ in error_next:
                            if next_action_id not in queued_actions:
                                execution_queue.append((next_action_id, action_id))
                                queued_actions.add(next_action_id)
                    else:
                        # No error path and continue_on_error is False - stop execution
                        logger.error(f"Stopping execution due to error in action '{action_id}'")
                        raise

    def _execute_action(self, action: Action) -> dict[str, Any]:
        """Execute a single action.

        This method:
        1. Calls before_action hooks
        2. Executes the action using ActionExecutor
        3. Calls after_action hooks
        4. Returns the result

        Args:
            action: The action to execute

        Returns:
            Execution result dictionary

        Raises:
            Exception: If action execution fails
        """
        logger.info(f"Executing action '{action.id}' (type={action.type})")

        # Mark as executing
        self.execution_state.mark_executing(action.id)

        # Call before_action hooks
        for hook in self.hooks:
            try:
                hook.before_action(action, self.execution_state.context)
            except Exception as e:
                logger.warning(f"Hook {type(hook).__name__}.before_action failed: {e}")

        # Execute action
        try:
            success = self.action_executor.execute_action(action)

            # Build result
            result = {
                "success": success,
                "action_id": action.id,
                "action_type": action.type,
                "context": self.execution_state.context.copy(),
            }

            # Call after_action hooks
            for hook in self.hooks:
                try:
                    hook.after_action(action, self.execution_state.context, result)
                except Exception as e:
                    logger.warning(f"Hook {type(hook).__name__}.after_action failed: {e}")

            return result

        except Exception as e:
            # Call error hooks
            for hook in self.hooks:
                try:
                    hook.on_error(action, self.execution_state.context, e)
                except Exception as hook_error:
                    logger.warning(f"Hook {type(hook).__name__}.on_error failed: {hook_error}")

            raise

    def _route_to_next_actions(
        self, action: Action, result: dict[str, Any]
    ) -> list[tuple[str, int]]:
        """Determine next actions to execute based on result.

        Args:
            action: The action that just executed
            result: Execution result

        Returns:
            List of (action_id, input_index) tuples for next actions
        """
        # Use router to determine next actions
        routing_decision = self.router.route(action, result, self.execution_state.context)

        logger.debug(
            f"Routing decision for '{action.id}': {routing_decision.connection_type} -> "
            f"{len(routing_decision.next_actions)} actions ({routing_decision.reason})"
        )

        return routing_decision.next_actions

    def get_execution_progress(self) -> dict[str, Any]:
        """Get current execution progress.

        Returns:
            Dictionary with progress information
        """
        if not self.execution_state:
            return {"status": "not_started"}

        total = len(self.workflow.actions)
        completed = sum(
            1 for s in self.execution_state.action_states.values() if s == TraversalState.COMPLETED
        )
        failed = sum(
            1 for s in self.execution_state.action_states.values() if s == TraversalState.FAILED
        )

        progress_percent = (completed / total * 100) if total > 0 else 0

        return {
            "status": "executing" if self.execution_state.end_time is None else "completed",
            "total_actions": total,
            "completed": completed,
            "failed": failed,
            "progress_percent": progress_percent,
            "elapsed_time": self.execution_state.get_elapsed_time(),
            "current_action": (
                self.execution_state.execution_order[-1]
                if self.execution_state.execution_order
                else None
            ),
        }

    def validate_before_execution(self) -> tuple[bool, list[str]]:
        """Validate workflow before execution.

        Performs comprehensive validation including:
        - Workflow structure
        - Merge points
        - Routing configuration

        Returns:
            Tuple of (is_valid, list of errors/warnings)
        """
        all_errors = []

        # Validate workflow structure
        is_valid, errors = self.traverser.validate_workflow()
        all_errors.extend(errors)

        # Validate merge points
        is_valid_merge, merge_errors = self.merger.validate_merge_points()
        if not is_valid_merge:
            all_errors.extend(merge_errors)

        # Validate routing for critical actions
        for action in self.workflow.actions:
            is_valid_routing, routing_warnings = self.router.validate_routing(action.id)
            if not is_valid_routing:
                all_errors.extend(routing_warnings)

        return len(all_errors) == 0, all_errors

    def get_execution_statistics(self) -> dict[str, Any]:
        """Get detailed execution statistics.

        Returns:
            Dictionary with execution statistics
        """
        if not self.execution_state:
            return {"error": "No execution state available"}

        # Basic statistics
        stats = self.execution_state.get_summary()

        # Add graph statistics
        stats["graph"] = {
            "total_actions": len(self.workflow.actions),
            "entry_points": len(self.traverser.find_entry_points()),
            "exit_points": len(self.traverser.find_exit_points()),
            "merge_points": len(self.merger.get_all_merge_points()),
            "orphaned_actions": len(self.traverser.find_orphaned_actions()),
        }

        # Add merge statistics
        stats["merges"] = self.merger.get_merge_statistics()

        # Add action type distribution
        action_types = {}
        for action in self.workflow.actions:
            action_types[action.type] = action_types.get(action.type, 0) + 1
        stats["action_types"] = action_types

        # Add performance metrics
        if self.execution_state.execution_order:
            avg_time_per_action = self.execution_state.get_elapsed_time() / len(
                self.execution_state.execution_order
            )
            stats["performance"] = {
                "total_time": self.execution_state.get_elapsed_time(),
                "actions_executed": len(self.execution_state.execution_order),
                "avg_time_per_action": avg_time_per_action,
            }

        return stats
