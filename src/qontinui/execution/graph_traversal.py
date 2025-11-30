"""
Graph traversal engine for executing graph-based workflows.

This module provides the core execution engine for graph format workflows.
It handles:
- Finding entry points (actions with no incoming connections)
- Traversing the graph following connections
- Handling multi-output actions (IF, SWITCH, LOOP, TRY_CATCH)
- Detecting and preventing infinite loops
- Managing execution state and context
"""

import logging
from collections.abc import Callable
from typing import Any

from ..config.schema import Action, Workflow
from .connection_resolver import ConnectionResolver
from .execution_state import ExecutionState

logger = logging.getLogger(__name__)


class CycleDetectedError(Exception):
    """Raised when a cycle is detected in the workflow graph."""

    pass


class InfiniteLoopError(Exception):
    """Raised when iteration limit is exceeded."""

    pass


class OrphanedActionsError(Exception):
    """Raised when actions have no path from entry points."""

    pass


class GraphTraverser:
    """
    Executes graph-based workflows by traversing action connections.

    This is the main execution engine for graph format workflows. It handles
    complex control flow including branching and loops. All actions are executed
    sequentially, which is appropriate for GUI automation (single mouse/keyboard).
    """

    def __init__(
        self,
        workflow: Workflow,
        action_executor: Callable[[Action, dict[str, Any]], dict[str, Any]] | None = None,
        max_iterations: int = 10000,
    ) -> None:
        """
        Initialize the graph traverser.

        Args:
            workflow: The workflow to execute
            action_executor: Function to execute individual actions
            max_iterations: Maximum iterations before stopping
        """
        self.workflow = workflow
        self.resolver = ConnectionResolver(workflow)
        self.action_executor = action_executor or self._default_executor
        self.max_iterations = max_iterations

        # Execution state
        self.state: ExecutionState | None = None

        # Validation
        self._validate_workflow()

    def _validate_workflow(self) -> None:
        """
        Validate the workflow structure.

        Raises:
            ValueError: If workflow is invalid
            OrphanedActionsError: If orphaned actions exist
        """
        # Check for entry points
        entry_points = self.get_entry_actions()
        if not entry_points:
            raise ValueError("Workflow has no entry points (actions with no incoming connections)")

        # Check for orphaned actions
        orphaned = self._find_orphaned_actions()
        if orphaned:
            raise OrphanedActionsError(
                f"Workflow has orphaned actions (no path from entry points): {orphaned}"
            )

        # Check for simple cycles (action connecting to itself)
        self._check_simple_cycles()

    def _check_simple_cycles(self) -> None:
        """
        Check for simple cycles (action connecting to itself).

        Raises:
            CycleDetectedError: If a simple cycle is detected
        """
        for action in self.workflow.actions:
            for output_type, _ in self.resolver.get_all_outputs(action.id):
                connections = self.resolver.resolve_output_connection(action.id, output_type)
                for conn in connections:
                    if conn.action == action.id:
                        raise CycleDetectedError(
                            f"Simple cycle detected: action '{action.id}' connects to itself"
                        )

    def _find_orphaned_actions(self) -> list[str]:
        """
        Find actions that cannot be reached from any entry point.

        Returns:
            List of orphaned action IDs
        """
        reachable = set()
        entry_points = self.get_entry_actions()

        # BFS to find all reachable actions
        queue = [action.id for action in entry_points]
        while queue:
            action_id = queue.pop(0)
            if action_id in reachable:
                continue

            reachable.add(action_id)

            # Add connected actions
            for output_type, _ in self.resolver.get_all_outputs(action_id):
                connections = self.resolver.resolve_output_connection(action_id, output_type)
                for conn in connections:
                    if conn.action not in reachable:
                        queue.append(conn.action)

        # Find actions not reachable
        all_actions = {action.id for action in self.workflow.actions}
        orphaned = all_actions - reachable

        return list(orphaned)

    def get_entry_actions(self) -> list[Action]:
        """
        Get all entry point actions (actions with no incoming connections).

        Returns:
            List of entry point actions
        """
        entry_actions = []
        for action in self.workflow.actions:
            if not self.resolver.has_incoming_connections(action.id):
                entry_actions.append(action)
        return entry_actions

    def get_next_actions(
        self, action_id: str, output_type: str = "main", output_index: int = 0
    ) -> list[Action]:
        """
        Get the next actions to execute based on output type.

        Args:
            action_id: The current action ID
            output_type: The output type taken (main, true, false, etc.)
            output_index: The output index

        Returns:
            List of next actions to execute
        """
        return self.resolver.get_connected_actions(action_id, output_type, output_index)

    def traverse(
        self, context: dict[str, Any] | None = None, start_action_id: str | None = None
    ) -> dict[str, Any]:
        """
        Main entry point for workflow execution.

        Traverses the workflow graph starting from entry points (or specified action)
        and executes each action in order, following connections.

        Args:
            context: Initial execution context
            start_action_id: Optional action to start from (default: entry points)

        Returns:
            Execution result including status, context, and statistics

        Raises:
            InfiniteLoopError: If iteration limit exceeded
            CycleDetectedError: If a cycle is detected during execution
        """
        # Initialize state
        self.state = ExecutionState(
            workflow_id=self.workflow.id, max_iterations=self.max_iterations, enable_history=True
        )
        self.state.start()

        # Initialize context
        if context is None:
            context = {}

        # Add workflow variables to context
        if self.workflow.variables:
            if self.workflow.variables.local:
                context.update(self.workflow.variables.local)
            if self.workflow.variables.process:
                context.update(self.workflow.variables.process)
            if self.workflow.variables.global_vars:
                context.update(self.workflow.variables.global_vars)

        self.state.update_context(context)

        try:
            # Get starting actions
            if start_action_id:
                start_action = self.resolver.get_action_by_id(start_action_id)
                if not start_action:
                    raise ValueError(f"Start action '{start_action_id}' not found")
                starting_actions = [start_action]
            else:
                starting_actions = self.get_entry_actions()

            # Add starting actions to pending queue
            for action in starting_actions:
                self.state.add_pending(action.id, depth=0)

            # Execute actions
            while self.state.has_pending():
                # Check iteration limit
                if not self.state.increment_iteration():
                    raise InfiniteLoopError(
                        f"Iteration limit ({self.max_iterations}) exceeded. "
                        "Possible infinite loop in workflow."
                    )

                # Get next action
                pending = self.state.get_next_pending()
                if not pending:
                    break

                # Check for pause
                if self.state.should_pause_at(pending.action_id):
                    self.state.pause()
                    logger.info(f"Execution paused at action: {pending.action_id}")
                    break

                # Execute action
                self._execute_action(pending.action_id, pending.depth)

            # Mark as complete if not paused
            if not self.state.is_paused():
                self.state.complete()

            # Return result
            return {
                "status": self.state.status.value,
                "context": self.state.get_all_context(),
                "statistics": self.state.get_statistics(),
                "history": [self._record_to_dict(r) for r in self.state.get_history()],
                "errors": self.state.get_errors(),
            }

        except Exception as e:
            self.state.fail(str(e))
            logger.error(f"Workflow execution failed: {e}")
            raise

    def _execute_action(self, action_id: str, depth: int) -> None:
        """
        Execute a single action and queue next actions.

        Args:
            action_id: The action ID to execute
            depth: Current execution depth
        """
        # Get action
        action = self.resolver.get_action_by_id(action_id)
        if not action:
            logger.error(f"Action not found: {action_id}")
            return

        # Mark as current
        if self.state is not None:
            self.state.set_current_action(action_id)

        # Start recording
        record = self.state.start_action(action_id, action.type) if self.state is not None else None

        logger.info(f"Executing action: {action_id} (type: {action.type}, depth: {depth})")

        try:
            # Execute the action
            context = self.state.get_all_context() if self.state is not None else {}
            result = self.action_executor(action, context)

            # Update context with result
            if result and self.state is not None:
                self.state.update_context(result)

            # Determine output type based on action type and result
            output_type = self._determine_output_type(action, result)
            output_index = 0

            # Mark as completed
            if record is not None:
                record.complete(result, output_type, output_index)

            # Queue next actions
            next_actions = self.get_next_actions(action_id, output_type, output_index)

            for next_action in next_actions:
                # Check if already visited (cycle detection)
                if self.state is not None and self.state.is_visited(next_action.id):
                    # Allow revisiting for LOOP actions
                    if action.type != "LOOP":
                        logger.warning(f"Cycle detected: action '{next_action.id}' already visited")
                        continue

                if self.state is not None:
                    self.state.add_pending(next_action.id, depth=depth + 1)

            # Mark as visited (after queuing to allow loops)
            if self.state is not None:
                self.state.mark_visited(action_id)

        except Exception as e:
            logger.error(f"Action execution failed: {action_id} - {e}")
            if record is not None:
                record.fail(str(e))

            # Try error path if available
            if self.resolver.validate_output_exists(action_id, "error"):
                error_actions = self.get_next_actions(action_id, "error", 0)
                for error_action in error_actions:
                    if self.state is not None:
                        self.state.add_pending(error_action.id, depth=depth + 1)
            else:
                # Re-raise if no error handler
                raise

        finally:
            if self.state is not None:
                self.state.set_current_action(None)

    def _determine_output_type(self, action: Action, result: dict[str, Any]) -> str:
        """
        Determine which output type to follow based on action type and result.

        Args:
            action: The action that was executed
            result: The execution result

        Returns:
            The output type to follow (main, true, false, loop, error, case_N)
        """
        action_type = action.type

        # IF action
        if action_type == "IF":
            condition_met = result.get("condition_met", False)
            return "true" if condition_met else "false"

        # LOOP action
        if action_type == "LOOP":
            should_continue = result.get("should_continue", False)
            return "loop" if should_continue else "main"

        # SWITCH action
        if action_type == "SWITCH":
            case_index = result.get("case_index", -1)
            if case_index >= 0:
                return f"case_{case_index}"
            return "main"  # default case

        # TRY_CATCH action
        if action_type == "TRY_CATCH":
            had_error = result.get("had_error", False)
            return "error" if had_error else "main"

        # Default: main output
        return "main"

    def _default_executor(self, action: Action, context: dict[str, Any]) -> dict[str, Any]:
        """
        Default action executor (for testing).

        Args:
            action: The action to execute
            context: Execution context

        Returns:
            Execution result
        """
        logger.info(f"Default executor: {action.type} - {action.id}")
        return {"success": True, "action_id": action.id, "action_type": action.type}

    def _record_to_dict(self, record) -> dict[str, Any]:
        """
        Convert action execution record to dictionary.

        Args:
            record: The execution record

        Returns:
            Dictionary representation
        """
        return {
            "action_id": record.action_id,
            "action_type": record.action_type,
            "status": record.status.value,
            "start_time": record.start_time.isoformat() if record.start_time else None,
            "end_time": record.end_time.isoformat() if record.end_time else None,
            "duration_ms": record.duration_ms,
            "result": record.result,
            "error": record.error,
            "output_type": record.output_type,
            "output_index": record.output_index,
        }

    # ============================================================================
    # Pause/Resume Support
    # ============================================================================

    def pause(self) -> None:
        """Pause execution at the next action."""
        if self.state:
            self.state.pause()

    def resume(self) -> dict[str, Any]:
        """
        Resume paused execution.

        Returns:
            Execution result
        """
        if not self.state or not self.state.is_paused():
            raise ValueError("Cannot resume: execution not paused")

        self.state.resume()

        # Continue execution from where we left off
        while self.state.has_pending():
            if not self.state.increment_iteration():
                raise InfiniteLoopError(f"Iteration limit ({self.max_iterations}) exceeded")

            pending = self.state.get_next_pending()
            if not pending:
                break

            if self.state.should_pause_at(pending.action_id):
                self.state.pause()
                break

            self._execute_action(pending.action_id, pending.depth)

        if not self.state.is_paused():
            self.state.complete()

        return {
            "status": self.state.status.value,
            "context": self.state.get_all_context(),
            "statistics": self.state.get_statistics(),
            "history": [self._record_to_dict(r) for r in self.state.get_history()],
        }

    def set_breakpoint(self, action_id: str) -> None:
        """
        Set a breakpoint at an action.

        Args:
            action_id: The action ID to pause at
        """
        if self.state:
            self.state.set_pause_at_action(action_id)

    def clear_breakpoint(self) -> None:
        """Clear the current breakpoint."""
        if self.state:
            self.state.set_pause_at_action(None)

    # ============================================================================
    # Utilities
    # ============================================================================

    def get_execution_path(self) -> list[str]:
        """
        Get the execution path (list of action IDs in order).

        Returns:
            List of action IDs
        """
        if not self.state:
            return []

        history = self.state.get_history()
        return [record.action_id for record in history]

    def get_statistics(self) -> dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Dictionary of statistics
        """
        if not self.state:
            return {}

        return self.state.get_statistics()
