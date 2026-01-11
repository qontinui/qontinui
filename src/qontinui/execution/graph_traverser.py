"""Graph traversal for workflow execution.

This module provides depth-first traversal of workflow graphs with support for
branching, looping, and parallel execution paths.
"""

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Union

from ..config import Workflow

if TYPE_CHECKING:
    from qontinui_schemas.config.models.workflow import Workflow as SchemaWorkflow

# Accept both local Workflow and qontinui_schemas Workflow
WorkflowType = Union[Workflow, "SchemaWorkflow"]

logger = logging.getLogger(__name__)


class TraversalState(str, Enum):
    """State of action during traversal."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class ExecutionPath:
    """Represents a single execution path through the graph."""

    def __init__(self, workflow: WorkflowType) -> None:
        """Initialize execution path.

        Args:
            workflow: The workflow to traverse
        """
        self.workflow = workflow
        self.visited: set[str] = set()
        self.execution_order: list[str] = []
        self.action_states: dict[str, TraversalState] = {}
        self.action_results: dict[str, dict[str, Any]] = {}

        # Initialize all actions as pending
        for action in workflow.actions:
            self.action_states[action.id] = TraversalState.PENDING


class GraphTraverser:
    """Traverses workflow graph and manages execution order.

    The GraphTraverser is responsible for:
    - Finding entry points (actions with no incoming connections)
    - Determining execution order based on dependencies
    - Managing execution state (visited nodes, pending actions)
    - Handling cycles and infinite loops
    - Supporting both depth-first and breadth-first traversal

    Attributes:
        workflow: The workflow to traverse
        connections: Connection graph from workflow
        action_map: Quick lookup for actions by ID
        execution_paths: List of execution paths through graph
    """

    def __init__(self, workflow: WorkflowType) -> None:
        """Initialize graph traverser.

        Args:
            workflow: The workflow to traverse
        """
        self.workflow = workflow
        self.connections = workflow.connections
        self.action_map = {action.id: action for action in workflow.actions}
        self.execution_paths: list[ExecutionPath] = []

        logger.info(
            f"Initialized GraphTraverser for workflow '{workflow.name}' "
            f"with {len(workflow.actions)} actions"
        )

    def find_entry_points(self) -> list[str]:
        """Find all entry point actions (no incoming connections).

        Entry points are actions that have no incoming connections from other
        actions. These are the starting points for graph execution.

        Returns:
            List of action IDs that are entry points
        """
        # Get all action IDs that have incoming connections
        target_actions: set[str] = set()

        for _source_id, conn_types in self.connections.root.items():
            for _conn_type, conn_lists in conn_types.items():
                for conn_list in conn_lists:
                    for connection in conn_list:
                        target_actions.add(connection.action)

        # Entry points are actions NOT in the target set
        entry_points = [
            action.id for action in self.workflow.actions if action.id not in target_actions
        ]

        logger.info(f"Found {len(entry_points)} entry points: {entry_points}")
        return entry_points

    def find_exit_points(self) -> list[str]:
        """Find all exit point actions (no outgoing connections).

        Exit points are actions that have no outgoing connections to other
        actions. These indicate the end of execution paths.

        Returns:
            List of action IDs that are exit points
        """
        # Get all action IDs that have outgoing connections
        source_actions = set(self.connections.root.keys())

        # Exit points are actions NOT in the source set
        exit_points = [
            action.id for action in self.workflow.actions if action.id not in source_actions
        ]

        logger.info(f"Found {len(exit_points)} exit points: {exit_points}")
        return exit_points

    def get_next_actions(
        self,
        action_id: str,
        connection_type: str = "main",
        execution_path: ExecutionPath | None = None,
    ) -> list[tuple[str, int]]:
        """Get next actions to execute from current action.

        Args:
            action_id: Current action ID
            connection_type: Type of connection to follow (main, error, true, false, etc.)
            execution_path: Current execution path for state tracking

        Returns:
            List of tuples (action_id, input_index) for next actions
        """
        conn_lists = self.connections.get_connections(action_id, connection_type)

        next_actions = []
        for conn_list in conn_lists:
            for connection in conn_list:
                # Check if target action exists
                if connection.action not in self.action_map:
                    logger.warning(f"Target action '{connection.action}' not found in workflow")
                    continue

                # Skip if already executed in this path (cycle detection)
                if execution_path and connection.action in execution_path.visited:
                    logger.debug(f"Skipping action '{connection.action}' - already visited in path")
                    continue

                next_actions.append((connection.action, connection.index))

        logger.debug(
            f"Action '{action_id}' ({connection_type}) -> {len(next_actions)} next actions"
        )
        return next_actions

    def get_all_next_actions(
        self, action_id: str, execution_path: ExecutionPath | None = None
    ) -> dict[str, list[tuple[str, int]]]:
        """Get all next actions grouped by connection type.

        Args:
            action_id: Current action ID
            execution_path: Current execution path for state tracking

        Returns:
            Dictionary mapping connection type to list of (action_id, index) tuples
        """
        all_connections = self.connections.get_all_connections(action_id)

        result = {}
        for conn_type, _conn_lists in all_connections.items():
            next_actions = self.get_next_actions(action_id, conn_type, execution_path)
            if next_actions:
                result[conn_type] = next_actions

        return result

    def detect_cycles(self) -> list[list[str]]:
        """Detect cycles in the workflow graph.

        Uses depth-first search to detect cycles. A cycle exists if we
        encounter a node that's currently in our DFS path.

        Returns:
            List of cycles, where each cycle is a list of action IDs
        """
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(action_id: str) -> bool:
            """DFS helper to detect cycles."""
            visited.add(action_id)
            rec_stack.add(action_id)
            path.append(action_id)

            # Get all next actions
            all_next = self.get_all_next_actions(action_id)
            for _conn_type, next_actions in all_next.items():
                for next_id, _ in next_actions:
                    if next_id not in visited:
                        if dfs(next_id):
                            return True
                    elif next_id in rec_stack:
                        # Found a cycle
                        cycle_start = path.index(next_id)
                        cycle = path[cycle_start:] + [next_id]
                        cycles.append(cycle)
                        logger.warning(f"Detected cycle: {' -> '.join(cycle)}")
                        return True

            path.pop()
            rec_stack.remove(action_id)
            return False

        # Check from all entry points
        entry_points = self.find_entry_points()
        for entry_id in entry_points:
            if entry_id not in visited:
                dfs(entry_id)

        # Check any remaining unvisited nodes (disconnected components)
        for action in self.workflow.actions:
            if action.id not in visited:
                dfs(action.id)

        return cycles

    def find_orphaned_actions(self) -> list[str]:
        """Find actions with no connections (neither incoming nor outgoing).

        Returns:
            List of orphaned action IDs
        """
        # Get all actions with incoming connections
        actions_with_incoming = set()
        for _source_id, conn_types in self.connections.root.items():
            for _conn_type, conn_lists in conn_types.items():
                for conn_list in conn_lists:
                    for connection in conn_list:
                        actions_with_incoming.add(connection.action)

        # Get all actions with outgoing connections
        actions_with_outgoing = set(self.connections.root.keys())

        # Orphans have neither incoming nor outgoing connections
        orphaned = [
            action.id
            for action in self.workflow.actions
            if action.id not in actions_with_incoming and action.id not in actions_with_outgoing
        ]

        if orphaned:
            logger.warning(f"Found {len(orphaned)} orphaned actions: {orphaned}")

        return orphaned

    def get_topological_order(self) -> list[str]:
        """Get topological ordering of actions.

        Returns actions in an order such that for every directed edge u -> v,
        u comes before v in the ordering. Only works for DAGs (acyclic graphs).

        Returns:
            List of action IDs in topological order

        Raises:
            ValueError: If graph contains cycles
        """
        cycles = self.detect_cycles()
        if cycles:
            raise ValueError(f"Cannot compute topological order: graph contains cycles: {cycles}")

        # Kahn's algorithm
        in_degree = {action.id: 0 for action in self.workflow.actions}

        # Calculate in-degrees
        for _source_id, conn_types in self.connections.root.items():
            for _conn_type, conn_lists in conn_types.items():
                for conn_list in conn_lists:
                    for connection in conn_list:
                        if connection.action in in_degree:
                            in_degree[connection.action] += 1

        # Queue for processing (start with zero in-degree nodes)
        queue = [action_id for action_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Process node with zero in-degree
            action_id = queue.pop(0)
            result.append(action_id)

            # Reduce in-degree for neighbors
            all_next = self.get_all_next_actions(action_id)
            for _conn_type, next_actions in all_next.items():
                for next_id, _ in next_actions:
                    in_degree[next_id] -= 1
                    if in_degree[next_id] == 0:
                        queue.append(next_id)

        logger.info(f"Computed topological order: {len(result)} actions")
        return result

    def calculate_action_depths(self) -> dict[str, int]:
        """Calculate depth of each action in the graph.

        Depth is the length of the longest path from any entry point to the action.
        Entry points have depth 0.

        Returns:
            Dictionary mapping action ID to depth
        """
        depths = {action.id: -1 for action in self.workflow.actions}

        def calculate_depth(action_id: str, current_depth: int, visited: set[str]) -> int:
            """Recursively calculate depth."""
            if action_id in visited:
                return depths[action_id]

            visited.add(action_id)
            depths[action_id] = max(depths[action_id], current_depth)

            # Calculate depth for all next actions
            all_next = self.get_all_next_actions(action_id)
            for _conn_type, next_actions in all_next.items():
                for next_id, _ in next_actions:
                    calculate_depth(next_id, current_depth + 1, visited.copy())

            return depths[action_id]

        # Start from entry points
        entry_points = self.find_entry_points()
        for entry_id in entry_points:
            calculate_depth(entry_id, 0, set())

        logger.info(f"Calculated depths for {len(depths)} actions")
        return depths

    def create_execution_path(self) -> ExecutionPath:
        """Create a new execution path for traversal.

        Returns:
            New ExecutionPath instance
        """
        path = ExecutionPath(self.workflow)
        self.execution_paths.append(path)
        return path

    def validate_workflow(self) -> tuple[bool, list[str]]:
        """Validate workflow structure.

        Checks for:
        - At least one entry point
        - No orphaned actions
        - All referenced actions exist
        - No cycles (warning only)

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check for entry points
        entry_points = self.find_entry_points()
        if not entry_points:
            errors.append("No entry points found - workflow has no starting actions")

        # Check for orphaned actions
        orphaned = self.find_orphaned_actions()
        if orphaned:
            errors.append(f"Found {len(orphaned)} orphaned actions: {orphaned}")

        # Check for invalid connections
        for source_id, conn_types in self.connections.root.items():
            if source_id not in self.action_map:
                errors.append(f"Connection source '{source_id}' does not exist")
                continue

            for _conn_type, conn_lists in conn_types.items():
                for conn_list in conn_lists:
                    for connection in conn_list:
                        if connection.action not in self.action_map:
                            errors.append(
                                f"Connection from '{source_id}' references non-existent "
                                f"action '{connection.action}'"
                            )

        # Check for cycles (warning only)
        cycles = self.detect_cycles()
        if cycles:
            logger.warning(f"Workflow contains {len(cycles)} cycle(s) - may cause infinite loops")

        is_valid = len(errors) == 0
        if is_valid:
            logger.info("Workflow validation passed")
        else:
            logger.error(f"Workflow validation failed with {len(errors)} errors")

        return is_valid, errors
