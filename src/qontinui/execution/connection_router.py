"""
Connection router for intelligent workflow graph traversal.

Routes execution flow through the workflow graph based on action results.
Handles different action types with specialized routing logic.

Clean, focused implementation for graph-based workflows only.
"""

from typing import Any

from ..config.schema import Action, Connections, Workflow
from .output_resolver import OutputResolver, OutputTypeValidator
from .routing_context import RoutingContext


class ConnectionRouter:
    """
    Routes workflow execution through the graph based on action results.

    Determines next actions to execute based on:
    - Action type and execution result
    - Connection graph structure
    - Output types (main, error, true/false, case_N, etc.)
    """

    def __init__(
        self,
        workflow: Workflow | None = None,
        resolver: OutputResolver | None = None,
        context: RoutingContext | None = None,
    ) -> None:
        """
        Initialize connection router.

        Args:
            workflow: Workflow to route through (optional, can be set per call)
            resolver: Output resolver instance (creates new if not provided)
            context: Routing context for tracking (creates new if not provided)
        """
        self.workflow = workflow
        self.resolver = resolver or OutputResolver()
        self.validator = OutputTypeValidator(self.resolver)
        self.context = context or RoutingContext()

    def route(
        self, action: Action, execution_result: dict[str, Any], connections: Connections
    ) -> list[tuple[str, str, int]]:
        """
        Determine next actions based on execution result.

        Args:
            action: The executed action
            execution_result: Result of action execution
            connections: Connection graph

        Returns:
            List of tuples (action_id, connection_type, input_index)
            representing next actions to execute

        Raises:
            ValueError: If routing fails due to invalid result or missing connections
        """
        # Validate result structure
        is_valid, error_msg = self.resolver.validate_result_structure(
            action, execution_result
        )
        if not is_valid:
            raise ValueError(
                f"Invalid execution result for {action.type} action: {error_msg}"
            )

        # Determine output type from result
        output_type = self.resolver.resolve(action, execution_result)

        # Get connections for this action and output type
        next_actions = self._get_next_actions(action.id, output_type, connections)

        # Record routing decisions
        for target_id, conn_type, input_idx in next_actions:
            self.context.record_route(
                from_action=action.id,
                to_action=target_id,
                output_type=conn_type,
                input_index=input_idx,
                execution_result=execution_result,
            )

        return next_actions

    def route_from_entry_point(
        self, action_id: str, connections: Connections
    ) -> list[tuple[str, str, int]]:
        """
        Route from an entry point (initial action with no incoming connections).

        Args:
            action_id: Entry point action ID
            connections: Connection graph

        Returns:
            List of next actions (usually just the entry point itself)
        """
        # Entry points execute themselves first
        return [(action_id, "main", 0)]

    def get_action_output_type(self, action: Action, result: dict[str, Any]) -> str:
        """
        Get output type for an action based on its execution result.

        Args:
            action: The executed action
            result: Execution result

        Returns:
            Output type string (e.g., 'main', 'true', 'error')
        """
        return self.resolver.resolve(action, result)

    def _get_next_actions(
        self, action_id: str, output_type: str, connections: Connections
    ) -> list[tuple[str, str, int]]:
        """
        Get next actions for a given action and output type.

        Args:
            action_id: Source action ID
            output_type: Output type to follow
            connections: Connection graph

        Returns:
            List of (action_id, connection_type, input_index) tuples
        """
        # Get connections for this action
        action_connections = connections.root.get(action_id, {})

        # Get connections for specific output type
        output_connections = action_connections.get(output_type, [])

        # Flatten connection lists and extract target info
        next_actions = []
        for conn_list in output_connections:
            for conn in conn_list:
                next_actions.append((conn.action, conn.type, conn.index))

        return next_actions

    def validate_routing(self, workflow: Workflow) -> tuple[bool, list[str]]:
        """
        Validate that all actions have proper routing connections.

        Args:
            workflow: Workflow to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        for action in workflow.actions:
            # Get valid output types for this action
            valid_outputs = self.resolver.get_valid_outputs(action)

            # Get used connection types
            action_connections = workflow.connections.root.get(action.id, {})
            used_types = list(action_connections.keys())

            # Check for invalid connection types
            invalid = self.validator.get_invalid_outputs(action, used_types)
            if invalid:
                errors.append(
                    f"Action '{action.id}' ({action.type}) has invalid connection types: {invalid}. "
                    f"Valid types: {valid_outputs}"
                )

            # Warn about missing outputs (informational, not an error)
            # Control flow actions should have all outputs defined
            if action.type in ["IF", "SWITCH", "TRY_CATCH"]:
                missing = self.validator.get_missing_outputs(action, used_types)
                # Remove 'error' from missing if not critical
                missing = [m for m in missing if m != "error"]
                if missing:
                    errors.append(
                        f"Action '{action.id}' ({action.type}) missing connections for outputs: {missing}"
                    )

        return len(errors) == 0, errors

    def get_routing_options(
        self, action: Action, connections: Connections
    ) -> dict[str, list[str]]:
        """
        Get all possible routing options for an action.

        Args:
            action: The action
            connections: Connection graph

        Returns:
            Dictionary mapping output types to lists of target action IDs
        """
        valid_outputs = self.resolver.get_valid_outputs(action)
        options = {}

        for output_type in valid_outputs:
            next_actions = self._get_next_actions(action.id, output_type, connections)
            if next_actions:
                options[output_type] = [aid for aid, _, _ in next_actions]

        return options

    def find_reachable_actions(
        self,
        start_action_id: str,
        connections: Connections,
        max_depth: int | None = None,
    ) -> set[str]:
        """
        Find all actions reachable from a starting action.

        Args:
            start_action_id: Starting action ID
            connections: Connection graph
            max_depth: Maximum depth to search (None for unlimited)

        Returns:
            Set of reachable action IDs
        """
        reachable = set()
        to_visit = [(start_action_id, 0)]  # (action_id, depth)
        visited = set()

        while to_visit:
            current_id, depth = to_visit.pop(0)

            if current_id in visited:
                continue

            if max_depth is not None and depth > max_depth:
                continue

            visited.add(current_id)
            reachable.add(current_id)

            # Get all connections from this action
            action_connections = connections.root.get(current_id, {})
            for _output_type, conn_lists in action_connections.items():
                for conn_list in conn_lists:
                    for conn in conn_list:
                        if conn.action not in visited:
                            to_visit.append((conn.action, depth + 1))

        return reachable

    def find_unreachable_actions(
        self,
        entry_points: list[str],
        all_action_ids: list[str],
        connections: Connections,
    ) -> list[str]:
        """
        Find actions that are unreachable from entry points.

        Args:
            entry_points: List of entry point action IDs
            all_action_ids: List of all action IDs in workflow
            connections: Connection graph

        Returns:
            List of unreachable action IDs
        """
        reachable = set()

        for entry in entry_points:
            reachable.update(self.find_reachable_actions(entry, connections))

        return [aid for aid in all_action_ids if aid not in reachable]

    def get_execution_paths(
        self,
        start_action_id: str,
        end_action_id: str,
        connections: Connections,
        max_paths: int = 10,
    ) -> list[list[str]]:
        """
        Find execution paths from start to end action.

        Args:
            start_action_id: Starting action ID
            end_action_id: Target action ID
            connections: Connection graph
            max_paths: Maximum number of paths to return

        Returns:
            List of paths, where each path is a list of action IDs
        """
        paths: list[list[str]] = []

        def dfs(current_id: str, path: list[str], visited: set[str]):
            if len(paths) >= max_paths:
                return

            if current_id == end_action_id:
                paths.append(path.copy())
                return

            if current_id in visited:
                return

            visited.add(current_id)

            # Explore all connections
            action_connections = connections.root.get(current_id, {})
            for _output_type, conn_lists in action_connections.items():
                for conn_list in conn_lists:
                    for conn in conn_list:
                        path.append(conn.action)
                        dfs(conn.action, path, visited.copy())
                        path.pop()

        dfs(start_action_id, [start_action_id], set())
        return paths

    def analyze_convergence_points(
        self, connections: Connections, all_action_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """
        Find convergence points (actions with multiple incoming connections).

        Args:
            connections: Connection graph
            all_action_ids: List of all action IDs

        Returns:
            Dictionary mapping action IDs to list of (source_action, output_type) tuples
        """
        incoming: dict[str, list[tuple[str, str]]] = {aid: [] for aid in all_action_ids}

        for source_id, output_types in connections.root.items():
            for output_type, conn_lists in output_types.items():
                for conn_list in conn_lists:
                    for conn in conn_list:
                        incoming[conn.action].append((source_id, output_type))

        # Return only convergence points (multiple incoming)
        return {
            action_id: sources
            for action_id, sources in incoming.items()
            if len(sources) > 1
        }

    def get_critical_path(self, workflow: Workflow) -> list[str]:
        """
        Get critical path (longest path from entry to exit).

        Args:
            workflow: Workflow to analyze

        Returns:
            List of action IDs forming the critical path
        """
        from ..config.workflow_utils import find_entry_points, find_exit_points

        entry_points = find_entry_points(workflow)
        exit_points = find_exit_points(workflow)

        if not entry_points or not exit_points:
            return []

        # Find longest path
        longest_path: list[str] = []

        for entry in entry_points:
            for exit_point in exit_points:
                paths = self.get_execution_paths(
                    entry, exit_point, workflow.connections, max_paths=100
                )
                for path in paths:
                    if len(path) > len(longest_path):
                        longest_path = path

        return longest_path

    def estimate_execution_complexity(self, workflow: Workflow) -> dict[str, Any]:
        """
        Estimate execution complexity of workflow.

        Args:
            workflow: Workflow to analyze

        Returns:
            Dictionary with complexity metrics
        """
        from ..config.workflow_utils import find_entry_points

        entry_points = find_entry_points(workflow)
        all_action_ids = [a.id for a in workflow.actions]

        # Calculate metrics
        convergence = self.analyze_convergence_points(
            workflow.connections, all_action_ids
        )
        critical_path = self.get_critical_path(workflow)

        metrics = {
            "total_actions": len(workflow.actions),
            "entry_points": len(entry_points),
            "convergence_points": len(convergence),
            "critical_path_length": len(critical_path),
            "max_convergence": (
                max([len(v) for v in convergence.values()]) if convergence else 0
            ),
        }

        # Count branching points
        branching_count = 0
        for action in workflow.actions:
            if action.type in ["IF", "SWITCH"]:
                branching_count += 1

        metrics["branching_points"] = branching_count

        # Calculate cyclomatic complexity estimate
        # V(G) = E - N + 2P (edges - nodes + 2*connected_components)
        edge_count = sum(
            len(conn)  # type: ignore[arg-type,misc]
            for output_types in workflow.connections.root.values()
            for conn_lists in output_types.values()
            for conn_list in conn_lists
            for conn in conn_list
        )

        metrics["edge_count"] = edge_count
        metrics["cyclomatic_complexity"] = edge_count - len(workflow.actions) + 2

        return metrics
