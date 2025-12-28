"""
Utility functions for working with workflows - graph format only.

Clean utilities for graph-based workflows.
No format detection, no sequential format support.
"""

from typing import Any, Literal

from .schema import Action, Workflow

# Type alias for initial states source
InitialStatesSource = Literal["override", "workflow", "defaults"]


class ResolvedInitialStates:
    """Result of resolving initial states for a workflow.

    Attributes:
        state_ids: List of state IDs that will be active initially
        source: Where the initial states came from ("override", "workflow", or "defaults")
        states: Optional list of state info dicts with id and name
    """

    def __init__(
        self,
        state_ids: list[str],
        source: InitialStatesSource,
        states: list[dict[str, str]] | None = None,
    ) -> None:
        self.state_ids = state_ids
        self.source = source
        self.states = states or [{"id": sid, "name": sid} for sid in state_ids]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "stateIds": self.state_ids,
            "source": self.source,
            "states": self.states,
        }


def resolve_initial_states(
    config: dict[str, Any],
    workflow_id: str,
    override_ids: list[str] | None = None,
) -> ResolvedInitialStates:
    """Resolve initial state IDs for a workflow with priority system.

    This function implements the same logic as the runner's resolve_initial_states
    function, enabling consistent behavior across the entire qontinui ecosystem.

    Priority (highest to lowest):
    1. Override IDs passed directly (runner session override)
    2. Workflow's `initialStateIds` field (workflow-level configuration)
    3. States with `is_initial: true` or `isInitial: true` (state machine defaults)

    Args:
        config: Configuration dict containing 'workflows' and 'states' keys.
            Can be a raw dict from JSON or a validated config object converted to dict.
        workflow_id: The workflow ID to resolve initial states for
        override_ids: Optional override from runner UI (highest priority)

    Returns:
        ResolvedInitialStates with state_ids, source, and state info

    Example:
        >>> config = {"workflows": [...], "states": [...]}
        >>> result = resolve_initial_states(config, "my-workflow")
        >>> print(result.state_ids)  # ['state-1', 'state-2']
        >>> print(result.source)      # 'workflow' or 'defaults'
    """
    # Priority 1: Use override if provided
    if override_ids and len(override_ids) > 0:
        states_info = _get_state_info_for_ids(config, override_ids)
        return ResolvedInitialStates(
            state_ids=override_ids,
            source="override",
            states=states_info,
        )

    # Priority 2: Check workflow.initialStateIds
    workflows = config.get("workflows", [])
    for workflow in workflows:
        wf_id = workflow.get("id")
        if wf_id == workflow_id:
            # Check both camelCase and snake_case
            initial_ids = workflow.get("initialStateIds") or workflow.get("initial_state_ids")
            if initial_ids and len(initial_ids) > 0:
                states_info = _get_state_info_for_ids(config, initial_ids)
                return ResolvedInitialStates(
                    state_ids=initial_ids,
                    source="workflow",
                    states=states_info,
                )
            break

    # Priority 3: Fall back to states with is_initial=true
    states = config.get("states", [])
    default_ids: list[str] = []
    default_states: list[dict[str, str]] = []

    for state in states:
        # Check both camelCase and snake_case
        is_initial = state.get("isInitial") or state.get("is_initial")
        if is_initial:
            state_id = state.get("id")
            if state_id:
                default_ids.append(state_id)
                default_states.append(
                    {
                        "id": state_id,
                        "name": state.get("name", state_id),
                    }
                )

    return ResolvedInitialStates(
        state_ids=default_ids,
        source="defaults",
        states=default_states,
    )


def get_initial_states_source(
    config: dict[str, Any],
    workflow_id: str,
    override_ids: list[str] | None = None,
) -> InitialStatesSource:
    """Determine the source of resolved initial states.

    This is a convenience function that returns only the source type
    without computing full state information.

    Args:
        config: Configuration dict containing 'workflows' and 'states' keys
        workflow_id: The workflow ID to check
        override_ids: Optional override IDs to check

    Returns:
        The source type: "override", "workflow", or "defaults"
    """
    # Check override first
    if override_ids and len(override_ids) > 0:
        return "override"

    # Check workflow.initialStateIds
    workflows = config.get("workflows", [])
    for workflow in workflows:
        if workflow.get("id") == workflow_id:
            initial_ids = workflow.get("initialStateIds") or workflow.get("initial_state_ids")
            if initial_ids and len(initial_ids) > 0:
                return "workflow"
            break

    return "defaults"


def _get_state_info_for_ids(
    config: dict[str, Any],
    state_ids: list[str],
) -> list[dict[str, str]]:
    """Get state info (id and name) for a list of state IDs.

    Args:
        config: Configuration dict containing 'states' key
        state_ids: List of state IDs to look up

    Returns:
        List of dicts with 'id' and 'name' keys
    """
    states = config.get("states", [])
    state_lookup = {s.get("id"): s.get("name", s.get("id")) for s in states}

    return [{"id": sid, "name": state_lookup.get(sid, sid)} for sid in state_ids]


def get_action_output_count(action: Action) -> int:
    """
    Get number of outputs for an action type.

    Different action types have different numbers of outputs:
    - IF: 2 outputs (true, false)
    - SWITCH: N outputs (one per case + default)
    - TRY_CATCH: 3 outputs (try, catch, finally)
    - Most actions: 1 output (main)
    """
    if action.type == "IF":
        return 2

    elif action.type == "SWITCH":
        cases = action.config.get("cases", [])
        return len(cases) + 1  # +1 for default case

    elif action.type == "TRY_CATCH":
        return 3  # try, catch, finally

    else:
        return 1  # Most actions have single output


def has_merge_nodes(workflow: Workflow) -> bool:
    """
    Check if workflow has merge points (multiple actions connecting to same target).
    """
    if workflow.connections is None:
        return False

    # Count incoming connections for each action
    incoming_counts: dict[str, int] = {action.id: 0 for action in workflow.actions}

    for connection_types in workflow.connections.root.values():
        for connections_list in connection_types.values():
            for connections in connections_list:
                for conn in connections:
                    if conn.action in incoming_counts:
                        incoming_counts[conn.action] += 1

    return any(count > 1 for count in incoming_counts.values())


def find_entry_points(workflow: Workflow) -> list[str]:
    """
    Find entry point actions (actions with no incoming connections).
    """
    if workflow.connections is None:
        return [action.id for action in workflow.actions]

    # Find actions with no incoming connections
    has_incoming: set[str] = set()

    for connection_types in workflow.connections.root.values():
        for connections_list in connection_types.values():
            for connections in connections_list:
                for conn in connections:
                    has_incoming.add(conn.action)

    return [action.id for action in workflow.actions if action.id not in has_incoming]


def find_exit_points(workflow: Workflow) -> list[str]:
    """
    Find exit point actions (actions with no outgoing connections).
    """
    if workflow.connections is None:
        return [action.id for action in workflow.actions]

    # Find actions with no outgoing connections
    has_outgoing = set(workflow.connections.root.keys())

    return [action.id for action in workflow.actions if action.id not in has_outgoing]


def get_workflow_statistics(workflow: Workflow) -> dict[str, Any]:
    """
    Get statistics about a workflow.
    """
    from .workflow_validation import detect_cycles

    stats = {
        "format": "graph",
        "total_actions": len(workflow.actions),
        "action_types": {},
        "entry_points": find_entry_points(workflow),
        "exit_points": find_exit_points(workflow),
        "entry_point_count": len(find_entry_points(workflow)),
        "exit_point_count": len(find_exit_points(workflow)),
    }

    # Count action types
    for action in workflow.actions:
        action_type = action.type
        stats["action_types"][action_type] = stats["action_types"].get(action_type, 0) + 1  # type: ignore[attr-defined,index]

    # Count total connections
    total_connections = 0
    if workflow.connections:
        for connection_types in workflow.connections.root.values():
            for connections_list in connection_types.values():
                for connections in connections_list:
                    total_connections += len(connections)

    stats["connection_count"] = total_connections
    stats["has_cycles"] = detect_cycles(workflow)
    stats["has_merge_points"] = has_merge_nodes(workflow)

    # Calculate max path length (depth)
    max_depth = calculate_max_depth(workflow)
    stats["max_path_length"] = max_depth

    # Variable statistics
    if workflow.variables:
        stats["variable_scopes"] = []
        if workflow.variables.local:
            stats["variable_scopes"].append("local")  # type: ignore[attr-defined]
            stats["local_variable_count"] = len(workflow.variables.local)
        if workflow.variables.process:
            stats["variable_scopes"].append("process")  # type: ignore[attr-defined]
            stats["process_variable_count"] = len(workflow.variables.process)
        if workflow.variables.global_vars:
            stats["variable_scopes"].append("global")  # type: ignore[attr-defined]
            stats["global_variable_count"] = len(workflow.variables.global_vars)

    return stats


def calculate_max_depth(workflow: Workflow) -> int:
    """
    Calculate maximum path length (depth) in a graph workflow.

    Uses DFS to find the longest path from any entry point to any exit point.
    """
    if workflow.connections is None:
        return len(workflow.actions)

    # Build adjacency list
    graph: dict[str, list[str]] = {action.id: [] for action in workflow.actions}

    for source_id, connection_types in workflow.connections.root.items():
        for connections_list in connection_types.values():
            for connections in connections_list:
                for conn in connections:
                    if conn.action in graph:
                        graph[source_id].append(conn.action)

    # DFS to find max depth
    def dfs(node: str, visited: set[str]) -> int:
        if not graph[node]:  # Exit point
            return 1

        visited.add(node)
        max_child_depth = 0

        for neighbor in graph[node]:
            if neighbor not in visited:
                child_depth = dfs(neighbor, visited.copy())
                max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth + 1

    # Start from all entry points
    entry_points = find_entry_points(workflow)
    max_depth = 0

    for entry in entry_points:
        depth = dfs(entry, set())
        max_depth = max(max_depth, depth)

    return max_depth


def get_action_by_id(workflow: Workflow, action_id: str) -> Action | None:
    """Get action by ID."""
    for action in workflow.actions:
        if action.id == action_id:
            return action
    return None


def get_connected_actions(workflow: Workflow, action_id: str) -> dict[str, list[str]]:
    """
    Get all actions connected to a specific action.

    Returns dictionary mapping connection types to lists of target action IDs.
    """
    if workflow.connections is None:
        return {}

    result: dict[str, list[str]] = {}
    connections = workflow.connections.root.get(action_id, {})

    for conn_type, connections_list in connections.items():
        targets = []
        for connections in connections_list:  # type: ignore[assignment]
            for conn in connections:
                targets.append(conn.action)  # type: ignore[attr-defined]
        if targets:
            result[conn_type] = targets

    return result
