"""
Comprehensive tests for graph-based workflow execution.

Tests cover:
- Linear workflows (A → B → C)
- Branching workflows (IF actions)
- Loops (LOOP actions)
- Parallel outputs
- Cycle detection
- Orphaned action handling
- Error handling
"""

from typing import Any

import pytest

from qontinui.config.schema import Action, Connection, Connections, Variables, Workflow
from qontinui.execution import (
    ActionStatus,
    ConnectionResolver,
    CycleDetectedError,
    ExecutionState,
    ExecutionStatus,
    GraphTraverser,
    InfiniteLoopError,
    OrphanedActionsError,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_linear_workflow():
    """Create a simple linear workflow: action1 → action2 → action3"""
    return Workflow(
        id="test_linear",
        name="Linear Test",
        version="1.0.0",
        format="graph",
        actions=[
            Action(
                id="action1",
                type="CLICK",
                name="Action 1",
                config={"target": {"type": "coordinates", "coordinates": {"x": 0, "y": 0}}},
                position=(0, 0),
            ),
            Action(
                id="action2",
                type="WAIT",
                name="Action 2",
                config={"waitFor": "time", "duration": 1000},
                position=(100, 0),
            ),
            Action(id="action3", type="SCREENSHOT", name="Action 3", config={}, position=(200, 0)),
        ],
        connections=Connections(
            root={
                "action1": {"main": [[Connection(action="action2", type="main", index=0)]]},
                "action2": {"main": [[Connection(action="action3", type="main", index=0)]]},
            }
        ),
    )


@pytest.fixture
def branching_workflow():
    """Create a branching workflow with IF action"""
    return Workflow(
        id="test_branching",
        name="Branching Test",
        version="1.0.0",
        format="graph",
        actions=[
            Action(
                id="start",
                type="SET_VARIABLE",
                config={"variableName": "counter", "value": 5},
                position=(0, 0),
            ),
            Action(
                id="check",
                type="IF",
                config={
                    "condition": {"type": "expression", "expression": "counter > 3"},
                    "thenActions": [],
                    "elseActions": [],
                },
                position=(100, 0),
            ),
            Action(
                id="action_true",
                type="CLICK",
                config={"target": {"type": "coordinates", "coordinates": {"x": 100, "y": 0}}},
                position=(200, 50),
            ),
            Action(
                id="action_false",
                type="CLICK",
                config={"target": {"type": "coordinates", "coordinates": {"x": 0, "y": 100}}},
                position=(200, -50),
            ),
            Action(id="end", type="SCREENSHOT", config={}, position=(300, 0)),
        ],
        connections=Connections(
            root={
                "start": {"main": [[Connection(action="check", type="main", index=0)]]},
                "check": {
                    "true": [[Connection(action="action_true", type="true", index=0)]],
                    "false": [[Connection(action="action_false", type="false", index=0)]],
                },
                "action_true": {"main": [[Connection(action="end", type="main", index=0)]]},
                "action_false": {"main": [[Connection(action="end", type="main", index=0)]]},
            }
        ),
    )


@pytest.fixture
def loop_workflow():
    """Create a workflow with LOOP action"""
    return Workflow(
        id="test_loop",
        name="Loop Test",
        version="1.0.0",
        format="graph",
        actions=[
            Action(
                id="init",
                type="SET_VARIABLE",
                config={"variableName": "i", "value": 0},
                position=(0, 0),
            ),
            Action(
                id="loop",
                type="LOOP",
                config={"loopType": "FOR", "iterations": 3, "actions": []},
                position=(100, 0),
            ),
            Action(
                id="body",
                type="SET_VARIABLE",
                config={
                    "variableName": "i",
                    "valueSource": {"type": "expression", "expression": "i + 1"},
                },
                position=(200, 50),
            ),
            Action(id="end", type="SCREENSHOT", config={}, position=(300, 0)),
        ],
        connections=Connections(
            root={
                "init": {"main": [[Connection(action="loop", type="main", index=0)]]},
                "loop": {
                    "loop": [[Connection(action="body", type="loop", index=0)]],
                    "main": [[Connection(action="end", type="main", index=0)]],
                },
                "body": {"main": [[Connection(action="loop", type="main", index=0)]]},
            }
        ),
    )


@pytest.fixture
def cycle_workflow():
    """Create a workflow with a simple cycle (invalid)"""
    return Workflow(
        id="test_cycle",
        name="Cycle Test",
        version="1.0.0",
        format="graph",
        actions=[
            Action(
                id="action1",
                type="CLICK",
                config={"target": {"type": "coordinates", "coordinates": {"x": 0, "y": 0}}},
                position=(0, 0),
            )
        ],
        connections=Connections(
            root={"action1": {"main": [[Connection(action="action1", type="main", index=0)]]}}
        ),
    )


@pytest.fixture
def orphaned_workflow():
    """Create a workflow with orphaned actions"""
    return Workflow(
        id="test_orphaned",
        name="Orphaned Test",
        version="1.0.0",
        format="graph",
        actions=[
            Action(
                id="action1",
                type="CLICK",
                config={"target": {"type": "coordinates", "coordinates": {"x": 0, "y": 0}}},
                position=(0, 0),
            ),
            Action(
                id="action2",
                type="WAIT",
                config={"waitFor": "time", "duration": 1000},
                position=(100, 0),
            ),
            Action(id="orphan", type="SCREENSHOT", config={}, position=(200, 100)),
        ],
        connections=Connections(
            root={"action1": {"main": [[Connection(action="action2", type="main", index=0)]]}}
        ),
    )


@pytest.fixture
def parallel_outputs_workflow():
    """Create a workflow with parallel outputs"""
    return Workflow(
        id="test_parallel",
        name="Parallel Test",
        version="1.0.0",
        format="graph",
        actions=[
            Action(
                id="start",
                type="CLICK",
                config={"target": {"type": "coordinates", "coordinates": {"x": 0, "y": 0}}},
                position=(0, 0),
            ),
            Action(id="parallel1", type="SCREENSHOT", config={}, position=(100, 50)),
            Action(id="parallel2", type="SCREENSHOT", config={}, position=(100, -50)),
            Action(
                id="end",
                type="WAIT",
                config={"waitFor": "time", "duration": 100},
                position=(200, 0),
            ),
        ],
        connections=Connections(
            root={
                "start": {
                    "main": [
                        [Connection(action="parallel1", type="main", index=0)],
                        [Connection(action="parallel2", type="main", index=0)],
                    ]
                },
                "parallel1": {"main": [[Connection(action="end", type="main", index=0)]]},
                "parallel2": {"main": [[Connection(action="end", type="main", index=0)]]},
            }
        ),
    )


@pytest.fixture
def mock_executor():
    """Create a mock action executor that tracks execution"""
    executed_actions = []

    def executor(action: Action, context: dict[str, Any]) -> dict[str, Any]:
        executed_actions.append(action.id)

        # Mock different action types
        if action.type == "IF":
            # Check condition in context
            return {"success": True, "condition_met": context.get("counter", 0) > 3}
        elif action.type == "LOOP":
            # Mock loop behavior
            iterations = context.get("loop_iteration", 0)
            max_iterations = action.config.get("iterations", 3)
            should_continue = iterations < max_iterations
            context["loop_iteration"] = iterations + 1
            return {"success": True, "should_continue": should_continue}
        else:
            return {"success": True, "action_id": action.id}

    executor.executed_actions = executed_actions
    return executor


# ============================================================================
# ConnectionResolver Tests
# ============================================================================


def test_connection_resolver_basic(simple_linear_workflow):
    """Test basic connection resolution"""
    resolver = ConnectionResolver(simple_linear_workflow)

    # Test resolving main output
    connections = resolver.resolve_output_connection("action1", "main")
    assert len(connections) == 1
    assert connections[0].action == "action2"

    # Test getting connected actions
    actions = resolver.get_connected_actions("action1", "main")
    assert len(actions) == 1
    assert actions[0].id == "action2"


def test_connection_resolver_branching(branching_workflow):
    """Test connection resolution for branching"""
    resolver = ConnectionResolver(branching_workflow)

    # Test IF action outputs
    true_connections = resolver.resolve_output_connection("check", "true")
    assert len(true_connections) == 1
    assert true_connections[0].action == "action_true"

    false_connections = resolver.resolve_output_connection("check", "false")
    assert len(false_connections) == 1
    assert false_connections[0].action == "action_false"


def test_connection_resolver_invalid_output():
    """Test invalid output type"""
    workflow = Workflow(
        id="test",
        name="Test",
        version="1.0.0",
        format="graph",
        actions=[
            Action(
                id="action1",
                type="CLICK",
                config={"target": {"type": "coordinates", "coordinates": {"x": 0, "y": 0}}},
                position=(0, 0),
            )
        ],
        connections=Connections(root={}),
    )

    resolver = ConnectionResolver(workflow)

    # Should raise error for invalid output type
    with pytest.raises(ValueError, match="Invalid output type"):
        resolver.resolve_output_connection("action1", "invalid_output")


def test_connection_resolver_incoming_connections(simple_linear_workflow):
    """Test getting incoming connections"""
    resolver = ConnectionResolver(simple_linear_workflow)

    # action1 has no incoming connections
    incoming = resolver.get_incoming_connections("action1")
    assert len(incoming) == 0

    # action2 has one incoming connection from action1
    incoming = resolver.get_incoming_connections("action2")
    assert len(incoming) == 1
    assert incoming[0][0] == "action1"  # source action
    assert incoming[0][1] == "main"  # output type


def test_connection_resolver_multi_output(branching_workflow):
    """Test multi-output action detection"""
    resolver = ConnectionResolver(branching_workflow)

    # IF action is multi-output
    assert resolver.is_multi_output_action("check")

    # Regular actions are not
    assert not resolver.is_multi_output_action("start")


def test_connection_resolver_branch_actions(branching_workflow):
    """Test getting branch actions"""
    resolver = ConnectionResolver(branching_workflow)

    branches = resolver.get_branch_actions("check")
    assert "true" in branches
    assert "false" in branches
    assert len(branches["true"]) == 1
    assert len(branches["false"]) == 1


# ============================================================================
# ExecutionState Tests
# ============================================================================


def test_execution_state_initialization():
    """Test execution state initialization"""
    state = ExecutionState("test_workflow")

    assert state.workflow_id == "test_workflow"
    assert state.status == ExecutionStatus.PENDING
    assert not state.is_visited("action1")
    assert not state.has_pending()


def test_execution_state_visited_tracking():
    """Test visited action tracking"""
    state = ExecutionState("test_workflow")

    # Mark actions as visited
    state.mark_visited("action1")
    state.mark_visited("action2")

    assert state.is_visited("action1")
    assert state.is_visited("action2")
    assert not state.is_visited("action3")


def test_execution_state_pending_queue():
    """Test pending action queue"""
    state = ExecutionState("test_workflow")

    # Add pending actions
    state.add_pending("action1", depth=0)
    state.add_pending("action2", depth=1)
    state.add_pending("action3", depth=1)

    assert state.get_pending_count() == 3
    assert state.has_pending()

    # Get pending actions in order
    pending1 = state.get_next_pending()
    assert pending1.action_id == "action1"
    assert pending1.depth == 0

    pending2 = state.get_next_pending()
    assert pending2.action_id == "action2"

    assert state.get_pending_count() == 1


def test_execution_state_history():
    """Test execution history recording"""
    state = ExecutionState("test_workflow", enable_history=True)

    # Start and complete an action
    record = state.start_action("action1", "CLICK")
    record.complete({"success": True}, "main", 0)

    # Check history
    history = state.get_history()
    assert len(history) == 1
    assert history[0].action_id == "action1"
    assert history[0].status == ActionStatus.COMPLETED


def test_execution_state_context():
    """Test context management"""
    state = ExecutionState("test_workflow")

    # Set context values
    state.set_context("key1", "value1")
    state.set_context("key2", 42)

    assert state.get_context("key1") == "value1"
    assert state.get_context("key2") == 42
    assert state.get_context("missing", "default") == "default"

    # Update multiple values
    state.update_context({"key3": True, "key4": [1, 2, 3]})
    context = state.get_all_context()
    assert len(context) == 4


def test_execution_state_statistics():
    """Test statistics generation"""
    state = ExecutionState("test_workflow")
    state.start()

    state.mark_visited("action1")
    state.mark_visited("action2")
    state.add_pending("action3")

    record = state.start_action("action1", "CLICK")
    record.complete({"success": True}, "main", 0)

    stats = state.get_statistics()
    assert stats["workflow_id"] == "test_workflow"
    assert stats["status"] == ExecutionStatus.RUNNING.value
    assert stats["visited_count"] == 2
    assert stats["pending_count"] == 1
    assert stats["completed_count"] == 1


# ============================================================================
# GraphTraverser Tests
# ============================================================================


def test_graph_traverser_linear_workflow(simple_linear_workflow, mock_executor):
    """Test executing a simple linear workflow"""
    traverser = GraphTraverser(simple_linear_workflow, action_executor=mock_executor)

    result = traverser.traverse()

    assert result["status"] == "completed"
    assert len(mock_executor.executed_actions) == 3
    assert mock_executor.executed_actions == ["action1", "action2", "action3"]


def test_graph_traverser_entry_points(simple_linear_workflow):
    """Test finding entry points"""
    traverser = GraphTraverser(simple_linear_workflow)

    entry_points = traverser.get_entry_actions()
    assert len(entry_points) == 1
    assert entry_points[0].id == "action1"


def test_graph_traverser_branching_workflow(branching_workflow, mock_executor):
    """Test executing branching workflow"""
    traverser = GraphTraverser(branching_workflow, action_executor=mock_executor)

    # Execute with counter > 3 (should take true branch)
    result = traverser.traverse(context={"counter": 5})

    assert result["status"] == "completed"
    assert "action_true" in mock_executor.executed_actions
    assert "action_false" not in mock_executor.executed_actions


def test_graph_traverser_branching_false_branch(branching_workflow, mock_executor):
    """Test executing branching workflow (false branch)"""
    traverser = GraphTraverser(branching_workflow, action_executor=mock_executor)

    # Execute with counter <= 3 (should take false branch)
    result = traverser.traverse(context={"counter": 2})

    assert result["status"] == "completed"
    assert "action_false" in mock_executor.executed_actions
    assert "action_true" not in mock_executor.executed_actions


def test_graph_traverser_loop_workflow(loop_workflow, mock_executor):
    """Test executing loop workflow"""
    traverser = GraphTraverser(loop_workflow, action_executor=mock_executor)

    result = traverser.traverse()

    assert result["status"] == "completed"
    # Loop body should execute 3 times
    body_count = mock_executor.executed_actions.count("body")
    assert body_count == 3


def test_graph_traverser_cycle_detection(cycle_workflow):
    """Test cycle detection"""
    with pytest.raises(CycleDetectedError, match="Simple cycle detected"):
        GraphTraverser(cycle_workflow)


def test_graph_traverser_orphaned_actions(orphaned_workflow):
    """Test orphaned action detection"""
    with pytest.raises(OrphanedActionsError, match="orphaned actions"):
        GraphTraverser(orphaned_workflow)


def test_graph_traverser_no_entry_points():
    """Test workflow with no entry points"""
    workflow = Workflow(
        id="test",
        name="Test",
        version="1.0.0",
        format="graph",
        actions=[
            Action(
                id="action1",
                type="CLICK",
                config={"target": {"type": "coordinates", "coordinates": {"x": 0, "y": 0}}},
                position=(0, 0),
            ),
            Action(
                id="action2",
                type="WAIT",
                config={"waitFor": "time", "duration": 1000},
                position=(100, 0),
            ),
        ],
        connections=Connections(
            root={
                "action1": {"main": [[Connection(action="action2", type="main", index=0)]]},
                "action2": {"main": [[Connection(action="action1", type="main", index=0)]]},
            }
        ),
    )

    with pytest.raises(ValueError, match="no entry points"):
        GraphTraverser(workflow)


def test_graph_traverser_iteration_limit():
    """Test iteration limit enforcement"""
    # Create workflow with potential infinite loop
    workflow = Workflow(
        id="test_limit",
        name="Test Limit",
        version="1.0.0",
        format="graph",
        actions=[
            Action(
                id="loop",
                type="LOOP",
                config={
                    "loopType": "WHILE",
                    "condition": {"type": "expression", "expression": "True"},
                },
                position=(0, 0),
            ),
            Action(
                id="body", type="WAIT", config={"waitFor": "time", "duration": 1}, position=(100, 0)
            ),
        ],
        connections=Connections(
            root={
                "loop": {"loop": [[Connection(action="body", type="loop", index=0)]]},
                "body": {"main": [[Connection(action="loop", type="main", index=0)]]},
            }
        ),
    )

    def mock_loop_executor(action: Action, context: dict[str, Any]) -> dict[str, Any]:
        # Always continue loop
        return {"success": True, "should_continue": True}

    traverser = GraphTraverser(workflow, action_executor=mock_loop_executor, max_iterations=10)

    with pytest.raises(InfiniteLoopError, match="Iteration limit"):
        traverser.traverse()


def test_graph_traverser_error_handling():
    """Test error handling during execution"""
    workflow = Workflow(
        id="test_error",
        name="Test Error",
        version="1.0.0",
        format="graph",
        actions=[
            Action(
                id="action1",
                type="CLICK",
                config={"target": {"type": "coordinates", "coordinates": {"x": 0, "y": 0}}},
                position=(0, 0),
            ),
            Action(id="error_handler", type="SCREENSHOT", config={}, position=(100, 100)),
        ],
        connections=Connections(
            root={
                "action1": {"error": [[Connection(action="error_handler", type="error", index=0)]]}
            }
        ),
    )

    executed_actions = []

    def failing_executor(action: Action, context: dict[str, Any]) -> dict[str, Any]:
        executed_actions.append(action.id)
        if action.id == "action1":
            raise Exception("Test error")
        return {"success": True}

    traverser = GraphTraverser(workflow, action_executor=failing_executor)
    _result = traverser.traverse()

    # Should execute error handler
    assert "error_handler" in executed_actions


def test_graph_traverser_pause_resume(simple_linear_workflow, mock_executor):
    """Test pause/resume functionality"""
    traverser = GraphTraverser(simple_linear_workflow, action_executor=mock_executor)

    # Set breakpoint at action2
    traverser.set_breakpoint("action2")

    # Execute until breakpoint
    result = traverser.traverse()

    assert result["status"] == "paused"
    assert "action1" in mock_executor.executed_actions
    assert "action2" not in mock_executor.executed_actions

    # Resume execution
    result = traverser.resume()

    assert result["status"] == "completed"
    assert len(mock_executor.executed_actions) == 3


def test_graph_traverser_execution_path(simple_linear_workflow, mock_executor):
    """Test getting execution path"""
    traverser = GraphTraverser(simple_linear_workflow, action_executor=mock_executor)

    traverser.traverse()

    path = traverser.get_execution_path()
    assert path == ["action1", "action2", "action3"]


def test_graph_traverser_statistics(simple_linear_workflow, mock_executor):
    """Test getting execution statistics"""
    traverser = GraphTraverser(simple_linear_workflow, action_executor=mock_executor)

    traverser.traverse()

    stats = traverser.get_statistics()
    assert stats["workflow_id"] == "test_linear"
    assert stats["status"] == "completed"
    assert stats["completed_count"] == 3


def test_graph_traverser_context_propagation(simple_linear_workflow):
    """Test context propagation through workflow"""
    executed_contexts = []

    def context_tracking_executor(action: Action, context: dict[str, Any]) -> dict[str, Any]:
        executed_contexts.append(context.copy())
        return {"success": True, f"result_{action.id}": f"value_{action.id}"}

    traverser = GraphTraverser(simple_linear_workflow, action_executor=context_tracking_executor)

    _result = traverser.traverse(context={"initial": "value"})

    # Check that context propagates
    assert executed_contexts[0]["initial"] == "value"
    assert "result_action1" in executed_contexts[1]
    assert "result_action2" in executed_contexts[2]


def test_graph_traverser_with_variables():
    """Test workflow with initial variables"""
    workflow = Workflow(
        id="test_vars",
        name="Test Variables",
        version="1.0.0",
        format="graph",
        actions=[
            Action(
                id="action1",
                type="CLICK",
                config={"target": {"type": "coordinates", "coordinates": {"x": 0, "y": 0}}},
                position=(0, 0),
            )
        ],
        connections=Connections(root={}),
        variables=Variables(
            local={"local_var": "local_value"},
            process={"process_var": "process_value"},
            global_vars={"global_var": "global_value"},
        ),
    )

    context_received = []

    def var_checking_executor(action: Action, context: dict[str, Any]) -> dict[str, Any]:
        context_received.append(context.copy())
        return {"success": True}

    traverser = GraphTraverser(workflow, action_executor=var_checking_executor)
    _result = traverser.traverse()

    # Check variables are in context
    assert context_received[0]["local_var"] == "local_value"
    assert context_received[0]["process_var"] == "process_value"
    assert context_received[0]["global_var"] == "global_value"


def test_graph_traverser_start_from_specific_action(simple_linear_workflow, mock_executor):
    """Test starting execution from a specific action"""
    traverser = GraphTraverser(simple_linear_workflow, action_executor=mock_executor)

    # Start from action2
    result = traverser.traverse(start_action_id="action2")

    assert result["status"] == "completed"
    assert "action1" not in mock_executor.executed_actions
    assert "action2" in mock_executor.executed_actions
    assert "action3" in mock_executor.executed_actions


# ============================================================================
# Integration Tests
# ============================================================================


def test_complex_workflow_integration():
    """Test a complex workflow with multiple features"""
    workflow = Workflow(
        id="test_complex",
        name="Complex Test",
        version="1.0.0",
        format="graph",
        actions=[
            Action(
                id="init",
                type="SET_VARIABLE",
                config={"variableName": "counter", "value": 0},
                position=(0, 0),
            ),
            Action(
                id="loop",
                type="LOOP",
                config={"loopType": "FOR", "iterations": 3, "actions": []},
                position=(100, 0),
            ),
            Action(
                id="increment",
                type="SET_VARIABLE",
                config={
                    "variableName": "counter",
                    "valueSource": {"type": "expression", "expression": "counter + 1"},
                },
                position=(200, 50),
            ),
            Action(
                id="check",
                type="IF",
                config={
                    "condition": {"type": "expression", "expression": "counter >= 2"},
                    "thenActions": [],
                    "elseActions": [],
                },
                position=(300, 0),
            ),
            Action(id="action_true", type="SCREENSHOT", config={}, position=(400, 50)),
            Action(
                id="action_false",
                type="WAIT",
                config={"waitFor": "time", "duration": 100},
                position=(400, -50),
            ),
            Action(id="end", type="SCREENSHOT", config={}, position=(500, 0)),
        ],
        connections=Connections(
            root={
                "init": {"main": [[Connection(action="loop", type="main", index=0)]]},
                "loop": {
                    "loop": [[Connection(action="increment", type="loop", index=0)]],
                    "main": [[Connection(action="check", type="main", index=0)]],
                },
                "increment": {"main": [[Connection(action="loop", type="main", index=0)]]},
                "check": {
                    "true": [[Connection(action="action_true", type="true", index=0)]],
                    "false": [[Connection(action="action_false", type="false", index=0)]],
                },
                "action_true": {"main": [[Connection(action="end", type="main", index=0)]]},
                "action_false": {"main": [[Connection(action="end", type="main", index=0)]]},
            }
        ),
    )

    executed_actions = []

    def tracking_executor(action: Action, context: dict[str, Any]) -> dict[str, Any]:
        executed_actions.append(action.id)

        if action.type == "LOOP":
            iterations = context.get("loop_iteration", 0)
            max_iterations = action.config.get("iterations", 3)
            should_continue = iterations < max_iterations
            context["loop_iteration"] = iterations + 1
            return {"success": True, "should_continue": should_continue}
        elif action.type == "IF":
            condition_met = context.get("counter", 0) >= 2
            return {"success": True, "condition_met": condition_met}
        else:
            return {"success": True}

    traverser = GraphTraverser(workflow, action_executor=tracking_executor)
    result = traverser.traverse()

    assert result["status"] == "completed"
    assert "init" in executed_actions
    assert "loop" in executed_actions
    assert executed_actions.count("increment") == 3  # Loop 3 times
    assert "check" in executed_actions
    # After 3 increments, counter should be >= 2, so true branch
    assert "action_true" in executed_actions
    assert "end" in executed_actions
