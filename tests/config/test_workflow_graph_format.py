"""
Comprehensive tests for workflow graph format support.

Tests cover:
- Workflow model parsing and validation
- Connection validation
- Position validation
- Cycle detection
- Orphan detection
- Backward compatibility
- Utility functions
"""

import pytest
from pydantic import ValidationError as PydanticValidationError

from qontinui.config.schema import (
    Action,
    Connections,
    Workflow,
    WorkflowFormat,
)
from qontinui.config.workflow_utils import (
    convert_sequential_to_graph,
    detect_workflow_format,
    find_entry_points,
    find_exit_points,
    get_action_by_id,
    get_action_connection_types,
    get_action_output_count,
    get_connected_actions,
    get_workflow_statistics,
    has_merge_nodes,
)
from qontinui.config.workflow_validation import (
    detect_cycles,
    detect_orphans,
    validate_positions,
    validate_workflow,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_sequential_workflow():
    """Simple sequential workflow with 3 actions."""
    return {
        "id": "wf-seq-1",
        "name": "Simple Sequential",
        "version": "1.0.0",
        "actions": [
            {
                "id": "action1",
                "type": "CLICK",
                "config": {"target": {"type": "image", "imageId": "img1"}},
            },
            {"id": "action2", "type": "TYPE", "config": {"text": "hello"}},
            {
                "id": "action3",
                "type": "FIND",
                "config": {"target": {"type": "image", "imageId": "img2"}},
            },
        ],
    }


@pytest.fixture
def simple_graph_workflow():
    """Simple graph workflow with 3 actions."""
    return {
        "id": "wf-graph-1",
        "name": "Simple Graph",
        "version": "1.0.0",
        "format": "graph",
        "actions": [
            {
                "id": "action1",
                "type": "CLICK",
                "config": {"target": {"type": "image", "imageId": "img1"}},
                "position": [100, 100],
            },
            {
                "id": "action2",
                "type": "TYPE",
                "config": {"text": "hello"},
                "position": [100, 200],
            },
            {
                "id": "action3",
                "type": "FIND",
                "config": {"target": {"type": "image", "imageId": "img2"}},
                "position": [100, 300],
            },
        ],
        "connections": {
            "action1": {"main": [[{"action": "action2", "type": "main", "index": 0}]]},
            "action2": {"main": [[{"action": "action3", "type": "main", "index": 0}]]},
        },
    }


@pytest.fixture
def branching_graph_workflow():
    """Graph workflow with IF branching."""
    return {
        "id": "wf-branch-1",
        "name": "Branching Graph",
        "version": "1.0.0",
        "format": "graph",
        "actions": [
            {
                "id": "if1",
                "type": "IF",
                "config": {
                    "condition": {"type": "variable", "variableName": "test"},
                    "thenActions": [],
                    "elseActions": [],
                },
                "position": [100, 100],
            },
            {
                "id": "action_true",
                "type": "CLICK",
                "config": {"target": {"type": "image", "imageId": "img1"}},
                "position": [200, 200],
            },
            {
                "id": "action_false",
                "type": "TYPE",
                "config": {"text": "false"},
                "position": [200, 300],
            },
            {
                "id": "merge",
                "type": "FIND",
                "config": {"target": {"type": "image", "imageId": "img2"}},
                "position": [300, 250],
            },
        ],
        "connections": {
            "if1": {
                "true": [[{"action": "action_true", "type": "true", "index": 0}]],
                "false": [[{"action": "action_false", "type": "false", "index": 0}]],
            },
            "action_true": {
                "main": [[{"action": "merge", "type": "main", "index": 0}]]
            },
            "action_false": {
                "main": [[{"action": "merge", "type": "main", "index": 0}]]
            },
        },
    }


@pytest.fixture
def workflow_with_cycle():
    """Graph workflow with a cycle."""
    return {
        "id": "wf-cycle-1",
        "name": "Workflow With Cycle",
        "version": "1.0.0",
        "format": "graph",
        "actions": [
            {
                "id": "action1",
                "type": "CLICK",
                "config": {"target": {"type": "image", "imageId": "img1"}},
                "position": [100, 100],
            },
            {
                "id": "action2",
                "type": "IF",
                "config": {
                    "condition": {"type": "variable", "variableName": "continue"},
                    "thenActions": [],
                    "elseActions": [],
                },
                "position": [100, 200],
            },
            {
                "id": "action3",
                "type": "TYPE",
                "config": {"text": "test"},
                "position": [100, 300],
            },
        ],
        "connections": {
            "action1": {"main": [[{"action": "action2", "type": "main", "index": 0}]]},
            "action2": {
                "true": [[{"action": "action3", "type": "true", "index": 0}]],
                "false": [[{"action": "action1", "type": "false", "index": 0}]],
            },
            "action3": {"main": [[{"action": "action1", "type": "main", "index": 0}]]},
        },
    }


# ============================================================================
# Schema Parsing Tests
# ============================================================================


def test_parse_sequential_workflow(simple_sequential_workflow):
    """Test parsing a simple sequential workflow."""
    workflow = Workflow.model_validate(simple_sequential_workflow)

    assert workflow.id == "wf-seq-1"
    assert workflow.name == "Simple Sequential"
    assert workflow.format == WorkflowFormat.SEQUENTIAL
    assert len(workflow.actions) == 3
    assert workflow.connections is None


def test_parse_graph_workflow(simple_graph_workflow):
    """Test parsing a simple graph workflow."""
    workflow = Workflow.model_validate(simple_graph_workflow)

    assert workflow.id == "wf-graph-1"
    assert workflow.name == "Simple Graph"
    assert workflow.format == WorkflowFormat.GRAPH
    assert len(workflow.actions) == 3
    assert workflow.connections is not None

    # Check positions
    for action in workflow.actions:
        assert action.position is not None
        assert len(action.position) == 2


def test_workflow_default_format():
    """Test that workflow defaults to sequential format."""
    workflow_dict = {"id": "wf-1", "name": "Test", "version": "1.0.0", "actions": []}
    workflow = Workflow.model_validate(workflow_dict)
    assert workflow.format == WorkflowFormat.SEQUENTIAL


def test_workflow_with_metadata():
    """Test parsing workflow with metadata."""
    workflow_dict = {
        "id": "wf-1",
        "name": "Test",
        "version": "1.0.0",
        "actions": [],
        "metadata": {
            "author": "Test Author",
            "description": "Test workflow",
            "created": "2025-10-16",
            "version": "1.0.0",
        },
    }
    workflow = Workflow.model_validate(workflow_dict)
    assert workflow.metadata is not None
    assert workflow.metadata.author == "Test Author"
    assert workflow.metadata.description == "Test workflow"


def test_workflow_with_variables():
    """Test parsing workflow with variables."""
    workflow_dict = {
        "id": "wf-1",
        "name": "Test",
        "version": "1.0.0",
        "actions": [],
        "variables": {
            "local": {"counter": 0},
            "process": {"processVar": "value"},
            "global": {"globalVar": True},
        },
    }
    workflow = Workflow.model_validate(workflow_dict)
    assert workflow.variables is not None
    assert workflow.variables.local == {"counter": 0}
    assert workflow.variables.process == {"processVar": "value"}
    assert workflow.variables.global_vars == {"globalVar": True}


def test_workflow_with_settings():
    """Test parsing workflow with settings."""
    workflow_dict = {
        "id": "wf-1",
        "name": "Test",
        "version": "1.0.0",
        "actions": [],
        "settings": {"timeout": 30000, "retryCount": 3, "continueOnError": True},
    }
    workflow = Workflow.model_validate(workflow_dict)
    assert workflow.settings is not None
    assert workflow.settings.timeout == 30000
    assert workflow.settings.retry_count == 3
    assert workflow.settings.continue_on_error is True


# ============================================================================
# Validation Tests
# ============================================================================


def test_graph_workflow_missing_connections():
    """Test that graph workflow without connections fails validation."""
    workflow_dict = {
        "id": "wf-1",
        "name": "Test",
        "version": "1.0.0",
        "format": "graph",
        "actions": [
            {
                "id": "action1",
                "type": "CLICK",
                "config": {"target": {"type": "image", "imageId": "img1"}},
                "position": [100, 100],
            }
        ],
        # Missing connections field
    }

    with pytest.raises(PydanticValidationError) as exc_info:
        Workflow.model_validate(workflow_dict)

    assert "connections" in str(exc_info.value).lower()


def test_graph_workflow_missing_positions():
    """Test that graph workflow with actions missing positions fails validation."""
    workflow_dict = {
        "id": "wf-1",
        "name": "Test",
        "version": "1.0.0",
        "format": "graph",
        "actions": [
            {
                "id": "action1",
                "type": "CLICK",
                "config": {"target": {"type": "image", "imageId": "img1"}},
                # Missing position
            }
        ],
        "connections": {},
    }

    with pytest.raises(PydanticValidationError) as exc_info:
        Workflow.model_validate(workflow_dict)

    assert "position" in str(exc_info.value).lower()


def test_validate_valid_workflow(simple_graph_workflow):
    """Test validation of a valid graph workflow."""
    workflow = Workflow.model_validate(simple_graph_workflow)
    result = validate_workflow(workflow)

    assert result.valid is True
    assert len(result.errors) == 0


def test_validate_invalid_connection_target(simple_graph_workflow):
    """Test validation catches invalid connection target."""
    # Add invalid connection
    simple_graph_workflow["connections"]["action1"]["main"][0].append(
        {"action": "nonexistent_action", "type": "main", "index": 0}
    )

    workflow = Workflow.model_validate(simple_graph_workflow)
    result = validate_workflow(workflow)

    assert result.valid is False
    assert len(result.errors) > 0
    assert any("nonexistent_action" in str(err.message) for err in result.errors)


def test_validate_self_connection(simple_graph_workflow):
    """Test validation warns about self-connections."""
    # Add self-connection
    simple_graph_workflow["connections"]["action1"]["error"] = [
        [{"action": "action1", "type": "error", "index": 0}]
    ]

    workflow = Workflow.model_validate(simple_graph_workflow)
    result = validate_workflow(workflow)

    assert len(result.warnings) > 0
    assert any("self" in str(warn.message).lower() for warn in result.warnings)


def test_validate_negative_position():
    """Test validation catches negative positions."""
    workflow_dict = {
        "id": "wf-1",
        "name": "Test",
        "version": "1.0.0",
        "format": "graph",
        "actions": [
            {
                "id": "action1",
                "type": "CLICK",
                "config": {"target": {"type": "image", "imageId": "img1"}},
                "position": [-100, 100],  # Negative position
            }
        ],
        "connections": {},
    }

    workflow = Workflow.model_validate(workflow_dict)
    result = validate_positions(workflow)

    assert result.valid is False
    assert len(result.errors) > 0


def test_validate_overlapping_positions():
    """Test validation warns about overlapping positions."""
    workflow_dict = {
        "id": "wf-1",
        "name": "Test",
        "version": "1.0.0",
        "format": "graph",
        "actions": [
            {
                "id": "action1",
                "type": "CLICK",
                "config": {"target": {"type": "image", "imageId": "img1"}},
                "position": [100, 100],
            },
            {
                "id": "action2",
                "type": "TYPE",
                "config": {"text": "test"},
                "position": [100, 100],  # Same position
            },
        ],
        "connections": {},
    }

    workflow = Workflow.model_validate(workflow_dict)
    result = validate_positions(workflow)

    assert len(result.warnings) > 0
    assert any("overlap" in str(warn.message).lower() for warn in result.warnings)


# ============================================================================
# Cycle Detection Tests
# ============================================================================


def test_detect_no_cycle(simple_graph_workflow):
    """Test that simple linear workflow has no cycles."""
    workflow = Workflow.model_validate(simple_graph_workflow)
    assert detect_cycles(workflow) is False


def test_detect_cycle(workflow_with_cycle):
    """Test detection of cycle in workflow."""
    workflow = Workflow.model_validate(workflow_with_cycle)
    assert detect_cycles(workflow) is True


def test_detect_cycle_validation(workflow_with_cycle):
    """Test that validation detects cycles."""
    workflow = Workflow.model_validate(workflow_with_cycle)
    result = validate_workflow(workflow)

    assert result.valid is False
    assert any("cycle" in str(err.message).lower() for err in result.errors)


# ============================================================================
# Orphan Detection Tests
# ============================================================================


def test_detect_no_orphans(simple_graph_workflow):
    """Test that connected workflow has no orphans."""
    workflow = Workflow.model_validate(simple_graph_workflow)
    orphans = detect_orphans(workflow)

    assert len(orphans) == 0


def test_detect_orphan_action():
    """Test detection of orphaned action."""
    workflow_dict = {
        "id": "wf-1",
        "name": "Test",
        "version": "1.0.0",
        "format": "graph",
        "actions": [
            {
                "id": "action1",
                "type": "CLICK",
                "config": {"target": {"type": "image", "imageId": "img1"}},
                "position": [100, 100],
            },
            {
                "id": "action2",
                "type": "TYPE",
                "config": {"text": "test"},
                "position": [200, 100],
            },
            {
                "id": "orphan",
                "type": "FIND",
                "config": {"target": {"type": "image", "imageId": "img2"}},
                "position": [300, 100],
            },
        ],
        "connections": {
            "action1": {"main": [[{"action": "action2", "type": "main", "index": 0}]]}
        },
    }

    workflow = Workflow.model_validate(workflow_dict)
    orphans = detect_orphans(workflow)

    assert len(orphans) == 1
    assert "orphan" in orphans


def test_orphan_warning_in_validation():
    """Test that validation warns about orphaned actions."""
    workflow_dict = {
        "id": "wf-1",
        "name": "Test",
        "version": "1.0.0",
        "format": "graph",
        "actions": [
            {
                "id": "action1",
                "type": "CLICK",
                "config": {"target": {"type": "image", "imageId": "img1"}},
                "position": [100, 100],
            },
            {
                "id": "orphan",
                "type": "TYPE",
                "config": {"text": "test"},
                "position": [200, 100],
            },
        ],
        "connections": {},
    }

    workflow = Workflow.model_validate(workflow_dict)
    result = validate_workflow(workflow)

    assert len(result.warnings) > 0
    assert any("orphan" in str(warn.message).lower() for warn in result.warnings)


# ============================================================================
# Utility Function Tests
# ============================================================================


def test_detect_workflow_format_sequential(simple_sequential_workflow):
    """Test auto-detection of sequential format."""
    detected = detect_workflow_format(simple_sequential_workflow)
    assert detected == WorkflowFormat.SEQUENTIAL


def test_detect_workflow_format_graph(simple_graph_workflow):
    """Test auto-detection of graph format."""
    detected = detect_workflow_format(simple_graph_workflow)
    assert detected == WorkflowFormat.GRAPH


def test_detect_workflow_format_explicit():
    """Test detection with explicit format field."""
    workflow_dict = {"format": "graph", "actions": []}
    detected = detect_workflow_format(workflow_dict)
    assert detected == WorkflowFormat.GRAPH


def test_get_action_output_count():
    """Test getting output count for different action types."""
    click_action = Action(
        id="a1", type="CLICK", config={"target": {"type": "image", "imageId": "img1"}}
    )
    assert get_action_output_count(click_action) == 1

    if_action = Action(
        id="a2",
        type="IF",
        config={
            "condition": {"type": "variable", "variableName": "test"},
            "thenActions": [],
            "elseActions": [],
        },
    )
    assert get_action_output_count(if_action) == 2

    switch_action = Action(
        id="a3",
        type="SWITCH",
        config={
            "expression": "value",
            "cases": [{"value": 1}, {"value": 2}, {"value": 3}],
        },
    )
    assert get_action_output_count(switch_action) == 4  # 3 cases + 1 default


def test_get_action_connection_types():
    """Test getting valid connection types for actions."""
    click_types = get_action_connection_types(
        Action(
            id="a1",
            type="CLICK",
            config={"target": {"type": "image", "imageId": "img1"}},
        )
    )
    assert "main" in click_types
    assert "error" in click_types

    if_types = get_action_connection_types(
        Action(
            id="a2",
            type="IF",
            config={
                "condition": {"type": "variable", "variableName": "test"},
                "thenActions": [],
                "elseActions": [],
            },
        )
    )
    assert "true" in if_types
    assert "false" in if_types
    assert "error" in if_types


def test_has_merge_nodes(branching_graph_workflow):
    """Test detection of merge nodes."""
    workflow = Workflow.model_validate(branching_graph_workflow)
    assert has_merge_nodes(workflow) is True


def test_no_merge_nodes(simple_graph_workflow):
    """Test workflow without merge nodes."""
    workflow = Workflow.model_validate(simple_graph_workflow)
    assert has_merge_nodes(workflow) is False


def test_find_entry_points_sequential(simple_sequential_workflow):
    """Test finding entry points in sequential workflow."""
    workflow = Workflow.model_validate(simple_sequential_workflow)
    entry_points = find_entry_points(workflow)

    assert len(entry_points) == 1
    assert entry_points[0] == "action1"


def test_find_entry_points_graph(simple_graph_workflow):
    """Test finding entry points in graph workflow."""
    workflow = Workflow.model_validate(simple_graph_workflow)
    entry_points = find_entry_points(workflow)

    assert len(entry_points) == 1
    assert entry_points[0] == "action1"


def test_find_exit_points_sequential(simple_sequential_workflow):
    """Test finding exit points in sequential workflow."""
    workflow = Workflow.model_validate(simple_sequential_workflow)
    exit_points = find_exit_points(workflow)

    assert len(exit_points) == 1
    assert exit_points[0] == "action3"


def test_find_exit_points_graph(simple_graph_workflow):
    """Test finding exit points in graph workflow."""
    workflow = Workflow.model_validate(simple_graph_workflow)
    exit_points = find_exit_points(workflow)

    assert len(exit_points) == 1
    assert exit_points[0] == "action3"


def test_get_workflow_statistics(simple_graph_workflow):
    """Test getting workflow statistics."""
    workflow = Workflow.model_validate(simple_graph_workflow)
    stats = get_workflow_statistics(workflow)

    assert stats["format"] == "graph"
    assert stats["total_actions"] == 3
    assert stats["entry_point_count"] == 1
    assert stats["exit_point_count"] == 1
    assert "connection_count" in stats
    assert "has_cycles" in stats
    assert stats["has_cycles"] is False


def test_get_action_by_id(simple_graph_workflow):
    """Test finding action by ID."""
    workflow = Workflow.model_validate(simple_graph_workflow)
    action = get_action_by_id(workflow, "action2")

    assert action is not None
    assert action.id == "action2"
    assert action.type == "TYPE"


def test_get_connected_actions(simple_graph_workflow):
    """Test getting connected actions."""
    workflow = Workflow.model_validate(simple_graph_workflow)
    connections = get_connected_actions(workflow, "action1")

    assert "main" in connections
    assert "action2" in connections["main"]


def test_convert_sequential_to_graph(simple_sequential_workflow):
    """Test converting sequential workflow to graph format."""
    seq_workflow = Workflow.model_validate(simple_sequential_workflow)
    graph_workflow = convert_sequential_to_graph(seq_workflow)

    assert graph_workflow.format == WorkflowFormat.GRAPH
    assert graph_workflow.connections is not None
    assert len(graph_workflow.actions) == len(seq_workflow.actions)

    # Check all actions have positions
    for action in graph_workflow.actions:
        assert action.position is not None

    # Validate the converted workflow
    result = validate_workflow(graph_workflow)
    assert result.valid is True


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


def test_old_workflow_without_format_field():
    """Test that old workflows without format field still work."""
    old_workflow = {
        "id": "old-wf",
        "name": "Old Workflow",
        "version": "1.0.0",
        "actions": [
            {
                "id": "action1",
                "type": "CLICK",
                "config": {"target": {"type": "image", "imageId": "img1"}},
            }
        ],
    }

    workflow = Workflow.model_validate(old_workflow)
    assert workflow.format == WorkflowFormat.SEQUENTIAL
    assert len(workflow.actions) == 1


def test_old_workflow_without_position():
    """Test that sequential workflows don't require positions."""
    workflow_dict = {
        "id": "wf-1",
        "name": "Test",
        "version": "1.0.0",
        "format": "sequential",
        "actions": [
            {
                "id": "action1",
                "type": "CLICK",
                "config": {"target": {"type": "image", "imageId": "img1"}},
                # No position field
            }
        ],
    }

    workflow = Workflow.model_validate(workflow_dict)
    assert workflow.format == WorkflowFormat.SEQUENTIAL
    assert workflow.actions[0].position is None


def test_action_position_optional():
    """Test that action position is optional."""
    action_dict = {
        "id": "action1",
        "type": "CLICK",
        "config": {"target": {"type": "image", "imageId": "img1"}},
    }

    action = Action.model_validate(action_dict)
    assert action.position is None


def test_action_with_position():
    """Test parsing action with position."""
    action_dict = {
        "id": "action1",
        "type": "CLICK",
        "config": {"target": {"type": "image", "imageId": "img1"}},
        "position": [100, 200],
    }

    action = Action.model_validate(action_dict)
    assert action.position == (100, 200)


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_empty_workflow():
    """Test workflow with no actions."""
    workflow_dict = {
        "id": "empty-wf",
        "name": "Empty",
        "version": "1.0.0",
        "actions": [],
    }

    workflow = Workflow.model_validate(workflow_dict)
    assert len(workflow.actions) == 0


def test_workflow_with_tags():
    """Test workflow with tags."""
    workflow_dict = {
        "id": "wf-1",
        "name": "Test",
        "version": "1.0.0",
        "actions": [],
        "tags": ["automation", "test", "demo"],
    }

    workflow = Workflow.model_validate(workflow_dict)
    assert workflow.tags == ["automation", "test", "demo"]


def test_complex_branching(branching_graph_workflow):
    """Test complex branching workflow validation."""
    workflow = Workflow.model_validate(branching_graph_workflow)
    result = validate_workflow(workflow)

    assert result.valid is True
    assert has_merge_nodes(workflow) is True


def test_connections_get_methods():
    """Test Connections helper methods."""
    connections_dict = {
        "action1": {
            "main": [[{"action": "action2", "type": "main", "index": 0}]],
            "error": [[{"action": "action3", "type": "error", "index": 0}]],
        }
    }

    connections = Connections(root=connections_dict)

    # Test get_connections
    main_conns = connections.get_connections("action1", "main")
    assert len(main_conns) == 1
    assert main_conns[0][0].action == "action2"

    # Test get_all_connections
    all_conns = connections.get_all_connections("action1")
    assert "main" in all_conns
    assert "error" in all_conns


def test_workflow_statistics_with_variables():
    """Test statistics for workflow with variables."""
    workflow_dict = {
        "id": "wf-1",
        "name": "Test",
        "version": "1.0.0",
        "actions": [],
        "variables": {
            "local": {"var1": 1, "var2": 2},
            "process": {"pvar": "value"},
            "global": {"gvar": True},
        },
    }

    workflow = Workflow.model_validate(workflow_dict)
    stats = get_workflow_statistics(workflow)

    assert "local" in stats["variable_scopes"]
    assert "process" in stats["variable_scopes"]
    assert "global" in stats["variable_scopes"]
    assert stats["local_variable_count"] == 2
    assert stats["process_variable_count"] == 1
    assert stats["global_variable_count"] == 1


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_workflow_lifecycle(simple_graph_workflow):
    """Test complete workflow parsing, validation, and analysis."""
    # Parse
    workflow = Workflow.model_validate(simple_graph_workflow)

    # Validate
    result = validate_workflow(workflow)
    assert result.valid is True

    # Analyze
    stats = get_workflow_statistics(workflow)
    assert stats["total_actions"] == 3

    # Find structure
    entry_points = find_entry_points(workflow)
    exit_points = find_exit_points(workflow)
    assert len(entry_points) == 1
    assert len(exit_points) == 1

    # Check connections
    connections = get_connected_actions(workflow, "action1")
    assert "main" in connections


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
