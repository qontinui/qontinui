"""
Comprehensive tests for Connection Router.

Tests routing logic for all action types and edge cases.
"""

import pytest

from qontinui.config.schema import (
    Action,
    Connection,
    Connections,
    Workflow,
)
from qontinui.execution import ConnectionRouter, OutputResolver, RoutingContext

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def output_resolver():
    """Create OutputResolver instance."""
    return OutputResolver()


@pytest.fixture
def routing_context():
    """Create RoutingContext instance."""
    return RoutingContext()


@pytest.fixture
def simple_connections():
    """Create simple linear connections."""
    return Connections(
        root={
            "action1": {"main": [[Connection(action="action2", type="main", index=0)]]},
            "action2": {"main": [[Connection(action="action3", type="main", index=0)]]},
        }
    )


@pytest.fixture
def if_action():
    """Create an IF action."""
    return Action(
        id="if1",
        type="IF",
        config={
            "condition": {"type": "expression", "expression": "x > 10"},
            "thenActions": ["then1"],
            "elseActions": ["else1"],
        },
        position=(100, 100),
    )


@pytest.fixture
def if_connections():
    """Create IF action connections."""
    return Connections(
        root={
            "if1": {
                "true": [[Connection(action="then1", type="true", index=0)]],
                "false": [[Connection(action="else1", type="false", index=0)]],
            }
        }
    )


@pytest.fixture
def loop_action():
    """Create a LOOP action."""
    return Action(
        id="loop1",
        type="LOOP",
        config={"loopType": "FOR", "iterations": 5, "iteratorVariable": "i", "actions": ["inner1"]},
        position=(200, 200),
    )


@pytest.fixture
def loop_connections():
    """Create LOOP action connections."""
    return Connections(
        root={
            "loop1": {
                "loop": [[Connection(action="inner1", type="loop", index=0)]],
                "main": [[Connection(action="next1", type="main", index=0)]],
            }
        }
    )


@pytest.fixture
def switch_action():
    """Create a SWITCH action."""
    return Action(
        id="switch1",
        type="SWITCH",
        config={
            "expression": "status",
            "cases": [
                {"value": "ready", "actions": ["case0_action"]},
                {"value": "error", "actions": ["case1_action"]},
                {"value": "pending", "actions": ["case2_action"]},
            ],
            "defaultActions": ["default_action"],
        },
        position=(300, 300),
    )


@pytest.fixture
def switch_connections():
    """Create SWITCH action connections."""
    return Connections(
        root={
            "switch1": {
                "case_0": [[Connection(action="case0_action", type="case_0", index=0)]],
                "case_1": [[Connection(action="case1_action", type="case_1", index=0)]],
                "case_2": [[Connection(action="case2_action", type="case_2", index=0)]],
                "default": [[Connection(action="default_action", type="default", index=0)]],
            }
        }
    )


@pytest.fixture
def try_catch_action():
    """Create a TRY_CATCH action."""
    return Action(
        id="try1",
        type="TRY_CATCH",
        config={
            "tryActions": ["try_action"],
            "catchActions": ["catch_action"],
            "finallyActions": ["finally_action"],
        },
        position=(400, 400),
    )


@pytest.fixture
def try_catch_connections():
    """Create TRY_CATCH connections."""
    return Connections(
        root={
            "try1": {
                "main": [[Connection(action="success_action", type="main", index=0)]],
                "error": [[Connection(action="catch_action", type="error", index=0)]],
            }
        }
    )


# ============================================================================
# Output Resolver Tests
# ============================================================================


class TestOutputResolver:
    """Test OutputResolver functionality."""

    def test_resolve_standard_action_success(self, output_resolver):
        """Test resolving standard action with success."""
        action = Action(id="a1", type="CLICK", config={}, position=(0, 0))
        result = {"success": True}

        output_type = output_resolver.resolve(action, result)
        assert output_type == "main"

    def test_resolve_standard_action_error(self, output_resolver):
        """Test resolving standard action with error."""
        action = Action(id="a1", type="CLICK", config={}, position=(0, 0))
        result = {"success": False, "error": "Click failed"}

        output_type = output_resolver.resolve(action, result)
        assert output_type == "error"

    def test_resolve_if_true(self, output_resolver, if_action):
        """Test resolving IF action when condition is true."""
        result = {"success": True, "condition_result": True}

        output_type = output_resolver.resolve(if_action, result)
        assert output_type == "true"

    def test_resolve_if_false(self, output_resolver, if_action):
        """Test resolving IF action when condition is false."""
        result = {"success": True, "condition_result": False}

        output_type = output_resolver.resolve(if_action, result)
        assert output_type == "false"

    def test_resolve_if_missing_condition(self, output_resolver, if_action):
        """Test IF action fails without condition_result."""
        result = {"success": True}

        with pytest.raises(ValueError, match="condition_result"):
            output_resolver.resolve(if_action, result)

    def test_resolve_loop_continue(self, output_resolver, loop_action):
        """Test LOOP action continues looping."""
        result = {"success": True, "continue_loop": True, "iteration": 2}

        output_type = output_resolver.resolve(loop_action, result)
        assert output_type == "loop"

    def test_resolve_loop_exit(self, output_resolver, loop_action):
        """Test LOOP action exits."""
        result = {"success": True, "continue_loop": False, "iteration": 5}

        output_type = output_resolver.resolve(loop_action, result)
        assert output_type == "main"

    def test_resolve_loop_missing_flag(self, output_resolver, loop_action):
        """Test LOOP action fails without continue_loop."""
        result = {"success": True, "iteration": 3}

        with pytest.raises(ValueError, match="continue_loop"):
            output_resolver.resolve(loop_action, result)

    def test_resolve_switch_case_0(self, output_resolver, switch_action):
        """Test SWITCH matches case 0."""
        result = {"success": True, "case_index": 0}

        output_type = output_resolver.resolve(switch_action, result)
        assert output_type == "case_0"

    def test_resolve_switch_case_2(self, output_resolver, switch_action):
        """Test SWITCH matches case 2."""
        result = {"success": True, "case_index": 2}

        output_type = output_resolver.resolve(switch_action, result)
        assert output_type == "case_2"

    def test_resolve_switch_default(self, output_resolver, switch_action):
        """Test SWITCH uses default case."""
        result = {"success": True, "case_index": None}

        output_type = output_resolver.resolve(switch_action, result)
        assert output_type == "default"

    def test_resolve_switch_missing_index(self, output_resolver, switch_action):
        """Test SWITCH fails without case_index."""
        result = {"success": True}

        with pytest.raises(ValueError, match="case_index"):
            output_resolver.resolve(switch_action, result)

    def test_resolve_try_catch_success(self, output_resolver, try_catch_action):
        """Test TRY_CATCH on success."""
        result = {"success": True}

        output_type = output_resolver.resolve(try_catch_action, result)
        assert output_type == "main"

    def test_resolve_try_catch_error(self, output_resolver, try_catch_action):
        """Test TRY_CATCH on error."""
        result = {"success": False, "error": "Exception occurred"}

        output_type = output_resolver.resolve(try_catch_action, result)
        assert output_type == "error"

    def test_resolve_try_catch_exception_field(self, output_resolver, try_catch_action):
        """Test TRY_CATCH with exception field."""
        result = {"success": True, "exception": "ValueError"}

        output_type = output_resolver.resolve(try_catch_action, result)
        assert output_type == "error"

    def test_get_valid_outputs_if(self, output_resolver, if_action):
        """Test getting valid outputs for IF action."""
        outputs = output_resolver.get_valid_outputs(if_action)

        assert "true" in outputs
        assert "false" in outputs
        assert "error" in outputs

    def test_get_valid_outputs_loop(self, output_resolver, loop_action):
        """Test getting valid outputs for LOOP action."""
        outputs = output_resolver.get_valid_outputs(loop_action)

        assert "loop" in outputs
        assert "main" in outputs
        assert "error" in outputs

    def test_get_valid_outputs_switch(self, output_resolver, switch_action):
        """Test getting valid outputs for SWITCH action."""
        outputs = output_resolver.get_valid_outputs(switch_action)

        assert "case_0" in outputs
        assert "case_1" in outputs
        assert "case_2" in outputs
        assert "default" in outputs
        assert "error" in outputs

    def test_get_valid_outputs_try_catch(self, output_resolver, try_catch_action):
        """Test getting valid outputs for TRY_CATCH action."""
        outputs = output_resolver.get_valid_outputs(try_catch_action)

        assert "main" in outputs
        assert "error" in outputs

    def test_get_valid_outputs_standard(self, output_resolver):
        """Test getting valid outputs for standard action."""
        action = Action(id="a1", type="CLICK", config={}, position=(0, 0))
        outputs = output_resolver.get_valid_outputs(action)

        assert "main" in outputs
        assert "error" in outputs

    def test_validate_output_exists(self, output_resolver, if_action):
        """Test output type validation."""
        assert output_resolver.validate_output_exists(if_action, "true") is True
        assert output_resolver.validate_output_exists(if_action, "false") is True
        assert output_resolver.validate_output_exists(if_action, "invalid") is False

    def test_get_output_description(self, output_resolver, if_action):
        """Test output descriptions."""
        desc_true = output_resolver.get_output_description(if_action, "true")
        assert "true" in desc_true.lower()

        desc_error = output_resolver.get_output_description(if_action, "error")
        assert "error" in desc_error.lower()

    def test_validate_result_structure_if(self, output_resolver, if_action):
        """Test result structure validation for IF."""
        valid_result = {"success": True, "condition_result": True}
        is_valid, error = output_resolver.validate_result_structure(if_action, valid_result)
        assert is_valid is True
        assert error is None

        invalid_result = {"success": True}
        is_valid, error = output_resolver.validate_result_structure(if_action, invalid_result)
        assert is_valid is False
        assert "condition_result" in error


# ============================================================================
# Connection Router Tests
# ============================================================================


class TestConnectionRouter:
    """Test ConnectionRouter functionality."""

    def test_route_standard_action(self, simple_connections):
        """Test routing standard action."""
        router = ConnectionRouter()
        action = Action(id="action1", type="CLICK", config={}, position=(0, 0))
        result = {"success": True}

        next_actions = router.route(action, result, simple_connections)

        assert len(next_actions) == 1
        assert next_actions[0] == ("action2", "main", 0)

    def test_route_if_true_branch(self, if_action, if_connections):
        """Test routing IF action on true branch."""
        router = ConnectionRouter()
        result = {"success": True, "condition_result": True}

        next_actions = router.route(if_action, result, if_connections)

        assert len(next_actions) == 1
        assert next_actions[0][0] == "then1"
        assert next_actions[0][1] == "true"

    def test_route_if_false_branch(self, if_action, if_connections):
        """Test routing IF action on false branch."""
        router = ConnectionRouter()
        result = {"success": True, "condition_result": False}

        next_actions = router.route(if_action, result, if_connections)

        assert len(next_actions) == 1
        assert next_actions[0][0] == "else1"
        assert next_actions[0][1] == "false"

    def test_route_loop_continue(self, loop_action, loop_connections):
        """Test routing LOOP to continue looping."""
        router = ConnectionRouter()
        result = {"success": True, "continue_loop": True, "iteration": 2}

        next_actions = router.route(loop_action, result, loop_connections)

        assert len(next_actions) == 1
        assert next_actions[0][0] == "inner1"
        assert next_actions[0][1] == "loop"

    def test_route_loop_exit(self, loop_action, loop_connections):
        """Test routing LOOP to exit."""
        router = ConnectionRouter()
        result = {"success": True, "continue_loop": False, "iteration": 5}

        next_actions = router.route(loop_action, result, loop_connections)

        assert len(next_actions) == 1
        assert next_actions[0][0] == "next1"
        assert next_actions[0][1] == "main"

    def test_route_switch_case_match(self, switch_action, switch_connections):
        """Test routing SWITCH with case match."""
        router = ConnectionRouter()
        result = {"success": True, "case_index": 1}

        next_actions = router.route(switch_action, result, switch_connections)

        assert len(next_actions) == 1
        assert next_actions[0][0] == "case1_action"
        assert next_actions[0][1] == "case_1"

    def test_route_switch_default(self, switch_action, switch_connections):
        """Test routing SWITCH to default."""
        router = ConnectionRouter()
        result = {"success": True, "case_index": None}

        next_actions = router.route(switch_action, result, switch_connections)

        assert len(next_actions) == 1
        assert next_actions[0][0] == "default_action"
        assert next_actions[0][1] == "default"

    def test_route_try_catch_success(self, try_catch_action, try_catch_connections):
        """Test routing TRY_CATCH on success."""
        router = ConnectionRouter()
        result = {"success": True}

        next_actions = router.route(try_catch_action, result, try_catch_connections)

        assert len(next_actions) == 1
        assert next_actions[0][0] == "success_action"
        assert next_actions[0][1] == "main"

    def test_route_try_catch_error(self, try_catch_action, try_catch_connections):
        """Test routing TRY_CATCH on error."""
        router = ConnectionRouter()
        result = {"success": False, "error": "Exception"}

        next_actions = router.route(try_catch_action, result, try_catch_connections)

        assert len(next_actions) == 1
        assert next_actions[0][0] == "catch_action"
        assert next_actions[0][1] == "error"

    def test_route_with_context_tracking(self, if_action, if_connections):
        """Test routing records to context."""
        context = RoutingContext()
        router = ConnectionRouter(context=context)
        result = {"success": True, "condition_result": True}

        router.route(if_action, result, if_connections)

        assert len(context.records) == 1
        assert context.records[0].from_action == "if1"
        assert context.records[0].to_action == "then1"
        assert context.records[0].output_type == "true"

    def test_route_invalid_result_structure(self, if_action, if_connections):
        """Test routing fails with invalid result."""
        router = ConnectionRouter()
        result = {"success": True}  # Missing condition_result

        with pytest.raises(ValueError, match="Invalid execution result"):
            router.route(if_action, result, if_connections)

    def test_get_routing_options(self, if_action, if_connections):
        """Test getting routing options."""
        router = ConnectionRouter()

        options = router.get_routing_options(if_action, if_connections)

        assert "true" in options
        assert "false" in options
        assert options["true"] == ["then1"]
        assert options["false"] == ["else1"]

    def test_find_reachable_actions(self):
        """Test finding reachable actions."""
        connections = Connections(
            root={
                "a1": {"main": [[Connection(action="a2", type="main", index=0)]]},
                "a2": {"main": [[Connection(action="a3", type="main", index=0)]]},
                "a3": {"main": [[Connection(action="a4", type="main", index=0)]]},
            }
        )

        router = ConnectionRouter()
        reachable = router.find_reachable_actions("a1", connections)

        assert "a1" in reachable
        assert "a2" in reachable
        assert "a3" in reachable
        assert "a4" in reachable

    def test_find_unreachable_actions(self):
        """Test finding unreachable actions."""
        connections = Connections(
            root={
                "a1": {"main": [[Connection(action="a2", type="main", index=0)]]}
                # a3 is orphaned
            }
        )

        router = ConnectionRouter()
        unreachable = router.find_unreachable_actions(["a1"], ["a1", "a2", "a3"], connections)

        assert "a3" in unreachable
        assert "a1" not in unreachable
        assert "a2" not in unreachable


# ============================================================================
# Routing Context Tests
# ============================================================================


class TestRoutingContext:
    """Test RoutingContext functionality."""

    def test_record_route(self):
        """Test recording a route."""
        context = RoutingContext()

        context.record_route("a1", "a2", "main", 0, {"success": True})

        assert len(context.records) == 1
        assert context.records[0].from_action == "a1"
        assert context.records[0].to_action == "a2"
        assert context.records[0].output_type == "main"

    def test_visit_tracking(self):
        """Test action visit tracking."""
        context = RoutingContext()

        context.record_route("a1", "a2", "main")
        context.record_route("a2", "a3", "main")
        context.record_route("a3", "a2", "loop")  # Revisit a2

        assert context.was_action_visited("a1")
        assert context.was_action_visited("a2")
        assert context.was_action_visited("a3")
        assert context.get_visit_count("a2") == 2
        assert context.get_visit_count("a1") == 1

    def test_output_usage_tracking(self):
        """Test output usage tracking."""
        context = RoutingContext()

        context.record_route("if1", "then1", "true")
        context.record_route("if1", "else1", "false")
        context.record_route("if1", "then1", "true")  # Use true again

        usage = context.get_output_usage("if1")
        assert usage["true"] == 2
        assert usage["false"] == 1

    def test_get_execution_path(self):
        """Test getting execution path."""
        context = RoutingContext()

        context.enter_action("a1", "main")
        context.enter_action("a2", "main")
        context.enter_action("a3", "loop")

        path = context.get_execution_path()
        assert path == ["a1", "a2", "a3"]

    def test_detect_loops(self):
        """Test loop detection in path."""
        context = RoutingContext()

        context.enter_action("loop1", "main")
        context.enter_action("inner1", "loop")
        context.enter_action("loop1", "loop")  # Loop back
        context.enter_action("inner1", "loop")

        loops = context.detect_loops()
        assert len(loops) > 0

    def test_get_branch_decisions(self):
        """Test getting branch decisions."""
        context = RoutingContext()

        context.record_route("if1", "then1", "true")
        context.record_route("if2", "else1", "false")
        context.record_route("switch1", "case0", "case_0")

        decisions = context.get_branch_decisions()
        assert len(decisions) == 3
        assert ("if1", "true") in decisions
        assert ("if2", "false") in decisions
        assert ("switch1", "case_0") in decisions

    def test_get_error_routes(self):
        """Test getting error routes."""
        context = RoutingContext()

        context.record_route("a1", "a2", "main")
        context.record_route("a2", "error_handler", "error")
        context.record_route("a3", "error_handler2", "error")

        error_routes = context.get_error_routes()
        assert len(error_routes) == 2

    def test_get_statistics(self):
        """Test getting execution statistics."""
        context = RoutingContext()

        context.record_route("a1", "a2", "main")
        context.record_route("a2", "a3", "main")
        context.record_route("if1", "then1", "true")

        stats = context.get_statistics()
        assert stats["total_routes"] == 3
        assert stats["unique_actions_visited"] >= 4

    def test_visual_path(self):
        """Test visual path representation."""
        context = RoutingContext()

        context.enter_action("a1", "main")
        context.enter_action("a2", "main")
        context.enter_action("a3", "true")

        visual = context.get_visual_path()
        assert "a1" in visual
        assert "a2" in visual
        assert "a3" in visual

    def test_to_dict(self):
        """Test serialization to dict."""
        context = RoutingContext()

        context.record_route("a1", "a2", "main", 0, {"success": True})

        data = context.to_dict()
        assert "records" in data
        assert "path" in data
        assert "statistics" in data
        assert len(data["records"]) == 1


# ============================================================================
# Integration Tests
# ============================================================================


class TestConnectionRouterIntegration:
    """Integration tests for complete routing scenarios."""

    def test_complex_workflow_routing(self):
        """Test routing through complex workflow."""
        # Create workflow with IF -> LOOP -> SWITCH
        workflow = Workflow(
            id="test_workflow",
            name="Complex Routing Test",
            version="1.0.0",
            format="graph",
            actions=[
                Action(id="start", type="CLICK", config={}, position=(0, 0)),
                Action(
                    id="if1",
                    type="IF",
                    config={
                        "condition": {"type": "expression", "expression": "x > 0"},
                        "thenActions": ["loop1"],
                        "elseActions": ["end"],
                    },
                    position=(100, 0),
                ),
                Action(
                    id="loop1",
                    type="LOOP",
                    config={"loopType": "FOR", "iterations": 3, "actions": ["inner"]},
                    position=(200, 0),
                ),
                Action(
                    id="end",
                    type="WAIT",
                    config={"waitFor": "time", "duration": 100},
                    position=(300, 0),
                ),
            ],
            connections=Connections(
                root={
                    "start": {"main": [[Connection(action="if1", type="main", index=0)]]},
                    "if1": {
                        "true": [[Connection(action="loop1", type="true", index=0)]],
                        "false": [[Connection(action="end", type="false", index=0)]],
                    },
                    "loop1": {"main": [[Connection(action="end", type="main", index=0)]]},
                }
            ),
        )

        context = RoutingContext()
        router = ConnectionRouter(workflow=workflow, context=context)

        # Route through start
        start_action = workflow.actions[0]
        next_actions = router.route(start_action, {"success": True}, workflow.connections)
        assert next_actions[0][0] == "if1"

        # Route through IF (true branch)
        if_action = workflow.actions[1]
        next_actions = router.route(
            if_action, {"success": True, "condition_result": True}, workflow.connections
        )
        assert next_actions[0][0] == "loop1"

        # Check context recorded the path
        assert len(context.records) == 2
        assert context.was_action_visited("start")
        assert context.was_action_visited("if1")

    def test_validate_workflow_routing(self):
        """Test workflow routing validation."""
        workflow = Workflow(
            id="test",
            name="Validation Test",
            version="1.0.0",
            format="graph",
            actions=[
                Action(
                    id="if1",
                    type="IF",
                    config={
                        "condition": {"type": "expression", "expression": "true"},
                        "thenActions": ["a1"],
                        "elseActions": ["a2"],
                    },
                    position=(0, 0),
                )
            ],
            connections=Connections(
                root={
                    "if1": {
                        "true": [[Connection(action="a1", type="true", index=0)]],
                        "false": [[Connection(action="a2", type="false", index=0)]],
                    }
                }
            ),
        )

        router = ConnectionRouter()
        is_valid, errors = router.validate_routing(workflow)

        # Should be valid (all outputs connected)
        assert is_valid is True
        assert len(errors) == 0

    def test_missing_connections_error(self):
        """Test validation catches missing connections."""
        workflow = Workflow(
            id="test",
            name="Missing Connections",
            version="1.0.0",
            format="graph",
            actions=[
                Action(
                    id="if1",
                    type="IF",
                    config={
                        "condition": {"type": "expression", "expression": "true"},
                        "thenActions": ["a1"],
                        "elseActions": [],
                    },
                    position=(0, 0),
                )
            ],
            connections=Connections(
                root={
                    "if1": {
                        "true": [[Connection(action="a1", type="true", index=0)]]
                        # Missing false connection
                    }
                }
            ),
        )

        router = ConnectionRouter()
        is_valid, errors = router.validate_routing(workflow)

        assert is_valid is False
        assert len(errors) > 0
        assert "false" in errors[0].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
