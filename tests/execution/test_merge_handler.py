"""
Comprehensive tests for merge node support.

This test suite covers:
- Simple merge (2 paths converge)
- Complex merge (3+ paths)
- Merge after IF (true/false branches reconverge)
- Nested merges
- Context merging (variable conflicts)
- Different merge strategies
- Timeout scenarios
- Error handling (one path fails)
"""

import threading
import time
from typing import Any

import pytest

from qontinui.config.schema import Action, Connection, Connections, Workflow
from qontinui.execution.merge_context import (
    MergeContext,
    VariableConflictResolution,
)
from qontinui.execution.merge_handler import MergeHandler
from qontinui.execution.merge_strategies import (
    MajorityStrategy,
    TimeoutStrategy,
    WaitAllStrategy,
    WaitAnyStrategy,
    WaitFirstStrategy,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_merge_workflow():
    """
    Simple workflow with 2 paths merging:

        action1 ──┐
                  ├─> merge_action -> action4
        action2 ──┘
    """
    actions = [
        Action(id="action1", type="CLICK", config={}, position=(0, 0)),
        Action(id="action2", type="CLICK", config={}, position=(0, 100)),
        Action(id="merge_action", type="CLICK", config={}, position=(200, 50)),
        Action(id="action4", type="CLICK", config={}, position=(400, 50)),
    ]

    connections = Connections(
        root={
            "action1": {"main": [[Connection(action="merge_action", type="main", index=0)]]},
            "action2": {"main": [[Connection(action="merge_action", type="main", index=1)]]},
            "merge_action": {"main": [[Connection(action="action4", type="main", index=0)]]},
        }
    )

    return Workflow(
        id="simple_merge",
        name="Simple Merge Test",
        version="1.0.0",
        format="graph",
        actions=actions,
        connections=connections,
    )


@pytest.fixture
def complex_merge_workflow():
    """
    Complex workflow with 3 paths merging:

        action1 ──┐
        action2 ──┼─> merge_action -> action5
        action3 ──┘
    """
    actions = [
        Action(id="action1", type="CLICK", config={}, position=(0, 0)),
        Action(id="action2", type="CLICK", config={}, position=(0, 50)),
        Action(id="action3", type="CLICK", config={}, position=(0, 100)),
        Action(id="merge_action", type="CLICK", config={}, position=(200, 50)),
        Action(id="action5", type="CLICK", config={}, position=(400, 50)),
    ]

    connections = Connections(
        root={
            "action1": {"main": [[Connection(action="merge_action", type="main", index=0)]]},
            "action2": {"main": [[Connection(action="merge_action", type="main", index=1)]]},
            "action3": {"main": [[Connection(action="merge_action", type="main", index=2)]]},
            "merge_action": {"main": [[Connection(action="action5", type="main", index=0)]]},
        }
    )

    return Workflow(
        id="complex_merge",
        name="Complex Merge Test",
        version="1.0.0",
        format="graph",
        actions=actions,
        connections=connections,
    )


@pytest.fixture
def if_merge_workflow():
    """
    IF statement with branches that reconverge:

        action1 -> if_action ──┬─(true)─> action2 ──┐
                               │                     ├─> merge_action
                               └─(false)> action3 ──┘
    """
    actions = [
        Action(id="action1", type="CLICK", config={}, position=(0, 50)),
        Action(
            id="if_action",
            type="IF",
            config={"condition": {"type": "expression", "expression": "True"}},
            position=(100, 50),
        ),
        Action(id="action2", type="CLICK", config={}, position=(200, 0)),
        Action(id="action3", type="CLICK", config={}, position=(200, 100)),
        Action(id="merge_action", type="CLICK", config={}, position=(300, 50)),
    ]

    connections = Connections(
        root={
            "action1": {"main": [[Connection(action="if_action", type="main", index=0)]]},
            "if_action": {
                "true": [[Connection(action="action2", type="true", index=0)]],
                "false": [[Connection(action="action3", type="false", index=0)]],
            },
            "action2": {"main": [[Connection(action="merge_action", type="main", index=0)]]},
            "action3": {"main": [[Connection(action="merge_action", type="main", index=1)]]},
        }
    )

    return Workflow(
        id="if_merge",
        name="IF Merge Test",
        version="1.0.0",
        format="graph",
        actions=actions,
        connections=connections,
    )


# ============================================================================
# MergeContext Tests
# ============================================================================


class TestMergeContext:
    """Test MergeContext class."""

    def test_basic_creation(self):
        """Test creating a merge context."""
        ctx = MergeContext(action_id="merge1", expected_inputs={"action1", "action2"})

        assert ctx.action_id == "merge1"
        assert ctx.expected_inputs == {"action1", "action2"}
        assert ctx.get_input_count() == 0
        assert not ctx.is_complete()

    def test_register_input(self):
        """Test registering inputs."""
        ctx = MergeContext(action_id="merge1", expected_inputs={"action1", "action2"})

        ctx.register_input("action1", {"var1": "value1"})
        assert ctx.has_input_from("action1")
        assert not ctx.has_input_from("action2")
        assert ctx.get_input_count() == 1
        assert not ctx.is_complete()

        ctx.register_input("action2", {"var2": "value2"})
        assert ctx.has_input_from("action2")
        assert ctx.get_input_count() == 2
        assert ctx.is_complete()

    def test_register_unexpected_input(self):
        """Test registering input from unexpected action."""
        ctx = MergeContext(action_id="merge1", expected_inputs={"action1", "action2"})

        with pytest.raises(ValueError, match="Unexpected input"):
            ctx.register_input("action3", {"var": "value"})

    def test_get_merged_context_last_wins(self):
        """Test merging contexts with LAST_WINS strategy."""
        ctx = MergeContext(
            action_id="merge1",
            expected_inputs={"action1", "action2"},
            conflict_resolution=VariableConflictResolution.LAST_WINS,
        )

        ctx.register_input("action1", {"var1": "from_action1", "shared": "value1"})
        ctx.register_input("action2", {"var2": "from_action2", "shared": "value2"})

        merged = ctx.get_merged_context()
        assert merged["var1"] == "from_action1"
        assert merged["var2"] == "from_action2"
        assert merged["shared"] == "value2"  # Last wins

    def test_get_merged_context_first_wins(self):
        """Test merging contexts with FIRST_WINS strategy."""
        ctx = MergeContext(
            action_id="merge1",
            expected_inputs={"action1", "action2"},
            conflict_resolution=VariableConflictResolution.FIRST_WINS,
        )

        ctx.register_input("action1", {"shared": "value1"})
        ctx.register_input("action2", {"shared": "value2"})

        merged = ctx.get_merged_context()
        assert merged["shared"] == "value1"  # First wins

    def test_get_merged_context_merge_lists(self):
        """Test merging contexts with MERGE_LISTS strategy."""
        ctx = MergeContext(
            action_id="merge1",
            expected_inputs={"action1", "action2"},
            conflict_resolution=VariableConflictResolution.MERGE_LISTS,
        )

        ctx.register_input("action1", {"shared": "value1"})
        ctx.register_input("action2", {"shared": "value2"})

        merged = ctx.get_merged_context()
        assert merged["shared"] == ["value1", "value2"]

    def test_get_merged_context_error_on_conflict(self):
        """Test ERROR_ON_CONFLICT strategy."""
        ctx = MergeContext(
            action_id="merge1",
            expected_inputs={"action1", "action2"},
            conflict_resolution=VariableConflictResolution.ERROR_ON_CONFLICT,
        )

        ctx.register_input("action1", {"shared": "value1"})
        ctx.register_input("action2", {"shared": "value2"})

        with pytest.raises(ValueError, match="Variable conflict"):
            ctx.get_merged_context()

    def test_get_input_contexts(self):
        """Test getting individual input contexts."""
        ctx = MergeContext(action_id="merge1", expected_inputs={"action1", "action2"})

        ctx.register_input("action1", {"var1": "value1"})
        ctx.register_input("action2", {"var2": "value2"})

        contexts = ctx.get_input_contexts()
        assert len(contexts) == 2
        assert contexts["action1"] == {"var1": "value1"}
        assert contexts["action2"] == {"var2": "value2"}

    def test_thread_safety(self):
        """Test thread-safe operations."""
        ctx = MergeContext(action_id="merge1", expected_inputs={"action1", "action2", "action3"})

        def register_input(action_id: str):
            time.sleep(0.01)  # Simulate work
            ctx.register_input(action_id, {f"var_{action_id}": f"value_{action_id}"})

        threads = [
            threading.Thread(target=register_input, args=(f"action{i}",)) for i in range(1, 4)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert ctx.is_complete()
        assert ctx.get_input_count() == 3


# ============================================================================
# MergeStrategy Tests
# ============================================================================


class TestMergeStrategies:
    """Test merge strategy implementations."""

    def test_wait_all_strategy(self):
        """Test WaitAllStrategy."""
        strategy = WaitAllStrategy()

        assert not strategy.should_execute(0, 3)
        assert not strategy.should_execute(1, 3)
        assert not strategy.should_execute(2, 3)
        assert strategy.should_execute(3, 3)
        assert strategy.should_execute(4, 3)  # More than expected is OK

    def test_wait_any_strategy(self):
        """Test WaitAnyStrategy."""
        strategy = WaitAnyStrategy(allow_multiple_executions=False)

        assert not strategy.should_execute(0, 3)
        assert strategy.should_execute(1, 3)

        # Mark as executed
        strategy.mark_executed()
        assert not strategy.should_execute(2, 3)  # Already executed

        # Reset and try again
        strategy.reset()
        assert strategy.should_execute(1, 3)

    def test_wait_first_strategy(self):
        """Test WaitFirstStrategy."""
        strategy = WaitFirstStrategy()

        assert not strategy.should_execute(0, 3)
        assert strategy.should_execute(1, 3, first_action_id="action1")

        # Mark as executed
        strategy.mark_executed()
        assert not strategy.should_execute(2, 3)  # Already executed
        assert strategy.get_first_action_id() == "action1"

    def test_timeout_strategy_all_inputs(self):
        """Test TimeoutStrategy when all inputs arrive before timeout."""
        strategy = TimeoutStrategy(timeout_seconds=1.0, minimum_inputs=1)

        # All inputs arrive quickly
        assert strategy.should_execute(3, 3)

    def test_timeout_strategy_timeout_expires(self):
        """Test TimeoutStrategy when timeout expires."""
        strategy = TimeoutStrategy(timeout_seconds=0.1, minimum_inputs=1, timeout_mode="absolute")

        # Not enough inputs yet
        assert not strategy.should_execute(0, 3)

        # Wait for timeout
        time.sleep(0.15)

        # Should execute with partial inputs
        assert strategy.should_execute(1, 3)

    def test_majority_strategy(self):
        """Test MajorityStrategy."""
        strategy = MajorityStrategy(threshold=0.5)

        assert not strategy.should_execute(0, 4)
        assert not strategy.should_execute(1, 4)  # 25% < 50%
        assert strategy.should_execute(2, 4)  # 50% >= 50%
        assert strategy.should_execute(3, 4)  # 75% > 50%
        assert strategy.should_execute(4, 4)  # 100% > 50%


# ============================================================================
# MergeHandler Tests
# ============================================================================


class TestMergeHandler:
    """Test MergeHandler class."""

    def test_detect_simple_merge(self, simple_merge_workflow):
        """Test detecting a simple merge node."""
        action_map = {action.id: action for action in simple_merge_workflow.actions}
        handler = MergeHandler(simple_merge_workflow.connections, action_map)

        # merge_action should be detected as merge point
        assert handler.is_merge_point("merge_action")
        assert not handler.is_merge_point("action1")
        assert not handler.is_merge_point("action2")
        assert not handler.is_merge_point("action4")

        merge_points = handler.get_all_merge_points()
        assert len(merge_points) == 1
        assert "merge_action" in merge_points

    def test_detect_complex_merge(self, complex_merge_workflow):
        """Test detecting merge with 3 inputs."""
        action_map = {action.id: action for action in complex_merge_workflow.actions}
        handler = MergeHandler(complex_merge_workflow.connections, action_map)

        assert handler.is_merge_point("merge_action")

        status = handler.get_merge_status("merge_action")
        assert status is not None
        assert status["total_paths"] == 3
        assert status["arrived_paths"] == 0

    def test_register_arrivals(self, simple_merge_workflow):
        """Test registering path arrivals."""
        action_map = {action.id: action for action in simple_merge_workflow.actions}
        handler = MergeHandler(simple_merge_workflow.connections, action_map)

        # Register first arrival
        is_ready = handler.register_arrival(
            "merge_action", "action1", {"success": True, "context": {"var1": "value1"}}
        )
        assert not is_ready  # Need both paths

        # Register second arrival
        is_ready = handler.register_arrival(
            "merge_action", "action2", {"success": True, "context": {"var2": "value2"}}
        )
        assert is_ready  # Both paths arrived

    def test_get_merged_context(self, simple_merge_workflow):
        """Test getting merged context."""
        action_map = {action.id: action for action in simple_merge_workflow.actions}
        handler = MergeHandler(simple_merge_workflow.connections, action_map)

        handler.register_arrival(
            "merge_action", "action1", {"success": True, "context": {"var1": "value1"}}
        )
        handler.register_arrival(
            "merge_action", "action2", {"success": True, "context": {"var2": "value2"}}
        )

        merged = handler.get_merged_context("merge_action")
        assert "var1" in merged or "var2" in merged  # At least one variable present

    def test_if_merge_detection(self, if_merge_workflow):
        """Test detecting merge after IF statement."""
        action_map = {action.id: action for action in if_merge_workflow.actions}
        handler = MergeHandler(if_merge_workflow.connections, action_map)

        assert handler.is_merge_point("merge_action")

        status = handler.get_merge_status("merge_action")
        assert status["total_paths"] == 2  # True and false branches

    def test_blocking_paths(self, complex_merge_workflow):
        """Test getting blocking paths."""
        action_map = {action.id: action for action in complex_merge_workflow.actions}
        handler = MergeHandler(complex_merge_workflow.connections, action_map)

        # Initially all paths are blocking
        blocking = handler.get_blocking_paths("merge_action")
        assert len(blocking) == 3
        assert set(blocking) == {"action1", "action2", "action3"}

        # Register one arrival
        handler.register_arrival("merge_action", "action1", {"success": True, "context": {}})

        blocking = handler.get_blocking_paths("merge_action")
        assert len(blocking) == 2
        assert set(blocking) == {"action2", "action3"}

    def test_reset_merge_point(self, simple_merge_workflow):
        """Test resetting a merge point."""
        action_map = {action.id: action for action in simple_merge_workflow.actions}
        handler = MergeHandler(simple_merge_workflow.connections, action_map)

        # Register arrivals
        handler.register_arrival("merge_action", "action1", {"success": True, "context": {}})
        handler.register_arrival("merge_action", "action2", {"success": True, "context": {}})

        status = handler.get_merge_status("merge_action")
        assert status["arrived_paths"] == 2
        assert status["is_ready"]

        # Reset
        handler.reset_merge_point("merge_action")

        status = handler.get_merge_status("merge_action")
        assert status["arrived_paths"] == 0
        assert not status["is_ready"]

    def test_merge_statistics(self, complex_merge_workflow):
        """Test getting merge statistics."""
        action_map = {action.id: action for action in complex_merge_workflow.actions}
        handler = MergeHandler(complex_merge_workflow.connections, action_map)

        stats = handler.get_merge_statistics()
        assert stats["total_merge_points"] == 1
        assert stats["ready"] == 0
        assert stats["waiting"] == 1

        # Register all arrivals
        handler.register_arrival("merge_action", "action1", {"success": True, "context": {}})
        handler.register_arrival("merge_action", "action2", {"success": True, "context": {}})
        handler.register_arrival("merge_action", "action3", {"success": True, "context": {}})

        stats = handler.get_merge_statistics()
        assert stats["ready"] == 1
        assert stats["waiting"] == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestMergeIntegration:
    """Integration tests for merge functionality."""

    def test_simple_parallel_merge(self, simple_merge_workflow):
        """Test simple parallel execution with merge."""
        action_map = {action.id: action for action in simple_merge_workflow.actions}
        handler = MergeHandler(simple_merge_workflow.connections, action_map)

        # Simulate parallel execution
        results = {}

        def execute_action(action_id: str, context: dict[str, Any]):
            time.sleep(0.01)  # Simulate work
            result = {"success": True, "context": {f"{action_id}_var": f"{action_id}_value"}}
            results[action_id] = result

            # Check if this completes a merge
            # Find downstream merge points
            if action_id in ["action1", "action2"]:
                handler.register_arrival("merge_action", action_id, result)

        # Execute action1 and action2 in parallel
        threads = [
            threading.Thread(target=execute_action, args=("action1", {})),
            threading.Thread(target=execute_action, args=("action2", {})),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Merge point should be ready
        status = handler.get_merge_status("merge_action")
        assert status["is_ready"]

        # Get merged context
        merged = handler.get_merged_context("merge_action")
        # Both variables should be present (or merged depending on strategy)
        assert len(merged) >= 1

    def test_nested_merges(self):
        """Test workflow with nested merge points."""
        # Create workflow with nested structure:
        # a1 ─┐     c1 ─┐
        #     ├─b1─┐    ├─d1
        # a2 ─┘     │    │
        #           ├─c2─┘
        # a3 ───b2─┘

        actions = [
            Action(id="a1", type="CLICK", config={}, position=(0, 0)),
            Action(id="a2", type="CLICK", config={}, position=(0, 50)),
            Action(id="a3", type="CLICK", config={}, position=(0, 100)),
            Action(id="b1", type="CLICK", config={}, position=(100, 25)),
            Action(id="b2", type="CLICK", config={}, position=(100, 100)),
            Action(id="c1", type="CLICK", config={}, position=(200, 0)),
            Action(id="c2", type="CLICK", config={}, position=(200, 75)),
            Action(id="d1", type="CLICK", config={}, position=(300, 50)),
        ]

        connections = Connections(
            root={
                "a1": {"main": [[Connection(action="b1", type="main", index=0)]]},
                "a2": {"main": [[Connection(action="b1", type="main", index=1)]]},
                "a3": {"main": [[Connection(action="b2", type="main", index=0)]]},
                "b1": {"main": [[Connection(action="c2", type="main", index=0)]]},
                "b2": {"main": [[Connection(action="c2", type="main", index=1)]]},
                "c1": {"main": [[Connection(action="d1", type="main", index=0)]]},
                "c2": {"main": [[Connection(action="d1", type="main", index=1)]]},
            }
        )

        workflow = Workflow(
            id="nested_merge",
            name="Nested Merge Test",
            version="1.0.0",
            format="graph",
            actions=actions,
            connections=connections,
        )

        action_map = {action.id: action for action in workflow.actions}
        handler = MergeHandler(connections, action_map)

        # Should detect two merge points
        merge_points = handler.get_all_merge_points()
        assert len(merge_points) == 2
        assert "b1" in merge_points
        assert "c2" in merge_points or "d1" in merge_points

    def test_error_handling_in_merge(self, simple_merge_workflow):
        """Test merge behavior when one path fails."""
        action_map = {action.id: action for action in simple_merge_workflow.actions}
        handler = MergeHandler(simple_merge_workflow.connections, action_map)

        # First path succeeds
        handler.register_arrival(
            "merge_action", "action1", {"success": True, "context": {"var1": "value1"}}
        )

        # Second path fails
        handler.register_arrival(
            "merge_action",
            "action2",
            {"success": False, "error": "Something went wrong", "context": {}},
        )

        # Merge should still be ready (both paths arrived)
        status = handler.get_merge_status("merge_action")
        assert status["is_ready"]

        # Can still get merged context
        merged = handler.get_merged_context("merge_action")
        assert isinstance(merged, dict)


# ============================================================================
# Performance Tests
# ============================================================================


class TestMergePerformance:
    """Performance tests for merge operations."""

    def test_large_merge_performance(self):
        """Test performance with many parallel paths."""
        num_paths = 100

        # Create workflow with many parallel paths merging
        actions = [
            Action(id=f"action{i}", type="CLICK", config={}, position=(0, i * 10))
            for i in range(num_paths)
        ]
        actions.append(Action(id="merge", type="CLICK", config={}, position=(200, num_paths * 5)))

        connections_dict = {}
        for i in range(num_paths):
            connections_dict[f"action{i}"] = {
                "main": [[Connection(action="merge", type="main", index=i)]]
            }

        connections = Connections(root=connections_dict)

        workflow = Workflow(
            id="large_merge",
            name="Large Merge Test",
            version="1.0.0",
            format="graph",
            actions=actions,
            connections=connections,
        )

        action_map = {action.id: action for action in workflow.actions}

        # Measure detection time
        start = time.time()
        handler = MergeHandler(connections, action_map)
        detection_time = time.time() - start

        assert detection_time < 1.0  # Should detect in under 1 second

        # Measure registration time
        start = time.time()
        for i in range(num_paths):
            handler.register_arrival(
                "merge", f"action{i}", {"success": True, "context": {f"var{i}": f"value{i}"}}
            )
        registration_time = time.time() - start

        assert registration_time < 1.0  # Should register all in under 1 second

        # Verify merge is ready
        status = handler.get_merge_status("merge")
        assert status["is_ready"]


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestMergeEdgeCases:
    """Test edge cases and error conditions."""

    def test_no_merge_points(self):
        """Test workflow with no merge points."""
        actions = [
            Action(id="a1", type="CLICK", config={}, position=(0, 0)),
            Action(id="a2", type="CLICK", config={}, position=(100, 0)),
            Action(id="a3", type="CLICK", config={}, position=(200, 0)),
        ]

        connections = Connections(
            root={
                "a1": {"main": [[Connection(action="a2", type="main", index=0)]]},
                "a2": {"main": [[Connection(action="a3", type="main", index=0)]]},
            }
        )

        action_map = {action.id: action for action in actions}
        handler = MergeHandler(connections, action_map)

        assert len(handler.get_all_merge_points()) == 0

        stats = handler.get_merge_statistics()
        assert stats["total_merge_points"] == 0

    def test_single_input_not_merge(self):
        """Test that action with single input is not considered a merge."""
        actions = [
            Action(id="a1", type="CLICK", config={}, position=(0, 0)),
            Action(id="a2", type="CLICK", config={}, position=(100, 0)),
        ]

        connections = Connections(
            root={
                "a1": {"main": [[Connection(action="a2", type="main", index=0)]]},
            }
        )

        action_map = {action.id: action for action in actions}
        handler = MergeHandler(connections, action_map)

        assert not handler.is_merge_point("a2")

    def test_validate_merge_points(self, simple_merge_workflow):
        """Test merge point validation."""
        action_map = {action.id: action for action in simple_merge_workflow.actions}
        handler = MergeHandler(simple_merge_workflow.connections, action_map)

        is_valid, warnings = handler.validate_merge_points()
        assert is_valid
        assert len(warnings) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
