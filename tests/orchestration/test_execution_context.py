"""Tests for execution context."""

import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import directly from module to avoid cv2 dependency
from qontinui.orchestration.execution_context import (
    ActionState,
    ExecutionContext,
    ExecutionStatistics,
)


class TestExecutionStatistics:
    """Test ExecutionStatistics class."""

    def test_initial_statistics(self):
        """Test initial statistics values."""
        stats = ExecutionStatistics()

        assert stats.total_actions == 0
        assert stats.successful_actions == 0
        assert stats.failed_actions == 0
        assert stats.retried_actions == 0
        assert stats.total_retries == 0
        assert stats.start_time is None
        assert stats.end_time is None

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        stats = ExecutionStatistics()
        stats.total_actions = 10
        stats.successful_actions = 7
        stats.failed_actions = 3

        assert stats.success_rate == 70.0

    def test_success_rate_zero_actions(self):
        """Test success rate with zero actions."""
        stats = ExecutionStatistics()

        assert stats.success_rate == 0.0

    def test_success_rate_perfect(self):
        """Test success rate with all successful."""
        stats = ExecutionStatistics()
        stats.total_actions = 5
        stats.successful_actions = 5

        assert stats.success_rate == 100.0

    def test_duration_calculation(self):
        """Test duration calculation."""
        stats = ExecutionStatistics()
        stats.start_time = datetime(2025, 1, 1, 12, 0, 0)
        stats.end_time = datetime(2025, 1, 1, 12, 0, 30)

        assert stats.duration_seconds == 30.0

    def test_duration_not_completed(self):
        """Test duration when not completed."""
        stats = ExecutionStatistics()
        stats.start_time = datetime.now()

        assert stats.duration_seconds == 0.0

    def test_string_representation(self):
        """Test string representation."""
        stats = ExecutionStatistics()
        stats.total_actions = 10
        stats.successful_actions = 8
        stats.failed_actions = 2
        stats.retried_actions = 3
        stats.start_time = datetime(2025, 1, 1, 12, 0, 0)
        stats.end_time = datetime(2025, 1, 1, 12, 1, 0)

        str_repr = str(stats)
        assert "total=10" in str_repr
        assert "successful=8" in str_repr
        assert "failed=2" in str_repr
        assert "retried=3" in str_repr
        assert "success_rate=80.0%" in str_repr
        assert "duration=60.00s" in str_repr


class TestActionState:
    """Test ActionState class."""

    def test_initial_action_state(self):
        """Test initial action state."""
        state = ActionState(action_index=0, action_name="test_action")

        assert state.action_index == 0
        assert state.action_name == "test_action"
        assert state.attempt_count == 0
        assert not state.success
        assert state.error is None
        assert state.start_time is None
        assert state.end_time is None

    def test_duration_calculation(self):
        """Test duration calculation."""
        state = ActionState(action_index=0, action_name="test")
        state.start_time = time.time()
        time.sleep(0.1)
        state.end_time = time.time()

        assert 0.08 < state.duration < 0.15

    def test_duration_not_completed(self):
        """Test duration when not completed."""
        state = ActionState(action_index=0, action_name="test")
        state.start_time = time.time()

        assert state.duration == 0.0

    def test_action_state_with_error(self):
        """Test action state with error."""
        error = ValueError("test error")
        state = ActionState(action_index=1, action_name="failing_action")
        state.error = error
        state.success = False

        assert state.error is error
        assert not state.success


class TestExecutionContext:
    """Test ExecutionContext class."""

    def test_initial_context(self):
        """Test initial context state."""
        context = ExecutionContext()

        assert len(context.variables) == 0
        assert len(context.action_states) == 0
        assert context.statistics.total_actions == 0

    def test_initial_context_with_variables(self):
        """Test context with initial variables."""
        initial_vars = {"var1": "value1", "var2": 42}
        context = ExecutionContext(initial_variables=initial_vars)

        assert context.get_variable("var1") == "value1"
        assert context.get_variable("var2") == 42

    def test_set_and_get_variable(self):
        """Test setting and getting variables."""
        context = ExecutionContext()

        context.set_variable("name", "test")
        assert context.get_variable("name") == "test"

        context.set_variable("count", 42)
        assert context.get_variable("count") == 42

    def test_get_variable_default(self):
        """Test getting variable with default."""
        context = ExecutionContext()

        assert context.get_variable("missing") is None
        assert context.get_variable("missing", "default") == "default"

    def test_has_variable(self):
        """Test checking variable existence."""
        context = ExecutionContext()

        assert not context.has_variable("test")

        context.set_variable("test", "value")
        assert context.has_variable("test")

    def test_delete_variable(self):
        """Test deleting variables."""
        context = ExecutionContext()
        context.set_variable("temp", "value")

        assert context.has_variable("temp")

        context.delete_variable("temp")
        assert not context.has_variable("temp")

    def test_delete_nonexistent_variable(self):
        """Test deleting nonexistent variable doesn't error."""
        context = ExecutionContext()
        context.delete_variable("nonexistent")  # Should not raise

    def test_clear_variables(self):
        """Test clearing all variables."""
        context = ExecutionContext()
        context.set_variable("var1", "value1")
        context.set_variable("var2", "value2")

        assert len(context.variables) == 2

        context.clear_variables()
        assert len(context.variables) == 0

    def test_substitute_variables_simple(self):
        """Test simple variable substitution."""
        context = ExecutionContext()
        context.set_variable("name", "Alice")
        context.set_variable("age", 30)

        result = context.substitute_variables("Hello ${name}, you are ${age}")
        assert result == "Hello Alice, you are 30"

    def test_substitute_variables_missing(self):
        """Test substitution with missing variables."""
        context = ExecutionContext()
        context.set_variable("name", "Bob")

        result = context.substitute_variables("Hello ${name}, age ${age}")
        # Missing variables should remain as placeholders
        assert result == "Hello Bob, age ${age}"

    def test_substitute_variables_empty_string(self):
        """Test substitution with empty string."""
        context = ExecutionContext()

        result = context.substitute_variables("")
        assert result == ""

    def test_substitute_variables_no_placeholders(self):
        """Test substitution with no placeholders."""
        context = ExecutionContext()
        context.set_variable("var", "value")

        result = context.substitute_variables("No variables here")
        assert result == "No variables here"

    def test_substitute_variables_multiple_occurrences(self):
        """Test substitution with multiple occurrences."""
        context = ExecutionContext()
        context.set_variable("x", "test")

        result = context.substitute_variables("${x} and ${x} again")
        assert result == "test and test again"

    def test_start_action(self):
        """Test starting an action."""
        context = ExecutionContext()

        state = context.start_action(0, "test_action")

        assert state.action_index == 0
        assert state.action_name == "test_action"
        assert state.start_time is not None
        assert len(context.action_states) == 1
        assert context.statistics.total_actions == 1

    def test_complete_action_success(self):
        """Test completing action successfully."""
        context = ExecutionContext()
        state = context.start_action(0, "test_action")

        context.complete_action(state, success=True)

        assert state.success
        assert state.end_time is not None
        assert context.statistics.successful_actions == 1
        assert context.statistics.failed_actions == 0

    def test_complete_action_failure(self):
        """Test completing action with failure."""
        context = ExecutionContext()
        state = context.start_action(0, "test_action")
        error = ValueError("test error")

        context.complete_action(state, success=False, error=error)

        assert not state.success
        assert state.error is error
        assert context.statistics.successful_actions == 0
        assert context.statistics.failed_actions == 1

    def test_record_retry(self):
        """Test recording retries."""
        context = ExecutionContext()
        state = context.start_action(0, "test_action")

        context.record_retry(state)
        assert state.attempt_count == 1
        assert context.statistics.total_retries == 1
        assert context.statistics.retried_actions == 1

        context.record_retry(state)
        assert state.attempt_count == 2
        assert context.statistics.total_retries == 2
        # Should still be 1 retried action (not incremented again)
        assert context.statistics.retried_actions == 1

    def test_start_and_complete_workflow(self):
        """Test workflow lifecycle."""
        context = ExecutionContext()

        assert context.statistics.start_time is None
        assert context.statistics.end_time is None

        context.start_workflow()
        assert context.statistics.start_time is not None

        time.sleep(0.01)

        context.complete_workflow()
        assert context.statistics.end_time is not None
        assert context.statistics.duration_seconds > 0

    def test_metadata_operations(self):
        """Test metadata get/set."""
        context = ExecutionContext()

        context.set_metadata("key1", "value1")
        assert context.get_metadata("key1") == "value1"

        context.set_metadata("key2", 123)
        assert context.get_metadata("key2") == 123

        assert context.get_metadata("missing") is None
        assert context.get_metadata("missing", "default") == "default"

    def test_get_last_action_state(self):
        """Test getting last action state."""
        context = ExecutionContext()

        assert context.get_last_action_state() is None

        state1 = context.start_action(0, "action1")
        assert context.get_last_action_state() is state1

        state2 = context.start_action(1, "action2")
        assert context.get_last_action_state() is state2

    def test_get_failed_actions(self):
        """Test getting failed actions."""
        context = ExecutionContext()

        state1 = context.start_action(0, "action1")
        context.complete_action(state1, success=True)

        state2 = context.start_action(1, "action2")
        context.complete_action(state2, success=False)

        state3 = context.start_action(2, "action3")
        context.complete_action(state3, success=False)

        failed = context.get_failed_actions()
        assert len(failed) == 2
        assert state2 in failed
        assert state3 in failed
        assert state1 not in failed

    def test_variables_property_returns_copy(self):
        """Test that variables property returns a copy."""
        context = ExecutionContext()
        context.set_variable("test", "value")

        vars1 = context.variables
        vars1["test"] = "modified"

        # Original should be unchanged
        assert context.get_variable("test") == "value"

    def test_action_states_property_returns_copy(self):
        """Test that action_states property returns a copy."""
        context = ExecutionContext()
        state = context.start_action(0, "test")

        states = context.action_states
        states.clear()

        # Original should be unchanged
        assert len(context.action_states) == 1

    def test_string_representation(self):
        """Test string representation."""
        context = ExecutionContext()
        context.set_variable("var1", "value")
        context.start_action(0, "action1")

        str_repr = str(context)
        assert "ExecutionContext" in str_repr
        assert "variables=1" in str_repr


class TestExecutionContextIntegration:
    """Integration tests for ExecutionContext."""

    def test_complete_workflow_simulation(self):
        """Test a complete workflow simulation."""
        context = ExecutionContext(initial_variables={"user": "test_user"})
        context.start_workflow()

        # Action 1: Success
        state1 = context.start_action(0, "login")
        context.complete_action(state1, success=True)

        # Action 2: Fail with retry
        state2 = context.start_action(1, "fetch_data")
        context.record_retry(state2)
        context.record_retry(state2)
        context.complete_action(state2, success=True)

        # Action 3: Fail
        state3 = context.start_action(2, "process")
        error = RuntimeError("processing failed")
        context.complete_action(state3, success=False, error=error)

        context.complete_workflow()

        # Verify statistics
        assert context.statistics.total_actions == 3
        assert context.statistics.successful_actions == 2
        assert context.statistics.failed_actions == 1
        assert context.statistics.retried_actions == 1
        assert context.statistics.total_retries == 2
        assert context.statistics.success_rate == pytest.approx(66.67, rel=0.1)
        assert context.statistics.duration_seconds > 0

        # Verify failed actions
        failed = context.get_failed_actions()
        assert len(failed) == 1
        assert failed[0].action_name == "process"
        assert failed[0].error is error
