"""Tests for workflow orchestrator."""

import sys
from pathlib import Path
from typing import Any

import pytest

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import directly from module to avoid cv2 dependency
from qontinui.orchestration.execution_context import ExecutionContext
from qontinui.orchestration.retry_policy import RetryPolicy
from qontinui.orchestration.workflow_orchestrator import (
    WorkflowOrchestrator,
    WorkflowResult,
)


class MockAction:
    """Mock action for testing."""

    def __init__(self, name: str):
        """Initialize mock action.

        Args:
            name: Action name
        """
        self.name = name


class MockActionExecutor:
    """Mock action executor for testing."""

    def __init__(self, success_sequence: list[bool] | None = None):
        """Initialize mock executor.

        Args:
            success_sequence: Sequence of success/failure results
        """
        self.success_sequence = success_sequence or []
        self.call_index = 0
        self.executed_actions = []

    def execute(self, action: Any, target: Any | None = None) -> bool:
        """Execute action with predetermined result.

        Args:
            action: Action to execute
            target: Optional target

        Returns:
            Success result from sequence
        """
        self.executed_actions.append(action)

        if self.call_index < len(self.success_sequence):
            result = self.success_sequence[self.call_index]
            self.call_index += 1
            return result

        return True


class MockEventEmitter:
    """Mock event emitter for testing."""

    def __init__(self):
        """Initialize mock emitter."""
        self.events = []

    def emit(self, event_type: str, **kwargs: Any) -> None:
        """Record emitted event.

        Args:
            event_type: Type of event
            kwargs: Event data
        """
        self.events.append({"type": event_type, "data": kwargs})


class TestWorkflowOrchestrator:
    """Test WorkflowOrchestrator class."""

    def test_execute_workflow_all_success(self):
        """Test executing workflow where all actions succeed."""
        executor = MockActionExecutor(success_sequence=[True, True, True])
        orchestrator = WorkflowOrchestrator(action_executor=executor)

        actions = [MockAction("action1"), MockAction("action2"), MockAction("action3")]
        result = orchestrator.execute_workflow(actions)

        assert result.success
        assert result.error is None
        assert result.completed_actions == 3
        assert result.failed_action_index is None
        assert len(executor.executed_actions) == 3

    def test_execute_workflow_with_failure_stop(self):
        """Test workflow stops on first failure."""
        executor = MockActionExecutor(success_sequence=[True, False, True])
        orchestrator = WorkflowOrchestrator(action_executor=executor)

        actions = [MockAction("action1"), MockAction("action2"), MockAction("action3")]
        result = orchestrator.execute_workflow(actions)

        assert not result.success
        assert result.completed_actions == 1
        assert result.failed_action_index == 1
        # Should only execute 2 actions (1 success, 1 failure)
        assert len(executor.executed_actions) == 2

    def test_execute_workflow_with_continue_on_error(self):
        """Test workflow continues after failure with continue_on_error."""
        executor = MockActionExecutor(success_sequence=[True, False, True])
        policy = RetryPolicy(max_retries=0, continue_on_error=True)
        orchestrator = WorkflowOrchestrator(action_executor=executor, retry_policy=policy)

        actions = [MockAction("action1"), MockAction("action2"), MockAction("action3")]
        result = orchestrator.execute_workflow(actions)

        assert result.success
        assert result.completed_actions == 3
        # All actions should be executed despite failure
        assert len(executor.executed_actions) == 3

    def test_execute_workflow_with_retry(self):
        """Test workflow with retry on failure."""
        # First call fails, second succeeds
        executor = MockActionExecutor(success_sequence=[False, True])
        policy = RetryPolicy(max_retries=1, base_delay=0.01)
        orchestrator = WorkflowOrchestrator(action_executor=executor, retry_policy=policy)

        actions = [MockAction("action1")]
        result = orchestrator.execute_workflow(actions, retry_policy=policy)

        assert result.success
        # Should have executed twice (initial + 1 retry)
        assert len(executor.executed_actions) == 2
        assert result.context.statistics.total_retries == 1

    def test_execute_workflow_with_failed_retries(self):
        """Test workflow when all retries fail."""
        executor = MockActionExecutor(success_sequence=[False, False, False])
        policy = RetryPolicy(max_retries=2, base_delay=0.01)
        orchestrator = WorkflowOrchestrator(action_executor=executor, retry_policy=policy)

        actions = [MockAction("action1")]
        result = orchestrator.execute_workflow(actions, retry_policy=policy)

        assert not result.success
        # Should execute 3 times (initial + 2 retries)
        assert len(executor.executed_actions) == 3
        assert result.context.statistics.total_retries == 2

    def test_execute_workflow_with_event_emitter(self):
        """Test that events are emitted during execution."""
        executor = MockActionExecutor(success_sequence=[True, True])
        emitter = MockEventEmitter()
        orchestrator = WorkflowOrchestrator(action_executor=executor, event_emitter=emitter)

        actions = [MockAction("action1"), MockAction("action2")]
        result = orchestrator.execute_workflow(actions)

        assert result.success
        # Should have workflow_started, 2x action_started, 2x action_completed, workflow_completed
        assert len(emitter.events) >= 5
        event_types = [e["type"] for e in emitter.events]
        assert "workflow_started" in event_types
        assert "action_started" in event_types
        assert "action_completed" in event_types
        assert "workflow_completed" in event_types

    def test_execute_workflow_with_context(self):
        """Test executing workflow with provided context."""
        executor = MockActionExecutor(success_sequence=[True])
        orchestrator = WorkflowOrchestrator(action_executor=executor)

        context = ExecutionContext(initial_variables={"test": "value"})
        actions = [MockAction("action1")]
        result = orchestrator.execute_workflow(actions, context=context)

        assert result.success
        assert result.context is context
        assert result.context.get_variable("test") == "value"

    def test_execute_workflow_statistics(self):
        """Test that execution statistics are tracked."""
        executor = MockActionExecutor(success_sequence=[True, False, True])
        policy = RetryPolicy(max_retries=0, continue_on_error=True)
        orchestrator = WorkflowOrchestrator(action_executor=executor, retry_policy=policy)

        actions = [MockAction("a1"), MockAction("a2"), MockAction("a3")]
        result = orchestrator.execute_workflow(actions)

        stats = result.context.statistics
        assert stats.total_actions == 3
        assert stats.successful_actions == 2
        assert stats.failed_actions == 1
        assert stats.success_rate == pytest.approx(66.67, rel=0.1)
        assert stats.duration_seconds > 0

    def test_execute_workflow_empty_actions(self):
        """Test executing workflow with no actions."""
        executor = MockActionExecutor()
        orchestrator = WorkflowOrchestrator(action_executor=executor)

        result = orchestrator.execute_workflow([])

        assert result.success
        assert result.completed_actions == 0
        assert len(executor.executed_actions) == 0

    def test_execute_workflow_with_exception(self):
        """Test handling of unexpected exceptions."""

        class FailingExecutor:
            def execute(self, action: Any, target: Any | None = None) -> bool:
                raise RuntimeError("Unexpected error")

        executor = FailingExecutor()
        orchestrator = WorkflowOrchestrator(action_executor=executor)

        actions = [MockAction("action1")]
        result = orchestrator.execute_workflow(actions)

        assert not result.success
        assert isinstance(result.error, RuntimeError)
        assert result.context.statistics.failed_actions == 1

    def test_execute_with_condition_true(self):
        """Test conditional execution when condition is met."""
        executor = MockActionExecutor(success_sequence=[True])
        orchestrator = WorkflowOrchestrator(action_executor=executor)

        actions = [MockAction("action1")]
        result = orchestrator.execute_with_condition(
            actions, condition=lambda ctx: True  # Always execute
        )

        assert result.success
        assert result.completed_actions == 1

    def test_execute_with_condition_false(self):
        """Test conditional execution when condition is not met."""
        executor = MockActionExecutor()
        orchestrator = WorkflowOrchestrator(action_executor=executor)

        actions = [MockAction("action1")]
        result = orchestrator.execute_with_condition(
            actions, condition=lambda ctx: False  # Never execute
        )

        assert result.success
        assert result.completed_actions == 0
        assert len(executor.executed_actions) == 0

    def test_execute_with_condition_uses_context(self):
        """Test that condition receives execution context."""
        executor = MockActionExecutor(success_sequence=[True])
        orchestrator = WorkflowOrchestrator(action_executor=executor)

        context = ExecutionContext(initial_variables={"execute": True})
        actions = [MockAction("action1")]

        result = orchestrator.execute_with_condition(
            actions, condition=lambda ctx: ctx.get_variable("execute"), context=context
        )

        assert result.success
        assert result.completed_actions == 1

    def test_execute_parallel_sequential_fallback(self):
        """Test parallel execution (currently sequential fallback)."""
        executor = MockActionExecutor(success_sequence=[True, True, True])
        orchestrator = WorkflowOrchestrator(action_executor=executor)

        action_groups = [
            [MockAction("g1_a1"), MockAction("g1_a2")],
            [MockAction("g2_a1")],
        ]
        results = orchestrator.execute_parallel(action_groups)

        assert len(results) == 2
        assert all(r.success for r in results)
        assert results[0].completed_actions == 2
        assert results[1].completed_actions == 1

    def test_retry_policy_override(self):
        """Test that retry policy can be overridden per workflow."""
        executor = MockActionExecutor(success_sequence=[False, True])
        # Default policy: no retry
        default_policy = RetryPolicy.no_retry()
        orchestrator = WorkflowOrchestrator(action_executor=executor, retry_policy=default_policy)

        # Override with retry policy
        override_policy = RetryPolicy(max_retries=1, base_delay=0.01)
        actions = [MockAction("action1")]
        result = orchestrator.execute_workflow(actions, retry_policy=override_policy)

        assert result.success
        # Should have retried
        assert len(executor.executed_actions) == 2

    def test_action_name_extraction(self):
        """Test extraction of action names for logging."""
        executor = MockActionExecutor(success_sequence=[True])
        emitter = MockEventEmitter()
        orchestrator = WorkflowOrchestrator(action_executor=executor, event_emitter=emitter)

        actions = [MockAction("test_action")]
        orchestrator.execute_workflow(actions)

        # Check that action name appears in events
        action_events = [e for e in emitter.events if e["type"] == "action_started"]
        assert len(action_events) == 1
        assert action_events[0]["data"]["action"] == "test_action"

    def test_event_emission_failure_handling(self):
        """Test that event emission failures don't crash workflow."""

        class FailingEmitter:
            def emit(self, event_type: str, **kwargs: Any) -> None:
                raise RuntimeError("Event emission failed")

        executor = MockActionExecutor(success_sequence=[True])
        emitter = FailingEmitter()
        orchestrator = WorkflowOrchestrator(action_executor=executor, event_emitter=emitter)

        actions = [MockAction("action1")]
        # Should not raise despite emitter failures
        result = orchestrator.execute_workflow(actions)

        assert result.success

    def test_workflow_result_string_representation(self):
        """Test WorkflowResult string representation."""
        context = ExecutionContext()
        context.start_workflow()
        context.complete_workflow()

        result = WorkflowResult(success=True, context=context, completed_actions=5)
        str_repr = str(result)

        assert "SUCCESS" in str_repr
        assert "ExecutionStatistics" in str_repr

    def test_multiple_workflows_with_same_orchestrator(self):
        """Test executing multiple workflows with same orchestrator."""
        executor = MockActionExecutor(success_sequence=[True, True, True])
        orchestrator = WorkflowOrchestrator(action_executor=executor)

        # First workflow
        result1 = orchestrator.execute_workflow([MockAction("a1")])
        assert result1.success

        # Second workflow
        result2 = orchestrator.execute_workflow([MockAction("a2"), MockAction("a3")])
        assert result2.success

        # Total executions
        assert len(executor.executed_actions) == 3


class TestWorkflowOrchestratorIntegration:
    """Integration tests for WorkflowOrchestrator."""

    def test_complex_workflow_with_retries_and_failures(self):
        """Test complex workflow with mixed success, retries, and failures."""
        # Action 1: Success
        # Action 2: Fail twice, succeed on third attempt
        # Action 3: Success
        # Action 4: Fail all attempts (stop workflow)
        executor = MockActionExecutor(
            success_sequence=[
                True,  # Action 1
                False,
                False,
                True,  # Action 2 (2 retries)
                True,  # Action 3
                False,
                False,
                False,  # Action 4 (all fail)
            ]
        )

        policy = RetryPolicy(max_retries=2, base_delay=0.01)
        emitter = MockEventEmitter()
        orchestrator = WorkflowOrchestrator(
            action_executor=executor, retry_policy=policy, event_emitter=emitter
        )

        actions = [
            MockAction("login"),
            MockAction("fetch_data"),
            MockAction("process"),
            MockAction("save"),
        ]

        result = orchestrator.execute_workflow(actions)

        # Should fail on action 4
        assert not result.success
        assert result.failed_action_index == 3
        assert result.completed_actions == 3

        # Check statistics
        stats = result.context.statistics
        assert stats.total_actions == 4
        assert stats.successful_actions == 3
        assert stats.failed_actions == 1
        assert stats.retried_actions == 2  # Actions 2 and 4
        assert stats.total_retries == 4  # 2 for action 2, 2 for action 4

        # Check events
        retry_events = [e for e in emitter.events if e["type"] == "action_retrying"]
        assert len(retry_events) == 4

    def test_workflow_with_variable_substitution(self):
        """Test workflow execution with variable context."""
        executor = MockActionExecutor(success_sequence=[True, True])
        orchestrator = WorkflowOrchestrator(action_executor=executor)

        context = ExecutionContext(initial_variables={"user": "alice", "env": "prod"})

        actions = [MockAction("action1"), MockAction("action2")]
        result = orchestrator.execute_workflow(actions, context=context)

        assert result.success
        # Variables should be preserved
        assert result.context.get_variable("user") == "alice"
        assert result.context.get_variable("env") == "prod"

        # Can add new variables during workflow
        result.context.set_variable("result", "success")
        assert result.context.get_variable("result") == "success"
