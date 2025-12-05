"""Tests for success criteria evaluation.

This module tests the success criteria system including:
- All supported criteria types
- Statistics computation
- Custom expression evaluation
- Workflow result generation
"""

import pytest

from qontinui.config import Action, Connections, Workflow
from qontinui.execution import (
    SuccessCriteria,
    SuccessCriteriaEvaluator,
    SuccessCriteriaType,
    WorkflowResult,
    evaluate_workflow_success,
)
from qontinui.execution.graph_executor import ExecutionState


@pytest.fixture
def sample_workflow():
    """Create a sample workflow for testing."""
    return Workflow(
        id="test_workflow",
        name="Test Workflow",
        version="1.0.0",
        format="graph",
        actions=[
            Action(
                id="action1",
                type="FIND",
                name="Find buttons",
                config={"target": {"type": "image"}},
            ),
            Action(
                id="action2",
                type="CLICK",
                name="Click button",
                config={"target": {"type": "image"}},
            ),
            Action(
                id="checkpoint1",
                type="FIND",
                name="Login checkpoint",
                config={"target": {"type": "image"}},
            ),
        ],
        connections=Connections(root={}),
    )


@pytest.fixture
def execution_state_all_pass(sample_workflow):
    """Create execution state where all actions pass."""
    state = ExecutionState(sample_workflow)
    state.start_time = 0.0
    state.end_time = 1.0

    # Mark all actions as completed
    state.mark_completed(
        "action1", {"success": True, "matches": [{"x": 0, "y": 0}], "match_count": 3}
    )
    state.mark_completed(
        "action2", {"success": True, "matches": [{"x": 0, "y": 0}], "match_count": 2}
    )
    state.mark_completed(
        "checkpoint1",
        {"success": True, "matches": [{"x": 0, "y": 0}], "match_count": 1},
    )

    return state


@pytest.fixture
def execution_state_with_failures(sample_workflow):
    """Create execution state with some failures."""
    state = ExecutionState(sample_workflow)
    state.start_time = 0.0
    state.end_time = 1.0

    state.mark_completed(
        "action1", {"success": True, "matches": [{"x": 0, "y": 0}], "match_count": 3}
    )
    state.mark_failed("action2", "Button not found")
    state.mark_completed(
        "checkpoint1",
        {"success": True, "matches": [{"x": 0, "y": 0}], "match_count": 1},
    )

    return state


@pytest.fixture
def execution_state_checkpoint_failed(sample_workflow):
    """Create execution state where checkpoint fails."""
    state = ExecutionState(sample_workflow)
    state.start_time = 0.0
    state.end_time = 1.0

    state.mark_completed(
        "action1", {"success": True, "matches": [{"x": 0, "y": 0}], "match_count": 3}
    )
    state.mark_completed(
        "action2", {"success": True, "matches": [{"x": 0, "y": 0}], "match_count": 2}
    )
    state.mark_failed("checkpoint1", "Login failed")

    return state


@pytest.fixture
def execution_state_with_states(sample_workflow):
    """Create execution state with states reached."""
    state = ExecutionState(sample_workflow)
    state.start_time = 0.0
    state.end_time = 1.0

    state.mark_completed(
        "action1",
        {
            "success": True,
            "matches": [{"x": 0, "y": 0}],
            "match_count": 3,
            "active_states": ["logged_in"],
        },
    )
    state.mark_completed(
        "action2",
        {
            "success": True,
            "matches": [{"x": 0, "y": 0}],
            "match_count": 2,
            "states_reached": ["dashboard", "loaded"],
        },
    )
    state.mark_completed(
        "checkpoint1",
        {"success": True, "matches": [{"x": 0, "y": 0}], "match_count": 1},
    )

    return state


class TestSuccessCriteriaEvaluator:
    """Tests for SuccessCriteriaEvaluator."""

    def test_all_actions_pass_success(self, execution_state_all_pass):
        """Test ALL_ACTIONS_PASS criteria with all passing."""
        criteria = SuccessCriteria(criteria_type=SuccessCriteriaType.ALL_ACTIONS_PASS)
        evaluator = SuccessCriteriaEvaluator()

        success, explanation = evaluator.evaluate(criteria, execution_state_all_pass)

        assert success is True
        assert "3 actions passed" in explanation

    def test_all_actions_pass_failure(self, execution_state_with_failures):
        """Test ALL_ACTIONS_PASS criteria with failures."""
        criteria = SuccessCriteria(criteria_type=SuccessCriteriaType.ALL_ACTIONS_PASS)
        evaluator = SuccessCriteriaEvaluator()

        success, explanation = evaluator.evaluate(criteria, execution_state_with_failures)

        assert success is False
        assert "1 of 3 actions failed" in explanation

    def test_min_matches_success(self, execution_state_all_pass):
        """Test MIN_MATCHES criteria with sufficient matches."""
        criteria = SuccessCriteria(
            criteria_type=SuccessCriteriaType.MIN_MATCHES,
            min_matches=5,
            description="Need at least 5 matches",
        )
        evaluator = SuccessCriteriaEvaluator()

        success, explanation = evaluator.evaluate(criteria, execution_state_all_pass)

        assert success is True
        assert "6 matches" in explanation
        assert "required: 5" in explanation

    def test_min_matches_failure(self, execution_state_all_pass):
        """Test MIN_MATCHES criteria with insufficient matches."""
        criteria = SuccessCriteria(criteria_type=SuccessCriteriaType.MIN_MATCHES, min_matches=10)
        evaluator = SuccessCriteriaEvaluator()

        success, explanation = evaluator.evaluate(criteria, execution_state_all_pass)

        assert success is False
        assert "Only found 6 matches" in explanation
        assert "needed 10" in explanation

    def test_min_matches_missing_parameter(self, execution_state_all_pass):
        """Test MIN_MATCHES criteria without min_matches parameter."""
        criteria = SuccessCriteria(criteria_type=SuccessCriteriaType.MIN_MATCHES)
        evaluator = SuccessCriteriaEvaluator()

        with pytest.raises(ValueError, match="requires min_matches parameter"):
            evaluator.evaluate(criteria, execution_state_all_pass)

    def test_max_failures_success(self, execution_state_with_failures):
        """Test MAX_FAILURES criteria with acceptable failures."""
        criteria = SuccessCriteria(
            criteria_type=SuccessCriteriaType.MAX_FAILURES,
            max_failures=1,
            description="Allow 1 failure",
        )
        evaluator = SuccessCriteriaEvaluator()

        success, explanation = evaluator.evaluate(criteria, execution_state_with_failures)

        assert success is True
        assert "1 failures" in explanation
        assert "allowed: 1" in explanation

    def test_max_failures_exceeded(self, execution_state_with_failures):
        """Test MAX_FAILURES criteria with too many failures."""
        criteria = SuccessCriteria(criteria_type=SuccessCriteriaType.MAX_FAILURES, max_failures=0)
        evaluator = SuccessCriteriaEvaluator()

        success, explanation = evaluator.evaluate(criteria, execution_state_with_failures)

        assert success is False
        assert "1 failures exceeds limit of 0" in explanation

    def test_checkpoint_passed(self, execution_state_all_pass):
        """Test CHECKPOINT_PASSED criteria with passing checkpoint."""
        criteria = SuccessCriteria(
            criteria_type=SuccessCriteriaType.CHECKPOINT_PASSED,
            checkpoint_name="Login checkpoint",
        )
        evaluator = SuccessCriteriaEvaluator()

        success, explanation = evaluator.evaluate(criteria, execution_state_all_pass)

        assert success is True
        assert "Checkpoint 'Login checkpoint' passed" in explanation

    def test_checkpoint_failed(self, execution_state_checkpoint_failed):
        """Test CHECKPOINT_PASSED criteria with failing checkpoint."""
        criteria = SuccessCriteria(
            criteria_type=SuccessCriteriaType.CHECKPOINT_PASSED,
            checkpoint_name="Login checkpoint",
        )
        evaluator = SuccessCriteriaEvaluator()

        success, explanation = evaluator.evaluate(criteria, execution_state_checkpoint_failed)

        assert success is False
        assert "Checkpoint 'Login checkpoint' failed" in explanation

    def test_checkpoint_not_executed(self, execution_state_all_pass):
        """Test CHECKPOINT_PASSED criteria with non-existent checkpoint."""
        criteria = SuccessCriteria(
            criteria_type=SuccessCriteriaType.CHECKPOINT_PASSED,
            checkpoint_name="Nonexistent checkpoint",
        )
        evaluator = SuccessCriteriaEvaluator()

        success, explanation = evaluator.evaluate(criteria, execution_state_all_pass)

        assert success is False
        assert "was not executed" in explanation

    def test_required_states_success(self, execution_state_with_states):
        """Test REQUIRED_STATES criteria with all states reached."""
        criteria = SuccessCriteria(
            criteria_type=SuccessCriteriaType.REQUIRED_STATES,
            required_states=("logged_in", "dashboard"),
        )
        evaluator = SuccessCriteriaEvaluator()

        success, explanation = evaluator.evaluate(criteria, execution_state_with_states)

        assert success is True
        assert "All required states reached" in explanation

    def test_required_states_missing(self, execution_state_with_states):
        """Test REQUIRED_STATES criteria with missing states."""
        criteria = SuccessCriteria(
            criteria_type=SuccessCriteriaType.REQUIRED_STATES,
            required_states=("logged_in", "dashboard", "admin_panel"),
        )
        evaluator = SuccessCriteriaEvaluator()

        success, explanation = evaluator.evaluate(criteria, execution_state_with_states)

        assert success is False
        assert "Missing required states" in explanation
        assert "admin_panel" in explanation

    def test_custom_condition_success(self, execution_state_all_pass):
        """Test CUSTOM criteria with successful condition."""
        criteria = SuccessCriteria(
            criteria_type=SuccessCriteriaType.CUSTOM,
            custom_condition="match_count >= 5 and failed_actions == 0",
        )
        evaluator = SuccessCriteriaEvaluator()

        success, explanation = evaluator.evaluate(criteria, execution_state_all_pass)

        assert success is True
        assert "evaluated to True" in explanation

    def test_custom_condition_failure(self, execution_state_with_failures):
        """Test CUSTOM criteria with failing condition."""
        criteria = SuccessCriteria(
            criteria_type=SuccessCriteriaType.CUSTOM,
            custom_condition="failed_actions == 0",
        )
        evaluator = SuccessCriteriaEvaluator()

        success, explanation = evaluator.evaluate(criteria, execution_state_with_failures)

        assert success is False
        assert "evaluated to False" in explanation

    def test_custom_condition_complex(self, execution_state_all_pass):
        """Test CUSTOM criteria with complex condition."""
        criteria = SuccessCriteria(
            criteria_type=SuccessCriteriaType.CUSTOM,
            custom_condition=(
                "match_count > 3 and "
                "successful_actions >= total_actions - 1 and "
                "failed_actions <= 1"
            ),
        )
        evaluator = SuccessCriteriaEvaluator()

        success, explanation = evaluator.evaluate(criteria, execution_state_all_pass)

        assert success is True

    def test_custom_condition_invalid_syntax(self, execution_state_all_pass):
        """Test CUSTOM criteria with invalid syntax."""
        criteria = SuccessCriteria(
            criteria_type=SuccessCriteriaType.CUSTOM,
            custom_condition="invalid syntax here!",
        )
        evaluator = SuccessCriteriaEvaluator()

        success, explanation = evaluator.evaluate(criteria, execution_state_all_pass)

        assert success is False
        assert "Invalid custom condition syntax" in explanation

    def test_custom_condition_unsafe_operation(self, execution_state_all_pass):
        """Test CUSTOM criteria with unsafe operation."""
        criteria = SuccessCriteria(
            criteria_type=SuccessCriteriaType.CUSTOM,
            custom_condition="__import__('os').system('ls')",
        )
        evaluator = SuccessCriteriaEvaluator()

        success, explanation = evaluator.evaluate(criteria, execution_state_all_pass)

        assert success is False
        assert "Unsafe operation" in explanation

    def test_custom_condition_non_boolean(self, execution_state_all_pass):
        """Test CUSTOM criteria returning non-boolean."""
        criteria = SuccessCriteria(
            criteria_type=SuccessCriteriaType.CUSTOM, custom_condition="match_count"
        )
        evaluator = SuccessCriteriaEvaluator()

        success, explanation = evaluator.evaluate(criteria, execution_state_all_pass)

        assert success is False
        assert "non-boolean" in explanation


class TestWorkflowResult:
    """Tests for WorkflowResult creation."""

    def test_create_workflow_result_all_pass(self, execution_state_all_pass):
        """Test creating workflow result for all-pass scenario."""
        criteria = SuccessCriteria(criteria_type=SuccessCriteriaType.ALL_ACTIONS_PASS)
        evaluator = SuccessCriteriaEvaluator()

        result = evaluator.create_workflow_result(criteria, execution_state_all_pass)

        assert isinstance(result, WorkflowResult)
        assert result.workflow_name == "Test Workflow"
        assert result.success is True
        assert result.total_actions == 3
        assert result.successful_actions == 3
        assert result.failed_actions == 0
        assert result.skipped_actions == 0
        assert result.total_matches == 6
        assert result.duration_ms == 1000  # 1 second

    def test_create_workflow_result_with_failures(self, execution_state_with_failures):
        """Test creating workflow result with failures."""
        criteria = SuccessCriteria(criteria_type=SuccessCriteriaType.MAX_FAILURES, max_failures=1)
        evaluator = SuccessCriteriaEvaluator()

        result = evaluator.create_workflow_result(criteria, execution_state_with_failures)

        assert result.success is True
        assert result.total_actions == 3
        assert result.successful_actions == 2
        assert result.failed_actions == 1
        assert result.total_matches == 4
        assert result.error is not None
        assert "Button not found" in result.error

    def test_create_workflow_result_default_criteria(self, execution_state_all_pass):
        """Test creating workflow result with default criteria."""
        evaluator = SuccessCriteriaEvaluator()

        result = evaluator.create_workflow_result(None, execution_state_all_pass)

        assert result.success is True
        assert result.success_criteria is not None
        assert result.success_criteria.criteria_type == SuccessCriteriaType.ALL_ACTIONS_PASS

    def test_workflow_result_str(self, execution_state_all_pass):
        """Test WorkflowResult string representation."""
        evaluator = SuccessCriteriaEvaluator()
        result = evaluator.create_workflow_result(None, execution_state_all_pass)

        result_str = str(result)

        assert "Test Workflow" in result_str
        assert "SUCCESS" in result_str
        assert "3/3 successful" in result_str
        assert "6" in result_str  # match count

    def test_workflow_result_checkpoints(self, execution_state_all_pass):
        """Test WorkflowResult checkpoint tracking."""
        evaluator = SuccessCriteriaEvaluator()
        result = evaluator.create_workflow_result(None, execution_state_all_pass)

        assert "Login checkpoint" in result.checkpoints_passed
        assert len(result.checkpoints_failed) == 0

    def test_workflow_result_states(self, execution_state_with_states):
        """Test WorkflowResult state tracking."""
        evaluator = SuccessCriteriaEvaluator()
        result = evaluator.create_workflow_result(None, execution_state_with_states)

        assert "logged_in" in result.states_reached
        assert "dashboard" in result.states_reached
        assert "loaded" in result.states_reached


class TestConvenienceFunction:
    """Tests for evaluate_workflow_success convenience function."""

    def test_evaluate_workflow_success_default(self, execution_state_all_pass):
        """Test convenience function with default criteria."""
        result = evaluate_workflow_success(execution_state_all_pass)

        assert isinstance(result, WorkflowResult)
        assert result.success is True

    def test_evaluate_workflow_success_custom_criteria(self, execution_state_all_pass):
        """Test convenience function with custom criteria."""
        criteria = SuccessCriteria(criteria_type=SuccessCriteriaType.MIN_MATCHES, min_matches=5)

        result = evaluate_workflow_success(execution_state_all_pass, criteria)

        assert result.success is True
        assert "6 matches" in result.criteria_evaluation


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_workflow(self):
        """Test evaluation with empty workflow."""
        workflow = Workflow(
            id="empty",
            name="Empty Workflow",
            version="1.0.0",
            format="graph",
            actions=[],
            connections=Connections(root={}),
        )

        state = ExecutionState(workflow)
        state.start_time = 0.0
        state.end_time = 0.1

        result = evaluate_workflow_success(state)

        assert result.success is True
        assert result.total_actions == 0
        assert result.total_matches == 0

    def test_no_matches_found(self, sample_workflow):
        """Test evaluation when no matches are found."""
        state = ExecutionState(sample_workflow)
        state.start_time = 0.0
        state.end_time = 1.0

        state.mark_completed("action1", {"success": True, "matches": []})
        state.mark_completed("action2", {"success": True, "matches": []})
        state.mark_completed("checkpoint1", {"success": True, "matches": []})

        criteria = SuccessCriteria(criteria_type=SuccessCriteriaType.MIN_MATCHES, min_matches=1)

        result = evaluate_workflow_success(state, criteria)

        assert result.success is False
        assert result.total_matches == 0

    def test_mixed_result_formats(self, sample_workflow):
        """Test handling of different result formats."""
        state = ExecutionState(sample_workflow)
        state.start_time = 0.0
        state.end_time = 1.0

        # Different formats for matches
        state.mark_completed("action1", {"success": True, "match_count": 3})
        state.mark_completed("action2", {"success": True, "matches": [{"x": 0}, {"x": 1}]})
        state.mark_completed("checkpoint1", {"success": True})

        result = evaluate_workflow_success(state)

        assert result.total_matches == 5  # 3 + 2 + 0

    def test_skipped_actions(self, sample_workflow):
        """Test evaluation with skipped actions."""
        state = ExecutionState(sample_workflow)
        state.start_time = 0.0
        state.end_time = 1.0

        state.mark_completed("action1", {"success": True, "match_count": 3})
        state.mark_skipped("action2", "Condition not met")
        state.mark_completed("checkpoint1", {"success": True, "match_count": 1})

        result = evaluate_workflow_success(state)

        assert result.total_actions == 3
        assert result.successful_actions == 2
        assert result.skipped_actions == 1
        assert result.success is True


class TestImmutability:
    """Tests for immutability of result objects."""

    def test_success_criteria_immutable(self):
        """Test that SuccessCriteria is immutable."""
        criteria = SuccessCriteria(criteria_type=SuccessCriteriaType.MIN_MATCHES, min_matches=5)

        with pytest.raises(AttributeError):
            criteria.min_matches = 10  # type: ignore[misc]

    def test_workflow_result_immutable(self, execution_state_all_pass):
        """Test that WorkflowResult is immutable."""
        result = evaluate_workflow_success(execution_state_all_pass)

        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]

        with pytest.raises(AttributeError):
            result.total_matches = 100  # type: ignore[misc]
