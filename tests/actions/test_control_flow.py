"""Tests for control flow action executor.

Tests cover LOOP (FOR, WHILE, FOREACH), IF, BREAK, and CONTINUE actions
with various configurations and edge cases.
"""

from typing import Any

import pytest

from qontinui.actions.control_flow import (
    BreakLoop,
    ContinueLoop,
    ControlFlowExecutor,
)
from qontinui.config import Action

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_action_executor():
    """Create a mock action executor that tracks calls."""
    executed_actions = []

    def executor(action_id: str, variables: dict[str, Any]) -> dict[str, Any]:
        executed_actions.append({"action_id": action_id, "variables": variables.copy()})
        return {"success": True}

    executor.executed_actions = executed_actions  # type: ignore
    return executor


@pytest.fixture
def failing_action_executor():
    """Create a mock action executor that fails."""

    def executor(action_id: str, variables: dict[str, Any]) -> dict[str, Any]:
        return {"success": False, "error": "Mock action failed"}

    return executor


@pytest.fixture
def break_action_executor():
    """Create a mock action executor that raises BreakLoop on specific action."""

    def executor(action_id: str, variables: dict[str, Any]) -> dict[str, Any]:
        if action_id == "break-action":
            raise BreakLoop("Test break")
        return {"success": True}

    return executor


@pytest.fixture
def continue_action_executor():
    """Create a mock action executor that raises ContinueLoop on specific action."""

    def executor(action_id: str, variables: dict[str, Any]) -> dict[str, Any]:
        if action_id == "continue-action":
            raise ContinueLoop("Test continue")
        return {"success": True}

    return executor


# ============================================================================
# FOR Loop Tests
# ============================================================================


def test_for_loop_basic(mock_action_executor):
    """Test basic FOR loop execution."""
    executor = ControlFlowExecutor(action_executor=mock_action_executor)

    action = Action(
        id="loop-1",
        type="LOOP",
        config={
            "loopType": "FOR",
            "iterations": 3,
            "iteratorVariable": "i",
            "actions": ["action-1"],
        },
    )

    result = executor.execute_loop(action)

    assert result["success"] is True
    assert result["iterations_completed"] == 3
    assert result["stopped_early"] is False
    assert len(mock_action_executor.executed_actions) == 3  # type: ignore

    # Check iterator variable was set correctly
    for i, exec_info in enumerate(mock_action_executor.executed_actions):  # type: ignore
        assert exec_info["variables"]["i"] == i


def test_for_loop_multiple_actions(mock_action_executor):
    """Test FOR loop with multiple actions per iteration."""
    executor = ControlFlowExecutor(action_executor=mock_action_executor)

    action = Action(
        id="loop-1",
        type="LOOP",
        config={
            "loopType": "FOR",
            "iterations": 2,
            "actions": ["action-1", "action-2", "action-3"],
        },
    )

    result = executor.execute_loop(action)

    assert result["success"] is True
    assert result["iterations_completed"] == 2
    assert (
        len(mock_action_executor.executed_actions) == 6
    )  # 2 iterations * 3 actions  # type: ignore


def test_for_loop_no_iterations():
    """Test FOR loop with zero iterations."""
    executor = ControlFlowExecutor()

    action = Action(
        id="loop-1",
        type="LOOP",
        config={
            "loopType": "FOR",
            "iterations": 0,
            "actions": ["action-1"],
        },
    )

    result = executor.execute_loop(action)

    assert result["success"] is True
    assert result["iterations_completed"] == 0


def test_for_loop_max_iterations_limit(mock_action_executor):
    """Test FOR loop respects max_iterations limit."""
    executor = ControlFlowExecutor(action_executor=mock_action_executor)

    action = Action(
        id="loop-1",
        type="LOOP",
        config={
            "loopType": "FOR",
            "iterations": 100,
            "maxIterations": 5,
            "actions": ["action-1"],
        },
    )

    result = executor.execute_loop(action)

    assert result["success"] is True
    assert result["iterations_completed"] == 5  # Capped at max
    assert len(mock_action_executor.executed_actions) == 5  # type: ignore


def test_for_loop_break_on_error(failing_action_executor):
    """Test FOR loop breaks on error when break_on_error=True."""
    executor = ControlFlowExecutor(action_executor=failing_action_executor)

    action = Action(
        id="loop-1",
        type="LOOP",
        config={
            "loopType": "FOR",
            "iterations": 5,
            "breakOnError": True,
            "actions": ["failing-action"],
        },
    )

    result = executor.execute_loop(action)

    assert result["success"] is True  # Loop itself succeeded (didn't crash)
    assert result["iterations_completed"] == 1  # Only first iteration
    assert len(result["errors"]) > 0


def test_for_loop_continue_on_error(failing_action_executor):
    """Test FOR loop continues on error when break_on_error=False."""
    executor = ControlFlowExecutor(action_executor=failing_action_executor)

    action = Action(
        id="loop-1",
        type="LOOP",
        config={
            "loopType": "FOR",
            "iterations": 3,
            "breakOnError": False,
            "actions": ["failing-action"],
        },
    )

    result = executor.execute_loop(action)

    assert result["success"] is True
    assert result["iterations_completed"] == 3  # All iterations
    assert len(result["errors"]) == 3  # One error per iteration


def test_for_loop_break(break_action_executor):
    """Test FOR loop with BREAK action."""
    executor = ControlFlowExecutor(action_executor=break_action_executor)

    action = Action(
        id="loop-1",
        type="LOOP",
        config={
            "loopType": "FOR",
            "iterations": 10,
            "actions": ["normal-action", "break-action", "should-not-execute"],
        },
    )

    result = executor.execute_loop(action)

    assert result["success"] is True
    assert result["stopped_early"] is True
    assert result["iterations_completed"] == 1
    assert "break_message" in result


def test_for_loop_continue(continue_action_executor):
    """Test FOR loop with CONTINUE action."""
    executor = ControlFlowExecutor(action_executor=continue_action_executor)

    action = Action(
        id="loop-1",
        type="LOOP",
        config={
            "loopType": "FOR",
            "iterations": 3,
            "actions": ["continue-action", "should-skip"],
        },
    )

    result = executor.execute_loop(action)

    assert result["success"] is True
    assert result["iterations_completed"] == 3


# ============================================================================
# WHILE Loop Tests
# ============================================================================


def test_while_loop_variable_condition(mock_action_executor):
    """Test WHILE loop with variable condition."""
    executor = ControlFlowExecutor(
        action_executor=mock_action_executor, variables={"counter": 0}
    )

    # Mock executor that increments counter
    def incrementing_executor(
        action_id: str, variables: dict[str, Any]
    ) -> dict[str, Any]:
        variables["counter"] = variables.get("counter", 0) + 1
        mock_action_executor(action_id, variables)
        return {"success": True}

    executor.action_executor = incrementing_executor

    action = Action(
        id="loop-1",
        type="LOOP",
        config={
            "loopType": "WHILE",
            "condition": {
                "type": "variable",
                "variableName": "counter",
                "operator": "<",
                "expectedValue": 5,
            },
            "actions": ["increment-action"],
        },
    )

    result = executor.execute_loop(action)

    assert result["success"] is True
    assert result["iterations_completed"] == 5
    assert executor.variables["counter"] == 5


def test_while_loop_expression_condition(mock_action_executor):
    """Test WHILE loop with expression condition."""
    executor = ControlFlowExecutor(
        action_executor=mock_action_executor, variables={"x": 0}
    )

    def incrementing_executor(
        action_id: str, variables: dict[str, Any]
    ) -> dict[str, Any]:
        variables["x"] = variables.get("x", 0) + 2
        mock_action_executor(action_id, variables)
        return {"success": True}

    executor.action_executor = incrementing_executor

    action = Action(
        id="loop-1",
        type="LOOP",
        config={
            "loopType": "WHILE",
            "condition": {"type": "expression", "expression": "x < 10"},
            "actions": ["increment-action"],
        },
    )

    result = executor.execute_loop(action)

    assert result["success"] is True
    assert result["iterations_completed"] == 5
    assert executor.variables["x"] == 10


def test_while_loop_max_iterations_safety():
    """Test WHILE loop respects max_iterations safety limit."""
    # Create an always-true condition
    executor = ControlFlowExecutor(variables={"always_true": True})

    action = Action(
        id="loop-1",
        type="LOOP",
        config={
            "loopType": "WHILE",
            "condition": {
                "type": "variable",
                "variableName": "always_true",
                "operator": "==",
                "expectedValue": True,
            },
            "maxIterations": 10,
            "actions": ["action-1"],
        },
    )

    result = executor.execute_loop(action)

    assert result["success"] is True
    assert result["iterations_completed"] == 10
    assert any(err["type"] == "MaxIterationsExceeded" for err in result["errors"])


def test_while_loop_false_initially():
    """Test WHILE loop that never executes."""
    executor = ControlFlowExecutor(variables={"value": 10})

    action = Action(
        id="loop-1",
        type="LOOP",
        config={
            "loopType": "WHILE",
            "condition": {
                "type": "variable",
                "variableName": "value",
                "operator": "<",
                "expectedValue": 5,
            },
            "actions": ["action-1"],
        },
    )

    result = executor.execute_loop(action)

    assert result["success"] is True
    assert result["iterations_completed"] == 0


# ============================================================================
# FOREACH Loop Tests
# ============================================================================


def test_foreach_loop_variable_collection(mock_action_executor):
    """Test FOREACH loop with variable collection."""
    items = ["apple", "banana", "cherry"]
    executor = ControlFlowExecutor(
        action_executor=mock_action_executor, variables={"fruits": items}
    )

    action = Action(
        id="loop-1",
        type="LOOP",
        config={
            "loopType": "FOREACH",
            "collection": {"type": "variable", "variableName": "fruits"},
            "iteratorVariable": "fruit",
            "actions": ["process-fruit"],
        },
    )

    result = executor.execute_loop(action)

    assert result["success"] is True
    assert result["iterations_completed"] == 3

    # Check that iterator variable was set to each item
    for i, exec_info in enumerate(mock_action_executor.executed_actions):  # type: ignore
        assert exec_info["variables"]["fruit"] == items[i]


def test_foreach_loop_range_collection(mock_action_executor):
    """Test FOREACH loop with range collection."""
    executor = ControlFlowExecutor(action_executor=mock_action_executor)

    action = Action(
        id="loop-1",
        type="LOOP",
        config={
            "loopType": "FOREACH",
            "collection": {"type": "range", "start": 0, "end": 5, "step": 1},
            "iteratorVariable": "num",
            "actions": ["process-number"],
        },
    )

    result = executor.execute_loop(action)

    assert result["success"] is True
    assert result["iterations_completed"] == 5

    # Check range values
    for i, exec_info in enumerate(mock_action_executor.executed_actions):  # type: ignore
        assert exec_info["variables"]["num"] == i


def test_foreach_loop_empty_collection():
    """Test FOREACH loop with empty collection."""
    executor = ControlFlowExecutor(variables={"empty_list": []})

    action = Action(
        id="loop-1",
        type="LOOP",
        config={
            "loopType": "FOREACH",
            "collection": {"type": "variable", "variableName": "empty_list"},
            "actions": ["action-1"],
        },
    )

    result = executor.execute_loop(action)

    assert result["success"] is True
    assert result["iterations_completed"] == 0


def test_foreach_loop_max_iterations_limit(mock_action_executor):
    """Test FOREACH loop respects max_iterations."""
    items = list(range(100))
    executor = ControlFlowExecutor(
        action_executor=mock_action_executor, variables={"numbers": items}
    )

    action = Action(
        id="loop-1",
        type="LOOP",
        config={
            "loopType": "FOREACH",
            "collection": {"type": "variable", "variableName": "numbers"},
            "maxIterations": 10,
            "actions": ["action-1"],
        },
    )

    result = executor.execute_loop(action)

    assert result["success"] is True
    assert result["iterations_completed"] == 10
    assert len(mock_action_executor.executed_actions) == 10  # type: ignore


# ============================================================================
# IF Action Tests
# ============================================================================


def test_if_then_true_condition(mock_action_executor):
    """Test IF action with true condition executes THEN branch."""
    executor = ControlFlowExecutor(
        action_executor=mock_action_executor, variables={"value": 10}
    )

    action = Action(
        id="if-1",
        type="IF",
        config={
            "condition": {
                "type": "variable",
                "variableName": "value",
                "operator": ">",
                "expectedValue": 5,
            },
            "thenActions": ["then-action-1", "then-action-2"],
            "elseActions": ["else-action-1"],
        },
    )

    result = executor.execute_if(action)

    assert result["success"] is True
    assert result["condition_result"] is True
    assert result["branch_taken"] == "then"
    assert result["actions_executed"] == 2

    executed_ids = [a["action_id"] for a in mock_action_executor.executed_actions]  # type: ignore
    assert "then-action-1" in executed_ids
    assert "then-action-2" in executed_ids
    assert "else-action-1" not in executed_ids


def test_if_else_false_condition(mock_action_executor):
    """Test IF action with false condition executes ELSE branch."""
    executor = ControlFlowExecutor(
        action_executor=mock_action_executor, variables={"value": 3}
    )

    action = Action(
        id="if-1",
        type="IF",
        config={
            "condition": {
                "type": "variable",
                "variableName": "value",
                "operator": ">",
                "expectedValue": 5,
            },
            "thenActions": ["then-action-1"],
            "elseActions": ["else-action-1", "else-action-2"],
        },
    )

    result = executor.execute_if(action)

    assert result["success"] is True
    assert result["condition_result"] is False
    assert result["branch_taken"] == "else"
    assert result["actions_executed"] == 2

    executed_ids = [a["action_id"] for a in mock_action_executor.executed_actions]  # type: ignore
    assert "else-action-1" in executed_ids
    assert "else-action-2" in executed_ids
    assert "then-action-1" not in executed_ids


def test_if_no_else_branch(mock_action_executor):
    """Test IF action without else branch."""
    executor = ControlFlowExecutor(
        action_executor=mock_action_executor, variables={"value": 3}
    )

    action = Action(
        id="if-1",
        type="IF",
        config={
            "condition": {
                "type": "variable",
                "variableName": "value",
                "operator": ">",
                "expectedValue": 5,
            },
            "thenActions": ["then-action-1"],
        },
    )

    result = executor.execute_if(action)

    assert result["success"] is True
    assert result["condition_result"] is False
    assert result["branch_taken"] == "else"
    assert result["actions_executed"] == 0


def test_if_expression_condition(mock_action_executor):
    """Test IF action with expression condition."""
    executor = ControlFlowExecutor(
        action_executor=mock_action_executor, variables={"x": 5, "y": 10}
    )

    action = Action(
        id="if-1",
        type="IF",
        config={
            "condition": {"type": "expression", "expression": "x + y > 10"},
            "thenActions": ["action-1"],
        },
    )

    result = executor.execute_if(action)

    assert result["success"] is True
    assert result["condition_result"] is True
    assert result["actions_executed"] == 1


# ============================================================================
# BREAK Action Tests
# ============================================================================


def test_break_unconditional():
    """Test unconditional BREAK action."""
    executor = ControlFlowExecutor()

    action = Action(id="break-1", type="BREAK", config={"message": "Breaking now"})

    with pytest.raises(BreakLoop) as exc_info:
        executor.execute_break(action)

    assert "Breaking now" in str(exc_info.value)


def test_break_conditional_true():
    """Test conditional BREAK with true condition."""
    executor = ControlFlowExecutor(variables={"should_break": True})

    action = Action(
        id="break-1",
        type="BREAK",
        config={
            "condition": {
                "type": "variable",
                "variableName": "should_break",
                "operator": "==",
                "expectedValue": True,
            }
        },
    )

    with pytest.raises(BreakLoop):
        executor.execute_break(action)


def test_break_conditional_false():
    """Test conditional BREAK with false condition (should not break)."""
    executor = ControlFlowExecutor(variables={"should_break": False})

    action = Action(
        id="break-1",
        type="BREAK",
        config={
            "condition": {
                "type": "variable",
                "variableName": "should_break",
                "operator": "==",
                "expectedValue": True,
            }
        },
    )

    # Should not raise - condition is false
    executor.execute_break(action)


# ============================================================================
# CONTINUE Action Tests
# ============================================================================


def test_continue_unconditional():
    """Test unconditional CONTINUE action."""
    executor = ControlFlowExecutor()

    action = Action(id="continue-1", type="CONTINUE", config={"message": "Continuing"})

    with pytest.raises(ContinueLoop) as exc_info:
        executor.execute_continue(action)

    assert "Continuing" in str(exc_info.value)


def test_continue_conditional_true():
    """Test conditional CONTINUE with true condition."""
    executor = ControlFlowExecutor(variables={"should_continue": True})

    action = Action(
        id="continue-1",
        type="CONTINUE",
        config={
            "condition": {
                "type": "variable",
                "variableName": "should_continue",
                "operator": "==",
                "expectedValue": True,
            }
        },
    )

    with pytest.raises(ContinueLoop):
        executor.execute_continue(action)


def test_continue_conditional_false():
    """Test conditional CONTINUE with false condition (should not continue)."""
    executor = ControlFlowExecutor(variables={"should_continue": False})

    action = Action(
        id="continue-1",
        type="CONTINUE",
        config={
            "condition": {
                "type": "variable",
                "variableName": "should_continue",
                "operator": "==",
                "expectedValue": True,
            }
        },
    )

    # Should not raise - condition is false
    executor.execute_continue(action)


# ============================================================================
# Condition Evaluation Tests
# ============================================================================


def test_condition_operators():
    """Test all comparison operators."""
    executor = ControlFlowExecutor(variables={"value": 10})

    test_cases = [
        ("==", 10, True),
        ("==", 5, False),
        ("!=", 5, True),
        ("!=", 10, False),
        (">", 5, True),
        (">", 15, False),
        ("<", 15, True),
        ("<", 5, False),
        (">=", 10, True),
        (">=", 5, True),
        (">=", 15, False),
        ("<=", 10, True),
        ("<=", 15, True),
        ("<=", 5, False),
    ]

    for operator, expected, should_match in test_cases:
        result = executor._compare_values(
            executor.variables["value"], operator, expected
        )
        assert (
            result == should_match
        ), f"Failed: 10 {operator} {expected} should be {should_match}"


def test_condition_contains_operator():
    """Test contains operator."""
    executor = ControlFlowExecutor(variables={"text": "hello world"})

    result = executor._compare_values("hello world", "contains", "world")
    assert result is True

    result = executor._compare_values("hello world", "contains", "goodbye")
    assert result is False


def test_condition_matches_operator():
    """Test regex matches operator."""
    executor = ControlFlowExecutor()

    result = executor._compare_values("test123", "matches", r"test\d+")
    assert result is True

    result = executor._compare_values("test", "matches", r"test\d+")
    assert result is False


def test_expression_with_variables():
    """Test expression evaluation with variables."""
    executor = ControlFlowExecutor(variables={"x": 5, "y": 10, "z": 15})

    action = Action(
        id="if-1",
        type="IF",
        config={
            "condition": {"type": "expression", "expression": "x + y == z"},
            "thenActions": ["action-1"],
        },
    )

    result = executor.execute_if(action)
    assert result["condition_result"] is True


# ============================================================================
# Variable Management Tests
# ============================================================================


def test_set_and_get_variable():
    """Test setting and getting variables."""
    executor = ControlFlowExecutor()

    executor.set_variable("test", 42)
    assert executor.get_variable("test") == 42


def test_get_variable_default():
    """Test getting variable with default."""
    executor = ControlFlowExecutor()

    assert executor.get_variable("nonexistent", "default") == "default"


def test_clear_variables():
    """Test clearing all variables."""
    executor = ControlFlowExecutor(variables={"a": 1, "b": 2})

    executor.clear_variables()
    assert len(executor.variables) == 0


def test_get_all_variables():
    """Test getting all variables."""
    variables = {"a": 1, "b": 2, "c": 3}
    executor = ControlFlowExecutor(variables=variables)

    all_vars = executor.get_all_variables()
    assert all_vars == variables
    assert all_vars is not executor.variables  # Should be a copy


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_invalid_loop_type():
    """Test error handling for invalid loop type."""
    executor = ControlFlowExecutor()

    action = Action(
        id="loop-1",
        type="LOOP",
        config={"loopType": "INVALID", "actions": []},
    )

    result = executor.execute_loop(action)

    assert result["success"] is False
    assert len(result["errors"]) > 0


def test_for_loop_missing_iterations():
    """Test FOR loop without iterations specified."""
    executor = ControlFlowExecutor()

    action = Action(
        id="loop-1",
        type="LOOP",
        config={"loopType": "FOR", "actions": []},
    )

    result = executor.execute_loop(action)

    assert result["success"] is False


def test_while_loop_missing_condition():
    """Test WHILE loop without condition."""
    executor = ControlFlowExecutor()

    action = Action(
        id="loop-1",
        type="LOOP",
        config={"loopType": "WHILE", "actions": []},
    )

    result = executor.execute_loop(action)

    assert result["success"] is False


def test_foreach_loop_missing_collection():
    """Test FOREACH loop without collection."""
    executor = ControlFlowExecutor()

    action = Action(
        id="loop-1",
        type="LOOP",
        config={"loopType": "FOREACH", "actions": []},
    )

    result = executor.execute_loop(action)

    assert result["success"] is False


def test_foreach_loop_nonexistent_variable():
    """Test FOREACH loop with nonexistent variable."""
    executor = ControlFlowExecutor()

    action = Action(
        id="loop-1",
        type="LOOP",
        config={
            "loopType": "FOREACH",
            "collection": {"type": "variable", "variableName": "nonexistent"},
            "actions": [],
        },
    )

    result = executor.execute_loop(action)

    assert result["success"] is False
    assert len(result["errors"]) > 0


def test_invalid_expression():
    """Test error handling for invalid expression."""
    executor = ControlFlowExecutor()

    action = Action(
        id="if-1",
        type="IF",
        config={
            "condition": {"type": "expression", "expression": "invalid syntax !!!"},
            "thenActions": [],
        },
    )

    result = executor.execute_if(action)

    assert result["success"] is False
    assert len(result["errors"]) > 0


# ============================================================================
# Integration Tests
# ============================================================================


def test_nested_loop_simulation(mock_action_executor):
    """Test simulation of nested loops using sequential execution."""
    executor = ControlFlowExecutor(action_executor=mock_action_executor)

    # Outer loop
    outer_action = Action(
        id="outer-loop",
        type="LOOP",
        config={
            "loopType": "FOR",
            "iterations": 3,
            "iteratorVariable": "i",
            "actions": ["inner-loop-ref"],
        },
    )

    outer_result = executor.execute_loop(outer_action)

    assert outer_result["success"] is True
    assert outer_result["iterations_completed"] == 3


def test_loop_with_if_simulation(mock_action_executor):
    """Test loop containing IF statement."""
    executor = ControlFlowExecutor(
        action_executor=mock_action_executor, variables={"threshold": 5}
    )

    action = Action(
        id="loop-1",
        type="LOOP",
        config={
            "loopType": "FOR",
            "iterations": 3,
            "iteratorVariable": "i",
            "actions": ["if-action-ref"],
        },
    )

    result = executor.execute_loop(action)

    assert result["success"] is True
    assert result["iterations_completed"] == 3
