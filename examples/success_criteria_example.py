"""Example usage of workflow-level success criteria.

This example demonstrates how to use the success criteria system to evaluate
workflow execution beyond simple all-actions-pass logic.
"""

from qontinui.config import Action, Connections, Workflow
from qontinui.execution import (
    SuccessCriteria,
    SuccessCriteriaType,
    evaluate_workflow_success,
)
from qontinui.execution.graph_executor import ExecutionState, GraphExecutor


def create_state_discovery_workflow() -> Workflow:
    """Create a workflow for state discovery use case."""
    return Workflow(
        id="state_discovery",
        name="State Discovery Workflow",
        version="1.0.0",
        format="graph",
        actions=[
            Action(
                id="find_buttons",
                type="FIND",
                name="Find all buttons",
                config={
                    "target": {"type": "image", "path": "button_pattern.png"},
                    "find_all": True,
                },
            ),
            Action(
                id="find_icons",
                type="FIND",
                name="Find all icons",
                config={
                    "target": {"type": "image", "path": "icon_pattern.png"},
                    "find_all": True,
                },
            ),
            Action(
                id="find_text_fields",
                type="FIND",
                name="Find all text fields",
                config={
                    "target": {"type": "text", "pattern": ".*"},
                    "find_all": True,
                },
            ),
        ],
        connections=Connections(
            root={
                "find_buttons": {"main": [[{"action": "find_icons", "type": "main", "index": 0}]]},
                "find_icons": {"main": [[{"action": "find_text_fields", "type": "main", "index": 0}]]},
            }
        ),
    )


def example_default_criteria():
    """Example: Default behavior (all actions must pass)."""
    print("=== Example 1: Default Criteria (All Actions Pass) ===")

    workflow = create_state_discovery_workflow()
    execution_state = ExecutionState(workflow)
    execution_state.start_time = 0.0

    # Simulate successful execution with matches
    execution_state.mark_completed(
        "find_buttons",
        {"success": True, "match_count": 5},
    )
    execution_state.mark_completed(
        "find_icons",
        {"success": True, "match_count": 3},
    )
    execution_state.mark_completed(
        "find_text_fields",
        {"success": True, "match_count": 2},
    )

    execution_state.end_time = 1.0

    # Evaluate with default criteria
    result = evaluate_workflow_success(execution_state)

    print(f"Success: {result.success}")
    print(f"Total matches: {result.total_matches}")
    print(f"Evaluation: {result.criteria_evaluation}")
    print()


def example_min_matches_criteria():
    """Example: MIN_MATCHES criteria for state discovery."""
    print("=== Example 2: MIN_MATCHES Criteria (State Discovery) ===")

    workflow = create_state_discovery_workflow()
    execution_state = ExecutionState(workflow)
    execution_state.start_time = 0.0

    # Simulate execution where some find actions fail but we found enough matches
    execution_state.mark_completed(
        "find_buttons",
        {"success": True, "match_count": 8},
    )
    execution_state.mark_failed("find_icons", "Pattern not found")
    execution_state.mark_completed(
        "find_text_fields",
        {"success": True, "match_count": 4},
    )

    execution_state.end_time = 1.0

    # Create criteria: require at least 10 matches
    criteria = SuccessCriteria(
        criteria_type=SuccessCriteriaType.MIN_MATCHES,
        min_matches=10,
        description="State discovery requires at least 10 UI elements",
    )

    # Evaluate
    result = evaluate_workflow_success(execution_state, criteria)

    print(f"Success: {result.success}")
    print(f"Total matches: {result.total_matches} (required: 10)")
    print(f"Failed actions: {result.failed_actions}")
    print(f"Evaluation: {result.criteria_evaluation}")
    print()


def example_max_failures_criteria():
    """Example: MAX_FAILURES criteria for failure tolerance."""
    print("=== Example 3: MAX_FAILURES Criteria (Failure Tolerance) ===")

    workflow = create_state_discovery_workflow()
    execution_state = ExecutionState(workflow)
    execution_state.start_time = 0.0

    # Simulate execution with 1 failure
    execution_state.mark_completed(
        "find_buttons",
        {"success": True, "match_count": 5},
    )
    execution_state.mark_failed("find_icons", "Pattern not found")
    execution_state.mark_completed(
        "find_text_fields",
        {"success": True, "match_count": 2},
    )

    execution_state.end_time = 1.0

    # Create criteria: allow up to 1 failure
    criteria = SuccessCriteria(
        criteria_type=SuccessCriteriaType.MAX_FAILURES,
        max_failures=1,
        description="Allow 1 action to fail",
    )

    # Evaluate
    result = evaluate_workflow_success(execution_state, criteria)

    print(f"Success: {result.success}")
    print(f"Failed actions: {result.failed_actions} (allowed: 1)")
    print(f"Successful actions: {result.successful_actions}")
    print(f"Evaluation: {result.criteria_evaluation}")
    print()


def example_checkpoint_criteria():
    """Example: CHECKPOINT_PASSED criteria for critical actions."""
    print("=== Example 4: CHECKPOINT_PASSED Criteria ===")

    workflow = Workflow(
        id="login_flow",
        name="Login Flow",
        version="1.0.0",
        format="graph",
        actions=[
            Action(
                id="enter_username",
                type="TYPE",
                name="Enter username",
                config={"text": "user@example.com"},
            ),
            Action(
                id="enter_password",
                type="TYPE",
                name="Enter password",
                config={"text": "password123"},
            ),
            Action(
                id="verify_login_checkpoint",
                type="FIND",
                name="Login success checkpoint",
                config={"target": {"type": "image", "path": "dashboard.png"}},
            ),
        ],
        connections=Connections(
            root={
                "enter_username": {
                    "main": [[{"action": "enter_password", "type": "main", "index": 0}]]
                },
                "enter_password": {
                    "main": [[{"action": "verify_login_checkpoint", "type": "main", "index": 0}]]
                },
            }
        ),
    )

    execution_state = ExecutionState(workflow)
    execution_state.start_time = 0.0

    # Simulate successful login
    execution_state.mark_completed("enter_username", {"success": True})
    execution_state.mark_completed("enter_password", {"success": True})
    execution_state.mark_completed(
        "verify_login_checkpoint",
        {"success": True, "match_count": 1},
    )

    execution_state.end_time = 1.0

    # Create criteria: checkpoint must pass
    criteria = SuccessCriteria(
        criteria_type=SuccessCriteriaType.CHECKPOINT_PASSED,
        checkpoint_name="Login success checkpoint",
        description="Login must succeed",
    )

    # Evaluate
    result = evaluate_workflow_success(execution_state, criteria)

    print(f"Success: {result.success}")
    print(f"Checkpoints passed: {result.checkpoints_passed}")
    print(f"Evaluation: {result.criteria_evaluation}")
    print()


def example_custom_criteria():
    """Example: CUSTOM criteria with Python expression."""
    print("=== Example 5: CUSTOM Criteria ===")

    workflow = create_state_discovery_workflow()
    execution_state = ExecutionState(workflow)
    execution_state.start_time = 0.0

    # Simulate execution
    execution_state.mark_completed(
        "find_buttons",
        {"success": True, "match_count": 5},
    )
    execution_state.mark_completed(
        "find_icons",
        {"success": True, "match_count": 3},
    )
    execution_state.mark_failed("find_text_fields", "Timeout")

    execution_state.end_time = 1.0

    # Create custom criteria: at least 7 matches and no more than 1 failure
    criteria = SuccessCriteria(
        criteria_type=SuccessCriteriaType.CUSTOM,
        custom_condition="match_count >= 7 and failed_actions <= 1",
        description="Custom: 7+ matches and ≤1 failure",
    )

    # Evaluate
    result = evaluate_workflow_success(execution_state, criteria)

    print(f"Success: {result.success}")
    print(f"Total matches: {result.total_matches}")
    print(f"Failed actions: {result.failed_actions}")
    print(f"Evaluation: {result.criteria_evaluation}")
    print()


if __name__ == "__main__":
    example_default_criteria()
    example_min_matches_criteria()
    example_max_failures_criteria()
    example_checkpoint_criteria()
    example_custom_criteria()
