"""Integration tests for new action schema.

Tests the complete pipeline:
1. Create actions using new schema format
2. Export to JSON
3. Import from JSON
4. Execute actions with json_executor
5. Verify results

This validates that the new Pydantic schema works end-to-end.
"""

import json
import sys
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qontinui.actions.control_flow import ControlFlowExecutor
from qontinui.actions.data_operations import DataOperationsExecutor
from qontinui.config import (
    Action,
    ClickActionConfig,
    SetVariableActionConfig,
    SortActionConfig,
    get_typed_config,
    load_actions_from_string,
)


class IntegrationTestRunner:
    """Runner for integration tests."""

    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results: list[dict[str, Any]] = []

    def run_test(self, name: str, test_func):
        """Run a single test and track results."""
        print(f"\n{'=' * 70}")
        print(f"Test: {name}")
        print("=" * 70)
        try:
            test_func()
            print(f"✅ PASSED: {name}")
            self.passed_tests += 1
            self.test_results.append({"name": name, "status": "PASSED"})
        except AssertionError as e:
            print(f"❌ FAILED: {name}")
            print(f"   Error: {e}")
            self.failed_tests += 1
            self.test_results.append(
                {"name": name, "status": "FAILED", "error": str(e)}
            )
        except Exception as e:
            print(f"❌ ERROR: {name}")
            print(f"   Error: {e}")
            self.failed_tests += 1
            self.test_results.append({"name": name, "status": "ERROR", "error": str(e)})

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"Total tests: {self.passed_tests + self.failed_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print("=" * 70)

        if self.failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! 🎉\n")
        else:
            print("\n⚠️ Some tests failed. Review output above.\n")


def test_action_creation_and_serialization():
    """Test 1: Create actions and serialize to JSON."""
    print("\nCreating actions with new schema...")

    # Create CLICK action
    click_action = Action(
        id="click-1",
        type="CLICK",
        config={
            "target": {"type": "image", "imageId": "button-submit"},
            "numberOfClicks": 1,
            "mouseButton": "LEFT",
        },
    )

    # Create LOOP action
    loop_action = Action(
        id="loop-1",
        type="LOOP",
        config={
            "loopType": "FOR",
            "iterations": 3,
            "iteratorVariable": "i",
            "actions": ["click-1"],
        },
    )

    # Create IF action
    if_action = Action(
        id="if-1",
        type="IF",
        config={
            "condition": {
                "type": "variable",
                "variableName": "counter",
                "operator": ">",
                "expectedValue": 5,
            },
            "thenActions": ["loop-1"],
            "elseActions": [],
        },
    )

    # Serialize to JSON
    actions = [click_action, loop_action, if_action]
    json_str = json.dumps([action.model_dump() for action in actions], indent=2)

    print(f"Created {len(actions)} actions")
    print(f"JSON size: {len(json_str)} bytes")
    print(f"Sample JSON:\n{json_str[:300]}...")

    # Verify JSON structure
    parsed = json.loads(json_str)
    assert len(parsed) == 3, "Should have 3 actions"
    assert parsed[0]["type"] == "CLICK", "First action should be CLICK"
    assert parsed[1]["type"] == "LOOP", "Second action should be LOOP"
    assert parsed[2]["type"] == "IF", "Third action should be IF"

    print("✓ Actions created and serialized successfully")


def test_json_import_and_validation():
    """Test 2: Import JSON and validate with Pydantic."""
    print("\nImporting actions from JSON...")

    json_data = """
    [
        {
            "id": "set-var-1",
            "type": "SET_VARIABLE",
            "config": {
                "variableName": "total",
                "value": 100,
                "scope": "local"
            }
        },
        {
            "id": "sort-1",
            "type": "SORT",
            "config": {
                "target": "variable",
                "variableName": "items",
                "sortBy": "price",
                "order": "ASC",
                "comparator": "NUMERIC"
            }
        }
    ]
    """

    # Load actions
    actions = load_actions_from_string(json_data)

    print(f"Loaded {len(actions)} actions from JSON")

    # Validate structure
    assert len(actions) == 2, "Should load 2 actions"
    assert actions[0].type == "SET_VARIABLE", "First action should be SET_VARIABLE"
    assert actions[1].type == "SORT", "Second action should be SORT"

    # Get typed configs
    set_var_config = get_typed_config(actions[0])
    sort_config = get_typed_config(actions[1])

    print(f"SET_VARIABLE config type: {type(set_var_config).__name__}")
    print(f"SORT config type: {type(sort_config).__name__}")

    # Verify typed config properties
    assert isinstance(
        set_var_config, SetVariableActionConfig
    ), "Should be SetVariableActionConfig"
    assert set_var_config.variable_name == "total", "Variable name should be 'total'"

    assert isinstance(sort_config, SortActionConfig), "Should be SortActionConfig"
    assert sort_config.sort_by == "price", "Sort by should be 'price'"

    print("✓ JSON imported and validated successfully")


def test_loop_execution():
    """Test 3: Execute LOOP action."""
    print("\nExecuting LOOP action...")

    # Track executed actions
    executed_actions = []

    def mock_executor(action_id: str, variables: dict) -> dict:
        executed_actions.append({"action_id": action_id, "variables": variables.copy()})
        return {"success": True}

    # Create executor
    executor = ControlFlowExecutor(action_executor=mock_executor)

    # Create loop action
    loop_action = Action(
        id="test-loop",
        type="LOOP",
        config={
            "loopType": "FOR",
            "iterations": 5,
            "iteratorVariable": "i",
            "actions": ["process-item"],
        },
    )

    # Execute
    result = executor.execute_loop(loop_action)

    print(f"Loop completed: {result['iterations_completed']} iterations")
    print(f"Actions executed: {len(executed_actions)}")

    # Verify
    assert result["success"], "Loop should succeed"
    assert result["iterations_completed"] == 5, "Should complete 5 iterations"
    assert len(executed_actions) == 5, "Should execute action 5 times"

    # Verify iterator variable
    for i, execution in enumerate(executed_actions):
        assert execution["variables"]["i"] == i, f"Iterator should be {i}"

    print("✓ LOOP action executed successfully")


def test_if_else_execution():
    """Test 4: Execute IF/ELSE action."""
    print("\nExecuting IF/ELSE action...")

    executed_actions = []

    def mock_executor(action_id: str, variables: dict) -> dict:
        executed_actions.append(action_id)
        return {"success": True}

    # Create executor
    executor = ControlFlowExecutor(action_executor=mock_executor)

    # Test THEN branch
    print("\n  Testing THEN branch (condition true)...")
    executed_actions.clear()
    executor.variables = {"temperature": 25}

    if_action = Action(
        id="test-if",
        type="IF",
        config={
            "condition": {
                "type": "variable",
                "variableName": "temperature",
                "operator": ">",
                "expectedValue": 20,
            },
            "thenActions": ["turn-on-ac"],
            "elseActions": ["turn-on-heater"],
        },
    )

    result = executor.execute_if(if_action)

    print(f"    Condition result: {result['condition_result']}")
    print(f"    Branch taken: {result['branch_taken']}")
    print(f"    Actions executed: {executed_actions}")

    assert result["condition_result"], "Condition should be true"
    assert result["branch_taken"] == "then", "Should take THEN branch"
    assert "turn-on-ac" in executed_actions, "Should execute AC action"
    assert "turn-on-heater" not in executed_actions, "Should not execute heater"

    # Test ELSE branch
    print("\n  Testing ELSE branch (condition false)...")
    executed_actions.clear()
    executor.variables = {"temperature": 15}

    result = executor.execute_if(if_action)

    print(f"    Condition result: {result['condition_result']}")
    print(f"    Branch taken: {result['branch_taken']}")
    print(f"    Actions executed: {executed_actions}")

    assert not result["condition_result"], "Condition should be false"
    assert result["branch_taken"] == "else", "Should take ELSE branch"
    assert "turn-on-heater" in executed_actions, "Should execute heater action"
    assert "turn-on-ac" not in executed_actions, "Should not execute AC"

    print("✓ IF/ELSE action executed successfully")


def test_variable_operations():
    """Test 5: Execute variable SET/GET operations."""
    print("\nExecuting variable operations...")

    executor = DataOperationsExecutor()
    context = {}

    # SET_VARIABLE
    print("\n  Setting variable 'score' = 100...")
    set_action = Action(
        id="set-1",
        type="SET_VARIABLE",
        config={"variableName": "score", "value": 100, "scope": "local"},
    )

    result = executor.execute_set_variable(set_action, context)

    print(f"    Result: {result}")
    assert result["success"], "SET_VARIABLE should succeed"
    assert (
        executor.variable_context.get("score") == 100
    ), "Variable should be set to 100"

    # SET_VARIABLE with expression
    print("\n  Setting variable 'total' from expression 'score * 2'...")
    set_expr_action = Action(
        id="set-2",
        type="SET_VARIABLE",
        config={
            "variableName": "total",
            "valueSource": {"type": "expression", "expression": "score * 2"},
            "scope": "local",
        },
    )

    result = executor.execute_set_variable(set_expr_action, context)

    print(f"    Result: {result}")
    assert result["success"], "SET_VARIABLE with expression should succeed"
    assert executor.variable_context.get("total") == 200, "Total should be 200"

    # GET_VARIABLE
    print("\n  Getting variable 'score'...")
    get_action = Action(
        id="get-1", type="GET_VARIABLE", config={"variableName": "score"}
    )

    result = executor.execute_get_variable(get_action, context)

    print(f"    Result: {result}")
    assert result["success"], "GET_VARIABLE should succeed"
    assert result["value"] == 100, "Should get value 100"

    print("✓ Variable operations executed successfully")


def test_sort_operation():
    """Test 6: Execute SORT operation."""
    print("\nExecuting SORT operation...")

    executor = DataOperationsExecutor()
    context = {}

    # Set up test data
    items = [
        {"name": "sword", "price": 150},
        {"name": "shield", "price": 200},
        {"name": "potion", "price": 50},
        {"name": "armor", "price": 300},
    ]

    executor.variable_context.set("items", items, "local")
    print(f"  Original items: {[item['name'] for item in items]}")

    # Create SORT action
    sort_action = Action(
        id="sort-1",
        type="SORT",
        config={
            "target": "variable",
            "variableName": "items",
            "sortBy": "price",
            "order": "ASC",
            "comparator": "NUMERIC",
            "outputVariable": "sorted_items",
        },
    )

    # Execute
    result = executor.execute_sort(sort_action, context)

    sorted_items = result["sorted_collection"]
    print(f"  Sorted items: {[item['name'] for item in sorted_items]}")
    print(f"  Prices: {[item['price'] for item in sorted_items]}")

    # Verify
    assert result["success"], "SORT should succeed"
    assert len(sorted_items) == 4, "Should have 4 items"
    assert sorted_items[0]["name"] == "potion", "First should be potion (50)"
    assert sorted_items[1]["name"] == "sword", "Second should be sword (150)"
    assert sorted_items[2]["name"] == "shield", "Third should be shield (200)"
    assert sorted_items[3]["name"] == "armor", "Fourth should be armor (300)"

    print("✓ SORT operation executed successfully")


def test_filter_operation():
    """Test 7: Execute FILTER operation."""
    print("\nExecuting FILTER operation...")

    executor = DataOperationsExecutor()
    context = {}

    # Set up test data
    items = [
        {"name": "sword", "price": 150, "level": 5},
        {"name": "shield", "price": 200, "level": 3},
        {"name": "potion", "price": 50, "level": 1},
        {"name": "armor", "price": 300, "level": 7},
    ]

    executor.variable_context.set("items", items, "local")
    print(f"  Original items: {len(items)}")

    # Create FILTER action (price > 100)
    filter_action = Action(
        id="filter-1",
        type="FILTER",
        config={
            "variableName": "items",
            "condition": {
                "type": "property",
                "property": "price",
                "operator": ">",
                "value": 100,
            },
            "outputVariable": "expensive_items",
        },
    )

    # Execute
    result = executor.execute_filter(filter_action, context)

    filtered_items = result["filtered_collection"]
    print(f"  Filtered items (price > 100): {len(filtered_items)}")
    print(f"  Items: {[(item['name'], item['price']) for item in filtered_items]}")

    # Verify
    assert result["success"], "FILTER should succeed"
    assert len(filtered_items) == 3, "Should have 3 items"

    # All items should have price > 100
    for item in filtered_items:
        assert item["price"] > 100, f"{item['name']} should have price > 100"

    print("✓ FILTER operation executed successfully")


def test_complex_workflow():
    """Test 8: Execute complex workflow with multiple action types."""
    print("\nExecuting complex workflow...")

    # This simulates a real workflow:
    # 1. SET variables
    # 2. LOOP to process items
    # 3. IF to check conditions
    # 4. SORT results
    # 5. FILTER results

    data_executor = DataOperationsExecutor()

    executed_steps = []

    # Step 1: Set up initial data
    print("\n  Step 1: Setting up data...")
    items = [
        {"name": "item-A", "value": 75},
        {"name": "item-B", "value": 120},
        {"name": "item-C", "value": 45},
        {"name": "item-D", "value": 200},
        {"name": "item-E", "value": 90},
    ]

    data_executor.variable_context.set("items", items, "local")
    data_executor.variable_context.set("threshold", 80, "local")
    executed_steps.append("set_data")

    # Step 2: Filter items (value > threshold)
    print("\n  Step 2: Filtering items (value > 80)...")
    filter_action = Action(
        id="filter",
        type="FILTER",
        config={
            "variableName": "items",
            "condition": {
                "type": "expression",
                "expression": "item['value'] > threshold",
            },
            "outputVariable": "filtered_items",
        },
    )

    filter_result = data_executor.execute_filter(filter_action, {})
    filtered_items = filter_result["filtered_collection"]
    print(f"    Filtered to {len(filtered_items)} items")
    executed_steps.append("filter")

    # Step 3: Sort filtered items by value (descending)
    print("\n  Step 3: Sorting by value (descending)...")
    data_executor.variable_context.set("filtered_items", filtered_items, "local")

    sort_action = Action(
        id="sort",
        type="SORT",
        config={
            "target": "variable",
            "variableName": "filtered_items",
            "sortBy": "value",
            "order": "DESC",
            "comparator": "NUMERIC",
            "outputVariable": "sorted_items",
        },
    )

    sort_result = data_executor.execute_sort(sort_action, {})
    sorted_items = sort_result["sorted_collection"]
    print(
        f"    Sorted items: {[(item['name'], item['value']) for item in sorted_items]}"
    )
    executed_steps.append("sort")

    # Step 4: Store result
    print("\n  Step 4: Storing final result...")
    data_executor.variable_context.set("final_result", sorted_items, "global")
    executed_steps.append("store_result")

    # Verify workflow
    print(f"\n  Workflow steps executed: {executed_steps}")

    assert len(executed_steps) == 4, "Should execute 4 steps"
    assert len(sorted_items) == 3, "Should have 3 filtered items"
    assert sorted_items[0]["name"] == "item-D", "First should be item-D (200)"
    assert sorted_items[1]["name"] == "item-B", "Second should be item-B (120)"
    assert sorted_items[2]["name"] == "item-E", "Third should be item-E (90)"

    # Verify all values > 80
    for item in sorted_items:
        assert item["value"] > 80, f"{item['name']} value should be > 80"

    print("✓ Complex workflow executed successfully")


def test_new_format_validation():
    """Test 9: Verify new format validation works correctly."""
    print("\nTesting new format validation...")

    # New format JSON with structured config
    new_format_json = """
    [
        {
            "id": "new-click",
            "type": "CLICK",
            "config": {
                "target": {
                    "type": "image",
                    "imageId": "button-1"
                },
                "numberOfClicks": 1,
                "mouseButton": "LEFT"
            },
            "base": {
                "pauseBeforeBegin": 100
            },
            "execution": {
                "timeout": 5000
            }
        }
    ]
    """

    # Load and validate
    actions = load_actions_from_string(new_format_json)

    print(f"  Loaded {len(actions)} action(s) from new format")
    assert len(actions) == 1, "Should load 1 action"

    action = actions[0]
    print(f"  Action type: {action.type}")
    print(f"  Has base settings: {action.base is not None}")
    print(f"  Has execution settings: {action.execution is not None}")

    # Get typed config
    typed_config = get_typed_config(action)
    print(f"  Typed config type: {type(typed_config).__name__}")

    assert isinstance(typed_config, ClickActionConfig), "Should be ClickActionConfig"
    assert typed_config.target.type == "image", "Target type should be 'image'"
    assert typed_config.number_of_clicks == 1, "Should have 1 click"

    # Verify base and execution settings
    assert action.base.pause_before_begin == 100, "Should have pause of 100ms"
    assert action.execution.timeout == 5000, "Should have timeout of 5000ms"

    print("✓ New format validation successful")


def test_json_roundtrip():
    """Test 10: Create → Export → Import → Verify."""
    print("\nTesting JSON roundtrip...")

    # Create actions
    original_actions = [
        Action(
            id="action-1",
            type="LOOP",
            config={
                "loopType": "FOR",
                "iterations": 10,
                "iteratorVariable": "i",
                "actions": ["sub-action"],
            },
        ),
        Action(
            id="action-2",
            type="SET_VARIABLE",
            config={"variableName": "result", "value": 42, "scope": "global"},
        ),
    ]

    print(f"  Created {len(original_actions)} actions")

    # Export to JSON
    json_str = json.dumps(
        [action.model_dump() for action in original_actions], indent=2
    )
    print(f"  Exported to JSON ({len(json_str)} bytes)")

    # Import from JSON
    imported_actions = load_actions_from_string(json_str)
    print(f"  Imported {len(imported_actions)} actions")

    # Verify match
    assert len(imported_actions) == len(original_actions), "Should have same count"

    for i, (original, imported) in enumerate(
        zip(original_actions, imported_actions, strict=False)
    ):
        assert imported.id == original.id, f"Action {i} ID should match"
        assert imported.type == original.type, f"Action {i} type should match"

        # Get typed configs and compare
        original_config = get_typed_config(original)
        imported_config = get_typed_config(imported)

        assert type(original_config) is type(
            imported_config
        ), f"Action {i} config type should match"

        print(f"  ✓ Action {i} ({imported.type}) matches")

    print("✓ JSON roundtrip successful")


def main():
    """Run all integration tests."""
    print("=" * 70)
    print("QONTINUI INTEGRATION TESTS - NEW ACTION SCHEMA")
    print("=" * 70)

    runner = IntegrationTestRunner()

    # Run all tests
    runner.run_test(
        "Action Creation and Serialization", test_action_creation_and_serialization
    )
    runner.run_test("JSON Import and Validation", test_json_import_and_validation)
    runner.run_test("LOOP Execution", test_loop_execution)
    runner.run_test("IF/ELSE Execution", test_if_else_execution)
    runner.run_test("Variable Operations", test_variable_operations)
    runner.run_test("SORT Operation", test_sort_operation)
    runner.run_test("FILTER Operation", test_filter_operation)
    runner.run_test("Complex Workflow", test_complex_workflow)
    runner.run_test("New Format Validation", test_new_format_validation)
    runner.run_test("JSON Roundtrip", test_json_roundtrip)

    # Print summary
    runner.print_summary()

    # Exit with appropriate code
    return 0 if runner.failed_tests == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
