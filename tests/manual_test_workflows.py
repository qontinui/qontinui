#!/usr/bin/env python
"""Manual test script to verify new action integration.

This script runs the test workflows directly without pytest to validate
the integration works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qontinui.json_executor.json_runner import JSONRunner
from qontinui.mock.mock_mode_manager import MockModeManager


def test_loop_workflow():
    """Test the FOR loop workflow."""
    print("\n" + "=" * 70)
    print("TEST: FOR Loop Workflow")
    print("=" * 70)

    MockModeManager.enable_mock_mode()

    workflow_path = Path(__file__).parent / "test_workflow_loop.json"
    runner = JSONRunner(str(workflow_path))

    if not runner.load_configuration():
        print("FAILED: Could not load loop workflow")
        return False

    result = runner.run(process_id="process-loop-test")

    if result:
        print("\nSUCCESS: Loop workflow executed successfully")
        # Check variables
        action_executor = runner.state_executor.action_executor
        counter = action_executor.variable_context.get("counter")
        print(f"  Counter value: {counter}")
        return True
    else:
        print("\nFAILED: Loop workflow execution failed")
        return False


def test_if_workflow():
    """Test the IF/ELSE workflow."""
    print("\n" + "=" * 70)
    print("TEST: IF/ELSE Workflow")
    print("=" * 70)

    MockModeManager.enable_mock_mode()

    workflow_path = Path(__file__).parent / "test_workflow_if.json"
    runner = JSONRunner(str(workflow_path))

    if not runner.load_configuration():
        print("FAILED: Could not load if/else workflow")
        return False

    result = runner.run(process_id="process-if-test")

    if result:
        print("\nSUCCESS: If/else workflow executed successfully")
        # Check variables
        action_executor = runner.state_executor.action_executor
        test_value = action_executor.variable_context.get("testValue")
        result_var = action_executor.variable_context.get("result")
        print(f"  Test value: {test_value}")
        print(f"  Result: {result_var}")
        return True
    else:
        print("\nFAILED: If/else workflow execution failed")
        return False


def test_variables_workflow():
    """Test variable operations workflow."""
    print("\n" + "=" * 70)
    print("TEST: Variable Operations Workflow")
    print("=" * 70)

    MockModeManager.enable_mock_mode()

    workflow_path = Path(__file__).parent / "test_workflow_variables.json"
    runner = JSONRunner(str(workflow_path))

    if not runner.load_configuration():
        print("FAILED: Could not load variables workflow")
        return False

    result = runner.run(process_id="process-variables-test")

    if result:
        print("\nSUCCESS: Variables workflow executed successfully")
        # Check variables
        action_executor = runner.state_executor.action_executor
        greeting = action_executor.variable_context.get("greeting")
        count = action_executor.variable_context.get("count")
        doubled = action_executor.variable_context.get("doubled")
        sum_result = action_executor.variable_context.get("sum")
        message = action_executor.variable_context.get("message")
        print(f"  Greeting: {greeting}")
        print(f"  Count: {count}")
        print(f"  Doubled: {doubled}")
        print(f"  Sum: {sum_result}")
        print(f"  Message: {message}")
        return True
    else:
        print("\nFAILED: Variables workflow execution failed")
        return False


def test_collections_workflow():
    """Test collection operations workflow."""
    print("\n" + "=" * 70)
    print("TEST: Collection Operations Workflow")
    print("=" * 70)

    MockModeManager.enable_mock_mode()

    workflow_path = Path(__file__).parent / "test_workflow_collections.json"
    runner = JSONRunner(str(workflow_path))

    if not runner.load_configuration():
        print("FAILED: Could not load collections workflow")
        return False

    result = runner.run(process_id="process-collections-test")

    if result:
        print("\nSUCCESS: Collections workflow executed successfully")
        # Check variables
        action_executor = runner.state_executor.action_executor
        numbers = action_executor.variable_context.get("numbers")
        sorted_numbers = action_executor.variable_context.get("sortedNumbers")
        filtered = action_executor.variable_context.get("filtered")
        doubled = action_executor.variable_context.get("doubled")
        total = action_executor.variable_context.get("total")
        print(f"  Original: {numbers}")
        print(f"  Sorted: {sorted_numbers}")
        print(f"  Filtered (>5): {filtered}")
        print(f"  Doubled: {doubled}")
        print(f"  Total: {total}")
        return True
    else:
        print("\nFAILED: Collections workflow execution failed")
        return False


def main():
    """Run all manual tests."""
    print("\n" + "=" * 70)
    print("MANUAL WORKFLOW INTEGRATION TESTS")
    print("=" * 70)

    results = []
    tests = [
        ("Loop", test_loop_workflow),
        ("If/Else", test_if_workflow),
        ("Variables", test_variables_workflow),
        ("Collections", test_collections_workflow),
    ]

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\nEXCEPTION in {name} test: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {name}: {status}")

    total = len(results)
    passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 70 + "\n")

    return all(success for _, success in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
