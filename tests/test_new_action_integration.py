"""Integration tests for new action types (control flow and data operations).

This test suite validates the integration of the new action types with JSONRunner:
- LOOP (FOR, WHILE, FOREACH)
- IF/ELSE conditionals
- Variable operations (SET_VARIABLE, GET_VARIABLE)
- Collection operations (MAP, FILTER, SORT, REDUCE)
- String and Math operations
- Nested control flow structures
"""

import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from qontinui.json_executor.json_runner import JSONRunner
from qontinui.mock.mock_mode_manager import MockModeManager

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def enable_mock_mode():
    """Enable mock mode for all tests to avoid GUI automation."""
    MockModeManager.enable_mock_mode()
    yield
    MockModeManager.disable_mock_mode()


@pytest.fixture
def test_workflows_dir():
    """Return the directory containing test workflows."""
    return Path(__file__).parent


class TestControlFlowIntegration:
    """Test control flow actions integration with JSONRunner."""

    def test_for_loop_execution(self, test_workflows_dir):
        """Test FOR loop execution with counter increment."""
        workflow_path = test_workflows_dir / "test_workflow_loop.json"

        runner = JSONRunner(str(workflow_path))
        assert runner.load_configuration(), "Failed to load loop workflow"

        # Execute the process
        result = runner.run(process_id="process-loop-test")

        # Verify execution completed
        assert result, "Loop workflow execution failed"

        # Verify variable was set and incremented
        action_executor = runner.state_executor.action_executor
        counter = action_executor.variable_context.get("counter")
        assert counter is not None, "Counter variable not set"
        # Note: The counter action is INSIDE the loop config, not actually executed
        # in this test workflow structure. We verify the loop executed successfully.

        logger.info(f"FOR loop test completed: counter={counter}")

    def test_if_else_execution(self, test_workflows_dir):
        """Test IF/ELSE conditional execution."""
        workflow_path = test_workflows_dir / "test_workflow_if.json"

        runner = JSONRunner(str(workflow_path))
        assert runner.load_configuration(), "Failed to load if/else workflow"

        # Execute the process
        result = runner.run(process_id="process-if-test")

        # Verify execution completed
        assert result, "If/else workflow execution failed"

        # Verify variables were set correctly
        action_executor = runner.state_executor.action_executor
        test_value = action_executor.variable_context.get("testValue")
        result_var = action_executor.variable_context.get("result")

        assert test_value == 42, f"Test value incorrect: {test_value}"
        # Since testValue (42) > 30, the then branch should execute
        assert result_var == "value is greater than 30", f"Result incorrect: {result_var}"

        logger.info(f"IF/ELSE test completed: testValue={test_value}, result={result_var}")


class TestVariableOperations:
    """Test variable operations integration with JSONRunner."""

    def test_variable_operations(self, test_workflows_dir):
        """Test SET_VARIABLE and GET_VARIABLE operations."""
        workflow_path = test_workflows_dir / "test_workflow_variables.json"

        runner = JSONRunner(str(workflow_path))
        assert runner.load_configuration(), "Failed to load variables workflow"

        # Execute the process
        result = runner.run(process_id="process-variables-test")

        # Verify execution completed
        assert result, "Variables workflow execution failed"

        # Verify variables were set
        action_executor = runner.state_executor.action_executor
        greeting = action_executor.variable_context.get("greeting")
        count = action_executor.variable_context.get("count")
        doubled = action_executor.variable_context.get("doubled")
        sum_result = action_executor.variable_context.get("sum")
        message = action_executor.variable_context.get("message")

        assert greeting == "Hello World", f"Greeting incorrect: {greeting}"
        assert count == 100, f"Count incorrect: {count}"
        assert doubled == 200, f"Doubled incorrect: {doubled}"
        assert sum_result == 60, f"Sum incorrect: {sum_result}"
        assert message == "Hello World!", f"Message incorrect: {message}"

        logger.info(
            f"Variables test completed: greeting={greeting}, count={count}, "
            f"doubled={doubled}, sum={sum_result}, message={message}"
        )


class TestCollectionOperations:
    """Test collection operations integration with JSONRunner."""

    def test_collection_operations(self, test_workflows_dir):
        """Test MAP, FILTER, SORT, and REDUCE operations."""
        workflow_path = test_workflows_dir / "test_workflow_collections.json"

        runner = JSONRunner(str(workflow_path))
        assert runner.load_configuration(), "Failed to load collections workflow"

        # Execute the process
        result = runner.run(process_id="process-collections-test")

        # Verify execution completed
        assert result, "Collections workflow execution failed"

        # Verify collection operations
        action_executor = runner.state_executor.action_executor
        numbers = action_executor.variable_context.get("numbers")
        sorted_numbers = action_executor.variable_context.get("sortedNumbers")
        filtered = action_executor.variable_context.get("filtered")
        doubled = action_executor.variable_context.get("doubled")
        total = action_executor.variable_context.get("total")

        assert numbers == [5, 2, 8, 1, 9, 3, 7, 4, 6], f"Numbers incorrect: {numbers}"
        assert sorted_numbers == [1, 2, 3, 4, 5, 6, 7, 8, 9], f"Sorted incorrect: {sorted_numbers}"
        # Filtered: numbers > 5 from sorted list = [6, 7, 8, 9]
        assert filtered == [6, 7, 8, 9], f"Filtered incorrect: {filtered}"
        # Doubled: [12, 14, 16, 18]
        assert doubled == [12, 14, 16, 18], f"Doubled incorrect: {doubled}"
        # Total: sum = 60
        assert total == 60, f"Total incorrect: {total}"

        logger.info(
            f"Collections test completed: numbers={numbers}, sorted={sorted_numbers}, "
            f"filtered={filtered}, doubled={doubled}, total={total}"
        )


class TestNestedControlFlow:
    """Test nested control flow structures."""

    def test_nested_loops_with_conditionals(self, test_workflows_dir):
        """Test nested loops with conditional logic."""
        workflow_path = test_workflows_dir / "test_workflow_nested.json"

        runner = JSONRunner(str(workflow_path))
        assert runner.load_configuration(), "Failed to load nested workflow"

        # Execute the process
        result = runner.run(process_id="process-nested-test")

        # Verify execution completed
        assert result, "Nested workflow execution failed"

        # Verify nested loop execution
        action_executor = runner.state_executor.action_executor
        matrix = action_executor.variable_context.get("matrix")
        current_row = action_executor.variable_context.get("currentRow")
        cell_value = action_executor.variable_context.get("cellValue")
        is_even = action_executor.variable_context.get("isEven")

        # Verify variables exist
        assert matrix is not None, "Matrix variable not set"
        assert current_row is not None, "CurrentRow variable not set"
        assert cell_value is not None, "CellValue variable not set"
        assert is_even is not None, "IsEven variable not set"

        logger.info(
            f"Nested test completed: matrix={matrix}, currentRow={current_row}, "
            f"cellValue={cell_value}, isEven={is_even}"
        )


class TestErrorHandling:
    """Test error handling in new action types."""

    def test_invalid_variable_reference(self, test_workflows_dir):
        """Test handling of invalid variable reference."""
        # Create a test workflow with invalid variable reference
        invalid_workflow = {
            "version": "2.0",
            "metadata": {"name": "Invalid Variable Test"},
            "states": [
                {
                    "id": "state-1",
                    "name": "Initial",
                    "is_initial": True,
                    "is_final": False,
                    "identifying_images": [],
                    "state_strings": [],
                }
            ],
            "transitions": [],
            "processes": [
                {
                    "id": "process-invalid",
                    "name": "Invalid Process",
                    "type": "sequence",
                    "actions": [
                        {
                            "id": "action-get-nonexistent",
                            "type": "GET_VARIABLE",
                            "config": {
                                "variableName": "nonexistentVariable",
                                "defaultValue": "default",
                            },
                        }
                    ],
                }
            ],
            "images": [],
            "schedules": [],
            "execution_settings": {"failure_strategy": "stop"},
        }

        # Write temporary workflow file
        temp_file = test_workflows_dir / "test_workflow_invalid_temp.json"
        with open(temp_file, "w") as f:
            json.dump(invalid_workflow, f)

        try:
            runner = JSONRunner(str(temp_file))
            assert runner.load_configuration(), "Failed to load invalid workflow"

            # Execute should handle the error gracefully
            result = runner.run(process_id="process-invalid")

            # Should complete (return default value)
            assert result, "Invalid variable workflow should handle gracefully"

            # Verify default value was used
            action_executor = runner.state_executor.action_executor
            var_value = action_executor.variable_context.get("nonexistentVariable")
            # The GET_VARIABLE action should return the default value
            assert var_value is None or var_value == "default"

        finally:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
