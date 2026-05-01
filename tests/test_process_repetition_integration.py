"""Integration tests for process repetition in JSON executor."""

import sys
from dataclasses import dataclass

sys.path.insert(0, "/home/jspinak/qontinui_parent_directory/qontinui/src")

from qontinui.json_executor.action_executor import ActionExecutor
from qontinui.json_executor.config_parser import Action, Process, QontinuiConfig


@dataclass
class MockImage:
    """Mock image for testing."""

    id: str
    file_path: str | None = None


@dataclass
class MockState:
    """Mock state for testing."""

    id: str
    name: str


class TestProcessRepetitionIntegration:
    """Test process repetition in JSON executor."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock config
        self.mock_image = MockImage(id="test_image", file_path=None)
        self.mock_state = MockState(id="state1", name="State 1")

        self.config = QontinuiConfig(
            states=[self.mock_state],
            transitions=[],
            processes=[],
            images=[self.mock_image],
            state_map={"state1": self.mock_state},
            transition_map={},
            process_map={},
            image_map={"test_image": self.mock_image},
        )

    def test_no_repetition_executes_once(self):
        """Test that process without repetition executes once."""
        # Create a simple process with one WAIT action
        wait_action = Action(
            id="wait1",
            type="WAIT",
            config={"duration": 100},
            timeout=5000,
            retry_count=1,
            continue_on_error=False,
        )

        test_process = Process(
            id="test_process",
            name="Test Process",
            description="Test",
            type="sequence",
            actions=[wait_action],
        )

        self.config.processes.append(test_process)
        self.config.process_map[test_process.id] = test_process

        # Create RUN_PROCESS action without repetition
        run_process_action = Action(
            id="run1",
            type="RUN_PROCESS",
            config={"process": "test_process"},
            timeout=10000,
            retry_count=1,
            continue_on_error=False,
        )

        executor = ActionExecutor(self.config)

        # Track execution count
        original_wait = executor._execute_wait
        execution_count = {"count": 0}

        def counted_wait(action):
            execution_count["count"] += 1
            return original_wait(action)

        executor._execute_wait = counted_wait

        # Execute
        result = executor._execute_run_process(run_process_action)

        # Verify
        assert result is True
        assert execution_count["count"] == 1, "Should execute once without repetition"

    def test_fixed_count_repetition(self):
        """Test fixed count repetition (run all repeats)."""
        # Create a process that always succeeds
        wait_action = Action(
            id="wait1",
            type="WAIT",
            config={"duration": 50},
            timeout=5000,
            retry_count=1,
            continue_on_error=False,
        )

        test_process = Process(
            id="test_process",
            name="Test Process",
            description="Test",
            type="sequence",
            actions=[wait_action],
        )

        self.config.processes.append(test_process)
        self.config.process_map[test_process.id] = test_process

        # Create RUN_PROCESS with fixed count repetition
        run_process_action = Action(
            id="run1",
            type="RUN_PROCESS",
            config={
                "process": "test_process",
                "processRepetition": {
                    "enabled": True,
                    "maxRepeats": 3,
                    "delay": 0,
                    "untilSuccess": False,
                },
            },
            timeout=10000,
            retry_count=1,
            continue_on_error=False,
        )

        executor = ActionExecutor(self.config)

        # Track execution count
        original_wait = executor._execute_wait
        execution_count = {"count": 0}

        def counted_wait(action):
            execution_count["count"] += 1
            return original_wait(action)

        executor._execute_wait = counted_wait

        # Execute
        result = executor._execute_run_process(run_process_action)

        # Verify
        assert result is True
        assert (
            execution_count["count"] == 4
        ), "Should execute 4 times (1 initial + 3 repeats)"

    def test_until_success_stops_early(self):
        """Test until_success mode stops on first success."""
        # Create a process that fails first 2 times, then succeeds
        test_process = Process(
            id="test_process",
            name="Test Process",
            description="Test",
            type="sequence",
            actions=[],
        )

        self.config.processes.append(test_process)
        self.config.process_map[test_process.id] = test_process

        # Create RUN_PROCESS with until_success repetition
        run_process_action = Action(
            id="run1",
            type="RUN_PROCESS",
            config={
                "process": "test_process",
                "processRepetition": {
                    "enabled": True,
                    "maxRepeats": 10,
                    "delay": 0,
                    "untilSuccess": True,
                },
            },
            timeout=10000,
            retry_count=1,
            continue_on_error=False,
        )

        executor = ActionExecutor(self.config)

        # Mock _execute_process_once to fail twice then succeed
        execution_count = {"count": 0}

        def mock_execute_once(process, process_id, run_num, total_runs):
            execution_count["count"] += 1
            # Fail first 2 times, succeed on 3rd
            return execution_count["count"] >= 3

        executor._execute_process_once = mock_execute_once

        # Execute
        result = executor._execute_run_process(run_process_action)

        # Verify
        assert result is True
        assert (
            execution_count["count"] == 3
        ), "Should stop after 3rd attempt (first success)"

    def test_until_success_reaches_max_repeats(self):
        """Test until_success mode uses all attempts if never succeeds."""
        # Create a process that always fails
        test_process = Process(
            id="test_process",
            name="Test Process",
            description="Test",
            type="sequence",
            actions=[],
        )

        self.config.processes.append(test_process)
        self.config.process_map[test_process.id] = test_process

        # Create RUN_PROCESS with until_success repetition
        run_process_action = Action(
            id="run1",
            type="RUN_PROCESS",
            config={
                "process": "test_process",
                "processRepetition": {
                    "enabled": True,
                    "maxRepeats": 5,
                    "delay": 0,
                    "untilSuccess": True,
                },
            },
            timeout=10000,
            retry_count=1,
            continue_on_error=False,
        )

        executor = ActionExecutor(self.config)

        # Mock _execute_process_once to always fail
        execution_count = {"count": 0}

        def mock_execute_once(process, process_id, run_num, total_runs):
            execution_count["count"] += 1
            return False  # Always fail

        executor._execute_process_once = mock_execute_once

        # Execute
        result = executor._execute_run_process(run_process_action)

        # Verify
        assert result is False
        assert (
            execution_count["count"] == 6
        ), "Should try 6 times (1 initial + 5 repeats)"

    def test_fixed_count_success_if_any_succeed(self):
        """Test fixed count mode succeeds if at least one run succeeds."""
        # Create a process
        test_process = Process(
            id="test_process",
            name="Test Process",
            description="Test",
            type="sequence",
            actions=[],
        )

        self.config.processes.append(test_process)
        self.config.process_map[test_process.id] = test_process

        # Create RUN_PROCESS with fixed count repetition
        run_process_action = Action(
            id="run1",
            type="RUN_PROCESS",
            config={
                "process": "test_process",
                "processRepetition": {
                    "enabled": True,
                    "maxRepeats": 4,
                    "delay": 0,
                    "untilSuccess": False,
                },
            },
            timeout=10000,
            retry_count=1,
            continue_on_error=False,
        )

        executor = ActionExecutor(self.config)

        # Mock _execute_process_once: fail, fail, succeed, fail, fail
        execution_count = {"count": 0}

        def mock_execute_once(process, process_id, run_num, total_runs):
            execution_count["count"] += 1
            return execution_count["count"] == 3  # Succeed only on 3rd run

        executor._execute_process_once = mock_execute_once

        # Execute
        result = executor._execute_run_process(run_process_action)

        # Verify
        assert result is True, "Should succeed if at least one run succeeded"
        assert execution_count["count"] == 5, "Should execute all 5 runs"

    def test_json_config_parsing(self):
        """Test that JSON config is properly parsed and executed."""
        # Create a simple process
        wait_action = Action(
            id="wait1",
            type="WAIT",
            config={"duration": 50},
            timeout=5000,
            retry_count=1,
            continue_on_error=False,
        )

        test_process = Process(
            id="test_process",
            name="Test Process",
            description="Test",
            type="sequence",
            actions=[wait_action],
        )

        self.config.processes.append(test_process)
        self.config.process_map[test_process.id] = test_process

        # Test JSON-like config with milliseconds for delay
        run_process_action = Action(
            id="run1",
            type="RUN_PROCESS",
            config={
                "process": "test_process",
                "processRepetition": {
                    "enabled": True,
                    "maxRepeats": 2,
                    "delay": 100,  # milliseconds in JSON
                    "untilSuccess": False,
                },
            },
            timeout=10000,
            retry_count=1,
            continue_on_error=False,
        )

        executor = ActionExecutor(self.config)

        # Track executions
        original_wait = executor._execute_wait
        execution_count = {"count": 0}

        def counted_wait(action):
            execution_count["count"] += 1
            return original_wait(action)

        executor._execute_wait = counted_wait

        # Execute
        result = executor._execute_run_process(run_process_action)

        # Verify
        assert result is True
        assert (
            execution_count["count"] == 3
        ), "Should execute 3 times (1 initial + 2 repeats)"


def main():
    """Run tests manually."""
    import traceback

    print("=" * 60)
    print("Process Repetition Integration Tests")
    print("=" * 60)

    test_suite = TestProcessRepetitionIntegration()

    tests = [
        ("No repetition executes once", test_suite.test_no_repetition_executes_once),
        ("Fixed count repetition", test_suite.test_fixed_count_repetition),
        ("Until success stops early", test_suite.test_until_success_stops_early),
        (
            "Until success reaches max repeats",
            test_suite.test_until_success_reaches_max_repeats,
        ),
        (
            "Fixed count success if any succeed",
            test_suite.test_fixed_count_success_if_any_succeed,
        ),
        ("JSON config parsing", test_suite.test_json_config_parsing),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\nTesting: {test_name}")
        test_suite.setup_method()
        try:
            test_func()
            print("  ✓ PASSED")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
