"""Threading tests for ExecutionContext.

Tests thread safety of ExecutionContext under concurrent access.
"""

import sys
import threading
import time
from pathlib import Path

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from qontinui.orchestration.execution_context import (
    ExecutionContext,
)


class TestExecutionContextThreading:
    """Test thread safety of ExecutionContext."""

    def test_concurrent_variable_modifications(self):
        """Test concurrent variable set/get operations."""
        context = ExecutionContext()
        num_threads = 10
        iterations = 100
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(iterations):
                    # Set thread-specific variable
                    context.set_variable(f"var_{thread_id}", i)
                    # Read it back
                    value = context.get_variable(f"var_{thread_id}")
                    # Value should match what we just set
                    if value != i:
                        errors.append(f"Thread {thread_id}: Expected {i}, got {value}")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Check all variables were set correctly
        assert len(errors) == 0, f"Errors occurred: {errors}"
        for i in range(num_threads):
            var_name = f"var_{i}"
            assert context.has_variable(var_name)
            assert context.get_variable(var_name) == iterations - 1

    def test_concurrent_reads_during_writes(self):
        """Test concurrent reads while variables are being written."""
        context = ExecutionContext()
        context.set_variable("counter", 0)
        num_readers = 5
        num_writers = 5
        iterations = 50
        read_errors = []
        write_errors = []

        def reader(reader_id: int):
            try:
                for _ in range(iterations):
                    # Read all variables
                    variables = context.variables
                    # Verify counter exists and is non-negative
                    counter = variables.get("counter", -1)
                    if counter < 0:
                        read_errors.append(f"Reader {reader_id}: Invalid counter {counter}")
                    time.sleep(0.001)
            except Exception as e:
                read_errors.append(f"Reader {reader_id}: {e}")

        def writer(writer_id: int):
            try:
                for i in range(iterations):
                    # Increment counter
                    current = context.get_variable("counter", 0)
                    context.set_variable("counter", current + 1)
                    # Set writer-specific variable
                    context.set_variable(f"writer_{writer_id}", i)
                    time.sleep(0.001)
            except Exception as e:
                write_errors.append(f"Writer {writer_id}: {e}")

        threads = []
        # Start readers
        for i in range(num_readers):
            t = threading.Thread(target=reader, args=(i,))
            threads.append(t)
            t.start()

        # Start writers
        for i in range(num_writers):
            t = threading.Thread(target=writer, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(read_errors) == 0, f"Read errors: {read_errors}"
        assert len(write_errors) == 0, f"Write errors: {write_errors}"

    def test_concurrent_action_tracking(self):
        """Test concurrent action start/complete operations."""
        context = ExecutionContext()
        num_threads = 10
        actions_per_thread = 50
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(actions_per_thread):
                    action_index = thread_id * actions_per_thread + i
                    state = context.start_action(action_index, f"action_{thread_id}_{i}")
                    # Simulate some work
                    time.sleep(0.001)
                    # Complete action
                    success = i % 2 == 0  # Alternate success/failure
                    error = ValueError(f"Error {i}") if not success else None
                    context.complete_action(state, success=success, error=error)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify statistics
        stats = context.statistics
        expected_total = num_threads * actions_per_thread
        assert stats.total_actions == expected_total
        # Half should be successful, half failed (due to i % 2 == 0)
        assert stats.successful_actions == expected_total // 2
        assert stats.failed_actions == expected_total // 2

        # Verify action states
        action_states = context.action_states
        assert len(action_states) == expected_total

    def test_stress_mixed_operations(self):
        """Stress test with mixed operations from multiple threads."""
        context = ExecutionContext(initial_variables={"shared_counter": 0})
        num_threads = 10
        iterations = 100
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(iterations):
                    # Mix different operations
                    if i % 4 == 0:
                        # Variable operations
                        context.set_variable(f"thread_{thread_id}", i)
                        _ = context.get_variable(f"thread_{thread_id}")
                    elif i % 4 == 1:
                        # Action operations
                        state = context.start_action(thread_id * iterations + i, f"action_{i}")
                        context.complete_action(state, success=True)
                    elif i % 4 == 2:
                        # Metadata operations
                        context.set_metadata(f"meta_{thread_id}", i)
                        _ = context.get_metadata(f"meta_{thread_id}")
                    else:
                        # Read operations
                        _ = context.variables
                        _ = context.action_states
                        _ = context.statistics
                        _ = context.metadata
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_no_data_corruption_verification(self):
        """Verify no data corruption occurs under concurrent modifications."""
        context = ExecutionContext()
        num_threads = 10
        iterations = 100
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(iterations):
                    # Set a specific pattern
                    key = f"thread_{thread_id}"
                    value = [thread_id] * 10  # List with thread_id repeated
                    context.set_variable(key, value)

                    # Read it back immediately
                    retrieved = context.get_variable(key)

                    # Verify integrity
                    if retrieved != value:
                        errors.append(
                            f"Thread {thread_id} iteration {i}: "
                            f"Data corruption detected. Expected {value}, got {retrieved}"
                        )
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Data corruption errors: {errors}"

        # Verify final state
        for i in range(num_threads):
            key = f"thread_{i}"
            value = context.get_variable(key)
            expected = [i] * 10
            assert value == expected, f"Final state corrupted for {key}"

    def test_concurrent_retry_tracking(self):
        """Test concurrent retry recording."""
        context = ExecutionContext()
        num_threads = 10
        retries_per_thread = 20
        errors = []

        def worker(thread_id: int):
            try:
                state = context.start_action(thread_id, f"action_{thread_id}")
                for _ in range(retries_per_thread):
                    context.record_retry(state)
                    time.sleep(0.001)
                context.complete_action(state, success=True)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify retry statistics
        stats = context.statistics
        expected_total_retries = num_threads * retries_per_thread
        assert stats.total_retries == expected_total_retries
        # Each thread had one action that was retried
        assert stats.retried_actions == num_threads

    def test_concurrent_workflow_lifecycle(self):
        """Test concurrent workflow start/complete operations."""
        context = ExecutionContext()
        num_threads = 10
        errors = []

        def worker(thread_id: int):
            try:
                # Multiple threads trying to mark workflow start/complete
                if thread_id % 2 == 0:
                    context.start_workflow()
                else:
                    time.sleep(0.01)
                    context.complete_workflow()
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify workflow was marked
        stats = context.statistics
        assert stats.start_time is not None
        assert stats.end_time is not None

    def test_concurrent_variable_substitution(self):
        """Test concurrent variable substitution operations."""
        context = ExecutionContext()
        context.set_variable("name", "test")
        context.set_variable("count", 42)
        num_threads = 10
        iterations = 50
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(iterations):
                    # Concurrent substitutions
                    result = context.substitute_variables("Hello ${name}, count is ${count}")
                    if result != "Hello test, count is 42":
                        errors.append(
                            f"Thread {thread_id}: Unexpected substitution result: {result}"
                        )

                    # Also modify variables
                    if i % 10 == 0:
                        context.set_variable(f"thread_{thread_id}", i)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_concurrent_get_failed_actions(self):
        """Test concurrent access to failed actions list."""
        context = ExecutionContext()
        num_threads = 10
        actions_per_thread = 20
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(actions_per_thread):
                    state = context.start_action(
                        thread_id * actions_per_thread + i, f"action_{thread_id}_{i}"
                    )
                    # Half succeed, half fail
                    success = i % 2 == 0
                    error = ValueError(f"Error {i}") if not success else None
                    context.complete_action(state, success=success, error=error)

                    # Concurrently read failed actions
                    if i % 5 == 0:
                        failed = context.get_failed_actions()
                        # Just verify it doesn't crash
                        _ = len(failed)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify failed actions count
        failed = context.get_failed_actions()
        expected_failures = (num_threads * actions_per_thread) // 2
        assert len(failed) == expected_failures

    def test_concurrent_clear_variables(self):
        """Test concurrent variable clear operations."""
        context = ExecutionContext()
        num_threads = 5
        iterations = 20
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(iterations):
                    # Set variables
                    context.set_variable(f"var_{thread_id}_{i}", i)
                    time.sleep(0.001)

                    # Occasionally clear all
                    if thread_id == 0 and i % 10 == 0:
                        context.clear_variables()
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        # No assertion on final state as clear is called concurrently
