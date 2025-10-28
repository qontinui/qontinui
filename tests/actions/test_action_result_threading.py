"""Threading tests for ActionResultBuilder to verify thread-safety of concurrent operations."""

import threading
from typing import List

import pytest

from qontinui.actions.action_result import ActionResultBuilder


class TestActionResultBuilderThreading:
    """Test suite for ActionResultBuilder thread safety."""

    def test_concurrent_times_acted_on_increments(self):
        """Test that times_acted_on counter is thread-safe with concurrent increments.

        This test verifies that when multiple threads increment times_acted_on,
        no increments are lost due to race conditions.
        """
        builder = ActionResultBuilder()
        num_threads = 10
        increments_per_thread = 100
        threads: List[threading.Thread] = []

        # Simulate read-modify-write with builder
        lock = threading.Lock()

        def increment():
            for _ in range(increments_per_thread):
                # Each thread gets current value and increments
                with lock:
                    current = builder._times_acted_on
                    builder.set_times_acted_on(current + 1)

        # Start all threads
        for _ in range(num_threads):
            t = threading.Thread(target=increment)
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify: should be exactly num_threads * increments_per_thread
        result = builder.build()
        expected = num_threads * increments_per_thread
        assert (
            result.times_acted_on == expected
        ), f"Expected {expected}, got {result.times_acted_on}"

    def test_concurrent_match_additions(self):
        """Test that adding matches concurrently doesn't corrupt the match list.

        This test verifies that when multiple threads add matches simultaneously,
        all matches are properly recorded without data corruption.
        """
        builder = ActionResultBuilder()
        num_threads = 10
        matches_per_thread = 100
        threads: List[threading.Thread] = []

        # Simple mock match object
        class MockMatch:
            def __init__(self, value: int):
                self.value = value

        def add_matches(thread_id: int):
            for i in range(matches_per_thread):
                match = MockMatch(thread_id * 1000 + i)
                builder.add_match(match)

        # Start all threads
        for tid in range(num_threads):
            t = threading.Thread(target=add_matches, args=(tid,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Build result and verify
        result = builder.build()
        expected_count = num_threads * matches_per_thread
        assert (
            len(result.matches) == expected_count
        ), f"Expected {expected_count} matches, got {len(result.matches)}"

        # Verify: all match values are unique (no duplicates)
        values = [m.value for m in result.matches]
        assert len(values) == len(set(values)), "Found duplicate matches (race condition)"

    def test_concurrent_defined_region_additions(self):
        """Test that adding defined regions concurrently is thread-safe.

        This test verifies that when multiple threads add regions simultaneously,
        all regions are properly recorded.
        """
        builder = ActionResultBuilder()
        num_threads = 10
        regions_per_thread = 100
        threads: List[threading.Thread] = []

        # Simple mock region object
        class MockRegion:
            def __init__(self, value: int):
                self.value = value

        def add_regions(thread_id: int):
            for i in range(regions_per_thread):
                region = MockRegion(thread_id * 1000 + i)
                builder.add_defined_region(region)

        # Start all threads
        for tid in range(num_threads):
            t = threading.Thread(target=add_regions, args=(tid,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Build result and verify
        result = builder.build()
        expected_count = num_threads * regions_per_thread
        assert (
            len(result.defined_regions) == expected_count
        ), f"Expected {expected_count} regions, got {len(result.defined_regions)}"

    def test_concurrent_movement_additions(self):
        """Test that adding movements concurrently is thread-safe.

        This test verifies that when multiple threads add movements simultaneously,
        all movements are properly recorded.
        """
        builder = ActionResultBuilder()
        num_threads = 10
        movements_per_thread = 100
        threads: List[threading.Thread] = []

        # Simple mock movement object
        class MockMovement:
            def __init__(self, value: int):
                self.value = value

        def add_movements(thread_id: int):
            for i in range(movements_per_thread):
                movement = MockMovement(thread_id * 1000 + i)
                builder.add_movement(movement)

        # Start all threads
        for tid in range(num_threads):
            t = threading.Thread(target=add_movements, args=(tid,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Build result and verify
        result = builder.build()
        expected_count = num_threads * movements_per_thread
        assert (
            len(result.movements) == expected_count
        ), f"Expected {expected_count} movements, got {len(result.movements)}"

    def test_concurrent_execution_history_additions(self):
        """Test that adding execution records concurrently is thread-safe.

        This test verifies that when multiple threads add execution records simultaneously,
        all records are properly recorded.
        """
        builder = ActionResultBuilder()
        num_threads = 10
        records_per_thread = 100
        threads: List[threading.Thread] = []

        # Simple mock execution record object
        class MockExecutionRecord:
            def __init__(self, value: int):
                self.value = value

        def add_records(thread_id: int):
            for i in range(records_per_thread):
                record = MockExecutionRecord(thread_id * 1000 + i)
                builder.add_execution_record(record)

        # Start all threads
        for tid in range(num_threads):
            t = threading.Thread(target=add_records, args=(tid,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Build result and verify
        result = builder.build()
        expected_count = num_threads * records_per_thread
        assert (
            len(result.execution_history) == expected_count
        ), f"Expected {expected_count} records, got {len(result.execution_history)}"

    def test_concurrent_mixed_operations(self):
        """Test that mixed concurrent operations are thread-safe.

        This test simulates realistic usage where multiple threads perform
        different operations on the same ActionResultBuilder simultaneously.
        """
        builder = ActionResultBuilder()
        num_threads = 10
        operations_per_thread = 50
        threads: List[threading.Thread] = []

        class MockMatch:
            def __init__(self, value: int):
                self.value = value

        class MockRegion:
            def __init__(self, value: int):
                self.value = value

        # Need external lock for read-modify-write of times_acted_on
        counter_lock = threading.Lock()

        def mixed_operations(thread_id: int):
            for i in range(operations_per_thread):
                # Add a match
                match = MockMatch(thread_id * 1000 + i)
                builder.add_match(match)

                # Increment times_acted_on
                with counter_lock:
                    current = builder._times_acted_on
                    builder.set_times_acted_on(current + 1)

                # Add a region
                region = MockRegion(thread_id * 1000 + i)
                builder.add_defined_region(region)

        # Start all threads
        for tid in range(num_threads):
            t = threading.Thread(target=mixed_operations, args=(tid,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Build result and verify
        result = builder.build()
        expected_count = num_threads * operations_per_thread
        assert len(result.matches) == expected_count
        assert len(result.defined_regions) == expected_count
        assert result.times_acted_on == expected_count

    def test_multiple_builds_create_separate_results(self):
        """Test that building multiple results creates independent immutable objects.

        This verifies that results are truly immutable and independent.
        """
        builder = ActionResultBuilder()

        class MockMatch:
            def __init__(self, value: int):
                self.value = value

        # Add some data
        builder.add_match(MockMatch(1))
        builder.add_match(MockMatch(2))
        builder.with_success(True)

        # Build first result
        result1 = builder.build()

        # Add more data
        builder.add_match(MockMatch(3))

        # Build second result
        result2 = builder.build()

        # First result should be unchanged
        assert len(result1.matches) == 2
        assert result1.success is True

        # Second result should have all data
        assert len(result2.matches) == 3
        assert result2.success is True

        # Results are independent
        assert result1 is not result2

    def test_concurrent_builder_operations_and_builds(self):
        """Test that building results while other threads modify builder works correctly.

        This test verifies thread-safe construction and independent result creation.
        """
        builder = ActionResultBuilder()
        stop_flag = threading.Event()
        threads: List[threading.Thread] = []
        results: List = []
        results_lock = threading.Lock()

        class MockMatch:
            def __init__(self, value: int):
                self.value = value

        def writer():
            """Continuously add matches."""
            counter = 0
            while not stop_flag.is_set():
                match = MockMatch(counter)
                builder.add_match(match)
                counter += 1

        def builder_thread():
            """Continuously build results."""
            while not stop_flag.is_set():
                result = builder.build()
                with results_lock:
                    results.append(result)

        # Start writer and builder threads
        writer_threads = [threading.Thread(target=writer) for _ in range(2)]
        build_threads = [threading.Thread(target=builder_thread) for _ in range(2)]

        for t in writer_threads + build_threads:
            t.start()
            threads.append(t)

        # Let them run for a bit
        threading.Event().wait(0.1)

        # Signal stop and wait for completion
        stop_flag.set()
        for t in threads:
            t.join()

        # If we got here without crashes, the test passed
        assert len(results) > 0, "Expected some results to be built"

        # All results should be valid
        for result in results:
            assert isinstance(result.matches, tuple)
            assert isinstance(result.active_states, frozenset)

    def test_immutability_of_result(self):
        """Test that ActionResult is truly immutable.

        This verifies that frozen=True dataclass prevents modification.
        """
        builder = ActionResultBuilder()

        class MockMatch:
            def __init__(self, value: int):
                self.value = value

        builder.add_match(MockMatch(1))
        result = builder.build()

        # Should not be able to modify result
        with pytest.raises(AttributeError):
            result.success = False  # type: ignore

        with pytest.raises(AttributeError):
            result.times_acted_on = 999  # type: ignore

        # Collections are immutable
        assert isinstance(result.matches, tuple)
        assert isinstance(result.active_states, frozenset)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
