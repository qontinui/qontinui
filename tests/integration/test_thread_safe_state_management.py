"""Comprehensive thread safety tests for state management.

Stress tests for StateRegistry with concurrent registration, transitions,
and mixed read/write operations with 100 threads.
"""

import threading
import time

import pytest

from qontinui.annotations.enhanced_state import state
from qontinui.annotations.state_registry import StateRegistry


class TestStateRegistryStressTests:
    """Stress tests for StateRegistry with high concurrency."""

    def test_stress_concurrent_state_registration_100_threads(self):
        """Stress test: 100 threads registering states concurrently."""
        registry = StateRegistry()
        num_threads = 100
        states_per_thread = 10
        errors: list[Exception] = []
        all_state_ids: list[int] = []
        lock = threading.Lock()

        def register_states(thread_id: int):
            """Register multiple states."""
            try:
                thread_ids: list[int] = []
                for i in range(states_per_thread):

                    @state(name=f"stress_state_{thread_id}_{i}")
                    class StressState:
                        pass

                    state_id = registry.register_state(StressState)
                    thread_ids.append(state_id)

                # Collect IDs atomically
                with lock:
                    all_state_ids.extend(thread_ids)

            except Exception as e:
                with lock:
                    errors.append(e)

        # Start all threads
        threads = [threading.Thread(target=register_states, args=(i,)) for i in range(num_threads)]
        start_time = time.time()

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        elapsed = time.time() - start_time

        # Verify results
        expected_count = num_threads * states_per_thread
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(all_state_ids) == expected_count
        assert len(set(all_state_ids)) == expected_count, "Duplicate state IDs found"
        assert len(registry.states) == expected_count
        assert registry.next_state_id == expected_count + 1

        print(f"Stress test: {expected_count} states registered in {elapsed:.2f}s")

    def test_stress_concurrent_state_transitions_100_threads(self):
        """Stress test: 100 threads performing state lookups concurrently."""
        registry = StateRegistry()

        # Pre-register states
        states_to_register = []
        for i in range(100):

            @state(name=f"lookup_state_{i}", group=f"group_{i % 10}")
            class LookupState:
                pass

            states_to_register.append(LookupState)

        for s in states_to_register:
            registry.register_state(s)

        # Now perform concurrent lookups
        num_threads = 100
        lookups_per_thread = 100
        errors: list[Exception] = []
        lock = threading.Lock()

        def perform_lookups(thread_id: int):
            """Perform multiple state lookups."""
            try:
                for i in range(lookups_per_thread):
                    state_name = f"lookup_state_{i % 100}"

                    # Various lookup operations
                    state_class = registry.get_state(state_name)
                    state_id = registry.get_state_id(state_name)
                    state_by_id = registry.get_state_by_id(state_id) if state_id else None

                    # Verify
                    assert state_class is not None
                    assert state_id is not None
                    assert state_by_id is not None

            except Exception as e:
                with lock:
                    errors.append(e)

        # Execute lookups
        threads = [threading.Thread(target=perform_lookups, args=(i,)) for i in range(num_threads)]
        start_time = time.time()

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        elapsed = time.time() - start_time
        total_lookups = num_threads * lookups_per_thread

        # Verify
        assert len(errors) == 0, f"Errors occurred: {errors}"
        print(f"Stress test: {total_lookups} lookups in {elapsed:.2f}s")

    def test_stress_mixed_read_write_operations(self):
        """Stress test: Mixed read and write operations."""
        registry = StateRegistry()
        stop_flag = threading.Event()
        errors: list[Exception] = []
        stats = {"registrations": 0, "lookups": 0}
        lock = threading.Lock()

        # Pre-register some states
        for i in range(20):

            @state(name=f"initial_state_{i}")
            class InitialState:
                pass

            registry.register_state(InitialState)

        def writer_worker(worker_id: int):
            """Continuously register new states."""
            counter = 0
            while not stop_flag.is_set():
                try:

                    @state(name=f"dynamic_state_{worker_id}_{counter}")
                    class DynamicState:
                        pass

                    registry.register_state(DynamicState)

                    with lock:
                        stats["registrations"] += 1

                    counter += 1
                    time.sleep(0.001)

                except Exception as e:
                    with lock:
                        errors.append(e)
                    break

        def reader_worker():
            """Continuously read state information."""
            while not stop_flag.is_set():
                try:
                    # Get statistics
                    _ = registry.get_statistics()

                    # List states
                    state_list = list(registry.states.keys())

                    # Lookup random states
                    if state_list:
                        for state_name in state_list[:10]:
                            _ = registry.get_state(state_name)
                            _ = registry.get_state_id(state_name)

                    with lock:
                        stats["lookups"] += 1

                    time.sleep(0.001)

                except Exception as e:
                    with lock:
                        errors.append(e)
                    break

        # Start workers
        writers = [threading.Thread(target=writer_worker, args=(i,)) for i in range(10)]
        readers = [threading.Thread(target=reader_worker) for _ in range(20)]

        all_threads = writers + readers
        for t in all_threads:
            t.start()

        # Let them run
        time.sleep(1.0)
        stop_flag.set()

        # Wait for completion
        for t in all_threads:
            t.join()

        # Verify
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert stats["registrations"] > 0
        assert stats["lookups"] > 0
        print(
            f"Mixed operations: {stats['registrations']} registrations, {stats['lookups']} lookups"
        )


class TestConcurrentGroupOperations:
    """Test concurrent group operations."""

    def test_concurrent_group_registration(self):
        """Multiple threads registering states in the same groups."""
        registry = StateRegistry()
        num_threads = 50
        states_per_thread = 10
        num_groups = 5
        errors: list[Exception] = []
        lock = threading.Lock()

        def register_grouped_states(thread_id: int):
            """Register states in groups."""
            try:
                for i in range(states_per_thread):
                    group_name = f"group_{i % num_groups}"

                    @state(name=f"grouped_{thread_id}_{i}", group=group_name)
                    class GroupedState:
                        pass

                    registry.register_state(GroupedState)

            except Exception as e:
                with lock:
                    errors.append(e)

        # Execute registrations
        threads = [
            threading.Thread(target=register_grouped_states, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(registry.groups) == num_groups

        # Verify each group has correct number of states
        total_states = num_threads * states_per_thread
        states_per_group = total_states // num_groups

        for i in range(num_groups):
            group_name = f"group_{i}"
            assert group_name in registry.groups
            assert len(registry.groups[group_name]) == states_per_group

    def test_concurrent_group_queries(self):
        """Multiple threads querying group states concurrently."""
        registry = StateRegistry()

        # Pre-register states in groups
        for i in range(100):

            @state(name=f"query_state_{i}", group=f"group_{i % 10}")
            class QueryState:
                pass

            registry.register_state(QueryState)

        # Concurrent queries
        errors: list[Exception] = []
        query_counts: list[int] = []
        lock = threading.Lock()

        def query_groups():
            """Query group states."""
            try:
                count = 0
                for i in range(100):
                    group_states = registry.get_group_states(f"group_{i % 10}")
                    assert len(group_states) == 10
                    count += 1

                with lock:
                    query_counts.append(count)

            except Exception as e:
                with lock:
                    errors.append(e)

        # Execute queries
        threads = [threading.Thread(target=query_groups) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify
        assert len(errors) == 0
        assert len(query_counts) == 50
        assert all(c == 100 for c in query_counts)


class TestConcurrentProfileOperations:
    """Test concurrent profile operations."""

    def test_concurrent_initial_state_registration(self):
        """Multiple threads registering initial states in profiles."""
        registry = StateRegistry()
        num_threads = 40
        profiles = ["profile_a", "profile_b", "profile_c", "profile_d"]
        errors: list[Exception] = []
        lock = threading.Lock()

        def register_initial_states(thread_id: int):
            """Register initial states."""
            try:
                for i, profile in enumerate(profiles):

                    @state(
                        name=f"initial_{thread_id}_{i}",
                        initial=True,
                        profiles=[profile],
                        priority=thread_id,
                    )
                    class InitialState:
                        pass

                    registry.register_state(InitialState)

            except Exception as e:
                with lock:
                    errors.append(e)

        # Execute registrations
        threads = [
            threading.Thread(target=register_initial_states, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify
        assert len(errors) == 0
        assert len(registry.profiles) == len(profiles)

        # Each profile should have num_threads states
        for profile in profiles:
            assert profile in registry.profiles
            assert len(registry.profiles[profile]) == num_threads

    def test_concurrent_profile_queries(self):
        """Multiple threads querying initial states for profiles."""
        registry = StateRegistry()

        # Pre-register initial states
        for i in range(50):

            @state(name=f"profile_state_{i}", initial=True, profiles=["test_profile"], priority=i)
            class ProfileState:
                pass

            registry.register_state(ProfileState)

        # Concurrent queries
        errors: list[Exception] = []
        results: list[int] = []
        lock = threading.Lock()

        def query_initial_states():
            """Query initial states."""
            try:
                initial = registry.get_initial_states("test_profile")
                with lock:
                    results.append(len(initial))

            except Exception as e:
                with lock:
                    errors.append(e)

        # Execute queries
        threads = [threading.Thread(target=query_initial_states) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify
        assert len(errors) == 0
        assert all(r == 50 for r in results)


class TestStateRegistryDataIntegrity:
    """Test data integrity under concurrent load."""

    def test_state_id_uniqueness_under_load(self):
        """Verify state IDs remain unique under heavy concurrent load."""
        registry = StateRegistry()
        num_threads = 100
        all_ids: set[int] = set()
        errors: list[Exception] = []
        lock = threading.Lock()

        def register_and_collect_id(thread_id: int):
            """Register state and collect ID."""
            try:

                @state(name=f"unique_id_state_{thread_id}")
                class UniqueState:
                    pass

                state_id = registry.register_state(UniqueState)

                with lock:
                    if state_id in all_ids:
                        errors.append(ValueError(f"Duplicate ID: {state_id}"))
                    all_ids.add(state_id)

            except Exception as e:
                with lock:
                    errors.append(e)

        # Execute
        threads = [
            threading.Thread(target=register_and_collect_id, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(all_ids) == num_threads
        assert all_ids == set(range(1, num_threads + 1))

    def test_state_count_consistency(self):
        """Verify state counts remain consistent."""
        registry = StateRegistry()
        num_threads = 50
        states_per_thread = 20
        errors: list[Exception] = []
        lock = threading.Lock()

        def register_states(thread_id: int):
            """Register states."""
            try:
                for i in range(states_per_thread):

                    @state(name=f"count_state_{thread_id}_{i}")
                    class CountState:
                        pass

                    registry.register_state(CountState)

            except Exception as e:
                with lock:
                    errors.append(e)

        # Execute
        threads = [threading.Thread(target=register_states, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify counts
        expected = num_threads * states_per_thread
        assert len(errors) == 0
        assert len(registry.states) == expected
        assert len(registry.state_ids) == expected
        assert registry.next_state_id == expected + 1

    def test_no_data_corruption_under_load(self):
        """Verify no data corruption occurs under heavy load."""
        registry = StateRegistry()
        num_threads = 50
        errors: list[Exception] = []
        lock = threading.Lock()

        def complex_operations(thread_id: int):
            """Perform various operations."""
            try:
                # Register states
                @state(name=f"corruption_test_{thread_id}_a", group="group_a")
                class StateA:
                    pass

                @state(name=f"corruption_test_{thread_id}_b", initial=True, profiles=["test"])
                class StateB:
                    pass

                id_a = registry.register_state(StateA)
                id_b = registry.register_state(StateB)

                # Verify immediately
                assert registry.get_state_id(f"corruption_test_{thread_id}_a") == id_a
                assert registry.get_state_id(f"corruption_test_{thread_id}_b") == id_b

                # Query operations
                _ = registry.get_state(f"corruption_test_{thread_id}_a")
                _ = registry.get_group_states("group_a")
                _ = registry.get_initial_states("test")
                _ = registry.get_statistics()

            except Exception as e:
                with lock:
                    errors.append(e)

        # Execute
        threads = [
            threading.Thread(target=complex_operations, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no errors
        assert len(errors) == 0, f"Data corruption detected: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
