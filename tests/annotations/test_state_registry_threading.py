"""Threading tests for StateRegistry to verify thread-safety of concurrent state registration."""

import threading

import pytest

from qontinui.annotations.enhanced_state import state
from qontinui.annotations.state_registry import RegistryFrozenError, StateRegistry
from qontinui.state_exceptions import (
    StateAlreadyExistsException,
    StateNotFoundException,
)


class TestStateRegistryThreading:
    """Test suite for StateRegistry thread safety."""

    def test_concurrent_state_registration_no_duplicates(self):
        """Test that concurrent state registration doesn't create duplicate IDs.

        This is the critical race condition: when multiple threads try to register
        states simultaneously, the check-then-act pattern could cause duplicate IDs
        or lost registrations. The lock should prevent this.
        """
        registry = StateRegistry()
        num_threads = 10
        states_per_thread = 10
        threads: list[threading.Thread] = []
        all_state_ids: list[int] = []
        lock = threading.Lock()

        def register_states(thread_id: int):
            """Each thread registers multiple unique states."""
            for i in range(states_per_thread):
                # Create a unique state class for this thread and iteration

                @state(name=f"state_{thread_id}_{i}")
                class DynamicState:
                    pass

                # Register and collect the ID
                state_id = registry.register_state(DynamicState)
                with lock:
                    all_state_ids.append(state_id)

        # Start all threads
        for tid in range(num_threads):
            t = threading.Thread(target=register_states, args=(tid,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify: all IDs should be unique (no duplicates)
        expected_count = num_threads * states_per_thread
        assert len(all_state_ids) == expected_count, (
            f"Expected {expected_count} state IDs, got {len(all_state_ids)}"
        )

        unique_ids = set(all_state_ids)
        assert len(unique_ids) == expected_count, (
            f"Found duplicate IDs! Expected {expected_count} unique IDs, got {len(unique_ids)}"
        )

        # Verify: IDs should be sequential from 1 to expected_count
        assert unique_ids == set(range(1, expected_count + 1)), (
            f"IDs are not sequential: {sorted(unique_ids)}"
        )

        # Verify: registry state count matches
        assert len(registry.states) == expected_count, (
            f"Registry has {len(registry.states)} states, expected {expected_count}"
        )

    def test_concurrent_same_state_registration(self):
        """Test that registering the same state concurrently is idempotent.

        When multiple threads try to register the same state, only one should
        succeed and all should get the same ID back.
        """
        registry = StateRegistry()
        num_threads = 20
        threads: list[threading.Thread] = []
        collected_ids: list[int] = []
        lock = threading.Lock()

        # Define a single state that all threads will try to register
        @state(name="shared_state")
        class SharedState:
            pass

        def register_same_state():
            """Multiple threads register the same state."""
            state_id = registry.register_state(SharedState)
            with lock:
                collected_ids.append(state_id)

        # Start all threads
        for _ in range(num_threads):
            t = threading.Thread(target=register_same_state)
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify: all threads should get the same ID
        assert len(collected_ids) == num_threads
        unique_ids = set(collected_ids)
        assert len(unique_ids) == 1, f"Expected 1 unique ID, got {len(unique_ids)}: {unique_ids}"

        # Verify: state is only registered once
        assert len(registry.states) == 1
        assert "shared_state" in registry.states

    def test_concurrent_state_id_counter_integrity(self):
        """Test that the next_state_id counter maintains integrity under load.

        This verifies that the counter increments correctly without skipping
        or reusing numbers when states are registered concurrently.
        """
        registry = StateRegistry()
        num_threads = 10
        states_per_thread = 50
        threads: list[threading.Thread] = []

        def register_states(thread_id: int):
            """Each thread registers unique states."""
            for i in range(states_per_thread):

                @state(name=f"counter_test_state_{thread_id}_{i}")
                class CounterTestState:
                    pass

                registry.register_state(CounterTestState)

        # Start all threads
        for tid in range(num_threads):
            t = threading.Thread(target=register_states, args=(tid,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify: final counter value should be total_states + 1
        expected_count = num_threads * states_per_thread
        expected_next_id = expected_count + 1
        assert registry.next_state_id == expected_next_id, (
            f"Expected next_state_id={expected_next_id}, got {registry.next_state_id}"
        )

        # Verify: all assigned IDs are in the correct range
        all_ids = set(registry.state_ids.values())
        assert all_ids == set(range(1, expected_count + 1)), "State IDs are not in expected range"

    def test_concurrent_state_registration_with_groups(self):
        """Test that group registration is thread-safe.

        When multiple threads register states in the same group concurrently,
        all states should be properly added to the group without loss.
        """
        registry = StateRegistry()
        num_threads = 10
        states_per_thread = 10
        threads: list[threading.Thread] = []

        def register_states_with_group(thread_id: int):
            """Each thread registers states in the same group."""
            for i in range(states_per_thread):

                @state(name=f"grouped_state_{thread_id}_{i}", group="test_group")
                class GroupedState:
                    pass

                registry.register_state(GroupedState)

        # Start all threads
        for tid in range(num_threads):
            t = threading.Thread(target=register_states_with_group, args=(tid,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify: all states are in the group
        expected_count = num_threads * states_per_thread
        assert "test_group" in registry.groups
        assert len(registry.groups["test_group"]) == expected_count, (
            f"Expected {expected_count} states in group, got {len(registry.groups['test_group'])}"
        )

    def test_concurrent_state_registration_with_profiles(self):
        """Test that profile registration is thread-safe.

        When multiple threads register initial states in profiles concurrently,
        all states should be properly added without loss.
        """
        registry = StateRegistry()
        num_threads = 10
        states_per_thread = 10
        threads: list[threading.Thread] = []

        def register_initial_states(thread_id: int):
            """Each thread registers initial states."""
            for i in range(states_per_thread):

                @state(
                    name=f"initial_state_{thread_id}_{i}",
                    initial=True,
                    profiles=["test_profile"],
                )
                class InitialState:
                    pass

                registry.register_state(InitialState)

        # Start all threads
        for tid in range(num_threads):
            t = threading.Thread(target=register_initial_states, args=(tid,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify: all states are in the profile
        expected_count = num_threads * states_per_thread
        assert "test_profile" in registry.profiles
        assert len(registry.profiles["test_profile"]) == expected_count, (
            f"Expected {expected_count} states in profile, got {len(registry.profiles['test_profile'])}"
        )

    def test_freeze_prevents_registration(self):
        """Test that freezing registry prevents further registrations.

        This verifies the lifecycle: construction → frozen.
        """
        registry = StateRegistry()

        @state(name="state1")
        class State1:
            pass

        # Register before freeze
        state_id = registry.register_state(State1)
        assert state_id == 1

        # Freeze registry
        registry.freeze()
        assert registry.is_frozen()

        # Try to register after freeze
        @state(name="state2")
        class State2:
            pass

        with pytest.raises(RegistryFrozenError):
            registry.register_state(State2)

    def test_concurrent_freeze_and_register(self):
        """Test that concurrent freeze and register operations work correctly.

        This tests the race between freezing and registering.
        """
        registry = StateRegistry()
        num_threads = 10
        threads: list[threading.Thread] = []
        exceptions: list[Exception] = []
        exceptions_lock = threading.Lock()
        success_count = [0]
        success_lock = threading.Lock()

        def register_state_safe(thread_id: int):
            """Try to register state, catch exceptions."""
            try:

                @state(name=f"state_{thread_id}")
                class DynamicState:
                    pass

                registry.register_state(DynamicState)
                with success_lock:
                    success_count[0] += 1
            except RegistryFrozenError as e:
                with exceptions_lock:
                    exceptions.append(e)

        def freeze_registry():
            """Freeze the registry."""
            threading.Event().wait(0.005)  # Small delay to allow some registrations
            registry.freeze()

        # Start register threads
        for tid in range(num_threads):
            t = threading.Thread(target=register_state_safe, args=(tid,))
            threads.append(t)
            t.start()

        # Start freeze thread
        freeze_thread = threading.Thread(target=freeze_registry)
        freeze_thread.start()
        threads.append(freeze_thread)

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify: registry is frozen
        assert registry.is_frozen()

        # Verify: Total should match threads
        assert success_count[0] + len(exceptions) == num_threads

        # With the timing, we expect both some successes and some failures
        # but this is a race condition test, so we just verify all threads completed

    def test_get_state_raises_exception_for_missing(self):
        """Test that get_state raises exception instead of returning None."""
        registry = StateRegistry()

        with pytest.raises(StateNotFoundException):
            registry.get_state("nonexistent")

    def test_get_state_id_raises_exception_for_missing(self):
        """Test that get_state_id raises exception instead of returning None."""
        registry = StateRegistry()

        with pytest.raises(StateNotFoundException):
            registry.get_state_id("nonexistent")

    def test_get_state_by_id_raises_exception_for_missing(self):
        """Test that get_state_by_id raises exception instead of returning None."""
        registry = StateRegistry()

        with pytest.raises(StateNotFoundException):
            registry.get_state_by_id(999)

    def test_has_state_for_existence_check(self):
        """Test that has_state provides safe existence check."""
        registry = StateRegistry()

        @state(name="test_state")
        class TestState:
            pass

        # Before registration
        assert not registry.has_state("test_state")

        # After registration
        registry.register_state(TestState)
        assert registry.has_state("test_state")

    def test_clear_on_frozen_registry_raises_error(self):
        """Test that clearing a frozen registry raises error."""
        registry = StateRegistry()

        @state(name="state1")
        class State1:
            pass

        registry.register_state(State1)
        registry.freeze()

        with pytest.raises(RegistryFrozenError):
            registry.clear()

    def test_concurrent_clear_and_register(self):
        """Test that concurrent clear and register operations don't cause corruption.

        This tests a more complex scenario where some threads are clearing
        while others are registering.
        """
        registry = StateRegistry()
        num_iterations = 50
        threads: list[threading.Thread] = []
        stop_flag = threading.Event()

        def register_continuously():
            """Continuously register states."""
            counter = 0
            while not stop_flag.is_set():

                @state(name=f"continuous_state_{counter}_{threading.get_ident()}")
                class ContinuousState:
                    pass

                try:
                    registry.register_state(ContinuousState)
                    counter += 1
                except (RegistryFrozenError, StateAlreadyExistsException):
                    # May fail if cleared or frozen during registration, that's ok
                    pass

        def clear_periodically():
            """Periodically clear the registry."""
            for _ in range(num_iterations):
                threading.Event().wait(0.001)  # Small delay
                try:
                    registry.clear()
                except RegistryFrozenError:
                    # Registry was frozen, that's ok
                    pass

        # Start threads
        register_threads = [threading.Thread(target=register_continuously) for _ in range(3)]
        clear_thread = threading.Thread(target=clear_periodically)

        for t in register_threads:
            t.start()
            threads.append(t)

        clear_thread.start()
        clear_thread.join()

        # Signal stop and wait for register threads
        stop_flag.set()
        for t in register_threads:
            t.join()

        # If we got here without crashes or exceptions, the test passed
        # The final state may vary, but should be consistent
        assert registry.next_state_id >= 1

    def test_concurrent_mixed_operations(self):
        """Test that mixed concurrent operations maintain data integrity.

        This simulates realistic usage where multiple threads perform
        different registry operations simultaneously.
        """
        registry = StateRegistry()
        num_threads = 5
        operations_per_thread = 20
        threads: list[threading.Thread] = []

        def mixed_operations(thread_id: int):
            """Each thread performs various registry operations."""
            for i in range(operations_per_thread):
                # Register a state
                @state(name=f"mixed_state_{thread_id}_{i}")
                class MixedState:
                    pass

                state_id = registry.register_state(MixedState)

                # Query operations - use has_state to check first
                if registry.has_state(f"mixed_state_{thread_id}_{i}"):
                    _ = registry.get_state(f"mixed_state_{thread_id}_{i}")
                    _ = registry.get_state_id(f"mixed_state_{thread_id}_{i}")
                    _ = registry.get_state_by_id(state_id)
                _ = registry.get_statistics()

        # Start all threads
        for tid in range(num_threads):
            t = threading.Thread(target=mixed_operations, args=(tid,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify: all states were registered correctly
        expected_count = num_threads * operations_per_thread
        assert len(registry.states) == expected_count
        assert len(registry.state_ids) == expected_count
        assert registry.next_state_id == expected_count + 1

    def test_duplicate_state_name_different_class_raises_error(self):
        """Test that registering different classes with same name raises error."""
        registry = StateRegistry()

        @state(name="duplicate")
        class State1:
            pass

        @state(name="duplicate")
        class State2:
            pass

        # First registration succeeds
        registry.register_state(State1)

        # Second registration with same name but different class fails
        with pytest.raises(StateAlreadyExistsException):
            registry.register_state(State2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
