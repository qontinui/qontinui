"""Comprehensive integration tests for concurrent workflow execution.

Tests complete workflows running concurrently with ActionResult, StateRegistry,
and input operations all working together without race conditions.
"""

import threading
import time

import pytest

from qontinui.actions.action_result import ActionResult
from qontinui.actions.result_builder import ActionResultBuilder
from qontinui.annotations.enhanced_state import state
from qontinui.annotations.state_registry import StateRegistry
from qontinui.model.element.location import Location
from qontinui.model.element.region import Region


# Mock match class for testing
class MockMatch:
    """Mock Match object for testing."""

    def __init__(self, x: int, y: int, name: str):
        self.x = x
        self.y = y
        self.name = name
        self.region = Region(x, y, 100, 100)
        self.location = Location(x, y)
        self.times_acted_on = 0

    def set_times_acted_on(self, times: int):
        self.times_acted_on = times

    def __repr__(self):
        return f"MockMatch({self.name} at {self.x},{self.y})"


class TestConcurrentActionWorkflows:
    """Test complete workflows running concurrently."""

    def test_concurrent_action_workflows_basic(self):
        """10 threads executing action workflows concurrently."""
        results: list[ActionResult] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def execute_workflow(thread_id: int):
            """Execute a complete workflow."""
            try:
                # Create action result with matches
                builder = ActionResultBuilder()
                for i in range(5):
                    match = MockMatch(thread_id * 100 + i * 10, i * 10, f"match_{thread_id}_{i}")
                    builder.add_match(match)

                result = (
                    builder.with_success(True).with_description(f"Workflow {thread_id}").build()
                )

                # Register some states
                @state(name=f"state_workflow_{thread_id}_a")
                class StateA:
                    pass

                @state(name=f"state_workflow_{thread_id}_b")
                class StateB:
                    pass

                registry = StateRegistry()
                registry.register_state(StateA)
                registry.register_state(StateB)

                # Simulate some processing
                time.sleep(0.01)

                # Add to results
                with lock:
                    results.append(result)

            except Exception as e:
                with lock:
                    errors.append(e)

        # Start all threads
        threads = [threading.Thread(target=execute_workflow, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        assert all(r.success for r in results), "Not all workflows succeeded"
        assert all(len(r.match_list) == 5 for r in results), "Not all workflows have 5 matches"

    def test_concurrent_workflows_with_state_transitions(self):
        """Test workflows that register and transition states concurrently."""
        registries: list[StateRegistry] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def execute_state_workflow(thread_id: int):
            """Execute workflow with state transitions."""
            try:
                # Create registry
                registry = StateRegistry()

                # Define states
                @state(name=f"initial_state_{thread_id}", initial=True, profiles=["test"])
                class InitialState:
                    pass

                @state(name=f"processing_state_{thread_id}", group=f"group_{thread_id}")
                class ProcessingState:
                    pass

                @state(name=f"final_state_{thread_id}", group=f"group_{thread_id}")
                class FinalState:
                    pass

                # Register states
                registry.register_state(InitialState)
                registry.register_state(ProcessingState)
                registry.register_state(FinalState)

                # Get initial states
                initial = registry.get_initial_states("test")

                # Get group states
                group_states = registry.get_group_states(f"group_{thread_id}")

                # Verify
                assert len(initial) == 1
                assert len(group_states) == 2

                with lock:
                    registries.append(registry)

            except Exception as e:
                with lock:
                    errors.append(e)

        # Execute workflows
        threads = [threading.Thread(target=execute_state_workflow, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(registries) == 10
        for registry in registries:
            stats = registry.get_statistics()
            assert stats["total_states"] == 3
            assert stats["total_groups"] == 1
            assert stats["total_profiles"] == 1

    def test_concurrent_workflows_with_action_results(self):
        """Test workflows that build and merge action results concurrently."""
        final_result = ActionResult()
        errors: list[Exception] = []
        lock = threading.Lock()

        def execute_result_workflow(thread_id: int):
            """Execute workflow that creates and merges results."""
            try:
                # Create partial result
                builder = ActionResultBuilder()

                for i in range(3):
                    match = MockMatch(thread_id * 50 + i * 10, i * 5, f"result_{thread_id}_{i}")
                    builder.add_match(match)

                partial = builder.with_success(True).build()

                # Add a region
                region = Region(thread_id * 100, thread_id * 50, 50, 50)
                partial.add_defined_region(region)

                # Merge into final result
                with lock:
                    final_result.add_all_results(partial)

            except Exception as e:
                with lock:
                    errors.append(e)

        # Execute workflows
        threads = [threading.Thread(target=execute_result_workflow, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify merged result
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(final_result.match_list) == 30  # 10 threads * 3 matches
        assert len(final_result.defined_regions) == 10  # 10 threads * 1 region

    def test_concurrent_workflows_mixed_operations(self):
        """Test realistic workflows with mixed operations."""
        results: list[ActionResult] = []
        registries: list[StateRegistry] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def execute_complex_workflow(thread_id: int):
            """Execute complex workflow with multiple operations."""
            try:
                # 1. Create action result
                result = ActionResult()

                # 2. Create and register states
                @state(name=f"complex_state_{thread_id}_start", initial=True)
                class StartState:
                    pass

                @state(name=f"complex_state_{thread_id}_middle")
                class MiddleState:
                    pass

                @state(name=f"complex_state_{thread_id}_end")
                class EndState:
                    pass

                registry = StateRegistry()
                registry.register_state(StartState)
                registry.register_state(MiddleState)
                registry.register_state(EndState)

                # 3. Add matches to result
                for i in range(3):
                    match = MockMatch(i * 10, i * 20, f"complex_{thread_id}_{i}")
                    result.add(match)

                # 4. Update action counts
                result.set_times_acted_on(thread_id + 1)

                # 5. Add regions
                result.add_defined_region(Region(0, 0, 100, 100))

                # 6. Set success
                result.success = True

                # Store results
                with lock:
                    results.append(result)
                    registries.append(registry)

            except Exception as e:
                with lock:
                    errors.append(e)

        # Execute complex workflows
        threads = [threading.Thread(target=execute_complex_workflow, args=(i,)) for i in range(15)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 15
        assert len(registries) == 15

        for i, result in enumerate(results):
            assert result.success
            assert len(result.match_list) == 3
            assert result.times_acted_on == i + 1
            assert len(result.defined_regions) == 1

        for registry in registries:
            assert len(registry.states) == 3
            assert registry.next_state_id == 4  # 3 states registered

    def test_concurrent_workflows_with_result_properties(self):
        """Test concurrent access to result properties during modifications."""
        result = ActionResult()
        stop_flag = threading.Event()
        read_counts: list[int] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def writer():
            """Continuously add matches."""
            counter = 0
            while not stop_flag.is_set():
                try:
                    match = MockMatch(counter % 100, counter % 50, f"writer_{counter}")
                    result.add(match)
                    result.set_times_acted_on(counter)
                    counter += 1
                    time.sleep(0.001)
                except Exception as e:
                    with lock:
                        errors.append(e)
                    break

        def reader():
            """Continuously read properties."""
            count = 0
            while not stop_flag.is_set():
                try:
                    # Access properties - should never crash
                    _ = result.matches
                    _ = result.is_success
                    _ = len(result.match_list)
                    _ = result.times_acted_on
                    count += 1
                    time.sleep(0.001)
                except Exception as e:
                    with lock:
                        errors.append(e)
                    break

            with lock:
                read_counts.append(count)

        # Start threads
        writer_threads = [threading.Thread(target=writer) for _ in range(2)]
        reader_threads = [threading.Thread(target=reader) for _ in range(3)]

        for t in writer_threads + reader_threads:
            t.start()

        # Let them run
        time.sleep(0.2)
        stop_flag.set()

        # Wait for completion
        for t in writer_threads + reader_threads:
            t.join()

        # Verify
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(read_counts) == 3
        assert all(count > 0 for count in read_counts)
        assert len(result.match_list) > 0


class TestConcurrentStateManagement:
    """Test state management in concurrent workflows."""

    def test_concurrent_state_registration_across_workflows(self):
        """Multiple workflows registering states in shared registry."""
        shared_registry = StateRegistry()
        errors: list[Exception] = []
        lock = threading.Lock()

        def register_workflow_states(workflow_id: int):
            """Register states for a workflow."""
            try:
                for state_num in range(5):
                    state_name = f"workflow_{workflow_id}_state_{state_num}"

                    @state(name=state_name, group=f"workflow_{workflow_id}")
                    class DynamicState:
                        pass

                    shared_registry.register_state(DynamicState)

            except Exception as e:
                with lock:
                    errors.append(e)

        # Execute registrations
        threads = [threading.Thread(target=register_workflow_states, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(shared_registry.states) == 100  # 20 workflows * 5 states
        assert len(shared_registry.groups) == 20

        # Verify state IDs are unique
        all_ids = list(shared_registry.state_ids.values())
        assert len(all_ids) == len(set(all_ids)), "Duplicate state IDs found"

    def test_workflow_state_isolation(self):
        """Each workflow has its own registry - no interference."""
        registries: list[StateRegistry] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def isolated_workflow(workflow_id: int):
            """Workflow with isolated state registry."""
            try:
                # Each workflow creates its own registry
                registry = StateRegistry()

                # Register states
                @state(name=f"isolated_{workflow_id}_a")
                class StateA:
                    pass

                @state(name=f"isolated_{workflow_id}_b")
                class StateB:
                    pass

                registry.register_state(StateA)
                registry.register_state(StateB)

                # Verify isolation
                assert len(registry.states) == 2
                assert registry.next_state_id == 3

                with lock:
                    registries.append(registry)

            except Exception as e:
                with lock:
                    errors.append(e)

        # Execute workflows
        threads = [threading.Thread(target=isolated_workflow, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(registries) == 20

        # Each registry should be independent
        for registry in registries:
            assert len(registry.states) == 2
            assert registry.next_state_id == 3


class TestConcurrentResultOperations:
    """Test concurrent operations on ActionResult."""

    def test_concurrent_result_building(self):
        """Multiple threads building results simultaneously."""
        results: list[ActionResult] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def build_result(builder_id: int):
            """Build a result with multiple operations."""
            try:
                builder = ActionResultBuilder()

                # Add matches
                for i in range(10):
                    match = MockMatch(i * 5, i * 10, f"build_{builder_id}_{i}")
                    builder.add_match(match)

                # Build result
                result = (
                    builder.with_success(True)
                    .with_description(f"Builder {builder_id}")
                    .with_times_acted_on(builder_id)
                    .build()
                )

                with lock:
                    results.append(result)

            except Exception as e:
                with lock:
                    errors.append(e)

        # Build results concurrently
        threads = [threading.Thread(target=build_result, args=(i,)) for i in range(25)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 25

        for _i, result in enumerate(results):
            assert result.success
            assert len(result.match_list) == 10

    def test_concurrent_result_merging(self):
        """Test merging results from multiple threads."""
        master_result = ActionResult()
        errors: list[Exception] = []
        lock = threading.Lock()

        def create_and_merge(thread_id: int):
            """Create result and merge into master."""
            try:
                # Create partial result
                partial = ActionResult()

                for i in range(5):
                    match = MockMatch(thread_id * 20 + i, i, f"merge_{thread_id}_{i}")
                    partial.add(match)

                partial.success = True

                # Merge into master
                with lock:
                    master_result.add_all_results(partial)

            except Exception as e:
                with lock:
                    errors.append(e)

        # Merge from multiple threads
        threads = [threading.Thread(target=create_and_merge, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(master_result.match_list) == 100  # 20 threads * 5 matches


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
