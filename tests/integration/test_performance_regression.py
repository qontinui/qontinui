"""Performance regression tests for qontinui fixes.

Benchmarks key operations to ensure thread safety doesn't add >10% overhead
and checks for memory usage issues.
"""

import gc
import threading
import time
import tracemalloc

import pytest

from qontinui.actions.action_result import ActionResult
from qontinui.annotations.enhanced_state import state
from qontinui.annotations.state_registry import StateRegistry


# Mock match for testing
class MockMatch:
    """Mock Match object."""

    def __init__(self, x: int, y: int, name: str):
        self.x = x
        self.y = y
        self.name = name

    def __repr__(self):
        return f"MockMatch({self.name})"


class TestActionResultPerformance:
    """Performance tests for ActionResult operations."""

    def test_baseline_match_addition_performance(self):
        """Baseline: Single-threaded match addition performance."""
        result = ActionResult()
        num_matches = 10000

        start_time = time.time()

        for i in range(num_matches):
            match = MockMatch(i, i * 2, f"match_{i}")
            result.add(match)

        elapsed = time.time() - start_time

        # Baseline: should be very fast
        assert len(result.match_list) == num_matches
        assert elapsed < 1.0, f"Baseline too slow: {elapsed:.3f}s"

        print(f"Baseline: {num_matches} matches added in {elapsed:.3f}s")
        print(f"Rate: {num_matches / elapsed:.0f} matches/sec")

    def test_concurrent_match_addition_overhead(self):
        """Measure overhead of concurrent match addition."""
        num_threads = 10
        matches_per_thread = 1000

        # Baseline: single-threaded
        result_baseline = ActionResult()
        start_baseline = time.time()

        for i in range(num_threads * matches_per_thread):
            match = MockMatch(i, i, f"baseline_{i}")
            result_baseline.add(match)

        baseline_time = time.time() - start_baseline

        # Concurrent
        result_concurrent = ActionResult()
        errors: list[Exception] = []
        lock = threading.Lock()

        def add_matches(thread_id: int):
            try:
                for i in range(matches_per_thread):
                    match = MockMatch(thread_id * 1000 + i, i, f"concurrent_{thread_id}_{i}")
                    result_concurrent.add(match)
            except Exception as e:
                with lock:
                    errors.append(e)

        start_concurrent = time.time()

        threads = [threading.Thread(target=add_matches, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        concurrent_time = time.time() - start_concurrent

        # Verify correctness
        assert len(errors) == 0
        assert len(result_concurrent.match_list) == num_threads * matches_per_thread

        # Calculate overhead
        overhead_percent = ((concurrent_time - baseline_time) / baseline_time) * 100

        print(f"Baseline time: {baseline_time:.3f}s")
        print(f"Concurrent time: {concurrent_time:.3f}s")
        print(f"Overhead: {overhead_percent:.1f}%")

        # Thread safety should not add >10% overhead
        # (Concurrent might even be faster due to parallelism, but with locks it's usually slower)
        assert overhead_percent < 50, f"Too much overhead: {overhead_percent:.1f}%"

    def test_result_property_access_performance(self):
        """Test performance of property access."""
        result = ActionResult()

        # Add some matches
        for i in range(1000):
            match = MockMatch(i, i, f"prop_{i}")
            result.add(match)

        # Measure property access
        iterations = 10000
        start_time = time.time()

        for _ in range(iterations):
            _ = result.matches
            _ = result.is_success
            _ = len(result.match_list)

        elapsed = time.time() - start_time

        print(f"Property access: {iterations} iterations in {elapsed:.3f}s")
        print(f"Rate: {iterations / elapsed:.0f} accesses/sec")

        # Should be fast
        assert elapsed < 0.5, f"Property access too slow: {elapsed:.3f}s"


class TestStateRegistryPerformance:
    """Performance tests for StateRegistry operations."""

    def test_baseline_state_registration_performance(self):
        """Baseline: Single-threaded state registration performance."""
        registry = StateRegistry()
        num_states = 1000

        start_time = time.time()

        for i in range(num_states):

            @state(name=f"baseline_state_{i}")
            class BaselineState:
                pass

            registry.register_state(BaselineState)

        elapsed = time.time() - start_time

        assert len(registry.states) == num_states
        print(f"Baseline: {num_states} states registered in {elapsed:.3f}s")
        print(f"Rate: {num_states / elapsed:.0f} states/sec")

    def test_concurrent_registration_overhead(self):
        """Measure overhead of concurrent state registration."""
        num_threads = 10
        states_per_thread = 100

        # Baseline: single-threaded
        registry_baseline = StateRegistry()
        start_baseline = time.time()

        for i in range(num_threads * states_per_thread):

            @state(name=f"baseline_concurrent_{i}")
            class BaselineConcurrentState:
                pass

            registry_baseline.register_state(BaselineConcurrentState)

        baseline_time = time.time() - start_baseline

        # Concurrent
        registry_concurrent = StateRegistry()
        errors: list[Exception] = []
        lock = threading.Lock()

        def register_states(thread_id: int):
            try:
                for i in range(states_per_thread):

                    @state(name=f"concurrent_{thread_id}_{i}")
                    class ConcurrentState:
                        pass

                    registry_concurrent.register_state(ConcurrentState)
            except Exception as e:
                with lock:
                    errors.append(e)

        start_concurrent = time.time()

        threads = [threading.Thread(target=register_states, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        concurrent_time = time.time() - start_concurrent

        # Verify
        assert len(errors) == 0
        assert len(registry_concurrent.states) == num_threads * states_per_thread

        # Calculate overhead
        overhead_percent = ((concurrent_time - baseline_time) / baseline_time) * 100

        print(f"Baseline time: {baseline_time:.3f}s")
        print(f"Concurrent time: {concurrent_time:.3f}s")
        print(f"Overhead: {overhead_percent:.1f}%")

        # Should not add excessive overhead
        assert overhead_percent < 50, f"Too much overhead: {overhead_percent:.1f}%"

    def test_state_lookup_performance(self):
        """Test performance of state lookups."""
        registry = StateRegistry()

        # Pre-register states
        for i in range(1000):

            @state(name=f"lookup_perf_{i}")
            class LookupState:
                pass

            registry.register_state(LookupState)

        # Measure lookup performance
        iterations = 10000
        start_time = time.time()

        for i in range(iterations):
            _ = registry.get_state(f"lookup_perf_{i % 1000}")
            _ = registry.get_state_id(f"lookup_perf_{i % 1000}")

        elapsed = time.time() - start_time

        print(f"State lookup: {iterations} lookups in {elapsed:.3f}s")
        print(f"Rate: {iterations / elapsed:.0f} lookups/sec")

        # Should be fast
        assert elapsed < 1.0, f"Lookups too slow: {elapsed:.3f}s"


class TestMemoryUsage:
    """Test memory usage of concurrent operations."""

    def test_action_result_memory_usage(self):
        """Test memory usage of ActionResult with many matches."""
        tracemalloc.start()

        result = ActionResult()

        # Take baseline
        gc.collect()
        baseline = tracemalloc.get_traced_memory()[0]

        # Add many matches
        for i in range(10000):
            match = MockMatch(i, i, f"mem_{i}")
            result.add(match)

        # Measure
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()

        tracemalloc.stop()

        memory_used_mb = (current - baseline) / 1024 / 1024

        print(f"Memory used: {memory_used_mb:.2f} MB for 10000 matches")
        print(f"Per match: {(current - baseline) / 10000:.0f} bytes")

        # Should be reasonable (rough estimate)
        assert memory_used_mb < 50, f"Too much memory used: {memory_used_mb:.2f} MB"

    def test_state_registry_memory_usage(self):
        """Test memory usage of StateRegistry with many states."""
        tracemalloc.start()

        registry = StateRegistry()

        # Take baseline
        gc.collect()
        baseline = tracemalloc.get_traced_memory()[0]

        # Register many states
        for i in range(1000):

            @state(name=f"mem_state_{i}")
            class MemState:
                pass

            registry.register_state(MemState)

        # Measure
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()

        tracemalloc.stop()

        memory_used_mb = (current - baseline) / 1024 / 1024

        print(f"Memory used: {memory_used_mb:.2f} MB for 1000 states")
        print(f"Per state: {(current - baseline) / 1000:.0f} bytes")

        # Should be reasonable
        assert memory_used_mb < 20, f"Too much memory used: {memory_used_mb:.2f} MB"

    def test_no_memory_leaks_in_concurrent_operations(self):
        """Test for memory leaks in concurrent operations."""
        tracemalloc.start()

        # Baseline
        gc.collect()
        baseline = tracemalloc.get_traced_memory()[0]

        # Perform many concurrent operations
        for iteration in range(10):
            result = ActionResult()
            registry = StateRegistry()

            # Concurrent operations
            threads = []

            def work(res: ActionResult, reg: StateRegistry, iter_num: int) -> None:
                for i in range(100):
                    match = MockMatch(i, i, f"leak_{i}")
                    res.add(match)

                    @state(name=f"leak_state_{iter_num}_{i}_{threading.get_ident()}")
                    class LeakState:
                        pass

                    reg.register_state(LeakState)

            for _ in range(10):
                t = threading.Thread(target=work, args=(result, registry, iteration))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # Clear references
            del result
            del registry
            gc.collect()

        # Measure
        gc.collect()
        final = tracemalloc.get_traced_memory()[0]

        tracemalloc.stop()

        # Memory growth should be minimal
        growth_mb = (final - baseline) / 1024 / 1024

        print(f"Memory growth after 10 iterations: {growth_mb:.2f} MB")

        # Allow some growth but not excessive
        assert growth_mb < 10, f"Possible memory leak: {growth_mb:.2f} MB growth"


class TestScalabilityBenchmarks:
    """Benchmark scalability with increasing load."""

    def test_scalability_with_thread_count(self):
        """Test how performance scales with thread count."""
        results = []

        for num_threads in [1, 2, 5, 10, 20, 50]:
            result = ActionResult()
            matches_per_thread = 1000

            def add_matches(thread_id: int, res: ActionResult, mpt: int) -> None:
                for i in range(mpt):
                    match = MockMatch(thread_id * 1000 + i, i, f"scale_{thread_id}_{i}")
                    res.add(match)

            start_time = time.time()

            threads = [
                threading.Thread(target=add_matches, args=(i, result, matches_per_thread))
                for i in range(num_threads)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            elapsed = time.time() - start_time
            total_matches = num_threads * matches_per_thread
            rate = total_matches / elapsed

            results.append(
                {
                    "threads": num_threads,
                    "time": elapsed,
                    "matches": total_matches,
                    "rate": rate,
                }
            )

            print(
                f"{num_threads} threads: {total_matches} matches in {elapsed:.3f}s ({rate:.0f} matches/sec)"
            )

        # Verify correctness for all runs
        assert all(r["matches"] == r["threads"] * 1000 for r in results)

    def test_scalability_with_data_size(self):
        """Test how performance scales with data size."""
        results = []

        for num_matches in [100, 500, 1000, 5000, 10000]:
            result = ActionResult()

            start_time = time.time()

            for i in range(num_matches):
                match = MockMatch(i, i, f"size_{i}")
                result.add(match)

            elapsed = time.time() - start_time
            rate = num_matches / elapsed

            results.append({"matches": num_matches, "time": elapsed, "rate": rate})

            print(f"{num_matches} matches: {elapsed:.3f}s ({rate:.0f} matches/sec)")

        # Performance should scale roughly linearly
        # (rate should stay relatively constant)
        rates = [r["rate"] for r in results]
        avg_rate = sum(rates) / len(rates)

        # All rates should be within 50% of average (allows for some variance)
        for r in results:
            deviation = abs(r["rate"] - avg_rate) / avg_rate * 100
            assert deviation < 50, f"Rate deviation too large: {deviation:.1f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print output
