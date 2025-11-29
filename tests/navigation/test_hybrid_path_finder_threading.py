"""Threading tests for HybridPathFinder.

Tests thread safety of HybridPathFinder cache and weight access under concurrent operations.
"""

import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from qontinui.navigation.hybrid_path_finder import HybridPathFinder, PathStrategy  # noqa: E402


class TestHybridPathFinderThreading:
    """Test thread safety of HybridPathFinder cache and configuration."""

    def test_concurrent_cache_access(self):
        """Test concurrent access to path cache."""
        # Create finder with mock joint table
        mock_joint_table = Mock()
        finder = HybridPathFinder(joint_table=mock_joint_table, enable_caching=True)

        num_threads = 10
        iterations = 50
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(iterations):
                    # Simulate cache operations
                    cache_key = (frozenset({thread_id}), frozenset({i}))

                    # Write to cache
                    with finder._lock:
                        finder._path_cache[cache_key] = Mock()

                    # Read from cache
                    with finder._lock:
                        _ = finder._path_cache.get(cache_key)

                    time.sleep(0.001)
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

    def test_concurrent_clear_cache(self):
        """Test concurrent cache clear operations."""
        mock_joint_table = Mock()
        finder = HybridPathFinder(joint_table=mock_joint_table, enable_caching=True)

        # Pre-populate cache
        with finder._lock:
            for i in range(10):
                key = (frozenset({i}), frozenset({i + 1}))
                finder._path_cache[key] = Mock()

        num_threads = 5
        iterations = 20
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(iterations):
                    if i % 5 == 0:
                        # Clear cache
                        finder.clear_cache()
                    else:
                        # Read cache size
                        _ = str(finder)
                    time.sleep(0.001)
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

    def test_concurrent_weight_modifications(self):
        """Test concurrent access to weight configuration."""
        mock_joint_table = Mock()
        finder = HybridPathFinder(joint_table=mock_joint_table)

        num_threads = 10
        iterations = 50
        errors = []

        def worker(thread_id: int):
            try:
                for _i in range(iterations):
                    # Simulate weight modifications (similar to _find_most_reliable_path)
                    with finder._lock:
                        old_reliability = finder.reliability_weight
                        old_state_cost = finder.state_cost_weight

                        # Temporarily modify
                        finder.reliability_weight = 0.7
                        finder.state_cost_weight = 0.1

                    time.sleep(0.001)

                    # Restore
                    with finder._lock:
                        finder.reliability_weight = old_reliability
                        finder.state_cost_weight = old_state_cost
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

        # Verify weights are reasonable (may not be exactly defaults due to interleaving)
        assert 0.0 <= finder.reliability_weight <= 1.0
        assert 0.0 <= finder.state_cost_weight <= 1.0

    def test_stress_mixed_cache_operations(self):
        """Stress test with mixed cache operations."""
        mock_joint_table = Mock()
        finder = HybridPathFinder(joint_table=mock_joint_table, enable_caching=True)

        num_threads = 10
        iterations = 100
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(iterations):
                    if i % 4 == 0:
                        # Add to cache
                        key = (frozenset({thread_id}), frozenset({i}))
                        with finder._lock:
                            finder._path_cache[key] = Mock()
                    elif i % 4 == 1:
                        # Clear cache
                        finder.clear_cache()
                    elif i % 4 == 2:
                        # Read cache
                        with finder._lock:
                            _ = len(finder._path_cache)
                    else:
                        # Get string representation
                        _ = str(finder)
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

    def test_no_data_corruption_in_cache(self):
        """Verify no data corruption in cache under concurrent access."""
        mock_joint_table = Mock()
        finder = HybridPathFinder(joint_table=mock_joint_table, enable_caching=True)

        num_threads = 10
        iterations = 50
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(iterations):
                    # Each thread uses unique keys
                    key = (frozenset({thread_id, i}), frozenset({thread_id + 100, i + 100}))
                    value = f"path_{thread_id}_{i}"

                    # Write to cache
                    with finder._lock:
                        finder._path_cache[key] = value

                    # Read back immediately
                    with finder._lock:
                        retrieved = finder._path_cache.get(key)

                    # Verify integrity
                    if retrieved != value:
                        errors.append(
                            f"Thread {thread_id} iteration {i}: "
                            f"Expected {value}, got {retrieved}"
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

    def test_concurrent_string_representation(self):
        """Test concurrent access to string representation."""
        mock_joint_table = Mock()
        finder = HybridPathFinder(joint_table=mock_joint_table, enable_caching=True)

        num_threads = 10
        iterations = 50
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(iterations):
                    if i % 2 == 0:
                        # Add to cache
                        key = (frozenset({i}), frozenset({i + 1}))
                        with finder._lock:
                            finder._path_cache[key] = Mock()
                    else:
                        # Read string representation (accesses cache size)
                        str_repr = str(finder)
                        if "HybridPathFinder" not in str_repr:
                            errors.append(f"Thread {thread_id}: Invalid string representation")
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

    def test_concurrent_strategy_access(self):
        """Test concurrent access to strategy configuration."""
        mock_joint_table = Mock()
        finder = HybridPathFinder(joint_table=mock_joint_table)

        num_threads = 10
        iterations = 50
        errors = []
        strategies = [PathStrategy.SHORTEST, PathStrategy.OPTIMAL, PathStrategy.MOST_RELIABLE]

        def worker(thread_id: int):
            try:
                for _i in range(iterations):
                    # Read current strategy
                    with finder._lock:
                        current = finder.strategy

                    # Verify it's valid
                    if current not in strategies:
                        errors.append(f"Thread {thread_id}: Invalid strategy {current}")
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

    def test_concurrent_cache_size_checks(self):
        """Test concurrent cache size checks."""
        mock_joint_table = Mock()
        finder = HybridPathFinder(joint_table=mock_joint_table, enable_caching=True)

        num_threads = 10
        iterations = 50
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(iterations):
                    if i % 3 == 0:
                        # Add items
                        key = (frozenset({thread_id, i}), frozenset({i}))
                        with finder._lock:
                            finder._path_cache[key] = Mock()
                    elif i % 3 == 1:
                        # Check size
                        with finder._lock:
                            size = len(finder._path_cache)
                            if size < 0:
                                errors.append(f"Thread {thread_id}: Negative cache size")
                    else:
                        # Clear
                        if i % 9 == 2:
                            finder.clear_cache()
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

    def test_concurrent_weight_reads(self):
        """Test concurrent reads of weight configuration."""
        mock_joint_table = Mock()
        finder = HybridPathFinder(joint_table=mock_joint_table)

        num_threads = 20
        iterations = 100
        errors = []

        def worker(thread_id: int):
            try:
                for _ in range(iterations):
                    # Read all weights
                    with finder._lock:
                        state_cost = finder.state_cost_weight
                        transition_cost = finder.transition_cost_weight
                        _ = finder.probability_weight  # Read to verify thread safety
                        _ = finder.reliability_weight  # Read to verify thread safety

                    # Verify they're reasonable
                    if not (0 <= state_cost <= 1 and 0 <= transition_cost <= 1):
                        errors.append(f"Thread {thread_id}: Invalid weights")
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
