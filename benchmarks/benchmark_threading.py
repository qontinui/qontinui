"""Benchmark threading performance and overhead.

Tests:
- Verify no performance regression from threading.Lock
- Measure lock contention under load
- Compare single-threaded vs multi-threaded performance
- Test scalability with varying thread counts
"""

import threading
import time
from typing import Any


class ThreadSafeCounter:
    """Thread-safe counter for benchmarking."""

    def __init__(self, use_lock: bool = True):
        """Initialize counter.

        Args:
            use_lock: Whether to use threading.Lock for thread safety
        """
        self.count = 0
        self.use_lock = use_lock
        if use_lock:
            self.lock = threading.Lock()

    def increment(self):
        """Increment the counter."""
        if self.use_lock:
            with self.lock:
                self.count += 1
        else:
            self.count += 1

    def get(self) -> int:
        """Get the current count.

        Returns:
            Current count value
        """
        if self.use_lock:
            with self.lock:
                return self.count
        else:
            return self.count


def benchmark_lock_overhead(num_operations: int = 100000) -> dict[str, Any]:
    """Benchmark lock overhead vs no lock.

    Args:
        num_operations: Number of increment operations

    Returns:
        Benchmark results dictionary
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmarking lock overhead: {num_operations} operations")
    print(f"{'=' * 60}")

    # Without lock (baseline)
    counter_no_lock = ThreadSafeCounter(use_lock=False)
    start_time = time.time()
    for _ in range(num_operations):
        counter_no_lock.increment()
    no_lock_time = time.time() - start_time

    # With lock
    counter_with_lock = ThreadSafeCounter(use_lock=True)
    start_time = time.time()
    for _ in range(num_operations):
        counter_with_lock.increment()
    with_lock_time = time.time() - start_time

    # Calculate overhead
    overhead = with_lock_time - no_lock_time
    overhead_percent = (overhead / no_lock_time) * 100 if no_lock_time > 0 else 0
    overhead_per_op = (overhead / num_operations) * 1000 if num_operations > 0 else 0  # ms

    # Determine grade
    if overhead_percent <= 10:
        grade = "Excellent"
    elif overhead_percent <= 25:
        grade = "Good"
    elif overhead_percent <= 50:
        grade = "Fair"
    else:
        grade = "Poor"

    # Results
    results = {
        "num_operations": num_operations,
        "no_lock_time_ms": no_lock_time * 1000,
        "with_lock_time_ms": with_lock_time * 1000,
        "overhead_ms": overhead * 1000,
        "overhead_percent": overhead_percent,
        "overhead_per_op_us": overhead_per_op,
        "grade": grade,
    }

    # Print results
    print(f"Without lock:        {results['no_lock_time_ms']:.1f} ms")
    print(f"With lock:           {results['with_lock_time_ms']:.1f} ms")
    print(f"Overhead:            {results['overhead_ms']:.1f} ms ({overhead_percent:.1f}%)")
    print(f"Overhead/operation:  {results['overhead_per_op_us']:.3f} us")
    print(f"Grade:               {grade}")

    return results


def benchmark_lock_contention(
    num_threads: int, operations_per_thread: int = 10000
) -> dict[str, Any]:
    """Benchmark lock contention with multiple threads.

    Args:
        num_threads: Number of concurrent threads
        operations_per_thread: Number of operations per thread

    Returns:
        Benchmark results dictionary
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmarking lock contention: {num_threads} threads")
    print(f"{'=' * 60}")

    # Single-threaded baseline
    counter_baseline = ThreadSafeCounter(use_lock=False)
    start_time = time.time()
    for _ in range(num_threads * operations_per_thread):
        counter_baseline.increment()
    baseline_time = time.time() - start_time

    # Multi-threaded with locks
    counter_threaded = ThreadSafeCounter(use_lock=True)

    def worker():
        for _ in range(operations_per_thread):
            counter_threaded.increment()

    start_time = time.time()
    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    threaded_time = time.time() - start_time

    # Verify correctness
    expected_count = num_threads * operations_per_thread
    actual_count = counter_threaded.get()
    correct = expected_count == actual_count

    # Calculate overhead
    overhead = threaded_time - baseline_time
    overhead_percent = (overhead / baseline_time) * 100 if baseline_time > 0 else 0

    # Determine grade
    if overhead_percent <= 10:
        grade = "Excellent"
    elif overhead_percent <= 25:
        grade = "Good"
    elif overhead_percent <= 50:
        grade = "Fair"
    else:
        grade = "Poor"

    # Results
    results = {
        "num_threads": num_threads,
        "operations_per_thread": operations_per_thread,
        "baseline_time_ms": baseline_time * 1000,
        "threaded_time_ms": threaded_time * 1000,
        "overhead_ms": overhead * 1000,
        "overhead_percent": overhead_percent,
        "correct": correct,
        "expected_count": expected_count,
        "actual_count": actual_count,
        "grade": grade,
    }

    # Print results
    print(f"Baseline time:       {results['baseline_time_ms']:.1f} ms")
    print(f"Threaded time:       {results['threaded_time_ms']:.1f} ms")
    print(f"Overhead:            {results['overhead_ms']:.1f} ms ({overhead_percent:.1f}%)")
    print(f"Correctness:         {'PASS' if correct else 'FAIL'} ({actual_count}/{expected_count})")
    print(f"Grade:               {grade}")

    return results


def benchmark_lock_acquisition_time(num_acquisitions: int = 100000) -> dict[str, Any]:
    """Benchmark lock acquisition time.

    Args:
        num_acquisitions: Number of lock acquisitions to measure

    Returns:
        Benchmark results dictionary
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmarking lock acquisition time: {num_acquisitions} acquisitions")
    print(f"{'=' * 60}")

    lock = threading.Lock()

    start_time = time.time()
    for _ in range(num_acquisitions):
        with lock:
            pass  # Empty critical section
    elapsed = time.time() - start_time

    avg_acquisition_time = (elapsed / num_acquisitions) * 1000000  # microseconds

    # Determine grade
    if avg_acquisition_time <= 1.0:
        grade = "Excellent"
    elif avg_acquisition_time <= 5.0:
        grade = "Good"
    elif avg_acquisition_time <= 10.0:
        grade = "Fair"
    else:
        grade = "Poor"

    # Results
    results = {
        "num_acquisitions": num_acquisitions,
        "total_time_ms": elapsed * 1000,
        "avg_acquisition_time_us": avg_acquisition_time,
        "acquisitions_per_second": num_acquisitions / elapsed if elapsed > 0 else 0,
        "grade": grade,
    }

    # Print results
    print(f"Total time:          {results['total_time_ms']:.1f} ms")
    print(f"Avg acquisition:     {results['avg_acquisition_time_us']:.3f} us")
    print(f"Acquisitions/sec:    {results['acquisitions_per_second']:.0f}")
    print(f"Grade:               {grade}")

    return results


def benchmark_scalability(max_threads: int = 50) -> dict[str, Any]:
    """Benchmark scalability with varying thread counts.

    Args:
        max_threads: Maximum number of threads to test

    Returns:
        Benchmark results dictionary
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmarking scalability: 1 to {max_threads} threads")
    print(f"{'=' * 60}")

    operations_per_thread = 10000
    thread_counts = [1, 2, 5, 10, 20, max_threads]
    scalability_data = []

    for num_threads in thread_counts:
        counter = ThreadSafeCounter(use_lock=True)

        def worker(counter_ref=counter):
            for _ in range(operations_per_thread):
                counter_ref.increment()

        start_time = time.time()
        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start_time

        total_ops = num_threads * operations_per_thread
        ops_per_second = total_ops / elapsed if elapsed > 0 else 0

        scalability_data.append(
            {
                "threads": num_threads,
                "time_ms": elapsed * 1000,
                "ops_per_second": ops_per_second,
            }
        )

        print(
            f"{num_threads:>3} threads: {elapsed * 1000:>8.1f} ms ({ops_per_second:>10.0f} ops/sec)"
        )

    # Results
    results = {
        "max_threads": max_threads,
        "scalability_data": scalability_data,
    }

    return results


def run_all_benchmarks() -> list[dict[str, Any]]:
    """Run all threading benchmarks.

    Returns:
        List of benchmark results
    """
    print("\n" + "=" * 60)
    print("THREADING PERFORMANCE BENCHMARKS")
    print("=" * 60)

    results = []

    # Lock overhead
    result = benchmark_lock_overhead(100000)
    results.append(result)

    # Lock contention with different thread counts
    for num_threads in [2, 10, 50]:
        result = benchmark_lock_contention(num_threads, 10000)
        results.append(result)

    # Lock acquisition time
    result = benchmark_lock_acquisition_time(100000)
    results.append(result)

    # Scalability
    result = benchmark_scalability(50)
    results.append(result)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Test':<30} {'Overhead %':<15} {'Grade':<15}")
    print(f"{'-' * 60}")

    for result in results:
        if "overhead_percent" in result:
            test_name = (
                f"{result.get('num_threads', 'N/A')} threads"
                if "num_threads" in result
                else "lock_overhead"
            )
            print(f"{test_name:<30} {result['overhead_percent']:<15.1f} {result['grade']:<15}")

    return results


if __name__ == "__main__":
    import json
    from pathlib import Path

    results = run_all_benchmarks()

    # Save results
    output_file = Path("/tmp/benchmark_threading_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
