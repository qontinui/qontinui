"""Benchmark image finding performance.

Tests:
- Time to find image on screen (best case: image present)
- Time to timeout (worst case: image not present)
- Pattern matching performance
- Concurrent find operations
"""

import time
from pathlib import Path
from typing import Any


class MockImageFinder:
    """Mock image finder for benchmarking."""

    def __init__(self, find_delay: float = 0.05):
        """Initialize mock image finder.

        Args:
            find_delay: Simulated pattern matching delay in seconds
        """
        self.find_delay = find_delay
        self.cache = {}

    def find_image(
        self,
        pattern_name: str,
        timeout: float = 3.0,
        will_find: bool = True,
    ) -> dict[str, Any]:
        """Mock image finding operation.

        Args:
            pattern_name: Pattern to find
            timeout: Maximum time to search
            will_find: Whether the image will be found

        Returns:
            Find result dictionary
        """
        start_time = time.time()

        if will_find:
            # Simulate successful find
            time.sleep(self.find_delay)
            return {
                "found": True,
                "x": 100,
                "y": 200,
                "confidence": 0.95,
                "time_ms": (time.time() - start_time) * 1000,
            }
        else:
            # Simulate timeout
            time.sleep(timeout)
            return {
                "found": False,
                "x": None,
                "y": None,
                "confidence": 0.0,
                "time_ms": (time.time() - start_time) * 1000,
            }

    def find_multiple(self, pattern_names: list[str]) -> list[dict[str, Any]]:
        """Find multiple patterns sequentially.

        Args:
            pattern_names: List of patterns to find

        Returns:
            List of find results
        """
        results = []
        for pattern in pattern_names:
            result = self.find_image(pattern, will_find=True)
            results.append(result)
        return results

    def measure_pattern_matching_fps(self, duration: float = 1.0) -> float:
        """Measure pattern matching frames per second.

        Args:
            duration: Duration to measure in seconds

        Returns:
            Frames per second
        """
        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < duration:
            # Simulate pattern matching
            time.sleep(self.find_delay)
            frame_count += 1

        elapsed = time.time() - start_time
        return frame_count / elapsed if elapsed > 0 else 0


def benchmark_image_finding(scenario: str, will_find: bool, timeout: float = 3.0) -> dict[str, Any]:
    """Benchmark image finding performance.

    Args:
        scenario: Scenario label (best_case, average_case, worst_case)
        will_find: Whether image will be found
        timeout: Maximum search time

    Returns:
        Benchmark results dictionary
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmarking {scenario}: will_find={will_find}, timeout={timeout}s")
    print(f"{'=' * 60}")

    finder = MockImageFinder(find_delay=0.05)

    # Run find operation
    start_time = time.time()
    result = finder.find_image("test_pattern", timeout=timeout, will_find=will_find)
    execution_time = time.time() - start_time

    # Determine grade
    if will_find:
        # Best/average case - should be fast
        target = 0.5  # 500ms
        if execution_time <= 0.2:
            grade = "Excellent"
        elif execution_time <= target:
            grade = "Good"
        elif execution_time <= target * 1.5:
            grade = "Fair"
        else:
            grade = "Poor"
    else:
        # Worst case - should timeout appropriately
        target = timeout
        deviation = abs(execution_time - timeout) / timeout
        if deviation <= 0.1:
            grade = "Excellent"
        elif deviation <= 0.2:
            grade = "Good"
        else:
            grade = "Fair"

    # Results
    results = {
        "scenario": scenario,
        "found": result["found"],
        "execution_time_ms": execution_time * 1000,
        "confidence": result.get("confidence", 0.0),
        "target_time_ms": (target if will_find else timeout) * 1000,
        "grade": grade,
    }

    # Print results
    print(f"Found:               {result['found']}")
    print(f"Execution time:      {results['execution_time_ms']:.1f} ms")
    print(f"Confidence:          {result.get('confidence', 0.0):.2f}")
    print(f"Target time:         {results['target_time_ms']:.1f} ms")
    print(f"Grade:               {grade}")

    return results


def benchmark_multiple_images(num_images: int) -> dict[str, Any]:
    """Benchmark finding multiple images.

    Args:
        num_images: Number of images to find

    Returns:
        Benchmark results dictionary
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmarking multiple image finds: {num_images} images")
    print(f"{'=' * 60}")

    finder = MockImageFinder(find_delay=0.05)
    patterns = [f"pattern_{i}" for i in range(num_images)]

    # Sequential finds
    start_time = time.time()
    results = finder.find_multiple(patterns)
    execution_time = time.time() - start_time

    # Calculate metrics
    avg_time_per_image = execution_time / num_images if num_images > 0 else 0
    images_per_second = num_images / execution_time if execution_time > 0 else 0

    # Results
    result = {
        "num_images": num_images,
        "total_time_ms": execution_time * 1000,
        "avg_time_per_image_ms": avg_time_per_image * 1000,
        "images_per_second": images_per_second,
        "success_count": sum(1 for r in results if r["found"]),
    }

    # Print results
    print(f"Total time:          {result['total_time_ms']:.1f} ms")
    print(f"Avg per image:       {result['avg_time_per_image_ms']:.1f} ms")
    print(f"Images/second:       {result['images_per_second']:.1f}")
    print(f"Success count:       {result['success_count']}/{num_images}")

    return result


def benchmark_pattern_matching_fps() -> dict[str, Any]:
    """Benchmark pattern matching frames per second.

    Returns:
        Benchmark results dictionary
    """
    print(f"\n{'=' * 60}")
    print("Benchmarking pattern matching FPS")
    print(f"{'=' * 60}")

    finder = MockImageFinder(find_delay=0.05)

    # Measure FPS
    fps = finder.measure_pattern_matching_fps(duration=2.0)

    # Determine grade
    target_fps = 10
    if fps >= target_fps:
        grade = "Excellent"
    elif fps >= target_fps * 0.8:
        grade = "Good"
    elif fps >= target_fps * 0.6:
        grade = "Fair"
    else:
        grade = "Poor"

    # Results
    results = {
        "fps": fps,
        "target_fps": target_fps,
        "grade": grade,
    }

    # Print results
    print(f"FPS:                 {fps:.1f}")
    print(f"Target FPS:          {target_fps}")
    print(f"Grade:               {grade}")

    return results


def benchmark_concurrent_finds(num_threads: int = 10) -> dict[str, Any]:
    """Benchmark concurrent image finding.

    Args:
        num_threads: Number of concurrent find operations

    Returns:
        Benchmark results dictionary
    """
    import concurrent.futures

    print(f"\n{'=' * 60}")
    print(f"Benchmarking concurrent finds: {num_threads} threads")
    print(f"{'=' * 60}")

    finder = MockImageFinder(find_delay=0.05)

    # Sequential baseline
    start_time = time.time()
    for i in range(num_threads):
        finder.find_image(f"pattern_{i}", will_find=True)
    sequential_time = time.time() - start_time

    # Concurrent
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(finder.find_image, f"pattern_{i}", will_find=True)
            for i in range(num_threads)
        ]
        concurrent.futures.wait(futures)
    concurrent_time = time.time() - start_time

    # Calculate speedup
    speedup = sequential_time / concurrent_time if concurrent_time > 0 else 0
    efficiency = (speedup / num_threads) * 100

    # Results
    results = {
        "num_threads": num_threads,
        "sequential_time_ms": sequential_time * 1000,
        "concurrent_time_ms": concurrent_time * 1000,
        "speedup": speedup,
        "efficiency_percent": efficiency,
    }

    # Print results
    print(f"Sequential time:     {results['sequential_time_ms']:.1f} ms")
    print(f"Concurrent time:     {results['concurrent_time_ms']:.1f} ms")
    print(f"Speedup:             {results['speedup']:.2f}x")
    print(f"Efficiency:          {results['efficiency_percent']:.1f}%")

    return results


def run_all_benchmarks() -> list[dict[str, Any]]:
    """Run all image finding benchmarks.

    Returns:
        List of benchmark results
    """
    print("\n" + "=" * 60)
    print("IMAGE FINDING PERFORMANCE BENCHMARKS")
    print("=" * 60)

    results = []

    # Best case
    result = benchmark_image_finding("best_case", will_find=True, timeout=3.0)
    results.append(result)

    # Average case
    result = benchmark_image_finding("average_case", will_find=True, timeout=3.0)
    results.append(result)

    # Worst case
    result = benchmark_image_finding("worst_case", will_find=False, timeout=3.0)
    results.append(result)

    # Multiple images
    result = benchmark_multiple_images(10)
    results.append(result)

    # Pattern matching FPS
    result = benchmark_pattern_matching_fps()
    results.append(result)

    # Concurrent finds
    result = benchmark_concurrent_finds(10)
    results.append(result)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Scenario':<20} {'Time (ms)':<15} {'Grade':<15}")
    print(f"{'-' * 50}")

    for result in results:
        if "scenario" in result:
            print(
                f"{result['scenario']:<20} "
                f"{result['execution_time_ms']:<15.1f} "
                f"{result['grade']:<15}"
            )
        elif "fps" in result:
            print(f"{'pattern_matching_fps':<20} {result['fps']:<15.1f} {result['grade']:<15}")

    return results


if __name__ == "__main__":
    import json

    results = run_all_benchmarks()

    # Save results
    output_file = Path("/tmp/benchmark_image_finding_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
