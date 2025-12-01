"""Benchmark configuration export/import performance.

Tests:
- Time to export large configuration to JSON
- Time to import and validate large configuration
- Memory usage during export/import
- Round-trip accuracy
"""

import gc
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any


def generate_config(num_workflows: int, num_actions: int) -> dict[str, Any]:
    """Generate a configuration for export/import testing.

    Args:
        num_workflows: Number of workflows
        num_actions: Number of actions per workflow

    Returns:
        Configuration dictionary
    """
    config = {
        "version": "2.0.0",
        "metadata": {
            "name": "export_test_config",
            "description": "Configuration for export/import benchmarking",
        },
        "images": [],
        "workflows": [],
        "states": [],
        "categories": [],
        "execution_settings": {
            "defaultTimeout": 10000,
            "defaultRetryCount": 3,
        },
        "recognition_settings": {
            "defaultThreshold": 0.85,
        },
        "schedules": [],
        "transitions": [],
        "settings": {},
    }

    # Generate workflows
    for w in range(num_workflows):
        workflow = {
            "id": f"workflow_{w}",
            "name": f"Workflow {w}",
            "description": f"Test workflow {w}",
            "actions": [],
        }

        for a in range(num_actions):
            action = {
                "id": f"action_{w}_{a}",
                "type": "click",
                "name": f"Action {a}",
                "params": {"x": 100, "y": 200},
            }
            workflow["actions"].append(action)

        config["workflows"].append(workflow)

    return config


def benchmark_export(
    config_size: str, num_workflows: int, num_actions: int
) -> dict[str, Any]:
    """Benchmark configuration export performance.

    Args:
        config_size: Size label (small, large)
        num_workflows: Number of workflows
        num_actions: Number of actions per workflow

    Returns:
        Benchmark results dictionary
    """
    print(f"\n{'=' * 60}")
    print(
        f"Benchmarking {config_size} config export: {num_workflows} workflows, {num_actions} actions"
    )
    print(f"{'=' * 60}")

    # Generate config
    config = generate_config(num_workflows, num_actions)

    # Start memory tracking
    tracemalloc.start()
    gc.collect()
    baseline_memory = tracemalloc.get_traced_memory()[0]

    # Export to JSON
    output_file = Path(f"/tmp/benchmark_export_{config_size}.json")

    start_time = time.time()
    with open(output_file, "w") as f:
        json.dump(config, f, indent=2)
    export_time = time.time() - start_time

    # Measure memory
    gc.collect()
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory_used = (current_memory - baseline_memory) / 1024 / 1024  # MB
    peak_memory_used = (peak_memory - baseline_memory) / 1024 / 1024  # MB

    # Get file size
    file_size = output_file.stat().st_size

    # Cleanup
    output_file.unlink()

    # Determine grade
    target_times = {"small": 0.1, "large": 2.0}  # 100ms  # 2s
    target = target_times.get(config_size, 1.0)
    ratio = export_time / target

    if ratio <= 1.0:
        grade = "Excellent"
    elif ratio <= 1.5:
        grade = "Good"
    elif ratio <= 2.0:
        grade = "Fair"
    else:
        grade = "Poor"

    # Results
    results = {
        "operation": "export",
        "config_size": config_size,
        "num_workflows": num_workflows,
        "num_actions": num_actions,
        "export_time_ms": export_time * 1000,
        "file_size_kb": file_size / 1024,
        "memory_used_mb": memory_used,
        "peak_memory_mb": peak_memory_used,
        "target_time_ms": target * 1000,
        "grade": grade,
    }

    # Print results
    print(f"Export time:         {results['export_time_ms']:.1f} ms")
    print(f"File size:           {results['file_size_kb']:.1f} KB")
    print(f"Memory used:         {results['memory_used_mb']:.2f} MB")
    print(f"Peak memory:         {results['peak_memory_mb']:.2f} MB")
    print(f"Target time:         {results['target_time_ms']:.1f} ms")
    print(f"Grade:               {grade}")

    return results


def benchmark_import(
    config_size: str, num_workflows: int, num_actions: int
) -> dict[str, Any]:
    """Benchmark configuration import performance.

    Args:
        config_size: Size label (small, large)
        num_workflows: Number of workflows
        num_actions: Number of actions per workflow

    Returns:
        Benchmark results dictionary
    """
    print(f"\n{'=' * 60}")
    print(
        f"Benchmarking {config_size} config import: {num_workflows} workflows, {num_actions} actions"
    )
    print(f"{'=' * 60}")

    # Generate and save config
    config = generate_config(num_workflows, num_actions)
    input_file = Path(f"/tmp/benchmark_import_{config_size}.json")

    with open(input_file, "w") as f:
        json.dump(config, f, indent=2)

    file_size = input_file.stat().st_size

    # Start memory tracking
    tracemalloc.start()
    gc.collect()
    baseline_memory = tracemalloc.get_traced_memory()[0]

    # Import from JSON
    start_time = time.time()
    with open(input_file) as f:
        json.load(f)
    import_time = time.time() - start_time

    # Measure memory
    gc.collect()
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory_used = (current_memory - baseline_memory) / 1024 / 1024  # MB
    peak_memory_used = (peak_memory - baseline_memory) / 1024 / 1024  # MB

    # Cleanup
    input_file.unlink()

    # Determine grade
    target_times = {"small": 0.2, "large": 2.0}  # 200ms  # 2s
    target = target_times.get(config_size, 1.0)
    ratio = import_time / target

    if ratio <= 1.0:
        grade = "Excellent"
    elif ratio <= 1.5:
        grade = "Good"
    elif ratio <= 2.0:
        grade = "Fair"
    else:
        grade = "Poor"

    # Results
    results = {
        "operation": "import",
        "config_size": config_size,
        "num_workflows": num_workflows,
        "num_actions": num_actions,
        "import_time_ms": import_time * 1000,
        "file_size_kb": file_size / 1024,
        "memory_used_mb": memory_used,
        "peak_memory_mb": peak_memory_used,
        "target_time_ms": target * 1000,
        "grade": grade,
    }

    # Print results
    print(f"Import time:         {results['import_time_ms']:.1f} ms")
    print(f"File size:           {results['file_size_kb']:.1f} KB")
    print(f"Memory used:         {results['memory_used_mb']:.2f} MB")
    print(f"Peak memory:         {results['peak_memory_mb']:.2f} MB")
    print(f"Target time:         {results['target_time_ms']:.1f} ms")
    print(f"Grade:               {grade}")

    return results


def benchmark_roundtrip(
    config_size: str, num_workflows: int, num_actions: int
) -> dict[str, Any]:
    """Benchmark export then import round-trip.

    Args:
        config_size: Size label (small, large)
        num_workflows: Number of workflows
        num_actions: Number of actions per workflow

    Returns:
        Benchmark results dictionary
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmarking {config_size} config round-trip")
    print(f"{'=' * 60}")

    # Generate config
    original_config = generate_config(num_workflows, num_actions)
    temp_file = Path(f"/tmp/benchmark_roundtrip_{config_size}.json")

    # Export
    start_time = time.time()
    with open(temp_file, "w") as f:
        json.dump(original_config, f, indent=2)
    export_time = time.time() - start_time

    # Import
    start_time = time.time()
    with open(temp_file) as f:
        imported_config = json.load(f)
    import_time = time.time() - start_time

    total_time = export_time + import_time

    # Verify accuracy
    configs_match = original_config == imported_config

    # Cleanup
    temp_file.unlink()

    # Results
    results = {
        "operation": "roundtrip",
        "config_size": config_size,
        "export_time_ms": export_time * 1000,
        "import_time_ms": import_time * 1000,
        "total_time_ms": total_time * 1000,
        "configs_match": configs_match,
    }

    # Print results
    print(f"Export time:         {results['export_time_ms']:.1f} ms")
    print(f"Import time:         {results['import_time_ms']:.1f} ms")
    print(f"Total time:          {results['total_time_ms']:.1f} ms")
    print(f"Accuracy:            {'PASS' if configs_match else 'FAIL'}")

    return results


def run_all_benchmarks() -> list[dict[str, Any]]:
    """Run all export/import benchmarks.

    Returns:
        List of benchmark results
    """
    print("\n" + "=" * 60)
    print("EXPORT/IMPORT PERFORMANCE BENCHMARKS")
    print("=" * 60)

    results = []

    # Test cases
    test_cases = [
        ("small", 10, 5),
        ("large", 100, 10),
    ]

    for config_size, num_workflows, num_actions in test_cases:
        # Export
        result = benchmark_export(config_size, num_workflows, num_actions)
        results.append(result)

        # Import
        result = benchmark_import(config_size, num_workflows, num_actions)
        results.append(result)

        # Round-trip
        result = benchmark_roundtrip(config_size, num_workflows, num_actions)
        results.append(result)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Operation':<20} {'Config Size':<15} {'Time (ms)':<15} {'Grade':<15}")
    print(f"{'-' * 65}")

    for result in results:
        if result["operation"] in ["export", "import"]:
            time_key = f"{result['operation']}_time_ms"
            print(
                f"{result['operation']:<20} "
                f"{result['config_size']:<15} "
                f"{result[time_key]:<15.1f} "
                f"{result['grade']:<15}"
            )

    return results


if __name__ == "__main__":
    results = run_all_benchmarks()

    # Save results
    output_file = Path("/tmp/benchmark_export_import_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
