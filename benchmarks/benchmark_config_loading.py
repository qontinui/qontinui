"""Benchmark configuration loading performance.

Tests:
- Time to load small, medium, large, and very large configs
- Memory usage during parsing
- Schema validation overhead
"""

import gc
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

from ..src.qontinui.json_executor.config_parser import ConfigParser


def generate_mock_config(num_workflows: int, num_actions: int) -> dict[str, Any]:
    """Generate a mock configuration for testing.

    Args:
        num_workflows: Number of workflows to generate
        num_actions: Number of actions per workflow

    Returns:
        Mock configuration dictionary
    """
    config = {
        "version": "2.0.0",
        "metadata": {
            "name": f"benchmark_config_{num_workflows}w_{num_actions}a",
            "description": "Generated benchmark configuration",
        },
        "images": [],
        "workflows": [],
        "states": [],
        "categories": [],
        "execution_settings": {
            "defaultTimeout": 10000,
            "defaultRetryCount": 3,
            "actionDelay": 100,
            "failureStrategy": "stop",
            "headless": False,
        },
        "recognition_settings": {
            "defaultThreshold": 0.85,
            "searchAlgorithm": "template_matching",
            "multiScaleSearch": True,
            "colorSpace": "rgb",
            "edgeDetection": False,
            "ocrEnabled": False,
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

        # Generate actions for this workflow
        for a in range(num_actions):
            action = {
                "id": f"action_{w}_{a}",
                "type": "click",
                "name": f"Action {w}.{a}",
                "params": {
                    "x": 100 + a * 10,
                    "y": 100 + a * 10,
                    "button": "left",
                },
            }
            workflow["actions"].append(action)

        config["workflows"].append(workflow)

    return config


def benchmark_config_loading(
    config_size: str, num_workflows: int, num_actions: int
) -> dict[str, Any]:
    """Benchmark configuration loading performance.

    Args:
        config_size: Size label (small, medium, large, very_large)
        num_workflows: Number of workflows
        num_actions: Number of actions per workflow

    Returns:
        Benchmark results dictionary
    """
    print(f"\n{'=' * 60}")
    print(
        f"Benchmarking {config_size} config: {num_workflows} workflows, {num_actions} actions"
    )
    print(f"{'=' * 60}")

    # Generate mock config
    config_dict = generate_mock_config(num_workflows, num_actions)

    # Save to temporary file
    temp_file = Path(f"/tmp/benchmark_config_{config_size}.json")
    with open(temp_file, "w") as f:
        json.dump(config_dict, f)

    file_size = temp_file.stat().st_size

    # Start memory tracking
    tracemalloc.start()
    gc.collect()
    baseline_memory = tracemalloc.get_traced_memory()[0]

    # Parse configuration
    parser = ConfigParser()

    start_time = time.time()
    parser.parse_file(str(temp_file))
    parse_time = time.time() - start_time

    # Measure memory
    gc.collect()
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory_used = (current_memory - baseline_memory) / 1024 / 1024  # MB
    peak_memory_used = (peak_memory - baseline_memory) / 1024 / 1024  # MB

    # Cleanup
    parser.cleanup()
    temp_file.unlink()

    # Calculate metrics
    total_actions = num_workflows * num_actions
    actions_per_second = total_actions / parse_time if parse_time > 0 else 0

    # Determine grade
    target_times = {
        "small": 0.1,  # 100ms
        "medium": 0.3,  # 300ms
        "large": 1.0,  # 1s
        "very_large": 5.0,  # 5s
    }
    target = target_times.get(config_size, 1.0)
    ratio = parse_time / target

    if ratio <= 1.0:
        grade = "Excellent"
    elif ratio <= 1.5:
        grade = "Good"
    elif ratio <= 2.0:
        grade = "Fair"
    elif ratio <= 5.0:
        grade = "Poor"
    else:
        grade = "Critical"

    # Results
    results = {
        "config_size": config_size,
        "num_workflows": num_workflows,
        "num_actions": num_actions,
        "total_actions": total_actions,
        "file_size_kb": file_size / 1024,
        "parse_time_ms": parse_time * 1000,
        "actions_per_second": actions_per_second,
        "memory_used_mb": memory_used,
        "peak_memory_mb": peak_memory_used,
        "target_time_ms": target * 1000,
        "grade": grade,
    }

    # Print results
    print(f"File size:           {results['file_size_kb']:.1f} KB")
    print(f"Parse time:          {results['parse_time_ms']:.1f} ms")
    print(f"Target time:         {results['target_time_ms']:.1f} ms")
    print(f"Actions/second:      {results['actions_per_second']:.0f}")
    print(f"Memory used:         {results['memory_used_mb']:.2f} MB")
    print(f"Peak memory:         {results['peak_memory_mb']:.2f} MB")
    print(f"Grade:               {grade}")

    return results


def run_all_benchmarks() -> list[dict[str, Any]]:
    """Run all configuration loading benchmarks.

    Returns:
        List of benchmark results
    """
    print("\n" + "=" * 60)
    print("CONFIGURATION LOADING PERFORMANCE BENCHMARKS")
    print("=" * 60)

    test_cases = [
        ("small", 10, 5),  # 10 workflows, 5 actions = 50 total
        ("medium", 50, 5),  # 50 workflows, 5 actions = 250 total
        ("large", 100, 10),  # 100 workflows, 10 actions = 1000 total
        ("very_large", 500, 10),  # 500 workflows, 10 actions = 5000 total
    ]

    results = []
    for config_size, num_workflows, num_actions in test_cases:
        result = benchmark_config_loading(config_size, num_workflows, num_actions)
        results.append(result)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Config Size':<15} {'Parse Time':<15} {'Grade':<15}")
    print(f"{'-' * 45}")

    for result in results:
        print(
            f"{result['config_size']:<15} "
            f"{result['parse_time_ms']:<15.1f} "
            f"{result['grade']:<15}"
        )

    return results


if __name__ == "__main__":
    results = run_all_benchmarks()

    # Save results
    output_file = Path("/tmp/benchmark_config_loading_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
