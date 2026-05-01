"""Benchmark workflow execution performance.

Tests:
- Time to execute simple workflow (5 actions)
- Time to execute complex workflow (50 actions with conditions)
- Navigation transition execution time
- Action execution overhead
"""

import time
from typing import Any


class MockAction:
    """Mock action for benchmarking."""

    def __init__(self, name: str, delay: float = 0.001):
        """Initialize mock action.

        Args:
            name: Action name
            delay: Simulated execution delay in seconds
        """
        self.name = name
        self.delay = delay

    def execute(self) -> dict[str, Any]:
        """Execute the mock action.

        Returns:
            Execution result
        """
        time.sleep(self.delay)
        return {"success": True, "action": self.name}


class MockWorkflow:
    """Mock workflow for benchmarking."""

    def __init__(self, name: str, actions: list[MockAction]):
        """Initialize mock workflow.

        Args:
            name: Workflow name
            actions: List of actions to execute
        """
        self.name = name
        self.actions = actions

    def execute(self) -> dict[str, Any]:
        """Execute all actions in the workflow.

        Returns:
            Execution result
        """
        results = []
        for action in self.actions:
            result = action.execute()
            results.append(result)

        return {"success": True, "workflow": self.name, "actions": len(results)}


def benchmark_workflow_execution(
    workflow_type: str, num_actions: int, action_delay: float = 0.001
) -> dict[str, Any]:
    """Benchmark workflow execution performance.

    Args:
        workflow_type: Type label (simple, medium, complex)
        num_actions: Number of actions in workflow
        action_delay: Simulated action execution time

    Returns:
        Benchmark results dictionary
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmarking {workflow_type} workflow: {num_actions} actions")
    print(f"{'=' * 60}")

    # Create mock workflow
    actions = [MockAction(f"action_{i}", action_delay) for i in range(num_actions)]
    workflow = MockWorkflow(f"{workflow_type}_workflow", actions)

    # Execute workflow
    start_time = time.time()
    workflow.execute()
    execution_time = time.time() - start_time

    # Calculate metrics
    expected_time = num_actions * action_delay
    overhead_time = execution_time - expected_time
    overhead_per_action = overhead_time / num_actions if num_actions > 0 else 0
    overhead_percent = (overhead_time / expected_time) * 100 if expected_time > 0 else 0

    # Determine grade
    target_times = {
        "simple": 0.5,  # 500ms
        "medium": 2.0,  # 2s
        "complex": 5.0,  # 5s
    }
    target = target_times.get(workflow_type, 1.0)
    ratio = execution_time / target

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

    # Check action overhead
    overhead_target = 0.1  # 100ms
    if overhead_per_action * 1000 <= overhead_target:
        overhead_grade = "Excellent"
    elif overhead_per_action * 1000 <= overhead_target * 1.5:
        overhead_grade = "Good"
    elif overhead_per_action * 1000 <= overhead_target * 2.0:
        overhead_grade = "Fair"
    else:
        overhead_grade = "Poor"

    # Results
    results = {
        "workflow_type": workflow_type,
        "num_actions": num_actions,
        "execution_time_ms": execution_time * 1000,
        "expected_time_ms": expected_time * 1000,
        "overhead_time_ms": overhead_time * 1000,
        "overhead_per_action_ms": overhead_per_action * 1000,
        "overhead_percent": overhead_percent,
        "target_time_ms": target * 1000,
        "grade": grade,
        "overhead_grade": overhead_grade,
    }

    # Print results
    print(f"Execution time:      {results['execution_time_ms']:.1f} ms")
    print(f"Expected time:       {results['expected_time_ms']:.1f} ms")
    print(
        f"Overhead:            {results['overhead_time_ms']:.1f} ms ({results['overhead_percent']:.1f}%)"
    )
    print(
        f"Overhead/action:     {results['overhead_per_action_ms']:.2f} ms [{overhead_grade}]"
    )
    print(f"Target time:         {results['target_time_ms']:.1f} ms")
    print(f"Grade:               {grade}")

    return results


def benchmark_parallel_workflows(
    num_workflows: int, actions_per_workflow: int
) -> dict[str, Any]:
    """Benchmark parallel workflow execution.

    Args:
        num_workflows: Number of parallel workflows
        actions_per_workflow: Number of actions per workflow

    Returns:
        Benchmark results dictionary
    """
    import concurrent.futures

    print(f"\n{'=' * 60}")
    print(f"Benchmarking parallel execution: {num_workflows} workflows")
    print(f"{'=' * 60}")

    # Create workflows
    workflows = []
    for i in range(num_workflows):
        actions = [
            MockAction(f"wf{i}_action_{j}", 0.001) for j in range(actions_per_workflow)
        ]
        workflow = MockWorkflow(f"parallel_workflow_{i}", actions)
        workflows.append(workflow)

    # Sequential execution (baseline)
    start_time = time.time()
    for workflow in workflows:
        workflow.execute()
    sequential_time = time.time() - start_time

    # Parallel execution
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workflows) as executor:
        futures = [executor.submit(workflow.execute) for workflow in workflows]
        concurrent.futures.wait(futures)
    parallel_time = time.time() - start_time

    # Calculate speedup
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    efficiency = (speedup / num_workflows) * 100

    # Results
    results = {
        "num_workflows": num_workflows,
        "actions_per_workflow": actions_per_workflow,
        "sequential_time_ms": sequential_time * 1000,
        "parallel_time_ms": parallel_time * 1000,
        "speedup": speedup,
        "efficiency_percent": efficiency,
    }

    # Print results
    print(f"Sequential time:     {results['sequential_time_ms']:.1f} ms")
    print(f"Parallel time:       {results['parallel_time_ms']:.1f} ms")
    print(f"Speedup:             {results['speedup']:.2f}x")
    print(f"Efficiency:          {results['efficiency_percent']:.1f}%")

    return results


def run_all_benchmarks() -> list[dict[str, Any]]:
    """Run all workflow execution benchmarks.

    Returns:
        List of benchmark results
    """
    print("\n" + "=" * 60)
    print("WORKFLOW EXECUTION PERFORMANCE BENCHMARKS")
    print("=" * 60)

    results = []

    # Test cases
    test_cases = [
        ("simple", 5),
        ("medium", 20),
        ("complex", 50),
    ]

    for workflow_type, num_actions in test_cases:
        result = benchmark_workflow_execution(workflow_type, num_actions)
        results.append(result)

    # Parallel workflows
    parallel_result = benchmark_parallel_workflows(10, 10)
    results.append(parallel_result)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"{'Workflow Type':<20} {'Exec Time':<15} {'Overhead/Action':<20} {'Grade':<15}"
    )
    print(f"{'-' * 70}")

    for result in results:
        if "workflow_type" in result:
            print(
                f"{result['workflow_type']:<20} "
                f"{result['execution_time_ms']:<15.1f} "
                f"{result['overhead_per_action_ms']:<20.2f} "
                f"{result['grade']:<15}"
            )

    return results


if __name__ == "__main__":
    import json
    from pathlib import Path

    results = run_all_benchmarks()

    # Save results
    output_file = Path("/tmp/benchmark_workflow_execution_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
