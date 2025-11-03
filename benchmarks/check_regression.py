"""Check for performance regressions in benchmark results.

This script compares current benchmark results against baseline
to detect performance regressions.
"""

import json
import sys
from pathlib import Path
from typing import Any


def load_results(file_path: Path) -> dict[str, Any]:
    """Load benchmark results from JSON file.

    Args:
        file_path: Path to JSON results file

    Returns:
        Benchmark results dictionary
    """
    with open(file_path) as f:
        return json.load(f)


def compare_results(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Compare current results against baseline.

    Args:
        baseline: Baseline benchmark results
        current: Current benchmark results

    Returns:
        Comparison results with regressions detected
    """
    regressions = []
    improvements = []

    # Define regression threshold (10% slower is a regression)
    regression_threshold = 1.10

    # Compare each benchmark suite
    for suite_name in baseline.get("results", {}).keys():
        if suite_name not in current.get("results", {}):
            continue

        baseline_suite = baseline["results"][suite_name]
        current_suite = current["results"][suite_name]

        if not isinstance(baseline_suite, list) or not isinstance(current_suite, list):
            continue

        # Compare each test in the suite
        for i, (baseline_test, current_test) in enumerate(
            zip(baseline_suite, current_suite, strict=False)
        ):
            # Find time metric
            time_keys = [
                "parse_time_ms",
                "execution_time_ms",
                "export_time_ms",
                "import_time_ms",
                "overhead_percent",
            ]

            for key in time_keys:
                if key in baseline_test and key in current_test:
                    baseline_value = baseline_test[key]
                    current_value = current_test[key]

                    # Calculate ratio
                    if baseline_value > 0:
                        ratio = current_value / baseline_value

                        # Check for regression
                        if ratio >= regression_threshold:
                            regression_percent = (ratio - 1.0) * 100
                            regressions.append(
                                {
                                    "suite": suite_name,
                                    "test": baseline_test.get("config_size")
                                    or baseline_test.get("workflow_type")
                                    or baseline_test.get("scenario")
                                    or f"test_{i}",
                                    "metric": key,
                                    "baseline": baseline_value,
                                    "current": current_value,
                                    "regression_percent": regression_percent,
                                }
                            )
                        elif ratio <= 0.9:  # 10% improvement
                            improvement_percent = (1.0 - ratio) * 100
                            improvements.append(
                                {
                                    "suite": suite_name,
                                    "test": baseline_test.get("config_size")
                                    or baseline_test.get("workflow_type")
                                    or baseline_test.get("scenario")
                                    or f"test_{i}",
                                    "metric": key,
                                    "baseline": baseline_value,
                                    "current": current_value,
                                    "improvement_percent": improvement_percent,
                                }
                            )
                    break

    return {
        "regressions": regressions,
        "improvements": improvements,
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
    }


def print_comparison_report(comparison: dict[str, Any]):
    """Print comparison report.

    Args:
        comparison: Comparison results
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "PERFORMANCE REGRESSION CHECK")
    print("=" * 80)

    # Print regressions
    if comparison["regressions"]:
        print("\nREGRESSIONS DETECTED:")
        print("-" * 80)
        print(f"{'Suite':<20} {'Test':<20} {'Metric':<20} {'Regression':<15}")
        print("-" * 80)

        for reg in comparison["regressions"]:
            print(
                f"{reg['suite']:<20} "
                f"{reg['test']:<20} "
                f"{reg['metric']:<20} "
                f"+{reg['regression_percent']:.1f}%"
            )

        print(f"\nTotal regressions: {comparison['regression_count']}")
    else:
        print("\nNo regressions detected! ✓")

    # Print improvements
    if comparison["improvements"]:
        print("\nIMPROVEMENTS DETECTED:")
        print("-" * 80)
        print(f"{'Suite':<20} {'Test':<20} {'Metric':<20} {'Improvement':<15}")
        print("-" * 80)

        for imp in comparison["improvements"]:
            print(
                f"{imp['suite']:<20} "
                f"{imp['test']:<20} "
                f"{imp['metric']:<20} "
                f"-{imp['improvement_percent']:.1f}%"
            )

        print(f"\nTotal improvements: {comparison['improvement_count']}")

    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    # Check for baseline and current results
    baseline_file = Path("/tmp/benchmark_results_baseline.json")
    current_file = Path("/tmp/benchmark_results.json")

    if not baseline_file.exists():
        print("ERROR: Baseline results not found at:", baseline_file)
        print("\nTo create a baseline, run:")
        print("  python -m benchmarks.run_all_benchmarks")
        print(f"  cp /tmp/benchmark_results.json {baseline_file}")
        sys.exit(1)

    if not current_file.exists():
        print("ERROR: Current results not found at:", current_file)
        print("\nRun benchmarks first:")
        print("  python -m benchmarks.run_all_benchmarks")
        sys.exit(1)

    # Load results
    baseline = load_results(baseline_file)
    current = load_results(current_file)

    # Compare
    comparison = compare_results(baseline, current)

    # Print report
    print_comparison_report(comparison)

    # Save comparison
    output_file = Path("/tmp/benchmark_comparison.json")
    with open(output_file, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison saved to: {output_file}")

    # Exit with error code if regressions detected
    if comparison["regression_count"] > 0:
        print("\nERROR: Performance regressions detected!")
        sys.exit(1)
    else:
        print("\nSUCCESS: No performance regressions detected!")
        sys.exit(0)


if __name__ == "__main__":
    main()
