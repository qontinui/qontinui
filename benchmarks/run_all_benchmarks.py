"""Run all performance benchmarks and generate comprehensive report.

This script runs all benchmark suites and generates:
- Console output with results
- JSON report with detailed metrics
- Summary report with pass/fail status
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Import benchmark modules
try:
    from . import (
        benchmark_config_loading,
        benchmark_export_import,
        benchmark_image_finding,
        benchmark_threading,
        benchmark_workflow_execution,
    )
except ImportError:
    # If running as script directly
    import benchmark_config_loading
    import benchmark_export_import
    import benchmark_image_finding
    import benchmark_threading
    import benchmark_workflow_execution


def generate_html_report(all_results: dict[str, Any], output_file: Path):
    """Generate HTML report from benchmark results.

    Args:
        all_results: Dictionary containing all benchmark results
        output_file: Path to output HTML file
    """
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Qontinui Performance Benchmarks</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
        }
        .summary {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .benchmark-section {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #007bff;
            color: white;
        }
        .excellent { color: #28a745; font-weight: bold; }
        .good { color: #17a2b8; font-weight: bold; }
        .fair { color: #ffc107; font-weight: bold; }
        .poor { color: #dc3545; font-weight: bold; }
        .critical { color: #721c24; font-weight: bold; background-color: #f8d7da; }
        .metric {
            display: inline-block;
            margin: 10px 20px 10px 0;
        }
        .metric-label {
            font-weight: bold;
            color: #666;
        }
        .metric-value {
            font-size: 1.2em;
            color: #007bff;
        }
    </style>
</head>
<body>
    <h1>Qontinui Performance Benchmarks</h1>

    <div class="summary">
        <h2>Summary</h2>
        <div class="metric">
            <span class="metric-label">Date:</span>
            <span class="metric-value">{{timestamp}}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Total Duration:</span>
            <span class="metric-value">{{duration:.2f}}s</span>
        </div>
        <div class="metric">
            <span class="metric-label">Benchmark Suites:</span>
            <span class="metric-value">{{num_suites}}</span>
        </div>
    </div>
"""

    html = html.format(
        timestamp=all_results["metadata"]["timestamp"],
        duration=all_results["metadata"]["total_duration"],
        num_suites=len(all_results["results"]),
    )

    # Add each benchmark section
    for suite_name, suite_results in all_results["results"].items():
        html += f"""
    <div class="benchmark-section">
        <h2>{suite_name.replace('_', ' ').title()}</h2>
        <table>
            <tr>
                <th>Test</th>
                <th>Metric</th>
                <th>Value</th>
                <th>Grade</th>
            </tr>
"""

        if isinstance(suite_results, list):
            for result in suite_results:
                # Extract key metrics
                test_name = (
                    result.get("config_size")
                    or result.get("workflow_type")
                    or result.get("scenario")
                    or result.get("operation")
                    or "test"
                )

                # Find the main time metric
                time_key = None
                for key in [
                    "parse_time_ms",
                    "execution_time_ms",
                    "export_time_ms",
                    "import_time_ms",
                    "overhead_percent",
                ]:
                    if key in result:
                        time_key = key
                        break

                if time_key:
                    grade = result.get("grade", "N/A")
                    grade_class = grade.lower()
                    value = result[time_key]
                    unit = "ms" if "time" in time_key else "%"

                    html += f"""
            <tr>
                <td>{test_name}</td>
                <td>{time_key.replace('_', ' ').title()}</td>
                <td>{value:.2f} {unit}</td>
                <td class="{grade_class}">{grade}</td>
            </tr>
"""

        html += """
        </table>
    </div>
"""

    html += """
</body>
</html>
"""

    # Write HTML file
    with open(output_file, "w") as f:
        f.write(html)


def run_all_benchmarks() -> dict[str, Any]:
    """Run all benchmark suites.

    Returns:
        Dictionary containing all benchmark results
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "QONTINUI PERFORMANCE BENCHMARKS")
    print("=" * 80)

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    all_results = {
        "metadata": {
            "timestamp": timestamp,
            "version": "1.0.0",
        },
        "results": {},
    }

    # Run configuration loading benchmarks
    print("\n[1/5] Running configuration loading benchmarks...")
    try:
        config_results = benchmark_config_loading.run_all_benchmarks()
        all_results["results"]["configuration_loading"] = config_results
    except Exception as e:
        print(f"ERROR: Configuration loading benchmarks failed: {e}")
        all_results["results"]["configuration_loading"] = {"error": str(e)}

    # Run workflow execution benchmarks
    print("\n[2/5] Running workflow execution benchmarks...")
    try:
        workflow_results = benchmark_workflow_execution.run_all_benchmarks()
        all_results["results"]["workflow_execution"] = workflow_results
    except Exception as e:
        print(f"ERROR: Workflow execution benchmarks failed: {e}")
        all_results["results"]["workflow_execution"] = {"error": str(e)}

    # Run image finding benchmarks
    print("\n[3/5] Running image finding benchmarks...")
    try:
        image_results = benchmark_image_finding.run_all_benchmarks()
        all_results["results"]["image_finding"] = image_results
    except Exception as e:
        print(f"ERROR: Image finding benchmarks failed: {e}")
        all_results["results"]["image_finding"] = {"error": str(e)}

    # Run export/import benchmarks
    print("\n[4/5] Running export/import benchmarks...")
    try:
        export_results = benchmark_export_import.run_all_benchmarks()
        all_results["results"]["export_import"] = export_results
    except Exception as e:
        print(f"ERROR: Export/import benchmarks failed: {e}")
        all_results["results"]["export_import"] = {"error": str(e)}

    # Run threading benchmarks
    print("\n[5/5] Running threading benchmarks...")
    try:
        threading_results = benchmark_threading.run_all_benchmarks()
        all_results["results"]["threading"] = threading_results
    except Exception as e:
        print(f"ERROR: Threading benchmarks failed: {e}")
        all_results["results"]["threading"] = {"error": str(e)}

    # Calculate total duration
    total_duration = time.time() - start_time
    all_results["metadata"]["total_duration"] = total_duration

    # Print final summary
    print("\n" + "=" * 80)
    print(" " * 30 + "FINAL SUMMARY")
    print("=" * 80)
    print(f"Total benchmark duration: {total_duration:.2f} seconds")
    print(f"Timestamp: {timestamp}")
    print("\nBenchmark suites completed:")

    for suite_name, suite_results in all_results["results"].items():
        if isinstance(suite_results, dict) and "error" in suite_results:
            print(f"  - {suite_name}: FAILED")
        else:
            print(f"  - {suite_name}: SUCCESS")

    return all_results


def main():
    """Main entry point."""
    # Run all benchmarks
    all_results = run_all_benchmarks()

    # Save JSON report
    output_dir = Path("/tmp")
    json_file = output_dir / "benchmark_results.json"

    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nJSON report saved to: {json_file}")

    # Generate HTML report
    html_file = output_dir / "benchmark_report.html"
    try:
        generate_html_report(all_results, html_file)
        print(f"HTML report saved to: {html_file}")
    except Exception as e:
        print(f"ERROR: Failed to generate HTML report: {e}")

    # Print locations
    print("\n" + "=" * 80)
    print("REPORTS GENERATED:")
    print("=" * 80)
    print(f"JSON Report: {json_file}")
    print(f"HTML Report: {html_file}")
    print("\nOpen the HTML report in a browser to view detailed results.")


if __name__ == "__main__":
    main()
