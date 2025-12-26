"""Result formatters for CLI output.

Provides formatting for test results in multiple formats:
- JSON: Machine-readable format
- JUnit XML: CI/CD integration format
- TAP: Test Anything Protocol format
"""

import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any


def format_results(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    format_type: str,
) -> str:
    """Format test results in the specified format.

    Args:
        results: List of test result dictionaries
        summary: Summary statistics dictionary
        format_type: Output format ("json", "junit", or "tap")

    Returns:
        Formatted string output

    Raises:
        ValueError: If format_type is not recognized
    """
    if format_type == "json":
        return _format_json(results, summary)
    elif format_type == "junit":
        return _format_junit(results, summary)
    elif format_type == "tap":
        return _format_tap(results, summary)
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def _format_json(results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    """Format results as JSON.

    Args:
        results: List of test result dictionaries
        summary: Summary statistics dictionary

    Returns:
        JSON formatted string
    """
    timestamp = summary.get("timestamp", time.time())
    timestamp_iso = datetime.fromtimestamp(timestamp).isoformat()

    output = {
        "summary": summary,
        "tests": results,
        "timestamp_iso": timestamp_iso,
    }

    return json.dumps(output, indent=2)


def _format_junit(results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    """Format results as JUnit XML.

    Args:
        results: List of test result dictionaries
        summary: Summary statistics dictionary

    Returns:
        JUnit XML formatted string
    """
    # Create root testsuites element
    testsuites = ET.Element("testsuites")
    testsuites.set("tests", str(summary.get("total_tests", 0)))
    testsuites.set("failures", str(summary.get("failed", 0)))
    testsuites.set("time", f"{summary.get('total_duration', 0):.3f}")

    # Create testsuite element
    testsuite = ET.SubElement(testsuites, "testsuite")
    testsuite.set("name", "Qontinui Workflows")
    testsuite.set("tests", str(summary.get("total_tests", 0)))
    testsuite.set("failures", str(summary.get("failed", 0)))
    testsuite.set("time", f"{summary.get('total_duration', 0):.3f}")

    # Add test cases
    for result in results:
        testcase = ET.SubElement(testsuite, "testcase")
        testcase.set("name", result.get("workflow_name", "Unknown"))
        testcase.set("classname", "qontinui.workflows")
        testcase.set("time", f"{result.get('duration', 0):.2f}")

        if not result.get("success", False):
            failure = ET.SubElement(testcase, "failure")
            failure.set("message", result.get("error", "Workflow execution failed"))
            failure.set("type", "WorkflowError")
            failure.text = result.get("error", "")

    # Generate XML string with declaration
    xml_string = ET.tostring(testsuites, encoding="unicode")
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_string}'


def _format_tap(results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    """Format results as TAP (Test Anything Protocol).

    Args:
        results: List of test result dictionaries
        summary: Summary statistics dictionary

    Returns:
        TAP formatted string
    """
    lines = []

    # TAP version header
    lines.append("TAP version 13")

    # Test plan
    total = summary.get("total_tests", 0)
    lines.append(f"1..{total}")

    # Test results
    for i, result in enumerate(results, 1):
        name = result.get("workflow_name", "Unknown")
        success = result.get("success", False)

        if success:
            lines.append(f"ok {i} - {name}")
        else:
            lines.append(f"not ok {i} - {name}")

        # Add YAML diagnostics block
        lines.append("  ---")
        lines.append(f"  workflow_id: {result.get('workflow_id', 'unknown')}")
        lines.append(f"  duration_ms: {result.get('duration', 0) * 1000:.2f}")
        if not success and result.get("error"):
            lines.append(f"  error: {result.get('error')}")
        lines.append("  ...")

    # Summary
    lines.append("")
    lines.append(f"# Total: {summary.get('total_tests', 0)}")
    lines.append(f"# Passed: {summary.get('passed', 0)}")
    lines.append(f"# Failed: {summary.get('failed', 0)}")

    return "\n".join(lines)


def format_integration_test_results(
    execution_log: list[dict[str, Any]],
    summary: dict[str, Any],
    verbose: bool = False,
) -> str:
    """Format integration test results for console output.

    Provides a human-readable step-by-step view of the workflow execution
    in mock mode, showing state transitions, path calculations, and action results.

    Args:
        execution_log: List of execution step dictionaries
        summary: Summary statistics dictionary
        verbose: Include detailed information for each step

    Returns:
        Formatted string for console output
    """
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append("INTEGRATION TEST RESULTS")
    lines.append("=" * 60)
    lines.append("")

    # Summary
    lines.append(f"Workflow: {summary.get('workflow_name', 'Unknown')}")
    lines.append("Mode: MOCK (historical data playback)")
    lines.append(f"Duration: {summary.get('duration_ms', 0):.0f}ms (virtual)")
    lines.append("")

    # Stats
    lines.append("--- Statistics ---")
    lines.append(f"Total Steps: {summary.get('total_steps', 0)}")
    lines.append(f"Actions Executed: {summary.get('actions_executed', 0)}")
    lines.append(f"State Transitions: {summary.get('state_transitions', 0)}")
    lines.append(f"Success: {'Yes' if summary.get('success', False) else 'No'}")
    lines.append("")

    # Execution log
    if execution_log:
        lines.append("--- Execution Log ---")
        for i, step in enumerate(execution_log, 1):
            step_type = step.get("type", "unknown")
            lines.append(f"[Step {i}] {step_type.upper()}")

            if step_type == "state_discovery":
                active_states = step.get("active_states", [])
                lines.append(f"  Active states: {{{', '.join(active_states)}}}")
                if step.get("initial_states_match"):
                    lines.append("  Initial states matched workflow definition")

            elif step_type == "path_calculation":
                target = step.get("target_state", "Unknown")
                lines.append(f"  Target: {target}")
                if verbose and step.get("available_paths"):
                    lines.append("  Available paths:")
                    for path in step.get("available_paths", []):
                        cost = path.get("cost", "?")
                        route = " -> ".join(path.get("route", []))
                        lines.append(f"    - {route} [cost: {cost}]")
                selected = step.get("selected_path")
                if selected:
                    lines.append(f"  Selected: {selected}")

            elif step_type == "action":
                action = step.get("action", {})
                action_type = action.get("type", "UNKNOWN")
                action_name = action.get("name", action.get("id", ""))
                lines.append(f"  Action: {action_type} {action_name}")
                if step.get("historical_stats"):
                    stats = step.get("historical_stats", {})
                    lines.append(
                        f"    Historical: {stats.get('total_runs', 0)} runs, "
                        f"{stats.get('success_rate', 0):.0f}% success"
                    )
                result = step.get("result", {})
                if result.get("success"):
                    lines.append("    Result: SUCCESS")
                else:
                    lines.append(f"    Result: FAILED - {result.get('error', 'Unknown error')}")

            elif step_type == "state_update":
                activated = step.get("activated", [])
                deactivated = step.get("deactivated", [])
                if activated:
                    lines.append(f"  Activated: {{{', '.join(activated)}}}")
                if deactivated:
                    lines.append(f"  Deactivated: {{{', '.join(deactivated)}}}")
                new_states = step.get("new_active_states", [])
                lines.append(f"  Active states: {{{', '.join(new_states)}}}")

            lines.append("")

    # Insights
    if summary.get("insights"):
        lines.append("--- Insights ---")
        for insight in summary.get("insights", []):
            severity = insight.get("severity", "info")
            message = insight.get("message", "")
            lines.append(f"[{severity.upper()}] {message}")
        lines.append("")

    # Final status
    lines.append("=" * 60)
    if summary.get("success"):
        lines.append("RESULT: PASSED")
    else:
        lines.append("RESULT: FAILED")
        if summary.get("failure_reason"):
            lines.append(f"Reason: {summary.get('failure_reason')}")
    lines.append("=" * 60)

    return "\n".join(lines)
