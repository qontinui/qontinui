"""Result formatters for test output.

Provides formatters for different test result formats:
- JSON: Detailed machine-readable results
- JUnit XML: Standard format for CI/CD systems (Jenkins, GitLab CI, etc.)
- TAP: Test Anything Protocol for Perl-compatible test runners
"""

import json
import xml.etree.ElementTree as ET
from datetime import datetime


def format_results(test_results: list[dict], summary: dict, format_type: str) -> str:
    """Format test results in the specified format.

    Args:
        test_results: List of test result dictionaries
        summary: Summary statistics
        format_type: Output format ('json', 'junit', or 'tap')

    Returns:
        Formatted test results as string

    Raises:
        ValueError: If format_type is not recognized
    """
    if format_type == "json":
        return _format_json(test_results, summary)
    elif format_type == "junit":
        return _format_junit(test_results, summary)
    elif format_type == "tap":
        return _format_tap(test_results, summary)
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def _format_json(test_results: list[dict], summary: dict) -> str:
    """Format results as JSON.

    Args:
        test_results: List of test result dictionaries
        summary: Summary statistics

    Returns:
        JSON-formatted string
    """
    output = {
        "summary": summary,
        "tests": test_results,
        "timestamp_iso": datetime.fromtimestamp(summary["timestamp"]).isoformat(),
    }

    return json.dumps(output, indent=2)


def _format_junit(test_results: list[dict], summary: dict) -> str:
    """Format results as JUnit XML.

    Args:
        test_results: List of test result dictionaries
        summary: Summary statistics

    Returns:
        JUnit XML formatted string
    """
    # Create root testsuites element
    testsuites = ET.Element("testsuites")
    testsuites.set("name", "Qontinui Test Suite")
    testsuites.set("tests", str(summary["total_tests"]))
    testsuites.set("failures", str(summary["failed"]))
    testsuites.set("time", f"{summary['total_duration']:.3f}")
    testsuites.set("timestamp", datetime.fromtimestamp(summary["timestamp"]).isoformat())

    # Create testsuite element
    testsuite = ET.SubElement(testsuites, "testsuite")
    testsuite.set("name", "Qontinui Workflows")
    testsuite.set("tests", str(summary["total_tests"]))
    testsuite.set("failures", str(summary["failed"]))
    testsuite.set("time", f"{summary['total_duration']:.3f}")
    testsuite.set("file", summary["config_file"])

    # Add test cases
    for result in test_results:
        testcase = ET.SubElement(testsuite, "testcase")
        testcase.set("name", result["workflow_name"])
        testcase.set("classname", f"qontinui.{result['workflow_id']}")
        testcase.set("time", f"{result['duration']:.3f}")

        if not result["success"]:
            failure = ET.SubElement(testcase, "failure")
            failure.set("message", "Workflow execution failed")
            if result.get("error"):
                failure.set("type", "ExecutionError")
                failure.text = result["error"]
            else:
                failure.set("type", "WorkflowFailure")
                failure.text = "Workflow did not complete successfully"

    # Format with indentation
    _indent_xml(testsuites)
    return ET.tostring(testsuites, encoding="unicode", xml_declaration=True)


def _format_tap(test_results: list[dict], summary: dict) -> str:
    """Format results as TAP (Test Anything Protocol).

    Args:
        test_results: List of test result dictionaries
        summary: Summary statistics

    Returns:
        TAP formatted string
    """
    lines = []

    # TAP version
    lines.append("TAP version 13")

    # Test plan
    lines.append(f"1..{summary['total_tests']}")

    # Test results
    for idx, result in enumerate(test_results, 1):
        if result["success"]:
            lines.append(f"ok {idx} - {result['workflow_name']}")
        else:
            lines.append(f"not ok {idx} - {result['workflow_name']}")

        # Add diagnostic information
        lines.append("  ---")
        lines.append(f"  duration_ms: {result['duration'] * 1000:.0f}")
        lines.append(f"  workflow_id: {result['workflow_id']}")
        if result.get("error"):
            lines.append(f"  error: {result['error']}")
        lines.append("  ...")

    # Add summary as diagnostic
    lines.append("")
    lines.append("# Test Summary")
    lines.append(f"# Total: {summary['total_tests']}")
    lines.append(f"# Passed: {summary['passed']}")
    lines.append(f"# Failed: {summary['failed']}")
    lines.append(f"# Duration: {summary['total_duration']:.2f}s")

    return "\n".join(lines)


def _indent_xml(elem: ET.Element, level: int = 0):
    """Add pretty-printing indentation to XML element tree.

    Args:
        elem: XML element to indent
        level: Current indentation level
    """
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent
