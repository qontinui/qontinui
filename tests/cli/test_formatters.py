"""Tests for result formatters."""

import json
import xml.etree.ElementTree as ET

import pytest

from qontinui.cli.formatters import format_results


@pytest.fixture
def sample_results():
    """Create sample test results."""
    return [
        {
            "workflow_id": "wf-1",
            "workflow_name": "Login Test",
            "success": True,
            "duration": 12.34,
            "error": None,
            "start_time": 1703001234.567,
        },
        {
            "workflow_id": "wf-2",
            "workflow_name": "Failed Test",
            "success": False,
            "duration": 5.67,
            "error": "Workflow execution failed",
            "start_time": 1703001246.907,
        },
    ]


@pytest.fixture
def sample_summary():
    """Create sample summary."""
    return {
        "total_tests": 2,
        "passed": 1,
        "failed": 1,
        "total_duration": 18.01,
        "config_file": "test.json",
        "timestamp": 1703001234.567,
    }


def test_format_json(sample_results, sample_summary):
    """Test JSON formatting."""
    output = format_results(sample_results, sample_summary, "json")

    # Parse and validate JSON
    data = json.loads(output)

    assert "summary" in data
    assert "tests" in data
    assert "timestamp_iso" in data

    assert data["summary"]["total_tests"] == 2
    assert data["summary"]["passed"] == 1
    assert data["summary"]["failed"] == 1

    assert len(data["tests"]) == 2
    assert data["tests"][0]["workflow_name"] == "Login Test"
    assert data["tests"][0]["success"] is True
    assert data["tests"][1]["success"] is False


def test_format_junit(sample_results, sample_summary):
    """Test JUnit XML formatting."""
    output = format_results(sample_results, sample_summary, "junit")

    # Parse XML
    root = ET.fromstring(output)

    # Check testsuites element
    assert root.tag == "testsuites"
    assert root.get("tests") == "2"
    assert root.get("failures") == "1"

    # Check testsuite element
    testsuite = root.find("testsuite")
    assert testsuite is not None
    assert testsuite.get("name") == "Qontinui Workflows"

    # Check test cases
    testcases = testsuite.findall("testcase")
    assert len(testcases) == 2

    # First test (passed)
    assert testcases[0].get("name") == "Login Test"
    assert testcases[0].find("failure") is None

    # Second test (failed)
    assert testcases[1].get("name") == "Failed Test"
    failure = testcases[1].find("failure")
    assert failure is not None
    assert "failed" in failure.get("message").lower()


def test_format_tap(sample_results, sample_summary):
    """Test TAP formatting."""
    output = format_results(sample_results, sample_summary, "tap")

    lines = output.split("\n")

    # Check TAP version
    assert lines[0] == "TAP version 13"

    # Check test plan
    assert lines[1] == "1..2"

    # Check test results
    assert "ok 1 - Login Test" in output
    assert "not ok 2 - Failed Test" in output

    # Check summary
    assert "# Total: 2" in output
    assert "# Passed: 1" in output
    assert "# Failed: 1" in output


def test_format_invalid_type(sample_results, sample_summary):
    """Test error handling for invalid format type."""
    with pytest.raises(ValueError, match="Unknown format type"):
        format_results(sample_results, sample_summary, "invalid")


def test_format_empty_results(sample_summary):
    """Test formatting with no test results."""
    empty_results = []
    summary = {**sample_summary, "total_tests": 0, "passed": 0, "failed": 0}

    # JSON format
    output = format_results(empty_results, summary, "json")
    data = json.loads(output)
    assert len(data["tests"]) == 0

    # JUnit format
    output = format_results(empty_results, summary, "junit")
    root = ET.fromstring(output)
    testsuite = root.find("testsuite")
    assert len(testsuite.findall("testcase")) == 0

    # TAP format
    output = format_results(empty_results, summary, "tap")
    assert "1..0" in output


def test_junit_xml_structure(sample_results, sample_summary):
    """Test detailed JUnit XML structure."""
    output = format_results(sample_results, sample_summary, "junit")
    root = ET.fromstring(output)

    # Check XML declaration is present
    assert output.startswith("<?xml")

    # Check time formatting
    testsuite = root.find("testsuite")
    time_attr = testsuite.get("time")
    assert "." in time_attr  # Should have decimal places

    # Check testcase time
    testcase = testsuite.find("testcase")
    assert float(testcase.get("time")) == pytest.approx(12.34, rel=0.01)


def test_tap_diagnostics(sample_results, sample_summary):
    """Test TAP diagnostic information."""
    output = format_results(sample_results, sample_summary, "tap")

    # Check for YAML-like diagnostics
    assert "---" in output
    assert "..." in output
    assert "duration_ms" in output
    assert "workflow_id" in output

    # Check error is included for failed test
    assert "error:" in output.lower()
