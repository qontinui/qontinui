"""Test command - Run workflows in test mode with result reporting."""

import json
import sys
import time
from pathlib import Path

import click

from ...json_executor import JSONRunner
from ..exit_codes import ExitCode
from ..formatters import format_results
from ..result_streamer import ResultStreamer
from ..utils import configure_logging, print_error, print_success, print_warning


@click.command()
@click.argument("config", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--workflow",
    "-w",
    help="Workflow name or ID to test. If not specified, runs all workflows.",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "junit", "tap"], case_sensitive=False),
    default="json",
    help="Output format for test results. Default: json",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Directory or file path for test results. Default: stdout",
)
@click.option(
    "--stream",
    is_flag=True,
    help="Enable result streaming to cloud API.",
)
@click.option(
    "--cloud-url",
    type=str,
    help="Qontinui web API URL for streaming results (e.g., http://localhost:8000)",
)
@click.option(
    "--api-token",
    type=str,
    help="Authentication token for cloud API access.",
)
@click.option(
    "--project-id",
    type=str,
    help="Project ID to stream results to.",
)
@click.option(
    "--stream-to",
    type=str,
    help="[DEPRECATED] URL to stream results to. Use --cloud-url instead.",
)
@click.option(
    "--monitor",
    "-m",
    type=int,
    default=0,
    help="Monitor index to run tests on (0-based). Default: 0",
)
@click.option(
    "--timeout",
    "-t",
    type=int,
    help="Maximum execution time per workflow in seconds.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose test output.",
)
@click.option(
    "--headless",
    is_flag=True,
    help="Run in headless mode (for CI environments).",
)
@click.option(
    "--continue-on-failure",
    is_flag=True,
    help="Continue running remaining workflows even if one fails.",
)
def test(
    config: Path,
    workflow: str | None,
    format: str,
    output: Path | None,
    stream: bool,
    cloud_url: str | None,
    api_token: str | None,
    project_id: str | None,
    stream_to: str | None,
    monitor: int,
    timeout: int | None,
    verbose: bool,
    headless: bool,
    continue_on_failure: bool,
):
    """Run workflows in test mode with detailed result reporting.

    This command executes workflows and generates test reports in various formats
    (JSON, JUnit XML, TAP). Results can be saved to files or streamed to a server.

    Examples:

      # Run all workflows and output JSON results
      qontinui test automation.json

      # Run specific workflow with JUnit output
      qontinui test automation.json --workflow "Login Test" --format junit

      # Save results to file
      qontinui test automation.json --format junit --output ./test-results/

      # Stream results to qontinui-web with authentication
      qontinui test automation.json --stream --cloud-url http://localhost:8000 \\
        --api-token YOUR_TOKEN --project-id PROJECT_ID

      # Run all workflows with timeout, continue on failure
      qontinui test automation.json --timeout 60 --continue-on-failure
    """
    configure_logging(verbose)

    # Validate streaming options
    if stream:
        if not cloud_url:
            print_error("--cloud-url is required when --stream is enabled")
            sys.exit(ExitCode.CONFIG_ERROR)
        if not api_token:
            print_error("--api-token is required when --stream is enabled")
            sys.exit(ExitCode.CONFIG_ERROR)
        if not project_id:
            print_error("--project-id is required when --stream is enabled")
            sys.exit(ExitCode.CONFIG_ERROR)

    try:
        # Initialize result streamer if needed
        streamer = None

        # Handle deprecated --stream-to option
        if stream_to:
            print_warning("--stream-to is deprecated. Use --stream --cloud-url --api-token --project-id instead.")
            if not stream:
                streamer = ResultStreamer(stream_to)
                click.echo(f"Results will be streamed to: {stream_to}")

        # Handle new streaming options
        if stream:
            try:
                # Build streaming endpoint URL
                stream_url = f"{cloud_url.rstrip('/')}/api/v1/projects/{project_id}/results"
                streamer = ResultStreamer(stream_url, api_token=api_token)
                click.echo(f"Cloud streaming enabled: {stream_url}")
            except Exception as e:
                print_warning(f"Failed to initialize cloud streaming: {e}")
                if not headless:
                    click.echo("Continuing without streaming...")
                streamer = None

        # Load configuration
        click.echo(f"Loading configuration from: {config}")
        runner = JSONRunner(str(config))

        if not runner.load_configuration():
            print_error("Failed to load configuration")
            sys.exit(ExitCode.CONFIG_ERROR)

        if not runner.config:
            print_error("Configuration not loaded")
            sys.exit(ExitCode.CONFIG_ERROR)

        # Determine which workflows to test
        workflows_to_test = _select_workflows(runner, workflow)
        if not workflows_to_test:
            sys.exit(ExitCode.CONFIG_ERROR)

        # Run tests
        click.echo(f"\nRunning {len(workflows_to_test)} workflow(s) in test mode...")
        test_results = []

        for idx, wf_id in enumerate(workflows_to_test, 1):
            wf_name = next(
                (w.name for w in runner.config.workflows if w.id == wf_id),
                wf_id,
            )
            click.echo(f"\n[{idx}/{len(workflows_to_test)}] Testing: {wf_name}")

            result = _run_test(runner, wf_id, wf_name, monitor, timeout, verbose)
            test_results.append(result)

            # Stream result if enabled
            if streamer:
                try:
                    streamer.stream_result(result)
                except Exception as e:
                    print_warning(f"Failed to stream result: {e}")

            # Check if we should continue
            if not result["success"] and not continue_on_failure:
                print_warning("Test failed and --continue-on-failure not set. Stopping.")
                break

        # Generate test report
        overall_success = all(r["success"] for r in test_results)
        total_time = sum(r["duration"] for r in test_results)

        summary = {
            "total_tests": len(test_results),
            "passed": sum(1 for r in test_results if r["success"]),
            "failed": sum(1 for r in test_results if not r["success"]),
            "total_duration": total_time,
            "config_file": str(config),
            "timestamp": time.time(),
        }

        # Format and output results
        formatted_output = format_results(
            test_results=test_results,
            summary=summary,
            format_type=format,
        )

        if output:
            _save_results(formatted_output, output, format)
            click.echo(f"\nResults saved to: {output}")
        else:
            click.echo("\n" + "=" * 80)
            click.echo("TEST RESULTS")
            click.echo("=" * 80)
            click.echo(formatted_output)

        # Stream summary if enabled
        if streamer:
            try:
                streamer.stream_summary(summary)
            except Exception as e:
                print_warning(f"Failed to stream summary: {e}")

        # Print summary
        click.echo("\n" + "=" * 80)
        click.echo("SUMMARY")
        click.echo("=" * 80)
        click.echo(f"Total tests: {summary['total_tests']}")
        click.echo(f"Passed: {summary['passed']}")
        click.echo(f"Failed: {summary['failed']}")
        click.echo(f"Duration: {summary['total_duration']:.2f}s")

        if overall_success:
            print_success("\nAll tests passed!")
            sys.exit(ExitCode.SUCCESS)
        else:
            print_error("\nSome tests failed")
            sys.exit(ExitCode.TEST_FAILURE)

    except KeyboardInterrupt:
        print_warning("\nTest execution interrupted by user")
        sys.exit(ExitCode.EXECUTION_ERROR)
    except Exception as e:
        print_error(f"Test execution error: {e}")
        if verbose:
            import traceback

            click.echo(traceback.format_exc(), err=True)
        sys.exit(ExitCode.EXECUTION_ERROR)
    finally:
        if "runner" in locals():
            runner.cleanup()


def _select_workflows(runner: JSONRunner, workflow_name: str | None) -> list[str]:
    """Select which workflows to test.

    Args:
        runner: JSONRunner instance with loaded config
        workflow_name: Optional workflow name/ID from user

    Returns:
        List of workflow IDs to test
    """
    if not runner.config:
        return []

    workflows = runner.config.workflows

    if not workflows:
        print_error("No workflows found in configuration")
        return []

    # If no workflow specified, test all
    if not workflow_name:
        click.echo(f"Testing all {len(workflows)} workflows")
        return [wf.id for wf in workflows]

    # Search by ID or name
    for wf in workflows:
        if wf.id == workflow_name or wf.name == workflow_name:
            return [wf.id]

    # Not found
    print_error(f"Workflow not found: {workflow_name}")
    click.echo("\nAvailable workflows:")
    for wf in workflows:
        click.echo(f"  - {wf.name} (ID: {wf.id})")

    return []


def _run_test(
    runner: JSONRunner,
    workflow_id: str,
    workflow_name: str,
    monitor: int,
    timeout: int | None,
    verbose: bool,
) -> dict:
    """Run a single workflow test.

    Args:
        runner: JSONRunner instance
        workflow_id: Workflow ID to test
        workflow_name: Workflow name for reporting
        monitor: Monitor index
        timeout: Optional timeout in seconds
        verbose: Verbose output flag

    Returns:
        Test result dictionary
    """
    start_time = time.time()
    result = {
        "workflow_id": workflow_id,
        "workflow_name": workflow_name,
        "success": False,
        "duration": 0.0,
        "error": None,
        "start_time": start_time,
    }

    try:
        if timeout:
            success = _run_with_timeout(runner, workflow_id, monitor, timeout, verbose)
        else:
            success = runner.run(process_id=workflow_id, monitor_index=monitor)

        result["success"] = success
        result["duration"] = time.time() - start_time

        if success:
            click.echo(f"  ✓ Passed ({result['duration']:.2f}s)")
        else:
            click.echo(f"  ✗ Failed ({result['duration']:.2f}s)")

    except Exception as e:
        result["success"] = False
        result["duration"] = time.time() - start_time
        result["error"] = str(e)
        click.echo(f"  ✗ Error: {e}")

    return result


def _run_with_timeout(
    runner: JSONRunner,
    workflow_id: str,
    monitor: int,
    timeout: int,
    verbose: bool,
) -> bool:
    """Run workflow with timeout."""
    import threading

    result = {"success": False, "error": None}

    def run_workflow():
        try:
            result["success"] = runner.run(process_id=workflow_id, monitor_index=monitor)
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=run_workflow, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        runner.request_stop()
        thread.join(timeout=5)
        if verbose:
            click.echo(f"  ⚠ Timeout after {timeout}s")
        return False

    if result["error"]:
        raise result["error"]

    return result["success"]


def _save_results(content: str, output_path: Path, format_type: str):
    """Save test results to file.

    Args:
        content: Formatted test results
        output_path: Path to save to (file or directory)
        format_type: Result format type
    """
    # If output is a directory, create filename
    if output_path.is_dir() or str(output_path).endswith(("/", "\\")):
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if format_type == "junit":
            filename = f"qontinui_test_results_{timestamp}.xml"
        elif format_type == "tap":
            filename = f"qontinui_test_results_{timestamp}.tap"
        else:
            filename = f"qontinui_test_results_{timestamp}.json"
        output_path = output_path / filename
    else:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(content, encoding="utf-8")
