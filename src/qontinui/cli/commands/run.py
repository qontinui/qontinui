"""Run command - Execute a workflow from a JSON configuration."""

import sys
import time
from pathlib import Path

import click

from ...json_executor import JSONRunner
from ..exit_codes import ExitCode
from ..result_streamer import ResultStreamer
from ..utils import configure_logging, print_error, print_success, print_warning


@click.command()
@click.argument("config", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--workflow",
    "-w",
    help="Workflow name or ID to execute. If not specified, runs the first workflow.",
)
@click.option(
    "--monitor",
    "-m",
    type=int,
    default=0,
    help="Monitor index to run automation on (0-based). Default: 0 (primary monitor)",
)
@click.option(
    "--timeout",
    "-t",
    type=int,
    help="Maximum execution time in seconds. Automation will be stopped if it exceeds this.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging output.",
)
@click.option(
    "--headless",
    is_flag=True,
    help="Run in headless mode (for CI environments). Disables interactive prompts.",
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
def run(
    config: Path,
    workflow: str | None,
    monitor: int,
    timeout: int | None,
    verbose: bool,
    headless: bool,
    stream: bool,
    cloud_url: str | None,
    api_token: str | None,
    project_id: str | None,
):
    """Run a workflow from a JSON configuration file.

    This command loads a Qontinui configuration and executes a specific workflow.
    By default, it runs the first workflow in the configuration.

    Examples:

      # Run the first workflow in the config
      qontinui run automation.json

      # Run a specific workflow by name
      qontinui run automation.json --workflow "Login Workflow"

      # Run on a specific monitor with timeout
      qontinui run automation.json --workflow "Test" --monitor 1 --timeout 300

      # Verbose headless execution for CI
      qontinui run automation.json --verbose --headless

      # Stream results to cloud API
      qontinui run automation.json --stream --cloud-url http://localhost:8000 \\
        --api-token YOUR_TOKEN --project-id PROJECT_ID
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
        # Initialize result streamer if enabled
        streamer = None
        if stream:
            try:
                # Build streaming endpoint URL
                if cloud_url:
                    stream_url = f"{cloud_url.rstrip('/')}/api/v1/projects/{project_id}/results"
                    streamer = ResultStreamer(stream_url, api_token=api_token)
                    click.echo(f"Cloud streaming enabled: {stream_url}")
                else:
                    raise ValueError("cloud_url is required for streaming")
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

        # Determine which workflow to run
        if not runner.config:
            print_error("Configuration not loaded")
            sys.exit(ExitCode.CONFIG_ERROR)

        workflow_id = _select_workflow(runner, workflow)
        if not workflow_id:
            sys.exit(ExitCode.CONFIG_ERROR)

        # Get workflow name for reporting
        workflow_name = next(
            (w.name for w in runner.config.workflows if w.id == workflow_id),
            workflow_id,
        )

        # Run the workflow
        click.echo(f"\nStarting workflow: {workflow_name}")
        click.echo(f"Monitor: {monitor}")
        if timeout:
            click.echo(f"Timeout: {timeout}s")

        start_time = time.time()
        success = False
        error_message = None

        try:
            if timeout:
                # Run with timeout using signal or threading
                success = _run_with_timeout(runner, workflow_id, monitor, timeout)
            else:
                success = runner.run(process_id=workflow_id, monitor_index=monitor)

        except KeyboardInterrupt:
            print_warning("\nExecution interrupted by user")
            error_message = "Execution interrupted by user"
            success = False
        except Exception as e:
            error_message = str(e)
            success = False
            raise

        elapsed_time = time.time() - start_time

        # Stream result if enabled
        if streamer:
            result = {
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                "success": success,
                "duration": elapsed_time,
                "start_time": start_time,
                "error": error_message,
            }
            try:
                streamer.stream_result(result)
            except Exception as e:
                print_warning(f"Failed to stream result: {e}")

        # Report results
        click.echo(f"\nExecution completed in {elapsed_time:.2f}s")

        if success:
            print_success("Workflow executed successfully")
            sys.exit(ExitCode.SUCCESS)
        else:
            print_error("Workflow execution failed")
            sys.exit(ExitCode.TEST_FAILURE)

    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if verbose:
            import traceback

            click.echo(traceback.format_exc(), err=True)
        sys.exit(ExitCode.EXECUTION_ERROR)
    finally:
        if "runner" in locals():
            runner.cleanup()


def _select_workflow(runner: JSONRunner, workflow_name: str | None) -> str | None:
    """Select which workflow to run.

    Args:
        runner: JSONRunner instance with loaded config
        workflow_name: Optional workflow name/ID from user

    Returns:
        Workflow ID to execute, or None if not found
    """
    if not runner.config:
        return None

    workflows = runner.config.workflows

    if not workflows:
        print_error("No workflows found in configuration")
        return None

    # If no workflow specified, use the first one
    if not workflow_name:
        workflow_id = workflows[0].id
        click.echo(f"No workflow specified, using first workflow: {workflows[0].name}")
        return workflow_id

    # Search by ID or name
    for wf in workflows:
        if wf.id == workflow_name or wf.name == workflow_name:
            return wf.id

    # Not found
    print_error(f"Workflow not found: {workflow_name}")
    click.echo("\nAvailable workflows:")
    for wf in workflows:
        click.echo(f"  - {wf.name} (ID: {wf.id})")

    return None


def _run_with_timeout(runner: JSONRunner, workflow_id: str, monitor: int, timeout: int) -> bool:
    """Run workflow with a timeout.

    Args:
        runner: JSONRunner instance
        workflow_id: Workflow ID to execute
        monitor: Monitor index
        timeout: Timeout in seconds

    Returns:
        True if successful, False otherwise
    """
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
        # Timeout occurred
        runner.request_stop()
        thread.join(timeout=5)  # Give it a few seconds to stop gracefully
        print_error(f"Workflow execution timed out after {timeout}s")
        return False

    if result["error"]:
        error = result["error"]
        if isinstance(error, BaseException):
            raise error

    return bool(result["success"])
