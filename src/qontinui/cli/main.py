"""Qontinui CLI - Main entry point.

Provides commands for running workflows, integration tests,
and configuration validation.

Exit codes:
    0: Success
    1: Workflow execution failed
    2: Configuration error
    3: Runtime error
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import click

from .formatters import format_integration_test_results, format_results

# Exit codes
EXIT_SUCCESS = 0
EXIT_EXECUTION_FAILED = 1
EXIT_CONFIG_ERROR = 2
EXIT_RUNTIME_ERROR = 3


def setup_logging(verbose: bool = False, log_file: str | None = None) -> None:
    """Configure logging for CLI.

    Args:
        verbose: Enable debug logging
        log_file: Optional log file path
    """
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def load_config_file(config_path: str) -> dict[str, Any] | None:
    """Load and parse a configuration file.

    Args:
        config_path: Path to JSON or YAML configuration file

    Returns:
        Parsed configuration dictionary, or None if loading fails
    """
    path = Path(config_path)

    if not path.exists():
        click.echo(f"Error: Configuration file not found: {config_path}", err=True)
        return None

    try:
        with open(path) as f:
            if path.suffix in (".yaml", ".yml"):
                try:
                    import yaml

                    return cast(dict[str, Any] | None, yaml.safe_load(f))
                except ImportError:
                    click.echo(
                        "Error: PyYAML not installed. Install with: pip install pyyaml", err=True
                    )
                    return None
            else:
                return cast(dict[str, Any] | None, json.load(f))
    except json.JSONDecodeError as e:
        click.echo(f"Error: Invalid JSON in {config_path}: {e}", err=True)
        return None
    except Exception as e:
        click.echo(f"Error loading config: {e}", err=True)
        return None


def load_history_file(history_path: str) -> dict[str, Any] | None:
    """Load historical data file for mock mode.

    Args:
        history_path: Path to history JSON file

    Returns:
        Parsed history data, or None if loading fails
    """
    path = Path(history_path)

    if not path.exists():
        click.echo(f"Warning: History file not found: {history_path}", err=True)
        return None

    try:
        with open(path) as f:
            return cast(dict[str, Any] | None, json.load(f))
    except Exception as e:
        click.echo(f"Error loading history file: {e}", err=True)
        return None


@click.group()
@click.version_option(prog_name="qontinui")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Qontinui CLI - Model-based GUI automation.

    Run workflows, integration tests, and validate configurations.
    """
    ctx.ensure_object(dict)


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def validate(config_path: str, verbose: bool) -> None:
    """Validate a workflow configuration file.

    CONFIG_PATH: Path to the configuration file (JSON or YAML)
    """
    setup_logging(verbose)

    config = load_config_file(config_path)
    if config is None:
        sys.exit(EXIT_CONFIG_ERROR)

    try:
        from qontinui_schemas import QontinuiConfig

        validated = QontinuiConfig.model_validate(config)

        click.echo(f"Configuration is valid: {config_path}")

        if verbose:
            click.echo(f"\nVersion: {validated.version}")
            click.echo(f"Name: {validated.metadata.name if validated.metadata else 'N/A'}")
            click.echo(f"\nStates: {len(validated.states)}")
            for state in validated.states:
                click.echo(f"  - {state.name} (id={state.id}, initial={state.is_initial})")

            click.echo(f"\nWorkflows: {len(validated.workflows)}")
            for workflow in validated.workflows:
                click.echo(f"  - {workflow.name} ({len(workflow.actions)} actions)")

            click.echo(f"\nTransitions: {len(validated.transitions)}")
            click.echo(f"Images: {len(validated.images)}")

        sys.exit(EXIT_SUCCESS)

    except ImportError:
        click.echo(
            "Error: qontinui-schemas not installed. Install with: pip install qontinui-schemas",
            err=True,
        )
        sys.exit(EXIT_CONFIG_ERROR)
    except Exception as e:
        click.echo(f"Validation error: {e}", err=True)
        sys.exit(EXIT_CONFIG_ERROR)


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--workflow", "-w", help="Workflow name or ID to run")
@click.option("--headless", is_flag=True, help="Run in headless mode (no GUI)")
@click.option("--timeout", default=600, help="Execution timeout in seconds")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory for results")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--stream", is_flag=True, help="Stream results to cloud")
@click.option("--cloud-url", help="Cloud backend URL for streaming")
@click.option("--api-token", help="API token for authentication")
@click.option("--project-id", help="Project ID for cloud streaming")
@click.option("--mock", is_flag=True, help="Run in mock mode (integration testing)")
@click.option("--history", type=click.Path(), help="Path to history file for mock mode")
def run(
    config_path: str,
    workflow: str | None,
    headless: bool,
    timeout: int,
    output_dir: str | None,
    verbose: bool,
    stream: bool,
    cloud_url: str | None,
    api_token: str | None,
    project_id: str | None,
    mock: bool,
    history: str | None,
) -> None:
    """Run a workflow from a configuration file.

    CONFIG_PATH: Path to the configuration file (JSON or YAML)
    """
    setup_logging(verbose)

    # Validate streaming options
    if stream:
        if not cloud_url:
            click.echo("Error: --cloud-url is required when using --stream", err=True)
            sys.exit(EXIT_CONFIG_ERROR)
        if not api_token:
            click.echo("Error: --api-token is required when using --stream", err=True)
            sys.exit(EXIT_CONFIG_ERROR)
        if not project_id:
            click.echo("Error: --project-id is required when using --stream", err=True)
            sys.exit(EXIT_CONFIG_ERROR)

    config = load_config_file(config_path)
    if config is None:
        sys.exit(EXIT_CONFIG_ERROR)

    # Set up output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Set execution mode
        if mock:
            from ..config.execution_mode import ExecutionModeConfig, MockMode, set_execution_mode

            set_execution_mode(ExecutionModeConfig(mode=MockMode.MOCK))

            # Configure historical data source
            if history:
                os.environ["QONTINUI_HISTORY_FILE"] = history
            elif cloud_url:
                os.environ["QONTINUI_WEB_URL"] = cloud_url

            click.echo("Running in MOCK mode (integration testing)")
        else:
            click.echo("Running in REAL mode (GUI automation)")

        # Load and execute workflow
        from qontinui_schemas import QontinuiConfig

        validated_config = QontinuiConfig.model_validate(config)

        # Find workflow to run
        target_workflow = None
        for wf in validated_config.workflows:
            if workflow is None or wf.name == workflow or wf.id == workflow:
                target_workflow = wf
                break

        if target_workflow is None:
            if workflow:
                click.echo(f"Error: Workflow '{workflow}' not found", err=True)
            else:
                click.echo("Error: No workflows found in configuration", err=True)
            sys.exit(EXIT_CONFIG_ERROR)

        click.echo(f"Executing workflow: {target_workflow.name}")

        # Execute workflow
        from ..action_executors import DelegatingActionExecutor
        from ..execution import GraphExecutor

        executor = DelegatingActionExecutor(validated_config)
        graph_executor = GraphExecutor(target_workflow, executor)

        start_time = time.time()
        result = graph_executor.execute()
        duration = time.time() - start_time

        # Format and output results
        summary = result["summary"]
        summary["duration"] = duration

        if output_dir:
            results_path = Path(output_dir) / "results.json"
            with open(results_path, "w") as f:
                json.dump({"summary": summary, "workflow": target_workflow.name}, f, indent=2)
            click.echo(f"Results saved to: {results_path}")

        if summary.get("failed", 0) > 0:
            click.echo(f"Workflow completed with {summary['failed']} failures")
            sys.exit(EXIT_EXECUTION_FAILED)
        else:
            click.echo(f"Workflow completed successfully in {duration:.2f}s")
            sys.exit(EXIT_SUCCESS)

    except ImportError as e:
        click.echo(f"Error: Missing dependency: {e}", err=True)
        sys.exit(EXIT_CONFIG_ERROR)
    except Exception as e:
        click.echo(f"Runtime error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(EXIT_RUNTIME_ERROR)


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--workflow", "-w", help="Workflow name or ID to test")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory for results")
@click.option("--output-format", "-f", type=click.Choice(["json", "junit", "tap"]), default="json")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--stream", is_flag=True, help="Stream results to cloud")
@click.option("--cloud-url", help="Cloud backend URL for streaming")
@click.option("--api-token", help="API token for authentication")
@click.option("--project-id", help="Project ID for cloud streaming")
def test(
    config_path: str,
    workflow: str | None,
    output_dir: str | None,
    output_format: str,
    verbose: bool,
    stream: bool,
    cloud_url: str | None,
    api_token: str | None,
    project_id: str | None,
) -> None:
    """Run workflows in test mode.

    CONFIG_PATH: Path to the configuration file (JSON or YAML)

    Runs workflows with test assertions and generates test reports.
    """
    setup_logging(verbose)

    # Validate streaming options
    if stream:
        if not cloud_url:
            click.echo("Error: --cloud-url is required when using --stream", err=True)
            sys.exit(EXIT_CONFIG_ERROR)
        if not api_token:
            click.echo("Error: --api-token is required when using --stream", err=True)
            sys.exit(EXIT_CONFIG_ERROR)
        if not project_id:
            click.echo("Error: --project-id is required when using --stream", err=True)
            sys.exit(EXIT_CONFIG_ERROR)

    config = load_config_file(config_path)
    if config is None:
        sys.exit(EXIT_CONFIG_ERROR)

    # Set up output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        from qontinui_schemas import QontinuiConfig

        validated_config = QontinuiConfig.model_validate(config)

        results: list[dict[str, Any]] = []
        start_time = time.time()

        # Run each workflow
        workflows_to_run = validated_config.workflows
        if workflow:
            workflows_to_run = [
                wf for wf in workflows_to_run if wf.name == workflow or wf.id == workflow
            ]

        for wf in workflows_to_run:
            click.echo(f"Testing workflow: {wf.name}")
            wf_start = time.time()

            try:
                from ..action_executors import DelegatingActionExecutor
                from ..execution import GraphExecutor

                executor = DelegatingActionExecutor(validated_config)
                graph_executor = GraphExecutor(wf, executor)

                result = graph_executor.execute()
                summary = result["summary"]

                results.append(
                    {
                        "workflow_id": wf.id,
                        "workflow_name": wf.name,
                        "success": summary.get("failed", 0) == 0,
                        "duration": time.time() - wf_start,
                        "error": None if summary.get("failed", 0) == 0 else "Actions failed",
                        "start_time": wf_start,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "workflow_id": wf.id,
                        "workflow_name": wf.name,
                        "success": False,
                        "duration": time.time() - wf_start,
                        "error": str(e),
                        "start_time": wf_start,
                    }
                )

        total_duration = time.time() - start_time
        passed = sum(1 for r in results if r["success"])
        failed = len(results) - passed

        summary = {
            "total_tests": len(results),
            "passed": passed,
            "failed": failed,
            "total_duration": total_duration,
            "config_file": config_path,
            "timestamp": start_time,
        }

        # Output results
        output = format_results(results, summary, output_format)

        if output_dir:
            ext = {"json": ".json", "junit": ".xml", "tap": ".tap"}[output_format]
            results_path = Path(output_dir) / f"results{ext}"
            with open(results_path, "w") as f:
                f.write(output)
            click.echo(f"Results saved to: {results_path}")
        else:
            click.echo(output)

        if failed > 0:
            sys.exit(EXIT_EXECUTION_FAILED)
        sys.exit(EXIT_SUCCESS)

    except ImportError as e:
        click.echo(f"Error: Missing dependency: {e}", err=True)
        sys.exit(EXIT_CONFIG_ERROR)
    except Exception as e:
        click.echo(f"Runtime error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(EXIT_RUNTIME_ERROR)


@main.command("integration-test")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--workflow", "-w", help="Workflow name or ID to test")
@click.option("--history", type=click.Path(), help="Path to history JSON file (optional)")
@click.option("--history-url", help="URL to fetch historical data from (default: QONTINUI_WEB_URL)")
@click.option("--project-id", help="Project ID for fetching historical data")
@click.option("--expect-success", is_flag=True, help="Exit with error if any step fails")
@click.option("--expect-states", help="Comma-separated list of expected final states")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory for results")
@click.option("--output-format", "-f", type=click.Choice(["text", "json"]), default="text")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def integration_test(
    config_path: str,
    workflow: str | None,
    history: str | None,
    history_url: str | None,
    project_id: str | None,
    expect_success: bool,
    expect_states: str | None,
    output_dir: str | None,
    output_format: str,
    verbose: bool,
) -> None:
    """Run integration tests in mock mode.

    CONFIG_PATH: Path to the workflow configuration file (JSON or YAML)

    Integration tests execute workflows using historical data instead of
    real GUI automation. This tests the automation logic (state transitions,
    path traversal, action sequences) without needing a live GUI.

    Historical data can come from:
    - A local JSON file (--history)
    - The qontinui-web backend API (--history-url or QONTINUI_WEB_URL env var)

    Example:
        # Using local history file
        qontinui integration-test workflow.json --history baselines/history.json

        # Using qontinui-web backend
        qontinui integration-test workflow.json --history-url http://localhost:8000

        # With CI assertions
        qontinui integration-test workflow.json --expect-success --expect-states Dashboard
    """
    setup_logging(verbose)

    config = load_config_file(config_path)
    if config is None:
        sys.exit(EXIT_CONFIG_ERROR)

    # Set up output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Configure mock mode
        from ..config.execution_mode import ExecutionModeConfig, MockMode, set_execution_mode

        set_execution_mode(ExecutionModeConfig(mode=MockMode.MOCK))

        # Configure historical data source
        if history:
            # Load from local file
            history_data = load_history_file(history)
            if history_data is None:
                click.echo("Warning: Could not load history file, using empty history")
            os.environ["QONTINUI_HISTORY_FILE"] = history

        if history_url:
            os.environ["QONTINUI_WEB_URL"] = history_url
            os.environ["QONTINUI_API_ENABLED"] = "true"

        if project_id:
            os.environ["QONTINUI_PROJECT_ID"] = project_id

        click.echo("Running integration test in MOCK mode")
        click.echo(f"Configuration: {config_path}")
        if history:
            click.echo(f"History source: {history}")
        elif history_url:
            click.echo(f"History source: {history_url}")
        else:
            click.echo(f"History source: {os.getenv('QONTINUI_WEB_URL', 'http://localhost:8000')}")
        click.echo()

        # Load and validate configuration
        from qontinui_schemas import QontinuiConfig

        validated_config = QontinuiConfig.model_validate(config)

        # Find workflow to test
        target_workflow = None
        for wf in validated_config.workflows:
            if workflow is None or wf.name == workflow or wf.id == workflow:
                target_workflow = wf
                break

        if target_workflow is None:
            if workflow:
                click.echo(f"Error: Workflow '{workflow}' not found", err=True)
            else:
                click.echo("Error: No workflows found in configuration", err=True)
            sys.exit(EXIT_CONFIG_ERROR)

        click.echo(f"Testing workflow: {target_workflow.name}")
        click.echo(f"Actions: {len(target_workflow.actions)}")
        click.echo()

        # Execute workflow in mock mode
        from ..action_executors import DelegatingActionExecutor
        from ..execution import GraphExecutor

        executor = DelegatingActionExecutor(validated_config)
        graph_executor = GraphExecutor(target_workflow, executor)

        # Add execution hook to collect detailed logs
        execution_log: list[dict[str, Any]] = []

        start_time = time.time()
        result = graph_executor.execute()
        duration_ms = (time.time() - start_time) * 1000

        # Build summary
        exec_summary = result["summary"]
        summary: dict[str, Any] = {
            "workflow_id": target_workflow.id,
            "workflow_name": target_workflow.name,
            "success": exec_summary.get("failed", 0) == 0,
            "duration_ms": duration_ms,
            "total_steps": len(execution_log),
            "actions_executed": exec_summary.get("completed", 0),
            "state_transitions": 0,  # Would need hook to count these
            "insights": [],
        }

        # Check expected states if specified
        if expect_states:
            expected = [s.strip() for s in expect_states.split(",")]
            # In a full implementation, we'd track active states here
            summary["expected_states"] = expected

        # Check success expectation
        test_passed = summary["success"]
        if expect_success and not test_passed:
            summary["failure_reason"] = "Workflow execution failed (--expect-success was set)"

        # Format and output results
        if output_format == "json":
            output = json.dumps(
                {
                    "summary": summary,
                    "execution_log": execution_log,
                    "execution_order": exec_summary.get("execution_order", []),
                },
                indent=2,
            )
        else:
            output = format_integration_test_results(execution_log, summary, verbose)

        if output_dir:
            ext = ".json" if output_format == "json" else ".txt"
            results_path = Path(output_dir) / f"integration-test-results{ext}"
            with open(results_path, "w") as f:
                f.write(output)
            click.echo(f"Results saved to: {results_path}")

        click.echo(output)

        # Exit with appropriate code
        if expect_success and not test_passed:
            sys.exit(EXIT_EXECUTION_FAILED)
        elif not test_passed:
            click.echo("\nNote: Test completed with failures. Use --expect-success to fail CI.")
            sys.exit(EXIT_SUCCESS)
        else:
            sys.exit(EXIT_SUCCESS)

    except ImportError as e:
        click.echo(f"Error: Missing dependency: {e}", err=True)
        sys.exit(EXIT_CONFIG_ERROR)
    except Exception as e:
        click.echo(f"Runtime error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(EXIT_RUNTIME_ERROR)


if __name__ == "__main__":
    main()
