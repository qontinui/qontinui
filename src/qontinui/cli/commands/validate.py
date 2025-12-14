"""Validate command - Validate a JSON configuration without executing."""

import sys
from pathlib import Path

import click

from ...json_executor import JSONRunner
from ..exit_codes import ExitCode
from ..utils import configure_logging, print_error, print_success


@click.command()
@click.argument("config", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose validation output.",
)
def validate(config: Path, verbose: bool):
    """Validate a JSON configuration file without executing it.

    This command loads and validates a Qontinui configuration file, checking
    for structural errors, missing references, and invalid settings.

    Examples:

      # Validate a configuration
      qontinui validate automation.json

      # Verbose validation with detailed output
      qontinui validate automation.json --verbose
    """
    configure_logging(verbose)

    try:
        click.echo(f"Validating configuration: {config}")

        # Load and validate
        runner = JSONRunner(str(config))
        if not runner.load_configuration():
            print_error("Configuration validation failed")
            sys.exit(ExitCode.CONFIG_ERROR)

        # Get summary
        summary = runner.get_summary()

        click.echo("\nValidation successful!")
        click.echo("\nConfiguration summary:")
        click.echo(f"  Name: {summary.get('config_name', 'Unknown')}")
        click.echo(f"  Version: {summary.get('version', 'Unknown')}")
        click.echo(f"  States: {summary.get('states', 0)}")
        click.echo(f"  Workflows: {summary.get('workflows', 0)}")
        click.echo(f"  Transitions: {summary.get('transitions', 0)}")
        click.echo(f"  Images: {summary.get('images', 0)}")
        click.echo(f"  Schedules: {summary.get('schedules', 0)}")

        if verbose and runner.config:
            click.echo("\nWorkflows:")
            for workflow in runner.config.workflows:
                click.echo(f"  - {workflow.name} (ID: {workflow.id})")
                click.echo(f"    Actions: {len(workflow.actions)}")

            click.echo("\nStates:")
            for state in runner.config.states:
                click.echo(f"  - {state.name} (ID: {state.id})")
                if state.is_initial:
                    click.echo("    [INITIAL STATE]")
                click.echo(f"    Outgoing transitions: {len(state.outgoing_transitions)}")
                click.echo(f"    Incoming transitions: {len(state.incoming_transitions)}")

        print_success("\nConfiguration is valid and ready to run")
        sys.exit(ExitCode.SUCCESS)

    except Exception as e:
        print_error(f"Validation error: {e}")
        if verbose:
            import traceback

            click.echo(traceback.format_exc(), err=True)
        sys.exit(ExitCode.CONFIG_ERROR)
    finally:
        if "runner" in locals():
            runner.cleanup()
