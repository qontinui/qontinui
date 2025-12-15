"""Basic tests for Qontinui CLI commands."""

import json

import pytest
from click.testing import CliRunner

from qontinui.cli.main import main


@pytest.fixture
def cli_runner():
    """Create CLI runner."""
    return CliRunner()


@pytest.fixture
def sample_config(tmp_path):
    """Create a minimal valid config for testing."""
    config = {
        "version": "2.0.0",
        "metadata": {"name": "Test Config", "description": "Test automation"},
        "states": [
            {
                "id": "state-1",
                "name": "Initial State",
                "is_initial": True,
                "outgoing_transitions": [],
                "incoming_transitions": [],
            }
        ],
        "workflows": [
            {
                "id": "wf-1",
                "name": "Test Workflow",
                "version": "1.0.0",
                "actions": [{"id": "act-1", "type": "WAIT", "config": {"duration": 100}}],
                "connections": {},  # Required for v2.0.0 format
            }
        ],
        "transitions": [],
        "images": [],
        "schedules": [],
    }

    config_path = tmp_path / "test_config.json"
    config_path.write_text(json.dumps(config))
    return config_path


def test_cli_help(cli_runner):
    """Test that CLI help works."""
    result = cli_runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Qontinui CLI" in result.output


def test_cli_version(cli_runner):
    """Test that version command works."""
    result = cli_runner.invoke(main, ["--version"])
    assert result.exit_code == 0


def test_validate_command_help(cli_runner):
    """Test validate command help."""
    result = cli_runner.invoke(main, ["validate", "--help"])
    assert result.exit_code == 0
    assert "Validate" in result.output


def test_run_command_help(cli_runner):
    """Test run command help."""
    result = cli_runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "Run a workflow" in result.output


def test_test_command_help(cli_runner):
    """Test test command help."""
    result = cli_runner.invoke(main, ["test", "--help"])
    assert result.exit_code == 0
    assert "test mode" in result.output


def test_validate_valid_config(cli_runner, sample_config):
    """Test validation of a valid config."""
    result = cli_runner.invoke(main, ["validate", str(sample_config)])
    if result.exit_code != 0:
        print(f"\nOutput: {result.output}")
        print(f"\nException: {result.exception}")
        if result.exception:
            import traceback

            traceback.print_exception(
                type(result.exception), result.exception, result.exception.__traceback__
            )
    assert result.exit_code == 0
    assert "valid" in result.output.lower()


def test_validate_missing_config(cli_runner, tmp_path):
    """Test validation with missing config file."""
    missing_config = tmp_path / "nonexistent.json"
    result = cli_runner.invoke(main, ["validate", str(missing_config)])
    assert result.exit_code == 2  # CONFIG_ERROR


def test_validate_invalid_json(cli_runner, tmp_path):
    """Test validation with invalid JSON."""
    invalid_config = tmp_path / "invalid.json"
    invalid_config.write_text("{ invalid json }")

    result = cli_runner.invoke(main, ["validate", str(invalid_config)])
    assert result.exit_code == 2  # CONFIG_ERROR


def test_validate_verbose(cli_runner, sample_config):
    """Test verbose validation output."""
    result = cli_runner.invoke(main, ["validate", str(sample_config), "--verbose"])
    assert result.exit_code == 0
    assert "Test Workflow" in result.output
    assert "Initial State" in result.output


def test_run_command_stream_validation(cli_runner, sample_config):
    """Test that run command validates streaming options."""
    # Test missing cloud-url
    result = cli_runner.invoke(
        main,
        [
            "run",
            str(sample_config),
            "--stream",
            "--api-token",
            "test-token",
            "--project-id",
            "test-project",
        ],
    )
    assert result.exit_code == 2  # CONFIG_ERROR
    assert "--cloud-url is required" in result.output

    # Test missing api-token
    result = cli_runner.invoke(
        main,
        [
            "run",
            str(sample_config),
            "--stream",
            "--cloud-url",
            "http://localhost:8000",
            "--project-id",
            "test-project",
        ],
    )
    assert result.exit_code == 2  # CONFIG_ERROR
    assert "--api-token is required" in result.output

    # Test missing project-id
    result = cli_runner.invoke(
        main,
        [
            "run",
            str(sample_config),
            "--stream",
            "--cloud-url",
            "http://localhost:8000",
            "--api-token",
            "test-token",
        ],
    )
    assert result.exit_code == 2  # CONFIG_ERROR
    assert "--project-id is required" in result.output


def test_test_command_stream_validation(cli_runner, sample_config):
    """Test that test command validates streaming options."""
    # Test missing cloud-url
    result = cli_runner.invoke(
        main,
        [
            "test",
            str(sample_config),
            "--stream",
            "--api-token",
            "test-token",
            "--project-id",
            "test-project",
        ],
    )
    assert result.exit_code == 2  # CONFIG_ERROR
    assert "--cloud-url is required" in result.output

    # Test missing api-token
    result = cli_runner.invoke(
        main,
        [
            "test",
            str(sample_config),
            "--stream",
            "--cloud-url",
            "http://localhost:8000",
            "--project-id",
            "test-project",
        ],
    )
    assert result.exit_code == 2  # CONFIG_ERROR
    assert "--api-token is required" in result.output

    # Test missing project-id
    result = cli_runner.invoke(
        main,
        [
            "test",
            str(sample_config),
            "--stream",
            "--cloud-url",
            "http://localhost:8000",
            "--api-token",
            "test-token",
        ],
    )
    assert result.exit_code == 2  # CONFIG_ERROR
    assert "--project-id is required" in result.output


def test_run_command_help_shows_streaming_options(cli_runner):
    """Test that run command help shows streaming options."""
    result = cli_runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--stream" in result.output
    assert "--cloud-url" in result.output
    assert "--api-token" in result.output
    assert "--project-id" in result.output


def test_test_command_help_shows_streaming_options(cli_runner):
    """Test that test command help shows streaming options."""
    result = cli_runner.invoke(main, ["test", "--help"])
    assert result.exit_code == 0
    assert "--stream" in result.output
    assert "--cloud-url" in result.output
    assert "--api-token" in result.output
    assert "--project-id" in result.output
