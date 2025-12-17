"""
Shell/command execution action configuration models.

This module provides configuration models for executing shell commands
and capturing their output for use in automation workflows.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ShellActionConfig(BaseModel):
    """SHELL action configuration.

    Executes a shell command and captures its output. Supports various
    output formats including text, JSON, and streaming.

    Use cases:
        - Execute CLI tools and capture their output
        - Run scripts and process results
        - Integrate with external APIs via curl
        - Automate command-line applications

    Example config:
        {
            "type": "SHELL",
            "config": {
                "command": "echo 'Hello World'",
                "shell": "bash",
                "outputFormat": "text",
                "timeout": 30000
            }
        }
    """

    # Command to execute
    command: str = Field(
        ...,
        description="The shell command to execute",
    )

    # Shell to use (bash, sh, powershell, cmd)
    shell: Literal["bash", "sh", "powershell", "cmd", "zsh"] | None = Field(
        None,
        description="Shell to use for execution. If None, uses system default.",
    )

    # Working directory
    working_directory: str | None = Field(
        None,
        alias="workingDirectory",
        description="Working directory for command execution",
    )

    # Environment variables to set
    environment: dict[str, str] | None = Field(
        None,
        description="Additional environment variables for the command",
    )

    # Output handling
    output_format: Literal["text", "json", "lines", "none"] | None = Field(
        "text",
        alias="outputFormat",
        description=(
            "How to parse the command output:\n"
            "- text: Return as plain string\n"
            "- json: Parse as JSON object\n"
            "- lines: Split into list of lines\n"
            "- none: Discard output"
        ),
    )

    # Store output in variable
    output_variable: str | None = Field(
        None,
        alias="outputVariable",
        description="Variable name to store the command output",
    )

    # Store exit code in variable
    exit_code_variable: str | None = Field(
        None,
        alias="exitCodeVariable",
        description="Variable name to store the exit code",
    )

    # Capture stderr separately
    capture_stderr: bool | None = Field(
        False,
        alias="captureStderr",
        description="Whether to capture stderr separately from stdout",
    )

    # Store stderr in variable (if capture_stderr is True)
    stderr_variable: str | None = Field(
        None,
        alias="stderrVariable",
        description="Variable name to store stderr output",
    )

    # Timeout in milliseconds
    timeout: int | None = Field(
        30000,
        description="Command timeout in milliseconds (default: 30 seconds)",
    )

    # Fail on non-zero exit code
    fail_on_error: bool | None = Field(
        True,
        alias="failOnError",
        description="Whether to fail the action if command returns non-zero exit code",
    )

    # Input to send to stdin
    stdin: str | None = Field(
        None,
        description="Input to send to the command's stdin",
    )

    # Description for logging
    description: str | None = Field(
        None,
        description="Human-readable description of what this command does",
    )

    model_config = {"populate_by_name": True}


class TriggerAiAnalysisActionConfig(BaseModel):
    """TRIGGER_AI_ANALYSIS action configuration.

    Triggers an AI assistant to analyze the automation results and fix any issues.
    This action is designed to be used at the end of automation workflows to
    enable autonomous debugging and improvement.

    The action:
    1. Reads execution results from .automation-results/latest/
    2. Invokes the configured AI provider with the analyze-automation prompt
    3. The AI reviews screenshots, logs, and errors
    4. The AI fixes issues and re-runs automation if needed

    Supported providers:
        - claude: Claude Code CLI (default)
        - (future providers can be added here)

    Example config:
        {
            "type": "TRIGGER_AI_ANALYSIS",
            "config": {
                "provider": "claude",
                "timeout": 600000,
                "resultsDirectory": ".automation-results/latest"
            }
        }

    Prerequisites:
        - For Claude: Claude Code CLI must be installed
        - On Windows: Claude Code should be installed in WSL with MCP configured
        - Automation results must exist in the results directory
    """

    # AI provider to use for analysis
    provider: Literal["claude"] | None = Field(
        "claude",
        description=(
            "AI provider to use for analysis:\n" "- claude: Claude Code CLI (default)"
        ),
    )

    # Prompt or command to send to the AI
    prompt: str | None = Field(
        None,
        description=(
            "The prompt or command to send to the AI. This can be:\n"
            "- A slash command (e.g., '/analyze-automation', '/qa')\n"
            "- A natural language prompt\n"
            "- Any text that will be passed to the AI\n\n"
            "IMPORTANT: The AI runs with bypassed permissions when executing this prompt."
        ),
    )

    # Timeout in milliseconds (default: 10 minutes for analysis)
    timeout: int | None = Field(
        600000,
        description="Analysis timeout in milliseconds (default: 10 minutes)",
    )

    # Results directory (relative to working directory or absolute)
    results_directory: str | None = Field(
        None,
        alias="resultsDirectory",
        description=(
            "Path to automation results directory. "
            "Defaults to .automation-results/latest relative to project root."
        ),
    )

    # Working directory for AI execution
    working_directory: str | None = Field(
        None,
        alias="workingDirectory",
        description="Working directory for AI execution",
    )

    # Whether to fail the action if analysis reports issues
    fail_on_issues: bool | None = Field(
        False,
        alias="failOnIssues",
        description="Whether to fail the action if the AI reports issues found",
    )

    # Store analysis output in variable
    output_variable: str | None = Field(
        None,
        alias="outputVariable",
        description="Variable name to store the analysis output",
    )

    # Description for logging
    description: str | None = Field(
        None,
        description="Human-readable description of this analysis trigger",
    )

    model_config = {"populate_by_name": True}


class ShellScriptActionConfig(BaseModel):
    """SHELL_SCRIPT action configuration.

    Executes a multi-line shell script. Similar to SHELL but optimized
    for longer scripts with multiple commands.

    Example config:
        {
            "type": "SHELL_SCRIPT",
            "config": {
                "script": "#!/bin/bash\\necho 'Line 1'\\necho 'Line 2'",
                "shell": "bash"
            }
        }
    """

    # Script content
    script: str = Field(
        ...,
        description="The shell script to execute (multi-line supported)",
    )

    # Shell to use
    shell: Literal["bash", "sh", "powershell", "cmd", "zsh"] | None = Field(
        "bash",
        description="Shell to use for script execution",
    )

    # Working directory
    working_directory: str | None = Field(
        None,
        alias="workingDirectory",
        description="Working directory for script execution",
    )

    # Environment variables
    environment: dict[str, str] | None = Field(
        None,
        description="Additional environment variables for the script",
    )

    # Output handling (same as SHELL)
    output_format: Literal["text", "json", "lines", "none"] | None = Field(
        "text",
        alias="outputFormat",
    )
    output_variable: str | None = Field(None, alias="outputVariable")
    exit_code_variable: str | None = Field(None, alias="exitCodeVariable")
    capture_stderr: bool | None = Field(False, alias="captureStderr")
    stderr_variable: str | None = Field(None, alias="stderrVariable")
    timeout: int | None = Field(
        60000, description="Script timeout in ms (default: 60s)"
    )
    fail_on_error: bool | None = Field(True, alias="failOnError")
    description: str | None = None

    model_config = {"populate_by_name": True}
