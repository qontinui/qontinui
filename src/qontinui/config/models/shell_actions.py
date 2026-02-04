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


class AIPromptActionConfig(BaseModel):
    """AI_PROMPT action configuration.

    Execute a single AI prompt via the qontinui-runner API.
    This action is designed to invoke AI assistants for various tasks.

    The action:
    1. POSTs to the qontinui-runner /prompts/run API
    2. Polls for completion via GET /task-runs/{id}
    3. Returns the accumulated output_log from the task

    Supported providers:
        - claude: Claude Code via qontinui-runner (default)
        - (future providers can be added here)

    Example config:
        {
            "type": "AI_PROMPT",
            "config": {
                "provider": "claude",
                "prompt": "Analyze the automation results and fix any issues",
                "name": "ai-analysis",
                "maxSessions": 1,
                "timeout": 600000
            }
        }

    Prerequisites:
        - qontinui-runner must be running (default: http://localhost:9876)
        - Claude CLI must be configured in the runner settings
    """

    # AI provider to use for analysis
    provider: Literal["claude"] | None = Field(
        "claude",
        description=(
            "AI provider to use for analysis:\n- claude: Claude Code via runner (default)"
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

    # Task name for the runner
    name: str | None = Field(
        "ai-analysis",
        description="Name for the task (used in runner UI and logs)",
    )

    # Maximum sessions (1 = one-shot, None = unlimited)
    max_sessions: int | None = Field(
        1,
        alias="maxSessions",
        description="Maximum number of sessions. 1 for one-shot, None for unlimited auto-continuation.",
    )

    # Runner API URL
    runner_url: str | None = Field(
        "http://localhost:9876",
        alias="runnerUrl",
        description="URL of the qontinui-runner API",
    )

    # Image paths for analysis
    image_paths: list[str] | None = Field(
        None,
        alias="imagePaths",
        description="Paths to images for the AI to analyze",
    )

    # Video paths for analysis
    video_paths: list[str] | None = Field(
        None,
        alias="videoPaths",
        description="Paths to videos for frame extraction and analysis",
    )

    # Trace path for Playwright traces
    trace_path: str | None = Field(
        None,
        alias="tracePath",
        description="Path to Playwright trace file for analysis",
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
    timeout: int | None = Field(60000, description="Script timeout in ms (default: 60s)")
    fail_on_error: bool | None = Field(True, alias="failOnError")
    description: str | None = None

    model_config = {"populate_by_name": True}
