"""Shell action executor for executing shell commands and scripts.

This module handles SHELL and SHELL_SCRIPT actions, enabling automation
workflows to execute command-line tools and capture their output.
"""

import json
import logging
import os
import platform
import subprocess
import tempfile
from typing import Any

from ..ai_providers import AIProviderRegistry, AnalysisRequest
from ..config.models.shell_actions import (
    ShellActionConfig,
    ShellScriptActionConfig,
    TriggerAiAnalysisActionConfig,
)
from ..config.schema import Action
from ..exceptions import ActionExecutionError
from .base import ActionExecutorBase
from .registry import register_executor

logger = logging.getLogger(__name__)


@register_executor
class ShellActionExecutor(ActionExecutorBase):
    """Executor for shell commands and scripts.

    Handles:
        - SHELL: Execute a single shell command
        - SHELL_SCRIPT: Execute a multi-line shell script

    Features:
        - Multiple shell support (bash, sh, powershell, cmd, zsh)
        - Output capture in various formats (text, json, lines)
        - Timeout handling
        - Environment variable injection
        - Exit code capture
        - Stderr capture

    Example:
        context = ExecutionContext(...)
        executor = ShellActionExecutor(context)

        # Execute a command
        action = Action(type="SHELL", config={"command": "echo hello"})
        executor.execute(action, ShellActionConfig(command="echo hello"))

        # Execute with JSON output
        action = Action(type="SHELL", config={
            "command": "cat data.json",
            "outputFormat": "json",
            "outputVariable": "data"
        })
        executor.execute(action, config)
    """

    def get_supported_action_types(self) -> list[str]:
        """Get list of shell action types this executor handles.

        Returns:
            List containing: SHELL, SHELL_SCRIPT, TRIGGER_AI_ANALYSIS
        """
        return ["SHELL", "SHELL_SCRIPT", "TRIGGER_AI_ANALYSIS"]

    def execute(self, action: Action, typed_config: Any) -> bool:
        """Execute a shell action with validated configuration.

        Args:
            action: Pydantic Action model with type, config, etc.
            typed_config: Type-specific validated configuration object

        Returns:
            True if action succeeded, False otherwise

        Raises:
            ActionExecutionError: If action execution fails critically
        """
        action_type = action.type

        try:
            if action_type == "SHELL":
                return self._execute_shell(action, typed_config)
            elif action_type == "SHELL_SCRIPT":
                return self._execute_shell_script(action, typed_config)
            elif action_type == "TRIGGER_AI_ANALYSIS":
                return self._execute_trigger_ai_analysis(action, typed_config)
            else:
                raise ActionExecutionError(
                    action_type=action_type,
                    reason=f"Unsupported action type: {action_type}",
                )

        except ActionExecutionError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error executing {action_type}: {e}", exc_info=True)
            raise ActionExecutionError(
                action_type=action_type,
                reason=f"Unexpected error: {e}",
            ) from e

    def _execute_shell(self, action: Action, typed_config: ShellActionConfig | None) -> bool:
        """Execute SHELL action - run a single command.

        Args:
            action: Action model
            typed_config: Validated ShellActionConfig

        Returns:
            True if successful (or if fail_on_error is False)

        Raises:
            ActionExecutionError: If command fails and fail_on_error is True
        """
        if not typed_config:
            raise ActionExecutionError(
                action_type="SHELL",
                reason="SHELL action requires valid ShellActionConfig",
            )

        command = typed_config.command
        if not command:
            raise ActionExecutionError(
                action_type="SHELL",
                reason="Command is required for SHELL action",
            )

        description = typed_config.description or command[:50]
        logger.info(f"Executing SHELL: {description}")

        # Build shell command
        shell_executable, shell_flag = self._get_shell_executable(typed_config.shell)

        # Prepare subprocess arguments
        timeout_seconds = (typed_config.timeout or 30000) / 1000.0
        env = self._build_environment(typed_config.environment)
        # Normalize working directory path for current platform
        cwd = self._normalize_path(typed_config.working_directory)

        try:
            # Execute command
            result = subprocess.run(
                [shell_executable, shell_flag, command],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
                cwd=cwd,
                input=typed_config.stdin,
            )

            # Process output
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.returncode

            logger.debug(f"Command completed with exit code: {exit_code}")

            # Store exit code if variable specified
            if typed_config.exit_code_variable:
                self._store_variable(typed_config.exit_code_variable, exit_code)

            # Store stderr if configured
            if typed_config.capture_stderr and typed_config.stderr_variable:
                self._store_variable(typed_config.stderr_variable, stderr)

            # Process and store stdout
            output = self._process_output(stdout, typed_config.output_format)
            if typed_config.output_variable:
                self._store_variable(typed_config.output_variable, output)

            # Check for errors
            if exit_code != 0 and typed_config.fail_on_error:
                error_msg = f"Command exited with code {exit_code}"
                if stderr:
                    error_msg += f": {stderr[:200]}"
                logger.error(error_msg)
                self._emit_action_failure(
                    action,
                    error_msg,
                    {
                        "exit_code": exit_code,
                        "stderr": stderr[:500] if stderr else None,
                    },
                )
                return False

            self._emit_action_success(
                action,
                {
                    "exit_code": exit_code,
                    "output_length": len(stdout) if stdout else 0,
                    "output_format": typed_config.output_format,
                },
            )
            return True

        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after {timeout_seconds}s"
            logger.error(error_msg)
            self._emit_action_failure(action, error_msg, {"timeout": timeout_seconds})
            if typed_config.fail_on_error:
                return False
            return True

        except Exception as e:
            error_msg = f"Failed to execute command: {e}"
            logger.error(error_msg, exc_info=True)
            self._emit_action_failure(action, error_msg)
            return False

    def _execute_shell_script(
        self, action: Action, typed_config: ShellScriptActionConfig | None
    ) -> bool:
        """Execute SHELL_SCRIPT action - run a multi-line script.

        Args:
            action: Action model
            typed_config: Validated ShellScriptActionConfig

        Returns:
            True if successful

        Raises:
            ActionExecutionError: If script fails
        """
        if not typed_config:
            raise ActionExecutionError(
                action_type="SHELL_SCRIPT",
                reason="SHELL_SCRIPT action requires valid ShellScriptActionConfig",
            )

        script = typed_config.script
        if not script:
            raise ActionExecutionError(
                action_type="SHELL_SCRIPT",
                reason="Script content is required for SHELL_SCRIPT action",
            )

        description = typed_config.description or "Shell script execution"
        logger.info(f"Executing SHELL_SCRIPT: {description}")

        # Determine shell and file extension
        shell = typed_config.shell or "bash"
        extension = self._get_script_extension(shell)

        # Create temporary script file
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=extension,
                delete=False,
            ) as script_file:
                script_file.write(script)
                script_path = script_file.name

            # Make executable on Unix
            if platform.system() != "Windows":
                os.chmod(script_path, 0o755)

            # Build shell command to execute script
            shell_executable, _ = self._get_shell_executable(shell)
            timeout_seconds = (typed_config.timeout or 60000) / 1000.0
            env = self._build_environment(typed_config.environment)
            # Normalize working directory path for current platform
            cwd = self._normalize_path(typed_config.working_directory)

            # Execute script
            result = subprocess.run(
                [shell_executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
                cwd=cwd,
            )

            # Process output (same as SHELL)
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.returncode

            logger.debug(f"Script completed with exit code: {exit_code}")

            # Store variables
            if typed_config.exit_code_variable:
                self._store_variable(typed_config.exit_code_variable, exit_code)

            if typed_config.capture_stderr and typed_config.stderr_variable:
                self._store_variable(typed_config.stderr_variable, stderr)

            output = self._process_output(stdout, typed_config.output_format)
            if typed_config.output_variable:
                self._store_variable(typed_config.output_variable, output)

            # Check for errors
            if exit_code != 0 and typed_config.fail_on_error:
                error_msg = f"Script exited with code {exit_code}"
                if stderr:
                    error_msg += f": {stderr[:200]}"
                logger.error(error_msg)
                self._emit_action_failure(
                    action,
                    error_msg,
                    {
                        "exit_code": exit_code,
                        "stderr": stderr[:500] if stderr else None,
                    },
                )
                return False

            self._emit_action_success(
                action,
                {
                    "exit_code": exit_code,
                    "output_length": len(stdout) if stdout else 0,
                },
            )
            return True

        except subprocess.TimeoutExpired:
            error_msg = f"Script timed out after {timeout_seconds}s"
            logger.error(error_msg)
            self._emit_action_failure(action, error_msg)
            if typed_config.fail_on_error:
                return False
            return True

        except Exception as e:
            error_msg = f"Failed to execute script: {e}"
            logger.error(error_msg, exc_info=True)
            self._emit_action_failure(action, error_msg)
            return False

        finally:
            # Clean up temp file
            try:
                if "script_path" in locals():
                    os.unlink(script_path)
            except OSError:
                pass

    def _get_shell_executable(self, shell: str | None) -> tuple[str, str]:
        """Get shell executable path and command flag.

        Args:
            shell: Shell name (bash, sh, powershell, cmd, zsh)

        Returns:
            Tuple of (executable, flag)
        """
        system = platform.system()

        if shell is None:
            # Use system default
            if system == "Windows":
                return ("cmd.exe", "/c")
            else:
                return ("/bin/sh", "-c")

        shell_map = {
            "bash": ("/bin/bash" if system != "Windows" else "bash.exe", "-c"),
            "sh": ("/bin/sh" if system != "Windows" else "sh.exe", "-c"),
            "zsh": ("/bin/zsh" if system != "Windows" else "zsh.exe", "-c"),
            "powershell": (
                "powershell.exe" if system == "Windows" else "pwsh",
                "-Command",
            ),
            "cmd": ("cmd.exe", "/c"),
        }

        return shell_map.get(shell, ("/bin/sh", "-c"))

    def _get_script_extension(self, shell: str) -> str:
        """Get file extension for script based on shell.

        Args:
            shell: Shell name

        Returns:
            File extension including dot
        """
        extensions = {
            "bash": ".sh",
            "sh": ".sh",
            "zsh": ".zsh",
            "powershell": ".ps1",
            "cmd": ".bat",
        }
        return extensions.get(shell, ".sh")

    def _build_environment(self, extra_env: dict[str, str] | None) -> dict[str, str]:
        """Build environment dict with current env plus extras.

        Args:
            extra_env: Additional environment variables

        Returns:
            Combined environment dict
        """
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        return env

    def _normalize_path(self, path: str | None) -> str | None:
        """Normalize a path for the current platform.

        Handles conversion between WSL paths (/mnt/c/...) and Windows paths (C:\\...)
        depending on the current platform.

        Args:
            path: Path to normalize, or None

        Returns:
            Normalized path for the current platform, or None if input is None
        """
        if path is None:
            return None

        system = platform.system()

        if system == "Windows":
            # Convert WSL path to Windows path if needed
            # /mnt/c/Users/... -> C:\Users\...
            if path.startswith("/mnt/"):
                # Extract drive letter and rest of path
                # /mnt/c/path/to/dir -> c, path/to/dir
                parts = path.split("/")
                if len(parts) >= 3:
                    drive_letter = parts[2]  # 'c' from /mnt/c/...
                    rest_of_path = "/".join(parts[3:])  # path/to/dir
                    windows_path = f"{drive_letter.upper()}:\\{rest_of_path.replace('/', '\\')}"
                    logger.debug(f"Converted WSL path '{path}' to Windows path '{windows_path}'")
                    return windows_path

            # Also handle forward slashes that might be intended as Windows paths
            # /c/Users/... -> C:\Users\...
            if path.startswith("/") and len(path) >= 2 and path[1].isalpha():
                if len(path) == 2 or path[2] == "/":
                    drive_letter = path[1]
                    rest_of_path = path[3:] if len(path) > 3 else ""
                    windows_path = f"{drive_letter.upper()}:\\{rest_of_path.replace('/', '\\')}"
                    logger.debug(f"Converted path '{path}' to Windows path '{windows_path}'")
                    return windows_path

        else:
            # On Unix-like systems, convert Windows paths to Unix paths if needed
            # C:\Users\... -> /mnt/c/Users/...
            if len(path) >= 2 and path[1] == ":" and path[0].isalpha():
                drive_letter = path[0].lower()
                rest_of_path = path[2:].replace("\\", "/")
                if rest_of_path.startswith("/"):
                    rest_of_path = rest_of_path[1:]
                unix_path = f"/mnt/{drive_letter}/{rest_of_path}"
                logger.debug(f"Converted Windows path '{path}' to Unix path '{unix_path}'")
                return unix_path

        return path

    def _process_output(self, output: str, output_format: str | None) -> Any:
        """Process command output based on format specification.

        Args:
            output: Raw command output
            output_format: Format type (text, json, lines, none)

        Returns:
            Processed output
        """
        if not output:
            return None

        format_type = output_format or "text"

        if format_type == "none":
            return None
        elif format_type == "text":
            return output.strip()
        elif format_type == "lines":
            return [line for line in output.strip().split("\n") if line]
        elif format_type == "json":
            try:
                return json.loads(output)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON output: {e}")
                # Return raw text if JSON parsing fails
                return output.strip()
        else:
            return output.strip()

    def _store_variable(self, name: str, value: Any) -> None:
        """Store a value in the variable context.

        Args:
            name: Variable name
            value: Value to store
        """
        if self.context.variable_context:
            self.context.variable_context.set(name, value)
            logger.debug(f"Stored variable '{name}' = {repr(value)[:100]}")

    def _emit_stream_event(self, action: Action, line: str) -> None:
        """Emit a streaming output event for real-time display in runner.

        Args:
            action: The action being executed
            line: A line of output from the subprocess
        """
        self.context.emit_event(
            "ai_output_stream",
            {
                "action_id": action.id,
                "action_type": action.type,
                "line": line,
                "source": "claude",
            },
        )

    def _execute_trigger_ai_analysis(
        self, action: Action, typed_config: TriggerAiAnalysisActionConfig | None
    ) -> bool:
        """Execute TRIGGER_AI_ANALYSIS action - invoke an AI to analyze results.

        This action triggers an AI assistant to analyze the automation results, review
        screenshots and logs, identify issues, and potentially fix them.

        Uses the extensible AI provider system - providers are registered in
        qontinui.ai_providers and can be selected via the 'provider' config option.

        Args:
            action: Action model
            typed_config: Validated TriggerAiAnalysisActionConfig

        Returns:
            True if analysis was triggered successfully

        Raises:
            ActionExecutionError: If AI invocation fails
        """
        if not typed_config:
            raise ActionExecutionError(
                action_type="TRIGGER_AI_ANALYSIS",
                reason="TRIGGER_AI_ANALYSIS action requires valid config",
            )

        # Map legacy provider name "claude" to new name "claude_code"
        provider_name = typed_config.provider or "claude_code"
        if provider_name == "claude":
            provider_name = "claude_code"

        description = typed_config.description or f"Trigger {provider_name} analysis"
        logger.info(f"Executing TRIGGER_AI_ANALYSIS with provider '{provider_name}': {description}")

        # Get provider from registry
        try:
            provider = AIProviderRegistry.get_provider(provider_name)
        except KeyError as e:
            available = AIProviderRegistry.list_available_providers()
            raise ActionExecutionError(
                action_type="TRIGGER_AI_ANALYSIS",
                reason=f"AI provider '{provider_name}' not found. Available: {available}",
            ) from e

        # Check if provider is available
        if not provider.is_available():
            available = AIProviderRegistry.list_available_providers()
            error_msg = f"AI provider '{provider_name}' is not available on this system."
            if available:
                error_msg += f" Available providers: {available}"
            else:
                error_msg += " No providers are currently available."
            logger.error(error_msg)
            self._emit_action_failure(action, error_msg)
            return False

        # Determine working directory
        if typed_config.working_directory:
            working_dir = self._normalize_path(typed_config.working_directory)
            logger.info(f"Using configured working directory: {working_dir}")
        else:
            working_dir = self._find_project_root()
            logger.info(f"Found project root: {working_dir}")

        # Determine results directory
        results_dir = typed_config.results_directory or ".automation-results/latest"
        logger.info(f"Results directory: {results_dir}")

        # Validate results exist
        if working_dir:
            results_path = os.path.join(working_dir, results_dir)
            execution_json = os.path.join(results_path, "execution.json")

            if not os.path.exists(execution_json):
                error_msg = f"No automation results found at {execution_json}"
                logger.warning(error_msg)
                self._emit_action_failure(action, error_msg)
                return False

            # Log execution metadata
            try:
                with open(execution_json) as f:
                    metadata = json.load(f)
                logger.info(
                    f"Found results for workflow: {metadata.get('workflow_name', 'unknown')}"
                )
                logger.info(f"Workflow success: {metadata.get('success', 'unknown')}")
            except Exception as e:
                logger.warning(f"Could not read execution metadata: {e}")

        # Determine the prompt to use
        if typed_config.prompt:
            prompt_text = typed_config.prompt
        else:
            prompt_text = (
                f"Analyze the automation results in {results_dir} and fix any issues found."
            )

        logger.info(f"Using prompt: {prompt_text[:100]}...")

        # Emit the prompt to the runner UI
        self.context.emit_event(
            "ai_output_stream",
            {
                "action_id": action.id,
                "action_type": action.type,
                "line": prompt_text,
                "source": "prompt",
            },
        )

        # Create analysis request
        timeout_seconds = (typed_config.timeout or 600000) / 1000.0
        request = AnalysisRequest(
            prompt=prompt_text,
            working_directory=working_dir,
            results_directory=results_dir,
            timeout_seconds=int(timeout_seconds),
            output_format="text",
        )

        # Execute analysis
        return self._execute_streaming_analysis(action, provider, request, typed_config)

    def _execute_streaming_analysis(
        self,
        action: Action,
        provider: Any,  # AIProvider type, using Any to avoid circular import
        request: AnalysisRequest,
        typed_config: TriggerAiAnalysisActionConfig,
    ) -> bool:
        """Execute AI analysis using the provider system.

        This method calls the provider's analyze() method and streams
        output to the runner UI line-by-line.

        Args:
            action: Action model
            provider: AI provider instance
            request: Analysis request
            typed_config: Original action config for output variable storage

        Returns:
            True if analysis was successful
        """
        try:
            logger.info(f"Starting analysis with provider: {provider.name}")

            # Call the provider's synchronous analyze method
            result = provider.analyze(request)

            # Stream output lines to the runner
            if result.output:
                for line in result.output.splitlines():
                    logger.info(f"[{provider.name}] {line}")
                    self._emit_stream_event(action, line)

            # Store output if variable specified
            if typed_config.output_variable:
                self._store_variable(typed_config.output_variable, result.output)

            # Handle result
            if result.success:
                self._emit_action_success(
                    action,
                    {
                        "provider": provider.name,
                        "output_length": len(result.output) if result.output else 0,
                        "metadata": result.metadata,
                    },
                )
                return True
            else:
                # Analysis failed
                if typed_config.fail_on_issues:
                    error_msg = result.error or "Analysis reported issues"
                    logger.warning(f"Analysis failed: {error_msg}")
                    self._emit_action_failure(action, error_msg, {"provider": provider.name})
                    return False
                else:
                    # Log but don't fail - analysis completing is success
                    logger.info(f"Analysis completed with findings: {result.error}")
                    self._emit_action_success(
                        action,
                        {
                            "provider": provider.name,
                            "output_length": len(result.output) if result.output else 0,
                            "had_issues": True,
                        },
                    )
                    return True

        except subprocess.TimeoutExpired:
            error_msg = f"Analysis timed out after {request.timeout_seconds}s"
            logger.error(error_msg)
            self._emit_action_failure(action, error_msg, {"timeout": request.timeout_seconds})
            return False

        except FileNotFoundError as e:
            error_msg = f"AI provider executable not found: {e}"
            logger.error(error_msg)
            self._emit_action_failure(action, error_msg)
            return False

        except Exception as e:
            error_msg = f"Failed to execute analysis: {e}"
            logger.error(error_msg, exc_info=True)
            self._emit_action_failure(action, error_msg)
            return False

    def _find_project_root(self) -> str | None:
        """Find the project root by looking for CLAUDE.md or .claude directory.

        Returns:
            Path to project root, or None if not found
        """
        # Start from current directory and look upward
        current = os.getcwd()

        for _ in range(10):  # Limit search depth
            if os.path.exists(os.path.join(current, "CLAUDE.md")):
                return current
            if os.path.exists(os.path.join(current, ".claude")):
                return current

            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent

        return None

    def _to_wsl_path(self, windows_path: str) -> str:
        """Convert a Windows path to WSL path format.

        Args:
            windows_path: Windows path (e.g., C:\\Users\\...)

        Returns:
            WSL path (e.g., /mnt/c/Users/...)
        """
        if not windows_path:
            return windows_path

        # Handle paths that are already WSL format
        if windows_path.startswith("/mnt/"):
            return windows_path

        # Convert C:\... to /mnt/c/...
        path = windows_path.replace("\\", "/")
        if len(path) >= 2 and path[1] == ":":
            drive = path[0].lower()
            rest = path[2:]
            if rest.startswith("/"):
                rest = rest[1:]
            return f"/mnt/{drive}/{rest}"

        return path
