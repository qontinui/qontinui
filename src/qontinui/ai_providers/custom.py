"""Custom command AI provider implementation.

This provider allows users to define their own commands for AI analysis,
enabling integration with any AI system via custom scripts or commands.

Security:
    IMPORTANT: The shell=True option should be avoided when possible.
    When shell=False (default), commands are executed directly without shell
    interpretation, which prevents command injection attacks.

    When shell=True is required (e.g., for shell features like pipes or
    environment variable expansion), placeholder values are escaped using
    shlex.quote() to mitigate command injection. However, this escaping is
    only effective on POSIX systems and does not protect against all attack
    vectors. The command template itself is NOT validated.

    Recommendation: Use shell=False with properly structured arguments.
"""

import asyncio
import logging
import re
import shlex
import subprocess
import sys
import warnings
from collections.abc import AsyncIterator

from .base import AIProvider, AnalysisRequest, AnalysisResult

logger = logging.getLogger(__name__)

# Security warning for shell=True usage
_SHELL_WARNING = (
    "Using shell=True with CustomCommandProvider. This can be a security risk "
    "if the command template or placeholder values come from untrusted sources. "
    "Consider using shell=False with explicit argument lists instead."
)

# Pattern to detect potentially dangerous characters in placeholder values
# Control characters (except tab, newline, carriage return) could be used for injection
_DANGEROUS_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _validate_placeholder_value(name: str, value: str) -> None:
    """Validate a placeholder value for dangerous characters.

    Args:
        name: Name of the placeholder for error messages
        value: The value to validate

    Raises:
        ValueError: If the value contains dangerous characters
    """
    if _DANGEROUS_CHARS_PATTERN.search(value):
        raise ValueError(
            f"Placeholder {name} contains potentially dangerous control characters. "
            "This could indicate an injection attack attempt."
        )


class CustomCommandProvider(AIProvider):
    """AI provider that executes a user-defined command.

    This provider enables integration with any AI system by executing
    a custom command with configurable arguments.

    The command can use placeholders that will be replaced:
        {prompt} - The analysis prompt
        {working_directory} - The working directory
        {results_directory} - The automation results directory
        {timeout} - The timeout in seconds

    Example:
        # Use a custom Python script
        provider = CustomCommandProvider(
            command="python /path/to/my_ai.py --prompt {prompt}"
        )

        # Use a shell script
        provider = CustomCommandProvider(
            command="/path/to/analyze.sh {results_directory}"
        )

    Configuration:
        command: The command template to execute
        shell: Whether to execute via shell (default: False)
        capture_stderr: Whether to capture stderr (default: True)

    Security Warning:
        Using shell=True is discouraged. If you must use it, ensure that
        all placeholder values come from trusted sources only.
    """

    def __init__(
        self,
        command: str,
        shell: bool = False,
        capture_stderr: bool = True,
    ):
        """Initialize custom command provider.

        Args:
            command: Command template with optional placeholders
            shell: Execute via shell (SECURITY WARNING: use with caution)
            capture_stderr: Capture stderr output
        """
        self._command = command
        self._shell = shell
        self._capture_stderr = capture_stderr

        # Emit security warning when shell=True
        if shell:
            warnings.warn(_SHELL_WARNING, SecurityWarning, stacklevel=2)
            logger.warning(_SHELL_WARNING)

    @property
    def name(self) -> str:
        """Get the provider name."""
        return "custom"

    @property
    def description(self) -> str:
        """Get provider description."""
        return f"Custom command: {self._command[:50]}..."

    def is_available(self) -> bool:
        """Check if the custom command is available.

        Returns:
            Always True (command availability checked at runtime)
        """
        # We cannot pre-check availability without knowing the placeholders
        # Runtime execution will fail if command is invalid
        return True

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Run analysis synchronously.

        Args:
            request: The analysis request

        Returns:
            The analysis result
        """
        result = AnalysisResult(success=False, provider=self.name)

        try:
            # Build command with placeholders replaced (includes validation)
            cmd, cmd_args = self._build_command(request)

            logger.info("Executing custom AI command")
            logger.debug(f"Command: {cmd if self._shell else cmd_args}")

            # Execute command - prefer list form over shell string
            if self._shell:
                # Shell execution - use list form with explicit shell for safety
                # This avoids shell=True by explicitly invoking the shell
                if sys.platform == "win32":
                    shell_cmd = ["cmd.exe", "/c", cmd]
                else:
                    shell_cmd = ["/bin/sh", "-c", cmd]
                proc = subprocess.run(
                    shell_cmd,
                    capture_output=True,
                    text=True,
                    timeout=request.timeout_seconds,
                    cwd=request.working_directory,
                )
            else:
                proc = subprocess.run(
                    cmd_args,
                    capture_output=True,
                    text=True,
                    timeout=request.timeout_seconds,
                    cwd=request.working_directory,
                )

            result.output = proc.stdout
            result.metadata["exit_code"] = proc.returncode

            if self._capture_stderr and proc.stderr:
                result.metadata["stderr"] = proc.stderr

            if proc.returncode == 0:
                result.success = True
                logger.info("Custom AI command completed successfully")
            else:
                result.error = proc.stderr or f"Command exited with code {proc.returncode}"
                logger.error(f"Custom AI command failed: {result.error}")

        except subprocess.TimeoutExpired:
            result.error = f"Command timed out after {request.timeout_seconds} seconds"
            logger.error(result.error)

        except ValueError as e:
            # Validation errors (e.g., dangerous characters)
            result.error = f"Command validation failed: {e}"
            logger.error(result.error)

        except Exception as e:
            result.error = f"Failed to execute custom command: {e}"
            logger.error(result.error, exc_info=True)

        return result

    async def stream_analyze(self, request: AnalysisRequest) -> AsyncIterator[str]:
        """Stream analysis output asynchronously.

        Args:
            request: The analysis request

        Yields:
            Lines of output from the command
        """
        try:
            cmd, cmd_args = self._build_command(request)
            logger.info("Starting streaming custom AI command")

            if self._shell:
                # Shell execution - use explicit shell invocation for safety
                if sys.platform == "win32":
                    shell_cmd = ["cmd.exe", "/c", cmd]
                else:
                    shell_cmd = ["/bin/sh", "-c", cmd]
                process = await asyncio.create_subprocess_exec(
                    *shell_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT if self._capture_stderr else None,
                    cwd=request.working_directory,
                )
            else:
                # Direct execution
                process = await asyncio.create_subprocess_exec(
                    *cmd_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT if self._capture_stderr else None,
                    cwd=request.working_directory,
                )

            # Stream output
            if process.stdout:
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    yield line.decode("utf-8", errors="replace")

            await process.wait()

        except TimeoutError:
            yield f"\n[ERROR: Command timed out after {request.timeout_seconds}s]\n"
            logger.error("Streaming command timed out")

        except ValueError as e:
            yield f"\n[ERROR: Command validation failed: {e}]\n"
            logger.error(f"Command validation failed: {e}")

        except Exception as e:
            yield f"\n[ERROR: {e}]\n"
            logger.error(f"Streaming command failed: {e}", exc_info=True)

    def _build_command(self, request: AnalysisRequest) -> tuple[str, list[str]]:
        """Build command with placeholders replaced.

        When shell mode is enabled, placeholder values are shell-escaped
        to prevent command injection attacks. All values are validated
        for dangerous characters.

        Args:
            request: The analysis request

        Returns:
            Tuple of (shell command string, argument list for direct execution)

        Raises:
            ValueError: If placeholder values contain dangerous characters
        """
        cmd = self._command

        # Define replacements with validation
        replacements = {
            "{prompt}": request.prompt,
            "{working_directory}": request.working_directory or "",
            "{results_directory}": request.results_directory,
            "{timeout}": str(request.timeout_seconds),
        }

        # Validate all placeholder values for dangerous characters
        for placeholder, value in replacements.items():
            _validate_placeholder_value(placeholder, value)

        # Build command string with escaped values for shell mode
        for placeholder, value in replacements.items():
            # Escape values when shell=True to prevent command injection
            if self._shell:
                escaped_value = shlex.quote(value)
            else:
                escaped_value = value
            cmd = cmd.replace(placeholder, escaped_value)

        # Parse into argument list for non-shell execution
        cmd_args = shlex.split(cmd)

        return cmd, cmd_args


class CustomScriptProvider(CustomCommandProvider):
    """Provider that executes a custom script file.

    This is a convenience wrapper around CustomCommandProvider that
    handles script path resolution and platform-specific execution.

    Example:
        provider = CustomScriptProvider(
            script_path="/path/to/analyze.py",
            interpreter="python",
        )
    """

    def __init__(
        self,
        script_path: str,
        interpreter: str | None = None,
        args: str = "",
    ):
        """Initialize custom script provider.

        Args:
            script_path: Path to script file
            interpreter: Script interpreter (e.g., "python", "bash")
            args: Additional arguments for the script
        """
        if interpreter:
            command = f"{interpreter} {script_path} {args}"
        else:
            command = f"{script_path} {args}"

        super().__init__(command=command, shell=False, capture_stderr=True)

        self._script_path = script_path
        self._interpreter = interpreter

    @property
    def description(self) -> str:
        """Get provider description."""
        if self._interpreter:
            return f"Custom script: {self._interpreter} {self._script_path}"
        return f"Custom script: {self._script_path}"
