"""Custom command AI provider implementation.

This provider allows users to define their own commands for AI analysis,
enabling integration with any AI system via custom scripts or commands.
"""

import asyncio
import logging
import subprocess
from collections.abc import AsyncIterator

from .base import AIProvider, AnalysisRequest, AnalysisResult

logger = logging.getLogger(__name__)


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
            shell: Execute via shell
            capture_stderr: Capture stderr output
        """
        self._command = command
        self._shell = shell
        self._capture_stderr = capture_stderr

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
        # We can't pre-check availability without knowing the placeholders
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
            # Build command with placeholders replaced
            cmd = self._build_command(request)

            logger.info("Executing custom AI command")
            logger.debug(f"Command: {cmd}")

            # Execute command
            if self._shell:
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=request.timeout_seconds,
                    cwd=request.working_directory,
                )
            else:
                proc = subprocess.run(
                    cmd.split(),
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
            cmd = self._build_command(request)
            logger.info("Starting streaming custom AI command")

            if self._shell:
                # Shell execution
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT if self._capture_stderr else None,
                    cwd=request.working_directory,
                )
            else:
                # Direct execution
                cmd_parts = cmd.split()
                process = await asyncio.create_subprocess_exec(
                    *cmd_parts,
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

        except Exception as e:
            yield f"\n[ERROR: {e}]\n"
            logger.error(f"Streaming command failed: {e}", exc_info=True)

    def _build_command(self, request: AnalysisRequest) -> str:
        """Build command with placeholders replaced.

        Args:
            request: The analysis request

        Returns:
            Command string with placeholders replaced
        """
        cmd = self._command

        # Replace placeholders
        replacements = {
            "{prompt}": request.prompt,
            "{working_directory}": request.working_directory or "",
            "{results_directory}": request.results_directory,
            "{timeout}": str(request.timeout_seconds),
        }

        for placeholder, value in replacements.items():
            cmd = cmd.replace(placeholder, value)

        return cmd


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
