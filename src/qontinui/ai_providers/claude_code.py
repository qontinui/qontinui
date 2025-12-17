"""Claude Code AI provider implementation.

This provider invokes the Claude Code CLI to analyze automation results.
It handles platform-specific invocation (Windows via WSL, Unix directly).
"""

import asyncio
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path

from .base import AIProvider, AnalysisRequest, AnalysisResult

logger = logging.getLogger(__name__)


class ClaudeCodeProvider(AIProvider):
    """AI provider that uses Claude Code CLI.

    This provider invokes the `claude` CLI command to perform analysis.
    On Windows, it invokes via WSL where Claude Code has MCP configured.
    On Unix systems, it invokes directly.

    Configuration via environment variables:
        CLAUDE_CODE_PATH: Path to claude executable (default: auto-detect)
    """

    @property
    def name(self) -> str:
        """Get the provider name."""
        return "claude_code"

    @property
    def description(self) -> str:
        """Get provider description."""
        return "Claude Code CLI via WSL (Windows) or direct invocation (Unix)"

    def is_available(self) -> bool:
        """Check if Claude Code CLI is available.

        Returns:
            True if claude command can be executed
        """
        system = platform.system()

        if system == "Windows":
            # On Windows, check if we can invoke via WSL
            try:
                result = subprocess.run(
                    ["wsl.exe", "which", "claude"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                return result.returncode == 0
            except Exception:
                return False
        else:
            # On Unix, check directly
            if custom_path := os.environ.get("CLAUDE_CODE_PATH"):
                return Path(custom_path).exists()
            return shutil.which("claude") is not None

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Run analysis synchronously.

        Args:
            request: The analysis request

        Returns:
            The analysis result
        """
        system = platform.system()
        result = AnalysisResult(success=False, provider=self.name)

        try:
            # Build command based on platform
            cmd = self._build_command(request)
            cwd = request.working_directory

            logger.info(
                f"Executing Claude Code analysis (timeout: {request.timeout_seconds}s)"
            )
            logger.debug(f"Command: {cmd}")

            if system == "Windows":
                # On Windows, use special handling for WSL
                output, error = self._execute_windows(cmd, request)
            else:
                # On Unix, execute directly
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=request.timeout_seconds,
                    cwd=cwd,
                )
                output = proc.stdout
                error = proc.stderr if proc.returncode != 0 else ""

            if error:
                result.error = error
                result.metadata["stderr"] = error
                logger.error(f"Claude Code analysis failed: {error}")
            else:
                result.success = True
                result.output = output
                logger.info("Claude Code analysis completed successfully")

        except subprocess.TimeoutExpired:
            result.error = f"Analysis timed out after {request.timeout_seconds} seconds"
            logger.error(result.error)

        except Exception as e:
            result.error = f"Failed to invoke Claude Code: {e}"
            logger.error(result.error, exc_info=True)

        return result

    async def stream_analyze(self, request: AnalysisRequest) -> AsyncIterator[str]:
        """Stream analysis output asynchronously.

        Args:
            request: The analysis request

        Yields:
            Lines of output from Claude Code
        """
        system = platform.system()

        try:
            cmd = self._build_command(request)
            cwd = request.working_directory

            logger.info("Starting streaming Claude Code analysis")

            if system == "Windows":
                # On Windows, use file-based streaming
                async for line in self._stream_windows(cmd, request):
                    yield line
            else:
                # On Unix, use process stdout streaming
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=cwd,
                )

                if process.stdout:
                    async for line in self._read_stream(process.stdout):
                        yield line

                await process.wait()

        except TimeoutError:
            yield f"\n[ERROR: Analysis timed out after {request.timeout_seconds}s]\n"
            logger.error("Streaming analysis timed out")

        except Exception as e:
            yield f"\n[ERROR: {e}]\n"
            logger.error(f"Streaming analysis failed: {e}", exc_info=True)

    def _build_command(self, request: AnalysisRequest) -> list[str]:
        """Build the Claude Code command based on platform and request.

        Args:
            request: The analysis request

        Returns:
            Command as list of arguments
        """
        system = platform.system()
        prompt = request.prompt

        if system == "Windows":
            # On Windows, invoke via WSL
            wsl_working_dir = self._to_wsl_path(request.working_directory)

            # Escape prompt for shell
            escaped_prompt = prompt.replace('"', '\\"').replace("'", "'\\''")

            # Build bash command with PATH setup for npm globals
            npm_paths = (
                "$HOME/.npm-global/lib/bin:$HOME/.npm-global/bin:$HOME/.local/bin"
            )
            path_setup = f'export PATH="{npm_paths}:$PATH"'
            env_setup = "export CI=true TERM=dumb FORCE_COLOR=0"

            if wsl_working_dir:
                bash_cmd = (
                    f'{path_setup}; cd "{wsl_working_dir}" && '
                    f'{env_setup}; claude -p "{escaped_prompt}" '
                    f"--output-format {request.output_format} "
                    f"--permission-mode bypassPermissions --print < /dev/null 2>&1"
                )
            else:
                bash_cmd = (
                    f'{path_setup}; {env_setup}; claude -p "{escaped_prompt}" '
                    f"--output-format {request.output_format} "
                    f"--permission-mode bypassPermissions --print < /dev/null 2>&1"
                )

            return ["wsl.exe", "bash", "-lc", bash_cmd]

        else:
            # On Unix, invoke directly
            claude_path = os.environ.get("CLAUDE_CODE_PATH", "claude")
            return [
                claude_path,
                "-p",
                prompt,
                "--output-format",
                request.output_format,
                "--permission-mode",
                "bypassPermissions",
            ]

    def _execute_windows(
        self, cmd: list[str], request: AnalysisRequest
    ) -> tuple[str, str]:
        """Execute command on Windows with special WSL handling.

        Args:
            cmd: Command to execute
            request: Analysis request

        Returns:
            Tuple of (stdout, stderr)
        """
        # Create temp file for output
        output_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        output_file_path = output_file.name
        output_file.close()

        try:
            # Modify command to redirect output to file
            wsl_output_path = self._to_wsl_path(output_file_path)

            # Build command with file redirection and completion marker
            original_cmd_str = cmd[3]  # The bash -lc argument
            modified_cmd_str = original_cmd_str.replace(" 2>&1", "")
            modified_cmd_str += f' >> "{wsl_output_path}" 2>&1'
            modified_cmd_str += (
                f'; echo "___QONTINUI_DONE_$$___" >> "{wsl_output_path}"'
            )

            modified_cmd = ["wsl.exe", "bash", "-lc", modified_cmd_str]

            # Start process with hidden console
            creationflags = subprocess.CREATE_NEW_CONSOLE
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

            process = subprocess.Popen(
                modified_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                startupinfo=startupinfo,
            )

            logger.debug(f"Started WSL process (pid: {process.pid})")

            # Poll output file for completion
            output = self._poll_output_file(output_file_path, request.timeout_seconds)

            return output, ""

        finally:
            # Clean up temp file
            try:
                os.unlink(output_file_path)
            except Exception:
                pass

    async def _stream_windows(
        self, cmd: list[str], request: AnalysisRequest
    ) -> AsyncIterator[str]:
        """Stream output from Windows/WSL process.

        Args:
            cmd: Command to execute
            request: Analysis request

        Yields:
            Lines of output
        """
        # Create temp file for output
        output_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        output_file_path = output_file.name
        output_file.close()

        try:
            # Modify command for file output
            wsl_output_path = self._to_wsl_path(output_file_path)

            original_cmd_str = cmd[3]
            modified_cmd_str = original_cmd_str.replace(" 2>&1", "")
            modified_cmd_str += f' >> "{wsl_output_path}" 2>&1'
            modified_cmd_str += (
                f'; echo "___QONTINUI_DONE_$$___" >> "{wsl_output_path}"'
            )

            modified_cmd = ["wsl.exe", "bash", "-lc", modified_cmd_str]

            # Start process
            creationflags = subprocess.CREATE_NEW_CONSOLE
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

            subprocess.Popen(
                modified_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                startupinfo=startupinfo,
            )

            # Stream output file
            async for line in self._poll_output_file_async(
                output_file_path, request.timeout_seconds
            ):
                yield line

        finally:
            try:
                os.unlink(output_file_path)
            except Exception:
                pass

    def _poll_output_file(self, file_path: str, timeout: int) -> str:
        """Poll output file until completion marker found.

        Args:
            file_path: Path to output file
            timeout: Timeout in seconds

        Returns:
            Complete output from file
        """
        start_time = time.time()
        last_position = 0
        output_lines = []
        completion_marker = "___QONTINUI_DONE_"

        while True:
            if time.time() - start_time > timeout:
                raise subprocess.TimeoutExpired(cmd=[], timeout=timeout)

            try:
                with open(file_path, encoding="utf-8", errors="replace") as f:
                    f.seek(last_position)
                    new_content = f.read()

                    if new_content:
                        last_position = f.tell()
                        output_lines.append(new_content)

                        # Check for completion marker
                        if completion_marker in new_content:
                            break

            except FileNotFoundError:
                pass  # File not created yet

            time.sleep(0.1)

        # Join all output and remove completion marker
        full_output = "".join(output_lines)
        if completion_marker in full_output:
            full_output = full_output[: full_output.find(completion_marker)]

        return full_output.strip()

    async def _poll_output_file_async(
        self, file_path: str, timeout: int
    ) -> AsyncIterator[str]:
        """Asynchronously poll output file and yield lines.

        Args:
            file_path: Path to output file
            timeout: Timeout in seconds

        Yields:
            Lines from the file
        """
        start_time = time.time()
        last_position = 0
        completion_marker = "___QONTINUI_DONE_"

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError()

            try:
                with open(file_path, encoding="utf-8", errors="replace") as f:
                    f.seek(last_position)
                    new_content = f.read()

                    if new_content:
                        last_position = f.tell()

                        # Check for completion
                        if completion_marker in new_content:
                            # Remove marker and yield final content
                            final_content = new_content[
                                : new_content.find(completion_marker)
                            ]
                            if final_content:
                                yield final_content
                            break
                        else:
                            yield new_content

            except FileNotFoundError:
                pass

            await asyncio.sleep(0.1)

    async def _read_stream(self, stream: asyncio.StreamReader) -> AsyncIterator[str]:
        """Read lines from async stream.

        Args:
            stream: Async stream reader

        Yields:
            Lines from the stream
        """
        while True:
            try:
                line = await stream.readline()
                if not line:
                    break
                yield line.decode("utf-8", errors="replace")
            except Exception as e:
                logger.error(f"Error reading stream: {e}")
                break

    def _to_wsl_path(self, windows_path: str | None) -> str | None:
        """Convert Windows path to WSL path.

        Args:
            windows_path: Windows path (e.g., C:\\Users\\...)

        Returns:
            WSL path (e.g., /mnt/c/Users/...) or None
        """
        if not windows_path:
            return None

        path = windows_path.replace("\\", "/")

        # C:/... -> /mnt/c/...
        if len(path) >= 2 and path[1] == ":" and path[0].isalpha():
            drive = path[0].lower()
            rest = path[2:].lstrip("/")
            return f"/mnt/{drive}/{rest}"

        return path
