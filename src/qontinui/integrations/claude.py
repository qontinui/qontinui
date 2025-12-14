"""Claude Code CLI integration for Qontinui.

This module provides cross-platform support for invoking Claude Code CLI
from Python. It handles:
- Finding Claude Code CLI on the system
- Cross-platform path handling (Windows, Linux, macOS, WSL)
- Native and WSL-based execution

The detection priority is:
1. Configured path (from runner settings or explicit configuration)
2. CLAUDE_PATH environment variable
3. Native claude in system PATH
4. WSL fallback on Windows (for users with claude only in WSL)

Requirements:
    - Claude Code CLI must be installed: https://docs.anthropic.com/en/docs/claude-code
    - Install via: npm install -g @anthropic-ai/claude-code

Usage:
    from qontinui.integrations.claude import find_claude, run_claude

    # Check if Claude is available
    info = find_claude()
    if info["found"]:
        result = run_claude(
            prompt="Analyze this code",
            working_directory="/path/to/project",
        )
"""

import logging
import os
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

# Module-level configured Claude path (set from runner UI settings or programmatically)
_configured_claude_path: str | None = None


class ClaudeInfo(TypedDict):
    """Information about Claude Code CLI availability."""

    found: bool
    method: str | None  # 'native', 'wsl', or None
    path: str | None
    error: str | None


class ClaudeResult(TypedDict):
    """Result from running Claude Code CLI."""

    success: bool
    output: str
    error: str
    return_code: int | None


def set_configured_claude_path(path: str | None) -> None:
    """Set the configured Claude path.

    This is typically called by the runner when the user configures
    a custom Claude CLI path in the settings UI.

    Args:
        path: Path to Claude CLI executable, or None to use auto-detection
    """
    global _configured_claude_path
    _configured_claude_path = path
    logger.info(f"Configured Claude path set to: {path or '(auto-detect)'}")


def get_configured_claude_path() -> str | None:
    """Get the currently configured Claude path.

    Returns:
        The configured path, or None if using auto-detection
    """
    return _configured_claude_path


def find_claude() -> ClaudeInfo:
    """Find the Claude Code CLI executable.

    Checks in order:
    1. Configured path from settings (highest priority)
    2. CLAUDE_PATH environment variable
    3. Native claude in system PATH (Windows, Linux, macOS)
    4. WSL fallback on Windows

    Returns:
        ClaudeInfo dictionary with:
        - found: bool - Whether claude was found
        - method: str - How claude will be invoked ('native', 'wsl', or None)
        - path: str - Path to claude executable (for native) or WSL path
        - error: str - Error message if not found
    """
    system = platform.system()

    # Debug: Log PATH info at debug level for troubleshooting
    current_path = os.environ.get("PATH", "")
    logger.debug(f"[find_claude] PATH length: {len(current_path)} chars")
    nodejs_in_path = "nodejs" in current_path.lower()
    logger.debug(f"[find_claude] nodejs in PATH: {nodejs_in_path}")

    # 1. Check configured path (highest priority)
    if configured_path := get_configured_claude_path():
        if os.path.isfile(configured_path):
            # On Windows, .cmd files may not be executable in the Unix sense
            if system == "Windows" or os.access(configured_path, os.X_OK):
                logger.info(f"Using claude from configured path: {configured_path}")
                return ClaudeInfo(found=True, method="native", path=configured_path, error=None)
        logger.warning(
            f"Configured Claude path {configured_path} not found, " "falling back to auto-detection"
        )

    # 2. Check CLAUDE_PATH environment variable
    if claude_path := os.environ.get("CLAUDE_PATH"):
        if os.path.isfile(claude_path):
            if system == "Windows" or os.access(claude_path, os.X_OK):
                logger.info(f"Using claude from CLAUDE_PATH: {claude_path}")
                return ClaudeInfo(found=True, method="native", path=claude_path, error=None)
        logger.warning(f"CLAUDE_PATH set to {claude_path} but file not found or not executable")

    # 3. Check native PATH (works on all platforms)
    native_claude = shutil.which("claude")
    if native_claude:
        logger.info(f"Found native claude in PATH: {native_claude}")
        return ClaudeInfo(found=True, method="native", path=native_claude, error=None)

    # On Windows, also check for claude.cmd (npm installs .cmd wrapper)
    if system == "Windows":
        claude_cmd = shutil.which("claude.cmd")
        if claude_cmd:
            logger.info(f"Found claude.cmd in PATH: {claude_cmd}")
            return ClaudeInfo(found=True, method="native", path=claude_cmd, error=None)

    # 4. WSL fallback on Windows
    if system == "Windows":
        wsl_result = _check_wsl_claude()
        if wsl_result["found"]:
            return wsl_result

    # Not found - provide helpful error message
    error_msg = (
        "Claude Code CLI not found. Please install it:\n"
        "  npm install -g @anthropic-ai/claude-code\n"
        "\n"
        "Or set CLAUDE_PATH environment variable to the claude executable path.\n"
        "\n"
        "Documentation: https://docs.anthropic.com/en/docs/claude-code"
    )

    return ClaudeInfo(found=False, method=None, path=None, error=error_msg)


def _check_wsl_claude() -> ClaudeInfo:
    """Check if Claude is available in WSL on Windows.

    Returns:
        ClaudeInfo with WSL claude info or not found
    """
    try:
        # Check if WSL is available
        wsl_check = subprocess.run(
            ["wsl.exe", "--status"],
            capture_output=True,
            timeout=5,
        )
        if wsl_check.returncode == 0:
            # Check if claude is available in WSL
            result = subprocess.run(
                [
                    "wsl.exe",
                    "bash",
                    "-c",
                    'source "$HOME/.bashrc" 2>/dev/null; which claude',
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                wsl_claude_path = result.stdout.strip()
                logger.info(f"Found claude in WSL: {wsl_claude_path}")
                return ClaudeInfo(found=True, method="wsl", path=wsl_claude_path, error=None)
    except subprocess.TimeoutExpired:
        logger.debug("WSL check timed out")
    except FileNotFoundError:
        logger.debug("wsl.exe not found")
    except Exception as e:
        logger.debug(f"WSL check failed: {e}")

    return ClaudeInfo(found=False, method=None, path=None, error="WSL claude not found")


def is_claude_available() -> bool:
    """Check if Claude Code CLI is available.

    Returns:
        True if Claude Code CLI is available
    """
    return find_claude()["found"]


def to_wsl_path(windows_path: str) -> str:
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


def run_claude(
    prompt: str,
    working_directory: str | Path | None = None,
    timeout_seconds: int = 600,
    output_format: str = "text",
    permission_mode: str = "bypassPermissions",
    extra_args: list[str] | None = None,
) -> ClaudeResult:
    """Run Claude Code CLI with the given prompt.

    Args:
        prompt: The prompt to send to Claude
        working_directory: Working directory for Claude (default: current directory)
        timeout_seconds: Maximum time to wait (default: 600s / 10min)
        output_format: Output format ('text', 'json', 'stream-json')
        permission_mode: Permission mode ('bypassPermissions', 'default')
        extra_args: Additional command line arguments

    Returns:
        ClaudeResult dictionary with:
        - success: bool - Whether execution succeeded
        - output: str - Output from Claude
        - error: str - Error message if failed
        - return_code: int - Process return code
    """
    result: ClaudeResult = {
        "success": False,
        "output": "",
        "error": "",
        "return_code": None,
    }

    # Find claude
    claude_info = find_claude()
    if not claude_info["found"]:
        result["error"] = claude_info["error"] or "Claude not found"
        logger.error(f"Claude not found: {result['error']}")
        return result

    # Determine working directory
    if working_directory:
        cwd = Path(working_directory)
    else:
        cwd = Path.cwd()

    logger.info(f"Running Claude Code (method: {claude_info['method']})...")
    logger.debug(f"Working directory: {cwd}")
    logger.debug(f"Prompt length: {len(prompt)} chars")

    try:
        if claude_info["method"] == "native":
            result = _run_native_claude(
                claude_path=claude_info["path"],  # type: ignore
                prompt=prompt,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                output_format=output_format,
                permission_mode=permission_mode,
                extra_args=extra_args,
            )
        elif claude_info["method"] == "wsl":
            result = _run_wsl_claude(
                prompt=prompt,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                output_format=output_format,
                permission_mode=permission_mode,
                extra_args=extra_args,
            )
        else:
            result["error"] = f"Unknown claude method: {claude_info['method']}"

    except subprocess.TimeoutExpired:
        result["error"] = f"Claude timed out after {timeout_seconds} seconds"
        logger.error(result["error"])

    except Exception as e:
        result["error"] = f"Failed to invoke Claude: {e}"
        logger.error(result["error"])

    return result


def _run_native_claude(
    claude_path: str,
    prompt: str,
    cwd: Path,
    timeout_seconds: int,
    output_format: str,
    permission_mode: str,
    extra_args: list[str] | None,
) -> ClaudeResult:
    """Run claude natively (Windows, Linux, macOS)."""
    cmd = [
        claude_path,
        "-p",
        prompt,
        "--output-format",
        output_format,
        "--permission-mode",
        permission_mode,
    ]

    if extra_args:
        cmd.extend(extra_args)

    # IMPORTANT: stdin=DEVNULL prevents Claude CLI from hanging waiting for input
    # when run as a subprocess (e.g., from Tauri/Rust IPC environment)
    proc = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        cwd=str(cwd),
    )

    result: ClaudeResult = {
        "success": proc.returncode == 0,
        "output": proc.stdout,
        "error": proc.stderr if proc.returncode != 0 else "",
        "return_code": proc.returncode,
    }

    if result["success"]:
        logger.info("Claude Code completed successfully")
    else:
        logger.error(f"Claude Code failed: {result['error'] or f'exit code {proc.returncode}'}")

    return result


def _run_wsl_claude(
    prompt: str,
    cwd: Path,
    timeout_seconds: int,
    output_format: str,
    permission_mode: str,
    extra_args: list[str] | None,
) -> ClaudeResult:
    """Run claude via WSL on Windows."""
    # Convert Windows path to WSL path
    wsl_cwd = to_wsl_path(str(cwd))

    # Build extra args string
    extra_args_str = " ".join(extra_args) if extra_args else ""

    # Create a temp script to handle the long prompt
    # Source shell profiles to ensure PATH is set correctly
    script_content = f"""#!/bin/bash
# Source shell profile to get PATH (needed for npm-global installations)
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
[ -f "$HOME/.profile" ] && source "$HOME/.profile"
[ -f "$HOME/.bash_profile" ] && source "$HOME/.bash_profile"

cd "{wsl_cwd}"
PROMPT=$(cat <<'ENDOFPROMPT'
{prompt}
ENDOFPROMPT
)
claude -p "$PROMPT" --output-format {output_format} --permission-mode {permission_mode} {extra_args_str} 2>&1
"""

    # Write script to temp file with Unix line endings
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, newline="\n") as f:
        f.write(script_content)
        script_path = f.name

    try:
        # Convert script path to WSL path
        wsl_script = to_wsl_path(script_path)

        cmd = ["wsl.exe", "bash", wsl_script]

        # stdin=DEVNULL prevents hanging on input
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        result: ClaudeResult = {
            "success": proc.returncode == 0,
            "output": proc.stdout,
            "error": proc.stderr if proc.returncode != 0 else "",
            "return_code": proc.returncode,
        }

        if result["success"]:
            logger.info("Claude Code (WSL) completed successfully")
        else:
            logger.error(
                f"Claude Code (WSL) failed: {result['error'] or f'exit code {proc.returncode}'}"
            )

        return result

    finally:
        # Clean up temp script
        try:
            os.unlink(script_path)
        except Exception:
            pass


def run_claude_streaming(
    prompt: str,
    working_directory: str | Path | None = None,
    timeout_seconds: int = 600,
    output_format: str = "text",
    permission_mode: str = "bypassPermissions",
    extra_args: list[str] | None = None,
    on_output: Callable[[str, str], None] | None = None,
) -> ClaudeResult:
    """Run Claude Code CLI with streaming output.

    This function streams output line by line as Claude produces it,
    calling on_output for each line. This enables real-time feedback.

    Args:
        prompt: The prompt to send to Claude
        working_directory: Working directory for Claude (default: current directory)
        timeout_seconds: Maximum time to wait (default: 600s / 10min)
        output_format: Output format ('text', 'json', 'stream-json')
        permission_mode: Permission mode ('bypassPermissions', 'default')
        extra_args: Additional command line arguments
        on_output: Callback function called for each line of output.
                   Signature: on_output(line: str, source: str) where source is 'stdout' or 'stderr'

    Returns:
        ClaudeResult dictionary with:
        - success: bool - Whether execution succeeded
        - output: str - Complete output from Claude
        - error: str - Error message if failed
        - return_code: int - Process return code
    """
    result: ClaudeResult = {
        "success": False,
        "output": "",
        "error": "",
        "return_code": None,
    }

    # Find claude
    claude_info = find_claude()
    if not claude_info["found"]:
        result["error"] = claude_info["error"] or "Claude not found"
        logger.error(f"Claude not found: {result['error']}")
        return result

    # Determine working directory
    if working_directory:
        cwd = Path(working_directory)
    else:
        cwd = Path.cwd()

    logger.info(f"Running Claude Code streaming (method: {claude_info['method']})...")
    logger.debug(f"Working directory: {cwd}")
    logger.debug(f"Prompt length: {len(prompt)} chars")

    try:
        if claude_info["method"] == "native":
            claude_path = claude_info["path"]
            if not claude_path:
                result["error"] = "Claude path is None"
                return result
            result = _run_native_claude_streaming(
                claude_path=claude_path,
                prompt=prompt,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                output_format=output_format,
                permission_mode=permission_mode,
                extra_args=extra_args,
                on_output=on_output,
            )
        elif claude_info["method"] == "wsl":
            result = _run_wsl_claude_streaming(
                prompt=prompt,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                output_format=output_format,
                permission_mode=permission_mode,
                extra_args=extra_args,
                on_output=on_output,
            )
        else:
            result["error"] = f"Unknown claude method: {claude_info['method']}"

    except subprocess.TimeoutExpired:
        result["error"] = f"Claude timed out after {timeout_seconds} seconds"
        logger.error(result["error"])

    except Exception as e:
        result["error"] = f"Failed to invoke Claude: {e}"
        logger.error(result["error"])

    return result


def _run_native_claude_streaming(
    claude_path: str,
    prompt: str,
    cwd: Path,
    timeout_seconds: int,
    output_format: str,
    permission_mode: str,
    extra_args: list[str] | None,
    on_output: Callable[[str, str], None] | None = None,
) -> ClaudeResult:
    """Run claude natively with streaming output.

    Uses stream-json format with --include-partial-messages for real-time output.
    The callback receives partial message content as it arrives.
    """
    import json
    import threading

    # Force stream-json for real-time streaming regardless of requested format
    # stream-json outputs JSON lines with partial message chunks
    # --verbose is required for stream-json format
    cmd = [
        claude_path,
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--verbose",
        "--permission-mode",
        permission_mode,
    ]

    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"Starting Claude process: {claude_path}")
    logger.debug(f"Working directory: {cwd}")

    # Set up environment with unbuffered Python output
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Use Popen for streaming
    # On Windows, we need shell=False and proper handling
    # IMPORTANT: stdin=DEVNULL prevents Claude CLI from hanging waiting for input
    # when run as a subprocess (e.g., from Tauri/Rust IPC environment)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd),
        bufsize=1,  # Line buffered
        env=env,
    )

    logger.info(f"Claude process started with PID: {proc.pid}")

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    output_text_parts: list[str] = []  # Accumulated text from stream-json events

    def read_stream(stream, lines_list, source):
        """Read from stream line by line, parsing stream-json format."""
        nonlocal output_text_parts
        logger.debug(f"Starting to read {source} stream")
        line_count = 0
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                line = line.rstrip("\n\r")
                lines_list.append(line)
                line_count += 1

                # For stdout, parse stream-json and extract text content
                if source == "stdout" and line.strip():
                    try:
                        event = json.loads(line)
                        event_type = event.get("type", "")

                        # Handle stream_event with nested event structure
                        # Format: {"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"..."}}}
                        if event_type == "stream_event":
                            inner_event = event.get("event", {})
                            inner_type = inner_event.get("type", "")

                            if inner_type == "content_block_delta":
                                delta = inner_event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text = delta.get("text", "")
                                    if text:
                                        if on_output:
                                            try:
                                                on_output(text, source)
                                            except Exception as e:
                                                logger.warning(f"on_output callback error: {e}")
                                        output_text_parts.append(text)

                        # Handle assistant message content (snapshot of full message so far)
                        # Don't use this for streaming - it's the full accumulated message
                        elif event_type == "assistant":
                            pass  # Skip - we're streaming incremental updates instead

                        # Handle result event (final output) - fallback if streaming didn't work
                        elif event_type == "result":
                            result_data = event.get("result", "")
                            if isinstance(result_data, str) and result_data:
                                # Result contains the full response - use only if we haven't accumulated
                                if not output_text_parts:
                                    output_text_parts.append(result_data)
                                    if on_output:
                                        try:
                                            on_output(result_data, source)
                                        except Exception as e:
                                            logger.warning(f"on_output callback error: {e}")

                        logger.debug(f"[{source}] Parsed event type: {event_type}")

                    except json.JSONDecodeError:
                        # Not valid JSON, pass through as-is
                        logger.debug(f"[{source}] Line {line_count}: {line[:100]}...")
                        if on_output:
                            try:
                                on_output(line, source)
                            except Exception as e:
                                logger.warning(f"on_output callback error: {e}")
                else:
                    # Stderr or empty lines
                    logger.debug(f"[{source}] Line {line_count}: {line[:100]}...")

        except Exception as e:
            logger.warning(f"Error reading {source}: {e}")
        finally:
            logger.debug(f"Finished reading {source} stream, got {line_count} lines")
            stream.close()

    # Start threads to read stdout and stderr
    stdout_thread = threading.Thread(target=read_stream, args=(proc.stdout, stdout_lines, "stdout"))
    stderr_thread = threading.Thread(target=read_stream, args=(proc.stderr, stderr_lines, "stderr"))

    stdout_thread.daemon = True
    stderr_thread.daemon = True

    stdout_thread.start()
    stderr_thread.start()

    # Wait for process with timeout
    try:
        logger.info(f"Waiting for Claude process (timeout: {timeout_seconds}s)...")
        return_code = proc.wait(timeout=timeout_seconds)
        logger.info(f"Claude process finished with return code: {return_code}")
    except subprocess.TimeoutExpired:
        logger.error(f"Claude process timed out after {timeout_seconds}s")
        proc.kill()
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        raise

    # Wait for threads to finish
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    logger.info(f"Collected {len(stdout_lines)} stdout lines, {len(stderr_lines)} stderr lines")
    logger.info(f"Accumulated {len(output_text_parts)} text parts from stream-json")

    # Use accumulated text from stream-json events as the output
    final_output = "".join(output_text_parts) if output_text_parts else "\n".join(stdout_lines)

    result: ClaudeResult = {
        "success": proc.returncode == 0,
        "output": final_output,
        "error": "\n".join(stderr_lines) if proc.returncode != 0 else "",
        "return_code": proc.returncode,
    }

    if result["success"]:
        logger.info("Claude Code streaming completed successfully")
        logger.debug(f"Output preview: {final_output[:200]}...")
    else:
        logger.error(
            f"Claude Code streaming failed: {result['error'] or f'exit code {proc.returncode}'}"
        )

    return result


def _run_wsl_claude_streaming(
    prompt: str,
    cwd: Path,
    timeout_seconds: int,
    output_format: str,
    permission_mode: str,
    extra_args: list[str] | None,
    on_output: Callable[[str, str], None] | None = None,
) -> ClaudeResult:
    """Run claude via WSL on Windows with streaming output.

    Uses stream-json format with --include-partial-messages for real-time output.
    """
    import json
    import threading

    # Convert Windows path to WSL path
    wsl_cwd = to_wsl_path(str(cwd))

    # Build extra args string
    extra_args_str = " ".join(extra_args) if extra_args else ""

    # Create a temp script to handle the long prompt
    # Force stream-json for real-time streaming
    script_content = f"""#!/bin/bash
# Source shell profile to get PATH (needed for npm-global installations)
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
[ -f "$HOME/.profile" ] && source "$HOME/.profile"
[ -f "$HOME/.bash_profile" ] && source "$HOME/.bash_profile"

cd "{wsl_cwd}"
PROMPT=$(cat <<'ENDOFPROMPT'
{prompt}
ENDOFPROMPT
)
claude -p "$PROMPT" --output-format stream-json --include-partial-messages --verbose --permission-mode {permission_mode} {extra_args_str} 2>&1
"""

    # Write script to temp file with Unix line endings
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, newline="\n") as f:
        f.write(script_content)
        script_path = f.name

    try:
        # Convert script path to WSL path
        wsl_script = to_wsl_path(script_path)

        cmd = ["wsl.exe", "bash", wsl_script]

        # Use Popen for streaming
        # stdin=DEVNULL prevents hanging on input
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        output_text_parts: list[str] = []  # Accumulated text from stream-json events

        def read_stream(stream, lines_list, source):
            """Read from stream line by line, parsing stream-json format."""
            nonlocal output_text_parts
            try:
                for line in iter(stream.readline, ""):
                    if not line:
                        break
                    line = line.rstrip("\n\r")
                    lines_list.append(line)

                    # Parse stream-json and extract text content
                    if line.strip():
                        try:
                            event = json.loads(line)
                            event_type = event.get("type", "")

                            # Handle stream_event with nested event structure
                            if event_type == "stream_event":
                                inner_event = event.get("event", {})
                                inner_type = inner_event.get("type", "")

                                if inner_type == "content_block_delta":
                                    delta = inner_event.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        text = delta.get("text", "")
                                        if text:
                                            if on_output:
                                                try:
                                                    on_output(text, source)
                                                except Exception as e:
                                                    logger.warning(f"on_output callback error: {e}")
                                            output_text_parts.append(text)

                            # Handle assistant message - skip for streaming
                            elif event_type == "assistant":
                                pass

                            # Handle result event (final output) - fallback
                            elif event_type == "result":
                                result_data = event.get("result", "")
                                if isinstance(result_data, str) and result_data:
                                    if not output_text_parts:
                                        output_text_parts.append(result_data)
                                        if on_output:
                                            try:
                                                on_output(result_data, source)
                                            except Exception as e:
                                                logger.warning(f"on_output callback error: {e}")

                        except json.JSONDecodeError:
                            # Not valid JSON, pass through as-is
                            if on_output:
                                try:
                                    on_output(line, source)
                                except Exception as e:
                                    logger.warning(f"on_output callback error: {e}")

            except Exception as e:
                logger.warning(f"Error reading {source}: {e}")
            finally:
                stream.close()

        # Start threads to read stdout and stderr
        stdout_thread = threading.Thread(
            target=read_stream, args=(proc.stdout, stdout_lines, "stdout")
        )
        stderr_thread = threading.Thread(
            target=read_stream, args=(proc.stderr, stderr_lines, "stderr")
        )

        stdout_thread.start()
        stderr_thread.start()

        # Wait for process with timeout
        try:
            proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            raise

        # Wait for threads to finish
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

        # Use accumulated text from stream-json events as the output
        final_output = "".join(output_text_parts) if output_text_parts else "\n".join(stdout_lines)

        result: ClaudeResult = {
            "success": proc.returncode == 0,
            "output": final_output,
            "error": "\n".join(stderr_lines) if proc.returncode != 0 else "",
            "return_code": proc.returncode,
        }

        if result["success"]:
            logger.info("Claude Code (WSL) streaming completed successfully")
        else:
            logger.error(
                f"Claude Code (WSL) streaming failed: {result['error'] or f'exit code {proc.returncode}'}"
            )

        return result

    finally:
        # Clean up temp script
        try:
            os.unlink(script_path)
        except Exception:
            pass
