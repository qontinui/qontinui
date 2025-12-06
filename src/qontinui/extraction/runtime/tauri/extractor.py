"""
TauriExtractor for runtime extraction of Tauri applications.

Extracts UI state from Tauri applications by connecting to their dev server
and injecting Tauri API mocks to enable browser-based testing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..playwright.extractor import PlaywrightExtractor
from .mock import generate_mock_script, get_default_mocks

if TYPE_CHECKING:
    from ..types import ExtractionTarget, RuntimeStateCapture

logger = logging.getLogger(__name__)


class TauriExtractor(PlaywrightExtractor):
    """
    Runtime extractor for Tauri applications.

    Works by:
    1. Starting the Tauri dev server (if app_dev_command provided)
    2. Connecting to it via Playwright
    3. Injecting Tauri API mocks
    4. Using Playwright's DOM extraction capabilities

    This allows Tauri apps to be tested in a browser environment
    without requiring the full Tauri runtime.
    """

    def __init__(self):
        """Initialize the Tauri extractor."""
        super().__init__()
        self.dev_process: subprocess.Popen | None = None
        self.mock_script: str | None = None
        self.tauri_config: dict[str, Any] = {}

    async def connect(self, target: ExtractionTarget) -> None:
        """
        Connect to Tauri app via dev server with mocks.

        Args:
            target: ExtractionTarget with Tauri configuration.

        Raises:
            ConnectionError: If unable to connect to the target.
        """
        try:
            # Load Tauri config if provided
            if target.tauri_config_path:
                await self._load_tauri_config(target.tauri_config_path)

            # Start dev server if command provided
            if target.app_dev_command:
                await self._start_dev_server(target.app_dev_command)

            # Wait for server to be ready
            if target.url:
                await self._wait_for_server(target.url)

            # Generate mock script
            self.mock_script = self._generate_mock_script(target)

            # Connect via Playwright (parent class)
            await super().connect(target)

            # Inject Tauri mocks
            if self.page and self.mock_script:
                await self.page.add_init_script(self.mock_script)
                logger.info("Tauri API mocks injected")

            logger.info("Connected to Tauri application")

        except Exception as e:
            logger.error(f"Failed to connect to Tauri app: {e}")
            await self._cleanup_tauri()
            raise ConnectionError("Failed to connect to Tauri app") from e

    async def disconnect(self) -> None:
        """Disconnect from Tauri app and cleanup resources."""
        await super().disconnect()
        await self._cleanup_tauri()
        logger.info("Disconnected from Tauri application")

    async def _cleanup_tauri(self) -> None:
        """Clean up Tauri-specific resources."""
        # Stop dev server
        if self.dev_process:
            try:
                self.dev_process.terminate()
                self.dev_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.dev_process.kill()
                self.dev_process.wait()
            except Exception as e:
                logger.warning(f"Error stopping dev server: {e}")
            finally:
                self.dev_process = None

    async def _load_tauri_config(self, config_path: str) -> None:
        """
        Load Tauri configuration from tauri.conf.json.

        Args:
            config_path: Path to tauri.conf.json or tauri.config.json.
        """
        try:
            path = Path(config_path)
            if not path.exists():
                logger.warning(f"Tauri config not found: {config_path}")
                return

            with open(path, encoding="utf-8") as f:
                self.tauri_config = json.load(f)

            logger.info(f"Loaded Tauri config from {config_path}")

        except Exception as e:
            logger.warning(f"Failed to load Tauri config: {e}")

    async def _start_dev_server(self, command: str) -> None:
        """
        Start the Tauri dev server.

        Args:
            command: Command to start the dev server (e.g., "npm run dev").
        """
        try:
            logger.info(f"Starting dev server: {command}")

            # Parse command
            cmd_parts = command.split()

            # Start process
            self.dev_process = subprocess.Popen(
                cmd_parts,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Give server time to start
            await asyncio.sleep(3)

            # Check if process is still running
            if self.dev_process.poll() is not None:
                stderr = self.dev_process.stderr.read() if self.dev_process.stderr else ""
                raise RuntimeError(f"Dev server failed to start: {stderr}")

            logger.info("Dev server started successfully")

        except Exception as e:
            logger.error(f"Failed to start dev server: {e}")
            raise RuntimeError("Failed to start dev server") from e

    async def _wait_for_server(self, url: str, timeout: int = 30) -> None:
        """
        Wait for the dev server to be ready.

        Args:
            url: URL to check.
            timeout: Maximum time to wait in seconds.
        """
        import aiohttp

        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=2) as response:
                        if response.status < 500:
                            logger.info(f"Server ready at {url}")
                            return
            except Exception:
                # Server not ready yet
                pass

            await asyncio.sleep(1)

        raise TimeoutError(f"Server at {url} did not become ready within {timeout}s")

    def _generate_mock_script(self, target: ExtractionTarget) -> str:
        """
        Generate Tauri mock script for this target.

        Args:
            target: ExtractionTarget with mock configuration.

        Returns:
            JavaScript code to inject.
        """
        # Start with default mocks
        mocks = get_default_mocks()

        # Add Tauri config values
        if self.tauri_config:
            package = self.tauri_config.get("package", {})
            mocks["get_version"] = package.get("version", "1.0.0")
            mocks["get_app_name"] = package.get("productName", "Tauri App")

        # Merge with custom mocks from target
        if target.tauri_mocks:
            mocks.update(target.tauri_mocks)

        # Generate script
        return generate_mock_script(mocks)

    async def extract_current_state(self) -> RuntimeStateCapture:
        """
        Extract current state of the Tauri application.

        Returns:
            RuntimeStateCapture with all extracted data.
        """
        # Ensure mocks are still active
        if self.page and self.mock_script:
            # Check if __TAURI__ exists
            has_tauri = await self.page.evaluate("() => typeof window.__TAURI__ !== 'undefined'")
            if not has_tauri:
                # Re-inject mocks
                await self.page.add_init_script(self.mock_script)
                logger.debug("Re-injected Tauri mocks")

        # Use parent class extraction
        return await super().extract_current_state()

    @classmethod
    def supports_target(cls, target: ExtractionTarget) -> bool:
        """
        Check if this extractor can handle the given target.

        Args:
            target: Target to check.

        Returns:
            True if this extractor can handle the target.
        """
        from ..types import RuntimeType

        # Support Tauri targets
        if target.runtime_type == RuntimeType.TAURI:
            return True

        # Also support if URL is provided and Tauri config exists
        if target.url and target.tauri_config_path:
            config_path = Path(target.tauri_config_path)
            if config_path.exists():
                return True

        return False

    async def inject_custom_mock(self, command: str, response: Any) -> None:
        """
        Inject a custom mock response for a Tauri command.

        This can be used to customize behavior during extraction.

        Args:
            command: Tauri command name.
            response: Mock response (value or function result).
        """
        if not self.page:
            raise RuntimeError("Not connected to target")

        # Convert response to JSON
        response_json = json.dumps(response)

        # Inject into page
        await self.page.evaluate(
            f"() => {{ window.__TAURI_MOCKS__['{command}'] = {response_json}; }}"
        )

        logger.debug(f"Injected custom mock for command: {command}")

    async def get_tauri_logs(self) -> list[str]:
        """
        Get Tauri mock logs from the console.

        Returns:
            List of log messages from Tauri mock calls.
        """
        if not self.page:
            return []

        # Get console logs (would need to set up console listener)
        # This is a placeholder - actual implementation would require
        # setting up console message listeners when creating the page
        return []

    async def simulate_tauri_event(self, event: str, payload: Any) -> None:
        """
        Simulate a Tauri event being emitted.

        Args:
            event: Event name.
            payload: Event payload.
        """
        if not self.page:
            raise RuntimeError("Not connected to target")

        # Convert payload to JSON
        payload_json = json.dumps(payload)

        # Emit event
        await self.page.evaluate(
            f"() => {{ window.__TAURI__.event.emit('{event}', {payload_json}); }}"
        )

        logger.debug(f"Simulated Tauri event: {event}")
