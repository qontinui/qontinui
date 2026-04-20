"""Fixtures for broken-accessibility live E2E tests.

Provides:
- ``omniparser_service`` (session scope): brings up the OmniParser
  docker-compose stack, waits for /health, tears down on session exit.
- ``live_app``: factory fixture that launches a Win32 application by
  path, waits for its main window, yields a ``LiveApp`` handle, and
  terminates the process on teardown.
- ``screen_capture``: mss-backed full-screen grabber returning BGR numpy.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger(__name__)

# Use the unified ai-proxy stack by default so the live suite also
# exercises the Caddy reverse proxy routing. The proxy fronts
# OmniParser at http://localhost:8888/parse. Override to the
# standalone compose by setting QONTINUI_E2E_OMNIPARSER_STACK=standalone.
_STACK = os.environ.get("QONTINUI_E2E_OMNIPARSER_STACK", "proxy").lower()
if _STACK == "standalone":
    OMNIPARSER_COMPOSE_FILE = (
        Path(__file__).resolve().parents[3] / "docker" / "omniparser" / "docker-compose.yml"
    )
    OMNIPARSER_COMPOSE_SERVICES: list[str] = []  # bring up everything
    OMNIPARSER_BASE_URL = "http://localhost:8080"
    OMNIPARSER_HEALTH_URL = "http://localhost:8080/health"
else:
    OMNIPARSER_COMPOSE_FILE = (
        Path(__file__).resolve().parents[3] / "docker" / "ai-proxy" / "docker-compose.yml"
    )
    # Start only proxy + omniparser — llama-swap is behind the ``full``
    # profile and not needed for the broken-accessibility suite.
    OMNIPARSER_COMPOSE_SERVICES = ["proxy", "omniparser"]
    OMNIPARSER_BASE_URL = "http://localhost:8888"
    OMNIPARSER_HEALTH_URL = "http://localhost:8888/health/omniparser"
OMNIPARSER_STARTUP_TIMEOUT_S = 300  # first pull + model load can be slow
OMNIPARSER_POLL_INTERVAL_S = 3.0


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _wait_for_health(url: str, timeout_s: float) -> bool:
    import httpx

    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=5.0)
            if resp.status_code == 200:
                return True
        except Exception as e:
            last_err = e
        time.sleep(OMNIPARSER_POLL_INTERVAL_S)
    if last_err is not None:
        logger.warning("OmniParser health probe never succeeded: %s", last_err)
    return False


@pytest.fixture(scope="session")
def omniparser_service() -> Iterator[str]:
    """Start the OmniParser docker-compose stack for the test session.

    Yields the base URL of the service. Skips the test if Docker or the
    compose file is unavailable. Tears the stack down on session exit.
    """
    if not OMNIPARSER_COMPOSE_FILE.exists():
        pytest.skip(f"OmniParser compose file not found: {OMNIPARSER_COMPOSE_FILE}")
    if not _docker_available():
        pytest.skip("Docker / docker compose not available on PATH")

    compose_dir = OMNIPARSER_COMPOSE_FILE.parent
    logger.info(
        "Starting OmniParser (%s stack) via docker compose in %s",
        _STACK,
        compose_dir,
    )
    up_cmd = ["docker", "compose", "up", "-d", *OMNIPARSER_COMPOSE_SERVICES]
    up = subprocess.run(
        up_cmd,
        cwd=str(compose_dir),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if up.returncode != 0:
        pytest.skip(f"docker compose up failed (rc={up.returncode}): " f"stderr={up.stderr[:500]}")

    ready = _wait_for_health(OMNIPARSER_HEALTH_URL, OMNIPARSER_STARTUP_TIMEOUT_S)
    if not ready:
        # Try to capture logs for diagnostics, then tear down.
        subprocess.run(
            ["docker", "compose", "logs", "--tail", "80"],
            cwd=str(compose_dir),
            timeout=30,
        )
        subprocess.run(
            ["docker", "compose", "down"],
            cwd=str(compose_dir),
            timeout=60,
        )
        pytest.skip(
            f"OmniParser service did not become healthy within " f"{OMNIPARSER_STARTUP_TIMEOUT_S}s"
        )

    os.environ["QONTINUI_OMNIPARSER_ENABLED"] = "true"
    os.environ["QONTINUI_OMNIPARSER_PROVIDER"] = "service"
    os.environ["QONTINUI_OMNIPARSER_SERVICE_URL"] = OMNIPARSER_BASE_URL

    try:
        yield OMNIPARSER_BASE_URL
    finally:
        logger.info("Stopping OmniParser (%s stack)", _STACK)
        subprocess.run(
            ["docker", "compose", "down"],
            cwd=str(compose_dir),
            capture_output=True,
            timeout=60,
        )


@dataclass
class LiveApp:
    """Handle for a launched native application under test."""

    process: subprocess.Popen[bytes]
    window_title_fragment: str
    pid: int

    def terminate(self) -> None:
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        except Exception:
            logger.exception(
                "Failed to terminate %s (pid=%s)", self.window_title_fragment, self.pid
            )


def _wait_for_window(title_fragment: str, timeout_s: float = 15.0) -> bool:
    """Poll for a visible window whose title contains ``title_fragment``."""
    try:
        import pygetwindow as gw
    except ImportError:
        # Without pygetwindow we can't verify — just sleep a conservative
        # amount and hope the app came up. Good enough for a skip-on-fail
        # live harness.
        time.sleep(3.0)
        return True

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        wins = [w for w in gw.getAllWindows() if title_fragment.lower() in (w.title or "").lower()]
        if any(w.visible and w.width > 0 for w in wins):
            return True
        time.sleep(0.25)
    return False


@pytest.fixture
def live_app() -> Iterator[Callable[..., LiveApp]]:
    """Factory fixture: launches an exe and waits for its window.

    Usage::

        def test_something(live_app):
            app = live_app("mspaint.exe", window_title="Paint")
            # ... do work ...
    """
    launched: list[LiveApp] = []

    def _launch(
        exe: str,
        *,
        window_title: str,
        args: list[str] | None = None,
        timeout_s: float = 15.0,
    ) -> LiveApp:
        if sys.platform != "win32":
            pytest.skip(f"{exe} live test only runs on Windows")
        cmd = [exe] + (args or [])
        logger.info("Launching %s", cmd)
        proc = subprocess.Popen(cmd)
        if not _wait_for_window(window_title, timeout_s=timeout_s):
            proc.kill()
            pytest.skip(f"Window with title fragment {window_title!r} never appeared")
        app = LiveApp(process=proc, window_title_fragment=window_title, pid=proc.pid)
        launched.append(app)
        return app

    yield _launch

    for app in launched:
        app.terminate()


@pytest.fixture
def screen_capture() -> Callable[[], Any]:
    """Return a callable that grabs the primary monitor as a BGR numpy array."""

    def _grab() -> Any:
        import mss
        import numpy as np

        with mss.mss() as sct:
            mon = sct.monitors[1]
            shot = sct.grab(mon)
            arr = np.array(shot)  # BGRA
            return arr[:, :, :3]  # BGR

    return _grab
