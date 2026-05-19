"""Environment probes for tests that require real OS resources.

Some tests genuinely need a functional screen-capture / input subsystem
(real `mss` screenshots, real `pynput` mouse, real UIA on Windows). On
a headless Linux CI runner (Xvfb, no GPU, no real monitors) those
subsystems either can't initialise or behave so differently that
asserting on their output is unsound. Such tests skip cleanly when the
required capability isn't present.

This module is imported by test files; keep it side-effect-free at
import time so collection-time errors don't cascade.
"""

from __future__ import annotations

import os
import sys


def _probe_display() -> bool:
    """True if mss can actually connect to a display and enumerate monitors.

    On Windows this is always true (WinAPI backend). On Linux it requires
    a working X connection (display + Xauthority). Failure modes vary
    (XError, XauthError, OSError); all are treated as "no display".
    """
    if sys.platform == "win32":
        return True
    try:
        import mss as _mss

        with _mss.mss() as sct:
            # Touching .monitors forces an actual connection.
            _ = sct.monitors
        return True
    except Exception:
        return False


HAS_FUNCTIONAL_DISPLAY: bool = _probe_display()
"""True iff `mss.mss().monitors` succeeds. Drives `skipif` on tests that
take real screenshots or drive real OS input."""

IS_CI: bool = bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))
"""True on hosted CI runners. Drives `skipif` on flaky perf assertions
whose ms-level thresholds aren't meaningful on shared/variable hardware."""

IS_WINDOWS: bool = sys.platform == "win32"
"""True on Windows. Drives `skipif` on Windows-only subsystems (UIA)."""
