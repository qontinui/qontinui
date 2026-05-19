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

IS_WINDOWS: bool = sys.platform == "win32"
"""True on Windows. Drives `skipif` on Windows-only subsystems (UIA)."""

IS_CI: bool = bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))
"""True on hosted CI runners. Drives `skipif` on flaky perf assertions
whose ms-level thresholds aren't meaningful on shared/variable hardware."""


def _probe_display() -> bool:
    """True if mss can actually capture a real 1x1 screenshot.

    We deliberately do a `sct.grab(...)` instead of just enumerating
    monitors: under Xvfb on the GitHub-hosted Linux runner, the cheap
    `sct.monitors` lookup succeeds against the virtual display, but the
    real capture path (`xcb.connect(display)` invoked by a fresh MSS
    handle later in test code) fails with `display = None` because the
    DISPLAY/XAUTHORITY env doesn't propagate consistently into
    subprocess-spawned mss instances. Probing the same path as production
    avoids the false-positive that made the previous probe useless.
    """
    if IS_WINDOWS:
        return True
    # Linux CI is treated as no-display unconditionally: even if Xvfb is
    # up enough to enumerate monitors, the grab path is consistently
    # flaky across mss versions / runner images. Skip is the robust
    # contract — tests run normally on a Linux desktop with a real X
    # session (CI=unset).
    if IS_CI:
        return False
    try:
        import mss as _mss

        with _mss.mss() as sct:
            # Force the same capture path production code uses; cheap
            # monitor enumeration alone gives false positives on Xvfb.
            sct.grab({"left": 0, "top": 0, "width": 1, "height": 1})
        return True
    except Exception:
        return False


HAS_FUNCTIONAL_DISPLAY: bool = _probe_display()
"""True iff a real `sct.grab(...)` succeeds. False on Linux CI by
construction (see `_probe_display`). Drives `skipif` on tests that
exercise real screenshot/input subsystems."""
