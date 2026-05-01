"""Live E2E test: World State Verifier refuses a wrong-target click.

Scenario: launch mspaint, take a PRE screenshot, click at a location
that is NOT the red color swatch (we click the canvas interior), take
a POST screenshot, then ask the World State Verifier to judge whether
the intent "Select the red color swatch from the color picker" was
achieved. Assert the verdict is NOT pass — the action was a no-op on
the intended target, so the WSM should return `refused` or `fail`.

The companion positive scenario runs the same test with a no-op intent
("Keep the mspaint window visible and idle") against two identical
screenshots and expects a `pass` verdict — this validates that the
WSM prompt contract works end-to-end.

Skip conditions:
- Not on Windows
- WSM endpoint not reachable (llama-swap isn't running or model isn't loaded)
- pyautogui / pygetwindow / mss / PIL not importable

Run with::

    pytest qontinui/tests/e2e/broken_accessibility/test_world_state_verifier.py \
        -v --run-live-e2e

Environment overrides:
- QONTINUI_WORLD_STATE_VERIFIER_ENDPOINT (default: http://127.0.0.1:8100)
- QONTINUI_WORLD_STATE_VERIFIER_MODEL (default: cua-wsm)
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time

import pytest

from ._wsm_client import WorldStateVerifierClient, WsmVerdict

pytestmark = pytest.mark.live_e2e

logger = logging.getLogger(__name__)


def _png_bytes_from_bgr(bgr_array) -> bytes:
    """Encode a BGR numpy array as a PNG byte string."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not available for PNG encoding")
    # BGR -> RGB
    rgb = bgr_array[:, :, ::-1]
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def wsm_client() -> WorldStateVerifierClient:
    """Shared WSM client. Skips the module if the endpoint isn't reachable."""
    endpoint = os.environ.get(
        "QONTINUI_WORLD_STATE_VERIFIER_ENDPOINT", "http://127.0.0.1:8100"
    )
    model = os.environ.get("QONTINUI_WORLD_STATE_VERIFIER_MODEL", "cua-wsm")
    client = WorldStateVerifierClient(endpoint=endpoint, model=model)
    if not client.is_reachable():
        pytest.skip(
            f"WSM endpoint {endpoint} not reachable — llama-swap not running "
            "or cua-wsm model not configured. See "
            "qontinui/docker/llama-swap/config.yaml."
        )
    return client


def test_wsm_refuses_wrong_target_click(
    wsm_client: WorldStateVerifierClient,
    live_app,
    screen_capture,
) -> None:
    """The verifier should refuse a click that missed the intended target.

    We launch mspaint, snapshot PRE, programmatically click the canvas
    interior (not the red color swatch), snapshot POST, and ask WSM
    whether the intent "Select the red color swatch" was achieved.
    The click lands on the drawing canvas, not the palette — so the
    intended state change (red selected) did not occur.
    """
    if sys.platform != "win32":
        pytest.skip("mspaint scenario only runs on Windows")

    try:
        import pyautogui
    except ImportError:
        pytest.skip("pyautogui not available")
    try:
        import pygetwindow as gw
    except ImportError:
        pytest.skip("pygetwindow not available")

    # Disable pyautogui failsafe (mouse-in-corner abort) for automated runs.
    pyautogui.FAILSAFE = False

    app = live_app("mspaint.exe", window_title="Paint")
    # Let the window fully paint before snapshotting.
    time.sleep(1.5)

    wins = [
        w
        for w in gw.getAllWindows()
        if "paint" in (w.title or "").lower() and w.visible and w.width > 0
    ]
    if not wins:
        pytest.skip("No visible Paint window found after launch")
    win = wins[0]
    try:
        win.activate()
    except Exception:
        pass  # activation is best-effort on some Windows configs
    time.sleep(0.3)

    # PRE screenshot: mspaint is open, no action yet.
    pre_bgr = screen_capture()
    pre_png = _png_bytes_from_bgr(pre_bgr)

    # Click in the middle of the canvas (not a palette swatch).
    # The canvas interior is roughly the lower-center of the Paint window.
    canvas_x = win.left + win.width // 2
    canvas_y = win.top + (win.height * 2) // 3
    logger.info(
        "Clicking mspaint canvas interior at (%d, %d) — deliberately NOT the red color swatch",
        canvas_x,
        canvas_y,
    )
    pyautogui.click(canvas_x, canvas_y)
    time.sleep(0.5)

    # POST screenshot.
    post_bgr = screen_capture()
    post_png = _png_bytes_from_bgr(post_bgr)

    verdict: WsmVerdict = wsm_client.verify(
        pre_png_bytes=pre_png,
        post_png_bytes=post_png,
        intent="Select the red color swatch from the color picker in the Paint ribbon",
        goal="Change the active drawing color to red",
    )
    logger.info(
        "WSM wrong-target verdict: status=%s confidence=%.2f observations=%r",
        verdict.status,
        verdict.confidence,
        verdict.observations[:200],
    )

    # Core assertion: a click that missed the color picker should never
    # be judged as pass. "refused" is the ideal answer (no-op on intended
    # target); "fail" and "partial" are also acceptable — all of them
    # mean the loop would not advance and would retry. The failure mode
    # we're guarding against is a false "pass" verdict on a wrong-target
    # click, which would cause infinite advancing without the intended
    # state change.
    assert verdict.status in (
        "refused",
        "fail",
        "partial",
    ), (
        f"WSM incorrectly judged wrong-target click as {verdict.status!r} "
        f"(confidence={verdict.confidence:.2f}). Observations: "
        f"{verdict.observations!r}"
    )

    # When WSM returns refused, it should also populate refused_reason
    # and the observations should mention something about the wrong
    # target or no color selection — this exercises the prompt schema.
    if verdict.status == "refused":
        assert (
            verdict.refused_reason
            or "red" in verdict.observations.lower()
            or "color" in verdict.observations.lower()
        ), (
            f"Refused verdict lacks reason or color/red mention: "
            f"observations={verdict.observations!r}, "
            f"refused_reason={verdict.refused_reason!r}"
        )


def test_wsm_passes_noop_intent_on_identical_screenshots(
    wsm_client: WorldStateVerifierClient,
    live_app,
    screen_capture,
) -> None:
    """Control: verify the WSM prompt contract with a trivially-passing case.

    Two identical screenshots + a no-op intent should produce a `pass`
    verdict. This is the positive regression guard against prompt drift
    or model-level bugs that would cause every verdict to be `fail`.
    """
    if sys.platform != "win32":
        pytest.skip("mspaint scenario only runs on Windows")

    _ = live_app("mspaint.exe", window_title="Paint")
    time.sleep(1.5)

    shot_bgr = screen_capture()
    shot_png = _png_bytes_from_bgr(shot_bgr)

    verdict = wsm_client.verify(
        pre_png_bytes=shot_png,
        post_png_bytes=shot_png,
        intent="Keep the mspaint window visible and idle (no state change required)",
        goal="Ensure mspaint is running and visible",
    )
    logger.info(
        "WSM noop verdict: status=%s confidence=%.2f observations=%r",
        verdict.status,
        verdict.confidence,
        verdict.observations[:200],
    )

    # An unchanged screen plus a no-op intent should be judged pass
    # (or at worst partial). If this returns refused/fail the prompt
    # contract is broken, not the caller.
    assert verdict.status in ("pass", "partial"), (
        f"WSM judged a no-op intent on identical screenshots as "
        f"{verdict.status!r} (confidence={verdict.confidence:.2f}). "
        f"Observations: {verdict.observations!r}"
    )
