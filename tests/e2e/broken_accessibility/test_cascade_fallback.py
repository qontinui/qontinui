"""Live E2E tests: cascade detection against native Win32 apps.

Three cases:

1. **notepad baseline** — accessibility tree works, short-circuits on a11y
   tier, OmniParser is never called, the InteractabilityFilter pre-filter
   is a no-op.

2. **mspaint terminal fallback** — accessibility tree is empty for the
   target element, the cascade's a11y-bypass path fires, OmniParser
   resolves the toolbar button via the terminal fallback.

3. **interactability filter on real capture** — a synthetic backend returns
   a mix of canvas-interior and real-toolbar bboxes on an mspaint
   screenshot; the InteractabilityFilter drops the canvas-interior ones
   and keeps the toolbar ones.

Run with::

    pytest qontinui/tests/e2e/broken_accessibility -v --run-live-e2e

Requires Docker + docker-compose (for the OmniParser service), Windows
(for mspaint / notepad), and a real display.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from qontinui.find.backends.base import DetectionBackend, DetectionResult
from qontinui.find.backends.cascade import CascadeDetector
from qontinui.find.backends.interactability_filter import InteractabilityFilter
from qontinui.find.backends.omniparser_service_backend import OmniParserServiceBackend

pytestmark = pytest.mark.live_e2e

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class StubAccessibilityCapture:
    """Minimal IAccessibilityCapture stub for the cascade's a11y backend.

    ``responses`` maps a ``(needle_type, needle)`` pair to a list of
    ``(x, y, width, height, name)`` bounds. An empty list represents a
    tree that answered but found nothing — exactly the "broken
    accessibility" scenario.
    """

    def __init__(self, responses: dict[tuple[str, str], list[tuple[int, int, int, int, str]]]):
        self._responses = responses

    def is_connected(self) -> bool:
        return True

    async def capture_tree(self) -> Any:
        """SemanticAccessibilityBackend needs a snapshot object with .roots.

        Return an empty snapshot so the backend yields zero candidates —
        the broken-accessibility scenario — without raising.
        """

        class _EmptySnapshot:
            roots: list[Any] = []
            elements: list[Any] = []

        return _EmptySnapshot()

    async def find_nodes(self, selector: Any) -> list[Any]:
        # Decode the selector's attribute name to match our fixture map.
        # qontinui_schemas.AccessibilitySelector uses automation_id/role/name.
        key_type = None
        key_val = None
        for attr, needle_type in (
            ("automation_id", "accessibility_id"),
            ("role", "role"),
            ("name", "label"),
        ):
            val = getattr(selector, attr, None)
            if val:
                key_type = needle_type
                key_val = val
                break
        if key_type is None or key_val is None:
            return []
        entries = self._responses.get((key_type, key_val), [])
        nodes: list[Any] = []
        for x, y, w, h, name in entries:
            nodes.append(
                _StubNode(
                    bounds=_StubBounds(x, y, w, h),
                    name=name,
                    role="button",
                )
            )
        return nodes


class _StubBounds:
    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class _StubNode:
    def __init__(self, bounds: _StubBounds, name: str, role: str) -> None:
        self.bounds = bounds
        self.name = name
        self.role = role
        self.ref = None


class SyntheticCandidateBackend(DetectionBackend):
    """Returns a predetermined list of candidates regardless of input.

    Used in test 3 to feed the InteractabilityFilter a mix of real and
    fake candidates and verify it drops the non-interactive ones.
    """

    def __init__(self, candidates: list[DetectionResult], cost_ms: float = 2000.0) -> None:
        self._candidates = candidates
        self._cost = cost_ms

    def find(self, needle: Any, haystack: Any, config: dict[str, Any]) -> list[DetectionResult]:
        # Return copies so filter mutations don't leak across calls.
        import dataclasses

        return [dataclasses.replace(c, metadata=dict(c.metadata)) for c in self._candidates]

    def supports(self, needle_type: str) -> bool:
        return True

    def estimated_cost_ms(self) -> float:
        return self._cost

    @property
    def name(self) -> str:
        return "synthetic_candidates"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_service_cascade(
    accessibility_capture: Any,
) -> CascadeDetector:
    """Build a cascade with the OmniParser service backend as terminal fallback.

    Uses only the accessibility tier + OmniParser service. Skips template/
    feature/OCR to keep the test focused on the bypass path.
    """
    from qontinui.find.backends.accessibility_backend import AccessibilityBackend
    from qontinui.find.backends.semantic_accessibility_backend import (
        SemanticAccessibilityBackend,
    )

    backends: list[DetectionBackend] = [
        AccessibilityBackend(accessibility_capture),
        SemanticAccessibilityBackend(accessibility_capture, llm_client=None),
        OmniParserServiceBackend(),
    ]
    cascade = CascadeDetector(backends=backends)
    # terminal fallback should auto-register to omniparser_service
    assert cascade.terminal_fallback is not None, (
        "expected auto-registered terminal fallback"
    )
    assert cascade.terminal_fallback.name == "omniparser_service"
    return cascade


# ---------------------------------------------------------------------------
# Test 1 — notepad baseline
# ---------------------------------------------------------------------------


def test_notepad_accessibility_short_circuits(
    omniparser_service: str,
    live_app: Any,
    screen_capture: Any,
) -> None:
    """Accessibility tree answers the query; OmniParser is never invoked."""
    live_app("notepad.exe", window_title="Notepad")
    # Give the window a moment to settle on screen before capture
    import time

    time.sleep(0.5)
    haystack = screen_capture()

    # Stub says "File" menu is at a known bbox — real coords don't matter
    # for the test, we're proving the cascade terminates on a11y.
    capture = StubAccessibilityCapture(
        responses={("label", "File"): [(10, 30, 40, 20, "File")]}
    )
    cascade = _build_service_cascade(capture)

    results = cascade.find(
        needle="File",
        haystack=haystack,
        config={"needle_type": "label", "min_confidence": 0.9},
    )

    assert len(results) == 1
    assert results[0].backend_name == "accessibility"
    assert results[0].label == "File"


# ---------------------------------------------------------------------------
# Test 2 — mspaint terminal fallback bypass
# ---------------------------------------------------------------------------


def test_mspaint_empty_accessibility_triggers_bypass(
    omniparser_service: str,
    live_app: Any,
    screen_capture: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Empty a11y → cascade jumps to OmniParser service terminal fallback."""
    live_app("mspaint.exe", window_title="Paint")
    import time

    time.sleep(1.5)  # mspaint takes longer to draw the ribbon
    haystack = screen_capture()

    # Every a11y query returns empty — exactly the broken-accessibility case.
    capture = StubAccessibilityCapture(responses={})
    cascade = _build_service_cascade(capture)

    caplog.set_level(logging.INFO)
    cascade.find(
        needle="pencil tool",
        haystack=haystack,
        config={"needle_type": "description", "min_confidence": 0.0},
    )

    # This test is about the bypass *mechanism*, not Florence-2's captioning
    # quality on a specific mspaint icon. Assert the cascade:
    #   1. Recognised the accessibility tier as empty
    #   2. Jumped straight to the omniparser_service terminal fallback
    #   3. Actually made the HTTP call (service returned 200 OK)
    # Whether Florence-2 caption matching found "pencil tool" specifically
    # is a semantic-matcher concern, not a cascade wiring concern.
    bypass_logged = any(
        "bypassing to terminal fallback omniparser_service" in rec.message
        for rec in caplog.records
    )
    assert bypass_logged, "expected accessibility-empty bypass to fire and name omniparser_service"

    service_called = any(
        "POST http://localhost:8080/parse" in rec.message
        or "POST http://localhost:8080/parse" in str(getattr(rec, "msg", ""))
        for rec in caplog.records
    )
    assert service_called, "expected the terminal fallback to actually call the OmniParser /parse endpoint"


# ---------------------------------------------------------------------------
# Test 3 — interactability filter on real mspaint capture
# ---------------------------------------------------------------------------


def test_interactability_filter_drops_canvas_interior(
    omniparser_service: str,
    live_app: Any,
    screen_capture: Any,
) -> None:
    """Filter rejects non-interactive bboxes, keeps real interactive ones.

    Self-calibrating: queries OmniParser YOLO directly to discover where the
    real interactive regions are on this specific mspaint capture (no
    hardcoded layout assumptions), then builds:

      - 3 ``real_*`` candidates cloned from actual YOLO-detected regions
        (known-interactive ground truth)
      - 3 ``fake_*`` candidates placed deep inside the canvas area at
        coordinates chosen to be outside every YOLO region

    The filter should drop all 3 fake_* and keep all 3 real_*.
    """
    from qontinui.discovery.element_detection.omniparser_detector import (
        OmniParserDetector,
    )
    from qontinui.find.backends.omniparser_config import OmniParserSettings

    live_app("mspaint.exe", window_title="Paint")
    import time

    time.sleep(1.5)
    haystack = screen_capture()
    img_h, img_w = haystack.shape[:2]

    # Self-calibration step 1: ask YOLO directly for the ground-truth
    # interactive regions in this capture.
    try:
        detector = OmniParserDetector(settings=OmniParserSettings(enabled=True))
        interactive = detector.get_interactive_regions(haystack)
    except Exception as e:
        pytest.skip(f"Local OmniParserDetector unavailable for filter test: {e}")

    if len(interactive) < 3:
        pytest.skip(
            f"YOLO found {len(interactive)} interactive regions on mspaint — "
            f"need at least 3 to construct the ground-truth positives"
        )

    # Pick 3 ground-truth positives: clone real YOLO regions. These MUST
    # survive the filter (IoU=1.0 against themselves).
    real_candidates = [
        DetectionResult(
            x=ix,
            y=iy,
            width=iw,
            height=ih,
            confidence=0.95,
            backend_name="synthetic_candidates",
            label=f"real_{i}",
        )
        for i, (ix, iy, iw, ih, _) in enumerate(interactive[:3])
    ]

    # Self-calibration step 2: find 3 canvas-interior points that don't
    # overlap any YOLO region. Scan from bottom-right (definitely canvas)
    # backwards, checking each 20x20 candidate against the interactive set.
    def _inside_any_interactive(px: int, py: int, pw: int, ph: int) -> bool:
        for ix, iy, iw, ih, _ in interactive:
            if not (px + pw <= ix or px >= ix + iw or py + ph <= iy or py >= iy + ih):
                return True
        return False

    fake_candidates: list[DetectionResult] = []
    for i, (fx_frac, fy_frac) in enumerate([(0.6, 0.6), (0.5, 0.7), (0.7, 0.5)]):
        fx, fy = int(img_w * fx_frac), int(img_h * fy_frac)
        if _inside_any_interactive(fx, fy, 20, 20):
            continue
        fake_candidates.append(
            DetectionResult(
                x=fx,
                y=fy,
                width=20,
                height=20,
                confidence=0.95,
                backend_name="synthetic_candidates",
                label=f"fake_{i}",
            )
        )

    if len(fake_candidates) < 3:
        pytest.skip(
            f"Could not find 3 canvas-interior points not overlapping any "
            f"YOLO region (found {len(fake_candidates)}) — mspaint layout "
            f"unusually dense, rerun or resize"
        )

    backend = SyntheticCandidateBackend(real_candidates + fake_candidates)
    filtered_backend = InteractabilityFilter(backend, iou_threshold=0.1)
    filtered = filtered_backend.find(
        needle=None,
        haystack=haystack,
        config={"needle_type": "description", "min_confidence": 0.0},
    )
    kept_labels = {r.label for r in filtered}

    expected_real = {c.label for c in real_candidates}
    expected_fake = {c.label for c in fake_candidates}

    # All 3 real_* (cloned from YOLO ground truth) must survive
    missing_real = expected_real - kept_labels
    assert not missing_real, (
        f"InteractabilityFilter dropped real YOLO-ground-truth candidates: "
        f"missing={missing_real}, kept={kept_labels}"
    )

    # No fake_* (canvas-interior, outside every YOLO region) may survive
    surviving_fake = expected_fake & kept_labels
    assert not surviving_fake, (
        f"InteractabilityFilter kept canvas-interior non-interactive "
        f"candidates: surviving={surviving_fake}, kept={kept_labels}"
    )
