"""VGA runtime — drives an external app through a saved state machine.

The runtime orchestrates: screenshot capture → resolve active state →
ground the target element → drift-check → HAL action. Each step emits a
structured :class:`VgaStepEvent` both as a list entry on the return
value AND via an optional ``on_event`` callback. The callback is what
the Rust step handler plugs into to stream per-step events into the
runner's event bus (plan §13 "structured events").

Blocking-state enforcement (plan §13 "corrected recommendation: modals
reuse existing ``blocking`` states"):
- If the state machine has any state with ``blocking=True``, the runtime
  treats it as the sole valid state to act in while it is active.
- The existing
  :class:`qontinui.state_management.enhanced_active_state_set.EnhancedActiveStateSet`
  is used verbatim — no new overlay abstraction.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict

from ..hal.interfaces.keyboard_controller import IKeyboardController
from ..hal.interfaces.mouse_controller import IMouseController, MouseButton
from ..hal.interfaces.screen_capture import IScreenCapture
from ..state_management.enhanced_active_state_set import EnhancedActiveStateSet
from .client import VgaClient, VgaClientError
from .drift import DriftDetector
from .shadow_log import ShadowSampleLogger
from .state_machine import BBox, VgaElement, VgaState, VgaStateMachine

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Action + event DTOs
# ----------------------------------------------------------------------


class VgaAction(BaseModel):
    """One step in a VGA action sequence.

    Three kinds are supported out of the gate:

    - ``click`` — ground + click the element identified by ``element_id``.
    - ``type`` — type ``text``; if ``element_id`` is set, click-to-focus
      first, otherwise type into whatever currently has focus.
    - ``wait_for`` — ground repeatedly until the element appears or
      ``timeout_ms`` elapses. Used to gate against slow app transitions.
    """

    model_config = ConfigDict(frozen=False)

    kind: Literal["click", "type", "wait_for"]
    element_id: UUID | None = None
    text: str | None = None
    timeout_ms: int = 10000


class VgaStepEvent(BaseModel):
    """One structured event emitted per runtime step.

    This is the payload the Rust handler streams up to the runner event
    bus. The ``kind`` prefix is intentional — it mirrors the shape of
    other runner events so they can be distinguished from non-VGA events
    on the same bus.
    """

    model_config = ConfigDict(frozen=False)

    kind: Literal["vga.step"] = "vga.step"
    action: VgaAction
    prompt: str
    bbox_pred: BBox | None = None
    bbox_last: BBox | None = None
    iou: float = 0.0
    template_similarity: float = 1.0
    status: Literal["ok", "drift", "failed"] = "ok"
    error: str | None = None


@dataclass
class VgaRunResult:
    """Aggregate outcome of :meth:`VgaRuntime.execute`."""

    run_id: UUID = field(default_factory=uuid4)
    status: Literal["success", "failed"] = "success"
    events: list[VgaStepEvent] = field(default_factory=list)
    error: str | None = None


class VgaRuntimeError(RuntimeError):
    """Raised when blocking-state policy rejects an action, or when a
    ``wait_for`` times out before the element appears. These are
    recoverable from the caller's perspective — the runtime maps them to
    a ``failed`` step event."""


# ----------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------


class VgaRuntime:
    """Executes a :class:`VgaStateMachine` against an external app.

    Args:
        client: VGA grounding client.
        hal_mouse: HAL mouse controller (any
            :class:`~qontinui.hal.interfaces.mouse_controller.IMouseController`).
        hal_keyboard: HAL keyboard controller.
        hal_capture: HAL screen-capture provider.
        drift_detector: Optional :class:`DriftDetector`. One is
            instantiated for the caller if omitted.
        active_state_set: Optional
            :class:`EnhancedActiveStateSet` so the caller can inspect
            blocking-state transitions after a run. One is instantiated
            if omitted.
    """

    def __init__(
        self,
        client: VgaClient,
        hal_mouse: IMouseController,
        hal_keyboard: IKeyboardController,
        hal_capture: IScreenCapture,
        drift_detector: DriftDetector | None = None,
        active_state_set: EnhancedActiveStateSet | None = None,
        shadow_logger: ShadowSampleLogger | None = None,
    ) -> None:
        self._client = client
        self._mouse = hal_mouse
        self._keyboard = hal_keyboard
        self._capture = hal_capture
        self._drift = drift_detector or DriftDetector()
        self._active = active_state_set or EnhancedActiveStateSet()
        # Optional shadow-sample writer. When present, every successful
        # grounding call is mirrored to ``runner.vga_shadow_samples`` so
        # the v6 trainer has production-distribution data for its no-
        # regression gate. Respects the SM's ``private`` flag — private
        # SMs never leak here.
        self._shadow = shadow_logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        sm: VgaStateMachine,
        actions: list[VgaAction],
        on_event: Callable[[VgaStepEvent], None] | None = None,
    ) -> VgaRunResult:
        """Run the sequence. Best-effort — failures surface as events.

        Args:
            sm: The state machine to act against. Blocking states are
                derived from its ``states`` list.
            actions: Sequence of actions to run.
            on_event: Optional per-step callback, called before the event
                is appended to the result. The Rust worker subprocess
                wires this to ``print(json.dumps(event))``.
        """
        result = VgaRunResult()

        # Seed the active-state tracker from the SM's blocking states so
        # enforcement uses the canonical abstraction. Non-blocking states
        # are implicitly "available".
        self._seed_active_states(sm)

        for action in actions:
            event = self._run_step(sm, action)
            if on_event is not None:
                try:
                    on_event(event)
                except Exception:
                    logger.exception("VgaRuntime: on_event callback raised")
            result.events.append(event)

            if event.status == "failed":
                result.status = "failed"
                result.error = event.error
                break

        return result

    # ------------------------------------------------------------------
    # Internal: per-step execution
    # ------------------------------------------------------------------

    def _run_step(self, sm: VgaStateMachine, action: VgaAction) -> VgaStepEvent:
        """Execute one action; produce exactly one :class:`VgaStepEvent`."""
        target: tuple[VgaState, VgaElement] | None = None
        if action.element_id is not None:
            target = sm.find_element(action.element_id)
            if target is None:
                return VgaStepEvent(
                    action=action,
                    prompt="",
                    status="failed",
                    error=f"unknown element_id {action.element_id}",
                )

            # Blocking-state enforcement: if any blocking state is active,
            # the target element must be in that state.
            try:
                self._enforce_blocking_policy(sm, target[0])
            except VgaRuntimeError as exc:
                return VgaStepEvent(
                    action=action,
                    prompt=target[1].prompt,
                    bbox_last=target[1].bbox,
                    status="failed",
                    error=str(exc),
                )

        try:
            if action.kind == "click":
                if target is None:
                    raise VgaRuntimeError("click requires element_id")
                return self._do_click(sm, action, *target)
            if action.kind == "type":
                return self._do_type(sm, action, target)
            if action.kind == "wait_for":
                if target is None:
                    raise VgaRuntimeError("wait_for requires element_id")
                return self._do_wait_for(sm, action, *target)
            raise VgaRuntimeError(f"unknown action kind {action.kind!r}")
        except VgaClientError as exc:
            return VgaStepEvent(
                action=action,
                prompt=target[1].prompt if target else "",
                bbox_last=target[1].bbox if target else None,
                status="failed",
                error=f"grounding failed: {exc}",
            )
        except VgaRuntimeError as exc:
            return VgaStepEvent(
                action=action,
                prompt=target[1].prompt if target else "",
                bbox_last=target[1].bbox if target else None,
                status="failed",
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Per-kind handlers
    # ------------------------------------------------------------------

    def _do_click(
        self,
        sm: VgaStateMachine,
        action: VgaAction,
        _state: VgaState,
        element: VgaElement,
    ) -> VgaStepEvent:
        screenshot = self._capture.capture_screen()
        pred_bbox = self._ground_element(sm, screenshot, element)
        drift = self._drift.check(pred_bbox, element.bbox, screenshot=screenshot)

        cx, cy = pred_bbox.center
        clicked = self._mouse.click_at(cx, cy, button=MouseButton.LEFT)
        status: Literal["ok", "drift", "failed"]
        error: str | None = None
        if not clicked:
            status, error = "failed", "HAL mouse click returned False"
        else:
            status = "drift" if drift.is_drift else "ok"

        return VgaStepEvent(
            action=action,
            prompt=element.prompt,
            bbox_pred=pred_bbox,
            bbox_last=element.bbox,
            iou=drift.iou,
            template_similarity=drift.template_similarity,
            status=status,
            error=error,
        )

    def _do_type(
        self,
        sm: VgaStateMachine,
        action: VgaAction,
        target: tuple[VgaState, VgaElement] | None,
    ) -> VgaStepEvent:
        text = action.text or ""

        pred_bbox: BBox | None = None
        bbox_last: BBox | None = None
        iou_val = 0.0
        sim_val = 1.0
        prompt = ""

        if target is not None:
            _state, element = target
            screenshot = self._capture.capture_screen()
            pred_bbox = self._ground_element(sm, screenshot, element)
            drift = self._drift.check(pred_bbox, element.bbox, screenshot=screenshot)
            bbox_last = element.bbox
            iou_val = drift.iou
            sim_val = drift.template_similarity
            prompt = element.prompt

            # Click to focus before typing.
            cx, cy = pred_bbox.center
            if not self._mouse.click_at(cx, cy, button=MouseButton.LEFT):
                return VgaStepEvent(
                    action=action,
                    prompt=prompt,
                    bbox_pred=pred_bbox,
                    bbox_last=bbox_last,
                    iou=iou_val,
                    template_similarity=sim_val,
                    status="failed",
                    error="HAL mouse click returned False (focus step)",
                )

        typed = self._keyboard.type_text(text)
        status: Literal["ok", "drift", "failed"]
        if not typed:
            status = "failed"
            error: str | None = "HAL keyboard type_text returned False"
        else:
            status = "ok"
            error = None

        return VgaStepEvent(
            action=action,
            prompt=prompt,
            bbox_pred=pred_bbox,
            bbox_last=bbox_last,
            iou=iou_val,
            template_similarity=sim_val,
            status=status,
            error=error,
        )

    def _do_wait_for(
        self,
        sm: VgaStateMachine,
        action: VgaAction,
        _state: VgaState,
        element: VgaElement,
    ) -> VgaStepEvent:
        deadline = time.monotonic() + max(0.0, action.timeout_ms / 1000.0)
        last_pred: BBox | None = None
        last_iou = 0.0
        last_sim = 1.0
        last_error: str | None = None

        while time.monotonic() < deadline:
            try:
                screenshot = self._capture.capture_screen()
                pred_bbox = self._ground_element(sm, screenshot, element)
                drift = self._drift.check(pred_bbox, element.bbox, screenshot=screenshot)
                last_pred = pred_bbox
                last_iou = drift.iou
                last_sim = drift.template_similarity
                if not drift.is_drift:
                    return VgaStepEvent(
                        action=action,
                        prompt=element.prompt,
                        bbox_pred=pred_bbox,
                        bbox_last=element.bbox,
                        iou=last_iou,
                        template_similarity=last_sim,
                        status="ok",
                    )
            except VgaClientError as exc:
                last_error = str(exc)

            time.sleep(0.25)

        return VgaStepEvent(
            action=action,
            prompt=element.prompt,
            bbox_pred=last_pred,
            bbox_last=element.bbox,
            iou=last_iou,
            template_similarity=last_sim,
            status="failed",
            error=last_error or "wait_for timeout",
        )

    # ------------------------------------------------------------------
    # Grounding + blocking-state policy
    # ------------------------------------------------------------------

    def _ground_element(self, sm: VgaStateMachine, screenshot: Any, element: VgaElement) -> BBox:
        """Ground ``element.prompt`` against the current screenshot.

        Returns a bbox centered on the predicted point, sized to match
        ``element.bbox`` (so IoU is meaningful against the stored bbox).

        Side effect: when a :class:`ShadowSampleLogger` is configured and
        the SM is not private, logs a shadow sample containing the
        captured PNG + the predicted bbox for offline v6 re-evaluation.
        """
        result = self._client.ground(screenshot, element.prompt)
        w = max(1, element.bbox.w)
        h = max(1, element.bbox.h)
        x = max(0, result.x - w // 2)
        y = max(0, result.y - h // 2)
        pred = BBox(x=x, y=y, w=w, h=h)
        self._maybe_log_shadow(sm, screenshot, element, pred, result.confidence)
        return pred

    def _maybe_log_shadow(
        self,
        sm: VgaStateMachine,
        screenshot: Any,
        element: VgaElement,
        pred: BBox,
        confidence: float,
    ) -> None:
        """Best-effort write to ``runner.vga_shadow_samples``.

        A capture failure or encode failure is swallowed — the shadow
        log is advisory, not load-bearing.
        """
        if self._shadow is None:
            return
        if getattr(sm, "private", True):
            return
        try:
            png_bytes = _to_png_bytes(screenshot)
        except Exception:
            logger.exception("VgaRuntime: could not encode screenshot for shadow log")
            return

        self._shadow.log_sample(
            image_png_bytes=png_bytes,
            state_machine_id=sm.id,
            target_process=sm.target_process,
            prompt=element.prompt,
            v5_bbox={"x": pred.x, "y": pred.y, "w": pred.w, "h": pred.h},
            v5_model=self._client.model,
            private=False,
            confidence=confidence,
        )

    def _seed_active_states(self, sm: VgaStateMachine) -> None:
        """Populate the shared :class:`EnhancedActiveStateSet` with any
        blocking state the SM already flags.

        VGA state ids are UUIDs but the existing set expects ``int``s, so
        we hash the UUIDs to stable ints. The mapping is bijective within
        the life of one run — good enough for enforcement.
        """
        self._state_id_map: dict[int, VgaState] = {}
        for state in sm.states:
            int_id = state.id.int & 0x7FFFFFFF
            self._state_id_map[int_id] = state
            if state.blocking:
                self._active.add_state(int_id, blocking=True)

    def _enforce_blocking_policy(self, _sm: VgaStateMachine, target_state: VgaState) -> None:
        """Reject actions that target a non-blocking state while any
        blocking state is active.

        Raises:
            VgaRuntimeError: With a descriptive message.
        """
        blocking_ids = self._active.blocking_states
        if not blocking_ids:
            return

        target_int_id = target_state.id.int & 0x7FFFFFFF
        if target_int_id in blocking_ids:
            return

        raise VgaRuntimeError(
            f"blocking state(s) {sorted(blocking_ids)} active; "
            f"cannot act on state {target_state.name!r} until dismissed"
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _to_png_bytes(screenshot: Any) -> bytes:
    """Encode a screenshot (PIL Image, numpy ndarray, or raw bytes) to PNG.

    The HAL's ``capture_screen`` returns a PIL ``Image.Image`` on all
    platforms but this helper is lenient so tests can pass raw bytes
    directly.
    """
    if isinstance(screenshot, bytes):
        return screenshot

    # PIL import is deferred so this module stays importable without PIL
    # in environments that never hit shadow logging (unlikely, but cheap
    # insurance).
    from PIL import Image as PILImage

    if isinstance(screenshot, PILImage.Image):
        buf = io.BytesIO()
        screenshot.save(buf, format="PNG")
        return buf.getvalue()

    # Fall back to numpy → PIL conversion (mirrors VgaClient._encode_image).
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover — numpy is a hard dep
        raise RuntimeError("numpy is required to encode ndarray screenshots") from exc

    if isinstance(screenshot, np.ndarray):
        pil_img = PILImage.fromarray(screenshot)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

    raise TypeError(f"Unsupported screenshot type for shadow log: {type(screenshot).__name__}")
