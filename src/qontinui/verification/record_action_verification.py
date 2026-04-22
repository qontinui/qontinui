"""Adapter between the WSM judge and grounding-capture records.

The dynamic grounding-capture pipeline lives in
``qontinui-runner/python-bridge/services/trajectory_logger.py`` and
writes :class:`qontinui_train.export.grounding_record.GroundingAction`
entries. This module is the glue that lets any dynamic capture entry
point (or a one-off script) run the WSM judge against a before/after
pair and stamp the resulting verdict onto a ``GroundingAction``.

Scope boundary — the dynamic-capture pipeline itself is not rebuilt
here. The existing trajectory logger consumes this adapter through
its ``wsm_client`` slot; a brand-new dynamic capture script would
call :func:`verify_and_stamp` directly around each action it executes.

HITL hand-off: the ``runner_url`` argument points at the Tauri runner's
HTTP surface. Low-confidence verdicts are POSTed to its deferred-question
enqueue endpoint so a human can decide later whether the action
succeeded. When the runner is unreachable, the verdict is still stamped
on the record with ``source="pixel_diff"`` (the safe default) and a
warning is logged — we never block capture on HITL availability.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from .wsm_client import (
    DEFAULT_VERIFY_TIMEOUT_S,
    WSMVerdict,
    verify_action,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stamping
# ---------------------------------------------------------------------------


def record_wsm_verdict(grounding_action: Any, verdict: WSMVerdict) -> None:
    """Write *verdict* onto a :class:`GroundingAction` in place.

    ``grounding_action`` is typed as ``Any`` because the qontinui core
    package avoids importing ``qontinui_train`` to keep dependency
    direction one-way (train depends on qontinui, never the reverse).
    Callers must pass a ``GroundingAction`` instance.
    """
    grounding_action.success = verdict.success
    grounding_action.success_source = verdict.source


async def verify_and_stamp(
    grounding_action: Any,
    before_png_bytes: bytes,
    after_png_bytes: bytes,
    intent: str,
    *,
    goal: str | None = None,
    endpoint: str | None = None,
    model: str | None = None,
    timeout_s: float = DEFAULT_VERIFY_TIMEOUT_S,
    runner_url: str | None = None,
    task_run_id: str | None = None,
    action_id: str | None = None,
) -> WSMVerdict:
    """Run the WSM against (*before*, *after*) and stamp the action.

    Also enqueues a deferred question when the verdict is low-confidence
    so the HITL queue can reconcile after the fact.
    """
    verdict = await verify_action(
        before_png_bytes,
        after_png_bytes,
        intent,
        goal=goal,
        endpoint=endpoint,
        model=model,
        timeout_s=timeout_s,
    )
    record_wsm_verdict(grounding_action, verdict)

    if verdict.source == "pixel_diff":
        # Low confidence or transport failure — enqueue for human review.
        # Fire-and-forget: capture must not block on HITL availability.
        await _maybe_enqueue_deferred(
            verdict=verdict,
            intent=intent,
            runner_url=runner_url,
            task_run_id=task_run_id,
            action_id=action_id,
        )

    return verdict


# ---------------------------------------------------------------------------
# HITL enqueue
# ---------------------------------------------------------------------------


def resolve_runner_url(explicit: str | None = None) -> str | None:
    """Pick the runner base URL for HITL enqueue, or ``None`` to skip.

    Order:
        1. Explicit argument.
        2. ``QONTINUI_RUNNER_URL`` env var.
    There is no default — if neither is set, HITL enqueue is skipped
    (verdict is still stamped on the grounding record).
    """
    if explicit:
        return explicit.rstrip("/")
    val = os.environ.get("QONTINUI_RUNNER_URL")
    if val:
        return val.rstrip("/")
    return None


async def _maybe_enqueue_deferred(
    *,
    verdict: WSMVerdict,
    intent: str,
    runner_url: str | None,
    task_run_id: str | None,
    action_id: str | None,
) -> None:
    """POST a deferred question to the runner, if a URL is available.

    The runner does not currently expose a dedicated HTTP route for
    external (non-Tauri) deferred-question enqueue; see
    ``qontinui-runner/src-tauri/src/database/pg/deferred_questions.rs``
    for the write path used in-process. Until that route lands, this
    function logs a warning when the enqueue would have happened so
    the gap is visible in logs rather than silently swallowed.

    The chosen route path — ``POST {runner_url}/hitl/deferred-questions``
    — is deliberate and documented here so the Rust side can pick it up
    when adding the passthrough. Body shape mirrors
    :class:`DeferredQuestion` minus server-populated fields.
    """
    url = resolve_runner_url(runner_url)
    if url is None:
        logger.debug(
            "HITL enqueue skipped — no runner URL (verdict=%s confidence=%.2f intent=%r)",
            verdict.source,
            verdict.confidence,
            intent[:80],
        )
        return

    body: dict[str, Any] = {
        "task_run_id": task_run_id,
        "action_id": action_id,
        "question": (f"Low-confidence WSM verdict for action {action_id or '<unknown>'}: {intent}"),
        "auto_decision_type": "pixel_diff_fallback",
        "auto_decision_detail": verdict.reason,
        "confidence": verdict.confidence,
        "risk_level": "low",
        "wsm_raw": verdict.raw,
    }

    endpoint = f"{url}/hitl/deferred-questions"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(endpoint, json=body)
            if resp.status_code == 404:
                # Route not yet exposed by the runner — log once and
                # keep going. Do NOT duplicate the DB write here: the
                # runner owns its postgres schema and writing around it
                # from Python would create drift.
                logger.warning(
                    "HITL enqueue endpoint %s returned 404 — runner needs "
                    "to expose POST /hitl/deferred-questions for Python "
                    "capture pipelines. Verdict stamped on record, not "
                    "enqueued for review.",
                    endpoint,
                )
                return
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning(
            "HITL enqueue failed against %s: %s (verdict still stamped on record)",
            endpoint,
            exc,
        )
