"""CLI entrypoint for the Rust ``vga_automate`` step handler.

Invocation shape (matches what the handler will fork):

.. code-block::

    python -m qontinui.vga.worker \
        --state-machine-id <UUID> \
        --action-sequence-json <path to json file> \
        --target-process <process name> \
        --pg-url <postgres url>

The worker:

1. Loads the state machine row from ``runner.vga_state_machines`` via
   psycopg (sync — the caller is a short-lived subprocess).
2. Constructs a :class:`VgaRuntime` wired to the platform-default HAL
   implementations.
3. Runs the action sequence, printing one JSON event per line to stdout
   for each :class:`VgaStepEvent` — the Rust handler line-parses these.
4. Emits a terminal ``{"kind": "vga.done", ...}`` event and exits 0 on
   success / 1 on failure.

psycopg is imported lazily so the worker module is safely importable in
test environments without a running Postgres.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from .client import VgaClient
from .runtime import VgaAction, VgaRuntime, VgaStepEvent
from .state_machine import VgaStateMachine

logger = logging.getLogger(__name__)


def _emit(event: dict[str, Any]) -> None:
    """Print a single JSON line to stdout and flush."""
    print(json.dumps(event, separators=(",", ":")), flush=True)


class _RunRecorder:
    """Append-only recorder for a ``runner.vga_runs`` row.

    Inserts a ``status='running'`` row at the start of a run and appends each
    ``VgaStepEvent`` to ``step_log`` as it occurs. On finish it updates
    ``status`` and ``ended_at``. Failures to write are logged and skipped —
    the runtime must not be blocked on PG availability.
    """

    def __init__(
        self,
        pg_url: str,
        run_id: UUID,
        state_machine_id: UUID,
        grounding_model: str,
    ) -> None:
        self._pg_url = pg_url
        self._run_id = run_id
        self._sm_id = state_machine_id
        self._grounding_model = grounding_model
        self._task_run_id = os.environ.get("QONTINUI_TASK_RUN_ID")
        self._disabled = False
        try:
            import psycopg  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            logger.warning(
                "psycopg not available — VGA run history will not be persisted"
            )
            self._disabled = True

    def start(self) -> None:
        if self._disabled:
            return
        try:
            import psycopg  # type: ignore[import-not-found]

            with psycopg.connect(self._pg_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("SET search_path TO runner, public")
                    cur.execute(
                        "INSERT INTO runner.vga_runs "
                        "(id, state_machine_id, task_run_id, grounding_model, "
                        " status, step_log, started_at) "
                        "VALUES (%s, %s, %s, %s, 'running', '[]'::jsonb, NOW())",
                        (
                            str(self._run_id),
                            str(self._sm_id),
                            self._task_run_id,
                            self._grounding_model,
                        ),
                    )
                conn.commit()
        except Exception:
            logger.exception("Failed to insert vga_runs row; continuing")
            self._disabled = True

    def append_event(self, event: dict[str, Any]) -> None:
        if self._disabled:
            return
        try:
            import psycopg  # type: ignore[import-not-found]

            with psycopg.connect(self._pg_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("SET search_path TO runner, public")
                    cur.execute(
                        "UPDATE runner.vga_runs "
                        "SET step_log = step_log || %s::jsonb "
                        "WHERE id = %s",
                        (json.dumps([event]), str(self._run_id)),
                    )
                conn.commit()
        except Exception:
            logger.exception("Failed to append vga_runs step; continuing")

    def finish(self, status: str, error: str | None = None) -> None:
        if self._disabled:
            return
        try:
            import psycopg  # type: ignore[import-not-found]

            with psycopg.connect(self._pg_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("SET search_path TO runner, public")
                    if error:
                        cur.execute(
                            "UPDATE runner.vga_runs "
                            "SET status = %s, ended_at = NOW(), "
                            "    step_log = step_log || %s::jsonb "
                            "WHERE id = %s",
                            (
                                status,
                                json.dumps(
                                    [
                                        {
                                            "kind": "vga.error",
                                            "error": error,
                                            "ts": datetime.now(UTC).isoformat(),
                                        }
                                    ]
                                ),
                                str(self._run_id),
                            ),
                        )
                    else:
                        cur.execute(
                            "UPDATE runner.vga_runs "
                            "SET status = %s, ended_at = NOW() "
                            "WHERE id = %s",
                            (status, str(self._run_id)),
                        )
                conn.commit()
        except Exception:
            logger.exception("Failed to finalize vga_runs row; continuing")


def _load_state_machine_from_pg(pg_url: str, sm_id: UUID) -> VgaStateMachine:
    """Load a VGA state machine row from ``runner.vga_state_machines``.

    The row's ``state_graph`` JSONB column holds the canonical JSON
    payload that :meth:`VgaStateMachine.from_canonical_json` consumes.

    Raises:
        RuntimeError: If psycopg is missing, the row isn't found, or the
            JSON is malformed.
    """
    try:
        import psycopg  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "psycopg is required to load VGA state machines from Postgres; "
            "install with `pip install psycopg[binary]`"
        ) from exc

    with psycopg.connect(pg_url) as conn:  # type: ignore[attr-defined]
        with conn.cursor() as cur:
            cur.execute("SET search_path TO runner, public")
            cur.execute(
                "SELECT id, name, target_process, target_os, grounding_model, "
                "private, state_graph "
                "FROM runner.vga_state_machines WHERE id = %s",
                (str(sm_id),),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"state machine {sm_id} not found")

            _id, name, target_process, target_os, grounding_model, private, state_graph = row

            # state_graph is JSONB — psycopg returns it as a dict already.
            # The builder stores the canonical JSON inside this column; we
            # let VgaStateMachine.from_canonical_json populate missing
            # timestamps if any.
            if isinstance(state_graph, str):
                state_graph = json.loads(state_graph)

            # Prefer the canonical graph for nested fields; override
            # top-level scalars from dedicated columns (source of truth).
            state_graph.setdefault("id", str(_id))
            state_graph["name"] = name
            state_graph["target_process"] = target_process
            state_graph["target_os"] = target_os
            state_graph["grounding_model"] = grounding_model
            state_graph["private"] = bool(private)

            return VgaStateMachine.from_canonical_json(state_graph)


def _load_actions(actions_path: Path) -> list[VgaAction]:
    """Parse the action-sequence JSON file into :class:`VgaAction` models."""
    payload = json.loads(actions_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(
            f"action sequence must be a JSON list; got {type(payload).__name__}"
        )
    return [VgaAction.model_validate(item) for item in payload]


def _build_runtime() -> VgaRuntime:
    """Wire a :class:`VgaRuntime` to the default HAL implementations.

    Uses ``pyautogui`` for input + ``mss`` for capture — both are already
    hard deps of qontinui on all platforms.
    """
    from ..hal.implementations.mss_capture import MSSCapture
    from ..hal.implementations.pyautogui_keyboard import PyAutoGUIKeyboard
    from ..hal.implementations.pyautogui_mouse import PyAutoGUIMouse

    client = VgaClient()
    return VgaRuntime(
        client=client,
        hal_mouse=PyAutoGUIMouse(),
        hal_keyboard=PyAutoGUIKeyboard(),
        hal_capture=MSSCapture(),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qontinui.vga.worker",
        description="Execute a VGA state machine against a running target process.",
    )
    parser.add_argument("--state-machine-id", required=True, type=UUID)
    parser.add_argument("--action-sequence-json", required=True, type=Path)
    parser.add_argument("--target-process", required=True, type=str)
    parser.add_argument("--pg-url", required=True, type=str)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns an exit code."""
    args = _parse_args(argv)
    run_id = uuid4()
    recorder: _RunRecorder | None = None

    try:
        sm = _load_state_machine_from_pg(args.pg_url, args.state_machine_id)
        actions = _load_actions(args.action_sequence_json)
        runtime = _build_runtime()
        recorder = _RunRecorder(
            pg_url=args.pg_url,
            run_id=run_id,
            state_machine_id=args.state_machine_id,
            grounding_model=sm.grounding_model,
        )
        recorder.start()
    except Exception as exc:
        if recorder is not None:
            recorder.finish("failed", error=f"{type(exc).__name__}: {exc}")
        _emit(
            {
                "kind": "vga.done",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "run_id": str(run_id),
            }
        )
        logger.exception("VGA worker setup failed")
        return 1

    def _on_event(event: VgaStepEvent) -> None:
        payload = event.model_dump(mode="json")
        _emit(payload)
        if recorder is not None:
            recorder.append_event(payload)

    result = runtime.execute(sm, actions, on_event=_on_event)
    # The worker-allocated run_id is the authoritative one (it's what landed
    # in the vga_runs row at recorder.start()); override the runtime's
    # factory-generated UUID so the stdout/Rust handler reports the same ID.
    result.run_id = run_id

    if recorder is not None:
        recorder.finish(result.status, error=result.error)

    _emit(
        {
            "kind": "vga.done",
            "status": result.status,
            "error": result.error,
            "run_id": str(run_id),
        }
    )
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
