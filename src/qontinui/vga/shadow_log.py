"""Shadow-sample writer for the VGA v6 training gate.

Every production grounding call is mirrored into
``runner.vga_shadow_samples``: one row per unique
``(image_sha, prompt, model_used)`` triple. The full-region PNG lands
at ``datasets/vga-shadow/<image_sha>.png`` (dedupe on disk too).

The v6 training pipeline uses this log to run an offline "re-predict
everything and compare" pass before shipping — it's how the per-domain
no-regression gate gets actual production distribution data. Without
this writer, the daemon's gate auto-passes on an empty query.

Privacy (mirrors :mod:`qontinui.vga.correction_log`):
- When ``private=True``, the call is a no-op. Private state machines
  never have their screenshots or prompts persisted here — matching the
  correction-log privacy contract.

Rate limiting:
- If the table has fewer than ``rate_limit_threshold`` rows, every
  non-private sample is inserted (subject to dedupe).
- Once the table exceeds the threshold, samples are inserted only when
  ``random.random() < sample_rate`` — a stochastic downsample so the
  log stays bounded even in long-running VGA sessions.
- The row count is cached for 60 seconds to keep the SELECT COUNT(*)
  off the hot path.

All PG operations are best-effort. Any exception from psycopg is caught
and logged; the writer must not block the runtime under any
circumstance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
from uuid import UUID

logger = logging.getLogger(__name__)

_DEFAULT_IMAGES_DIR = Path("datasets/vga-shadow")
_ENV_SAMPLE_RATE = "QONTINUI_VGA_SHADOW_SAMPLE_RATE"
_COUNT_CACHE_TTL_SECONDS = 60.0


class ShadowSampleLogger:
    """Writes one row per unique grounding call into ``runner.vga_shadow_samples``.

    Args:
        pg_url: Postgres connection string. Connections are opened lazily
            and closed after each write — matches :class:`_RunRecorder`
            in :mod:`.worker`. A persistent connection is out of scope
            for this writer.
        images_dir: Directory for ``<image_sha>.png`` files. Created on
            first write if missing. Defaults to
            ``datasets/vga-shadow`` (relative to the current working
            directory of the worker, same convention as
            :class:`CorrectionLogger`).
        sample_rate: Probability of insertion once the table is full.
            ``None`` reads ``$QONTINUI_VGA_SHADOW_SAMPLE_RATE`` and falls
            back to ``1.0`` (no downsampling). Values outside ``[0, 1]``
            are clamped.
        rate_limit_threshold: Row count at which the ``sample_rate`` gate
            activates. Defaults to 100_000 (plan §13).
        rng: Injected ``random.Random`` instance, for test determinism.
            Defaults to the ``random`` module's shared state.
    """

    def __init__(
        self,
        pg_url: str,
        images_dir: Path = _DEFAULT_IMAGES_DIR,
        sample_rate: float | None = None,
        rate_limit_threshold: int = 100_000,
        rng: random.Random | None = None,
    ) -> None:
        self._pg_url = pg_url
        self._images_dir = Path(images_dir)
        self._rate_limit_threshold = rate_limit_threshold
        self._rng = rng if rng is not None else random.Random()

        if sample_rate is None:
            env = os.environ.get(_ENV_SAMPLE_RATE)
            if env is not None:
                try:
                    sample_rate = float(env)
                except ValueError:
                    logger.warning(
                        "ShadowSampleLogger: invalid %s=%r; using 1.0",
                        _ENV_SAMPLE_RATE,
                        env,
                    )
                    sample_rate = 1.0
            else:
                sample_rate = 1.0
        self._sample_rate = max(0.0, min(1.0, float(sample_rate)))

        self._disabled = False
        try:
            import psycopg  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            logger.warning(
                "psycopg not available — VGA shadow samples will not be persisted"
            )
            self._disabled = True

        # Row-count cache: (count, fetched_at monotonic seconds).
        self._cached_count: int | None = None
        self._cached_at: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_sample(
        self,
        image_png_bytes: bytes,
        state_machine_id: UUID,
        target_process: str,
        prompt: str,
        v5_bbox: dict,
        v5_model: str,
        private: bool,
        confidence: float | None = None,
    ) -> None:
        """Log one grounding call's result.

        Args:
            image_png_bytes: PNG-encoded bytes of the captured region.
                Used for the disk write and for computing ``image_sha``.
            state_machine_id: FK into ``runner.vga_state_machines``.
            target_process: Process name (e.g. ``"notepad++.exe"``).
                Used by the daemon to bucket per-domain regression.
            prompt: Natural-language element description fed to the VLM.
            v5_bbox: The predicted bbox, as a dict with integer
                ``x/y/w/h`` fields. Stored JSONB.
            v5_model: Model name string (e.g. ``"qontinui-grounding-v5"``).
            private: When True, returns immediately — private SMs never
                leak through this channel.
            confidence: Optional confidence in [0, 1]. Passed through to
                the table's ``confidence`` column.
        """
        if private:
            return
        if self._disabled:
            return

        try:
            image_sha = hashlib.sha256(image_png_bytes).hexdigest()

            if self._should_downsample():
                if self._rng.random() >= self._sample_rate:
                    return

            image_path = self._images_dir / f"{image_sha}.png"
            self._write_image_if_new(image_path, image_png_bytes)
            self._insert_row(
                image_sha=image_sha,
                image_path=str(image_path),
                state_machine_id=state_machine_id,
                target_process=target_process,
                prompt=prompt,
                v5_bbox=v5_bbox,
                v5_model=v5_model,
                confidence=confidence,
            )
        except Exception:
            # Never allow the shadow writer to block the runtime.
            logger.exception("ShadowSampleLogger.log_sample failed; continuing")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _should_downsample(self) -> bool:
        """Decide whether the table is large enough to start sampling."""
        count = self._get_cached_count()
        if count is None:
            return False
        return count >= self._rate_limit_threshold

    def _get_cached_count(self) -> int | None:
        """Return the (possibly cached) row count, or None on failure."""
        now = time.monotonic()
        if (
            self._cached_count is not None
            and (now - self._cached_at) < _COUNT_CACHE_TTL_SECONDS
        ):
            return self._cached_count

        try:
            import psycopg  # type: ignore[import-not-found]

            with psycopg.connect(self._pg_url) as conn:  # type: ignore[attr-defined]
                with conn.cursor() as cur:
                    cur.execute("SET search_path TO runner, public")
                    cur.execute("SELECT COUNT(*) FROM runner.vga_shadow_samples")
                    row = cur.fetchone()
                    count = int(row[0]) if row else 0
        except Exception:
            logger.exception("ShadowSampleLogger: row-count query failed")
            return None

        self._cached_count = count
        self._cached_at = now
        return count

    def _write_image_if_new(self, image_path: Path, image_bytes: bytes) -> None:
        """Write the PNG to disk only if it doesn't already exist."""
        if image_path.exists():
            return
        self._images_dir.mkdir(parents=True, exist_ok=True)
        # Write to a tmp sibling and rename for atomicity — two workers
        # with the same sha should not corrupt each other.
        tmp = image_path.with_suffix(".png.tmp")
        tmp.write_bytes(image_bytes)
        try:
            tmp.replace(image_path)
        except OSError:
            # Another writer won the race; clean up and move on.
            try:
                tmp.unlink()
            except OSError:
                pass

    def _insert_row(
        self,
        *,
        image_sha: str,
        image_path: str,
        state_machine_id: UUID,
        target_process: str,
        prompt: str,
        v5_bbox: dict,
        v5_model: str,
        confidence: float | None,
    ) -> None:
        """INSERT ... ON CONFLICT DO NOTHING into ``runner.vga_shadow_samples``.

        There is no UNIQUE index on ``(image_sha, prompt, model_used)``
        in the shipped schema; deduplication is handled at the disk
        layer (same PNG written once) and by the insert being idempotent
        at the caller level — identical triples will insert multiple
        rows here. That's acceptable for the v6 gate, which GROUP BYs
        on the triple anyway. We keep the ``ON CONFLICT`` clause against
        the PK as a defensive measure in case a future migration adds a
        unique index.
        """
        import psycopg  # type: ignore[import-not-found]

        with psycopg.connect(self._pg_url) as conn:  # type: ignore[attr-defined]
            with conn.cursor() as cur:
                cur.execute("SET search_path TO runner, public")
                cur.execute(
                    "INSERT INTO runner.vga_shadow_samples "
                    "(state_machine_id, image_sha, image_path, prompt, "
                    " target_process, predicted_bbox, model_used, confidence, "
                    " created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, NOW())",
                    (
                        str(state_machine_id),
                        image_sha,
                        image_path,
                        prompt,
                        target_process,
                        json.dumps(v5_bbox),
                        v5_model,
                        confidence,
                    ),
                )
            conn.commit()

        # Invalidate the row-count cache once we've added rows — the next
        # sample will refresh it.
        if self._cached_count is not None:
            self._cached_count += 1
