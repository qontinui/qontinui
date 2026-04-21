"""Append-only JSONL correction log.

Every user correction (from the builder UI or at runtime) is appended
here. The v6 trainer consumes this log through an exporter (milestone
c.1) that converts each entry to vlm_sft format.

Privacy (plan §13 recommendation D):
- Every entry defaults to ``private=True``.
- When ``private=True``, a ``.private`` sidecar file is written next to
  the image referenced by the entry. The future exporter must skip any
  entry whose sidecar exists unless explicitly told to include it.
- Nothing uploads anywhere from this module. Centralized training
  storage is a future concern; the sidecar guarantees the exporter has
  the information it needs when that concern arrives.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Generator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from .state_machine import BBox

logger = logging.getLogger(__name__)

_DEFAULT_CORRECTIONS_DIR = Path("datasets/vga-corrections")
_JSONL_NAME = "corrections.jsonl"
_ENV_OVERRIDE = "QONTINUI_VGA_CORRECTIONS_DIR"


@dataclass(frozen=True)
class CorrectionStats:
    """Summary returned by :meth:`CorrectionLogger.stats`."""

    total: int
    per_target_process: dict[str, int]
    since_v5: int


class CorrectionLogger:
    """Append-only JSONL store for VGA corrections.

    Args:
        corrections_dir: Directory to write ``corrections.jsonl`` into.
            Falls back to ``$QONTINUI_VGA_CORRECTIONS_DIR`` or
            ``datasets/vga-corrections`` (relative to cwd).
    """

    def __init__(self, corrections_dir: Path | None = None) -> None:
        if corrections_dir is None:
            env_dir = os.environ.get(_ENV_OVERRIDE)
            corrections_dir = Path(env_dir) if env_dir else _DEFAULT_CORRECTIONS_DIR
        self._dir = Path(corrections_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        return self._dir

    @property
    def jsonl_path(self) -> Path:
        return self._dir / _JSONL_NAME

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def append(
        self,
        state_machine_id: UUID,
        image_sha: str,
        image_path: Path,
        prompt: str,
        corrected_bbox: BBox,
        source: Literal["builder", "runtime"],
        target_process: str,
        private: bool = True,
    ) -> None:
        """Append one correction entry as a JSONL line.

        If ``private=True``, writes a ``{image_path}.private`` sidecar
        file (empty marker) that the future exporter uses as its filter.
        """
        entry = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "state_machine_id": str(state_machine_id),
            "image_sha": image_sha,
            "image_path": str(image_path),
            "prompt": prompt,
            "corrected_bbox": {
                "x": corrected_bbox.x,
                "y": corrected_bbox.y,
                "w": corrected_bbox.w,
                "h": corrected_bbox.h,
            },
            "source": source,
            "target_process": target_process,
            "private": private,
        }

        line = json.dumps(entry, separators=(",", ":"), sort_keys=True)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")

        if private:
            try:
                sidecar = Path(f"{image_path}.private")
                # Create parent if it doesn't exist — the image may not
                # have been written yet, but the marker still needs its
                # directory.
                sidecar.parent.mkdir(parents=True, exist_ok=True)
                sidecar.touch(exist_ok=True)
            except OSError:
                logger.warning(
                    "CorrectionLogger: could not write sidecar for %s", image_path,
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def iter_entries(
        self, include_private: bool = False
    ) -> Generator[dict, None, None]:
        """Yield one parsed entry per JSONL line.

        Args:
            include_private: When False (default), entries with
                ``private=True`` or with an existing ``.private`` sidecar
                are skipped. This is the safe default for the exporter.
        """
        if not self.jsonl_path.exists():
            return

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("CorrectionLogger: skipping malformed line")
                    continue

                if not include_private:
                    if entry.get("private", True):
                        continue
                    image_path = entry.get("image_path")
                    if image_path and Path(f"{image_path}.private").exists():
                        continue

                yield entry

    def stats(self) -> CorrectionStats:
        """Summarize the correction log without loading it all at once.

        ``since_v5`` is the count of entries logged after v5 landed.
        Until we have a persistent marker for "v5 train timestamp", this
        is simply the total entry count — the trainer's exporter will
        do the cross-referencing when v6 lands.
        """
        total = 0
        per_target: dict[str, int] = {}

        if not self.jsonl_path.exists():
            return CorrectionStats(total=0, per_target_process={}, since_v5=0)

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                tp = entry.get("target_process", "unknown")
                per_target[tp] = per_target.get(tp, 0) + 1

        return CorrectionStats(
            total=total, per_target_process=per_target, since_v5=total
        )
