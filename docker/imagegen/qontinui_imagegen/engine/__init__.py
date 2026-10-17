"""Pipeline executors behind the facade.

:class:`Engine` is the seam: the facade (admission, the VRAM check, weight verification,
provenance, the marker, the wire) never imports torch, so its tests run on CPU with no
weights against a fake executor. :mod:`qontinui_imagegen.engine.diffusers_engine` is the
real one; it imports torch and diffusers lazily.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from qontinui_imagegen.jobs import JobPlan


class JobInterrupted(Exception):
    """The job stopped before it finished (its GPU-seconds cap, or its client went away)."""


class EngineOutOfMemory(Exception):
    """The card ran out of memory while loading or running the job."""


@dataclass(frozen=True)
class Rendered:
    """One finished candidate: a raster image (a PIL image) or an SVG document."""

    seed: int | None
    width: int
    height: int
    image: Any = None
    svg: str | None = None


StepCallback = Callable[[], None]
StopCheck = Callable[[], bool]


class Engine(Protocol):
    name: str

    def resident_key(self) -> str | None:
        """The plan.resident_key of what is loaded now, or None when nothing is."""

    def release(self) -> None:
        """Free everything this engine holds on the card."""

    def free_vram_mib(self) -> int | None:
        """Device-wide free VRAM in MiB, or None when it cannot be read."""

    def versions(self) -> Mapping[str, str]:
        """Engine and library versions for the provenance."""

    def run(self, plan: JobPlan, on_step: StepCallback, should_stop: StopCheck) -> list[Rendered]:
        """Load what ``plan`` needs (if not resident) and render its ``n`` candidates.

        ``on_step`` is called once per denoising step and once per post step;
        ``should_stop`` is polled between steps, and a True answer raises
        :class:`JobInterrupted`.
        """
