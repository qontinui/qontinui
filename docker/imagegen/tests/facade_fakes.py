"""A fake pipeline executor and request helpers for the facade tests (CPU, no weights)."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx
from PIL import Image

from qontinui_imagegen.admission import AdmissionController
from qontinui_imagegen.app import Facade, create_app
from qontinui_imagegen.config import FacadeConfig, load_config
from qontinui_imagegen.engine import EngineOutOfMemory, JobInterrupted, Rendered
from qontinui_imagegen.jobs import Catalogue, JobPlan
from qontinui_imagegen.manifest import MANIFEST_PATH, RECIPES_DIR, load_recipes, load_yaml

#: What the loopback override sets (rule R-bind); without it the facade answers `disabled`.
ATTESTED = {"QONTINUI_IMAGEGEN_LISTENER": "loopback"}
DAYTIME = datetime(2026, 10, 17, 14, 0)
NIGHT = datetime(2026, 10, 17, 2, 0)
FAKE_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"><rect width="8" height="8"/></svg>'


@dataclass
class FakeEngine:
    """Renders a solid colour per seed. ``gate`` holds a job open until it is set."""

    name: str = "fake"
    free_mib: int | None = 32_000
    gate: threading.Event | None = None
    started: threading.Event = field(default_factory=threading.Event)
    out_of_memory: bool = False
    #: Raised from run() as an unexpected engine failure (not a typed one).
    crash: Exception | None = None
    #: Called in run() BEFORE on_ready -- a cold load, e.g. advancing a fake clock.
    load: Callable[[], None] | None = None
    runs: list[JobPlan] = field(default_factory=list)
    releases: int = 0
    _resident: str | None = None

    def resident_key(self) -> str | None:
        return self._resident

    def release(self) -> None:
        self.releases += 1
        self._resident = None

    def free_vram_mib(self) -> int | None:
        return self.free_mib

    def versions(self) -> Mapping[str, str]:
        return {"engine": self.name, "diffusers": "0.41.0", "torch": "fake"}

    def run(
        self,
        plan: JobPlan,
        on_step: Callable[[], None],
        should_stop: Callable[[], bool],
        on_ready: Callable[[], None],
    ) -> list[Rendered]:
        self.runs.append(plan)
        self._resident = plan.resident_key
        if self.load is not None:
            self.load()
        on_ready()
        self.started.set()
        if self.crash is not None:
            raise self.crash
        if self.out_of_memory:
            raise EngineOutOfMemory("CUDA out of memory (fake)")
        outputs = []
        for seed in plan.seeds:
            for _ in range(plan.steps_per_image):
                if self.gate is not None:
                    while not self.gate.wait(0.01):
                        if should_stop():
                            raise JobInterrupted("stopped while gated")
                if should_stop():
                    raise JobInterrupted("stopped")
                on_step()
            if plan.output_format == "svg":
                outputs.append(Rendered(seed=seed, width=8, height=8, svg=FAKE_SVG))
                continue
            width, height = plan.size or (plan.inputs[0].width * 2, plan.inputs[0].height * 2)
            colour = ((seed or 0) % 256, 64, 128)
            outputs.append(
                Rendered(
                    seed=seed, width=width, height=height, image=Image.new("RGB", (8, 8), colour)
                )
            )
        return outputs


class FakeVerifier:
    """A verifier whose answer per model the test sets."""

    def __init__(self, failing: Mapping[str, list[str]] | None = None) -> None:
        from qontinui_imagegen.weights import VerifyProgress

        self.failing = dict(failing or {})
        self.verified: set[str] = set()
        self.progress = VerifyProgress()

    def verify_model(self, model_id: str, *, full: bool = False) -> None:
        from qontinui_imagegen.weights import WeightsError

        if model_id in self.failing:
            raise WeightsError(model_id, self.failing[model_id])
        self.verified.add(model_id)


def real_catalogue(config: FacadeConfig | None = None) -> Catalogue:
    manifest = load_yaml(MANIFEST_PATH)
    recipes = load_recipes(RECIPES_DIR)
    models = {str(c["id"]) for c in manifest["components"] if c.get("kind") == "model"}
    config = config or load_config(environ={}, known_models=models, known_recipes=recipes)
    return Catalogue(config, manifest, recipes)


def make_facade(
    *,
    engine: FakeEngine | None = None,
    verifier: Any = None,
    now: datetime = DAYTIME,
    environ: Mapping[str, str] | None = None,
    monotonic: Callable[[], float] | None = None,
    catalogue: Catalogue | None = None,
    attested: bool = True,
) -> Facade:
    catalogue = catalogue or real_catalogue()
    kwargs: dict[str, Any] = {}
    if monotonic is not None:
        kwargs["monotonic"] = monotonic
    admission = AdmissionController(catalogue.config.admission, now=lambda: now, **kwargs)
    return Facade(
        catalogue,
        engine or FakeEngine(),
        verifier or FakeVerifier(),
        admission,
        environ={**ATTESTED, **(environ or {})} if attested else dict(environ or {}),
        random_seed=lambda: 1234,
        **kwargs,
    )


def client(facade: Facade) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(facade)), base_url="http://facade"
    )


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def sse_events(text: str) -> list[tuple[str, dict[str, Any]]]:
    import json

    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


async def wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)
