"""The real executor's job orchestration, against fake torch and diffusers modules (plan Phase 10).

No GPU and no weights: ``torch`` and ``diffusers`` are stand-ins installed into
``sys.modules`` for one test, recording what the engine asks of them and when. What is
pinned here is ORDER and COUNT, which is where the engine's own logic lives:

* the ``sequential_text_encoder`` placement loads its text encoder per job, and that load is
  a load -- it happens before ``on_ready``, so the interactive cap never counts it, while
  the encode itself runs after (review round 3, M1);
* the encoder is loaded and run ONCE per job however many candidates it renders, and freed
  before the denoiser runs (M6);
* ``encode_prompt_kwargs``'s ZImagePipeline branch hands the denoiser embeddings, never the
  prompt text, with or without classifier-free guidance.
"""

from __future__ import annotations

import contextlib
import dataclasses
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from facade_fakes import FakeVerifier, make_facade
from PIL import Image

from qontinui_imagegen.admission import Ticket
from qontinui_imagegen.engine.diffusers_engine import (
    DiffusersEngine,
    _Loaded,
    encode_prompt_kwargs,
)
from qontinui_imagegen.jobs import build_plan

ENCODER_LOAD_S = 55.0  # Phase 0: Z-Image's text encoder loaded in 52-55 s
#: One candidate's denoising: inside the 120 s cap on its own, past it with the load added.
DENOISE_S = 100.0


@dataclasses.dataclass
class Runtime:
    """The fake torch + diffusers, and the log of what the engine did, in order."""

    clock: dict[str, float] = dataclasses.field(default_factory=lambda: {"now": 0.0})
    events: list[str] = dataclasses.field(default_factory=list)
    encode_calls: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    denoise_calls: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    denoise_s: float = DENOISE_S

    def advance(self, seconds: float) -> None:
        self.clock["now"] += seconds


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> Runtime:
    rt = Runtime()

    class Generator:
        def __init__(self, device: str) -> None:
            self.device = device

        def manual_seed(self, seed: int) -> Generator:
            self.seed = seed
            return self

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
    torch.no_grad = contextlib.nullcontext  # type: ignore[attr-defined]
    torch.Generator = Generator  # type: ignore[attr-defined]
    torch.device = lambda name: name  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        OutOfMemoryError=type("OutOfMemoryError", (RuntimeError,), {}),
        is_available=lambda: True,
        empty_cache=lambda: rt.events.append("empty_cache"),
        mem_get_info=lambda device: (30_000 * 2**20, 32_607 * 2**20),
    )

    class ZImagePipeline:
        """The encoder-only pipeline the sequential placement loads per job."""

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> ZImagePipeline:
            assert kwargs["transformer"] is None and kwargs["vae"] is None
            assert kwargs["local_files_only"] is True
            rt.events.append("encoder_load")
            rt.advance(ENCODER_LOAD_S)
            return cls()

        def to(self, device: str) -> ZImagePipeline:
            return self

        def encode_prompt(self, **kwargs: Any) -> tuple[str, str | None]:
            rt.events.append("encode")
            rt.encode_calls.append(kwargs)
            negative = "NEG" if kwargs["do_classifier_free_guidance"] else None
            return "EMBEDS", negative

        def __del__(self) -> None:
            rt.events.append("encoder_freed")

    diffusers = types.ModuleType("diffusers")
    diffusers.ZImagePipeline = ZImagePipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    return rt


class ResidentPipeline:
    """The resident (encoder-less) pipeline: denoises, calling the step callback."""

    def __init__(self, rt: Runtime) -> None:
        self.rt = rt
        self._interrupt = False

    def __call__(self, **call: Any) -> Any:
        self.rt.events.append("denoise")
        self.rt.denoise_calls.append(call)
        steps = int(call["num_inference_steps"])
        for step in range(steps):
            self.rt.advance(self.rt.denoise_s / steps)
            call["callback_on_step_end"](self, step, None, {})
        return types.SimpleNamespace(images=[Image.new("RGB", (8, 8))])


def _facade(rt: Runtime) -> Any:
    """A facade over the real DiffusersEngine, its cold load stubbed (no weights to read)."""
    facade = make_facade(verifier=FakeVerifier(), monotonic=lambda: rt.clock["now"])
    engine = DiffusersEngine(facade.catalogue, facade.config)

    def load(plan: Any) -> _Loaded:  # what _load builds, minus the weights
        rt.events.append("pipeline_load")
        return _Loaded(
            plan.resident_key, ResidentPipeline(rt), Path("/weights/z"), "sequential_text_encoder"
        )

    engine._load = load  # type: ignore[method-assign]
    facade.engine = engine
    return facade


def _run(facade: Any, plan: Any, rt: Runtime) -> list[Any]:
    """Facade.execute on a fresh interactive ticket, logging when the cap's clock starts."""
    ticket = Ticket(
        "interactive", facade.config.admission.interactive_gpu_seconds_cap, started_at=0.0
    )
    begin_compute = ticket.begin_compute

    def logged(now: float) -> None:
        rt.events.append("ready")
        begin_compute(now)

    ticket.begin_compute = logged  # type: ignore[method-assign]
    return facade.execute(plan, ticket, lambda step, total: None)


def _plan(facade: Any, **body: Any) -> Any:
    return build_plan(
        facade.catalogue, "generations", {"model": "z-image-turbo", "prompt": "a fox", **body}
    )


def test_the_encoder_load_is_outside_the_cap_and_the_encode_inside(runtime: Runtime) -> None:
    """M1: a 55 s encoder load plus 100 s of denoising passes a 120 s cap -- the cap counts
    compute only. Were the encoder stage inside the cap, 155 s would trip it."""
    facade = _facade(runtime)
    plan = _plan(facade)
    for warm in (False, True):  # cold, then warm: the encoder loads on both
        runtime.events.clear()
        rendered = _run(facade, plan, runtime)
        assert len(rendered) == 1
        events = runtime.events
        assert events.index("encoder_load") < events.index("ready") < events.index("encode")
        assert ("pipeline_load" in events) is (not warm)
        if not warm:
            assert events.index("pipeline_load") < events.index("ready")


def test_the_encoder_runs_once_per_job_and_is_freed_before_denoising(runtime: Runtime) -> None:
    """M6: three candidates share one encoder load and one encode; the denoiser never sees
    the prompt text, only the embeddings, and runs after the encoder is gone."""
    runtime.denoise_s = 10.0  # three candidates inside one cap
    facade = _facade(runtime)
    plan = _plan(facade, n=3)
    rendered = _run(facade, plan, runtime)
    assert [r.seed for r in rendered] == list(plan.seeds)
    assert runtime.events.count("encoder_load") == 1
    assert runtime.events.count("encode") == 1
    assert runtime.events.count("denoise") == 3
    assert runtime.events.index("encoder_freed") < runtime.events.index("denoise")
    for call in runtime.denoise_calls:
        assert call["prompt_embeds"] == "EMBEDS"
        assert "prompt" not in call and "negative_prompt_embeds" not in call


def test_the_z_image_encode_under_the_sequential_placement(runtime: Runtime) -> None:
    """The ZImagePipeline branch of encode_prompt_kwargs as the engine drives it: the
    recipe's rendered prompt, its max_sequence_length, and no guidance at Turbo's 0.0."""
    facade = _facade(runtime)
    plan = _plan(facade)
    _run(facade, plan, runtime)
    (call,) = runtime.encode_calls
    assert call == {
        "prompt": plan.rendered_prompt,
        "device": facade.config.device,
        "do_classifier_free_guidance": False,
        "negative_prompt": None,
        "max_sequence_length": 512,
    }


def test_the_z_image_encode_with_guidance_carries_the_negative_embeddings(
    runtime: Runtime,
) -> None:
    plan = _plan(_facade(runtime))
    guided = dataclasses.replace(
        plan,
        call_parameters={**plan.call_parameters, "guidance_scale": 4.0},
        negative_prompt="blurry",
    )
    encoder = sys.modules["diffusers"].ZImagePipeline()  # type: ignore[attr-defined]
    out = encode_prompt_kwargs(encoder, guided, "cuda")
    assert out == {"prompt_embeds": "EMBEDS", "negative_prompt_embeds": "NEG"}
    assert runtime.encode_calls[-1]["do_classifier_free_guidance"] is True
    assert runtime.encode_calls[-1]["negative_prompt"] == "blurry"
