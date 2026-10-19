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
  prompt text, with or without classifier-free guidance;
* the ``model_cpu_offload`` placement (Qwen) never holds the text encoder and the
  transformer on the card together, and finishes every host-to-card move before
  ``on_ready`` (review round 3b).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from facade_fakes import FakeVerifier, make_facade, real_catalogue
from PIL import Image

from qontinui_imagegen.admission import Ticket
from qontinui_imagegen.engine import HostMemory
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
    #: One host<->card move of a module under model_cpu_offload.
    move_s: float = 20.0
    #: Moments the encoder and the transformer were on the card together (must stay empty).
    co_resident: list[str] = dataclasses.field(default_factory=list)

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


class Module:
    """A pipeline component that moves between the host and the card."""

    def __init__(self, rt: Runtime, pipeline: OffloadPipeline, name: str) -> None:
        self.rt, self.pipeline, self.name, self.device = rt, pipeline, name, "cpu"

    def to(self, device: str) -> Module:
        if device != self.device:
            self.rt.events.append(f"{self.name}->{device}")
            self.rt.advance(self.rt.move_s)
            self.device = device
            on_card = {m.name for m in self.pipeline.modules() if m.device != "cpu"}
            if {"text_encoder", "transformer"} <= on_card:
                self.rt.co_resident.append(f"after moving {self.name} to {device}")
        return self


class OffloadPipeline(ResidentPipeline):
    """A Qwen pipeline as _load leaves it under model_cpu_offload: every module on the host."""

    def __init__(self, rt: Runtime) -> None:
        super().__init__(rt)
        self.text_encoder = Module(rt, self, "text_encoder")
        self.transformer = Module(rt, self, "transformer")
        self.vae = Module(rt, self, "vae")

    def modules(self) -> list[Module]:
        return [self.text_encoder, self.transformer, self.vae]

    def encode_prompt(self, **kwargs: Any) -> tuple[str, str]:
        assert self.text_encoder.device == "cuda" and self.transformer.device == "cpu"
        self.rt.events.append("encode")
        self.rt.encode_calls.append(kwargs)
        return "EMBEDS", "MASK"

    def __call__(self, **call: Any) -> Any:
        assert self.transformer.device == "cuda" and self.vae.device == "cuda"
        assert self.text_encoder.device == "cpu"
        return super().__call__(**call)


def _facade(rt: Runtime) -> Any:
    """A facade over the real DiffusersEngine, its cold load stubbed (no weights to read)."""
    facade = make_facade(verifier=FakeVerifier(), monotonic=lambda: rt.clock["now"])
    engine = DiffusersEngine(facade.catalogue, facade.config)

    def load(plan: Any) -> _Loaded:  # what _load builds, minus the weights
        rt.events.append("pipeline_load")
        placement = facade.config.models[plan.generator_model].placement
        pipeline = OffloadPipeline(rt) if placement == "model_cpu_offload" else ResidentPipeline(rt)
        return _Loaded(plan.resident_key, pipeline, Path("/weights/m"), placement)

    engine._load = load  # type: ignore[method-assign]
    engine.host_memory = lambda: HostMemory(60_000, 64_000, 2_000)  # type: ignore[method-assign]
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


def test_qwen_offload_never_shares_the_card_and_moves_before_the_cap(runtime: Runtime) -> None:
    """Qwen-Image under model_cpu_offload: encoder on, encode, encoder off, denoiser on, then
    the cap starts. Four 20 s moves plus 100 s of denoising pass the 120 s cap only because
    every move precedes ``on_ready``; and the encoder (15.4 GiB) and the FP8 transformer
    (19.0 GiB) are never on the card together, cold or warm."""
    facade = _facade(runtime)
    plan = build_plan(
        facade.catalogue, "generations", {"model": "qwen-image-2512", "prompt": "a fox", "n": 1}
    )
    assert facade.config.models["qwen-image-2512"].placement == "model_cpu_offload"
    for warm in (False, True):
        runtime.events.clear()
        assert len(_run(facade, plan, runtime)) == 1
        events = runtime.events
        expected = [
            *(["pipeline_load"] if not warm else ["transformer->cpu", "vae->cpu"]),
            "text_encoder->cuda",
            "encode",
            "encode",  # the recipe's negative prompt: two encodes, one job, one stage
            "text_encoder->cpu",
            "transformer->cuda",
            "vae->cuda",
            "ready",
            "denoise",
        ]
        assert [e for e in events if e != "empty_cache"] == expected
    assert runtime.co_resident == []


def test_a_qwen_offload_job_shares_one_encode_across_its_candidates(runtime: Runtime) -> None:
    runtime.denoise_s = 10.0
    facade = _facade(runtime)
    plan = build_plan(
        facade.catalogue, "generations", {"model": "qwen-image-2512", "prompt": "a fox", "n": 3}
    )
    _run(facade, plan, runtime)
    assert runtime.events.count("text_encoder->cuda") == 1
    assert runtime.events.count("denoise") == 3
    for call in runtime.denoise_calls:
        assert call["prompt_embeds"] == "EMBEDS" and call["negative_prompt_embeds"] == "EMBEDS"
        assert "prompt" not in call


def test_the_real_offload_load_puts_nothing_on_the_card(
    runtime: Runtime, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R4-6: DiffusersEngine._load itself (not a stub) under model_cpu_offload builds the FP8
    transformer on the host and leaves every module there; a job stages them later."""
    import qontinui_imagegen.engine.diffusers_engine as engine_module
    from qontinui_imagegen.config import WEIGHTS_ROOT_ENV, load_config
    from qontinui_imagegen.manifest import MANIFEST_PATH, RECIPES_DIR, load_recipes, load_yaml
    from qontinui_imagegen.weights import component_dir

    manifest = load_yaml(MANIFEST_PATH)
    recipes = load_recipes(RECIPES_DIR)
    models = {str(c["id"]) for c in manifest["components"] if c.get("kind") == "model"}
    config = load_config(
        environ={WEIGHTS_ROOT_ENV: str(tmp_path)}, known_models=models, known_recipes=recipes
    )
    catalogue = real_catalogue(config)
    model_dir = component_dir(tmp_path, catalogue.components["qwen-image-2512"])
    model_dir.mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({"transformer": ["diffusers", "QwenImageTransformer2DModel"]})
    )
    built: dict[str, Any] = {}

    def fake_fp8(cls: Any, folder: Path, device: str, pinned: set[Path]) -> Any:
        built["transformer_target"] = device
        return "FP8-TRANSFORMER"

    class QwenImagePipeline(OffloadPipeline):
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> QwenImagePipeline:
            built["kwargs"] = kwargs
            return cls(runtime)

        def to(self, device: str) -> QwenImagePipeline:  # the whole pipeline at once
            runtime.events.append(f"pipeline->{device}")
            for module in self.modules():
                module.to(device)
            return self

        def set_progress_bar_config(self, **kwargs: Any) -> None:
            pass

    diffusers = sys.modules["diffusers"]
    diffusers.QwenImagePipeline = QwenImagePipeline  # type: ignore[attr-defined]
    diffusers.QwenImageTransformer2DModel = object  # type: ignore[attr-defined]
    monkeypatch.setattr(engine_module, "load_fp8_transformer", fake_fp8)
    engine = DiffusersEngine(catalogue, config)
    plan = build_plan(catalogue, "generations", {"model": "qwen-image-2512", "prompt": "a"})

    loaded = engine._load(plan)

    assert loaded.placement == "model_cpu_offload"
    assert built["transformer_target"] == "cpu"  # the FP8 transformer is built on the host
    assert built["kwargs"]["transformer"] == "FP8-TRANSFORMER"
    assert "text_encoder" not in built["kwargs"]  # offload keeps its encoder in the pipeline
    assert not any(event.endswith("->cuda") for event in runtime.events), runtime.events
    assert {m.device for m in loaded.pipeline.modules()} == {"cpu"}
