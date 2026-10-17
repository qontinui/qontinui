"""The real executor: diffusers pipelines plus the named post runners, on one CUDA card.

Imported only by :mod:`qontinui_imagegen.serve`; torch, diffusers and transformers load
lazily so the facade and its tests never need them.

Loading follows Phase 0 item 5's recorded path (plan RESULT block, 2026-10-15) when a model's
``transformer_storage`` is ``fp8``:

1. the transformer is built on the meta device (``accelerate.init_empty_weights``);
2. each tensor is read from the pinned safetensors and placed on the card already cast --
   ``float8_e4m3fn`` for the weights of layer types diffusers' layerwise casting hooks
   (Linear, Conv, Embedding), bf16 for everything else (raw parameters such as pad tokens,
   and the modules the model's ``_skip_layerwise_casting_patterns`` and
   ``_keep_in_fp32_modules`` name, plus diffusers' default skip patterns) -- passing
   ``dtype=`` to ``set_module_tensor_to_device``, without which accelerate casts back to the
   meta tensor's fp32 (Phase 0 finding 2);
3. ``apply_layerwise_casting`` registers the bf16 compute hooks.

So the bf16 transformer is never materialised -- on the card or in host memory -- which is
what makes a 20B Qwen transformer fit the 32 GB card and the 24 GB WSL VM at all.
``QuantoConfig(float8)`` is deliberately not used: optimum-quanto's marlin-fp8 path
JIT-compiles a CUDA extension on sm_120 and hangs (Phase 0 finding 1).

``placement`` (facade.yaml) chooses between resident text encoders, Phase 0's sequential
text encoder (loaded alone per job, encodes, freed), and diffusers' model CPU offload.
"""

from __future__ import annotations

import gc
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from qontinui_imagegen.config import FacadeConfig
from qontinui_imagegen.engine import EngineOutOfMemory, JobInterrupted, Rendered
from qontinui_imagegen.engine.post import PostRunners
from qontinui_imagegen.jobs import Catalogue, JobPlan
from qontinui_imagegen.weights import component_dir

#: Pipeline components that encode the prompt: everything the sequential placement frees.
TEXT_COMPONENTS = ("text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "processor")


def _skip_patterns(model: Any) -> tuple[str, ...]:
    from diffusers.hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN

    patterns = list(DEFAULT_SKIP_MODULES_PATTERN)
    patterns += list(getattr(model, "_keep_in_fp32_modules", None) or ())
    patterns += list(getattr(model, "_skip_layerwise_casting_patterns", None) or ())
    return tuple(sorted(set(patterns)))


def fp8_parameter_names(model: Any, patterns: Sequence[str]) -> set[str]:
    """Names of the parameters diffusers' layerwise casting will store in FP8.

    Mirrors ``diffusers.hooks.layerwise_casting._apply_layerwise_casting``: a module whose
    dotted name matches a skip pattern is skipped with everything under it; a module of a
    supported layer type is hooked (its own parameters only); anything else is descended.
    """
    from diffusers.hooks._common import _GO_LC_SUPPORTED_PYTORCH_LAYERS

    names: set[str] = set()

    def walk(module: Any, prefix: str) -> None:
        if prefix and any(re.search(pattern, prefix) for pattern in patterns):
            return
        if isinstance(module, _GO_LC_SUPPORTED_PYTORCH_LAYERS):
            for name, _ in module.named_parameters(recurse=False):
                names.add(f"{prefix}.{name}" if prefix else name)
            return
        for child_name, child in module.named_children():
            walk(child, f"{prefix}.{child_name}" if prefix else child_name)

    walk(model, "")
    return names


def _shards(folder: Path, pinned: set[Path]) -> list[Path]:
    """The transformer's weight files: exactly the manifest's, never a stray file beside them.

    ``pinned`` holds the manifest-listed files of the component. A ``.safetensors`` file in
    ``folder`` that is not pinned is refused rather than loaded or ignored: an unlisted file
    is unverified bytes, and a glob would pick it up.
    """
    present = set(folder.glob("*.safetensors"))
    unlisted = sorted(present - pinned)
    if unlisted:
        raise RuntimeError(f"unpinned weight files beside the manifest's: {unlisted}")
    index = sorted(folder.glob("*.safetensors.index.json"))
    if index:
        weight_map = json.loads(index[0].read_text(encoding="utf-8"))["weight_map"]
        shards = [folder / name for name in sorted(set(weight_map.values()))]
    else:
        shards = sorted(present)
    missing = [shard for shard in shards if shard not in pinned]
    if missing or not shards:
        raise RuntimeError(f"weight files the manifest does not pin: {missing or 'none found'}")
    return shards


def load_fp8_transformer(transformer_cls: Any, folder: Path, device: str, pinned: set[Path]) -> Any:
    """Build ``transformer_cls`` from ``folder`` with FP8 storage, tensor by tensor."""
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers.hooks import apply_layerwise_casting
    from safetensors import safe_open

    config = transformer_cls.load_config(str(folder), local_files_only=True)
    with init_empty_weights():
        model = transformer_cls.from_config(config)
    patterns = _skip_patterns(model)
    fp8 = fp8_parameter_names(model, patterns)
    expected = {name for name, _ in model.named_parameters()}
    loaded: set[str] = set()
    for shard in _shards(folder, pinned):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                dtype = torch.float8_e4m3fn if key in fp8 else torch.bfloat16
                tensor = handle.get_tensor(key)
                set_module_tensor_to_device(model, key, device, value=tensor, dtype=dtype)
                loaded.add(key)
                del tensor
    missing = expected - loaded
    if missing:
        raise RuntimeError(
            f"{folder}: {len(missing)} parameters have no tensor, e.g. {sorted(missing)[:3]}"
        )
    model.to(device)  # buffers built at __init__ (rotary tables and the like)
    model.eval()
    apply_layerwise_casting(
        model,
        storage_dtype=torch.float8_e4m3fn,
        compute_dtype=torch.bfloat16,
        skip_modules_pattern=patterns,
    )
    return model


class _Loaded:
    """What is resident for one plan: the pipeline and the post runners' models."""

    def __init__(self, key: str, pipeline: Any, model_dir: Path, placement: str | None) -> None:
        self.key = key
        self.pipeline = pipeline
        self.model_dir = model_dir
        self.placement = placement


class DiffusersEngine:
    name = "diffusers"

    def __init__(self, catalogue: Catalogue, config: FacadeConfig) -> None:
        self.catalogue = catalogue
        self.config = config
        self.device = config.device
        self._loaded: _Loaded | None = None
        self.post = PostRunners(catalogue, config)
        vendor = Path(__file__).resolve().parents[2] / "vendor"
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))

    # --- Engine protocol --------------------------------------------------------------------

    def resident_key(self) -> str | None:
        return None if self._loaded is None else self._loaded.key

    def release(self) -> None:
        self._loaded = None
        self.post.release()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def free_vram_mib(self) -> int | None:
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            free, _total = torch.cuda.mem_get_info(torch.device(self.device))
            return int(free // (1024 * 1024))
        except (ImportError, RuntimeError):
            return None

    def versions(self) -> Mapping[str, str]:
        import accelerate
        import diffusers
        import torch
        import transformers

        return {
            "engine": self.name,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "torch": torch.__version__,
            "cuda": str(torch.version.cuda),
        }

    def run(
        self,
        plan: JobPlan,
        on_step: Callable[[], None],
        should_stop: Callable[[], bool],
        on_ready: Callable[[], None],
    ) -> list[Rendered]:
        import torch

        try:
            if self.resident_key() != plan.resident_key:
                self.release()
                if plan.pipeline_kind == "diffusers":
                    self._loaded = self._load(plan)
                else:
                    self._loaded = _Loaded(plan.resident_key, None, Path(), None)
            self.post.prepare(plan)  # post-step models load before the cap's clock starts
            on_ready()
            if plan.pipeline_kind == "diffusers":
                prompt_kwargs = self._prompt_kwargs(plan)  # ONE text-encoder pass per job
                images = [
                    (seed, self._generate(plan, seed, prompt_kwargs, on_step, should_stop))
                    for seed in plan.seeds
                ]
            else:
                import io

                from PIL import Image

                if len(plan.seeds) != 1:
                    raise ValueError("a deterministic named recipe renders exactly one image")
                source = Image.open(io.BytesIO(plan.inputs[0].data))
                source.load()
                images = [(None, self.post.upscale(plan.alias, source, plan.call_parameters))]
                on_step()
            rendered = []
            for seed, image in images:
                result: Any = image
                for step in plan.post:
                    if should_stop():
                        raise JobInterrupted(f"stopped before post step {step.op}")
                    result = self.post.apply(step, result)
                    on_step()
                if isinstance(result, str):
                    width, height = (image.width, image.height)
                    rendered.append(Rendered(seed=seed, width=width, height=height, svg=result))
                else:
                    rendered.append(
                        Rendered(seed=seed, width=result.width, height=result.height, image=result)
                    )
            return rendered
        except torch.cuda.OutOfMemoryError as error:
            raise EngineOutOfMemory(str(error)) from error

    # --- loading ----------------------------------------------------------------------------

    def _load(self, plan: JobPlan) -> _Loaded:
        import diffusers
        import torch

        component = self.catalogue.components[plan.generator_model]
        model_dir = component_dir(self.config.weights_root, component)
        load = self.config.models[plan.generator_model]
        pipeline_cls = getattr(diffusers, plan.pipeline)
        kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16, "local_files_only": True}
        if load.transformer_storage == "fp8":
            index = json.loads((model_dir / "model_index.json").read_text(encoding="utf-8"))
            library, class_name = index["transformer"]
            transformer_cls = getattr(diffusers, class_name)
            target = "cpu" if load.placement == "model_cpu_offload" else self.device
            kwargs["transformer"] = load_fp8_transformer(
                transformer_cls,
                model_dir / "transformer",
                target,
                {model_dir / str(entry["path"]) for entry in component["files"]},
            )
        if load.placement == "sequential_text_encoder":
            for name in ("text_encoder", "text_encoder_2"):
                if (model_dir / name).is_dir():
                    kwargs[name] = None
        pipeline = pipeline_cls.from_pretrained(str(model_dir), **kwargs)
        if load.placement == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=self.device)
        else:
            pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        return _Loaded(plan.resident_key, pipeline, model_dir, load.placement)

    # --- generation -------------------------------------------------------------------------

    def _encode_sequentially(self, plan: JobPlan) -> dict[str, Any]:
        """Phase 0's text-encoder stage: load the encoder alone in bf16, encode, free it."""
        import diffusers
        import torch

        assert self._loaded is not None
        pipeline_cls = getattr(diffusers, plan.pipeline)
        encoder = pipeline_cls.from_pretrained(
            str(self._loaded.model_dir),
            transformer=None,
            vae=None,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to(self.device)
        try:
            with torch.no_grad():
                return encode_prompt_kwargs(encoder, plan, self.device)
        finally:
            del encoder
            gc.collect()
            torch.cuda.empty_cache()

    def _prompt_kwargs(self, plan: JobPlan) -> dict[str, Any]:
        """The prompt arguments every candidate of the job shares, computed once.

        Under the sequential placement this is the one text-encoder load + encode of the job
        (Phase 0: about 52-55 s for Z-Image's encoder); every seed reuses the embeddings.
        """
        assert self._loaded is not None
        if self._loaded.placement == "sequential_text_encoder":
            return self._encode_sequentially(plan)
        kwargs: dict[str, Any] = {"prompt": plan.rendered_prompt}
        if plan.negative_prompt is not None:
            kwargs["negative_prompt"] = plan.negative_prompt
        return kwargs

    def _generate(
        self,
        plan: JobPlan,
        seed: int | None,
        prompt_kwargs: dict[str, Any],
        on_step: Callable[[], None],
        should_stop: Callable[[], bool],
    ) -> Any:
        import torch

        assert self._loaded is not None and seed is not None
        pipeline = self._loaded.pipeline
        call: dict[str, Any] = dict(plan.call_parameters)
        call.update(prompt_kwargs)
        if plan.size is not None:
            call["width"], call["height"] = plan.size
        if plan.inputs:
            call["image"] = _pil_inputs(plan)
        call["generator"] = torch.Generator(device="cpu").manual_seed(seed)
        call["output_type"] = "pil"

        def step_end(
            pipe: Any, step: int, timestep: Any, callback_kwargs: dict[str, Any]
        ) -> dict[str, Any]:
            on_step()
            if should_stop():
                pipe._interrupt = True
            return callback_kwargs

        call["callback_on_step_end"] = step_end
        with torch.no_grad():
            result = pipeline(**call)
        if should_stop():
            raise JobInterrupted("stopped during denoising")
        return result.images[0]


def _pil_inputs(plan: JobPlan) -> list[Any]:
    import io

    from PIL import Image

    images = []
    for item in plan.inputs:
        image = Image.open(io.BytesIO(item.data))
        image.load()
        images.append(image.convert("RGB"))
    return images


def encode_prompt_kwargs(pipeline: Any, plan: JobPlan, device: str) -> dict[str, Any]:
    """The pipeline-call keyword arguments carrying precomputed prompt embeddings.

    One branch per pipeline the manifest declares, each matching that pipeline's
    ``encode_prompt`` and ``__call__`` signatures in diffusers 0.41.0.
    """
    prompt = plan.rendered_prompt
    negative = plan.negative_prompt
    max_len = plan.call_parameters.get("max_sequence_length")
    name = plan.pipeline
    if name == "QwenImagePipeline":
        embeds, mask = pipeline.encode_prompt(
            prompt=prompt, device=device, max_sequence_length=max_len
        )
        out = {"prompt_embeds": embeds, "prompt_embeds_mask": mask}
        if negative is not None:
            neg, neg_mask = pipeline.encode_prompt(
                prompt=negative, device=device, max_sequence_length=max_len
            )
            out.update(negative_prompt_embeds=neg, negative_prompt_embeds_mask=neg_mask)
        return out
    if name == "QwenImageEditPlusPipeline":
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
            CONDITION_IMAGE_SIZE,
            calculate_dimensions,
        )

        conditions = []
        for image in _pil_inputs(plan):
            width, height = calculate_dimensions(CONDITION_IMAGE_SIZE, image.width / image.height)
            conditions.append(pipeline.image_processor.resize(image, height, width))
        embeds, mask = pipeline.encode_prompt(
            prompt=prompt, image=conditions, device=device, max_sequence_length=max_len
        )
        out = {"prompt_embeds": embeds, "prompt_embeds_mask": mask}
        if negative is not None:
            neg, neg_mask = pipeline.encode_prompt(
                prompt=negative, image=conditions, device=device, max_sequence_length=max_len
            )
            out.update(negative_prompt_embeds=neg, negative_prompt_embeds_mask=neg_mask)
        return out
    if name == "ZImagePipeline":
        guidance = float(plan.call_parameters.get("guidance_scale", 0.0))
        embeds, neg = pipeline.encode_prompt(
            prompt=prompt,
            device=device,
            do_classifier_free_guidance=guidance > 0,
            negative_prompt=negative,
            max_sequence_length=max_len,
        )
        out = {"prompt_embeds": embeds}
        if guidance > 0:
            out["negative_prompt_embeds"] = neg
        return out
    if name == "FluxPipeline":
        embeds, pooled, _text_ids = pipeline.encode_prompt(
            prompt=prompt, prompt_2=None, device=device, max_sequence_length=max_len
        )
        return {"prompt_embeds": embeds, "pooled_prompt_embeds": pooled}
    raise ValueError(f"no sequential text-encoder stage for {name!r}")
