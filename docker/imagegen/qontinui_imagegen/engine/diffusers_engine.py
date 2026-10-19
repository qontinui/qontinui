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

``placement`` (facade.yaml) chooses how the text encoder and the transformer share the card:

* ``resident`` -- everything on the card for the process's life;
* ``sequential_text_encoder`` (Phase 0's path) -- the pipeline is resident without its text
  encoder, which is loaded per job, encodes once for every candidate, and is freed before
  the denoiser runs. Its per-job LOAD happens before ``on_ready``, so the interactive cap
  never counts it; the encode runs after, inside the cap;
* ``model_cpu_offload`` -- every component waits in host RAM. Per job the text encoder is
  moved onto the card, encodes, and is moved off; then the transformer and VAE are moved
  on. The encoder and the transformer are never on the card together, which is what lets
  Qwen-Image (15.4 GiB encoder + 19.0 GiB FP8 transformer) run on a 32 GB card at all. The
  moves are loads, so all of this precedes ``on_ready``; the encode, which sits between
  the two moves, falls outside the cap with them (seconds, against the cap's minutes).

The staging is explicit here rather than diffusers' ``enable_model_cpu_offload`` hooks:
those move the transformer onto the card lazily inside the pipeline call, which would put
a ~19 GiB host-to-card copy inside the interactive cap.
"""

from __future__ import annotations

import gc
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from qontinui_imagegen.config import FacadeConfig
from qontinui_imagegen.engine import EngineOutOfMemory, HostMemory, JobInterrupted, Rendered
from qontinui_imagegen.engine.post import PostRunners
from qontinui_imagegen.jobs import (
    CARD_MODULE_DIRS,
    DENOISER_DIRS,
    TEXT_ENCODER_DIRS,
    Catalogue,
    JobPlan,
)
from qontinui_imagegen.weights import component_dir


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
        _free_device_memory()

    def free_vram_mib(self) -> int | None:
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            free, _total = torch.cuda.mem_get_info(torch.device(self.device))
            return int(free // (1024 * 1024))
        except (ImportError, RuntimeError):
            return None

    def host_memory(self) -> HostMemory | None:
        return read_host_memory()

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
            prompt_kwargs = self._ready_and_encode(plan, on_ready)  # ONE encode per job
            if plan.pipeline_kind == "diffusers":
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
            for name in TEXT_ENCODER_DIRS:
                if (model_dir / name).is_dir():
                    kwargs[name] = None
        pipeline = pipeline_cls.from_pretrained(str(model_dir), **kwargs)
        if load.placement != "model_cpu_offload":
            pipeline.to(self.device)  # offload keeps everything on the host until a job stages it
        pipeline.set_progress_bar_config(disable=True)
        return _Loaded(plan.resident_key, pipeline, model_dir, load.placement)

    # --- generation -------------------------------------------------------------------------

    def _ready_and_encode(self, plan: JobPlan, on_ready: Callable[[], None]) -> dict[str, Any]:
        """Do the job's remaining loads, call ``on_ready``, and return the prompt arguments.

        Every placement calls ``on_ready`` only once nothing is left to load, so the
        interactive cap never counts a load; the prompt is encoded once per job, and every
        candidate shares the result.
        """
        if plan.pipeline_kind != "diffusers":
            on_ready()
            return {}
        import torch

        assert self._loaded is not None
        placement = self._loaded.placement
        if placement == "model_cpu_offload":
            pipeline = self._loaded.pipeline
            self._stage(pipeline, TEXT_ENCODER_DIRS)  # a load: host -> card
            with torch.no_grad():
                prompt_kwargs = encode_prompt_kwargs(pipeline, plan, self.device)
            self._stage(pipeline, DENOISER_DIRS)  # a load: the encoder off, the denoiser on
            on_ready()
            return prompt_kwargs
        if placement == "sequential_text_encoder":
            encoder = self._load_text_encoder(plan)  # a load, every job: disk -> card
            try:
                on_ready()
                with torch.no_grad():
                    return encode_prompt_kwargs(encoder, plan, self.device)
            finally:
                del encoder  # freed before the denoiser runs
                _free_device_memory()
        on_ready()
        kwargs: dict[str, Any] = {"prompt": plan.rendered_prompt}
        if plan.negative_prompt is not None:
            kwargs["negative_prompt"] = plan.negative_prompt
        return kwargs

    def _stage(self, pipeline: Any, on_card: Sequence[str]) -> None:
        """Put exactly the ``on_card`` modules on the card: every other one off FIRST.

        Moving off before moving on is what keeps the text encoder and the transformer from
        ever sharing the card under model_cpu_offload.
        """
        modules = {
            name: module
            for name in CARD_MODULE_DIRS
            if (module := getattr(pipeline, name, None)) is not None
        }
        for name, module in modules.items():
            if name not in on_card:
                module.to("cpu")
        _free_device_memory()
        for name in on_card:
            if name in modules:
                modules[name].to(self.device)

    def _load_text_encoder(self, plan: JobPlan) -> Any:
        """The sequential placement's per-job text encoder: the encoder pipeline in bf16.

        Phase 0 measured this load at 52-55 s for Z-Image's encoder, every job, warm or
        cold -- so it runs before ``on_ready``.
        """
        import diffusers
        import torch

        assert self._loaded is not None
        pipeline_cls = getattr(diffusers, plan.pipeline)
        return pipeline_cls.from_pretrained(
            str(self._loaded.model_dir),
            transformer=None,
            vae=None,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to(self.device)

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


#: cgroup v1 reports "no limit" as a page-rounded LONG_MAX; anything this large is unlimited.
_CGROUP_V1_UNLIMITED = 1 << 60


def _stat_field(path: Path, *keys: str) -> int:
    """The first of ``keys`` present in a cgroup ``memory.stat`` file (bytes)."""
    fields: dict[str, int] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        key, _, value = line.partition(" ")
        if value:
            fields[key] = int(value)
    for key in keys:
        if key in fields:
            return fields[key]
    raise KeyError(f"{path}: none of {keys}")


def _cgroup_limit(cgroup: Path) -> tuple[int, int] | None:
    """(limit, usage) of this container's memory cgroup in bytes, or None when unlimited.

    Usage leaves out the inactive file cache, which the kernel reclaims before it would
    OOM-kill anything: weight verification reads tens of GB through the page cache just
    before a load, and counting that cache as used would refuse a load that fits.
    Raises OSError / ValueError / KeyError when a limit is set but the usage beside it
    cannot be read -- an unreadable usage under a real limit is not "no limit".
    """
    v2_max = cgroup / "memory.max"
    if v2_max.is_file():
        raw = v2_max.read_text(encoding="ascii").strip()
        if raw == "max":
            return None
        current = int((cgroup / "memory.current").read_text(encoding="ascii").strip())
        inactive = _stat_field(cgroup / "memory.stat", "inactive_file")
        return int(raw), max(0, current - inactive)
    v1 = cgroup / "memory"
    v1_limit = v1 / "memory.limit_in_bytes"
    if v1_limit.is_file():
        limit = int(v1_limit.read_text(encoding="ascii").strip())
        if limit >= _CGROUP_V1_UNLIMITED:
            return None
        usage = int((v1 / "memory.usage_in_bytes").read_text(encoding="ascii").strip())
        inactive = _stat_field(v1 / "memory.stat", "total_inactive_file", "inactive_file")
        return limit, max(0, usage - inactive)
    return None


def _own_rss_mib(statm: Path) -> int:
    """This process's resident set in MiB; 0 when unreadable (assume none is held yet)."""
    try:
        resident_pages = int(statm.read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") // (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return 0


def read_host_memory(
    meminfo: Path = Path("/proc/meminfo"),
    cgroup: Path = Path("/sys/fs/cgroup"),
    statm: Path = Path("/proc/self/statm"),
) -> HostMemory | None:
    """Host memory as this process may use it, or None when it cannot be read.

    ``MemAvailable`` / ``MemTotal``, narrowed by the container's memory cgroup limit when it
    has one (v2 ``memory.max``, or v1 ``memory/memory.limit_in_bytes``): a load past that
    limit is OOM-killed by the cgroup, never refused, so the limit is the real total. A
    limit whose usage cannot be read makes the whole reading None (unreadable), never a
    reading without the limit.
    """
    try:
        fields = {}
        for line in meminfo.read_text(encoding="ascii").splitlines():
            key, _, rest = line.partition(":")
            fields[key] = int(rest.split()[0]) * 1024  # kB
        available, total = fields["MemAvailable"], fields["MemTotal"]
    except (OSError, KeyError, ValueError, IndexError):
        return None
    try:
        limited = _cgroup_limit(cgroup)
    except (OSError, KeyError, ValueError):
        return None
    if limited is not None:
        limit, used = limited
        total = min(total, limit)
        available = min(available, max(0, limit - used))
    return HostMemory(
        available_mib=available // (1024 * 1024),
        total_mib=total // (1024 * 1024),
        own_rss_mib=_own_rss_mib(statm),
    )


def _free_device_memory() -> None:
    """Collect what was just dropped and hand the card's cached blocks back."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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
