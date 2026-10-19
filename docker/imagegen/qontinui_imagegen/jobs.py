"""From an OpenAI images request to a job plan the engine can run.

The request names a model ALIAS (``model``), which is what llama-swap routes on; the alias
picks its default recipe from ``facade.yaml`` unless the request names one (``recipe``, sent
by OpenAI clients through ``extra_body``). A recipe is the whole contract for the pixels:
steps, guidance, scheduler and sizes come from it and are never overridden per request, so
the recipe hash in the provenance identifies what generated the image. What a request may
set is what a recipe leaves open: the prompt, the seed, the size among the recipe's presets,
the candidate count, a negative prompt where the recipe uses one, and the priority.

Anything a request sends that this facade would not honour is refused with a typed 400
rather than ignored: a dropped parameter is a silent difference between what the caller
asked for and what the provenance says was made.
"""

from __future__ import annotations

import hashlib
import io
import secrets
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qontinui_imagegen import errors
from qontinui_imagegen.config import ConfigError, FacadeConfig
from qontinui_imagegen.manifest import (
    MANIFEST_PATH,
    RECIPES_DIR,
    load_recipes,
    load_yaml,
    recipe_hash,
)

ROUTES = ("generations", "edits")
PRIORITIES = ("interactive", "bulk")
MAX_INPUT_IMAGES = 4
MAX_INPUT_BYTES = 50 * 1024 * 1024
MAX_N = 10
SEED_LIMIT = 2**63
#: A diffusers checkpoint's text-encoder folders: what the sequential_text_encoder placement
#: leaves out of the resident pipeline and loads per job instead.
TEXT_ENCODER_DIRS = ("text_encoder", "text_encoder_2")
#: The folders that denoise and decode: what model_cpu_offload holds on the card to render.
DENOISER_DIRS = ("transformer", "vae")
#: A diffusers checkpoint's folders that become modules on the card (tokenizers, schedulers
#: and processors stay on the host).
CARD_MODULE_DIRS = (*DENOISER_DIRS, *TEXT_ENCODER_DIRS)
MIB = 1024 * 1024
#: Roles whose pipeline renders from a prompt alone; the others need input images (edits).
TEXT_TO_IMAGE_ROLES = frozenset({"text-to-image"})
INPUT_ROLES = frozenset({"image-edit", "upscale"})
#: OpenAI request fields accepted only at the value that changes nothing here.
NEUTRAL_FIELDS: dict[str, frozenset[Any]] = {
    "quality": frozenset({"auto"}),
    "background": frozenset({"auto"}),
    "partial_images": frozenset({0}),
    "response_format": frozenset({"b64_json"}),
}
#: Fields accepted and not used for the pixels (OpenAI's end-user id).
IGNORED_FIELDS = frozenset({"user"})
GENERATION_FIELDS = frozenset(
    {
        "model",
        "prompt",
        "n",
        "size",
        "size_preset",
        "output_format",
        "stream",
        "recipe",
        "seed",
        "negative_prompt",
        "priority",
        "parent_asset",
    }
)


@dataclass(frozen=True)
class InputImage:
    data: bytes
    sha256: str
    media_type: str
    width: int
    height: int


@dataclass(frozen=True)
class PostStep:
    op: str
    runner: str
    models: tuple[str, ...]
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class JobPlan:
    job_id: str
    route: str
    alias: str
    recipe: Mapping[str, Any]
    recipe_hash: str
    pipeline: str
    pipeline_kind: str
    generator_model: str
    prompt: str | None
    rendered_prompt: str | None
    prompt_used: bool
    negative_prompt: str | None
    size: tuple[int, int] | None
    n: int
    seeds: tuple[int | None, ...]
    call_parameters: Mapping[str, Any]
    post: tuple[PostStep, ...]
    inputs: tuple[InputImage, ...]
    priority: str
    parent_asset: str | None
    stream: bool
    output_format: str
    model_ids: tuple[str, ...]
    resident_key: str

    @property
    def steps_per_image(self) -> int:
        steps = self.call_parameters.get("num_inference_steps")
        return (int(steps) if isinstance(steps, int) else 1) + len(self.post)

    @property
    def total_steps(self) -> int:
        return self.n * self.steps_per_image

    @property
    def pixels(self) -> int:
        if self.size is not None:
            return self.size[0] * self.size[1]
        return max((image.width * image.height for image in self.inputs), default=0)


class Catalogue:
    """The manifest, the recipes and the configuration, cross-checked once at startup."""

    def __init__(
        self,
        config: FacadeConfig,
        manifest: Mapping[str, Any],
        recipes: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.config = config
        self.manifest = manifest
        self.recipes = dict(recipes)
        self.components: dict[str, Mapping[str, Any]] = {
            str(c["id"]): c for c in manifest["components"] if isinstance(c, Mapping)
        }
        self.runners: dict[str, Mapping[str, Any]] = {
            str(r["id"]): r for r in manifest["runners"] if isinstance(r, Mapping)
        }
        self.recipe_hashes = {
            recipe_id: recipe_hash(recipe, manifest) for recipe_id, recipe in self.recipes.items()
        }
        #: Output scale of every upscaler: the factor the engine's network applies
        #: (``architecture.netscale``), checked here so a request never meets a missing one.
        self.upscale_factors: dict[str, int] = {}
        for model_id, component in self.components.items():
            if component.get("role") != "upscale":
                continue
            netscale = (component.get("architecture") or {}).get("netscale")
            if isinstance(netscale, bool) or not isinstance(netscale, int) or netscale < 1:
                raise ValueError(
                    f"upscaler {model_id!r} declares no positive integer architecture.netscale"
                )
            self.upscale_factors[model_id] = netscale
        self._check_card_fit()
        for recipe_id, recipe in self.recipes.items():
            for step in recipe.get("post") or ():
                if step.get("op") == "upscale" and (
                    len(step.get("models") or ()) != 1
                    or step["models"][0] not in self.upscale_factors
                ):
                    raise ValueError(
                        f"recipe {recipe_id!r}: an upscale post step names one upscaler model"
                    )

    @classmethod
    def load(
        cls,
        config: FacadeConfig,
        manifest_path: Path = MANIFEST_PATH,
        recipes_dir: Path = RECIPES_DIR,
    ) -> Catalogue:
        return cls(config, load_yaml(manifest_path), load_recipes(recipes_dir))

    @property
    def aliases(self) -> tuple[str, ...]:
        return tuple(sorted(self.config.aliases))

    def weighted_models(self, plan_models: Sequence[str]) -> tuple[str, ...]:
        return tuple(model for model in plan_models if model in self.components)

    def vram_need_mib(self, model_ids: Sequence[str]) -> int:
        return sum(
            self.config.models[model].vram_need_mib
            for model in model_ids
            if model in self.config.models
        )

    # --- what fits on the card ---------------------------------------------------------------

    def module_bytes(self, model_id: str) -> dict[str, int]:
        """Held bytes of each on-card module folder: the transformer at its storage dtype.

        A model with no placement (an upscaler, a matting model) is one module of its files'
        size, under ``"."``.
        """
        component = self.components[model_id]
        load = self.config.models.get(model_id)
        sizes: dict[str, int] = {}
        for entry in component.get("files") or ():
            path = str(entry["path"])
            folder = path.split("/", 1)[0] if "/" in path else "."
            if load is not None and load.placement is not None and folder not in CARD_MODULE_DIRS:
                continue
            size = int(entry.get("bytes") or 0)
            if folder == "transformer" and load is not None:
                size = load.transformer_bytes_on_card(size)
            sizes[folder] = sizes.get(folder, 0) + size
        return sizes

    def card_peak_bytes(self, model_id: str) -> int:
        """The most of ``model_id``'s weights the card holds at once, by its placement.

        resident, and sequential_text_encoder on a WARM job (the per-job encoder is loaded
        while the transformer stays resident): encoders + transformer + VAE together.
        model_cpu_offload: the encoder stage or the denoising stage, never both.
        """
        sizes = self.module_bytes(model_id)
        load = self.config.models.get(model_id)
        encoders = sum(sizes.get(name, 0) for name in TEXT_ENCODER_DIRS)
        denoiser = sum(sizes.get(name, 0) for name in DENOISER_DIRS)
        if load is None or load.placement is None:
            return sum(sizes.values())
        if load.placement == "model_cpu_offload":
            return max(encoders, denoiser)
        return encoders + denoiser

    def host_peak_bytes(self, model_id: str) -> int:
        """Host memory a model keeps for its life: everything, under model_cpu_offload."""
        load = self.config.models.get(model_id)
        if load is None or load.placement != "model_cpu_offload":
            return 0
        return sum(self.module_bytes(model_id).values())

    def host_need_mib(self, model_ids: Sequence[str]) -> int:
        return -(-sum(self.host_peak_bytes(m) for m in model_ids if m in self.components) // MIB)

    def _check_card_fit(self) -> None:
        """Refuse, at startup, a configuration whose weights cannot fit the card.

        Per model: its weights' peak on the card (``card_peak_bytes``) must fit
        ``card_vram_budget_mib``, and its declared ``vram_need_mib`` (what the pre-load
        check asks for, activations included) must cover that peak and fit the budget too.
        Per recipe: every model it loads is on the card together, so their needs sum.
        Each refusal names its rule in brackets ([weights_over_budget], [need_below_weights],
        [need_over_budget], [recipe_over_budget]) so an operator and a test can tell them apart.
        """
        budget = self.config.card_vram_budget_mib
        problems: list[str] = []
        for model_id, load in self.config.models.items():
            if model_id not in self.components:
                continue
            peak = -(-self.card_peak_bytes(model_id) // MIB)
            where = f"facade.models.{model_id} ({load.placement or 'single module'})"
            if peak > budget:
                problems.append(
                    f"{where}: [weights_over_budget] its weights alone hold {peak} MiB on the "
                    f"card at once, over card_vram_budget_mib {budget}"
                )
            elif load.vram_need_mib < peak:
                problems.append(
                    f"{where}: [need_below_weights] vram_need_mib {load.vram_need_mib} is less "
                    f"than its weights' {peak} MiB peak on the card"
                )
            elif load.vram_need_mib > budget:
                problems.append(
                    f"{where}: [need_over_budget] vram_need_mib {load.vram_need_mib} is over "
                    f"card_vram_budget_mib {budget}, so no load could ever be admitted"
                )
        for recipe_id, recipe in self.recipes.items():
            post_models = [m for step in recipe["post"] for m in step.get("models") or ()]
            for generator in recipe["models"]:  # a job loads one generator plus the post models
                need = self.vram_need_mib([generator, *post_models])
                if need > budget:
                    problems.append(
                        f"recipe {recipe_id!r}: [recipe_over_budget] {generator} and its "
                        f"post-step models need {need} MiB together, over card_vram_budget_mib "
                        f"{budget}"
                    )
        if problems:
            raise ConfigError("the configuration cannot fit the card: " + "; ".join(problems))

    def _load_bytes(self, model_id: str) -> tuple[int, int]:
        """(bytes a cold start loads and keeps, bytes loaded again for every job)."""
        component = self.components.get(model_id)
        if component is None:
            return 0, 0
        load = self.config.models.get(model_id)
        per_job_dirs = (
            TEXT_ENCODER_DIRS
            if load is not None and load.placement == "sequential_text_encoder"
            else ()
        )
        resident = per_job = 0
        for entry in component.get("files") or ():
            size = int(entry.get("bytes") or 0)
            if str(entry["path"]).split("/", 1)[0] in per_job_dirs:
                per_job += size
            else:
                resident += size
        return resident, per_job

    def expected_load_s(self, plan: JobPlan, *, cold: bool) -> float:
        """How long ``plan``'s loads should take: what the interactive cap does not count.

        Every job pays its per-job loads (the sequential placement's text encoder); a cold
        job also pays everything it keeps resident. The rate is facade.yaml's load_estimate,
        a DISK rate: model_cpu_offload's per-job host<->card moves are not in it (PCIe
        copies, unmeasured), so for those models the estimate is short by them.
        """
        total = 0
        for model_id in plan.model_ids:
            resident, per_job = self._load_bytes(model_id)
            total += per_job + (resident if cold else 0)
        return self.config.load_estimate.seconds(total)

    def output_scale(self, alias: str, pipeline_kind: str, post: Sequence[PostStep]) -> int:
        """How many times wider (and taller) the output is than what the pipeline renders.

        A named upscaler's own scale, times every upscale post step's: the work an
        interactive cap bounds is the OUTPUT a job makes, not the size it starts from.
        """
        factor = self.upscale_factors.get(alias, 1) if pipeline_kind == "named" else 1
        for step in post:
            if step.op == "upscale":
                factor *= self.upscale_factors[step.models[0]]
        return factor


def _field_errors(body: Mapping[str, Any], allowed: frozenset[str]) -> None:
    for key, value in body.items():
        if key in allowed or key in IGNORED_FIELDS:
            continue
        neutral = NEUTRAL_FIELDS.get(key)
        if neutral is not None:
            if value is None or value in neutral:
                continue
            raise errors.invalid(
                "unsupported_value",
                f"{key}={value!r} is not supported by this provider; accepted: "
                f"{sorted(map(str, neutral))}",
                key,
            )
        raise errors.invalid("unknown_parameter", f"{key!r} is not a parameter here", key)


def _string(body: Mapping[str, Any], key: str, *, required: bool) -> str | None:
    value = body.get(key)
    if value is None:
        if required:
            raise errors.invalid("missing_parameter", f"{key!r} is required", key)
        return None
    if not isinstance(value, str) or (required and not value.strip()):
        raise errors.invalid("invalid_value", f"{key!r} must be a non-empty string", key)
    return value


def _integer(value: Any, key: str, lower: int, upper: int) -> int:
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        value = int(value)  # multipart form fields arrive as text
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value < upper:
        raise errors.invalid(
            "invalid_value", f"{key!r} must be an integer in [{lower}, {upper})", key
        )
    return value


def _boolean(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise errors.invalid("invalid_value", f"{key!r} must be a boolean", key)


def _size(parameters: Mapping[str, Any], size: Any, preset: Any) -> tuple[int, int] | None:
    if parameters.get("size") == "from_inputs" or "size_presets" not in parameters:
        if (size not in (None, "auto")) or preset is not None:
            raise errors.invalid(
                "size_from_inputs",
                "this recipe takes its size from the input images; send no size",
                "size",
            )
        return None
    presets: Mapping[str, Sequence[int]] = parameters["size_presets"]
    if preset is not None:
        if preset not in presets:
            raise errors.invalid(
                "unknown_size_preset",
                f"size_preset {preset!r} is not one of this recipe's {sorted(presets)}",
                "size_preset",
            )
        chosen = presets[preset]
        if size not in (None, "auto", f"{chosen[0]}x{chosen[1]}"):
            raise errors.invalid(
                "size_conflict", f"size {size!r} disagrees with size_preset {preset!r}", "size"
            )
        return int(chosen[0]), int(chosen[1])
    if size in (None, "auto"):
        chosen = presets[parameters["default_size"]]
        return int(chosen[0]), int(chosen[1])
    for dims in presets.values():
        if size == f"{dims[0]}x{dims[1]}":
            return int(dims[0]), int(dims[1])
    offered = sorted(f"{dims[0]}x{dims[1]}" for dims in presets.values())
    raise errors.invalid(
        "unsupported_size", f"size {size!r} is not one of this recipe's sizes {offered}", "size"
    )


def decode_input(data: bytes, param: str) -> InputImage:
    """Read one uploaded image; refuse anything Pillow cannot open as a still image."""
    from PIL import Image, UnidentifiedImageError

    if not data:
        raise errors.invalid("empty_image", "an uploaded image is empty", param)
    if len(data) > MAX_INPUT_BYTES:
        raise errors.invalid(
            "image_too_large", f"an uploaded image exceeds {MAX_INPUT_BYTES} bytes", param
        )
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
            fmt = (image.format or "").lower()
    except (UnidentifiedImageError, OSError) as error:
        raise errors.invalid(
            "unreadable_image", f"an uploaded image is unreadable: {error}", param
        ) from error
    return InputImage(
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        media_type=f"image/{fmt}" if fmt else "application/octet-stream",
        width=width,
        height=height,
    )


def build_plan(
    catalogue: Catalogue,
    route: str,
    body: Mapping[str, Any],
    inputs: Sequence[InputImage] = (),
    *,
    random_seed: Any = None,
) -> JobPlan:
    """Resolve ``body`` (JSON for generations, the form's text fields for edits) into a plan.

    ``random_seed`` returns the seed a ``request_or_random`` policy records when the request
    sends none (tests inject it; production draws from :mod:`secrets`).
    """
    if route not in ROUTES:
        raise ValueError(route)
    _field_errors(body, GENERATION_FIELDS)
    config = catalogue.config
    alias = _string(body, "model", required=True)
    assert alias is not None
    if alias not in config.aliases:
        raise errors.invalid(
            "model_not_found",
            f"model {alias!r} is not served here; served: {list(catalogue.aliases)}",
            "model",
        )
    recipe_id = _string(body, "recipe", required=False) or config.aliases[alias]
    recipe = catalogue.recipes.get(recipe_id)
    if recipe is None:
        raise errors.invalid("recipe_not_found", f"recipe {recipe_id!r} does not exist", "recipe")
    if alias not in recipe["models"]:
        raise errors.invalid(
            "recipe_model_mismatch",
            f"recipe {recipe_id!r} runs {list(recipe['models'])}, not {alias!r}",
            "recipe",
        )
    role = str(catalogue.components[alias].get("role"))
    wants_inputs = role in INPUT_ROLES
    if wants_inputs and route != "edits":
        raise errors.invalid(
            "wrong_route",
            f"model {alias!r} ({role}) works on input images: use POST /v1/images/edits",
            "model",
        )
    if not wants_inputs and route != "generations":
        raise errors.invalid(
            "wrong_route",
            f"model {alias!r} ({role}) renders from a prompt: use POST /v1/images/generations",
            "model",
        )
    runner = catalogue.runners[str(recipe["pipeline"])]
    pipeline_kind = str(runner["kind"])
    parameters: Mapping[str, Any] = recipe["parameters"]

    priority = body.get("priority") or "interactive"
    if priority not in PRIORITIES:
        raise errors.invalid("invalid_value", f"priority is one of {list(PRIORITIES)}", "priority")
    n = _integer(body.get("n", 1), "n", 1, MAX_N + 1)
    if priority == "interactive" and n > config.admission.interactive_max_images:
        raise errors.invalid(
            "interactive_cap_exceeded",
            f"an interactive request makes at most {config.admission.interactive_max_images} "
            "images; send priority 'bulk' for a larger batch",
            "n",
        )
    stream = _boolean(body.get("stream", False), "stream")
    if stream and n != 1:
        raise errors.invalid("stream_needs_one_image", "stream supports n=1 only", "stream")

    size = _size(parameters, body.get("size"), body.get("size_preset"))
    if wants_inputs:
        if not inputs:
            raise errors.invalid("missing_image", "this model needs an input image", "image")
        limit = 1 if pipeline_kind == "named" else MAX_INPUT_IMAGES
        if len(inputs) > limit:
            raise errors.invalid(
                "too_many_images", f"this recipe takes at most {limit} input image(s)", "image"
            )
    elif inputs:
        raise errors.invalid("unexpected_image", "this model takes no input image", "image")
    post = tuple(
        PostStep(
            op=str(step["op"]),
            runner=str(step["runner"]),
            models=tuple(step.get("models") or ()),
            parameters=dict(step.get("parameters") or {}),
        )
        for step in recipe["post"]
    )
    if priority == "interactive":
        # The work is the OUTPUT: what the pipeline renders, times the square of every
        # upscale (a named upscaler's own, and each upscale post step's).
        pixels = size[0] * size[1] if size else max(i.width * i.height for i in inputs)
        pixels *= catalogue.output_scale(alias, pipeline_kind, post) ** 2
        if pixels > config.admission.interactive_max_pixels:
            raise errors.invalid(
                "interactive_cap_exceeded",
                f"{pixels} output px exceeds the interactive cap of "
                f"{config.admission.interactive_max_pixels}; send priority 'bulk'",
                "size",
            )

    prompt = _string(body, "prompt", required=pipeline_kind == "diffusers")
    prompt_used = pipeline_kind == "diffusers"
    rendered_prompt: str | None = None
    if prompt_used and prompt is not None:
        template = parameters.get("prompt_template")
        rendered_prompt = template.format(prompt=prompt) if isinstance(template, str) else prompt
    negative = _string(body, "negative_prompt", required=False)
    if negative is not None and "negative_prompt" not in parameters:
        raise errors.invalid(
            "negative_prompt_unsupported",
            f"recipe {recipe_id!r} renders without a negative prompt (no classifier-free "
            "guidance), so one would change nothing",
            "negative_prompt",
        )
    if negative is None and isinstance(parameters.get("negative_prompt"), str):
        negative = str(parameters["negative_prompt"])

    seed_policy: Mapping[str, Any] = recipe["seed_policy"]
    seeds: tuple[int | None, ...]
    if seed_policy["mode"] == "seeded":
        if body.get("seed") is not None:
            base = _integer(body["seed"], "seed", 0, SEED_LIMIT - MAX_N)
        else:
            base = random_seed() if random_seed else secrets.randbelow(2**32)
        seeds = tuple(base + index for index in range(n))
    else:
        if body.get("seed") is not None:
            raise errors.invalid(
                "seed_unused", f"recipe {recipe_id!r} is deterministic and takes no seed", "seed"
            )
        if n != 1:
            raise errors.invalid(
                "n_unsupported",
                f"recipe {recipe_id!r} is deterministic: every candidate would be the same "
                "image, so it renders exactly one (n=1)",
                "n",
            )
        seeds = (None,)

    output_format = "svg" if post and post[-1].op == "vectorize" else "png"
    requested_format = body.get("output_format")
    if requested_format not in (None, output_format):
        raise errors.invalid(
            "unsupported_output_format",
            f"recipe {recipe_id!r} produces {output_format}",
            "output_format",
        )
    parent = _string(body, "parent_asset", required=False)
    if parent is not None and len(parent) > 512:
        raise errors.invalid(
            "invalid_value", "parent_asset is at most 512 characters", "parent_asset"
        )

    call_parameters = {
        key: value
        for key, value in parameters.items()
        if key
        not in {
            "scheduler",
            "size_presets",
            "default_size",
            "size",
            "prompt_template",
            "negative_prompt",
            "scale_models",
            "default_scale",
        }
    }
    if pipeline_kind == "named":
        scale_models: Mapping[str, str] = parameters.get("scale_models") or {}
        scale = next((int(s) for s, model in scale_models.items() if model == alias), None)
        if scale is not None:
            call_parameters["scale"] = scale
    model_ids = (alias, *(model for step in post for model in step.models))
    load_key = [
        f"{config.models[m].transformer_storage}/{config.models[m].placement}"
        for m in model_ids
        if m in config.models
    ]
    return JobPlan(
        job_id=uuid.uuid4().hex,
        route=route,
        alias=alias,
        recipe=recipe,
        recipe_hash=catalogue.recipe_hashes[recipe_id],
        pipeline=str(recipe["pipeline"]),
        pipeline_kind=pipeline_kind,
        generator_model=alias,
        prompt=prompt,
        rendered_prompt=rendered_prompt,
        prompt_used=prompt_used,
        negative_prompt=negative if prompt_used else None,
        size=size,
        n=n,
        seeds=seeds,
        call_parameters=call_parameters,
        post=post,
        inputs=tuple(inputs),
        priority=str(priority),
        parent_asset=parent,
        stream=stream,
        output_format=output_format,
        model_ids=model_ids,
        resident_key="|".join([str(recipe["pipeline"]), *model_ids, *load_key]),
    )
