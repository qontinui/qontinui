"""The facade's configuration: ``facade.yaml``, checked strictly at startup.

A value the facade cannot honour refuses startup (:class:`ConfigError`) instead of running
on a guess -- the facade is the only owner of a shared GPU, and a silently defaulted cap or
window is a desktop made sluggish without anyone having decided it (plan Phase 10, D1.3).

The machine switch ``QONTINUI_IMAGEGEN`` (plan Phase 12 step 4) is read per request by
:func:`machine_switch`, not here: the switch is an operator's lever, and the configuration
file is not where it lives.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from typing import Any

from qontinui_imagegen.manifest import IMAGEGEN_ROOT, load_yaml

DEFAULT_CONFIG_PATH = IMAGEGEN_ROOT / "facade.yaml"
CONFIG_ENV = "QONTINUI_IMAGEGEN_CONFIG"
WEIGHTS_ROOT_ENV = "QONTINUI_IMAGEGEN_WEIGHTS_ROOT"
#: The machine-scope off-switch (plan Phase 12 step 4, D1.7). ``0`` turns the facade off.
SWITCH_ENV = "QONTINUI_IMAGEGEN"
#: Who may reach llama-swap's listener, as attested by the deployment that bound it (rule
#: R-bind, D1.6). The facade has no authentication of its own: the authorization for image
#: work is that only the box reaches :8100. The file that makes that true -- the host-local
#: loopback override (docker/llama-swap/README.md, "Bind loopback only") -- sets this to
#: ``loopback``; a later device-JWT front door would set its own value. Until a deployment
#: attests a listener, the facade answers ``disabled``: Phase 0 read R-bind as UNKNOWN, and
#: an UNKNOWN bind is the off-box arm, where the image aliases must not be served.
LISTENER_ENV = "QONTINUI_IMAGEGEN_LISTENER"
ATTESTED_LISTENERS = frozenset({"loopback"})

TRANSFORMER_STORAGE = frozenset({"fp8", "bf16"})
PLACEMENTS = frozenset({"resident", "sequential_text_encoder", "model_cpu_offload"})
STAMP_FIELDS = frozenset({"size", "mtime_ns", "inode"})
_CLOCK = re.compile(r"([01]\d|2[0-3]):([0-5]\d)")


class ConfigError(ValueError):
    """The configuration cannot be honoured; the facade does not start."""


@dataclass(frozen=True)
class ModelLoad:
    """How one weighted model is loaded and the VRAM its load needs."""

    vram_need_mib: int
    vram_basis: str
    transformer_storage: str | None = None
    placement: str | None = None


@dataclass(frozen=True)
class Admission:
    interactive_gpu_seconds_cap: float
    interactive_max_pixels: int
    interactive_max_images: int
    max_interactive_waiters: int
    bulk_window_start: time
    bulk_window_end: time


@dataclass(frozen=True)
class FacadeConfig:
    weights_root: Path
    stamps_file: Path
    stamp_fields: tuple[str, ...]
    device: str
    admission: Admission
    vram_desktop_reserve_mib: int
    aliases: Mapping[str, str]
    models: Mapping[str, ModelLoad] = field(default_factory=dict)


@dataclass(frozen=True)
class SwitchState:
    """The machine switch as read: ``enabled`` plus the raw value, for the refusal's text."""

    enabled: bool
    raw: str | None
    recognised: bool


def machine_switch(environ: Mapping[str, str] | None = None) -> SwitchState:
    """Read ``QONTINUI_IMAGEGEN``: unset or ``1`` is on, ``0`` is off.

    Any other value is OFF and reported as unrecognised: a switch someone set to a value the
    facade does not understand was set by someone meaning to change something, and the
    direction that cannot surprise them with GPU load is off. The switch is never the
    authorization (the loopback bind is) -- it only removes the capability.
    """
    env = os.environ if environ is None else environ
    raw = env.get(SWITCH_ENV)
    if raw is None or raw.strip() == "1":
        return SwitchState(enabled=True, raw=raw, recognised=True)
    if raw.strip() == "0":
        return SwitchState(enabled=False, raw=raw, recognised=True)
    return SwitchState(enabled=False, raw=raw, recognised=False)


def listener_attestation(environ: Mapping[str, str] | None = None) -> str | None:
    """The attested listener (``loopback``), or None when no deployment attested one."""
    env = os.environ if environ is None else environ
    value = (env.get(LISTENER_ENV) or "").strip()
    return value if value in ATTESTED_LISTENERS else None


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{where}.{key} is required")
    return mapping[key]


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{where} must be a positive integer, not {value!r}")
    return value


def _clock(value: Any, where: str) -> time:
    if not isinstance(value, str) or not _CLOCK.fullmatch(value):
        raise ConfigError(f"{where} must be HH:MM on the 24-hour clock, not {value!r}")
    hours, minutes = value.split(":")
    return time(int(hours), int(minutes))


def _check_keys(mapping: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"{where} has unknown keys {unknown}")


def parse_config(
    document: Any,
    *,
    environ: Mapping[str, str] | None = None,
    known_models: set[str] | None = None,
    known_recipes: Mapping[str, Mapping[str, Any]] | None = None,
) -> FacadeConfig:
    """Check ``document`` (facade.yaml's content) and build the configuration.

    ``known_models`` and ``known_recipes`` (manifest model ids; recipes keyed by id) make
    the alias table and the per-model loads refer only to things that exist, and each
    alias's default recipe list that alias among its models.
    """
    env = os.environ if environ is None else environ
    if not isinstance(document, Mapping):
        raise ConfigError("the facade configuration is a mapping")
    _check_keys(
        document,
        {
            "schema_version",
            "weights_root",
            "stamps_file",
            "stamp_fields",
            "device",
            "admission",
            "vram_desktop_reserve_mib",
            "aliases",
            "models",
        },
        "facade",
    )
    if document.get("schema_version") != 1:
        raise ConfigError("facade.schema_version must be 1")
    weights_root = Path(env.get(WEIGHTS_ROOT_ENV) or _require(document, "weights_root", "facade"))
    stamps = Path(_require(document, "stamps_file", "facade"))
    if stamps.is_absolute() or ".." in stamps.parts:
        raise ConfigError("facade.stamps_file is a path relative to weights_root")
    fields = _require(document, "stamp_fields", "facade")
    if (
        not isinstance(fields, list)
        or not fields
        or not set(fields) <= STAMP_FIELDS
        or "size" not in fields
    ):
        raise ConfigError(
            f"facade.stamp_fields lists some of {sorted(STAMP_FIELDS)} and always size"
        )
    device = _require(document, "device", "facade")
    if not isinstance(device, str) or not device:
        raise ConfigError("facade.device names a torch device")

    raw_admission = _require(document, "admission", "facade")
    if not isinstance(raw_admission, Mapping):
        raise ConfigError("facade.admission is a mapping")
    _check_keys(
        raw_admission,
        {
            "interactive_gpu_seconds_cap",
            "interactive_max_pixels",
            "interactive_max_images",
            "max_interactive_waiters",
            "bulk_window",
        },
        "facade.admission",
    )
    window = _require(raw_admission, "bulk_window", "facade.admission")
    if not isinstance(window, Mapping) or set(window) != {"start", "end"}:
        raise ConfigError("facade.admission.bulk_window is {start: HH:MM, end: HH:MM}")
    start = _clock(window["start"], "facade.admission.bulk_window.start")
    end = _clock(window["end"], "facade.admission.bulk_window.end")
    if start == end:
        raise ConfigError("facade.admission.bulk_window is empty (start equals end)")
    waiters = _require(raw_admission, "max_interactive_waiters", "facade.admission")
    if isinstance(waiters, bool) or not isinstance(waiters, int) or waiters not in (0, 1):
        raise ConfigError(
            "facade.admission.max_interactive_waiters is 0 or 1: a deeper queue here "
            "would starve every other llama-swap alias on the card (plan Phase 10)"
        )
    admission = Admission(
        interactive_gpu_seconds_cap=float(
            _positive_int(
                _require(raw_admission, "interactive_gpu_seconds_cap", "facade.admission"),
                "facade.admission.interactive_gpu_seconds_cap",
            )
        ),
        interactive_max_pixels=_positive_int(
            _require(raw_admission, "interactive_max_pixels", "facade.admission"),
            "facade.admission.interactive_max_pixels",
        ),
        interactive_max_images=_positive_int(
            _require(raw_admission, "interactive_max_images", "facade.admission"),
            "facade.admission.interactive_max_images",
        ),
        max_interactive_waiters=waiters,
        bulk_window_start=start,
        bulk_window_end=end,
    )
    reserve = _require(document, "vram_desktop_reserve_mib", "facade")
    if isinstance(reserve, bool) or not isinstance(reserve, int) or reserve < 0:
        raise ConfigError("facade.vram_desktop_reserve_mib is a non-negative integer")

    raw_models = _require(document, "models", "facade")
    if not isinstance(raw_models, Mapping):
        raise ConfigError("facade.models is a mapping")
    models: dict[str, ModelLoad] = {}
    for model_id, raw in raw_models.items():
        where = f"facade.models.{model_id}"
        if known_models is not None and model_id not in known_models:
            raise ConfigError(f"{where}: {model_id!r} is not a model in the manifest")
        if not isinstance(raw, Mapping):
            raise ConfigError(f"{where} is a mapping")
        _check_keys(raw, {"vram_need_mib", "vram_basis", "transformer_storage", "placement"}, where)
        basis = _require(raw, "vram_basis", where)
        if not isinstance(basis, str) or not basis.startswith(("measured", "estimate")):
            raise ConfigError(f"{where}.vram_basis starts with 'measured' or 'estimate'")
        storage = raw.get("transformer_storage")
        placement = raw.get("placement")
        if (storage is None) != (placement is None):
            raise ConfigError(f"{where}: a diffusion model names transformer_storage AND placement")
        if storage is not None and storage not in TRANSFORMER_STORAGE:
            raise ConfigError(
                f"{where}.transformer_storage is one of {sorted(TRANSFORMER_STORAGE)}"
            )
        if placement is not None and placement not in PLACEMENTS:
            raise ConfigError(f"{where}.placement is one of {sorted(PLACEMENTS)}")
        models[str(model_id)] = ModelLoad(
            vram_need_mib=_positive_int(
                _require(raw, "vram_need_mib", where), f"{where}.vram_need_mib"
            ),
            vram_basis=basis,
            transformer_storage=storage,
            placement=placement,
        )

    raw_aliases = _require(document, "aliases", "facade")
    if not isinstance(raw_aliases, Mapping) or not raw_aliases:
        raise ConfigError("facade.aliases maps each model alias to its default recipe")
    aliases: dict[str, str] = {}
    for alias, recipe_id in raw_aliases.items():
        where = f"facade.aliases.{alias}"
        if known_models is not None and alias not in known_models:
            raise ConfigError(f"{where}: {alias!r} is not a model in the manifest")
        if known_recipes is not None:
            recipe = known_recipes.get(str(recipe_id))
            if recipe is None:
                raise ConfigError(f"{where}: {recipe_id!r} is not a recipe")
            if alias not in (recipe.get("models") or []):
                raise ConfigError(f"{where}: recipe {recipe_id!r} does not list {alias!r}")
        aliases[str(alias)] = str(recipe_id)
    return FacadeConfig(
        weights_root=weights_root,
        stamps_file=stamps,
        stamp_fields=tuple(fields),
        device=device,
        admission=admission,
        vram_desktop_reserve_mib=reserve,
        aliases=aliases,
        models=models,
    )


def load_config(
    path: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    known_models: set[str] | None = None,
    known_recipes: Mapping[str, Mapping[str, Any]] | None = None,
) -> FacadeConfig:
    env = os.environ if environ is None else environ
    chosen = path or Path(env.get(CONFIG_ENV) or DEFAULT_CONFIG_PATH)
    return parse_config(
        load_yaml(chosen),
        environ=env,
        known_models=known_models,
        known_recipes=known_recipes,
    )
