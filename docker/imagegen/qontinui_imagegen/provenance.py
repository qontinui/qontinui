"""``qontinui_provenance``: what made one image, attached to every result (plan Phase 10).

The set decision record a5ec7670 requires of every generated asset: the recipe (id, version,
hash), the models with the sha256 of every weight file loaded, the seed, the prompt, the
digests of the reference images, the parent asset, the SPDX licence, and the engine versions.
Phase 11's studio adapter embeds this object verbatim in the asset's sidecar.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from qontinui_imagegen import __version__, marker
from qontinui_imagegen.jobs import Catalogue, JobPlan
from qontinui_imagegen.manifest import (
    DECISION_RECORD_ID,
    DECISION_RECORD_NAME,
    DECISION_RECORD_VERSION,
)

SCHEMA = "qontinui.imagegen.provenance/1"
#: Keys every provenance object carries; tests/test_facade.py checks each is present.
REQUIRED_KEYS = (
    "schema",
    "job_id",
    "created_at",
    "recipe",
    "models",
    "licence",
    "seed",
    "candidate_index",
    "prompt",
    "references",
    "parent_asset",
    "engine",
    "marker",
)


def licence_expression(spdx_ids: list[str]) -> str:
    """The AND of the distinct licences of the components that produced the pixels."""
    distinct = sorted(set(spdx_ids))
    if len(distinct) == 1:
        return distinct[0]
    return " AND ".join(f"({item})" if " " in item else item for item in distinct)


def build(
    catalogue: Catalogue,
    plan: JobPlan,
    *,
    candidate_index: int,
    seed: int | None,
    engine_versions: Mapping[str, str],
    created_at: datetime | None = None,
) -> dict[str, Any]:
    models = []
    for model_id in plan.model_ids:
        component = catalogue.components[model_id]
        models.append(
            {
                "id": model_id,
                "role": component.get("role"),
                "source": dict(component["source"]),
                "spdx": component["spdx"],
                "weights": [
                    {"path": entry["path"], "sha256": entry["sha256"]}
                    for entry in component["files"]
                ],
            }
        )
    runner = catalogue.runners[plan.pipeline]
    code = [catalogue.components[code_id] for code_id in runner["code"]]
    when = created_at or datetime.now(timezone.utc)
    return {
        "schema": SCHEMA,
        "job_id": plan.job_id,
        "created_at": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "recipe": {
            "id": plan.recipe["id"],
            "version": plan.recipe["version"],
            "hash": plan.recipe_hash,
        },
        "models": models,
        "licence": {
            "spdx": licence_expression([model["spdx"] for model in models]),
            "code": {component["id"]: component["spdx"] for component in code},
            "decision_record": {
                "name": DECISION_RECORD_NAME,
                "id": DECISION_RECORD_ID,
                "version": DECISION_RECORD_VERSION,
            },
        },
        "seed": seed,
        "candidate_index": candidate_index,
        "prompt": plan.prompt,
        "prompt_used": plan.prompt_used,
        "rendered_prompt": plan.rendered_prompt,
        "negative_prompt": plan.negative_prompt,
        "size": list(plan.size) if plan.size else None,
        "priority": plan.priority,
        "references": [
            {"sha256": image.sha256, "media_type": image.media_type, "bytes": len(image.data)}
            for image in plan.inputs
        ],
        "parent_asset": plan.parent_asset,
        "engine": {"facade": __version__, **dict(engine_versions)},
        "marker": {"iptc_digital_source_type": marker.DIGITAL_SOURCE_TYPE},
    }
