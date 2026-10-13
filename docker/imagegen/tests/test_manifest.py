"""The image provider's licence manifest, pinned hashes and recipes (plan
2026-10-13-image-generation-and-visual-judgment, Phase 2).

Decision record D2 (decision_record/image-generation-weights-are-permissively-licensed,
a5ec7670) admits Apache-2.0, MIT and BSD-family licences only, and requires each licence to
sit beside its pinned hash, checked by a test. These tests check the real manifest and
recipes, and prove the checker can fail: every mutation fixture is refused with exactly the
violations it was built to cause, and nothing else.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest
import yaml
from qontinui_imagegen import manifest as m

FIXTURES = Path(__file__).resolve().parent / "fixtures"
RECIPE_FIXTURES = FIXTURES / "recipes"
HEX64 = re.compile(r"[0-9a-f]{64}")

#: The baseline recipe's hash. A change here means the canonical form changed, which makes
#: every recorded asset's provenance hash stale -- change it only on purpose.
BASELINE_RECIPE_HASH = "8dc441f2315988b596a88423fe172809d1caa19dbc1cb27250778c2030943e70"

MANIFEST_MUTATIONS: dict[str, set[str]] = {
    "mutation-flux-1-dev.yaml": {"licence_not_admitted", "excluded_component"},
    "mutation-flux-1-dev-labelled-apache.yaml": {"excluded_component"},
    "mutation-qwen-image-2.1.yaml": {"licence_not_admitted", "excluded_component"},
    "mutation-hunyuanimage-3.0.yaml": {"licence_not_admitted", "excluded_component"},
    "mutation-comfyui.yaml": {"licence_not_admitted", "excluded_component"},
    "mutation-rmbg-2.0.yaml": {"licence_not_admitted", "excluded_component"},
    "mutation-unpinned-file.yaml": {"unpinned_file"},
    "mutation-unpinned-revision.yaml": {"unpinned_revision"},
    "mutation-derived-licence.yaml": {"derived_invalid"},
    "mutation-derived-unknown-source.yaml": {"derived_invalid"},
    "mutation-runtime-online.yaml": {"runtime_not_offline"},
    "mutation-runtime-remote-code.yaml": {"runtime_not_offline"},
}

RECIPE_MUTATIONS: dict[str, set[str]] = {
    "mutation-trust-remote-code.yaml": {"network_fetch"},
    "mutation-hub-url.yaml": {"network_fetch"},
    "mutation-hub-online.yaml": {"network_fetch"},
    "mutation-local-files-only.yaml": {"network_fetch"},
    "mutation-hub-revision.yaml": {"network_fetch"},
    "mutation-hub-model-id.yaml": {"model_unresolved"},
    "mutation-size-off-grid.yaml": {"recipe_schema"},
    "mutation-unknown-runner.yaml": {"runner_unresolved", "pipeline_mismatch"},
    "mutation-unseeded-diffusion.yaml": {"recipe_schema"},
}


def codes(violations: list[m.Violation]) -> set[str]:
    return {violation.code for violation in violations}


def explain(violations: list[m.Violation]) -> str:
    return "\n".join(map(str, violations))


@pytest.fixture(scope="module")
def real_manifest() -> dict:
    return m.load_yaml(m.MANIFEST_PATH)


@pytest.fixture(scope="module")
def real_recipes() -> dict:
    return m.load_recipes(m.RECIPES_DIR)


@pytest.fixture
def baseline() -> dict:
    return m.load_yaml(FIXTURES / "baseline-manifest.yaml")


@pytest.fixture
def baseline_recipe() -> dict:
    return m.load_yaml(RECIPE_FIXTURES / "baseline-recipe.yaml")


# --- the real manifest and recipes ---------------------------------------------------------


def test_real_manifest_recipes_and_vendored_bytes_are_admitted() -> None:
    violations = m.check()
    assert violations == [], explain(violations)


def test_admitted_families_are_exactly_d2s() -> None:
    assert m.ADMITTED_SPDX == {
        "Apache-2.0",
        "MIT",
        "MIT-0",
        "MIT-CMU",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "0BSD",
    }


def test_every_component_carries_an_admitted_licence_beside_pinned_files(
    real_manifest: dict,
) -> None:
    for component in real_manifest["components"]:
        assert m.spdx_admitted(component["spdx"]), (component["id"], component["spdx"])
        assert m.excluded_reason(component["id"]) is None, component["id"]
        assert HEX64.fullmatch(component["licence_evidence"]["sha256"]), component["id"]
        for entry in component["files"]:
            assert HEX64.fullmatch(entry["sha256"]), (component["id"], entry["path"])
            assert entry["bytes"] > 0, (component["id"], entry["path"])


def test_the_six_recipes_are_present(real_recipes: dict) -> None:
    assert set(real_recipes) == {
        "hero",
        "illustration",
        "empty-state",
        "edit",
        "cut-out",
        "upscale",
    }


def test_every_recipe_model_resolves_to_a_pinned_manifest_entry(
    real_manifest: dict, real_recipes: dict
) -> None:
    components = {component["id"]: component for component in real_manifest["components"]}
    for name, recipe in real_recipes.items():
        referenced = [
            *recipe["models"],
            *(model for step in recipe["post"] for model in step["models"]),
        ]
        assert referenced, name
        for model_id in referenced:
            component = components[model_id]
            assert component["kind"] == "model", (name, model_id)
            assert component["files"], (name, model_id)
            assert all(HEX64.fullmatch(entry["sha256"]) for entry in component["files"]), (
                name,
                model_id,
            )


def test_no_recipe_enables_a_network_fetch(real_manifest: dict, real_recipes: dict) -> None:
    assert real_manifest["runtime"] == {
        "env": {"HF_HUB_OFFLINE": "1"},
        "from_pretrained": {"local_files_only": True, "trust_remote_code": False},
        "torch_load": {"weights_only": True},
    }
    for name, recipe in real_recipes.items():
        assert "network_fetch" not in codes(m.check_recipe(name, recipe, real_manifest)), name
        text = (m.RECIPES_DIR / f"{name}.yaml").read_text(encoding="utf-8")
        assert "trust_remote_code" not in text and "://" not in text, name


def test_matting_code_is_vendored_not_fetched(real_manifest: dict) -> None:
    components = {component["id"]: component for component in real_manifest["components"]}
    for model_id, code_id in (("birefnet", "birefnet-code"), ("ben2-base", "ben2-code")):
        module = components[model_id]["architecture"]["module"]
        vendored = {entry["path"]: entry for entry in components[code_id]["files"]}
        assert module in vendored, model_id
        assert vendored[module]["upstream"]["sha256"] == vendored[module]["sha256"], model_id
        on_disk = (m.IMAGEGEN_ROOT / module).read_bytes()
        assert hashlib.sha256(on_disk).hexdigest() == vendored[module]["sha256"], model_id


def test_the_qwen_generators_store_their_shared_encoder_once(real_manifest: dict) -> None:
    components = {component["id"]: component for component in real_manifest["components"]}
    generate = {entry["path"]: entry for entry in components["qwen-image-2512"]["files"]}
    edit = {entry["path"]: entry for entry in components["qwen-image-edit-2511"]["files"]}
    shared_paths = [
        path for path in generate if path.split("/")[0] in ("text_encoder", "vae", "tokenizer")
    ]
    assert sum(path.startswith("text_encoder/model-") for path in shared_paths) == 4
    for path in shared_paths:
        assert edit[path]["sha256"] == generate[path]["sha256"], path

    footprint = m.install_footprint(real_manifest, ["qwen-image-2512", "qwen-image-edit-2511"])
    for path in shared_paths:
        assert footprint.shared[generate[path]["sha256"]] == (
            "qwen-image-2512",
            "qwen-image-edit-2511",
        )
    distinct = {entry["sha256"]: entry["bytes"] for entry in [*generate.values(), *edit.values()]}
    assert footprint.unique_bytes == sum(distinct.values())
    assert footprint.listed_bytes == sum(
        entry["bytes"] for entry in [*generate.values(), *edit.values()]
    )
    assert footprint.unique_bytes < footprint.listed_bytes


def test_real_recipe_hashes_are_distinct(real_manifest: dict, real_recipes: dict) -> None:
    hashes = {name: m.recipe_hash(recipe, real_manifest) for name, recipe in real_recipes.items()}
    assert all(HEX64.fullmatch(value) for value in hashes.values())
    assert len(set(hashes.values())) == len(hashes)


# --- the checker can fail: manifest mutations ----------------------------------------------


def test_baseline_manifest_fixture_is_admitted(baseline: dict) -> None:
    violations = m.check_manifest(baseline)
    assert violations == [], explain(violations)


def test_every_manifest_mutation_fixture_is_exercised() -> None:
    assert {path.name for path in FIXTURES.glob("mutation-*.yaml")} == set(MANIFEST_MUTATIONS)


@pytest.mark.parametrize(("fixture", "expected"), sorted(MANIFEST_MUTATIONS.items()))
def test_mutation_manifest_is_refused(fixture: str, expected: set[str]) -> None:
    violations = m.check_manifest(m.load_yaml(FIXTURES / fixture))
    assert codes(violations) == expected, explain(violations)


# --- the checker can fail: recipe mutations ------------------------------------------------


def test_baseline_recipe_fixture_is_admitted(baseline: dict, baseline_recipe: dict) -> None:
    violations = m.check_recipe("example", baseline_recipe, baseline)
    assert violations == [], explain(violations)


def test_every_recipe_mutation_fixture_is_exercised() -> None:
    assert {path.name for path in RECIPE_FIXTURES.glob("mutation-*.yaml")} == set(RECIPE_MUTATIONS)


@pytest.mark.parametrize(("fixture", "expected"), sorted(RECIPE_MUTATIONS.items()))
def test_mutation_recipe_is_refused(fixture: str, expected: set[str], baseline: dict) -> None:
    recipe = m.load_yaml(RECIPE_FIXTURES / fixture)
    violations = m.check_recipe("example", recipe, baseline)
    assert codes(violations) == expected, explain(violations)


def test_a_recipe_id_must_match_its_file(baseline: dict, baseline_recipe: dict) -> None:
    assert codes(m.check_recipe("hero", baseline_recipe, baseline)) == {"recipe_schema"}


# --- vendored code on disk -----------------------------------------------------------------


@pytest.fixture
def vendored_tree(tmp_path: Path) -> Path:
    shutil.copy(FIXTURES / "vendored-manifest.yaml", tmp_path / "manifest.yaml")
    sample = tmp_path / "vendor" / "sample"
    sample.mkdir(parents=True)
    (sample / "sample.py").write_bytes(b"VALUE = 1\n")
    (sample / "__init__.py").write_bytes(b"")
    return tmp_path


def check_tree(root: Path) -> list[m.Violation]:
    return m.check_manifest(m.load_yaml(root / "manifest.yaml"), base_dir=root)


def test_vendored_files_matching_their_pins_are_admitted(vendored_tree: Path) -> None:
    violations = check_tree(vendored_tree)
    assert violations == [], explain(violations)


def test_vendored_bytes_that_drift_are_refused(vendored_tree: Path) -> None:
    (vendored_tree / "vendor" / "sample" / "sample.py").write_bytes(b"VALUE = 2\n")
    assert codes(check_tree(vendored_tree)) == {"vendored_mismatch"}


def test_a_missing_vendored_file_is_refused(vendored_tree: Path) -> None:
    (vendored_tree / "vendor" / "sample" / "sample.py").unlink()
    assert codes(check_tree(vendored_tree)) == {"vendored_mismatch"}


def test_an_unlisted_vendored_file_is_refused(vendored_tree: Path) -> None:
    (vendored_tree / "vendor" / "sample" / "helper.py").write_bytes(b"import os\n")
    assert codes(check_tree(vendored_tree)) == {"vendored_unlisted"}


def test_a_package_marker_with_content_is_refused(vendored_tree: Path) -> None:
    (vendored_tree / "vendor" / "sample" / "__init__.py").write_bytes(
        b"from .sample import VALUE\n"
    )
    assert codes(check_tree(vendored_tree)) == {"vendored_unlisted"}


# --- SPDX expressions and D2's named exclusions --------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "Apache-2.0",
        "MIT",
        "MIT-0",
        "MIT-CMU",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "0BSD",
        "MIT AND Apache-2.0",
        "BSD-2-Clause OR GPL-3.0-only",
        "Apache-2.0 WITH LLVM-exception",
        "(MIT OR GPL-2.0-only) AND BSD-3-Clause",
    ],
)
def test_admitted_expressions(expression: str) -> None:
    assert m.spdx_admitted(expression)


@pytest.mark.parametrize(
    ("expression", "blockers"),
    [
        ("GPL-3.0", ["GPL-3.0"]),
        ("GPL-3.0-only", ["GPL-3.0-only"]),
        ("LicenseRef-qwen-research", ["LicenseRef-qwen-research"]),
        (
            "LicenseRef-flux-1-dev-non-commercial-license",
            ["LicenseRef-flux-1-dev-non-commercial-license"],
        ),
        ("CC-BY-NC-4.0", ["CC-BY-NC-4.0"]),
        ("CC-BY-SA-4.0", ["CC-BY-SA-4.0"]),
        ("Apache-2.0 AND BSL-1.0", ["BSL-1.0"]),
        ("MIT AND (GPL-2.0-only OR LGPL-2.1-only)", ["GPL-2.0-only", "LGPL-2.1-only"]),
        ("GPL-2.0-only WITH Classpath-exception-2.0", ["GPL-2.0-only"]),
        ("mit", ["mit"]),
    ],
)
def test_refused_expressions_name_what_blocks_them(expression: str, blockers: list[str]) -> None:
    assert m.spdx_blockers(expression) == blockers


@pytest.mark.parametrize(
    "expression",
    ["", "MIT AND", "(MIT", "MIT)", "MIT WITH", "AND MIT", "MIT OR OR 0BSD", "MIT/BSD-3-Clause"],
)
def test_unparseable_expressions_raise(expression: str) -> None:
    with pytest.raises(m.SpdxError):
        m.spdx_blockers(expression)


@pytest.mark.parametrize(
    "name",
    [
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-Kontext-dev",
        "black-forest-labs/FLUX.1-Redux-dev",
        "black-forest-labs/FLUX.2-dev",
        "Qwen/Qwen-Image-2.1",
        "tencent/HunyuanImage-3.0",
        "briaai/RMBG-2.0",
        "OmniSVG/OmniSVG",
        "Comfy-Org/ComfyUI",
        "comfyui",
    ],
)
def test_d2s_named_exclusions_are_refused(name: str) -> None:
    assert m.excluded_reason(name)


@pytest.mark.parametrize(
    "name",
    [
        "black-forest-labs/FLUX.1-schnell",
        "Qwen/Qwen-Image-2512",
        "Qwen/Qwen-Image-Edit-2511",
        "Tongyi-MAI/Z-Image-Turbo",
        "stabilityai/sd-vae-ft-mse",
        "example/ComfyUI-custom-node",
    ],
)
def test_names_d2_does_not_exclude_pass_the_name_check(name: str) -> None:
    assert m.excluded_reason(name) is None


# --- strict YAML ---------------------------------------------------------------------------


def test_a_duplicate_key_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("spdx: MIT\nspdx: GPL-3.0\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        m.load_yaml(path)


def test_timestamps_stay_strings(tmp_path: Path) -> None:
    path = tmp_path / "timestamp.yaml"
    path.write_text("licence_read_at: 2026-10-13T07:40:11Z\n", encoding="utf-8")
    assert m.load_yaml(path) == {"licence_read_at": "2026-10-13T07:40:11Z"}


# --- the recipe hash -----------------------------------------------------------------------


def test_recipe_hash_is_sha256_of_the_canonical_recipe_and_its_entries(
    baseline: dict, baseline_recipe: dict
) -> None:
    components = {component["id"]: component for component in baseline["components"]}
    excluded = {"licence_read_at", "licence_evidence", "licence_url", "notes"}

    def loaded(component_id: str) -> dict:
        return {
            key: value for key, value in components[component_id].items() if key not in excluded
        }

    document = {
        "recipe": {key: value for key, value in baseline_recipe.items() if key != "description"},
        "manifest": {
            "runners": {"ExamplePipeline": baseline["runners"][0]},
            "components": {
                "diffusers": loaded("diffusers"),
                "example-generator": loaded("example-generator"),
            },
        },
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert m.recipe_hash(baseline_recipe, baseline) == expected == BASELINE_RECIPE_HASH


def test_recipe_hash_ignores_key_order_prose_and_licence_rereads(
    baseline: dict, baseline_recipe: dict
) -> None:
    reference = m.recipe_hash(baseline_recipe, baseline)
    assert m.recipe_hash(dict(reversed(list(baseline_recipe.items()))), baseline) == reference
    assert m.recipe_hash({**baseline_recipe, "description": "Other prose."}, baseline) == reference
    reread = copy.deepcopy(baseline)
    reread["components"][0]["licence_read_at"] = "2027-01-01T00:00:00Z"
    reread["components"][0]["notes"] = "Re-read; unchanged."
    assert m.recipe_hash(baseline_recipe, reread) == reference


def test_recipe_hash_changes_with_what_is_loaded(baseline: dict, baseline_recipe: dict) -> None:
    reference = m.recipe_hash(baseline_recipe, baseline)

    new_weights = copy.deepcopy(baseline)
    new_weights["components"][0]["files"][1]["sha256"] = "0" * 64
    assert m.recipe_hash(baseline_recipe, new_weights) != reference

    upgraded = copy.deepcopy(baseline)
    upgraded["components"][2]["source"]["version"] = "0.42.0"
    assert m.recipe_hash(baseline_recipe, upgraded) != reference

    retuned = copy.deepcopy(baseline_recipe)
    retuned["parameters"]["num_inference_steps"] = 9
    assert m.recipe_hash(retuned, baseline) != reference

    unreferenced = copy.deepcopy(baseline)
    unreferenced["components"][1]["files"][0]["bytes"] = 1
    assert m.recipe_hash(baseline_recipe, unreferenced) == reference


# --- footprint and the command line --------------------------------------------------------


def test_footprint_refuses_an_unknown_component(real_manifest: dict) -> None:
    with pytest.raises(KeyError):
        m.install_footprint(real_manifest, ["flux.1-dev"])


def test_cli_admits_the_real_tree(capsys: pytest.CaptureFixture[str]) -> None:
    assert m.main(["--hashes", "--footprint"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"^hero v1 [0-9a-f]{64}$", out, re.MULTILINE)
    assert "unique " in out


def test_cli_refuses_a_mutation_fixture(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    shutil.copy(RECIPE_FIXTURES / "baseline-recipe.yaml", recipes / "example.yaml")
    assert (
        m.main(["--manifest", str(FIXTURES / "baseline-manifest.yaml"), "--recipes", str(recipes)])
        == 0
    )
    capsys.readouterr()
    assert (
        m.main(["--manifest", str(FIXTURES / "mutation-comfyui.yaml"), "--recipes", str(recipes)])
        == 1
    )
    assert "licence_not_admitted at components[comfyui].spdx" in capsys.readouterr().out
