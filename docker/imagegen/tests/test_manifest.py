"""The image provider's licence manifest, pinned hashes and recipes (plan
2026-10-13-image-generation-and-visual-judgment, Phase 2).

Decision record D2 (decision_record/image-generation-weights-are-permissively-licensed,
a5ec7670 v2) admits Apache-2.0, MIT and BSD-family licences, plus BSL-1.0, Zlib, libpng-2.0
and IJG for code bundled inside admitted packages and ``Apache-2.0 WITH LLVM-exception`` as
its only exception, and requires each licence to sit beside its pinned hash, checked by a
test. These tests check the real manifest and recipes, and prove the checker can fail:
every mutation fixture is refused with exactly the violations it was built to cause, and
nothing else.
"""

from __future__ import annotations

import ast
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
BASELINE_RECIPE_HASH = "84d0b339404fc57968ed5eb9cf9df5cae17beb9582d7a1493a211938cedf9d42"

MANIFEST_MUTATIONS: dict[str, set[str]] = {
    # D2's named exclusions and licences it does not admit
    "mutation-flux-1-dev.yaml": {"licence_not_admitted", "excluded_component"},
    "mutation-flux-1-dev-labelled-apache.yaml": {"excluded_component"},
    "mutation-qwen-image-2.1.yaml": {"licence_not_admitted", "excluded_component"},
    "mutation-hunyuanimage-3.0.yaml": {"licence_not_admitted", "excluded_component"},
    "mutation-comfyui.yaml": {"licence_not_admitted", "excluded_component"},
    "mutation-rmbg-2.0.yaml": {"licence_not_admitted", "excluded_component"},
    "mutation-apache-commons-clause.yaml": {"licence_not_admitted"},
    "mutation-mit-licenseref-rider.yaml": {"licence_not_admitted"},
    "mutation-vendored-from-excluded.yaml": {"excluded_component"},
    # pins: one sha256 per file, full commit ids, licences read at the pin
    "mutation-unpinned-file.yaml": {"unpinned_file"},
    # A branch name pins nothing, so the evidence read at the real revision is unpinned too.
    "mutation-unpinned-revision.yaml": {"unpinned_revision", "licence_unpinned"},
    "mutation-hash-size-conflict.yaml": {"hash_size_conflict"},
    "mutation-evidence-at-main.yaml": {"licence_unpinned"},
    "mutation-evidence-stale-revision.yaml": {"licence_unpinned"},
    "mutation-platform-member-unpinned.yaml": {"unpinned_file"},
    # what a wheel bundles (a5ec7670 v2, "How a licence is read")
    "mutation-bundled-missing.yaml": {"schema"},
    "mutation-bundled-unread.yaml": {"licence_unread"},
    "mutation-bundled-lib-not-admitted.yaml": {"licence_not_admitted", "licence_incomplete"},
    "mutation-bundled-incomplete.yaml": {"licence_incomplete"},
    "mutation-bundled-lib-evidence-unpinned.yaml": {"licence_unpinned"},
    # statically linked dependencies
    "mutation-static-dependency-lgpl.yaml": {"licence_not_admitted"},
    "mutation-static-dependency-unknown.yaml": {"licence_unknown"},
    "mutation-static-dependency-other-version.yaml": {"licence_unpinned"},
    # derivations and the runtime contract
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
    "mutation-custom-pipeline.yaml": {"network_fetch"},
    "mutation-custom-revision.yaml": {"network_fetch"},
    "mutation-hub-model-id.yaml": {"model_unresolved"},
    "mutation-scale-models-outside.yaml": {"model_unresolved"},
    "mutation-size-off-grid.yaml": {"recipe_schema"},
    "mutation-unknown-runner.yaml": {"runner_unresolved", "pipeline_mismatch"},
    "mutation-unseeded-diffusion.yaml": {"recipe_schema"},
}

#: The helpers a5ec7670 v2 has removed from BEN2.py (code of unknown licence).
BEN2_REMOVED = frozenset(
    {
        "refine_foreground_process",
        "FB_blur_fusion_foreground_estimator",
        "FB_blur_fusion_foreground_estimator_2",
    }
)


def codes(violations: list[m.Violation]) -> set[str]:
    return {violation.code for violation in violations}


def explain(violations: list[m.Violation]) -> str:
    return "\n".join(map(str, violations))


def by_id(manifest: dict) -> dict[str, dict]:
    return {component["id"]: component for component in manifest["components"]}


def evidence_items(component: dict) -> list[dict]:
    evidence = component["licence_evidence"]
    return evidence if isinstance(evidence, list) else [evidence]


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


def test_the_checker_mirrors_decision_record_v2(real_manifest: dict) -> None:
    assert m.DECISION_RECORD_VERSION == 2
    assert real_manifest["decision_record"]["version"] == 2
    assert m.ADMITTED_SPDX == {
        "Apache-2.0",
        "MIT",
        "MIT-0",
        "MIT-CMU",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "0BSD",
        "BSL-1.0",
        "Zlib",
        "libpng-2.0",
        "IJG",
    }
    assert m.ADMITTED_EXCEPTIONS == {"Apache-2.0": {"LLVM-exception"}}


def test_every_component_carries_an_admitted_licence_beside_pinned_files(
    real_manifest: dict,
) -> None:
    for component in real_manifest["components"]:
        assert m.spdx_admitted(component["spdx"]), (component["id"], component["spdx"])
        assert m.excluded_reason(component["id"]) is None, component["id"]
        for item in evidence_items(component):
            assert HEX64.fullmatch(item["sha256"]), component["id"]
        for entry in component["files"]:
            assert HEX64.fullmatch(entry["sha256"]), (component["id"], entry["path"])
            assert entry["bytes"] > 0, (component["id"], entry["path"])


def test_every_wheel_records_its_licence_files_and_bundled_libraries(real_manifest: dict) -> None:
    for component in real_manifest["components"]:
        if component["kind"] != "library" or component["source"]["type"] != "pypi":
            continue
        bundled = component["bundled"]
        read = {item["sha256"] for item in evidence_items(component)}
        assert bundled["licence_files"], component["id"]
        for item in bundled["licence_files"]:
            assert item["sha256"] in read, (component["id"], item["member"])
        for library in bundled.get("native_libraries", []):
            assert m.spdx_admitted(library["spdx"]), (component["id"], library["member"])


def test_torch_and_torchvision_carry_what_their_wheels_bundle(real_manifest: dict) -> None:
    components = by_id(real_manifest)
    torch = components["torch"]
    assert {"BSL-1.0", "MIT", "BSD-2-Clause"} <= m._licence_ids(torch["spdx"])
    assert "Apache-2.0 WITH LLVM-exception" in torch["spdx"]
    assert [item["member"] for item in torch["bundled"]["licence_files"]] == [
        "torch-2.11.0+cu128.dist-info/licenses/LICENSE",
        "torch-2.11.0+cu128.dist-info/licenses/NOTICE",
    ]
    (cpr_test,) = torch["bundled"]["not_terms"]
    assert "cpr/test/LICENSE" in cpr_test["text"]

    torchvision = components["torchvision"]
    bundled = {
        library["member"].split("/")[1].split(".")[0]: library["spdx"]
        for library in torchvision["bundled"]["native_libraries"]
    }
    assert bundled == {
        "libpng16": "libpng-2.0",
        "libjpeg": "IJG AND BSD-3-Clause AND Zlib",
        "libwebp": "BSD-3-Clause",
        "libsharpyuv": "BSD-3-Clause",
        "libz": "Zlib",
    }
    assert m._licence_ids(torchvision["spdx"]) == {"BSD-3-Clause", "libpng-2.0", "IJG", "Zlib"}


def test_platform_libraries_a_wheel_bundles_are_pinned_outside_d2(real_manifest: dict) -> None:
    platform = real_manifest["platform"]
    assert platform["outside_d2_scope"] is True
    members = {item["member"]: item for item in platform["components"] if "member" in item}
    assert set(members) == {
        "torchvision.libs/libnvjpeg.8dd2b5e6.so.12",
        "torchvision.libs/libcudart.e8e8b82a.so.12",
        "torch/lib/libgomp.so.1",
    }
    for item in members.values():
        assert HEX64.fullmatch(item["sha256"]) and item["bytes"] > 0, item["member"]
    assert "GCC-exception-3.1" in members["torch/lib/libgomp.so.1"]["licence"]
    assert "GPL-3.0-or-later" not in m.ADMITTED_SPDX


def test_compiled_rust_libraries_record_their_linked_crates(real_manifest: dict) -> None:
    components = by_id(real_manifest)
    linked_counts = {}
    for component_id in ("safetensors", "vtracer"):
        deps = components[component_id]["static_dependencies"]
        source = components[component_id]["source"]
        assert deps["lockfile"]["member"].startswith(f"{source['package']}-{source['version']}/")
        linked = [crate for crate in deps["crates"] if crate["linked"]]
        for crate in linked:
            expression = m.crate_licence_expression(crate["licence"])
            assert m.spdx_admitted(expression), (component_id, crate["name"], crate["licence"])
        linked_counts[component_id] = len(linked)
    assert linked_counts == {"safetensors": 22, "vtracer": 51}


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
    components = by_id(real_manifest)
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
        assert "custom_pipeline" not in text and "custom_revision" not in text, name


def test_matting_code_is_vendored_not_fetched(real_manifest: dict) -> None:
    components = by_id(real_manifest)
    for model_id, code_id in (("birefnet", "birefnet-code"), ("ben2-base", "ben2-code")):
        module = components[model_id]["architecture"]["module"]
        vendored = {entry["path"]: entry for entry in components[code_id]["files"]}
        assert module in vendored, model_id
        on_disk = (m.IMAGEGEN_ROOT / module).read_bytes()
        assert hashlib.sha256(on_disk).hexdigest() == vendored[module]["sha256"], model_id
    birefnet = {entry["path"]: entry for entry in components["birefnet-code"]["files"]}
    module = birefnet["vendor/birefnet/birefnet.py"]
    assert module["upstream"]["sha256"] == module["sha256"]  # byte-identical
    assert "vendor/birefnet/LICENSE-PVT" in birefnet  # its PVTv2 backbone's Apache-2.0 text


def test_the_derived_ben2_holds_none_of_the_removed_helpers(real_manifest: dict) -> None:
    entries = {entry["path"]: entry for entry in by_id(real_manifest)["ben2-code"]["files"]}
    entry = entries["vendor/ben2/BEN2.py"]
    assert "upstream" not in entry
    (origin,) = entry["derived_from"]["upstream"]
    assert origin["repo"] == "PramaLLC/BEN2" and origin["path"] == "BEN2.py"

    source = (m.IMAGEGEN_ROOT / "vendor/ben2/BEN2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    named = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    named |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    named |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not BEN2_REMOVED & named
    assert "refine_foreground == True" not in source

    # BEN_Base.inference refuses refine_foreground=True before it does anything else.
    (inference,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "inference"
    ]
    guard = inference.body[0]
    assert isinstance(guard, ast.If)
    assert isinstance(guard.test, ast.Name) and guard.test.id == "refine_foreground"
    (raised,) = guard.body
    assert isinstance(raised, ast.Raise) and isinstance(raised.exc, ast.Call)
    assert isinstance(raised.exc.func, ast.Name) and raised.exc.func.id == "ValueError"


def test_the_qwen_generators_share_their_encoder_files(real_manifest: dict) -> None:
    components = by_id(real_manifest)
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


def test_the_default_footprint_counts_every_model_judge_included(real_manifest: dict) -> None:
    every_model = [c["id"] for c in real_manifest["components"] if c["kind"] == "model"]
    assert m.install_footprint(real_manifest) == m.install_footprint(real_manifest, every_model)
    judge = m.install_footprint(real_manifest, ["qwen3.8-27b", "qwen3.8-27b-fp8"])
    generators = m.install_footprint(
        real_manifest, ["qwen-image-2512", "qwen-image-edit-2511", "z-image-turbo"]
    )
    assert (
        generators.unique_bytes + judge.unique_bytes
        < m.install_footprint(real_manifest).unique_bytes
    )


def test_real_recipe_hashes_are_distinct(real_manifest: dict, real_recipes: dict) -> None:
    hashes = {name: m.recipe_hash(recipe, real_manifest) for name, recipe in real_recipes.items()}
    assert all(HEX64.fullmatch(value) for value in hashes.values())
    assert len(set(hashes.values())) == len(hashes)


# --- the checker can fail: manifest mutations ----------------------------------------------


def test_baseline_manifest_fixtures_are_admitted(baseline: dict) -> None:
    violations = m.check_manifest(baseline)
    assert violations == [], explain(violations)
    vendored = m.check_manifest(m.load_yaml(FIXTURES / "vendored-manifest.yaml"))
    assert vendored == [], explain(vendored)


def test_every_manifest_mutation_fixture_is_exercised() -> None:
    assert {path.name for path in FIXTURES.glob("mutation-*.yaml")} == set(MANIFEST_MUTATIONS)


@pytest.mark.parametrize(("fixture", "expected"), sorted(MANIFEST_MUTATIONS.items()))
def test_mutation_manifest_is_refused(fixture: str, expected: set[str]) -> None:
    violations = m.check_manifest(m.load_yaml(FIXTURES / fixture))
    assert codes(violations) == expected, explain(violations)


@pytest.mark.parametrize("excluded", ["briaai/RMBG-2.0", "black-forest-labs/FLUX.1-dev"])
def test_vendored_code_whose_byte_identical_upstream_is_excluded_is_refused(
    excluded: str,
) -> None:
    manifest = m.load_yaml(FIXTURES / "vendored-manifest.yaml")
    sample = by_id(manifest)["sample-code"]
    sample["files"][0]["upstream"]["repo"] = excluded
    assert codes(m.check_manifest(manifest)) == {"excluded_component"}


def test_footprint_refuses_a_sha256_listed_with_two_sizes() -> None:
    manifest = m.load_yaml(FIXTURES / "mutation-hash-size-conflict.yaml")
    with pytest.raises(ValueError, match="listed as"):
        m.install_footprint(manifest, ["example-generator"])


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


@pytest.mark.parametrize("stray", ["extra.yml", "extra.json", "nested"])
def test_the_recipes_directory_holds_yaml_recipes_only(tmp_path: Path, stray: str) -> None:
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    shutil.copy(RECIPE_FIXTURES / "baseline-recipe.yaml", recipes / "example.yaml")
    manifest = FIXTURES / "baseline-manifest.yaml"
    assert m.check(manifest, recipes) == []
    if stray == "nested":
        (recipes / stray).mkdir()
    else:
        shutil.copy(RECIPE_FIXTURES / "mutation-trust-remote-code.yaml", recipes / stray)
    violations = m.check(manifest, recipes)
    assert [(v.code, v.where) for v in violations] == [("recipe_schema", f"recipes/{stray}")]


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


@pytest.mark.parametrize("vendor_dir", ["/", "absolute-tmp", "../outside", "vendor/../.."])
def test_a_vendor_dir_outside_the_manifest_directory_is_refused_unswept(
    vendored_tree: Path, vendor_dir: str
) -> None:
    manifest = m.load_yaml(vendored_tree / "manifest.yaml")
    manifest["vendor_dir"] = str(vendored_tree) if vendor_dir == "absolute-tmp" else vendor_dir
    # A file the sweep would report if it ran over the manifest's own directory tree.
    (vendored_tree / "vendor" / "sample" / "helper.py").write_bytes(b"import os\n")
    violations = m.check_manifest(manifest, base_dir=vendored_tree)
    assert [(v.code, v.where) for v in violations if v.where == "vendor_dir"] == [
        ("schema", "vendor_dir")
    ]
    assert "vendored_unlisted" not in codes(violations)
    assert "vendored_mismatch" not in codes(violations)


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
        "BSL-1.0",
        "Zlib",
        "libpng-2.0",
        "IJG",
        "MIT AND Apache-2.0",
        "Apache-2.0 AND BSL-1.0",
        "IJG AND BSD-3-Clause AND Zlib",
        "BSD-2-Clause OR GPL-3.0-only",
        "Apache-2.0 WITH LLVM-exception",
        "Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT",
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
        ("Libpng", ["Libpng"]),
        ("Unicode-3.0", ["Unicode-3.0"]),
        ("MIT AND (GPL-2.0-only OR LGPL-2.1-only)", ["GPL-2.0-only", "LGPL-2.1-only"]),
        ("GPL-2.0-only WITH Classpath-exception-2.0", ["GPL-2.0-only"]),
        ("GPL-3.0-or-later WITH GCC-exception-3.1", ["GPL-3.0-or-later"]),
        ("Apache-2.0 WITH Commons-Clause", ["Apache-2.0 WITH Commons-Clause"]),
        (
            "MIT WITH LicenseRef-non-commercial-rider",
            ["MIT WITH LicenseRef-non-commercial-rider"],
        ),
        ("MIT WITH LLVM-exception", ["MIT WITH LLVM-exception"]),
        ("BSD-3-Clause AND Apache-2.0 WITH Commons-Clause", ["Apache-2.0 WITH Commons-Clause"]),
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
    ("licence", "expression"),
    [
        ("MIT/Apache-2.0", "MIT OR Apache-2.0"),
        ("MIT / Apache-2.0", "MIT OR Apache-2.0"),
        ("Apache-2.0/MIT", "Apache-2.0 OR MIT"),
        ("MIT OR Apache-2.0", "MIT OR Apache-2.0"),
        ("(MIT OR Apache-2.0) AND Unicode-3.0", "(MIT OR Apache-2.0) AND Unicode-3.0"),
    ],
)
def test_a_crates_io_licence_reads_slash_as_or(licence: str, expression: str) -> None:
    assert m.crate_licence_expression(licence) == expression


@pytest.mark.parametrize(
    "name",
    [
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-Kontext-dev",
        "black-forest-labs/FLUX.1-Redux-dev",
        "black-forest-labs/FLUX.1-Fill-dev",
        "black-forest-labs/FLUX.1-Krea-dev",
        "black-forest-labs/FLUX.2-dev",
        "flux-dev",
        "flux_1_dev",
        "Qwen/Qwen-Image-2.1",
        "qwen_image_2_1",
        "tencent/HunyuanImage-3.0",
        "briaai/RMBG-2.0",
        "rmbg_2",
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
        "whai362/PVT",
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

#: Written out here rather than read from the module, so the oracle is independent of it.
LICENCE_ONLY_FIELDS = {
    "spdx",
    "licence_url",
    "licence_evidence",
    "licence_read_at",
    "notes",
    "static_dependencies",
    "bundled",
}


def test_recipe_hash_is_sha256_of_the_canonical_recipe_and_its_entries(
    baseline: dict, baseline_recipe: dict
) -> None:
    components = by_id(baseline)

    def loaded(component_id: str) -> dict:
        return {
            key: value
            for key, value in components[component_id].items()
            if key not in LICENCE_ONLY_FIELDS
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
    assert m.HASH_EXCLUDED_COMPONENT_FIELDS == LICENCE_ONLY_FIELDS
    assert m.recipe_hash(baseline_recipe, baseline) == expected == BASELINE_RECIPE_HASH


def test_recipe_hash_ignores_key_order_prose_and_licence_records(
    baseline: dict, baseline_recipe: dict
) -> None:
    reference = m.recipe_hash(baseline_recipe, baseline)
    assert m.recipe_hash(dict(reversed(list(baseline_recipe.items()))), baseline) == reference
    assert m.recipe_hash({**baseline_recipe, "description": "Other prose."}, baseline) == reference
    reread = copy.deepcopy(baseline)
    generator, _, diffusers = reread["components"]
    generator["licence_read_at"] = "2027-01-01T00:00:00Z"
    generator["notes"] = "Re-read; unchanged."
    generator["spdx"] = "MIT"
    generator["licence_url"] = "https://example.org/LICENSE"
    generator["licence_evidence"] = [generator["licence_evidence"]]
    diffusers["bundled"] = {"licence_files": []}
    diffusers["static_dependencies"] = {"crates": []}
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


def test_canonical_json_is_pythons_not_rfc_8785() -> None:
    # Python's float repr: JCS (RFC 8785) would write 4, not 4.0. Keys sort by code point.
    assert m.canonical_json({"b": 4.0, "a": "é", "B": [1, 2]}) == '{"B":[1,2],"a":"é","b":4.0}'
    with pytest.raises(ValueError):
        m.canonical_json({"nan": float("nan")})


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
