"""facade.yaml, the llama-swap entry, the runtime lock and the manifest agree (plan Phase 10).

Four files describe one deployment; each test here fails when one of them drifts from the
others, so a drift is a red CI run rather than a facade that refuses an alias llama-swap
routes to it, or an image that installs a library the licence manifest never admitted.
"""

from __future__ import annotations

import copy
import re
import tomllib
from pathlib import Path

import pytest
import yaml
from facade_fakes import real_catalogue
from packaging.tags import Tag, compatible_tags, cpython_tags
from packaging.utils import parse_wheel_filename

from qontinui_imagegen import config as c
from qontinui_imagegen.manifest import (
    IMAGEGEN_ROOT,
    MANIFEST_PATH,
    RECIPES_DIR,
    load_recipes,
    load_yaml,
)

DOCKER = IMAGEGEN_ROOT.parent
LLAMA_SWAP_CONFIG = DOCKER / "llama-swap" / "config.yaml"
COMPOSE = DOCKER / "llama-swap" / "docker-compose.yml"
DOCKERFILE = DOCKER / "llama-swap" / "Dockerfile"
RUNTIME_LOCK = IMAGEGEN_ROOT / "requirements-runtime.txt"
PROJECT = IMAGEGEN_ROOT / "pyproject.toml"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return load_yaml(MANIFEST_PATH)


@pytest.fixture(scope="module")
def document() -> dict:
    return load_yaml(c.DEFAULT_CONFIG_PATH)


def _known(manifest: dict) -> dict:
    return {
        "known_models": {x["id"] for x in manifest["components"] if x["kind"] == "model"},
        "known_recipes": load_recipes(RECIPES_DIR),
    }


def test_the_shipped_config_parses_against_the_manifest_and_recipes(
    manifest: dict, document: dict
) -> None:
    config = c.parse_config(document, environ={}, **_known(manifest))
    assert config.weights_root == Path("/weights/imagegen")
    assert config.admission.interactive_gpu_seconds_cap == 120  # Phase 0 threshold T5
    assert (config.admission.bulk_window_start.hour, config.admission.bulk_window_end.hour) == (
        0,
        6,
    )
    assert config.admission.max_interactive_waiters == 1


def test_every_weighted_model_a_recipe_loads_has_a_declared_vram_need(
    manifest: dict, document: dict
) -> None:
    config = c.parse_config(document, environ={}, **_known(manifest))
    recipes = load_recipes(RECIPES_DIR)
    loaded = {
        model
        for recipe_id in set(config.aliases.values()) | set(recipes)
        for model in [
            *recipes[recipe_id]["models"],
            *(m for s in recipes[recipe_id]["post"] for m in s["models"]),
        ]
    }
    assert loaded <= set(config.models), sorted(loaded - set(config.models))
    diffusion = {
        model
        for recipe in recipes.values()
        if recipe["pipeline"] not in {"realesrgan"}
        for model in recipe["models"]
    }
    for model in diffusion:
        assert config.models[model].placement in c.PLACEMENTS
        assert config.models[model].transformer_storage in c.TRANSFORMER_STORAGE


def test_the_llama_swap_entry_lists_exactly_the_facade_aliases(document: dict) -> None:
    swap = yaml.safe_load(LLAMA_SWAP_CONFIG.read_text(encoding="utf-8"))
    entry = swap["models"]["qontinui-imagegen"]
    assert sorted(entry["aliases"]) == sorted(document["aliases"])
    assert "flux-1-schnell" in entry["aliases"]
    # useModelName unset: llama-swap forwards the alias, which picks the recipe.
    assert "useModelName" not in entry
    assert "--host 127.0.0.1" in entry["cmd"] and entry["proxy"].startswith("http://127.0.0.1:")
    assert entry["checkEndpoint"] == "/health"
    # Above the facade's own bound (1 running + 1 waiting) so its typed refusals answer first.
    assert entry["concurrencyLimit"] > 2
    assert entry["metadata"] == {"qontinui_capability": "image_generation"}
    assert "persistent" not in entry
    env = dict(item.split("=", 1) for item in entry["env"])
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["QONTINUI_IMAGEGEN_WEIGHTS_ROOT"] == document["weights_root"]
    # No other entry claims an image alias.
    others = [
        alias
        for name, model in swap["models"].items()
        if name != "qontinui-imagegen"
        for alias in [name, *model.get("aliases", [])]
    ]
    assert not set(others) & set(entry["aliases"])


def test_compose_passes_the_machine_switch_and_mounts_the_weights(document: dict) -> None:
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    service = compose["services"]["llama-swap"]
    assert "QONTINUI_IMAGEGEN=${QONTINUI_IMAGEGEN:-1}" in service["environment"]
    targets = [volume.rsplit(":", 1)[-1] for volume in service["volumes"]]
    assert document["weights_root"] in targets


def test_only_the_loopback_override_attests_the_listener() -> None:
    """The tracked compose publishes :8100 on every interface, so it must not attest a
    loopback listener; the README's host-local override, which binds 127.0.0.1, does."""
    compose = COMPOSE.read_text(encoding="utf-8")
    assert c.LISTENER_ENV not in compose
    readme = (DOCKER / "llama-swap" / "README.md").read_text(encoding="utf-8")
    override = readme.split("cat > docker/llama-swap/docker-compose.override.yml <<'YAML'")[1]
    override = yaml.safe_load(override.split("YAML", 1)[0].replace("!override", ""))
    service = override["services"]["llama-swap"]
    assert service["ports"] == ["127.0.0.1:8100:8100"]
    assert f"{c.LISTENER_ENV}=loopback" in service["environment"]
    assert c.listener_attestation({c.LISTENER_ENV: "loopback"}) == "loopback"
    assert c.listener_attestation({}) is None


def test_llama_swap_is_pinned_to_the_measured_version_and_checksum() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "ARG LLAMA_SWAP_VERSION=217\n" in text
    assert re.search(r"ARG LLAMA_SWAP_SHA256=[0-9a-f]{64}\n", text)
    assert "sha256sum -c" in text and "latest" not in text.split("# --- Stage 2")[0]


def _lock() -> dict[str, tuple[str, set[str]]]:
    pins: dict[str, tuple[str, set[str]]] = {}
    current: str | None = None
    for line in RUNTIME_LOCK.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==(\S+)", line)
        if match:
            current = match.group(1).lower().replace("_", "-")
            pins[current] = (match.group(2), set())
            continue
        digest = re.search(r"--hash=sha256:([0-9a-f]{64})", line)
        if digest and current:
            pins[current][1].add(digest.group(1))
    return pins


def test_the_runtime_lock_pins_every_manifest_library_with_its_wheel_hash(manifest: dict) -> None:
    pins = _lock()
    libraries = [x for x in manifest["components"] if x["kind"] == "library"]
    assert libraries
    for library in libraries:
        name = library["source"]["package"].lower()
        assert name in pins, f"{name} is in the manifest but not the runtime lock"
        version, hashes = pins[name]
        assert version == library["source"]["version"], name
        wheel = library["files"][0]["sha256"]
        assert wheel in hashes, f"{name}: the lock does not carry the manifest's wheel sha256"


def test_the_test_project_pins_the_manifest_web_stack(manifest: dict) -> None:
    text = PROJECT.read_text(encoding="utf-8")
    for package in ("fastapi", "starlette", "uvicorn", "python-multipart"):
        version = next(x for x in manifest["components"] if x["id"] == package)["source"]["version"]
        assert f'{package} = "{version}"' in text


def test_the_lock_check_can_fail(
    manifest: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drifted = tmp_path / "requirements-runtime.txt"
    drifted.write_text(
        RUNTIME_LOCK.read_text(encoding="utf-8").replace("fastapi==0.142.2", "fastapi==0.143.3"),
        encoding="utf-8",
    )
    monkeypatch.setattr(__import__(__name__), "RUNTIME_LOCK", drifted)
    with pytest.raises(AssertionError):
        test_the_runtime_lock_pins_every_manifest_library_with_its_wheel_hash(manifest)


# --- the config refuses what it cannot honour ----------------------------------------------------------


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["admission"].__setitem__("max_interactive_waiters", 2), "0 or 1"),
        (lambda d: d["admission"]["bulk_window"].__setitem__("start", "25:00"), "HH:MM"),
        (lambda d: d["admission"]["bulk_window"].__setitem__("end", "00:00"), "empty"),
        (lambda d: d["aliases"].__setitem__("z-image-turbo", "hero"), "does not list"),
        (lambda d: d["aliases"].__setitem__("flux-1-dev", "draft"), "not a model"),
        (lambda d: d["models"]["z-image-turbo"].__setitem__("placement", "gpu"), "placement"),
        (lambda d: d["models"]["z-image-turbo"].__setitem__("vram_basis", "trust me"), "measured"),
        (lambda d: d.__setitem__("stamp_fields", ["inode"]), "always size"),
        (lambda d: d.__setitem__("stamps_file", "../stamps.json"), "relative"),
        (lambda d: d.__setitem__("surprise", 1), "unknown keys"),
        (lambda d: d.__setitem__("host_reserve_mib", -1), "host_reserve_mib"),
    ],
)
def test_a_config_it_cannot_honour_refuses_to_start(
    manifest: dict, document: dict, mutate, message: str
) -> None:
    bad = copy.deepcopy(document)
    mutate(bad)
    with pytest.raises(c.ConfigError, match=message):
        c.parse_config(bad, environ={}, **_known(manifest))


def test_the_weights_root_env_overrides_the_file(manifest: dict, document: dict) -> None:
    config = c.parse_config(
        document,
        environ={c.WEIGHTS_ROOT_ENV: "D:/qontinui-root/imagegen-weights"},
        **_known(manifest),
    )
    assert config.weights_root == Path("D:/qontinui-root/imagegen-weights")


@pytest.mark.parametrize(
    ("environ", "enabled", "recognised"),
    [
        ({}, True, True),
        ({"QONTINUI_IMAGEGEN": "1"}, True, True),
        ({"QONTINUI_IMAGEGEN": "0"}, False, True),
        ({"QONTINUI_IMAGEGEN": " 0 "}, False, True),
        ({"QONTINUI_IMAGEGEN": "no"}, False, False),
        ({"QONTINUI_IMAGEGEN": ""}, False, False),
    ],
)
def test_the_machine_switch(environ: dict, enabled: bool, recognised: bool) -> None:
    switch = c.machine_switch(environ)
    assert (switch.enabled, switch.recognised) == (enabled, recognised)


def test_the_catalogue_hashes_every_recipe() -> None:
    catalogue = real_catalogue()
    assert set(catalogue.recipe_hashes) == set(load_recipes(RECIPES_DIR))
    assert len(set(catalogue.recipe_hashes.values())) == len(catalogue.recipe_hashes)


# --- the bulk window's clock (M2) and the loopback-only bind (M4) ---------------------------


def test_the_bulk_window_reads_its_configured_zone(manifest: dict, document: dict) -> None:
    from datetime import UTC, datetime

    from qontinui_imagegen.admission import AdmissionController, zone_clock

    config = c.parse_config(document, environ={c.TIMEZONE_ENV: "Asia/Tokyo"}, **_known(manifest))
    assert config.admission.bulk_window_timezone == "Asia/Tokyo"
    # 17:30 UTC is 02:30 in Tokyo: inside 00:00-06:00 there, outside it in UTC.
    instant = datetime(2026, 10, 17, 17, 30, tzinfo=UTC)
    tokyo = instant.astimezone(__import__("zoneinfo").ZoneInfo("Asia/Tokyo")).replace(tzinfo=None)
    controller = AdmissionController(config.admission, now=lambda: tokyo)
    assert controller.in_bulk_window()
    assert not controller.in_bulk_window(instant.replace(tzinfo=None))
    assert zone_clock("Asia/Tokyo")() is not None


def test_with_no_zone_bulk_is_deferred_not_run_on_utc(manifest: dict, document: dict) -> None:
    from qontinui_imagegen import errors
    from qontinui_imagegen.admission import AdmissionController

    config = c.parse_config(document, environ={c.TIMEZONE_ENV: ""}, **_known(manifest))
    assert config.admission.bulk_window_timezone is None
    controller = AdmissionController(config.admission)
    assert not controller.in_bulk_window()
    with pytest.raises(errors.FacadeError) as raised:
        controller.admit("bulk")
    assert (raised.value.error_type, raised.value.code) == (
        "deferred",
        "bulk_window_timezone_unset",
    )
    assert controller.admit("interactive").priority == "interactive"


def test_an_unknown_zone_refuses_to_start(manifest: dict, document: dict) -> None:
    with pytest.raises(c.ConfigError, match="not a known zone"):
        c.parse_config(document, environ={c.TIMEZONE_ENV: "Mars/Olympus"}, **_known(manifest))


def test_compose_passes_the_bulk_window_zone_through() -> None:
    service = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]["llama-swap"]
    assert f"{c.TIMEZONE_ENV}=${{{c.TIMEZONE_ENV}:-}}" in service["environment"]


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "100.111.28.125", "192.168.1.5", "example.com"])
def test_serve_refuses_a_non_loopback_host_before_loading_anything(
    host: str, capsys: pytest.CaptureFixture[str]
) -> None:
    from qontinui_imagegen import serve

    assert serve.main(["--host", host, "--port", "1"]) == 2
    assert "binds loopback only" in capsys.readouterr().err


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "localhost"])
def test_loopback_hosts_are_loopback(host: str) -> None:
    from qontinui_imagegen import serve

    assert serve._loopback(host)


def test_an_upscale_post_step_must_name_an_upscaler_with_a_scale() -> None:
    """The interactive pixel cap multiplies by every upscale's scale, so the catalogue
    refuses at startup a recipe whose upscale step has no scale to multiply by."""
    from qontinui_imagegen.jobs import Catalogue

    good = real_catalogue()
    assert good.upscale_factors == {"real-esrgan-x2plus": 2, "real-esrgan-general-x4v3": 4}
    recipes = copy.deepcopy(good.recipes)
    recipes["hero"]["post"][0]["models"] = ["birefnet"]
    with pytest.raises(ValueError, match="names one upscaler model"):
        Catalogue(good.config, good.manifest, recipes)
    manifest = copy.deepcopy(dict(good.manifest))
    for component in manifest["components"]:
        if component["id"] == "real-esrgan-x2plus":
            del component["architecture"]["netscale"]
    with pytest.raises(ValueError, match="netscale"):
        Catalogue(good.config, manifest, good.recipes)


def test_a_config_without_a_load_estimate_refuses_to_start(manifest: dict, document: dict) -> None:
    for bad_value in (
        None,
        {"read_mb_per_s": 0, "basis": "estimate: x"},
        {"read_mb_per_s": 150, "basis": "a hunch"},
    ):
        bad = copy.deepcopy(document)
        if bad_value is None:
            del bad["load_estimate"]
        else:
            bad["load_estimate"] = bad_value
        with pytest.raises(c.ConfigError, match="load_estimate"):
            c.parse_config(bad, environ={}, **_known(manifest))


# --- what fits on the card (review round 3b) -------------------------------------------------


def _catalogue_for(document: dict, manifest: dict):
    from qontinui_imagegen.jobs import Catalogue

    config = c.parse_config(document, environ={}, **_known(manifest))
    return Catalogue(config, manifest, load_recipes(RECIPES_DIR))


def test_the_shipped_config_fits_the_card_and_qwen_runs_offloaded(
    manifest: dict, document: dict
) -> None:
    """Qwen under model_cpu_offload holds one stage at a time: the 19,485 MiB FP8 transformer
    + VAE (19,727 MiB), against the 15,817 MiB encoder stage -- under the 30,559 MiB budget."""
    catalogue = _catalogue_for(document, manifest)  # the startup check passes
    mib = 1024 * 1024
    for qwen in ("qwen-image-2512", "qwen-image-edit-2511"):
        assert catalogue.config.models[qwen].placement == "model_cpu_offload"
        sizes = catalogue.module_bytes(qwen)
        assert -(-sizes["transformer"] // mib) == 19485  # 40.86 GB bf16 -> FP8: / 2
        assert -(-sizes["text_encoder"] // mib) == 15817
        assert -(-catalogue.card_peak_bytes(qwen) // mib) == 19727
        assert catalogue.host_need_mib([qwen]) == 35543  # the whole model waits on the host
    # Z-Image keeps the sequential path: encoder + FP8 transformer + VAE on a warm job.
    assert -(-catalogue.card_peak_bytes("z-image-turbo") // mib) == 13703
    assert catalogue.host_need_mib(["z-image-turbo"]) == 0


@pytest.mark.parametrize("placement", ["resident", "sequential_text_encoder"])
def test_a_qwen_config_holding_encoder_and_transformer_together_refuses_to_start(
    manifest: dict, document: dict, placement: str
) -> None:
    """15.4 GiB encoder + 19.0 GiB FP8 transformer + VAE = 35,543 MiB: over the card."""
    bad = copy.deepcopy(document)
    bad["models"]["qwen-image-2512"]["placement"] = placement
    with pytest.raises(c.ConfigError, match=r"qwen-image-2512 .*35543 MiB .*over"):
        _catalogue_for(bad, manifest)


@pytest.mark.parametrize(
    ("mutate", "message", "rules"),
    [
        # A transformer kept in bf16 instead of FP8 doubles: 38,969 MiB + the VAE.
        (
            lambda d: d["models"]["qwen-image-2512"].__setitem__("transformer_storage", "bf16"),
            "facade.models.qwen-image-2512 (model_cpu_offload): [weights_over_budget] its "
            "weights alone hold 39211 MiB on the card at once, over card_vram_budget_mib 30559",
            {"weights_over_budget"},
        ),
        # A declared need below the weights' own peak would let the pre-load check underask.
        (
            lambda d: d["models"]["z-image-turbo"].__setitem__("vram_need_mib", 8704),
            "facade.models.z-image-turbo (sequential_text_encoder): [need_below_weights] "
            "vram_need_mib 8704 is less than its weights' 13703 MiB peak on the card",
            {"need_below_weights"},
        ),
        # A need the budget can never admit, though the weights themselves fit (19,727 MiB).
        # Its recipes then outgrow the budget too, which is the recipe rule's own refusal.
        (
            lambda d: d["models"]["qwen-image-2512"].__setitem__("vram_need_mib", 31000),
            "facade.models.qwen-image-2512 (model_cpu_offload): [need_over_budget] "
            "vram_need_mib 31000 is over card_vram_budget_mib 30559, so no load could ever "
            "be admitted",
            {"need_over_budget", "recipe_over_budget"},
        ),
        # A recipe whose models together outgrow the card (hero = Qwen + the 2x upscaler),
        # though each model alone is admissible.
        (
            lambda d: d["models"]["real-esrgan-x2plus"].__setitem__("vram_need_mib", 8000),
            "recipe 'hero': [recipe_over_budget] qwen-image-2512 and its post-step models "
            "need 31552 MiB together, over card_vram_budget_mib 30559",
            {"recipe_over_budget"},
        ),
    ],
)
def test_each_card_fit_rule_refuses_with_its_own_message(
    manifest: dict, document: dict, mutate, message: str, rules: set[str]
) -> None:
    bad = copy.deepcopy(document)
    mutate(bad)
    with pytest.raises(c.ConfigError) as raised:
        _catalogue_for(bad, manifest)
    assert message in str(raised.value)
    assert set(re.findall(r"\[(\w+)\]", str(raised.value))) == rules


def test_a_diffusion_model_must_name_its_checkpoint_dtype(manifest: dict, document: dict) -> None:
    bad = copy.deepcopy(document)
    del bad["models"]["flux-1-schnell"]["transformer_checkpoint_dtype"]
    with pytest.raises(c.ConfigError, match="transformer_checkpoint_dtype"):
        c.parse_config(bad, environ={}, **_known(manifest))


def _accepted_tags(python: tuple[int, int], os_name: str, arch: str, glibc: int) -> set[Tag]:
    """The wheel tags pip accepts on CPython ``python`` for ``os_name``/``arch``.

    ``glibc`` is the minor version of the newest manylinux_2_N the image may use
    (the runtime lock's --python-platform floor). The platform list mirrors
    packaging's manylinux rule for x86 (glibc 2.5 floor and the legacy aliases);
    the interpreter and ABI tags come from packaging's own generators, the code
    pip vendors.
    """
    assert os_name == "linux", (
        f"only the linux arm defines its wheel tags; add {os_name}'s when its arm lands"
    )
    assert arch in {"x86_64", "i686"}, (
        f"only x86's manylinux floor (glibc 2.5) and legacy aliases are modelled; "
        f"add {arch}'s when its arm lands"
    )
    platforms = [f"manylinux_2_{minor}_{arch}" for minor in range(glibc, 4, -1)]
    legacy = {17: "manylinux2014", 12: "manylinux2010", 5: "manylinux1"}
    platforms += [f"{name}_{arch}" for floor, name in legacy.items() if floor <= glibc]
    platforms.append(f"linux_{arch}")
    interpreter = f"cp{python[0]}{python[1]}"
    return {
        *cpython_tags(python_version=python, abis=[interpreter], platforms=platforms),
        *compatible_tags(python_version=python, interpreter=interpreter, platforms=platforms),
    }


def _wheel_installs_on(wheel: str, accepted: set[Tag]) -> bool:
    return not parse_wheel_filename(wheel)[3].isdisjoint(accepted)


@pytest.mark.parametrize(
    ("wheel", "python", "installs"),
    [
        ("x-1.0-py3-none-any.whl", (3, 12), True),
        ("x-1.0-py2.py3-none-any.whl", (3, 12), True),
        ("x-1.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", (3, 12), True),
        ("x-1.0-cp312-cp312-manylinux_2_28_x86_64.whl", (3, 12), True),
        ("x-1.0-cp312-none-manylinux_2_17_x86_64.whl", (3, 12), True),
        ("x-1.0-cp313-abi3-manylinux_2_17_x86_64.whl", (3, 12), False),
        ("x-1.0-cp311-cp311-manylinux_2_28_x86_64.whl", (3, 12), False),
        ("x-1.0-cp311-none-manylinux_2_17_x86_64.whl", (3, 12), False),
        ("x-1.0-py313-none-any.whl", (3, 12), False),
        ("x-1.0-cp312-cp312-win_amd64.whl", (3, 12), False),
        ("x-1.0-cp312-cp312-manylinux_2_28_aarch64.whl", (3, 12), False),
        ("x-1.0-cp312-cp312-musllinux_1_2_x86_64.whl", (3, 12), False),
        ("x-1.0-cp312-cp312-manylinux_2_39_x86_64.whl", (3, 12), False),
        ("x-1.0-cp312-cp311-manylinux_2_28_x86_64.whl", (3, 12), False),
        ("x-1.0-cp313-cp313t-manylinux_2_28_x86_64.whl", (3, 13), False),
        ("x-1.0-py3-cp312-any.whl", (3, 12), False),
        ("x-1.0-cp312-cp312-manylinux_2_29_x86_64.whl", (3, 12), False),
        ("x-1.0-cp312-cp312-manylinux2014_x86_64.whl", (3, 12), True),
        ("x-1.0-cp312-cp312-linux_x86_64.whl", (3, 12), True),
        ("x-1.0-cp312-cp312-manylinux_2_5_x86_64.whl", (3, 12), True),
        ("x-1.0-cp312-cp312-manylinux2010_x86_64.whl", (3, 12), True),
        ("x-1.0-cp312-cp312-manylinux1_x86_64.whl", (3, 12), True),
    ],
)
def test_the_wheel_rule_can_fail(wheel: str, python: tuple[int, int], installs: bool) -> None:
    accepted = _accepted_tags(python, "linux", "x86_64", 28)
    assert _wheel_installs_on(wheel, accepted) is installs


def test_the_wheel_rule_refuses_an_os_it_does_not_model() -> None:
    with pytest.raises(AssertionError):
        _accepted_tags((3, 12), "windows", "x86_64", 28)


def test_the_wheel_rule_refuses_an_arch_it_does_not_model() -> None:
    with pytest.raises(AssertionError):
        _accepted_tags((3, 12), "linux", "aarch64", 28)


def _instructions(dockerfile: str) -> list[str]:
    """The Dockerfile's instructions, one per entry, joined as Docker's parser
    joins continuations (default ``\\`` escape, no heredocs; an escape directive
    or a heredoc is refused rather than misread).

    Comment lines and blank lines are dropped first (Docker skips both inside a
    continuation, and a trailing backslash on a comment continues nothing), then
    each backslash continuation, trailing spaces or tabs allowed, is joined to
    the next line with nothing between them.
    """
    assert not re.search(r"(?im)^\s*#\s*escape\s*=", dockerfile), (
        "an escape directive is not modelled"
    )
    assert not re.search(r"<<-?\s*[\"']?\w", dockerfile), "a heredoc is not modelled"
    kept = "\n".join(
        line
        for line in dockerfile.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    joined = re.sub(r"\\[ \t]*\n", "", kept)
    return [line.strip() for line in joined.splitlines()]


def test_instructions_joins_continuations_and_skips_comments_as_docker_does() -> None:
    text = (
        "ENV A=1 \\  \n"
        "    # a comment inside the continuation\n"
        "\n"
        "    B=2\n"
        "# a comment ending in a backslash \\\n"
        "ARG C=3\n"
        "ARG IMAGEGEN_\\\n"
        "PYTHON=3\n"
    )
    assert _instructions(text) == ["ENV A=1     B=2", "ARG C=3", "ARG IMAGEGEN_PYTHON=3"]


@pytest.mark.parametrize("text", ["# escape=`\nARG A=1\n", "RUN <<EOF\necho hi\nEOF\n"])
def test_instructions_refuses_what_it_does_not_model(text: str) -> None:
    with pytest.raises(AssertionError):
        _instructions(text)


def test_the_image_interpreter_matches_the_manifest_target(manifest: dict) -> None:
    """The facade's Python version is stated in several files; they must agree.

    The Dockerfile's IMAGEGEN_PYTHON (one value, on the global ARG before the first
    FROM), the manifest's target.python, the runtime lock's --python-version, the
    CI imagegen job's interpreter and ruff's target-version name the same minor
    version; the lock's --python-platform names the manifest's arch; and every
    pinned wheel installs on that interpreter, OS, arch and glibc floor.
    """
    text = DOCKERFILE.read_text(encoding="utf-8")
    # Every ARG/ENV instruction naming the variable (not a ${IMAGEGEN_PYTHON} or
    # $IMAGEGEN_PYTHON use). Any of them but a bare stage re-declaration gives it
    # a value, in either ENV spelling.
    declared = [
        line
        for line in _instructions(text)
        if re.match(r"(?i)(ARG|ENV)\b", line) and re.search(r"(?<![{$])\bIMAGEGEN_PYTHON\b", line)
    ]
    valued = [d for d in declared if " ".join(d.split()).upper() != "ARG IMAGEGEN_PYTHON"]
    assert len(valued) == 1, f"only the global ARG may give IMAGEGEN_PYTHON a value: {valued}"
    pinned = re.search(r"^ARG IMAGEGEN_PYTHON=(\d+)\.(\d+)\.\d+$", text, re.M)
    assert pinned, "the Dockerfile must pin IMAGEGEN_PYTHON to a patch release"
    first_from = re.search(r"(?im)^\s*FROM\s", text)
    assert first_from and pinned.start() < first_from.start(), (
        "IMAGEGEN_PYTHON's one value must sit on the global ARG, before the first FROM"
    )
    want = (int(pinned.group(1)), int(pinned.group(2)))
    minor = f"{want[0]}.{want[1]}"

    target = manifest["target"]
    assert target["python"] == minor
    header = RUNTIME_LOCK.read_text(encoding="utf-8").split("\n--", 1)[0]
    assert re.search(rf"--python-version {re.escape(minor)}(\s|$)", header), header
    platform = re.search(r"--python-platform (\w+?)-manylinux_2_(\d+)\b", header)
    assert platform, f"the runtime lock names no manylinux --python-platform: {header}"
    assert platform.group(1) == target["arch"], header
    glibc = int(platform.group(2))

    ci = yaml.safe_load((DOCKER.parent / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    setup = next(
        step
        for step in ci["jobs"]["imagegen"]["steps"]
        if str(step.get("uses", "")).startswith("actions/setup-python")
    )
    assert str(setup["with"]["python-version"]) == minor
    ruff = tomllib.loads((IMAGEGEN_ROOT / "ruff.toml").read_text(encoding="utf-8"))
    assert ruff["target-version"] == f"py{want[0]}{want[1]}"

    accepted = _accepted_tags(want, target["os"], target["arch"], glibc)
    for library in (x for x in manifest["components"] if x["kind"] == "library"):
        for pinned_file in library["files"]:
            wheel = pinned_file["path"]
            assert _wheel_installs_on(wheel, accepted), wheel


def test_uv_is_pinned_to_a_version_and_checksum() -> None:
    """The uv that installs the facade interpreter is pinned, downloaded at that
    version, and checked against its pinned sha256."""
    _, alias, rest = DOCKERFILE.read_text(encoding="utf-8").partition("AS imagegen-python")
    assert alias, "the Dockerfile has no imagegen-python stage"
    uv_stage, stage4, _ = rest.partition("# --- Stage 4")
    assert stage4, "the imagegen-python stage is not followed by Stage 4"
    assert re.search(r"^ARG UV_VERSION=\d+\.\d+\.\d+$", uv_stage, re.M), "UV_VERSION is not pinned"
    assert re.search(r"^ARG UV_SHA256=[0-9a-f]{64}$", uv_stage, re.M), "UV_SHA256 is not a sha256"
    assert 'echo "${UV_SHA256}  /tmp/uv.tar.gz" | sha256sum -c -' in uv_stage, (
        "the uv tarball is not checked against UV_SHA256"
    )
    # The whole file goes through _instructions once, so its escape-directive
    # and heredoc refusals see the top of the file; the stage-wide guards below
    # read the stage's joined instructions, so a keyword split across a
    # continuation or named in a comment is read as Docker reads it.
    _instructions(DOCKERFILE.read_text(encoding="utf-8"))
    instructions = _instructions(uv_stage)
    stage = "\n".join(instructions)
    assert "latest" not in stage, "the uv stage must not download 'latest'"
    # Textual guards, not a shell parse: the stage runs `uv python install` once,
    # in one RUN where the download, the sha256 check, the extraction and the
    # extracted uv's install are chained directly by `&&`, in that order, so a
    # failed check stops the chain; curl is named once in the apt install and
    # once in that RUN, never wget; the stage extracts once and has no ADD/COPY.
    assert stage.count("uv python install") == 1, "expected one `uv python install`"
    runs = [i for i in instructions if "uv python install" in i]
    assert len(runs) == 1, f"expected one RUN that installs the interpreter: {runs}"
    chain = (
        r'curl -fsSL -o /tmp/uv\.tar\.gz\s+"([^"]*)"\s*&&\s*'
        r'echo "\$\{UV_SHA256\}  /tmp/uv\.tar\.gz" \| sha256sum -c - &&\s*'
        r"tar xzf /tmp/uv\.tar\.gz -C /tmp &&\s*"
        r"UV_NO_CACHE=1 /tmp/uv-x86_64-unknown-linux-gnu/uv python install\b"
    )
    linked = re.search(chain, runs[0])
    assert linked, f"the uv download/check/extract/install chain is broken: {runs[0]}"
    # The URL curl actually downloads is exactly the UV_VERSION release asset.
    assert linked.group(1) == (
        "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/"
        "uv-x86_64-unknown-linux-gnu.tar.gz"
    ), f"the download ignores UV_VERSION: {linked.group(1)}"
    apt = [i for i in instructions if "apt-get install" in i]
    assert len(apt) == 1 and len(re.findall(r"\bcurl\b", apt[0])) == 1, apt
    assert len(re.findall(r"\bcurl\b", runs[0])) == 1, "a second curl in the install RUN"
    assert len(re.findall(r"\bcurl\b", stage)) == 2, (
        "curl named outside the apt install and the download"
    )
    assert not re.search(r"\bwget\b", stage), "the uv stage uses wget"
    assert len(re.findall(r"\btar\s", stage)) == 1, "a second extraction in the uv stage"
    assert not re.search(r"(?im)^(ADD|COPY)\b", stage), "the uv stage copies files in"
