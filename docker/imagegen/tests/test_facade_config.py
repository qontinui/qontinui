"""facade.yaml, the llama-swap entry, the runtime lock and the manifest agree (plan Phase 10).

Four files describe one deployment; each test here fails when one of them drifts from the
others, so a drift is a red CI run rather than a facade that refuses an alias llama-swap
routes to it, or an image that installs a library the licence manifest never admitted.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest
import yaml
from facade_fakes import real_catalogue

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
    from datetime import datetime, timezone

    from qontinui_imagegen.admission import AdmissionController, zone_clock

    config = c.parse_config(document, environ={c.TIMEZONE_ENV: "Asia/Tokyo"}, **_known(manifest))
    assert config.admission.bulk_window_timezone == "Asia/Tokyo"
    # 17:30 UTC is 02:30 in Tokyo: inside 00:00-06:00 there, outside it in UTC.
    instant = datetime(2026, 10, 17, 17, 30, tzinfo=timezone.utc)
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
