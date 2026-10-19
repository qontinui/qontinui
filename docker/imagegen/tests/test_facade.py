"""The image facade against a fake pipeline executor (plan Phase 10): no GPU, no weights.

Covers admission (one job on the card, one interactive waiter), both refusals (``busy``,
``deferred``) and the machine switch's ``disabled``, SSE event order, provenance
completeness, the weight-mismatch refusal, the out-of-band VRAM check, and the
AI-generated marker on every returned image.
"""

from __future__ import annotations

import asyncio
import base64
import io
import math
import re
import threading

import pytest
from facade_fakes import (
    DAYTIME,
    NIGHT,
    FakeEngine,
    FakeVerifier,
    client,
    make_facade,
    run,
    sse_events,
    wait_for,
)
from PIL import Image

from qontinui_imagegen import marker, provenance
from qontinui_imagegen.admission import Ticket
from qontinui_imagegen.engine import HostMemory
from qontinui_imagegen.jobs import INPUT_ROLES
from qontinui_imagegen.manifest import RECIPES_DIR, load_recipes


def _png(width: int = 16, height: int = 12) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 10, 10)).save(buffer, format="PNG")
    return buffer.getvalue()


async def _post(facade, path: str = "/v1/images/generations", **body):
    async with client(facade) as http:
        return await http.post(path, json=body)


# --- a generation, end to end ------------------------------------------------------------


def test_a_generation_returns_a_marked_png_with_complete_provenance() -> None:
    facade = make_facade()
    response = run(_post(facade, model="z-image-turbo", prompt="a lighthouse", seed=7))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["output_format"] == "png" and body["size"] == "1024x1024"
    (item,) = body["data"]
    data = base64.b64decode(item["b64_json"])
    assert marker.has_marker(data, "image/png")
    record = item["qontinui_provenance"]
    for key in provenance.REQUIRED_KEYS:
        assert key in record, key
    assert record["recipe"]["id"] == "draft"
    assert record["recipe"]["hash"] == facade.catalogue.recipe_hashes["draft"]
    assert record["seed"] == 7 and record["prompt"] == "a lighthouse"
    assert record["licence"]["spdx"] == "Apache-2.0"
    (model,) = record["models"]
    assert model["id"] == "z-image-turbo"
    pinned = {entry["sha256"] for entry in facade.catalogue.components["z-image-turbo"]["files"]}
    assert {weight["sha256"] for weight in model["weights"]} == pinned
    assert record["engine"]["facade"] and record["engine"]["diffusers"] == "0.41.0"
    assert record["marker"]["iptc_digital_source_type"].endswith("trainedAlgorithmicMedia")


def test_every_candidate_is_marked_and_seeded_from_the_base() -> None:
    facade = make_facade()
    response = run(_post(facade, model="z-image-turbo", prompt="x", n=3, seed=40))
    items = response.json()["data"]
    assert [item["qontinui_provenance"]["seed"] for item in items] == [40, 41, 42]
    assert [item["qontinui_provenance"]["candidate_index"] for item in items] == [0, 1, 2]
    assert all(marker.has_marker(base64.b64decode(i["b64_json"]), "image/png") for i in items)


def test_an_unseeded_request_records_the_seed_it_drew() -> None:
    facade = make_facade()
    item = run(_post(facade, model="z-image-turbo", prompt="x")).json()["data"][0]
    assert item["qontinui_provenance"]["seed"] == 1234


def test_a_post_chain_ending_in_vectorize_returns_a_marked_svg() -> None:
    facade = make_facade()
    response = run(_post(facade, model="z-image-turbo", recipe="empty-state", prompt="a box"))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["output_format"] == "svg"
    item = body["data"][0]
    assert item["qontinui_media_type"] == "image/svg+xml"
    assert marker.has_marker(base64.b64decode(item["b64_json"]), "image/svg+xml")
    models = [model["id"] for model in item["qontinui_provenance"]["models"]]
    assert models == ["z-image-turbo", "birefnet"]
    assert item["qontinui_provenance"]["licence"]["spdx"] == "Apache-2.0 AND MIT"


def test_flux_schnell_is_served_by_its_own_alias() -> None:
    facade = make_facade()
    response = run(_post(facade, model="flux-1-schnell", prompt="a fox"))
    assert response.status_code == 200, response.text
    record = response.json()["data"][0]["qontinui_provenance"]
    assert record["recipe"]["id"] == "draft-flux"
    assert record["models"][0]["source"]["repo"] == "black-forest-labs/FLUX.1-schnell"


# --- the marker's test can fail ------------------------------------------------------------


def test_the_marker_check_fails_on_a_png_without_the_marker() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="PNG")
    assert not marker.has_marker(buffer.getvalue(), "image/png")
    assert marker.has_marker(marker.encode_png(Image.new("RGB", (4, 4))), "image/png")
    assert not marker.has_marker(FAKE_SVG_BYTES, "image/svg+xml")


FAKE_SVG_BYTES = b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'


# --- edits -------------------------------------------------------------------------------


def test_an_edit_takes_multipart_images_and_records_their_digests() -> None:
    facade = make_facade()
    image = _png()

    async def go():
        async with client(facade) as http:
            return await http.post(
                "/v1/images/edits",
                data={"model": "qwen-image-edit-2511", "prompt": "make it blue", "seed": "3"},
                files=[("image[]", ("a.png", image, "application/octet-stream"))],
            )

    response = run(go())
    assert response.status_code == 200, response.text
    record = response.json()["data"][0]["qontinui_provenance"]
    assert record["recipe"]["id"] == "edit"
    assert record["references"][0]["sha256"] == __import__("hashlib").sha256(image).hexdigest()
    assert record["negative_prompt"] == " "  # the recipe's own, from its model card


def test_an_edit_refuses_images_split_across_field_names() -> None:
    facade = make_facade()

    async def go():
        async with client(facade) as http:
            return await http.post(
                "/v1/images/edits",
                data={"model": "qwen-image-edit-2511", "prompt": "p"},
                files=[
                    ("image", ("a.png", _png(), "application/octet-stream")),
                    ("image[]", ("b.png", _png(), "application/octet-stream")),
                ],
            )

    response = run(go())
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "ambiguous_image_order"


def test_upscale_runs_on_the_edits_route_with_no_seed() -> None:
    facade = make_facade()

    async def go():
        async with client(facade) as http:
            return await http.post(
                "/v1/images/edits",
                data={"model": "real-esrgan-general-x4v3", "prompt": "unused"},
                files=[("image", ("a.png", _png(), "application/octet-stream"))],
            )

    response = run(go())
    assert response.status_code == 200, response.text
    record = response.json()["data"][0]["qontinui_provenance"]
    assert record["recipe"]["id"] == "upscale" and record["seed"] is None
    assert record["prompt_used"] is False
    assert facade.engine.runs[-1].call_parameters["scale"] == 4


def test_edits_require_multipart() -> None:
    facade = make_facade()
    response = run(_post(facade, "/v1/images/edits", model="qwen-image-edit-2511", prompt="p"))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "multipart_required"


# --- typed 400s instead of silent drops --------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"model": "dall-e-3", "prompt": "x"}, "model_not_found"),
        ({"model": "z-image-turbo"}, "missing_parameter"),
        ({"model": "z-image-turbo", "prompt": "x", "size": "512x512"}, "unsupported_size"),
        ({"model": "z-image-turbo", "prompt": "x", "response_format": "url"}, "unsupported_value"),
        ({"model": "z-image-turbo", "prompt": "x", "style": "vivid"}, "unknown_parameter"),
        (
            {"model": "z-image-turbo", "prompt": "x", "negative_prompt": "blur"},
            "negative_prompt_unsupported",
        ),
        ({"model": "z-image-turbo", "prompt": "x", "recipe": "hero"}, "recipe_model_mismatch"),
        ({"model": "z-image-turbo", "prompt": "x", "n": 5}, "interactive_cap_exceeded"),
        (
            {"model": "z-image-turbo", "prompt": "x", "n": 2, "stream": True},
            "stream_needs_one_image",
        ),
        ({"model": "qwen-image-edit-2511", "prompt": "x"}, "wrong_route"),
        (
            {"model": "z-image-turbo", "prompt": "x", "output_format": "webp"},
            "unsupported_output_format",
        ),
    ],
)
def test_requests_the_facade_would_not_honour_are_refused(body: dict, code: str) -> None:
    facade = make_facade()
    response = run(_post(facade, **body))
    assert response.status_code == 400, response.text
    assert response.json()["error"] == {
        **response.json()["error"],
        "type": "invalid_request_error",
        "code": code,
    }
    assert facade.engine.runs == []


def test_a_negative_prompt_reaches_a_recipe_that_uses_one() -> None:
    facade = make_facade()
    response = run(_post(facade, model="qwen-image-2512", prompt="x", negative_prompt="blur"))
    assert response.status_code == 200
    assert response.json()["data"][0]["qontinui_provenance"]["negative_prompt"] == "blur"


# --- the machine switch (plan Phase 12 step 4) ----------------------------------------------------


@pytest.mark.parametrize(
    ("value", "code"), [("0", "machine_switch_off"), ("off", "machine_switch_unrecognised")]
)
def test_the_machine_switch_answers_disabled(value: str, code: str) -> None:
    facade = make_facade(environ={"QONTINUI_IMAGEGEN": value})
    response = run(_post(facade, model="z-image-turbo", prompt="x"))
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["type"] == "disabled" and error["code"] == code
    assert error["retry_after_s"] is None and "retry-after" not in response.headers
    assert facade.engine.runs == []


def test_without_a_loopback_attestation_the_facade_answers_disabled() -> None:
    """Rule R-bind read as the off-box arm: no attested loopback listener, no image work."""
    facade = make_facade(attested=False)
    response = run(_post(facade, model="z-image-turbo", prompt="x"))
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["type"] == "disabled" and error["code"] == "listener_not_attested"
    assert facade.engine.runs == []
    wrong = make_facade(attested=False, environ={"QONTINUI_IMAGEGEN_LISTENER": "tailnet"})
    assert run(_post(wrong, model="z-image-turbo", prompt="x")).json()["error"]["code"] == (
        "listener_not_attested"
    )


def test_the_machine_switch_off_wins_over_a_missing_attestation() -> None:
    facade = make_facade(attested=False, environ={"QONTINUI_IMAGEGEN": "0"})
    response = run(_post(facade, model="z-image-turbo", prompt="x"))
    assert response.json()["error"]["code"] == "machine_switch_off"


@pytest.mark.parametrize("value", [None, "1"])
def test_the_machine_switch_on_serves(value: str | None) -> None:
    environ = {} if value is None else {"QONTINUI_IMAGEGEN": value}
    facade = make_facade(environ=environ)
    assert run(_post(facade, model="z-image-turbo", prompt="x")).status_code == 200


# --- admission --------------------------------------------------------------------------------


def test_bulk_outside_the_window_is_deferred_with_retry_after() -> None:
    facade = make_facade(now=DAYTIME)
    response = run(_post(facade, model="z-image-turbo", prompt="x", priority="bulk"))
    assert response.status_code == 429
    error = response.json()["error"]
    assert error["type"] == "deferred" and error["code"] == "outside_bulk_window"
    # 14:00 -> the window opens at 00:00: ten hours.
    assert error["retry_after_s"] == 10 * 3600
    assert response.headers["retry-after"] == str(10 * 3600)
    assert facade.engine.runs == []


def test_bulk_inside_the_window_runs() -> None:
    facade = make_facade(now=NIGHT)
    response = run(_post(facade, model="z-image-turbo", prompt="x", priority="bulk", n=6))
    assert response.status_code == 200
    assert len(response.json()["data"]) == 6


def test_one_job_runs_one_interactive_request_waits_and_the_next_is_busy() -> None:
    gate = threading.Event()
    engine = FakeEngine(gate=gate)
    facade = make_facade(engine=engine, now=NIGHT)

    async def go():
        async with client(facade) as http:
            first = asyncio.ensure_future(
                http.post("/v1/images/generations", json={"model": "z-image-turbo", "prompt": "a"})
            )
            await wait_for(engine.started.is_set)
            second = asyncio.ensure_future(
                http.post("/v1/images/generations", json={"model": "z-image-turbo", "prompt": "b"})
            )
            await wait_for(lambda: facade.admission.snapshot()["waiting"])
            third = await http.post(
                "/v1/images/generations", json={"model": "z-image-turbo", "prompt": "c"}
            )
            bulk = await http.post(
                "/v1/images/generations",
                json={"model": "z-image-turbo", "prompt": "d", "priority": "bulk"},
            )
            gate.set()
            return await first, await second, third, bulk

    first, second, third, bulk = run(go())
    assert first.status_code == 200 and second.status_code == 200
    for refused in (third, bulk):
        assert refused.status_code == 503
        error = refused.json()["error"]
        assert error["type"] == "busy" and error["code"] == "card_busy"
        assert error["retry_after_s"] >= 1 and int(refused.headers["retry-after"]) >= 1
    assert [plan.prompt for plan in engine.runs] == ["a", "b"]


def test_bulk_never_waits_behind_a_running_job() -> None:
    gate = threading.Event()
    engine = FakeEngine(gate=gate)
    facade = make_facade(engine=engine, now=NIGHT)

    async def go():
        async with client(facade) as http:
            first = asyncio.ensure_future(
                http.post(
                    "/v1/images/generations",
                    json={"model": "z-image-turbo", "prompt": "a", "priority": "bulk"},
                )
            )
            await wait_for(engine.started.is_set)
            second = await http.post(
                "/v1/images/generations",
                json={"model": "z-image-turbo", "prompt": "b", "priority": "bulk"},
            )
            gate.set()
            return await first, second

    first, second = run(go())
    assert first.status_code == 200
    assert second.status_code == 503 and second.json()["error"]["type"] == "busy"


def test_the_card_frees_after_a_failed_job() -> None:
    engine = FakeEngine(out_of_memory=True)
    facade = make_facade(engine=engine)
    response = run(_post(facade, model="z-image-turbo", prompt="x"))
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "out_of_memory"
    assert facade.admission.snapshot()["running"] is None
    engine.out_of_memory = False
    assert run(_post(facade, model="z-image-turbo", prompt="x")).status_code == 200


def test_a_job_past_the_interactive_cap_is_stopped() -> None:
    clock = {"now": 0.0}
    gate = threading.Event()
    engine = FakeEngine(gate=gate)
    facade = make_facade(engine=engine, monotonic=lambda: clock["now"])

    async def go():
        async with client(facade) as http:
            pending = asyncio.ensure_future(
                http.post("/v1/images/generations", json={"model": "z-image-turbo", "prompt": "a"})
            )
            await wait_for(engine.started.is_set)
            clock["now"] = 121.0  # past the 120 s interactive cap
            return await pending

    response = run(go())
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "interactive_gpu_seconds_cap"


def test_a_cancelled_ticket_stops_its_job_as_client_gone() -> None:
    facade = make_facade()
    plan_response = run(_post(facade, model="z-image-turbo", prompt="warm"))
    assert plan_response.status_code == 200
    from qontinui_imagegen.jobs import build_plan

    plan = build_plan(facade.catalogue, "generations", {"model": "z-image-turbo", "prompt": "x"})
    ticket = Ticket("interactive", 120.0, started_at=0.0)
    ticket.cancelled.set()
    from qontinui_imagegen.errors import FacadeError

    with pytest.raises(FacadeError) as raised:
        facade.execute(plan, ticket, lambda step, total: None)
    assert raised.value.code == "client_gone"


# --- streaming ------------------------------------------------------------------------------------


def test_a_stream_reports_running_steps_then_the_completed_image() -> None:
    facade = make_facade()
    response = run(_post(facade, model="z-image-turbo", prompt="x", stream=True))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response.text)
    names = [name for name, _ in events]
    assert names == ["qontinui.running"] * 9 + ["image_generation.completed"]
    assert [data["step"] for _, data in events[:-1]] == list(range(1, 10))
    assert all(data["total"] == 9 for _, data in events[:-1])
    final = events[-1][1]
    assert final["type"] == "image_generation.completed"
    assert marker.has_marker(base64.b64decode(final["b64_json"]), "image/png")
    assert set(provenance.REQUIRED_KEYS) <= set(final["qontinui_provenance"])


def test_a_queued_stream_says_so_first() -> None:
    gate = threading.Event()
    engine = FakeEngine(gate=gate)
    facade = make_facade(engine=engine)

    async def go():
        async with client(facade) as http:
            first = asyncio.ensure_future(
                http.post("/v1/images/generations", json={"model": "z-image-turbo", "prompt": "a"})
            )
            await wait_for(engine.started.is_set)
            second = asyncio.ensure_future(
                http.post(
                    "/v1/images/generations",
                    json={"model": "z-image-turbo", "prompt": "b", "stream": True},
                )
            )
            await wait_for(lambda: facade.admission.snapshot()["waiting"])
            gate.set()
            return await first, await second

    _, second = run(go())
    names = [name for name, _ in sse_events(second.text)]
    assert names[0] == "qontinui.queued"
    assert names[1:] == ["qontinui.running"] * 9 + ["image_generation.completed"]


def test_an_edit_stream_completes_under_the_edit_event_name() -> None:
    facade = make_facade()

    async def go():
        async with client(facade) as http:
            return await http.post(
                "/v1/images/edits",
                data={"model": "qwen-image-edit-2511", "prompt": "p", "stream": "true"},
                files=[("image", ("a.png", _png(), "application/octet-stream"))],
            )

    events = sse_events(run(go()).text)
    assert events[-1][0] == "image_edit.completed"


def test_a_failure_inside_a_stream_is_an_error_event() -> None:
    facade = make_facade(engine=FakeEngine(out_of_memory=True))
    events = sse_events(run(_post(facade, model="z-image-turbo", prompt="x", stream=True)).text)
    assert events[-1][0] == "error"
    assert events[-1][1]["error"]["code"] == "out_of_memory"


# --- weights and VRAM ---------------------------------------------------------------------------


def test_unverified_weights_are_never_served() -> None:
    verifier = FakeVerifier({"z-image-turbo": ["vae/x.safetensors hashes to 00, pinned ff"]})
    facade = make_facade(verifier=verifier)
    response = run(_post(facade, model="z-image-turbo", prompt="x"))
    assert response.status_code == 500
    error = response.json()["error"]
    assert error["type"] == "failed" and error["code"] == "weights_unverified"
    assert error["model"] == "z-image-turbo"
    assert facade.engine.runs == []


def test_vram_held_out_of_band_answers_busy_after_releasing_our_own() -> None:
    engine = FakeEngine(free_mib=6_000)
    facade = make_facade(engine=engine)
    response = run(_post(facade, model="z-image-turbo", prompt="x"))
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["type"] == "busy" and error["code"] == "vram_held_by"
    assert error["vram_held_by"] == "out_of_band"
    need = facade.config.models["z-image-turbo"].vram_need_mib + 2048
    assert error["need_mib"] == need and error["free_mib"] == 6_000
    assert engine.releases == 1 and engine.runs == []


def test_a_resident_pipeline_skips_the_vram_check() -> None:
    engine = FakeEngine()
    facade = make_facade(engine=engine)
    assert run(_post(facade, model="z-image-turbo", prompt="x")).status_code == 200
    engine.free_mib = 0  # the facade's own pipeline holds the card now
    assert run(_post(facade, model="z-image-turbo", prompt="y")).status_code == 200
    assert engine.releases == 1


def test_unreadable_vram_is_a_typed_failure_not_a_guess() -> None:
    facade = make_facade(engine=FakeEngine(free_mib=None))
    response = run(_post(facade, model="z-image-turbo", prompt="x"))
    assert response.status_code == 500 and response.json()["error"]["code"] == "vram_unreadable"


# --- health and status ------------------------------------------------------------------------------


def test_health_is_non_200_with_progress_while_weights_verify() -> None:
    verifier = FakeVerifier()
    facade = make_facade(verifier=verifier)

    async def go():
        async with client(facade) as http:
            ok = await http.get("/health")
            verifier.progress.active = 1
            verifier.progress.total_bytes = 200
            verifier.progress.done_bytes = 50
            busy = await http.get("/health")
            return ok, busy

    ok, busy = run(go())
    assert ok.status_code == 200
    assert busy.status_code == 503 and busy.json()["progress"] == 0.25


def test_status_reports_switch_card_and_aliases() -> None:
    facade = make_facade(environ={"QONTINUI_IMAGEGEN": "0"})

    async def go():
        async with client(facade) as http:
            return await http.get("/qontinui/status")

    status = run(go()).json()
    assert status["switch"] == {"enabled": False, "raw": "0", "recognised": True}
    assert status["listener"] == "loopback"
    assert "flux-1-schnell" in status["aliases"] and status["admission"]["running"] is None


def _status(facade) -> dict:
    async def go():
        async with client(facade) as http:
            return await http.get("/qontinui/status")

    response = run(go())
    assert response.status_code == 200, response.text
    return response.json()


def _generate_with_recipe(facade, recipe_id: str) -> dict:
    """Run one job under ``recipe_id`` (bulk, so no interactive cap) and return its provenance."""
    recipe = load_recipes(RECIPES_DIR)[recipe_id]
    alias = recipe["models"][0]
    role = facade.catalogue.components[alias]["role"]

    async def go():
        async with client(facade) as http:
            if role in INPUT_ROLES:
                return await http.post(
                    "/v1/images/edits",
                    data={"model": alias, "recipe": recipe_id, "prompt": "x", "priority": "bulk"},
                    files=[("image", ("a.png", _png(), "application/octet-stream"))],
                )
            return await http.post(
                "/v1/images/generations",
                json={"model": alias, "recipe": recipe_id, "prompt": "x", "priority": "bulk"},
            )

    response = run(go())
    assert response.status_code == 200, (recipe_id, response.text)
    record = response.json()["data"][0]["qontinui_provenance"]
    assert record["recipe"]["id"] == recipe_id
    return record


def test_status_serves_a_hash_for_every_loaded_recipe() -> None:
    """Every recipe on disk, read independently of the facade -- not only the aliases' ones."""
    on_disk = set(load_recipes(RECIPES_DIR))
    facade = make_facade()
    request_only = on_disk - set(facade.catalogue.config.aliases.values())
    assert request_only, "no recipe is reachable only by request: this test would not cover one"
    served = _status(facade)["recipes"]
    assert set(served) == on_disk
    assert all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in served.values())


def test_status_recipe_hashes_are_the_ones_provenance_stamps() -> None:
    """Studio compares like with like: the served hash is what a result's provenance carries."""
    facade = make_facade(now=NIGHT)
    served = _status(facade)["recipes"]
    for recipe_id in sorted(load_recipes(RECIPES_DIR)):
        record = _generate_with_recipe(facade, recipe_id)
        assert served[recipe_id] == record["recipe"]["hash"], recipe_id
    # Stable: a second read, after jobs ran, serves the same mapping.
    assert _status(facade)["recipes"] == served


# --- review round 2 --------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True])
def test_an_unexpected_engine_failure_answers_and_frees_the_card(stream: bool) -> None:
    """H1: an exception that is not a typed one still answers; it never hangs the request."""
    engine = FakeEngine(crash=RuntimeError("driver fell over"))
    facade = make_facade(engine=engine)

    async def go():
        async with client(facade) as http:
            return await asyncio.wait_for(
                http.post(
                    "/v1/images/generations",
                    json={"model": "z-image-turbo", "prompt": "x", "stream": stream},
                ),
                timeout=10,
            )

    response = run(go())
    if stream:
        name, data = sse_events(response.text)[-1]
        assert name == "error"
        error = data["error"]
    else:
        assert response.status_code == 500
        error = response.json()["error"]
    assert error["type"] == "failed" and error["code"] == "engine_error"
    assert "driver fell over" in error["message"]
    assert facade.admission.snapshot()["running"] is None
    assert engine.releases >= 2  # the VRAM check's release, then the failure's
    engine.crash = None
    assert run(_post(facade, model="z-image-turbo", prompt="y")).status_code == 200


def test_a_cold_load_does_not_count_against_the_interactive_cap() -> None:
    """H2: the cap's clock starts when the pipeline is loaded, not at admission."""
    clock = {"now": 0.0}

    def cold_load() -> None:
        clock["now"] += 220.0  # Phase 0: a cold FP8 Z-Image-Turbo load took ~214-220 s

    facade = make_facade(engine=FakeEngine(load=cold_load), monotonic=lambda: clock["now"])
    response = run(_post(facade, model="z-image-turbo", prompt="x"))
    assert response.status_code == 200, response.text


def test_the_cap_still_stops_compute_that_runs_long_after_the_load() -> None:
    clock = {"now": 0.0}
    gate = threading.Event()
    engine = FakeEngine(gate=gate, load=lambda: clock.__setitem__("now", clock["now"] + 220.0))
    facade = make_facade(engine=engine, monotonic=lambda: clock["now"])

    async def go():
        async with client(facade) as http:
            pending = asyncio.ensure_future(
                http.post("/v1/images/generations", json={"model": "z-image-turbo", "prompt": "a"})
            )
            await wait_for(engine.started.is_set)
            clock["now"] = 220.0 + 121.0
            return await pending

    response = run(go())
    assert response.json()["error"]["code"] == "interactive_gpu_seconds_cap"


def test_a_deterministic_recipe_refuses_more_than_one_image() -> None:
    """M5: the upscaler renders one image; n>1 is refused, not silently dropped."""
    facade = make_facade()

    async def go(n: str):
        async with client(facade) as http:
            return await http.post(
                "/v1/images/edits",
                data={"model": "real-esrgan-x2plus", "prompt": "unused", "n": n},
                files=[("image", ("a.png", _png(), "application/octet-stream"))],
            )

    refused = run(go("2"))
    assert refused.status_code == 400 and refused.json()["error"]["code"] == "n_unsupported"
    assert run(go("1")).status_code == 200


def test_the_interactive_pixel_cap_counts_an_upscales_output() -> None:
    """L6: 800x800 is under the 2,359,296 px cap, but 4x it is 10,240,000 px."""
    facade = make_facade()

    async def go(model: str, size: int):
        async with client(facade) as http:
            return await http.post(
                "/v1/images/edits",
                data={"model": model, "prompt": "unused"},
                files=[("image", ("a.png", _png(size, size), "application/octet-stream"))],
            )

    refused = run(go("real-esrgan-general-x4v3", 800))
    assert refused.status_code == 400
    assert refused.json()["error"]["code"] == "interactive_cap_exceeded"
    assert run(go("real-esrgan-x2plus", 700)).status_code == 200  # 1400x1400 = 1,960,000 px


def test_a_non_stream_client_that_leaves_while_queued_gives_back_its_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L2: a disconnected waiter frees the waiting place instead of holding it."""
    import qontinui_imagegen.app as app_module

    monkeypatch.setattr(app_module, "DISCONNECT_POLL_S", 0.02)
    gate = threading.Event()
    engine = FakeEngine(gate=gate)
    facade = make_facade(engine=engine)

    async def go():
        async with client(facade) as http:
            first = asyncio.ensure_future(
                http.post("/v1/images/generations", json={"model": "z-image-turbo", "prompt": "a"})
            )
            await wait_for(engine.started.is_set)

            async def gone() -> bool:
                return True

            second = await facade.respond(
                "generations", {"model": "z-image-turbo", "prompt": "b"}, (), gone
            )
            waiting = facade.admission.snapshot()["waiting"]
            third = asyncio.ensure_future(
                http.post("/v1/images/generations", json={"model": "z-image-turbo", "prompt": "c"})
            )
            await wait_for(lambda: facade.admission.snapshot()["waiting"])
            gate.set()
            return await first, second, waiting, await third

    first, second, waiting_after_leave, third = run(go())
    assert first.status_code == 200 and third.status_code == 200
    assert second.status_code == 500 and b"client_gone" in second.body
    assert waiting_after_leave is False
    assert [plan.prompt for plan in engine.runs] == ["a", "c"]


def test_a_running_client_that_leaves_stops_its_job(monkeypatch: pytest.MonkeyPatch) -> None:
    import qontinui_imagegen.app as app_module

    monkeypatch.setattr(app_module, "DISCONNECT_POLL_S", 0.02)
    gate = threading.Event()
    engine = FakeEngine(gate=gate)
    facade = make_facade(engine=engine)

    async def go():
        async def gone() -> bool:
            return engine.started.is_set()

        response = await facade.respond(
            "generations", {"model": "z-image-turbo", "prompt": "a"}, (), gone
        )
        await wait_for(lambda: facade.admission.snapshot()["running"] is None)
        return response

    response = run(go())
    assert b"client_gone" in response.body


# --- review round 3 --------------------------------------------------------------------------


def test_bulk_with_no_zone_is_deferred_with_a_finite_retry_after() -> None:
    """R3-2: the shipped default (no zone) still answers with a retry time and its header."""
    facade = make_facade(now=None)  # no zone: the box clock cannot be read
    response = run(_post(facade, model="z-image-turbo", prompt="x", priority="bulk"))
    assert response.status_code == 429
    error = response.json()["error"]
    assert error["type"] == "deferred" and error["code"] == "bulk_window_timezone_unset"
    # The window's own length (00:00-06:00): a client re-asking that often cannot step
    # over a whole window once the zone is set.
    assert error["retry_after_s"] == 6 * 3600
    assert response.headers["retry-after"] == str(6 * 3600)
    assert facade.engine.runs == []
    # Interactive work needs no window and is served.
    assert run(_post(facade, model="z-image-turbo", prompt="x")).status_code == 200


def test_the_interactive_pixel_cap_counts_an_upscale_post_step() -> None:
    """R3-3: hero renders 1664x928 (1.54 MP) and upscales it 2x: 6.18 MP is past the cap."""
    facade = make_facade(now=NIGHT)
    refused = run(_post(facade, model="qwen-image-2512", prompt="x", recipe="hero"))
    assert refused.status_code == 400
    error = refused.json()["error"]
    assert error["code"] == "interactive_cap_exceeded" and "6176768" in error["message"]
    # The same render without the upscale is under the cap, and bulk work is not capped.
    assert run(_post(facade, model="qwen-image-2512", prompt="x")).status_code == 200
    bulk = run(_post(facade, model="qwen-image-2512", prompt="x", recipe="hero", priority="bulk"))
    assert bulk.status_code == 200


def test_a_stream_client_that_leaves_while_queued_gives_back_its_place() -> None:
    """R3-4: the stream path frees the waiting place the moment its client stops reading."""
    gate = threading.Event()
    engine = FakeEngine(gate=gate)
    facade = make_facade(engine=engine)

    async def go():
        async with client(facade) as http:
            first = asyncio.ensure_future(
                http.post("/v1/images/generations", json={"model": "z-image-turbo", "prompt": "a"})
            )
            await wait_for(engine.started.is_set)
            streaming = await facade.respond(
                "generations", {"model": "z-image-turbo", "prompt": "b", "stream": True}, ()
            )
            body = streaming.body_iterator
            queued = await body.__anext__()  # the client reads the queued event...
            await body.aclose()  # ...and goes away
            waiting_after_leave = facade.admission.snapshot()["waiting"]
            third = asyncio.ensure_future(
                http.post("/v1/images/generations", json={"model": "z-image-turbo", "prompt": "c"})
            )
            await wait_for(lambda: facade.admission.snapshot()["waiting"])
            gate.set()
            return queued, waiting_after_leave, await first, await third

    queued, waiting_after_leave, first, third = run(go())
    assert queued.startswith(b"event: qontinui.queued")
    assert waiting_after_leave is False
    assert first.status_code == 200 and third.status_code == 200
    assert [plan.prompt for plan in engine.runs] == ["a", "c"]


def test_busy_while_loading_counts_the_expected_load_then_says_unknown() -> None:
    """R3-5: a loading job's end is its remaining expected load plus the cap, not one cap."""
    clock = {"now": 0.0}
    loading = threading.Event()
    engine = FakeEngine(load=lambda: loading.wait(5))
    facade = make_facade(engine=engine, now=NIGHT, monotonic=lambda: clock["now"])
    from qontinui_imagegen.jobs import build_plan

    plan = build_plan(facade.catalogue, "generations", {"model": "z-image-turbo", "prompt": "a"})
    expected = facade.catalogue.expected_load_s(plan, cold=True)
    bulk = {"model": "z-image-turbo", "prompt": "b", "priority": "bulk"}

    async def go():
        async with client(facade) as http:
            first = asyncio.ensure_future(
                http.post("/v1/images/generations", json={"model": "z-image-turbo", "prompt": "a"})
            )
            await wait_for(lambda: bool(engine.runs))  # its load has begun
            clock["now"] = 10.0
            during = await http.post("/v1/images/generations", json=bulk)
            status_during = (await http.get("/qontinui/status")).json()["admission"]["running"]
            clock["now"] = expected + 50.0  # the load overran its estimate
            overrun = await http.post("/v1/images/generations", json=bulk)
            status_overrun = (await http.get("/qontinui/status")).json()["admission"]["running"]
            loading.set()
            return during, status_during, overrun, status_overrun, await first

    during, status_during, overrun, status_overrun, first = run(go())
    assert first.status_code == 200
    error = during.json()["error"]
    assert error["code"] == "card_busy" and error["retry_after_basis"] == "estimated_end"
    assert error["retry_after_s"] == math.ceil(expected - 10.0 + 120.0)
    assert during.headers["retry-after"] == str(math.ceil(expected - 10.0 + 120.0))
    assert status_during["phase"] == "loading" and status_during["deadline_in_s"] is None
    assert status_during["expected_end_in_s"] == pytest.approx(expected - 10.0 + 120.0)
    # Past the estimate nothing honest says when the load ends: unknown, and busy's
    # Retry-After is one cap as a re-ask interval, labelled as such.
    error = overrun.json()["error"]
    assert error["retry_after_basis"] == "reask_interval" and error["retry_after_s"] == 120
    assert status_overrun["phase"] == "loading" and status_overrun["expected_end_in_s"] is None


def test_the_load_estimate_reproduces_phase_0s_z_image_observation() -> None:
    """The rate in facade.yaml against the manifest's bytes: Phase 0 observed about 214-220 s
    for a cold Z-Image-Turbo load and 52-55 s for its text encoder alone; a warm
    sequential-placement job still pays the encoder, and a resident one pays nothing."""
    from qontinui_imagegen.jobs import build_plan

    facade = make_facade()
    zimage = build_plan(facade.catalogue, "generations", {"model": "z-image-turbo", "prompt": "a"})
    assert 210.0 <= facade.catalogue.expected_load_s(zimage, cold=True) <= 225.0
    assert 52.0 <= facade.catalogue.expected_load_s(zimage, cold=False) <= 55.0
    flux = build_plan(facade.catalogue, "generations", {"model": "flux-1-schnell", "prompt": "a"})
    assert facade.catalogue.expected_load_s(flux, cold=False) == 0.0
    assert facade.catalogue.expected_load_s(flux, cold=True) > 0.0


# --- host memory under model_cpu_offload (review rounds 3b and 4) ---------------------------

#: Qwen-Image under model_cpu_offload keeps 35,543 MiB of weights on the host, and the facade
#: reserves facade.yaml's host_reserve_mib (3,072) beside them: 38,615 MiB in all.
QWEN_HOST_NEED = 35_543


@pytest.mark.parametrize(
    ("host", "status", "code"),
    [
        # The 24 GB WSL VM: the weights alone never fit.
        (HostMemory(20_000, 24_000, 2_000), 500, "host_memory_insufficient"),
        # The weights fit the total, but not with the facade's own reserve beside them.
        (HostMemory(37_500, 38_000, 2_000), 500, "host_memory_insufficient"),
        # Fits in total; free memory is short only because others hold it: busy.
        (HostMemory(36_000, 64_000, 2_000), 503, "host_memory_held_by"),
        # The reserve this process does not yet hold is asked of free memory as well.
        (HostMemory(37_000, 64_000, 0), 503, "host_memory_held_by"),
        (None, 500, "host_memory_unreadable"),
        # 35,543 + (3,072 - 2,000 already held) = 36,615 <= 37,000 free.
        (HostMemory(37_000, 64_000, 2_000), 200, None),
    ],
)
def test_an_offloaded_qwen_load_checks_the_host_can_hold_it(
    host: HostMemory | None, status: int, code: str | None
) -> None:
    """A load the host cannot hold is a typed answer before it starts -- never an OOM kill
    in the shared VM -- and, being checked before weight verification, it hashes nothing."""
    engine = FakeEngine(host=host)
    verifier = FakeVerifier()
    facade = make_facade(engine=engine, verifier=verifier)
    assert facade.catalogue.host_need_mib(["qwen-image-2512"]) == QWEN_HOST_NEED
    assert facade.config.host_reserve_mib == 3_072
    response = run(_post(facade, model="qwen-image-2512", prompt="x"))
    assert response.status_code == status, response.text
    if code is not None:
        assert response.json()["error"]["code"] == code
        assert engine.runs == []
        assert verifier.verified == set()  # refused before any weight was hashed


def test_a_model_kept_on_the_card_needs_no_host_reading() -> None:
    facade = make_facade(engine=FakeEngine(host=None))
    assert run(_post(facade, model="z-image-turbo", prompt="x")).status_code == 200


# --- reading host memory -------------------------------------------------------------------------

GIB = 1024**3


def _meminfo(tmp_path, total_gib: int = 64, available_gib: int = 48):
    path = tmp_path / "meminfo"
    path.write_text(
        f"MemTotal:       {total_gib * 1024 * 1024} kB\n"
        f"MemAvailable:   {available_gib * 1024 * 1024} kB\n"
    )
    return path


def _statm(tmp_path, resident_pages: int = 0):
    path = tmp_path / "statm"
    path.write_text(f"100000 {resident_pages} 0 0 0 0 0\n")
    return path


def _read(tmp_path, cgroup):
    from qontinui_imagegen.engine.diffusers_engine import read_host_memory

    return read_host_memory(_meminfo(tmp_path), cgroup, _statm(tmp_path))


def test_host_memory_with_no_cgroup_limit_is_meminfo(tmp_path) -> None:
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    assert _read(tmp_path, cgroup) == HostMemory(48 * 1024, 64 * 1024, 0)
    (cgroup / "memory.max").write_text("max\n")  # v2, unlimited
    assert _read(tmp_path, cgroup) == HostMemory(48 * 1024, 64 * 1024, 0)


def test_a_cgroup_v2_limit_narrows_and_leaves_the_page_cache_out(tmp_path) -> None:
    """R4-1: memory.current counts reclaimable page cache -- verification just read tens of
    GB through it -- so used = current - inactive_file, or a load that fits waits forever."""
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.max").write_text(f"{40 * GIB}\n")
    (cgroup / "memory.current").write_text(f"{38 * GIB}\n")  # 30 GiB of it is file cache
    (cgroup / "memory.stat").write_text(f"anon {8 * GIB}\ninactive_file {30 * GIB}\n")
    assert _read(tmp_path, cgroup) == HostMemory(32 * 1024, 40 * 1024, 0)


def test_a_cgroup_v1_limit_narrows_and_leaves_the_page_cache_out(tmp_path) -> None:
    """R4-2: a v1 container's limit is honoured too."""
    v1 = tmp_path / "cgroup" / "memory"
    v1.mkdir(parents=True)
    (v1 / "memory.limit_in_bytes").write_text(f"{40 * GIB}\n")
    (v1 / "memory.usage_in_bytes").write_text(f"{38 * GIB}\n")
    (v1 / "memory.stat").write_text(f"cache {30 * GIB}\ntotal_inactive_file {30 * GIB}\n")
    assert _read(tmp_path, tmp_path / "cgroup") == HostMemory(32 * 1024, 40 * 1024, 0)
    (v1 / "memory.limit_in_bytes").write_text("9223372036854771712\n")  # v1's "no limit"
    assert _read(tmp_path, tmp_path / "cgroup") == HostMemory(48 * 1024, 64 * 1024, 0)


@pytest.mark.parametrize("missing", ["memory.current", "memory.stat"])
def test_a_limit_whose_usage_cannot_be_read_is_unreadable_not_unlimited(
    tmp_path, missing: str
) -> None:
    """R4-2: memory.max set but its usage unreadable must not fall back to no limit."""
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.max").write_text(f"{16 * GIB}\n")
    (cgroup / "memory.current").write_text(f"{4 * GIB}\n")
    (cgroup / "memory.stat").write_text(f"inactive_file {1 * GIB}\n")
    (cgroup / missing).unlink()
    assert _read(tmp_path, cgroup) is None


def test_host_memory_reads_its_own_resident_set(tmp_path) -> None:
    import os

    from qontinui_imagegen.engine.diffusers_engine import read_host_memory

    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    pages = 2 * GIB // os.sysconf("SC_PAGE_SIZE")
    memory = read_host_memory(_meminfo(tmp_path), cgroup, _statm(tmp_path, pages))
    assert memory is not None and memory.own_rss_mib == 2048
    assert read_host_memory(tmp_path / "absent", cgroup, _statm(tmp_path)) is None


def test_startup_names_every_model_the_host_cannot_hold() -> None:
    """R4-7: in the 24 GB VM both Qwen models are named at start, before any request."""
    from qontinui_imagegen.app import host_memory_notices

    catalogue = make_facade().catalogue
    small = host_memory_notices(catalogue, HostMemory(20_000, 24_000, 2_000))
    assert [n.split()[1] for n in small] == ["qwen-image-2512", "qwen-image-edit-2511"]
    assert all("host_memory_insufficient" in n for n in small)
    assert host_memory_notices(catalogue, HostMemory(60_000, 64_000, 2_000)) == []
    unreadable = host_memory_notices(catalogue, None)
    assert len(unreadable) == 2 and all("host_memory_unreadable" in n for n in unreadable)
