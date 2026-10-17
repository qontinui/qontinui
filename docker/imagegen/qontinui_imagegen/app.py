"""The image facade: OpenAI-compatible ``/v1/images/*`` over the engine (plan Phase 10, PD1).

llama-swap (pinned v217) routes ``POST /v1/images/generations`` by the JSON body's ``model``
and ``POST /v1/images/edits`` by the multipart form's ``model`` field, and with
``useModelName`` unset it forwards the alias unchanged, so the facade picks the recipe from
it. Two v217 details shape this module: the form is rebuilt from Go maps, so field ORDER is
not preserved across keys (multiple input images must share one key, whose order is kept),
and every uploaded file arrives as ``application/octet-stream`` (images are identified by
their bytes, never by a declared type).

Wire:

* ``POST /v1/images/generations`` (JSON) and ``POST /v1/images/edits`` (multipart) answer
  ``{"created", "data": [{"b64_json", "qontinui_provenance"}], "output_format", "size"}``;
* with ``stream: true`` they answer server-sent events under OpenAI's names --
  ``image_generation.completed`` / ``image_edit.completed`` carrying ``b64_json`` and
  ``qontinui_provenance`` -- preceded by the namespaced progress extensions
  ``qontinui.queued {position}`` and ``qontinui.running {step, total}``, or an ``error``
  event carrying the refusal envelope (:mod:`qontinui_imagegen.errors`);
* ``GET /health`` is llama-swap's ``checkEndpoint``: 200 when ready, 503 with
  ``{"progress": <0..1>}`` while a weight verification runs (v217 reads only the status;
  later llama-swap versions also read the progress body);
* ``GET /qontinui/status`` reports the switch, the card and what is resident (read-only).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.datastructures import UploadFile

from qontinui_imagegen import errors, marker, provenance
from qontinui_imagegen.admission import AdmissionController, Ticket
from qontinui_imagegen.config import (
    LISTENER_ENV,
    FacadeConfig,
    listener_attestation,
    machine_switch,
)
from qontinui_imagegen.engine import Engine, EngineOutOfMemory, JobInterrupted, Rendered
from qontinui_imagegen.jobs import Catalogue, InputImage, JobPlan, build_plan, decode_input
from qontinui_imagegen.weights import WeightsError, WeightVerifier

#: When the card is held out of band, how long to wait before retrying: llama-swap's ttl for
#: its other entries (config.yaml), after which an idle resident model is unloaded.
OUT_OF_BAND_RETRY_S = 300.0
COMPLETED_EVENT = {"generations": "image_generation.completed", "edits": "image_edit.completed"}
MEDIA_TYPES = {"png": "image/png", "svg": "image/svg+xml"}
IMAGE_FIELDS = ("image", "image[]")


def _sse(event: str, data: Mapping[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def _refusal(error: errors.FacadeError) -> JSONResponse:
    return JSONResponse(error.body(), status_code=error.status, headers=error.headers())


class Facade:
    """The request path: switch, plan, admission, verification, the VRAM check, the job."""

    def __init__(
        self,
        catalogue: Catalogue,
        engine: Engine,
        verifier: WeightVerifier,
        admission: AdmissionController,
        *,
        environ: Mapping[str, str] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        random_seed: Callable[[], int] | None = None,
    ) -> None:
        self.catalogue = catalogue
        self.config: FacadeConfig = catalogue.config
        self.engine = engine
        self.verifier = verifier
        self.admission = admission
        self.environ = os.environ if environ is None else environ
        self.monotonic = monotonic
        self.random_seed = random_seed
        # One job on the card at a time: every engine call runs on this one thread.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="imagegen-card")
        self._tasks: set[asyncio.Future[None]] = set()

    # --- the switch ------------------------------------------------------------------------

    def check_switch(self) -> None:
        """The machine switch, then the listener attestation (rule R-bind)."""
        switch = machine_switch(self.environ)
        if switch.enabled:
            if listener_attestation(self.environ) is None:
                raise errors.disabled(
                    "listener_not_attested",
                    f"no deployment attested that llama-swap's listener is loopback-only "
                    f"({LISTENER_ENV} is {self.environ.get(LISTENER_ENV)!r}); the facade has no "
                    "authentication of its own, so it serves only behind a loopback listener "
                    "(rule R-bind) -- apply the loopback override in "
                    "docker/llama-swap/README.md",
                )
            return
        if switch.recognised:
            raise errors.disabled(
                "machine_switch_off",
                "image generation is switched off on this machine (QONTINUI_IMAGEGEN=0)",
            )
        raise errors.disabled(
            "machine_switch_unrecognised",
            f"QONTINUI_IMAGEGEN={switch.raw!r} is not 0 or 1, so image generation is off on "
            "this machine until it is set to one of them",
        )

    # --- the job, on the card's thread ------------------------------------------------------

    def _should_stop(self, ticket: Ticket) -> Callable[[], bool]:
        def stop() -> bool:
            deadline = ticket.deadline()
            return ticket.cancelled.is_set() or (
                deadline is not None and self.monotonic() > deadline
            )

        return stop

    def execute(
        self, plan: JobPlan, ticket: Ticket, on_step: Callable[[int, int], None]
    ) -> list[Rendered]:
        """Verify weights, check VRAM, run. Raises :class:`errors.FacadeError` on refusal."""
        for model_id in plan.model_ids:
            try:
                self.verifier.verify_model(model_id)
            except WeightsError as error:
                raise errors.failed(
                    "weights_unverified",
                    f"the weights of {error.model} are not the pinned bytes, so they are not "
                    f"served: {'; '.join(error.problems)}",
                    model=error.model,
                ) from error
        if self.engine.resident_key() != plan.resident_key:
            # Only one pipeline is resident at a time: release ours before measuring, so VRAM
            # the facade itself holds is never reported as out of band.
            self.engine.release()
            free = self.engine.free_vram_mib()
            if free is None:
                raise errors.failed(
                    "vram_unreadable",
                    "free VRAM could not be read, so a load could not be checked against it",
                )
            need = (
                self.catalogue.vram_need_mib(plan.model_ids) + self.config.vram_desktop_reserve_mib
            )
            if free < need:
                raise errors.busy(
                    "vram_held_by",
                    f"{free} MiB of VRAM is free after the facade released its own pipeline; "
                    f"this job needs {need} MiB (its models plus the desktop reserve), so "
                    "another GPU owner holds the card",
                    OUT_OF_BAND_RETRY_S,
                    vram_held_by="out_of_band",
                    free_mib=free,
                    need_mib=need,
                )
        total = plan.total_steps
        done = 0

        def step() -> None:
            nonlocal done
            done += 1
            on_step(min(done, total), total)

        try:
            return self.engine.run(plan, step, self._should_stop(ticket))
        except JobInterrupted as error:
            if ticket.cancelled.is_set():
                raise errors.failed(
                    "client_gone", "the client went away; the job was stopped"
                ) from error
            raise errors.failed(
                "interactive_gpu_seconds_cap",
                f"the job ran past the interactive cap of "
                f"{self.config.admission.interactive_gpu_seconds_cap:.0f} GPU seconds; send "
                "priority 'bulk' for heavier work",
            ) from error
        except EngineOutOfMemory as error:
            self.engine.release()
            raise errors.failed("out_of_memory", f"the card ran out of memory: {error}") from error

    # --- results ----------------------------------------------------------------------------

    def encode(self, plan: JobPlan, outputs: Sequence[Rendered]) -> list[dict[str, Any]]:
        items = []
        versions = self.engine.versions()
        for index, output in enumerate(outputs):
            if output.svg is not None:
                data = marker.mark_svg(output.svg).encode("utf-8")
            else:
                data = marker.encode_png(output.image)
            items.append(
                {
                    "b64_json": base64.b64encode(data).decode("ascii"),
                    "qontinui_provenance": provenance.build(
                        self.catalogue,
                        plan,
                        candidate_index=index,
                        seed=output.seed,
                        engine_versions=versions,
                    ),
                    "qontinui_media_type": MEDIA_TYPES[plan.output_format],
                    "size": f"{output.width}x{output.height}",
                }
            )
        return items

    # --- the request ------------------------------------------------------------------------

    async def _run_job(
        self, plan: JobPlan, ticket: Ticket, events: asyncio.Queue[tuple[str, Any]]
    ) -> None:
        """Wait for the card if queued, run the job, and post every event to ``events``.

        A task of its own, so the card is released when the JOB ends whether or not anyone
        is still listening: a streaming client that never reads still frees the card.
        """
        loop = asyncio.get_running_loop()
        try:
            if ticket.queued:
                await events.put(("queued", 1))
                await self.admission.wait_turn(ticket)
            if ticket.cancelled.is_set():
                raise errors.failed("client_gone", "the client went away before its turn")

            def on_step(step: int, total: int) -> None:
                loop.call_soon_threadsafe(events.put_nowait, ("running", (step, total)))

            outputs = await loop.run_in_executor(self.executor, self.execute, plan, ticket, on_step)
            await events.put(("done", outputs))
        except errors.FacadeError as error:
            await events.put(("error", error))
        except asyncio.CancelledError:
            ticket.cancelled.set()
            raise
        finally:
            self.admission.abandon(ticket)
            self.admission.release(ticket)

    async def respond(
        self, route: str, body: Mapping[str, Any], inputs: Sequence[InputImage]
    ) -> Response:
        try:
            self.check_switch()
            plan = build_plan(self.catalogue, route, body, inputs, random_seed=self.random_seed)
            ticket = self.admission.admit(plan.priority)
        except errors.FacadeError as error:
            return _refusal(error)
        events: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        task = asyncio.ensure_future(self._run_job(plan, ticket, events))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        if plan.stream:
            return StreamingResponse(
                self._stream(plan, ticket, events),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        while True:
            kind, value = await events.get()
            if kind == "error":
                return _refusal(value)
            if kind == "done":
                break
        items = self.encode(plan, value)
        return JSONResponse(
            {
                "created": int(time.time()),
                "data": items,
                "output_format": plan.output_format,
                "size": items[0]["size"] if items else None,
            }
        )

    async def _stream(
        self, plan: JobPlan, ticket: Ticket, events: asyncio.Queue[tuple[str, Any]]
    ) -> AsyncIterator[bytes]:
        finished = False
        try:
            while True:
                kind, value = await events.get()
                if kind == "queued":
                    yield _sse("qontinui.queued", {"type": "qontinui.queued", "position": value})
                elif kind == "running":
                    step, total = value
                    yield _sse(
                        "qontinui.running",
                        {"type": "qontinui.running", "step": step, "total": total},
                    )
                elif kind == "error":
                    finished = True
                    yield _sse("error", {"type": "error", **value.body()})
                    return
                else:
                    finished = True
                    item = self.encode(plan, value)[0]
                    event = COMPLETED_EVENT[plan.route]
                    yield _sse(
                        event,
                        {
                            "type": event,
                            "b64_json": item["b64_json"],
                            "created_at": int(time.time()),
                            "size": item["size"],
                            "output_format": plan.output_format,
                            "qontinui_media_type": item["qontinui_media_type"],
                            "qontinui_provenance": item["qontinui_provenance"],
                        },
                    )
                    return
        finally:
            if not finished:
                # The client stopped listening: stop the job at its next step.
                ticket.cancelled.set()


async def _edit_inputs(request: Request) -> tuple[dict[str, Any], list[InputImage]]:
    form = await request.form()
    body: dict[str, Any] = {}
    files: dict[str, list[bytes]] = {}
    for key, value in form.multi_items():
        if isinstance(value, UploadFile):
            files.setdefault(key, []).append(await value.read())
        elif key in body:
            raise errors.invalid("repeated_field", f"form field {key!r} is sent twice", key)
        else:
            body[key] = value
    if "mask" in files or "mask" in body:
        raise errors.invalid("mask_unsupported", "no recipe here inpaints under a mask", "mask")
    unknown_files = sorted(set(files) - set(IMAGE_FIELDS))
    if unknown_files:
        raise errors.invalid(
            "unknown_parameter", f"unexpected file field(s) {unknown_files}", unknown_files[0]
        )
    if len([key for key in IMAGE_FIELDS if key in files]) > 1:
        raise errors.invalid(
            "ambiguous_image_order",
            "send every input image under ONE field name (image or image[]): llama-swap "
            "rebuilds the form and keeps the order within a field, not across fields",
            "image",
        )
    if "n" in body and isinstance(body["n"], str) and body["n"].isdigit():
        body["n"] = int(body["n"])
    key = next((name for name in IMAGE_FIELDS if name in files), None)
    inputs = [decode_input(data, key or "image") for data in files.get(key or "", [])]
    return body, inputs


def create_app(facade: Facade) -> FastAPI:
    app = FastAPI(title="qontinui image facade", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    async def health() -> Response:
        progress = facade.verifier.progress
        if progress.active:
            return JSONResponse(
                {"status": "verifying_weights", "progress": round(progress.fraction, 4)},
                status_code=503,
            )
        return JSONResponse({"status": "ok"})

    @app.get("/qontinui/status")
    async def status() -> Response:
        switch = machine_switch(facade.environ)
        return JSONResponse(
            {
                "listener": listener_attestation(facade.environ),
                "switch": {
                    "enabled": switch.enabled,
                    "raw": switch.raw,
                    "recognised": switch.recognised,
                },
                "admission": facade.admission.snapshot(),
                "resident": facade.engine.resident_key(),
                "engine": facade.engine.name,
                "aliases": list(facade.catalogue.aliases),
                "verified_models": sorted(facade.verifier.verified),
            }
        )

    @app.post("/v1/images/generations")
    async def generations(request: Request) -> Response:
        try:
            body = await request.json()
        except ValueError:
            return _refusal(errors.invalid("invalid_json", "the request body is not JSON"))
        if not isinstance(body, dict):
            return _refusal(errors.invalid("invalid_json", "the request body is a JSON object"))
        return await facade.respond("generations", body, ())

    @app.post("/v1/images/edits")
    async def edits(request: Request) -> Response:
        if not request.headers.get("content-type", "").startswith("multipart/form-data"):
            return _refusal(
                errors.invalid(
                    "multipart_required",
                    "POST /v1/images/edits takes multipart/form-data, as OpenAI's does",
                )
            )
        try:
            facade.check_switch()  # the switch answers before the upload is even read
            body, inputs = await _edit_inputs(request)
        except errors.FacadeError as error:
            return _refusal(error)
        return await facade.respond("edits", body, inputs)

    return app
