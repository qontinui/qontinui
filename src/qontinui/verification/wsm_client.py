"""Python client for the World State Verifier (WSM) judge.

Transport choice: **(b)** — this client POSTs directly to the llama-swap
``cua-wsm`` endpoint instead of going through a Rust HTTP passthrough.
Rationale: the grounding-capture pipeline may run without the Tauri
runner process (e.g. inside a CI container, from an E2E harness, or from
a one-off script), and prompt parity with the Rust judge is enforced by
a shared TOML artifact (see ``wsm_prompt.toml``) rather than by sharing
a process. The cost is that this client must be kept structurally
aligned with ``qontinui-runner/src-tauri/src/verification/world_state_verifier.rs``;
the shared TOML makes that alignment mechanical rather than by-hand.

Endpoint resolution order:
    1. Explicit ``endpoint`` argument to :class:`WSMClient`.
    2. ``QONTINUI_WSM_ENDPOINT`` env var.
    3. ``QONTINUI_WORLD_STATE_VERIFIER_ENDPOINT`` env var (matches the
       Rust judge's existing env var so a single export works for both).
    4. Default ``http://127.0.0.1:8100`` (llama-swap on localhost).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx

try:  # Python 3.11+ has tomllib in the stdlib; older envs can fall back.
    import tomllib as _toml_reader
except ModuleNotFoundError:  # pragma: no cover — qontinui targets >=3.12
    import tomli as _toml_reader  # type: ignore[no-redef]

from .pixel_diff import pixel_diff_verdict

logger = logging.getLogger(__name__)

# Path to the shared prompt artifact. The Rust judge mirrors this file's
# wording in-source (see WSM_SYSTEM_PROMPT in world_state_verifier.rs).
PROMPT_TOML_PATH = Path(__file__).with_name("wsm_prompt.toml")

# Default timeout for a single verify call when not overridden. Matches
# the guardrail requested by the WSM integration plan — WSM inference
# should not block grounding capture for longer than this.
DEFAULT_VERIFY_TIMEOUT_S = 30.0

SuccessSource = Literal["wsm", "pixel_diff", "record_flag"]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class WSMVerdict:
    """Outcome of a single verify call.

    ``source`` records which labelling strategy produced the verdict so
    callers can stamp ``GroundingAction.success_source`` directly from
    this field without re-deriving it.
    """

    success: bool
    confidence: float
    reason: str
    source: SuccessSource
    status: str = ""
    refused_reason: str | None = None
    next_priority: str | None = None
    raw: dict[str, Any] | None = None


@dataclass
class PromptConfig:
    """Parsed ``wsm_prompt.toml`` contents."""

    schema_version: str
    model_alias: str
    temperature: float
    max_tokens: int
    system_prompt: str
    user_template: str
    goal_section_template: str
    low_confidence_fallback_threshold: float
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

_PROMPT_CACHE: PromptConfig | None = None


def load_prompt_config(path: Path = PROMPT_TOML_PATH) -> PromptConfig:
    """Load and cache the shared WSM prompt config.

    The file is parsed once per process; tests that edit the TOML can
    call :func:`reload_prompt_config` to bust the cache.
    """
    global _PROMPT_CACHE
    if _PROMPT_CACHE is not None and path == PROMPT_TOML_PATH:
        return _PROMPT_CACHE

    with open(path, "rb") as f:
        data = _toml_reader.load(f)

    cfg = PromptConfig(
        schema_version=str(data.get("schema_version", "1")),
        model_alias=str(data.get("model_alias", "cua-wsm")),
        temperature=float(data.get("temperature", 0.0)),
        max_tokens=int(data.get("max_tokens", 512)),
        system_prompt=str(data["system_prompt"]).strip(),
        user_template=str(data["user_template"]).strip("\n"),
        goal_section_template=str(data.get("goal_section_template", "")).strip("\n"),
        low_confidence_fallback_threshold=float(
            data.get("low_confidence_fallback_threshold", 0.55)
        ),
        raw=data,
    )
    if path == PROMPT_TOML_PATH:
        _PROMPT_CACHE = cfg
    return cfg


def reload_prompt_config() -> PromptConfig:
    """Bust the prompt cache and reload from disk. For tests."""
    global _PROMPT_CACHE
    _PROMPT_CACHE = None
    return load_prompt_config()


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------

def resolve_endpoint(explicit: str | None = None) -> str:
    """Pick the WSM endpoint per the documented precedence order."""
    if explicit:
        return explicit.rstrip("/")
    for var in ("QONTINUI_WSM_ENDPOINT", "QONTINUI_WORLD_STATE_VERIFIER_ENDPOINT"):
        val = os.environ.get(var)
        if val:
            return val.rstrip("/")
    return "http://127.0.0.1:8100"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class WSMClient:
    """Async client for the WSM llama-swap endpoint.

    Thread-safe: httpx.AsyncClient is created lazily and reused across
    calls. Close via :meth:`aclose` (or use the context manager).
    """

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        timeout_s: float = DEFAULT_VERIFY_TIMEOUT_S,
        prompt: PromptConfig | None = None,
    ) -> None:
        self.prompt = prompt or load_prompt_config()
        self.endpoint = resolve_endpoint(endpoint)
        self.model = model or self.prompt.model_alias
        self.timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> WSMClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    def _build_user_text(self, intent: str, goal: str | None) -> str:
        if goal:
            goal_block = self.prompt.goal_section_template.format(goal=goal)
        else:
            goal_block = ""
        return self.prompt.user_template.format(
            goal_section=goal_block,
            intent=intent,
        )

    def _build_payload(
        self,
        pre_b64: str,
        post_b64: str,
        intent: str,
        goal: str | None,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.prompt.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._build_user_text(intent, goal)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{pre_b64}"
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{post_b64}"
                            },
                        },
                    ],
                },
            ],
            "temperature": self.prompt.temperature,
            "max_tokens": self.prompt.max_tokens,
        }

    # ------------------------------------------------------------------
    async def verify_raw(
        self,
        before_png_bytes: bytes,
        after_png_bytes: bytes,
        intent: str,
        goal: str | None = None,
    ) -> dict[str, Any]:
        """Call the WSM and return the parsed judgement dict.

        Raises :class:`httpx.HTTPError` on transport failure and
        :class:`ValueError` when the response cannot be parsed into the
        declared JSON schema.
        """
        pre_b64 = base64.b64encode(before_png_bytes).decode("ascii")
        post_b64 = base64.b64encode(after_png_bytes).decode("ascii")
        payload = self._build_payload(pre_b64, post_b64, intent, goal)

        url = f"{self.endpoint}/v1/chat/completions"
        client = self._ensure_client()
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        return _parse_wsm_json(content)

    async def verify(
        self,
        before_png_bytes: bytes,
        after_png_bytes: bytes,
        intent: str,
        goal: str | None = None,
    ) -> WSMVerdict:
        """Call the WSM and return a :class:`WSMVerdict`.

        Low-confidence judgments are transparently backfilled with a
        pixel-diff result so the caller always gets a usable boolean.
        ``source`` distinguishes the two cases:

        - ``"wsm"`` — confidence ≥ threshold, success taken from status
        - ``"pixel_diff"`` — confidence below threshold, success from
          pixel diff; ``raw`` still carries the WSM response for audit
        """
        parsed = await self.verify_raw(
            before_png_bytes, after_png_bytes, intent, goal
        )
        status = str(parsed.get("status", "")).lower()
        confidence = _clamp_confidence(parsed.get("confidence", 0.0))
        observations = str(parsed.get("observations", ""))
        refused_reason = parsed.get("refused_reason")
        next_priority = parsed.get("next_priority")

        threshold = self.prompt.low_confidence_fallback_threshold
        if confidence >= threshold and status in {"pass", "partial", "fail", "refused"}:
            # "partial" counts as success for the grounding-capture label
            # (the worker made visible progress toward intent). Rust
            # treats Partial the same way in `judgement_to_verdict`.
            success = status in {"pass", "partial"}
            reason = observations or f"wsm status={status} confidence={confidence:.2f}"
            return WSMVerdict(
                success=success,
                confidence=confidence,
                reason=reason,
                source="wsm",
                status=status,
                refused_reason=refused_reason,
                next_priority=next_priority,
                raw=parsed,
            )

        # Low confidence: fall back to pixel diff and preserve the raw
        # WSM verdict for audit so calibration work can analyze why.
        pd_success, pd_reason = pixel_diff_verdict(
            before_png_bytes, after_png_bytes
        )
        return WSMVerdict(
            success=pd_success,
            confidence=confidence,
            reason=(
                f"wsm confidence {confidence:.2f} < threshold {threshold:.2f}; "
                f"fell back to {pd_reason}"
            ),
            source="pixel_diff",
            status=status,
            refused_reason=refused_reason,
            next_priority=next_priority,
            raw=parsed,
        )


# ---------------------------------------------------------------------------
# Top-level functional API (preferred by callers that don't hold a client)
# ---------------------------------------------------------------------------

async def verify_action(
    before_png_bytes: bytes,
    after_png_bytes: bytes,
    intent: str,
    goal: str | None = None,
    *,
    endpoint: str | None = None,
    model: str | None = None,
    timeout_s: float = DEFAULT_VERIFY_TIMEOUT_S,
) -> WSMVerdict:
    """Judge whether *after_png_bytes* reflects the *intent* being achieved.

    Wrapped in :func:`asyncio.wait_for` with a hard *timeout_s* so it
    can never block a capture pipeline indefinitely. On timeout (or any
    transport error), falls back to a pure pixel-diff verdict with
    ``source="pixel_diff"`` — the grounding record is still stamped,
    just with a conservative label.
    """
    try:
        async with WSMClient(
            endpoint=endpoint, model=model, timeout_s=timeout_s
        ) as client:
            return await asyncio.wait_for(
                client.verify(before_png_bytes, after_png_bytes, intent, goal),
                timeout=timeout_s,
            )
    except TimeoutError:
        pd_success, pd_reason = pixel_diff_verdict(
            before_png_bytes, after_png_bytes
        )
        logger.warning(
            "WSM verify timed out after %.1fs; falling back to pixel diff", timeout_s
        )
        return WSMVerdict(
            success=pd_success,
            confidence=0.0,
            reason=f"wsm timeout after {timeout_s:.1f}s; {pd_reason}",
            source="pixel_diff",
        )
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        pd_success, pd_reason = pixel_diff_verdict(
            before_png_bytes, after_png_bytes
        )
        logger.warning("WSM verify failed (%s); falling back to pixel diff", exc)
        return WSMVerdict(
            success=pd_success,
            confidence=0.0,
            reason=f"wsm error {type(exc).__name__}: {exc}; {pd_reason}",
            source="pixel_diff",
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_wsm_json(content: str) -> dict[str, Any]:
    """Extract the JSON object from a WSM response.

    Accepts both fenced (```json ... ```) and raw responses. Mirrors the
    Rust parser in ``world_state_verifier.rs`` (parse_judgement).
    """
    m = _FENCED_JSON_RE.search(content)
    if m:
        return json.loads(m.group(1))

    start = content.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in WSM response: {content[:200]!r}")
    depth = 0
    for i, ch in enumerate(content[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(content[start : i + 1])
    raise ValueError(f"Unbalanced JSON in WSM response: {content[:200]!r}")


def _clamp_confidence(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v
