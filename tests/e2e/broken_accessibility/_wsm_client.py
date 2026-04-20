"""Thin Python client for the World State Verifier (WSM).

Test-only — not exported from qontinui. Mirrors the Rust
`WorldStateVerifier` that lives in qontinui-runner under
`src-tauri/src/verification/world_state_verifier.rs`. Use this helper
to exercise the WSM directly from Python E2E tests without having to
boot the full runner.

The WSM is served via llama-swap's OpenAI-compatible endpoint under
the model alias `cua-wsm` (see qontinui/docker/llama-swap/config.yaml).
Both pre- and post-action screenshots are sent in a single multimodal
chat message; the model returns a JSON verdict which this helper
parses into a plain dict.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

WSM_SYSTEM_PROMPT = (
    "You are a GUI world-state verifier. You receive two screenshots "
    "(PRE and POST) of an application and a text intent describing what "
    "action the agent attempted. Your job is to judge, by comparing the "
    "two screenshots, whether the POST state reflects that the intended "
    "action was actually performed on the correct target.\n\n"
    "Output STRICT JSON matching this schema:\n"
    "```json\n"
    "{\n"
    '  "status": "pass" | "fail" | "refused" | "partial",\n'
    '  "confidence": <float 0.0-1.0>,\n'
    '  "observations": "<short explanation>",\n'
    '  "refused_reason": "<only if status=refused>",\n'
    '  "next_priority": "<optional>"\n'
    "}\n"
    "```\n\n"
    "Status semantics:\n"
    "- pass: intended action performed on correct target, state matches intent.\n"
    "- partial: some visible progress, not complete.\n"
    "- fail: no visible progress or regression.\n"
    "- refused: action did NOT interact with the intended target (wrong "
    "button clicked, click missed, wrong element). Use this when the agent "
    "appears to have acted on the wrong element."
)


@dataclass
class WsmVerdict:
    status: str
    confidence: float
    observations: str
    refused_reason: str | None = None
    next_priority: str | None = None
    raw: dict[str, Any] | None = None


class WorldStateVerifierClient:
    """Thin client against a llama-swap-served WSM endpoint."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8100",
        model: str = "cua-wsm",
        timeout_s: float = 300.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def is_reachable(self) -> bool:
        """Quick probe — does llama-swap respond on /v1/models?"""
        try:
            resp = httpx.get(f"{self.endpoint}/v1/models", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def verify(
        self,
        pre_png_bytes: bytes,
        post_png_bytes: bytes,
        intent: str,
        goal: str | None = None,
    ) -> WsmVerdict:
        """Submit pre/post screenshots + intent, return a parsed verdict.

        Raises on HTTP failure or unparseable output so tests can
        fail-loudly rather than silently accept a bad judgment.
        """
        pre_b64 = base64.b64encode(pre_png_bytes).decode("ascii")
        post_b64 = base64.b64encode(post_png_bytes).decode("ascii")

        user_text_parts = []
        if goal:
            user_text_parts.append(f"## Overall Goal\n{goal}")
        user_text_parts.append(f"## Action Intent\n{intent}")
        user_text_parts.append(
            "The first image is PRE (before the action). The second image "
            "is POST (after the action). Compare them and return the JSON verdict."
        )
        user_text = "\n\n".join(user_text_parts)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": WSM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{pre_b64}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{post_b64}"},
                        },
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }

        resp = httpx.post(
            f"{self.endpoint}/v1/chat/completions",
            json=payload,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        parsed = _parse_wsm_json(content)
        return WsmVerdict(
            status=str(parsed["status"]).lower(),
            confidence=float(parsed["confidence"]),
            observations=str(parsed.get("observations", "")),
            refused_reason=parsed.get("refused_reason"),
            next_priority=parsed.get("next_priority"),
            raw=parsed,
        )


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_wsm_json(content: str) -> dict[str, Any]:
    """Extract the JSON object from a WSM response.

    Accepts both fenced (```json ... ```) and raw responses. Finds the
    first balanced {} block if no fence is present.
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
