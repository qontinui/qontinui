"""Python-side verification helpers for the World State Verifier (WSM).

This package bridges Python callers (grounding capture, training data
pipelines, E2E tests) to the WSM judge already implemented in Rust at
``qontinui-runner/src-tauri/src/verification/world_state_verifier.rs``.

Public surface:

- :class:`WSMVerdict`: structured judge output usable by callers.
- :func:`verify_action`: async call through to the WSM endpoint with a
  strict timeout and a pixel-diff fallback.
- :func:`record_wsm_verdict`: stamp a ``GroundingAction`` with the verdict
  (setting ``success_source="wsm"`` on confident verdicts, ``"pixel_diff"``
  on low-confidence ones).

The prompt template lives in ``wsm_prompt.toml`` and is the source of
truth for both the Rust judge and this Python client.
"""

from .pixel_diff import pixel_diff_verdict
from .record_action_verification import record_wsm_verdict
from .wsm_client import WSMClient, WSMVerdict, load_prompt_config, verify_action

__all__ = [
    "WSMVerdict",
    "WSMClient",
    "load_prompt_config",
    "verify_action",
    "record_wsm_verdict",
    "pixel_diff_verdict",
]
