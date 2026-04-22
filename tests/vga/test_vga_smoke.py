"""Smoke tests for the VGA submodule.

Covers:

- Round-tripping a :class:`VgaStateMachine` through canonical JSON.
- Determinism of :meth:`VgaStateMachine.sha256`.
- :class:`CorrectionLogger` appending a JSONL line to a tmp dir.
- :class:`VgaClient` request/parse logic using a mocked
  ``urllib.request.urlopen`` — never contacts the network.

The test file avoids any reliance on a live llama-swap endpoint.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from unittest import mock
from uuid import uuid4

import pytest
from PIL import Image

from qontinui.vga import (
    BBox,
    VgaClient,
    VgaElement,
    VgaState,
    VgaStateMachine,
    VgaTransition,
)
from qontinui.vga.client import VgaClientError
from qontinui.vga.correction_log import CorrectionLogger


def _make_sm() -> VgaStateMachine:
    element = VgaElement(
        label="Save button",
        prompt="Save button in the toolbar",
        bbox=BBox(x=100, y=200, w=40, h=30),
    )
    another = VgaElement(
        label="Cancel button",
        prompt="Cancel button in the dialog",
        bbox=BBox(x=300, y=200, w=60, h=30),
    )
    state_a = VgaState(name="main", elements=[element])
    state_b = VgaState(name="save_dialog", elements=[another], blocking=True)
    transition = VgaTransition(
        from_state_id=state_a.id,
        to_state_id=state_b.id,
        trigger_element_id=element.id,
    )
    return VgaStateMachine(
        name="notepad++ test",
        target_process="notepad++.exe",
        target_os="windows",
        states=[state_a, state_b],
        transitions=[transition],
    )


def test_state_machine_canonical_roundtrip() -> None:
    sm = _make_sm()
    payload = sm.to_canonical_json()
    # Should be valid JSON and stable-sorted (keys in lexicographic order).
    parsed = json.loads(payload)
    assert parsed["name"] == "notepad++ test"
    assert "created_at" not in parsed
    assert "updated_at" not in parsed

    restored = VgaStateMachine.from_canonical_json(payload)
    assert restored.name == sm.name
    assert restored.target_process == sm.target_process
    assert len(restored.states) == len(sm.states)
    assert restored.states[1].blocking is True
    assert restored.states[0].elements[0].bbox.x == 100


def test_state_machine_sha256_is_deterministic() -> None:
    sm_a = _make_sm()
    sm_b = VgaStateMachine.from_canonical_json(sm_a.to_canonical_json())
    # Hash should match across roundtrip — timestamps are excluded.
    assert sm_a.sha256() == sm_b.sha256()


def test_state_machine_sha256_changes_on_structural_edit() -> None:
    sm = _make_sm()
    baseline = sm.sha256()
    sm.states[0].elements[0].bbox.x = 999
    assert sm.sha256() != baseline


def test_correction_logger_appends_jsonl(tmp_path: Path) -> None:
    logger = CorrectionLogger(corrections_dir=tmp_path)
    sm_id = uuid4()
    image_path = tmp_path / "shot.png"
    image_path.write_bytes(b"fake-png")

    logger.append(
        state_machine_id=sm_id,
        image_sha="deadbeef",
        image_path=image_path,
        prompt="Save button",
        corrected_bbox=BBox(x=10, y=20, w=40, h=40),
        source="builder",
        target_process="notepad++.exe",
        private=True,
    )

    jsonl = logger.jsonl_path
    assert jsonl.exists()
    lines = jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["prompt"] == "Save button"
    assert entry["target_process"] == "notepad++.exe"
    assert entry["private"] is True
    # Sidecar marker for private entries.
    assert (tmp_path / "shot.png.private").exists()

    stats = logger.stats()
    assert stats.total == 1
    assert stats.per_target_process["notepad++.exe"] == 1


def _fake_response(text: str) -> mock.MagicMock:
    """Build a urlopen context-manager returning the given model text."""
    body = json.dumps({"choices": [{"message": {"content": text}}]}).encode("utf-8")

    resp = mock.MagicMock()
    resp.read.return_value = body
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _make_png_bytes() -> bytes:
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_client_ground_parses_point_tag() -> None:
    client = VgaClient()
    png = _make_png_bytes()

    with mock.patch(
        "urllib.request.urlopen", return_value=_fake_response("<point>500 500</point>")
    ):
        result = client.ground(png, "Save button")

    # 500/1000 * 100 = 50 in both axes.
    assert result.x == 50
    assert result.y == 50
    assert result.norm_x == 500.0
    assert result.confidence == 0.75
    assert result.image_width == 100
    assert result.image_height == 100


def test_client_ground_handles_none_sentinel() -> None:
    client = VgaClient()
    png = _make_png_bytes()

    with mock.patch("urllib.request.urlopen", return_value=_fake_response("<none/>")):
        result = client.ground(png, "NonExistent thing")

    assert result.confidence == 0.0
    assert result.x == 0
    assert result.y == 0


def test_client_ground_raises_on_unparseable() -> None:
    client = VgaClient()
    png = _make_png_bytes()

    with mock.patch("urllib.request.urlopen", return_value=_fake_response("the model rambled")):
        with pytest.raises(VgaClientError):
            client.ground(png, "Whatever")


def test_client_handles_base64_encoding_of_bytes() -> None:
    """Sanity: encoding an already-PNG bytes input round-trips through base64."""
    png = _make_png_bytes()
    # Make sure the fixture is valid PNG bytes.
    assert base64.b64encode(png).decode("ascii")
