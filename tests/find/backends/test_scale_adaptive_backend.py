"""Tests for the scale-adaptive template-matching backend.

Covers three layers:
  1. Shape tests — ABC compliance, env-flag gating, supports/cost contract.
  2. Feature-cache semantics — hash identity, LRU eviction, clear().
  3. A parametrized regression stub that opens grounding-eval records
     tagged ``dpi_changed`` or ``resolution_scaled`` (if the dataset is
     available via ``QONTINUI_GROUNDING_EVAL_DATASET``) and checks that
     the backend at least runs without exploding on them.  Results are
     split by tag so the plan's "may be weaker on dpi_changed than
     resolution_scaled" hypothesis can be evaluated later.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qontinui.find.backends._feature_cache import (
    ScreenshotFeatureCache,
    get_feature_cache,
    hash_screenshot,
)
from qontinui.find.backends.base import DetectionBackend
from qontinui.find.backends.scale_adaptive_backend import ENV_FLAG, ScaleAdaptiveBackend, is_enabled

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def enabled_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn on the scale-adaptive env flag for the duration of the test."""
    monkeypatch.setenv(ENV_FLAG, "1")


@pytest.fixture
def disabled_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the scale-adaptive env flag is explicitly unset."""
    monkeypatch.delenv(ENV_FLAG, raising=False)


def _torch_weights_available() -> bool:
    """True iff torchvision can actually fetch VGG-13 pretrained weights.

    We don't want to download multi-hundred-MB weights in CI.  We probe
    the torchvision weights registry rather than triggering the
    download.
    """
    try:
        import torchvision.models as models  # noqa: F401

        # Merely looking up the enum is enough — no download is issued.
        _ = models.VGG13_Weights.IMAGENET1K_V1
        return True
    except Exception:
        return False


# ===========================================================================
# 1. Shape tests
# ===========================================================================


class TestBackendShape:
    def test_implements_detection_backend(self) -> None:
        b = ScaleAdaptiveBackend()
        assert isinstance(b, DetectionBackend)

    def test_name(self) -> None:
        assert ScaleAdaptiveBackend().name == "scale_adaptive"

    def test_estimated_cost(self) -> None:
        assert ScaleAdaptiveBackend().estimated_cost_ms() == 140.0

    def test_supports_false_when_env_unset(self, disabled_env: None) -> None:
        b = ScaleAdaptiveBackend()
        assert b.supports("template") is False
        assert b.supports("text") is False
        assert b.supports("accessibility_id") is False

    def test_supports_template_when_env_set(self, enabled_env: None) -> None:
        b = ScaleAdaptiveBackend()
        assert b.supports("template") is True

    def test_supports_only_template_when_env_set(self, enabled_env: None) -> None:
        b = ScaleAdaptiveBackend()
        assert b.supports("text") is False
        assert b.supports("description") is False
        assert b.supports("accessibility_id") is False

    def test_is_enabled_helper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_FLAG, raising=False)
        assert is_enabled() is False
        monkeypatch.setenv(ENV_FLAG, "0")
        assert is_enabled() is False
        monkeypatch.setenv(ENV_FLAG, "true")
        # Strict "1"-only gate by design.
        assert is_enabled() is False
        monkeypatch.setenv(ENV_FLAG, "1")
        assert is_enabled() is True

    def test_find_noop_when_disabled(self, disabled_env: None) -> None:
        b = ScaleAdaptiveBackend()
        results = b.find(
            needle=np.zeros((8, 8, 3), dtype=np.uint8),
            haystack=np.zeros((64, 64, 3), dtype=np.uint8),
            config={"min_confidence": 0.5},
        )
        assert results == []

    def test_find_handles_unconvertible_inputs(self, enabled_env: None) -> None:
        b = ScaleAdaptiveBackend()
        # Strings should convert to None and yield an empty result.
        assert (
            b.find(
                needle="not_an_image",
                haystack=np.zeros((32, 32, 3), dtype=np.uint8),
                config={},
            )
            == []
        )
        assert (
            b.find(
                needle=np.zeros((8, 8, 3), dtype=np.uint8),
                haystack="not_an_image",
                config={},
            )
            == []
        )


# ===========================================================================
# 2. Feature cache
# ===========================================================================


class TestFeatureCache:
    def test_singleton_returns_same_instance(self) -> None:
        a = get_feature_cache()
        b = get_feature_cache()
        assert a is b

    def test_hash_same_for_same_bytes(self) -> None:
        img = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
        assert hash_screenshot(img) == hash_screenshot(img)

    def test_hash_differs_for_different_bytes(self) -> None:
        a = np.zeros((64, 64, 3), dtype=np.uint8)
        b = np.ones((64, 64, 3), dtype=np.uint8)
        assert hash_screenshot(a) != hash_screenshot(b)

    def test_hash_accepts_bytes_directly(self) -> None:
        assert hash_screenshot(b"hello") == hash_screenshot(b"hello")
        assert hash_screenshot(b"hello") != hash_screenshot(b"world")

    def test_hash_rejects_non_bytes(self) -> None:
        with pytest.raises(TypeError):
            hash_screenshot(42)  # type: ignore[arg-type]

    def test_put_and_get_roundtrip(self) -> None:
        cache = ScreenshotFeatureCache(max_entries=4)
        cache.put("hash_a", "backend_x", "payload_a")
        assert cache.get("hash_a", "backend_x") == "payload_a"

    def test_miss_on_unknown_hash(self) -> None:
        cache = ScreenshotFeatureCache(max_entries=4)
        cache.put("hash_a", "backend_x", "payload_a")
        assert cache.get("hash_b", "backend_x") is None

    def test_miss_on_unknown_backend_key(self) -> None:
        cache = ScreenshotFeatureCache(max_entries=4)
        cache.put("hash_a", "backend_x", "payload_a")
        assert cache.get("hash_a", "backend_y") is None

    def test_multiple_backends_per_hash(self) -> None:
        cache = ScreenshotFeatureCache(max_entries=4)
        cache.put("hash_a", "backend_x", 1)
        cache.put("hash_a", "backend_y", 2)
        assert cache.get("hash_a", "backend_x") == 1
        assert cache.get("hash_a", "backend_y") == 2

    def test_lru_eviction(self) -> None:
        cache = ScreenshotFeatureCache(max_entries=2)
        cache.put("h1", "k", "v1")
        cache.put("h2", "k", "v2")
        cache.put("h3", "k", "v3")  # evicts h1 (LRU)
        assert cache.get("h1", "k") is None
        assert cache.get("h2", "k") == "v2"
        assert cache.get("h3", "k") == "v3"

    def test_get_refreshes_lru(self) -> None:
        cache = ScreenshotFeatureCache(max_entries=2)
        cache.put("h1", "k", "v1")
        cache.put("h2", "k", "v2")
        assert cache.get("h1", "k") == "v1"  # touch h1
        cache.put("h3", "k", "v3")  # should evict h2 now (LRU)
        assert cache.get("h1", "k") == "v1"
        assert cache.get("h2", "k") is None
        assert cache.get("h3", "k") == "v3"

    def test_has(self) -> None:
        cache = ScreenshotFeatureCache(max_entries=2)
        cache.put("h1", "k", "v1")
        assert cache.has("h1", "k") is True
        assert cache.has("h1", "other") is False
        assert cache.has("other", "k") is False

    def test_clear(self) -> None:
        cache = ScreenshotFeatureCache(max_entries=2)
        cache.put("h1", "k", "v1")
        cache.put("h2", "k", "v2")
        cache.clear()
        assert len(cache) == 0
        assert cache.get("h1", "k") is None

    def test_max_entries_validation(self) -> None:
        with pytest.raises(ValueError):
            ScreenshotFeatureCache(max_entries=0)


# ===========================================================================
# 3. Regression-friendly stub over grounding-eval records (optional)
# ===========================================================================


def _load_dataset() -> list[dict[str, Any]]:
    """Load grounding-eval records tagged dpi_changed / resolution_scaled.

    Dataset format is flexible — we look for either a JSONL file
    (one record per line) or a JSON array.  Each record should expose
    ``tags`` (list[str]) and ``id`` (str).  Records without at least
    one of the target tags are filtered out.  Returns [] if the env
    var is unset or the file is missing.
    """
    path_str = os.environ.get("QONTINUI_GROUNDING_EVAL_DATASET")
    if not path_str:
        return []
    path = Path(path_str)
    if not path.is_file():
        return []

    raw: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl" or "\n{" in text:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                raw.append(json.loads(line))
        else:
            data = json.loads(text)
            if isinstance(data, list):
                raw = list(data)
    except (OSError, json.JSONDecodeError):
        return []

    target_tags = {"dpi_changed", "resolution_scaled"}
    filtered: list[dict[str, Any]] = []
    for rec in raw:
        tags = rec.get("tags") or []
        if any(t in target_tags for t in tags):
            filtered.append(rec)
    return filtered


_DATASET = _load_dataset()


@pytest.mark.parametrize(
    "record",
    _DATASET,
    ids=(
        [rec.get("id", f"rec_{i}") for i, rec in enumerate(_DATASET)]
        if _DATASET
        else None
    ),
)
def test_regression_by_tag(record: dict[str, Any], enabled_env: None) -> None:
    """Smoke-run the backend against a tagged eval record.

    Skips if the dataset isn't configured or if torchvision pretrained
    weights aren't reachable (CI without network access).  We do NOT
    assert confidence thresholds here — this is a regression stub whose
    purpose is to (a) make sure the backend runs on dpi_changed /
    resolution_scaled inputs, and (b) publish per-tag pass/confidence
    data that a later stage of the plan can evaluate.
    """
    if not _DATASET:  # pragma: no cover — guarded by skipif indirectly
        pytest.skip("QONTINUI_GROUNDING_EVAL_DATASET not set")
    _ = pytest.importorskip("torch")
    _ = pytest.importorskip("torchvision")
    if not _torch_weights_available():
        pytest.skip("VGG-13 pretrained weights unavailable")

    template_path = record.get("template_path") or record.get("needle_path")
    haystack_path = record.get("screenshot_path") or record.get("haystack_path")
    if not template_path or not haystack_path:
        pytest.skip("record missing template/haystack paths")

    try:
        import cv2
    except ImportError:
        pytest.skip("opencv-python unavailable")

    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    haystack = cv2.imread(str(haystack_path), cv2.IMREAD_COLOR)
    if template is None or haystack is None:
        pytest.skip("record image files not readable")

    tags = record.get("tags") or []
    tag_group = "resolution_scaled" if "resolution_scaled" in tags else "dpi_changed"
    # Tag is surfaced via user_properties so downstream analysis can
    # split pass/fail by tag group without re-parsing the dataset.
    # pytest's request fixture would let us record this more formally;
    # printing keeps the stub's dependencies minimal.
    print(f"[scale_adaptive_regression] tag_group={tag_group} id={record.get('id')}")

    backend = ScaleAdaptiveBackend()
    results = backend.find(
        needle=template,
        haystack=haystack,
        config={
            "needle_type": "template",
            "min_confidence": 0.0,  # accept anything — stub, not a pass/fail
        },
    )
    # Smoke: the call returned a list (possibly empty), no exceptions.
    assert isinstance(results, list)
