"""Weight verification by stamps, and the verify-on-download install (plan Phase 10)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from qontinui_imagegen import weights as w

REVISION = "0123456789abcdef0123456789abcdef01234567"


def _component(files: dict[str, bytes]) -> dict:
    return {
        "id": "tiny-model",
        "kind": "model",
        "source": {"type": "huggingface", "repo": "example/Tiny", "revision": REVISION},
        "files": [
            {"path": path, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for path, data in files.items()
        ],
    }


FILES = {"model_index.json": b"{}", "transformer/weights.safetensors": b"weights" * 100}


def _install_files(root: Path, files: dict[str, bytes]) -> Path:
    base = root / "hub" / "models--example--Tiny" / "snapshots" / REVISION
    for path, data in files.items():
        (base / path).parent.mkdir(parents=True, exist_ok=True)
        (base / path).write_bytes(data)
    return base


class CountingHasher:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def __call__(self, path: Path, progress=None) -> str:
        self.paths.append(path.name)
        return w.sha256_file(path, progress)


def _verifier(root: Path, component: dict, hasher=None, fields=("size", "mtime_ns", "inode")):
    return w.WeightVerifier(
        root,
        {"tiny-model": component},
        root / "stamps.json",
        fields,
        hasher=hasher or w.sha256_file,
    )


def test_the_layout_is_a_stock_hf_cache_and_github_release_assets(tmp_path: Path) -> None:
    assert w.component_dir(tmp_path, _component(FILES)) == (
        tmp_path / "hub" / "models--example--Tiny" / "snapshots" / REVISION
    )
    release = {
        "id": "r",
        "source": {"type": "github_release", "repo": "xinntao/Real-ESRGAN", "tag": "v0.2.1"},
    }
    assert (
        w.component_dir(tmp_path, release)
        == tmp_path / "github" / "xinntao--Real-ESRGAN" / "v0.2.1"
    )


def test_verified_files_are_stamped_and_not_rehashed_by_the_next_process(tmp_path: Path) -> None:
    _install_files(tmp_path, FILES)
    first = CountingHasher()
    _verifier(tmp_path, _component(FILES), first).verify_model("tiny-model")
    assert sorted(first.paths) == ["model_index.json", "weights.safetensors"]
    second = CountingHasher()  # a new process: llama-swap restarted the facade after its ttl
    verifier = _verifier(tmp_path, _component(FILES), second)
    verifier.verify_model("tiny-model")
    assert second.paths == [] and "tiny-model" in verifier.verified


def test_a_file_whose_stat_changed_is_rehashed_and_a_mismatch_is_refused(tmp_path: Path) -> None:
    base = _install_files(tmp_path, FILES)
    _verifier(tmp_path, _component(FILES)).verify_model("tiny-model")
    target = base / "transformer" / "weights.safetensors"
    target.write_bytes(b"WEIGHTS" * 100)  # same size, different bytes, new mtime
    os.utime(target, ns=(target.stat().st_atime_ns, target.stat().st_mtime_ns + 10**9))
    hasher = CountingHasher()
    with pytest.raises(w.WeightsError) as raised:
        _verifier(tmp_path, _component(FILES), hasher).verify_model("tiny-model")
    assert hasher.paths == ["weights.safetensors"]
    assert "hashes to" in raised.value.problems[0]


def test_a_missing_or_resized_file_is_refused_without_hashing(tmp_path: Path) -> None:
    base = _install_files(tmp_path, {"model_index.json": b"{}"})
    hasher = CountingHasher()
    with pytest.raises(w.WeightsError, match="not installed"):
        _verifier(tmp_path, _component(FILES), hasher).verify_model("tiny-model")
    (base / "transformer").mkdir()
    (base / "transformer" / "weights.safetensors").write_bytes(b"short")
    with pytest.raises(w.WeightsError, match="bytes, the manifest pins"):
        _verifier(tmp_path, _component(FILES), hasher).verify_model("tiny-model")
    assert hasher.paths == []


def test_without_inode_in_the_stamp_fields_a_new_inode_is_not_a_change(tmp_path: Path) -> None:
    base = _install_files(tmp_path, FILES)
    fields = ("size", "mtime_ns")
    _verifier(tmp_path, _component(FILES), fields=fields).verify_model("tiny-model")
    target = base / "model_index.json"
    stat = target.stat()
    replacement = target.with_name("copy")
    replacement.write_bytes(target.read_bytes())
    os.replace(replacement, target)  # a new inode, the same bytes
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    hasher = CountingHasher()
    _verifier(tmp_path, _component(FILES), hasher, fields=fields).verify_model("tiny-model")
    assert hasher.paths == []


def test_an_unreadable_stamps_file_vouches_for_nothing(tmp_path: Path) -> None:
    _install_files(tmp_path, FILES)
    (tmp_path / "stamps.json").write_text("not json", encoding="utf-8")
    hasher = CountingHasher()
    _verifier(tmp_path, _component(FILES), hasher).verify_model("tiny-model")
    assert len(hasher.paths) == 2


def test_progress_is_visible_while_hashing(tmp_path: Path) -> None:
    _install_files(tmp_path, FILES)
    seen = []
    verifier = _verifier(tmp_path, _component(FILES))

    def hasher(path: Path, progress=None) -> str:
        seen.append((verifier.progress.active, verifier.progress.total_bytes))
        return w.sha256_file(path, progress)

    verifier.hasher = hasher
    verifier.verify_model("tiny-model")
    assert seen and all(active == 1 for active, _ in seen)
    assert seen[0][1] == sum(len(data) for data in FILES.values())
    assert verifier.progress.active == 0 and verifier.progress.fraction == 1.0


# --- install ----------------------------------------------------------------------------------


def _fetcher(files: dict[str, bytes], calls: list[str]):
    def fetch(component, entry, root: Path) -> Path:
        calls.append(entry["path"])
        target = w.component_dir(root, component) / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[entry["path"]])
        return target

    return fetch


def test_install_fetches_verifies_and_stamps(tmp_path: Path) -> None:
    component = _component(FILES)
    verifier = _verifier(tmp_path, component)
    calls: list[str] = []
    report = w.install(
        tmp_path, [component], verifier, fetcher=_fetcher(FILES, calls), log=lambda _: None
    )
    assert sorted(calls) == sorted(FILES) and report.installed == ("tiny-model",)
    assert report.bytes_fetched == sum(len(data) for data in FILES.values())
    again = w.install(
        tmp_path,
        [component],
        _verifier(tmp_path, component),
        fetcher=_fetcher(FILES, calls),
        log=lambda _: None,
    )
    assert again.already_present == ("tiny-model",) and again.bytes_fetched == 0


def test_install_refuses_when_the_volume_cannot_hold_it(tmp_path: Path) -> None:
    component = _component(FILES)
    calls: list[str] = []
    with pytest.raises(OSError, match="refusing to install"):
        w.install(
            tmp_path,
            [component],
            _verifier(tmp_path, component),
            fetcher=_fetcher(FILES, calls),
            free_bytes=lambda _: 10,
            log=lambda _: None,
        )
    assert calls == []


def test_install_deletes_a_download_that_is_not_the_pinned_bytes(tmp_path: Path) -> None:
    component = _component(FILES)
    corrupt = {**FILES, "model_index.json": b"[]"}
    with pytest.raises(w.WeightsError, match="downloaded as"):
        w.install(
            tmp_path,
            [component],
            _verifier(tmp_path, component),
            fetcher=_fetcher(corrupt, []),
            log=lambda _: None,
        )
    assert not (w.component_dir(tmp_path, component) / "model_index.json").exists()


def test_the_hub_install_refuses_to_run_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    component = _component(FILES)
    with pytest.raises(OSError, match="HF_HUB_OFFLINE"):
        w.default_fetcher(component, component["files"][0], tmp_path)
