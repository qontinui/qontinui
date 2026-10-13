"""Tests for scripts/enforce_package_owners.py, the Xlib/cv2 install-race repair."""

import base64
import hashlib
import importlib.util
import sys
import tomllib
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "enforce_package_owners.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("enforce_package_owners", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before executing, as @dataclass resolves its module through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


epo = _load_script()


def _record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _install_fake_owner(root: Path, files: dict[str, bytes]) -> metadata.Distribution:
    """Lay out ``files`` plus a dist-info for ``fake-owner`` whose RECORD hashes them."""
    dist_info = root / "fake_owner-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: fake-owner\nVersion: 1.0\n")
    rows = []
    for relative, data in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        rows.append(f"{relative},{_record_hash(data)},{len(data)}")
    rows.append("fake_owner-1.0.dist-info/RECORD,,")
    (dist_info / "RECORD").write_text("\n".join(rows) + "\n")
    return metadata.PathDistribution(dist_info)


def _write_lock(path: Path, name: str, version: str, hashes: list[str]) -> Path:
    files = ", ".join(f'{{file = "f{i}", hash = "{h}"}}' for i, h in enumerate(hashes))
    path.write_text(f'[[package]]\nname = "{name}"\nversion = "{version}"\nfiles = [{files}]\n')
    return path


def test_intact_owner_has_no_damage_and_other_packages_are_ignored(tmp_path):
    dist = _install_fake_owner(
        tmp_path,
        {"Pkg/__init__.py": b"x = 1\n", "Pkg/sub/mod.py": b"y = 2\n", "Other/a.py": b"z\n"},
    )
    assert epo.damaged_files(dist, "Pkg") == (2, [])


def test_torn_and_missing_files_are_reported(tmp_path):
    owner_copy = b"__all__ = [\n    'display',\n    'rq',\n    ]\n"
    dist = _install_fake_owner(
        tmp_path, {"Pkg/protocol/__init__.py": owner_copy, "Pkg/X.py": b"A=1\n"}
    )
    # What two overlapping writers leave: the rival's shorter copy, then the owner's tail.
    rival_copy = b"__all__ = ['display', 'rq']\n"
    torn = rival_copy + owner_copy[len(rival_copy) :]
    (tmp_path / "Pkg/protocol/__init__.py").write_bytes(torn)
    (tmp_path / "Pkg/X.py").unlink()

    checked, damaged = epo.damaged_files(dist, "Pkg")

    assert checked == 2
    assert sorted(damaged) == ["Pkg/X.py", "Pkg/protocol/__init__.py"]


def test_bytecode_is_not_held_to_the_record(tmp_path):
    pyc = "Pkg/__pycache__/m.cpython-312.pyc"
    dist = _install_fake_owner(tmp_path, {"Pkg/m.py": b"z = 3\n", pyc: b"compiled at install"})
    (tmp_path / pyc).write_bytes(b"recompiled on import")
    assert epo.damaged_files(dist, "Pkg") == (1, [])


def test_locked_requirement_pins_the_version_with_every_artifact_hash(tmp_path):
    lock = _write_lock(tmp_path / "poetry.lock", "python-xlib", "0.33", ["sha256:aa", "sha256:bb"])
    requirement = epo.locked_requirement(lock, "Python_Xlib", "0.33")
    assert requirement == "python-xlib==0.33 --hash=sha256:aa --hash=sha256:bb"


def test_a_lock_for_another_version_is_refused_instead_of_changing_the_env(
    tmp_path, monkeypatch, capsys
):
    _install_fake_owner(tmp_path, {"Pkg/__init__.py": b"value = 'owner'\n"})
    monkeypatch.syspath_prepend(str(tmp_path))
    lock = _write_lock(tmp_path / "poetry.lock", "fake-owner", "2.0", ["sha256:dd"])
    contest = epo.Contest(package="Pkg", owner="fake-owner", rivals=())
    (tmp_path / "Pkg/__init__.py").write_bytes(b"value = 'rival'\n")
    requested = []
    monkeypatch.setattr(epo, "reinstall", requested.append)

    assert epo.enforce(contest, lock, check_only=False) is False
    assert requested == []
    assert "locks fake-owner 2.0, not the installed 1.0" in capsys.readouterr().out


def test_enforce_repairs_then_reverifies_and_check_mode_changes_nothing(
    tmp_path, monkeypatch, capsys
):
    owner_copy = b"value = 'owner'\n"
    _install_fake_owner(tmp_path, {"Pkg/__init__.py": owner_copy})
    monkeypatch.syspath_prepend(str(tmp_path))
    lock = _write_lock(tmp_path / "poetry.lock", "fake-owner", "1.0", ["sha256:cc"])
    contest = epo.Contest(package="Pkg", owner="fake-owner", rivals=("fake-rival",))
    (tmp_path / "Pkg/__init__.py").write_bytes(b"value = 'rival'\n")  # the rival wrote last
    requested = []

    def reinstall(requirement):
        requested.append(requirement)
        (tmp_path / "Pkg/__init__.py").write_bytes(owner_copy)

    monkeypatch.setattr(epo, "reinstall", reinstall)

    assert epo.enforce(contest, lock, check_only=True) is False
    assert requested == []
    assert epo.enforce(contest, lock, check_only=False) is True
    assert requested == ["fake-owner==1.0 --hash=sha256:cc"]
    assert (tmp_path / "Pkg/__init__.py").read_bytes() == owner_copy
    assert "repaired - reinstalled fake-owner 1.0" in capsys.readouterr().out


def test_enforce_fails_when_the_reinstall_leaves_damage(tmp_path, monkeypatch):
    _install_fake_owner(tmp_path, {"Pkg/__init__.py": b"value = 'owner'\n"})
    monkeypatch.syspath_prepend(str(tmp_path))
    lock = _write_lock(tmp_path / "poetry.lock", "fake-owner", "1.0", ["sha256:cc"])
    contest = epo.Contest(package="Pkg", owner="fake-owner", rivals=())
    (tmp_path / "Pkg/__init__.py").write_bytes(b"value = 'rival'\n")
    monkeypatch.setattr(epo, "reinstall", lambda requirement: None)

    assert epo.enforce(contest, lock, check_only=False) is False


def test_every_contest_is_still_in_the_lock():
    """The repair installs the owner from poetry.lock, so it must be locked. A rival
    that left the lock means the collision is gone: delete that contest entry."""
    lock = tomllib.loads((REPO_ROOT / "poetry.lock").read_text(encoding="utf-8"))
    locked = {epo.canonical(entry["name"]) for entry in lock["package"]}
    for contest in epo.CONTESTS:
        assert epo.canonical(contest.owner) in locked, contest
        for rival in contest.rivals:
            assert epo.canonical(rival) in locked, f"{rival} left poetry.lock: delete {contest}"
