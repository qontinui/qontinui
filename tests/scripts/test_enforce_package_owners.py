"""Tests for scripts/enforce_package_owners.py, the Xlib/cv2 install-race repair."""

import base64
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import tomllib
import zipfile
from importlib import metadata
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "enforce_package_owners.py"
OWNER_COPY = b"value = 'owner'\n"


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


def _install_fake_owner(
    root: Path, files: dict[str, bytes], *, hashed: bool = True
) -> metadata.Distribution:
    """Lay out ``files`` plus a dist-info for ``fake-owner`` 1.0 whose RECORD lists them."""
    dist_info = root / "fake_owner-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: fake-owner\nVersion: 1.0\n")
    rows = []
    for relative, data in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        rows.append(f"{relative},{_record_hash(data) if hashed else ''},{len(data)}")
    rows.append("fake_owner-1.0.dist-info/RECORD,,")
    (dist_info / "RECORD").write_text("\n".join(rows) + "\n")
    return metadata.PathDistribution(dist_info)


def _write_lock(path: Path, name: str, version: str, artifacts: list[tuple[str, str]]) -> Path:
    files = ", ".join(f'{{file = "{file}", hash = "{digest}"}}' for file, digest in artifacts)
    path.write_text(f'[[package]]\nname = "{name}"\nversion = "{version}"\nfiles = [{files}]\n')
    return path


def _fake_wheel(directory: Path) -> Path:
    """A minimal, valid ``fake-owner`` 1.0 wheel, for pip to find."""
    directory.mkdir(parents=True, exist_ok=True)
    wheel = directory / "fake_owner-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("fake_owner/__init__.py", "")
        info = "fake_owner-1.0.dist-info"
        archive.writestr(
            f"{info}/METADATA", "Metadata-Version: 2.1\nName: fake-owner\nVersion: 1.0\n"
        )
        archive.writestr(
            f"{info}/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{info}/RECORD", "")
    return wheel


@pytest.fixture
def damaged_owner(tmp_path, monkeypatch):
    """``fake-owner`` 1.0 on sys.path, its one file overwritten by a rival; empty Poetry cache."""
    _install_fake_owner(tmp_path, {"Pkg/__init__.py": OWNER_COPY})
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("POETRY_CACHE_DIR", str(tmp_path / "poetry-cache"))
    (tmp_path / "Pkg/__init__.py").write_bytes(b"value = 'rival'\n")
    return tmp_path


def _recording_reinstall(root: Path, calls: list, repair: bool = True):
    def reinstall(requirement, find_links):
        calls.append((requirement, list(find_links)))
        if repair:
            (root / "Pkg/__init__.py").write_bytes(OWNER_COPY)

    return reinstall


CONTEST = epo.Contest(package="Pkg", owner="fake-owner", rivals=("fake-rival",))


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


def test_requirement_pins_the_version_with_every_locked_hash(tmp_path):
    lock = _write_lock(
        tmp_path / "poetry.lock",
        "python-xlib",
        "0.33",
        [("a.whl", "sha256:aa"), ("a.tar.gz", "sha256:bb")],
    )
    entry = epo.locked_entry(lock, "Python_Xlib", "0.33")
    assert epo.requirement_line(entry) == "python-xlib==0.33 --hash=sha256:aa --hash=sha256:bb"


def test_lock_matching_names_a_missing_entry_apart_from_duplicate_ones(tmp_path):
    lock = _write_lock(tmp_path / "poetry.lock", "fake-owner", "2.0", [("f.whl", "sha256:dd")])
    at_lock = re.escape(f"{lock}: ")
    with pytest.raises(
        epo.LockError,
        match=at_lock + r"no matching lock entry for fake-owner 1\.0 \(it locks 2\.0\)",
    ):
        epo.locked_entry(lock, "fake-owner", "1.0")
    with pytest.raises(
        epo.LockError, match=at_lock + r"no matching lock entry for other 1\.0 \(it locks none\)"
    ):
        epo.locked_entry(lock, "other", "1.0")
    twice = tmp_path / "twice.lock"
    twice.write_text(lock.read_text() * 2)
    with pytest.raises(
        epo.LockError,
        match=re.escape(f"{twice}: ") + r"several matching lock entries for fake-owner 2\.0 \(2\)",
    ):
        epo.locked_entry(twice, "fake-owner", "2.0")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "cannot be read"),
        ("[[package]\n", "is not valid TOML"),
        ('[metadata]\nlock-version = "2.1"\n', "has no [[package]] table, so it locks nothing"),
        (
            '[[package]]\nname = "fake-owner"\nversion = "1.0"\nfiles = "not-a-list"\n',
            "the fake-owner 1.0 entry has files that are not a list",
        ),
        ("package = 3\n", "a [[package]] entry is malformed"),
        (
            '[[package]]\nname = "fake-owner"\nversion = "1.0"\n'
            'files = [{file = "f.whl", hash = "nohash"}]\n',
            "locks f.whl with hash 'nohash'; pip accepts only sha256, sha384, sha512",
        ),
    ],
    ids=[
        "unreadable",
        "bad-toml",
        "no-package-table",
        "files-not-a-list",
        "package-not-an-array",
        "bad-hash",
    ],
)
def test_every_lock_problem_is_a_lock_error_that_names_the_lock(tmp_path, content, message):
    lock = tmp_path / "poetry.lock"
    if content is not None:
        lock.write_text(content)

    with pytest.raises(epo.LockError) as raised:
        epo.locked_entry(lock, "fake-owner", "1.0")

    assert str(raised.value).startswith(f"{lock}: ")
    assert message in str(raised.value)


def test_a_lock_for_another_version_is_refused_instead_of_changing_the_env(
    damaged_owner, monkeypatch, capsys
):
    lock = _write_lock(damaged_owner / "poetry.lock", "fake-owner", "2.0", [("f.whl", "sha256:dd")])
    calls = []
    monkeypatch.setattr(epo, "reinstall", _recording_reinstall(damaged_owner, calls))

    assert epo.enforce(CONTEST, lock, check_only=False) is False
    assert calls == []
    assert (
        f"{lock}: no matching lock entry for fake-owner 1.0 (it locks 2.0)"
        in capsys.readouterr().out
    )


def test_a_malformed_lock_is_reported_not_raised(damaged_owner, monkeypatch, capsys):
    lock = damaged_owner / "poetry.lock"
    lock.write_text('[[package]]\nname = "fake-owner"\nversion = "1.0"\nfiles = "not-a-list"\n')
    calls = []
    monkeypatch.setattr(epo, "reinstall", _recording_reinstall(damaged_owner, calls))

    assert epo.enforce(CONTEST, lock, check_only=False) is False
    assert calls == []
    assert (
        f"{lock}: the fake-owner 1.0 entry has files that are not a list" in capsys.readouterr().out
    )


_XLIB_ENTRY = b'[[package]]\nname = "python-xlib"\nversion = "0.33"\n'


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"\xff\xfe" + _XLIB_ENTRY, "is not UTF-8 text"),
        (
            _XLIB_ENTRY + b'files = [{hash = "sha256:' + b"a" * 64 + b'"}]\n',
            "the python-xlib 0.33 entry lists an artifact that lacks a file or a hash",
        ),
        (
            _XLIB_ENTRY + b'files = [{file = "x.whl"}]\n',
            "the python-xlib 0.33 entry lists an artifact that lacks a file or a hash",
        ),
        (
            _XLIB_ENTRY + b'files = [{file = "x.whl", hash = "sha256:"}]\n',
            "the python-xlib 0.33 entry locks x.whl with an empty sha256 digest",
        ),
        (
            _XLIB_ENTRY + b'files = [{file = "x.whl", hash = "shake_128:abcd"}]\n',
            "with hash 'shake_128:abcd'; pip accepts only sha256, sha384, sha512",
        ),
        (
            _XLIB_ENTRY + b'files = [{file = "x.whl", hash = "md5:abcd"}]\n',
            "with hash 'md5:abcd'; pip accepts only sha256, sha384, sha512",
        ),
        # tomllib raises RecursionError on these (the dotted key only on some releases;
        # where it parses, the lock has no [[package]] table). Either way: refused, named.
        (b"x = " + b"[" * 1000 + b"]" * 1000 + b"\n", ""),
        (b"a" + b".a" * 1000 + b" = 1\n", ""),
    ],
    ids=[
        "invalid-utf8",
        "artifact-without-file",
        "artifact-without-hash",
        "empty-sha256-digest",
        "shake_128-hash",
        "md5-hash",
        "arrays-nested-1000-deep",
        "key-with-1001-dotted-parts",
    ],
)
def test_a_hand_made_lock_fails_the_cli_with_exit_1_and_no_traceback(tmp_path, content, message):
    """Runs the script itself, as CI does, over an environment whose python-xlib is
    damaged (a dist-info placed ahead of any real one), so the repair reads the lock."""
    site = tmp_path / "site"
    dist_info = site / "python_xlib-0.33.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: python-xlib\nVersion: 0.33\n")
    (dist_info / "RECORD").write_text(f"Xlib/__init__.py,{_record_hash(b'owner')},5\n")
    (site / "Xlib").mkdir()
    (site / "Xlib" / "__init__.py").write_bytes(b"rival")
    lock = tmp_path / "poetry.lock"
    lock.write_bytes(content)
    env = {**os.environ, "PYTHONPATH": str(site), "POETRY_CACHE_DIR": str(tmp_path / "cache")}

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--lock", str(lock)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "Traceback" not in result.stdout + result.stderr
    reports = [
        line.split("): ", 1)[1]
        for line in result.stdout.splitlines()
        if line.startswith("Xlib/ (owner python-xlib 0.33") and "): " in line
    ]
    assert any(report.startswith(f"{lock}: ") and message in report for report in reports), (
        result.stdout
    )


def test_enforce_repairs_then_reverifies_and_check_mode_changes_nothing(
    damaged_owner, monkeypatch, capsys
):
    lock = _write_lock(damaged_owner / "poetry.lock", "fake-owner", "1.0", [("f.whl", "sha256:cc")])
    calls = []
    monkeypatch.setattr(epo, "reinstall", _recording_reinstall(damaged_owner, calls))

    assert epo.enforce(CONTEST, lock, check_only=True) is False
    assert calls == []
    assert epo.enforce(CONTEST, lock, check_only=False) is True
    assert calls == [("fake-owner==1.0 --hash=sha256:cc", [])]
    assert (damaged_owner / "Pkg/__init__.py").read_bytes() == OWNER_COPY
    assert "repaired - reinstalled fake-owner 1.0 from the package index" in capsys.readouterr().out


def test_enforce_fails_when_the_reinstall_leaves_damage(damaged_owner, monkeypatch):
    lock = _write_lock(damaged_owner / "poetry.lock", "fake-owner", "1.0", [("f.whl", "sha256:cc")])
    calls = []
    monkeypatch.setattr(epo, "reinstall", _recording_reinstall(damaged_owner, calls, repair=False))

    assert epo.enforce(CONTEST, lock, check_only=False) is False
    assert len(calls) == 1


def test_repair_prefers_the_wheel_poetry_already_cached(damaged_owner, monkeypatch, capsys):
    wheel = _fake_wheel(damaged_owner / "poetry-cache" / "artifacts" / "ab" / "cd")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    lock = _write_lock(
        damaged_owner / "poetry.lock",
        "fake-owner",
        "1.0",
        [(wheel.name, f"sha256:{digest}"), ("fake_owner-1.0.tar.gz", "sha256:00")],
    )
    calls = []
    monkeypatch.setattr(epo, "reinstall", _recording_reinstall(damaged_owner, calls))

    assert epo.enforce(CONTEST, lock, check_only=False) is True
    assert calls == [
        (epo.requirement_line(epo.locked_entry(lock, "fake-owner", "1.0")), [wheel.parent])
    ]
    assert (
        "repaired - reinstalled fake-owner 1.0 from Poetry's artifact cache"
        in capsys.readouterr().out
    )


def test_repair_falls_back_to_the_index_when_no_cached_wheel_matches_the_lock(
    damaged_owner, monkeypatch, capsys
):
    wheel = _fake_wheel(damaged_owner / "poetry-cache" / "artifacts" / "ab" / "cd")
    # Same file name as the locked wheel, but not the locked bytes.
    lock = _write_lock(
        damaged_owner / "poetry.lock", "fake-owner", "1.0", [(wheel.name, "sha256:" + "0" * 64)]
    )
    calls = []
    monkeypatch.setattr(epo, "reinstall", _recording_reinstall(damaged_owner, calls))

    assert epo.enforce(CONTEST, lock, check_only=False) is True
    assert calls[0][1] == []
    assert "reinstalled fake-owner 1.0 from the package index" in capsys.readouterr().out


def _cache_holding_the_locked_wheel(root: Path) -> tuple[Path, Path]:
    wheel = _fake_wheel(root / "poetry-cache" / "artifacts" / "ab" / "cd")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    lock = _write_lock(
        root / "poetry.lock", "fake-owner", "1.0", [(wheel.name, f"sha256:{digest}")]
    )
    return wheel, lock


def test_a_cached_wheel_pip_cannot_install_is_retried_once_from_the_index(
    damaged_owner, monkeypatch, capsys
):
    wheel, lock = _cache_holding_the_locked_wheel(damaged_owner)
    calls = []

    def reinstall(requirement, find_links):
        calls.append((requirement, list(find_links)))
        if find_links:  # another platform's wheel: pip --no-index finds nothing to install
            raise subprocess.CalledProcessError(1, ["pip"])
        (damaged_owner / "Pkg/__init__.py").write_bytes(OWNER_COPY)

    monkeypatch.setattr(epo, "reinstall", reinstall)

    assert epo.enforce(CONTEST, lock, check_only=False) is True
    requirement = epo.requirement_line(epo.locked_entry(lock, "fake-owner", "1.0"))
    assert calls == [(requirement, [wheel.parent]), (requirement, [])]
    out = capsys.readouterr().out
    assert "retrying once from the package index, still hash-pinned" in out
    assert "repaired - reinstalled fake-owner 1.0 from the package index" in out


def test_the_index_is_tried_once_after_the_cache_and_its_failure_is_reported(
    damaged_owner, monkeypatch, capsys
):
    _, lock = _cache_holding_the_locked_wheel(damaged_owner)
    calls = []

    def reinstall(requirement, find_links):
        calls.append(list(find_links))
        raise subprocess.CalledProcessError(1, ["pip"])

    monkeypatch.setattr(epo, "reinstall", reinstall)

    assert epo.enforce(CONTEST, lock, check_only=False) is False
    assert len(calls) == 2
    assert calls[1] == []
    assert "reinstalling fake-owner failed" in capsys.readouterr().out


def test_a_cached_sdist_is_not_offered_to_pip(tmp_path):
    """Only wheels are taken from Poetry's cache: with --no-index, pip could not fetch
    the build backend an sdist needs."""
    cache = tmp_path / "artifacts"
    sdist = cache / "ef" / "fake_owner-1.0.tar.gz"
    sdist.parent.mkdir(parents=True)
    sdist.write_bytes(b"an sdist")
    digest = hashlib.sha256(b"an sdist").hexdigest()
    lock = _write_lock(
        tmp_path / "poetry.lock", "fake-owner", "1.0", [(sdist.name, f"sha256:{digest}")]
    )

    assert epo.cached_wheels(epo.locked_entry(lock, "fake-owner", "1.0"), cache) == []


def test_poetry_artifact_cache_follows_poetry_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("POETRY_CACHE_DIR", str(tmp_path))
    assert epo.poetry_artifact_cache() == tmp_path / "artifacts"


def test_reinstall_runs_pip_with_every_safety_flag_and_every_locked_hash(tmp_path, monkeypatch):
    lock = _write_lock(
        tmp_path / "poetry.lock",
        "fake-owner",
        "1.0",
        [("a.whl", "sha256:aa"), ("a.tar.gz", "sha256:bb")],
    )
    requirement = epo.requirement_line(epo.locked_entry(lock, "fake-owner", "1.0"))
    calls = []

    def run(command, **kwargs):
        written = Path(command[command.index("-r") + 1]).read_text()
        calls.append((command, kwargs, written))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(epo.subprocess, "run", run)

    epo.reinstall(requirement, [])
    epo.reinstall(requirement, [tmp_path / "wheels"])

    for command, kwargs, written in calls:
        assert command[:4] == [sys.executable, "-m", "pip", "install"]
        for flag in ("--force-reinstall", "--no-deps", "--require-hashes"):
            assert flag in command
        assert kwargs.get("check") is True
        assert written == "fake-owner==1.0 --hash=sha256:aa --hash=sha256:bb\n"
    assert "--no-index" not in calls[0][0]
    assert "--no-index" in calls[1][0]
    assert ["--find-links", str(tmp_path / "wheels")] == calls[1][0][-4:-2]


def test_pip_refuses_an_unhashed_requirement_because_of_require_hashes(tmp_path):
    """A lock entry without artifacts yields a requirement with no hash. pip installs
    such a requirement unless --require-hashes is passed, so this pins the flag."""
    lock = _write_lock(tmp_path / "poetry.lock", "fake-owner", "1.0", [])
    requirements = tmp_path / "owner.txt"
    requirements.write_text(
        epo.requirement_line(epo.locked_entry(lock, "fake-owner", "1.0")) + "\n"
    )
    wheels = _fake_wheel(tmp_path / "wheels").parent

    command = epo.pip_install_command(requirements, [wheels]) + ["--dry-run"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)

    assert result.returncode != 0, result.stdout
    assert "Hashes are required in --require-hashes mode" in result.stderr


def test_a_dist_info_without_a_name_is_skipped(tmp_path, monkeypatch):
    _install_fake_owner(tmp_path, {"Pkg/__init__.py": OWNER_COPY})
    nameless = tmp_path / "nameless-1.0.dist-info"
    nameless.mkdir()
    (nameless / "METADATA").write_text("Metadata-Version: 2.1\nVersion: 1.0\n")
    (nameless / "RECORD").write_text("Pkg/__init__.py,,\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    assert epo.other_writers("Pkg", "fake-owner") == []
    assert epo.enforce(CONTEST, tmp_path / "poetry.lock", check_only=True) is True


def _main(monkeypatch, lock: Path, *contests) -> int:
    monkeypatch.setattr(epo, "CONTESTS", contests)
    return epo.main(["--lock", str(lock)])


def test_main_exits_0_when_every_installed_owner_is_intact(tmp_path, monkeypatch):
    _install_fake_owner(tmp_path, {"Pkg/__init__.py": OWNER_COPY})
    monkeypatch.syspath_prepend(str(tmp_path))
    absent = epo.Contest(package="Gone", owner="no-such-distribution-installed", rivals=())

    assert _main(monkeypatch, tmp_path / "poetry.lock", CONTEST, absent) == 0


def test_main_exits_1_when_damage_is_left_unrepaired(damaged_owner, monkeypatch):
    lock = _write_lock(damaged_owner / "poetry.lock", "fake-owner", "1.0", [("f.whl", "sha256:cc")])
    monkeypatch.setattr(epo, "reinstall", _recording_reinstall(damaged_owner, [], repair=False))

    assert _main(monkeypatch, lock, CONTEST) == 1


def test_main_exits_1_when_an_owner_has_nothing_to_check(tmp_path, monkeypatch):
    _install_fake_owner(tmp_path, {"Pkg/__init__.py": OWNER_COPY}, hashed=False)
    monkeypatch.syspath_prepend(str(tmp_path))

    assert _main(monkeypatch, tmp_path / "poetry.lock", CONTEST) == 1


def test_every_contest_is_still_in_the_lock():
    """The repair installs the owner from poetry.lock, so it must be locked. A rival
    that left the lock means that collision is gone: drop it from the contest."""
    lock = tomllib.loads((REPO_ROOT / "poetry.lock").read_text(encoding="utf-8"))
    locked = {epo.canonical(entry["name"]) for entry in lock["package"]}
    for contest in epo.CONTESTS:
        assert epo.canonical(contest.owner) in locked, contest
        for rival in contest.rivals:
            assert epo.canonical(rival) in locked, (
                f"{rival} left poetry.lock: drop it from {contest}"
            )
