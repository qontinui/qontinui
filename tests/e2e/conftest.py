"""Live E2E conftest.

Live E2E tests launch real native applications (notepad, mspaint) and
capture real screenshots. They are opt-in via ``--run-live-e2e`` to keep
the default suite fast and hermetic.
"""

from __future__ import annotations

from typing import Any

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live-e2e",
        action="store_true",
        default=False,
        help="Run live E2E tests that launch real applications and a real "
        "OmniParser service container.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live_e2e: live end-to-end test requiring a real display, real "
        "applications, and (optionally) a running OmniParser service",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-live-e2e"):
        return
    skip_live = pytest.mark.skip(reason="live_e2e tests require --run-live-e2e flag")
    for item in items:
        if "live_e2e" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def _restore_real_display(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Undo the headless env vars set by the root conftest.

    The root ``tests/conftest.py`` pins ``DISPLAY=:99`` and
    ``XAUTHORITY`` via an autouse fixture. For live E2E tests we
    want the real display instead.
    """
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("XAUTHORITY", raising=False)
    yield
