"""Pytest configuration for integration tests."""

import pytest


@pytest.fixture(autouse=True)
def integration_test_marker(request):
    """Automatically mark all tests in integration as integration tests."""
    if "integration" in request.fspath.dirname:
        request.node.add_marker(pytest.mark.integration)


# Add integration marker
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
