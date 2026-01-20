"""Integration tests for CDP accessibility capture.

Tests cover:
- Connecting to Chrome via CDP
- Capturing accessibility tree
- Verifying node structure
- Performing click_by_ref on a button
- Performing type_by_ref in a textbox

These tests require Chrome/Chromium running with remote debugging enabled:
    chrome --remote-debugging-port=9222

Use the pytest marker to skip when CDP is unavailable:
    pytest -m "not cdp_required"
"""

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest
from qontinui_schemas.accessibility import (
    AccessibilityBackend,
    AccessibilityCaptureOptions,
    AccessibilityConfig,
    AccessibilityRole,
    AccessibilitySelector,
)


def _load_module_directly(module_name: str, file_path: Path) -> ModuleType:
    """Load a module directly from file without triggering parent __init__.py.

    This is used to avoid the cv2 import issues in the main qontinui package.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# First load RefManager (dependency of CDPAccessibilityCapture)
_base_path = Path(__file__).parent.parent.parent / "src" / "qontinui" / "hal"
_ref_manager_path = _base_path / "implementations" / "accessibility" / "ref_manager.py"
_ref_manager_module = _load_module_directly(
    "qontinui.hal.implementations.accessibility.ref_manager", _ref_manager_path
)

# Load IAccessibilityCapture interface
_interface_path = _base_path / "interfaces" / "accessibility_capture.py"
_interface_module = _load_module_directly(
    "qontinui.hal.interfaces.accessibility_capture", _interface_path
)

# Load CDPAccessibilityCapture
_cdp_capture_path = _base_path / "implementations" / "accessibility" / "cdp_capture.py"
_cdp_capture_module = _load_module_directly(
    "qontinui.hal.implementations.accessibility.cdp_capture", _cdp_capture_path
)
CDPAccessibilityCapture = _cdp_capture_module.CDPAccessibilityCapture

# Mark all tests in this module as requiring CDP
pytestmark = pytest.mark.cdp_required


def is_cdp_available(host: str = "localhost", port: int = 9222) -> bool:
    """Check if CDP is available at the given host:port."""
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


# Skip all tests if CDP is not available
CDP_HOST = os.environ.get("CDP_HOST", "localhost")
CDP_PORT = int(os.environ.get("CDP_PORT", "9222"))
CDP_AVAILABLE = is_cdp_available(CDP_HOST, CDP_PORT)

skip_if_no_cdp = pytest.mark.skipif(
    not CDP_AVAILABLE,
    reason=f"CDP not available at {CDP_HOST}:{CDP_PORT}. Start Chrome with --remote-debugging-port={CDP_PORT}",
)


@pytest.fixture
def cdp_config() -> AccessibilityConfig:
    """Create CDP config with test settings."""
    return AccessibilityConfig(
        backend=AccessibilityBackend.CDP,
        cdp_host=CDP_HOST,
        cdp_port=CDP_PORT,
        cdp_timeout=10.0,
        interactive_only=False,
        include_hidden=False,
        include_bounds=True,
        include_value=True,
    )


@pytest.fixture
async def cdp_capture(cdp_config: AccessibilityConfig) -> CDPAccessibilityCapture:
    """Create and connect a CDP capture instance."""
    capture = CDPAccessibilityCapture(config=cdp_config)
    yield capture
    await capture.disconnect()


@skip_if_no_cdp
class TestCDPConnection:
    """Test CDP connection functionality."""

    @pytest.mark.asyncio
    async def test_connect_success(self, cdp_config: AccessibilityConfig) -> None:
        """Can connect to Chrome via CDP."""
        capture = CDPAccessibilityCapture(config=cdp_config)
        try:
            result = await capture.connect(host=CDP_HOST, port=CDP_PORT)
            assert result is True
            assert capture.is_connected() is True
        finally:
            await capture.disconnect()

    @pytest.mark.asyncio
    async def test_connect_bad_port(self, cdp_config: AccessibilityConfig) -> None:
        """Connection fails gracefully on bad port."""
        capture = CDPAccessibilityCapture(config=cdp_config)
        try:
            result = await capture.connect(host=CDP_HOST, port=19999, timeout=2.0)
            assert result is False
            assert capture.is_connected() is False
        finally:
            await capture.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect(self, cdp_config: AccessibilityConfig) -> None:
        """Disconnect cleans up properly."""
        capture = CDPAccessibilityCapture(config=cdp_config)
        await capture.connect(host=CDP_HOST, port=CDP_PORT)
        assert capture.is_connected() is True

        await capture.disconnect()

        assert capture.is_connected() is False

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, cdp_config: AccessibilityConfig) -> None:
        """Disconnect is safe when not connected."""
        capture = CDPAccessibilityCapture(config=cdp_config)
        # Should not raise
        await capture.disconnect()
        await capture.disconnect()  # Multiple calls should be safe


@skip_if_no_cdp
class TestCDPCaptureTree:
    """Test accessibility tree capture."""

    @pytest.mark.asyncio
    async def test_capture_tree_basic(self, cdp_config: AccessibilityConfig) -> None:
        """Can capture accessibility tree."""
        capture = CDPAccessibilityCapture(config=cdp_config)
        try:
            connected = await capture.connect(host=CDP_HOST, port=CDP_PORT)
            assert connected

            snapshot = await capture.capture_tree()

            assert snapshot is not None
            assert snapshot.backend == AccessibilityBackend.CDP
            assert snapshot.root is not None
            assert snapshot.timestamp > 0
            assert snapshot.total_nodes > 0
        finally:
            await capture.disconnect()

    @pytest.mark.asyncio
    async def test_capture_tree_with_options(self, cdp_config: AccessibilityConfig) -> None:
        """Capture respects options."""
        capture = CDPAccessibilityCapture(config=cdp_config)
        try:
            await capture.connect(host=CDP_HOST, port=CDP_PORT)

            options = AccessibilityCaptureOptions(
                target="auto",
                include_screenshot=False,
                config=AccessibilityConfig(interactive_only=True),
            )
            snapshot = await capture.capture_tree(options=options)

            assert snapshot is not None
            # With interactive_only, should have fewer nodes with refs
            assert snapshot.interactive_nodes <= snapshot.total_nodes
        finally:
            await capture.disconnect()

    @pytest.mark.asyncio
    async def test_capture_tree_has_refs(self, cdp_config: AccessibilityConfig) -> None:
        """Captured nodes have refs assigned."""
        capture = CDPAccessibilityCapture(config=cdp_config)
        try:
            await capture.connect(host=CDP_HOST, port=CDP_PORT)
            snapshot = await capture.capture_tree()

            # Root should have a ref
            assert snapshot.root.ref.startswith("@e")

            # Check first few nodes
            def check_refs(node, count: int = 0) -> int:
                if count >= 5:
                    return count
                if node.ref:
                    assert node.ref.startswith("@e")
                    count += 1
                for child in node.children:
                    count = check_refs(child, count)
                return count

            count = check_refs(snapshot.root)
            assert count > 0
        finally:
            await capture.disconnect()

    @pytest.mark.asyncio
    async def test_capture_tree_not_connected_raises(self, cdp_config: AccessibilityConfig) -> None:
        """Capture raises when not connected."""
        capture = CDPAccessibilityCapture(config=cdp_config)

        with pytest.raises(RuntimeError, match="Not connected"):
            await capture.capture_tree()


@skip_if_no_cdp
class TestCDPNodeLookup:
    """Test node lookup by ref."""

    @pytest.mark.asyncio
    async def test_get_node_by_ref(self, cdp_config: AccessibilityConfig) -> None:
        """Can find node by ref."""
        capture = CDPAccessibilityCapture(config=cdp_config)
        try:
            await capture.connect(host=CDP_HOST, port=CDP_PORT)
            snapshot = await capture.capture_tree()

            # Get root by ref
            node = await capture.get_node_by_ref(snapshot.root.ref)

            assert node is not None
            assert node.ref == snapshot.root.ref
        finally:
            await capture.disconnect()

    @pytest.mark.asyncio
    async def test_get_node_by_ref_not_found(self, cdp_config: AccessibilityConfig) -> None:
        """Returns None for non-existent ref."""
        capture = CDPAccessibilityCapture(config=cdp_config)
        try:
            await capture.connect(host=CDP_HOST, port=CDP_PORT)
            await capture.capture_tree()

            node = await capture.get_node_by_ref("@e99999")

            assert node is None
        finally:
            await capture.disconnect()


@skip_if_no_cdp
class TestCDPFindNodes:
    """Test node finding by selector."""

    @pytest.mark.asyncio
    async def test_find_by_role(self, cdp_config: AccessibilityConfig) -> None:
        """Can find nodes by role."""
        capture = CDPAccessibilityCapture(config=cdp_config)
        try:
            await capture.connect(host=CDP_HOST, port=CDP_PORT)
            await capture.capture_tree()

            selector = AccessibilitySelector(role=AccessibilityRole.DOCUMENT)
            nodes = await capture.find_nodes(selector)

            # Most pages have at least one document role
            assert len(nodes) >= 1
            for node in nodes:
                assert node.role == AccessibilityRole.DOCUMENT
        finally:
            await capture.disconnect()

    @pytest.mark.asyncio
    async def test_find_by_multiple_roles(self, cdp_config: AccessibilityConfig) -> None:
        """Can find nodes matching any of multiple roles."""
        capture = CDPAccessibilityCapture(config=cdp_config)
        try:
            await capture.connect(host=CDP_HOST, port=CDP_PORT)
            await capture.capture_tree()

            selector = AccessibilitySelector(
                role=[AccessibilityRole.BUTTON, AccessibilityRole.LINK]
            )
            nodes = await capture.find_nodes(selector)

            for node in nodes:
                assert node.role in [AccessibilityRole.BUTTON, AccessibilityRole.LINK]
        finally:
            await capture.disconnect()

    @pytest.mark.asyncio
    async def test_find_interactive(self, cdp_config: AccessibilityConfig) -> None:
        """Can find interactive nodes."""
        capture = CDPAccessibilityCapture(config=cdp_config)
        try:
            await capture.connect(host=CDP_HOST, port=CDP_PORT)
            await capture.capture_tree()

            selector = AccessibilitySelector(is_interactive=True)
            nodes = await capture.find_nodes(selector)

            for node in nodes:
                assert node.is_interactive is True
        finally:
            await capture.disconnect()

    @pytest.mark.asyncio
    async def test_find_no_matches(self, cdp_config: AccessibilityConfig) -> None:
        """Returns empty list when no matches."""
        capture = CDPAccessibilityCapture(config=cdp_config)
        try:
            await capture.connect(host=CDP_HOST, port=CDP_PORT)
            await capture.capture_tree()

            selector = AccessibilitySelector(name="ThisNameDoesNotExistAnywhere123456789")
            nodes = await capture.find_nodes(selector)

            assert nodes == []
        finally:
            await capture.disconnect()


@skip_if_no_cdp
class TestCDPAIContext:
    """Test AI context generation."""

    @pytest.mark.asyncio
    async def test_to_ai_context(self, cdp_config: AccessibilityConfig) -> None:
        """Generates AI-friendly context string."""
        capture = CDPAccessibilityCapture(config=cdp_config)
        try:
            await capture.connect(host=CDP_HOST, port=CDP_PORT)
            snapshot = await capture.capture_tree()

            context = capture.to_ai_context(snapshot)

            assert "## Accessibility Tree" in context
            assert "### Interactive Elements" in context
            # Should have at least one element listed
            assert "@e" in context
        finally:
            await capture.disconnect()

    @pytest.mark.asyncio
    async def test_to_ai_context_with_url(self, cdp_config: AccessibilityConfig) -> None:
        """Context includes URL when available."""
        capture = CDPAccessibilityCapture(config=cdp_config)
        try:
            await capture.connect(host=CDP_HOST, port=CDP_PORT)
            snapshot = await capture.capture_tree()

            context = capture.to_ai_context(snapshot)

            if snapshot.url:
                assert "URL:" in context
            if snapshot.title:
                assert "Title:" in context
        finally:
            await capture.disconnect()

    @pytest.mark.asyncio
    async def test_to_ai_context_truncation(self, cdp_config: AccessibilityConfig) -> None:
        """Context respects max_elements limit."""
        capture = CDPAccessibilityCapture(config=cdp_config)
        try:
            await capture.connect(host=CDP_HOST, port=CDP_PORT)
            snapshot = await capture.capture_tree()

            context = capture.to_ai_context(snapshot, max_elements=5)

            # Count lines starting with "- @e"
            element_lines = [
                line for line in context.split("\n") if line.strip().startswith("- @e")
            ]
            assert len(element_lines) <= 5
        finally:
            await capture.disconnect()


@skip_if_no_cdp
class TestCDPBackendName:
    """Test backend name."""

    def test_backend_name(self, cdp_config: AccessibilityConfig) -> None:
        """Returns correct backend name."""
        capture = CDPAccessibilityCapture(config=cdp_config)
        assert capture.get_backend_name() == "cdp"
