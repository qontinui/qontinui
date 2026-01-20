"""Tests for AWAS discovery service."""

import pytest

from qontinui.awas.discovery import AwasDiscoveryService
from qontinui.awas.types import AwasManifest, ConformanceLevel


class TestAwasDiscoveryService:
    """Tests for AwasDiscoveryService."""

    def test_normalize_base_url(self):
        """Test URL normalization."""
        discovery = AwasDiscoveryService()

        assert discovery._normalize_base_url("https://example.com") == "https://example.com"
        assert discovery._normalize_base_url("https://example.com/") == "https://example.com"
        assert discovery._normalize_base_url("https://example.com/path") == "https://example.com"
        assert discovery._normalize_base_url("http://localhost:3000/api") == "http://localhost:3000"

    def test_extract_elements_basic(self):
        """Test extracting AWAS elements from HTML."""
        discovery = AwasDiscoveryService()

        html = """
        <html>
        <body>
            <button data-awas-action="submit" data-awas-element="submit-btn">Submit</button>
            <input data-awas-element="name-input" data-awas-param-name="username">
            <a href="#" data-awas-action="navigate" id="nav-link">Navigate</a>
        </body>
        </html>
        """

        elements = discovery.extract_elements(html)

        assert len(elements) == 3

        # Check button element
        submit_btn = next((e for e in elements if e.id == "submit-btn"), None)
        assert submit_btn is not None
        assert submit_btn.action_id == "submit"

        # Check input element
        name_input = next((e for e in elements if e.id == "name-input"), None)
        assert name_input is not None

        # Check link element (uses id as fallback)
        nav_link = next((e for e in elements if e.id == "nav-link"), None)
        assert nav_link is not None
        assert nav_link.action_id == "navigate"

    def test_extract_elements_with_params(self):
        """Test extracting elements with parameter bindings."""
        discovery = AwasDiscoveryService()

        html = """
        <div data-awas-element="form" data-awas-param-id="123" data-awas-param-type="user">
            Content
        </div>
        """

        elements = discovery.extract_elements(html)

        assert len(elements) == 1
        assert elements[0].param_bindings == {"id": "123", "type": "user"}

    def test_extract_elements_empty_html(self):
        """Test extracting from HTML without AWAS attributes."""
        discovery = AwasDiscoveryService()

        html = """
        <html>
        <body>
            <button>Regular Button</button>
            <input type="text">
        </body>
        </html>
        """

        elements = discovery.extract_elements(html)
        assert len(elements) == 0

    def test_build_selector_with_id(self):
        """Test selector building prioritizes ID."""
        discovery = AwasDiscoveryService()

        attrs = {
            "id": "my-button",
            "data-awas-element": "button-elem",
            "data-awas-action": "click",
        }

        selector = discovery._build_selector(attrs, "<button>")
        assert selector == "#my-button"

    def test_build_selector_with_awas_element(self):
        """Test selector building uses data-awas-element."""
        discovery = AwasDiscoveryService()

        attrs = {
            "data-awas-element": "my-elem",
            "data-awas-action": "click",
        }

        selector = discovery._build_selector(attrs, "<button>")
        assert selector == '[data-awas-element="my-elem"]'

    def test_cache_operations(self):
        """Test cache management."""
        discovery = AwasDiscoveryService()

        # Manually add to cache for testing
        from qontinui.awas.discovery import CacheEntry

        manifest = AwasManifest(
            app_name="Test",
            base_url="https://test.com",
        )
        discovery._cache["https://test.com"] = CacheEntry(manifest, ttl_seconds=3600)

        # Test cache hit
        cached = discovery.get_cached_manifest("https://test.com")
        assert cached is not None
        assert cached.app_name == "Test"

        # Test cache miss
        assert discovery.get_cached_manifest("https://other.com") is None

        # Test list cached domains
        domains = discovery.list_cached_domains()
        assert "https://test.com" in domains

        # Test clear specific
        discovery.clear_cache("https://test.com")
        assert discovery.get_cached_manifest("https://test.com") is None

    def test_cache_entry_expiration(self):
        """Test cache entry TTL."""
        from qontinui.awas.discovery import CacheEntry

        manifest = AwasManifest(
            app_name="Test",
            base_url="https://test.com",
        )

        # Create entry with 0 TTL (immediately expired)
        entry = CacheEntry(manifest, ttl_seconds=0)
        assert entry.is_expired is True

        # Create entry with large TTL
        entry = CacheEntry(manifest, ttl_seconds=3600)
        assert entry.is_expired is False


class TestAwasDiscoveryServiceAsync:
    """Async tests for AwasDiscoveryService."""

    @pytest.mark.asyncio
    async def test_check_awas_support_no_manifest(self):
        """Test checking support when no manifest exists."""
        discovery = AwasDiscoveryService()

        # This will fail to fetch (no server) but should handle gracefully
        result = await discovery.check_awas_support("https://nonexistent.invalid")

        assert result["supported"] is False
        assert "url" in result

    @pytest.mark.asyncio
    async def test_discover_from_cache(self):
        """Test discovery returns cached manifest."""
        discovery = AwasDiscoveryService()

        # Pre-populate cache
        from qontinui.awas.discovery import CacheEntry

        manifest = AwasManifest(
            app_name="Cached App",
            base_url="https://cached.example.com",
            conformance_level=ConformanceLevel.L2,
        )
        discovery._cache["https://cached.example.com"] = CacheEntry(manifest)

        # Discovery should return cached value
        result = await discovery.discover("https://cached.example.com")

        assert result is not None
        assert result.app_name == "Cached App"
        assert result.conformance_level == ConformanceLevel.L2
