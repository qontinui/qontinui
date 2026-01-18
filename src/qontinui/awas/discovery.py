"""
AWAS manifest discovery service.

Handles fetching, parsing, and caching of AWAS manifests from web applications.
Also extracts AWAS elements from HTML content.
"""

import logging
import re
import time
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from .types import AwasElement, AwasManifest

logger = logging.getLogger(__name__)


class CacheEntry:
    """Cache entry with TTL tracking."""

    def __init__(self, manifest: AwasManifest, ttl_seconds: float = 3600.0):
        self.manifest = manifest
        self.fetched_at = time.time()
        self.ttl_seconds = ttl_seconds

    @property
    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        return time.time() - self.fetched_at >= self.ttl_seconds


class AwasDiscoveryService:
    """
    Service for discovering and caching AWAS manifests.

    Handles:
    - Fetching /.well-known/ai-actions.json manifests
    - Caching manifests with configurable TTL
    - Extracting AWAS elements from HTML
    """

    MANIFEST_PATH = "/.well-known/ai-actions.json"
    DEFAULT_TTL_SECONDS = 3600.0  # 1 hour
    DEFAULT_TIMEOUT_SECONDS = 10.0

    def __init__(
        self,
        cache_ttl_seconds: float = DEFAULT_TTL_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        """
        Initialize the discovery service.

        Args:
            cache_ttl_seconds: How long to cache manifests (default 1 hour)
            timeout_seconds: HTTP request timeout (default 10 seconds)
        """
        self._cache: dict[str, CacheEntry] = {}
        self._cache_ttl = cache_ttl_seconds
        self._timeout = timeout_seconds

    def _normalize_base_url(self, url: str) -> str:
        """Normalize a URL to its base form (scheme + host)."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    async def discover(self, base_url: str) -> AwasManifest | None:
        """
        Discover AWAS manifest for a website.

        Fetches the manifest from /.well-known/ai-actions.json and caches it.

        Args:
            base_url: Base URL of the website (e.g., https://example.com)

        Returns:
            AwasManifest if found, None if not available or invalid
        """
        normalized_url = self._normalize_base_url(base_url)

        # Check cache first
        cached = self._get_cached(normalized_url)
        if cached is not None:
            logger.debug(f"AWAS manifest cache hit for {normalized_url}")
            return cached

        # Fetch manifest
        manifest_url = urljoin(normalized_url, self.MANIFEST_PATH)
        logger.info(f"Discovering AWAS manifest at {manifest_url}")

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    manifest_url,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "Qontinui/1.0 (AWAS-compatible)",
                    },
                )

                if response.status_code == 404:
                    logger.info(f"No AWAS manifest at {normalized_url} (404)")
                    return None

                if not response.is_success:
                    logger.warning(
                        f"AWAS manifest fetch failed: {response.status_code} from {manifest_url}"
                    )
                    return None

                # Parse manifest
                data = response.json()
                manifest = AwasManifest.model_validate(data)

                # Cache the manifest
                self._cache[normalized_url] = CacheEntry(manifest, self._cache_ttl)

                logger.info(
                    f"Discovered AWAS manifest for {manifest.app_name} "
                    f"with {len(manifest.actions)} actions (level {manifest.conformance_level.value})"
                )
                return manifest

        except httpx.TimeoutException:
            logger.warning(f"Timeout fetching AWAS manifest from {manifest_url}")
            return None
        except httpx.RequestError as e:
            logger.warning(f"Error fetching AWAS manifest from {manifest_url}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to parse AWAS manifest from {manifest_url}: {e}")
            return None

    def discover_sync(self, base_url: str) -> AwasManifest | None:
        """
        Synchronous version of discover().

        Args:
            base_url: Base URL of the website

        Returns:
            AwasManifest if found, None otherwise
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.discover(base_url))

    def _get_cached(self, base_url: str) -> AwasManifest | None:
        """Get cached manifest if available and not expired."""
        entry = self._cache.get(base_url)
        if entry is None:
            return None
        if entry.is_expired:
            del self._cache[base_url]
            return None
        return entry.manifest

    def get_cached_manifest(self, base_url: str) -> AwasManifest | None:
        """
        Get a cached manifest without fetching.

        Args:
            base_url: Base URL to look up

        Returns:
            Cached manifest or None
        """
        normalized_url = self._normalize_base_url(base_url)
        return self._get_cached(normalized_url)

    def clear_cache(self, base_url: str | None = None) -> None:
        """
        Clear cached manifests.

        Args:
            base_url: Specific URL to clear, or None to clear all
        """
        if base_url is None:
            self._cache.clear()
            logger.info("Cleared all AWAS manifest cache")
        else:
            normalized_url = self._normalize_base_url(base_url)
            if normalized_url in self._cache:
                del self._cache[normalized_url]
                logger.info(f"Cleared AWAS manifest cache for {normalized_url}")

    def list_cached_domains(self) -> list[str]:
        """List all domains with cached manifests."""
        return list(self._cache.keys())

    def extract_elements(self, html: str, page_url: str | None = None) -> list[AwasElement]:
        """
        Extract AWAS elements from HTML content.

        Looks for elements with data-awas-* attributes:
        - data-awas-element: Element identifier
        - data-awas-action: Associated action ID
        - data-awas-trigger: Trigger type (click, submit, etc.)
        - data-awas-param-*: Parameter bindings

        Args:
            html: HTML content to parse
            page_url: URL of the page (for context)

        Returns:
            List of extracted AWAS elements
        """
        elements: list[AwasElement] = []

        # Find all elements with data-awas-* attributes
        # This regex finds opening tags with data-awas attributes
        awas_pattern = r"<(\w+)[^>]*\bdata-awas-(?:element|action)[^>]*>"

        for match in re.finditer(awas_pattern, html, re.IGNORECASE):
            tag_html = match.group(0)
            element = self._parse_awas_element(tag_html)
            if element:
                elements.append(element)

        if elements:
            logger.debug(f"Extracted {len(elements)} AWAS elements from HTML")

        return elements

    def _parse_awas_element(self, tag_html: str) -> AwasElement | None:
        """Parse AWAS attributes from an HTML tag."""
        # Extract attributes
        attrs = self._extract_attributes(tag_html)

        # Get element ID (required)
        element_id = attrs.get("data-awas-element") or attrs.get("id")
        if not element_id:
            return None

        # Get action ID
        action_id = attrs.get("data-awas-action")

        # Get trigger type
        trigger = attrs.get("data-awas-trigger")

        # Build CSS selector
        selector = self._build_selector(attrs, tag_html)

        # Extract parameter bindings
        param_bindings: dict[str, str] = {}
        for key, value in attrs.items():
            if key.startswith("data-awas-param-"):
                param_name = key[len("data-awas-param-") :]
                param_bindings[param_name] = value

        return AwasElement(
            id=element_id,
            action_id=action_id,
            selector=selector,
            param_bindings=param_bindings,
            trigger=trigger,
        )

    def _extract_attributes(self, tag_html: str) -> dict[str, str]:
        """Extract attributes from an HTML tag string."""
        attrs: dict[str, str] = {}

        # Match attribute patterns: name="value" or name='value'
        attr_pattern = r'(\w[\w-]*)\s*=\s*["\']([^"\']*)["\']'
        for match in re.finditer(attr_pattern, tag_html):
            name = match.group(1).lower()
            value = match.group(2)
            attrs[name] = value

        return attrs

    def _build_selector(self, attrs: dict[str, str], tag_html: str) -> str:
        """Build a CSS selector for an element."""
        # Prefer ID
        if "id" in attrs:
            return f"#{attrs['id']}"

        # Use data-awas-element attribute
        if "data-awas-element" in attrs:
            return f'[data-awas-element="{attrs["data-awas-element"]}"]'

        # Use data-awas-action attribute
        if "data-awas-action" in attrs:
            return f'[data-awas-action="{attrs["data-awas-action"]}"]'

        # Fallback to tag name
        tag_match = re.match(r"<(\w+)", tag_html)
        if tag_match:
            return tag_match.group(1)

        return "*"

    async def check_awas_support(self, base_url: str) -> dict[str, Any]:
        """
        Check if a website supports AWAS and return summary info.

        Args:
            base_url: Base URL to check

        Returns:
            Dict with support info: {supported, app_name, action_count, conformance_level, etc.}
        """
        manifest = await self.discover(base_url)

        if manifest is None:
            return {
                "supported": False,
                "url": base_url,
                "message": "No AWAS manifest found",
            }

        return {
            "supported": True,
            "url": base_url,
            "app_name": manifest.app_name,
            "description": manifest.description,
            "action_count": len(manifest.actions),
            "conformance_level": manifest.conformance_level.value,
            "has_auth": manifest.auth is not None,
            "auth_type": manifest.auth.type.value if manifest.auth else None,
            "read_only_actions": len(manifest.get_read_only_actions()),
            "openapi_url": manifest.openapi_url,
            "mcp_manifest_url": manifest.mcp_manifest_url,
        }
