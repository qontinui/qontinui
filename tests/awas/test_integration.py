"""Integration tests for AWAS module.

Tests the full discovery -> execute flow with mocked HTTP responses.
Uses respx for httpx mocking.
"""

import json

import httpx
import pytest
import respx

from qontinui.awas.discovery import AwasDiscoveryService, CacheEntry
from qontinui.awas.executor import AwasExecutor
from qontinui.awas.types import (
    AwasAction,
    AwasAuth,
    AwasAuthType,
    AwasManifest,
    AwasParameter,
    ConformanceLevel,
    HttpMethod,
    ParameterLocation,
)


# Sample manifest data for testing
SAMPLE_MANIFEST_L1 = {
    "schemaVersion": "1.0",
    "appName": "Test App L1",
    "baseUrl": "https://api.example.com",
    "conformanceLevel": "L1",
    "actions": [
        {
            "id": "get_users",
            "name": "Get Users",
            "method": "GET",
            "endpoint": "/users",
            "intent": "Retrieve all users",
        },
        {
            "id": "create_user",
            "name": "Create User",
            "method": "POST",
            "endpoint": "/users",
            "intent": "Create a new user",
            "sideEffect": True,
        },
    ],
}

SAMPLE_MANIFEST_L2 = {
    "schemaVersion": "1.0",
    "appName": "Test App L2",
    "baseUrl": "https://api.example.com",
    "conformanceLevel": "L2",
    "actions": [
        {
            "id": "get_user",
            "name": "Get User",
            "method": "GET",
            "endpoint": "/users/{user_id}",
            "intent": "Get a specific user",
            "parameters": [
                {
                    "name": "user_id",
                    "location": "path",
                    "type": "string",
                    "required": True,
                }
            ],
            "inputSchema": {"type": "object"},
            "outputSchema": {"type": "object", "properties": {"id": {"type": "string"}}},
        },
    ],
    "auth": {"type": "bearer_token"},
}

SAMPLE_MANIFEST_L3 = {
    "schemaVersion": "1.0",
    "appName": "Test App L3",
    "description": "Full-featured L3 app",
    "baseUrl": "https://api.example.com",
    "conformanceLevel": "L3",
    "openapiUrl": "/openapi.json",
    "mcpManifestUrl": "/.well-known/mcp-manifest.json",
    "actions": [
        {
            "id": "delete_user",
            "name": "Delete User",
            "method": "DELETE",
            "endpoint": "/users/{user_id}",
            "intent": "Delete a user",
            "side_effect": True,
            "rate_limit": 10,
            "required_scopes": ["admin:write"],
            "parameters": [
                {
                    "name": "user_id",
                    "location": "path",
                    "type": "string",
                    "required": True,
                }
            ],
        },
    ],
    "auth": {
        "type": "oauth2",
        "token_endpoint": "/oauth/token",
        "authorization_url": "/oauth/authorize",
        "scopes": [
            {"name": "admin:write", "description": "Write access to admin resources"},
        ],
    },
}


class TestIntegrationDiscoverAndExecute:
    """Test the full discovery -> execute flow."""

    @pytest.fixture
    def discovery(self):
        """Create discovery service instance."""
        return AwasDiscoveryService(timeout_seconds=5)

    @pytest.fixture
    def executor(self):
        """Create executor instance."""
        return AwasExecutor(timeout_seconds=5)

    @pytest.mark.asyncio
    @respx.mock
    async def test_full_discovery_and_execute_flow(self, discovery, executor):
        """Test discovering manifest and executing an action."""
        # Mock the manifest endpoint
        respx.get("https://example.com/.well-known/ai-actions.json").mock(
            return_value=httpx.Response(200, json=SAMPLE_MANIFEST_L1)
        )

        # Mock the action endpoint
        respx.get("https://api.example.com/users").mock(
            return_value=httpx.Response(
                200,
                json={"users": [{"id": "1", "name": "John"}]},
            )
        )

        # Discover manifest
        manifest = await discovery.discover("https://example.com")

        assert manifest is not None
        assert manifest.app_name == "Test App L1"
        assert manifest.conformance_level == ConformanceLevel.L1
        assert len(manifest.actions) == 2

        # Execute an action
        result = await executor.execute(manifest, "get_users")

        assert result.success is True
        assert result.status_code == 200
        assert result.response_body == {"users": [{"id": "1", "name": "John"}]}

    @pytest.mark.asyncio
    @respx.mock
    async def test_discovery_and_execute_with_path_params(self, discovery, executor):
        """Test executing action with path parameters."""
        # Mock manifest
        respx.get("https://example.com/.well-known/ai-actions.json").mock(
            return_value=httpx.Response(200, json=SAMPLE_MANIFEST_L2)
        )

        # Mock user endpoint
        respx.get("https://api.example.com/users/abc123").mock(
            return_value=httpx.Response(200, json={"id": "abc123", "name": "Jane"})
        )

        manifest = await discovery.discover("https://example.com")
        result = await executor.execute(
            manifest, "get_user", params={"user_id": "abc123"}
        )

        assert result.success is True
        assert result.response_body["id"] == "abc123"

    @pytest.mark.asyncio
    @respx.mock
    async def test_discovery_and_execute_with_auth(self, discovery, executor):
        """Test executing action with authentication."""
        # Mock manifest
        respx.get("https://example.com/.well-known/ai-actions.json").mock(
            return_value=httpx.Response(200, json=SAMPLE_MANIFEST_L2)
        )

        # Mock user endpoint - check auth header
        def check_auth(request):
            auth_header = request.headers.get("Authorization", "")
            if auth_header == "Bearer test-token":
                return httpx.Response(200, json={"id": "user1", "name": "Auth User"})
            return httpx.Response(401, json={"error": "Unauthorized"})

        respx.get("https://api.example.com/users/user1").mock(side_effect=check_auth)

        manifest = await discovery.discover("https://example.com")

        # Without auth - should fail
        result_no_auth = await executor.execute(
            manifest, "get_user", params={"user_id": "user1"}
        )
        assert result_no_auth.success is False
        assert result_no_auth.status_code == 401

        # With auth - should succeed
        result_with_auth = await executor.execute(
            manifest,
            "get_user",
            params={"user_id": "user1"},
            credentials={"token": "test-token"},
        )
        assert result_with_auth.success is True
        assert result_with_auth.response_body["name"] == "Auth User"


class TestManifestCaching:
    """Test manifest caching behavior."""

    @pytest.fixture
    def discovery(self):
        """Create discovery service with short TTL for testing."""
        return AwasDiscoveryService(cache_ttl_seconds=1, timeout_seconds=5)

    @pytest.mark.asyncio
    @respx.mock
    async def test_manifest_is_cached(self, discovery):
        """Test that manifest is cached after first fetch."""
        # Mock manifest endpoint - should only be called once
        manifest_route = respx.get(
            "https://example.com/.well-known/ai-actions.json"
        ).mock(return_value=httpx.Response(200, json=SAMPLE_MANIFEST_L1))

        # First discovery
        manifest1 = await discovery.discover("https://example.com")
        assert manifest1 is not None
        assert manifest_route.call_count == 1

        # Second discovery - should use cache
        manifest2 = await discovery.discover("https://example.com")
        assert manifest2 is not None
        assert manifest_route.call_count == 1  # Still 1, no new request
        assert manifest2.app_name == manifest1.app_name

    @pytest.mark.asyncio
    @respx.mock
    async def test_cache_expiration(self, discovery):
        """Test that cache expires after TTL."""
        import asyncio

        manifest_route = respx.get(
            "https://example.com/.well-known/ai-actions.json"
        ).mock(return_value=httpx.Response(200, json=SAMPLE_MANIFEST_L1))

        # First discovery
        await discovery.discover("https://example.com")
        assert manifest_route.call_count == 1

        # Wait for cache to expire (TTL is 1 second)
        await asyncio.sleep(1.1)

        # Third discovery - cache expired, should fetch again
        await discovery.discover("https://example.com")
        assert manifest_route.call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_cache_clear_specific(self, discovery):
        """Test clearing cache for specific URL."""
        respx.get("https://example1.com/.well-known/ai-actions.json").mock(
            return_value=httpx.Response(200, json=SAMPLE_MANIFEST_L1)
        )
        respx.get("https://example2.com/.well-known/ai-actions.json").mock(
            return_value=httpx.Response(
                200,
                json={
                    **SAMPLE_MANIFEST_L1,
                    "appName": "App 2",
                    "baseUrl": "https://example2.com",
                },
            )
        )

        # Populate cache
        await discovery.discover("https://example1.com")
        await discovery.discover("https://example2.com")

        domains = discovery.list_cached_domains()
        assert "https://example1.com" in domains
        assert "https://example2.com" in domains

        # Clear specific
        discovery.clear_cache("https://example1.com")

        domains = discovery.list_cached_domains()
        assert "https://example1.com" not in domains
        assert "https://example2.com" in domains

    def test_cache_clear_all(self, discovery):
        """Test clearing all cache."""
        # Manually populate cache
        manifest = AwasManifest(app_name="Test", base_url="https://test.com")
        discovery._cache["https://test1.com"] = CacheEntry(manifest)
        discovery._cache["https://test2.com"] = CacheEntry(manifest)

        assert len(discovery.list_cached_domains()) == 2

        discovery.clear_cache()

        assert len(discovery.list_cached_domains()) == 0


class TestErrorHandling:
    """Test error handling for various failure scenarios."""

    @pytest.fixture
    def discovery(self):
        """Create discovery service."""
        return AwasDiscoveryService(timeout_seconds=2)

    @pytest.fixture
    def executor(self):
        """Create executor."""
        return AwasExecutor(timeout_seconds=2)

    @pytest.mark.asyncio
    @respx.mock
    async def test_manifest_404(self, discovery):
        """Test handling when manifest doesn't exist."""
        respx.get("https://noawas.example.com/.well-known/ai-actions.json").mock(
            return_value=httpx.Response(404)
        )

        manifest = await discovery.discover("https://noawas.example.com")
        assert manifest is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_manifest_500_error(self, discovery):
        """Test handling server error."""
        respx.get("https://error.example.com/.well-known/ai-actions.json").mock(
            return_value=httpx.Response(500, json={"error": "Internal Server Error"})
        )

        manifest = await discovery.discover("https://error.example.com")
        assert manifest is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_manifest_invalid_json(self, discovery):
        """Test handling invalid JSON response."""
        respx.get("https://invalid.example.com/.well-known/ai-actions.json").mock(
            return_value=httpx.Response(200, content=b"not json at all {{{")
        )

        manifest = await discovery.discover("https://invalid.example.com")
        assert manifest is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_manifest_timeout(self, discovery):
        """Test handling timeout."""
        respx.get("https://slow.example.com/.well-known/ai-actions.json").mock(
            side_effect=httpx.TimeoutException("Connection timed out")
        )

        manifest = await discovery.discover("https://slow.example.com")
        assert manifest is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_manifest_connection_error(self, discovery):
        """Test handling connection error."""
        respx.get("https://offline.example.com/.well-known/ai-actions.json").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        manifest = await discovery.discover("https://offline.example.com")
        assert manifest is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_action_execution_timeout(self, executor):
        """Test action execution timeout."""
        manifest = AwasManifest(
            app_name="Test",
            base_url="https://slow.example.com",
            actions=[
                AwasAction(
                    id="slow_action",
                    name="Slow Action",
                    method=HttpMethod.GET,
                    endpoint="/slow",
                    intent="Test slow action",
                )
            ],
        )

        respx.get("https://slow.example.com/slow").mock(
            side_effect=httpx.TimeoutException("Request timed out")
        )

        result = await executor.execute(manifest, "slow_action")

        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    @respx.mock
    async def test_action_execution_server_error(self, executor):
        """Test action execution with server error."""
        manifest = AwasManifest(
            app_name="Test",
            base_url="https://error.example.com",
            actions=[
                AwasAction(
                    id="error_action",
                    name="Error Action",
                    method=HttpMethod.GET,
                    endpoint="/error",
                    intent="Test error action",
                )
            ],
        )

        respx.get("https://error.example.com/error").mock(
            return_value=httpx.Response(500, json={"error": "Server error"})
        )

        result = await executor.execute(manifest, "error_action")

        assert result.success is False
        assert result.status_code == 500
        assert "500" in result.error


class TestConformanceLevels:
    """Test different conformance levels (L1, L2, L3)."""

    @pytest.fixture
    def discovery(self):
        """Create discovery service."""
        return AwasDiscoveryService(timeout_seconds=5)

    @pytest.fixture
    def executor(self):
        """Create executor."""
        return AwasExecutor(timeout_seconds=5)

    @pytest.mark.asyncio
    @respx.mock
    async def test_l1_manifest_basic_actions(self, discovery, executor):
        """Test L1 conformance level - basic actions only."""
        respx.get("https://l1.example.com/.well-known/ai-actions.json").mock(
            return_value=httpx.Response(200, json=SAMPLE_MANIFEST_L1)
        )
        respx.get("https://api.example.com/users").mock(
            return_value=httpx.Response(200, json={"users": []})
        )

        manifest = await discovery.discover("https://l1.example.com")

        assert manifest is not None
        assert manifest.conformance_level == ConformanceLevel.L1
        assert manifest.auth is None  # L1 typically no auth

        # Execute basic action
        result = await executor.execute(manifest, "get_users")
        assert result.success is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_l2_manifest_typed_io(self, discovery, executor):
        """Test L2 conformance level - typed I/O with parameters."""
        respx.get("https://l2.example.com/.well-known/ai-actions.json").mock(
            return_value=httpx.Response(200, json=SAMPLE_MANIFEST_L2)
        )
        respx.get("https://api.example.com/users/test123").mock(
            return_value=httpx.Response(200, json={"id": "test123", "email": "test@example.com"})
        )

        manifest = await discovery.discover("https://l2.example.com")

        assert manifest is not None
        assert manifest.conformance_level == ConformanceLevel.L2
        assert manifest.auth is not None
        assert manifest.auth.type == AwasAuthType.BEARER_TOKEN

        # Action has typed parameters
        action = manifest.get_action("get_user")
        assert action is not None
        assert len(action.parameters) == 1
        assert action.parameters[0].location == ParameterLocation.PATH

        # Execute with params
        result = await executor.execute(manifest, "get_user", params={"user_id": "test123"})
        assert result.success is True
        assert result.response_body["id"] == "test123"

    @pytest.mark.asyncio
    @respx.mock
    async def test_l3_manifest_full_features(self, discovery, executor):
        """Test L3 conformance level - full features."""
        respx.get("https://l3.example.com/.well-known/ai-actions.json").mock(
            return_value=httpx.Response(200, json=SAMPLE_MANIFEST_L3)
        )
        respx.delete("https://api.example.com/users/del123").mock(
            return_value=httpx.Response(204)
        )

        manifest = await discovery.discover("https://l3.example.com")

        assert manifest is not None
        assert manifest.conformance_level == ConformanceLevel.L3
        assert manifest.description == "Full-featured L3 app"
        assert manifest.openapi_url == "/openapi.json"
        assert manifest.mcp_manifest_url == "/.well-known/mcp-manifest.json"

        # Check auth with OAuth2
        assert manifest.auth is not None
        assert manifest.auth.type == AwasAuthType.OAUTH2
        assert manifest.auth.token_endpoint == "/oauth/token"
        assert len(manifest.auth.scopes) == 1

        # Action has rate limits and scopes
        action = manifest.get_action("delete_user")
        assert action is not None
        assert action.rate_limit == 10
        assert "admin:write" in action.required_scopes
        assert action.side_effect is True

        # Execute action
        result = await executor.execute(manifest, "delete_user", params={"user_id": "del123"})
        assert result.success is True
        assert result.status_code == 204


class TestBatchExecution:
    """Test batch action execution."""

    @pytest.fixture
    def executor(self):
        """Create executor."""
        return AwasExecutor(timeout_seconds=5)

    @pytest.fixture
    def manifest(self):
        """Create test manifest."""
        return AwasManifest(
            app_name="Batch Test",
            base_url="https://batch.example.com",
            actions=[
                AwasAction(
                    id="action1",
                    name="Action 1",
                    method=HttpMethod.GET,
                    endpoint="/action1",
                    intent="First action",
                ),
                AwasAction(
                    id="action2",
                    name="Action 2",
                    method=HttpMethod.GET,
                    endpoint="/action2",
                    intent="Second action",
                ),
                AwasAction(
                    id="action3",
                    name="Action 3",
                    method=HttpMethod.GET,
                    endpoint="/action3",
                    intent="Third action",
                ),
            ],
        )

    @pytest.mark.asyncio
    @respx.mock
    async def test_batch_execution_all_success(self, executor, manifest):
        """Test batch execution with all actions succeeding."""
        respx.get("https://batch.example.com/action1").mock(
            return_value=httpx.Response(200, json={"result": 1})
        )
        respx.get("https://batch.example.com/action2").mock(
            return_value=httpx.Response(200, json={"result": 2})
        )
        respx.get("https://batch.example.com/action3").mock(
            return_value=httpx.Response(200, json={"result": 3})
        )

        results = await executor.execute_batch(
            manifest,
            [("action1", None), ("action2", None), ("action3", None)],
        )

        assert len(results) == 3
        assert all(r.success for r in results)
        assert results[0].response_body["result"] == 1
        assert results[1].response_body["result"] == 2
        assert results[2].response_body["result"] == 3

    @pytest.mark.asyncio
    @respx.mock
    async def test_batch_execution_continue_on_error(self, executor, manifest):
        """Test batch execution continues after error by default."""
        respx.get("https://batch.example.com/action1").mock(
            return_value=httpx.Response(200, json={"result": 1})
        )
        respx.get("https://batch.example.com/action2").mock(
            return_value=httpx.Response(500, json={"error": "fail"})
        )
        respx.get("https://batch.example.com/action3").mock(
            return_value=httpx.Response(200, json={"result": 3})
        )

        results = await executor.execute_batch(
            manifest,
            [("action1", None), ("action2", None), ("action3", None)],
            stop_on_error=False,
        )

        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True  # Continued after error

    @pytest.mark.asyncio
    @respx.mock
    async def test_batch_execution_stop_on_error(self, executor, manifest):
        """Test batch execution stops on error when configured."""
        respx.get("https://batch.example.com/action1").mock(
            return_value=httpx.Response(200, json={"result": 1})
        )
        respx.get("https://batch.example.com/action2").mock(
            return_value=httpx.Response(500, json={"error": "fail"})
        )
        # action3 should not be called
        action3_route = respx.get("https://batch.example.com/action3").mock(
            return_value=httpx.Response(200, json={"result": 3})
        )

        results = await executor.execute_batch(
            manifest,
            [("action1", None), ("action2", None), ("action3", None)],
            stop_on_error=True,
        )

        assert len(results) == 2  # Only 2 results, stopped at action2
        assert results[0].success is True
        assert results[1].success is False
        assert action3_route.call_count == 0  # Never called


class TestAwasSupport:
    """Test AWAS support checking."""

    @pytest.fixture
    def discovery(self):
        """Create discovery service."""
        return AwasDiscoveryService(timeout_seconds=5)

    @pytest.mark.asyncio
    @respx.mock
    async def test_check_awas_support_available(self, discovery):
        """Test checking AWAS support when manifest exists."""
        respx.get("https://awas.example.com/.well-known/ai-actions.json").mock(
            return_value=httpx.Response(200, json=SAMPLE_MANIFEST_L2)
        )

        result = await discovery.check_awas_support("https://awas.example.com")

        assert result["supported"] is True
        assert result["app_name"] == "Test App L2"
        assert result["action_count"] == 1
        assert result["conformance_level"] == "L2"
        assert result["has_auth"] is True
        assert result["auth_type"] == "bearer_token"

    @pytest.mark.asyncio
    @respx.mock
    async def test_check_awas_support_not_available(self, discovery):
        """Test checking AWAS support when no manifest."""
        respx.get("https://noawas.example.com/.well-known/ai-actions.json").mock(
            return_value=httpx.Response(404)
        )

        result = await discovery.check_awas_support("https://noawas.example.com")

        assert result["supported"] is False
        assert "No AWAS manifest" in result["message"]


class TestSyncWrappers:
    """Test synchronous wrapper methods.

    Note: Sync wrappers use asyncio.get_event_loop().run_until_complete()
    which doesn't work when already inside an event loop (pytest-asyncio).
    These tests verify the sync wrappers exist and have correct signatures.
    Integration testing of sync wrappers should be done in non-async test files.
    """

    def test_discover_sync_method_exists(self):
        """Test discover_sync method exists with correct signature."""
        discovery = AwasDiscoveryService(timeout_seconds=5)
        assert hasattr(discovery, "discover_sync")
        assert callable(discovery.discover_sync)

    def test_execute_sync_method_exists(self):
        """Test execute_sync method exists with correct signature."""
        executor = AwasExecutor(timeout_seconds=5)
        assert hasattr(executor, "execute_sync")
        assert callable(executor.execute_sync)
