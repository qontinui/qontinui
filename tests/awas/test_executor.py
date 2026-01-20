"""Tests for AWAS action executor."""

import pytest

from qontinui.awas.executor import AwasExecutor
from qontinui.awas.types import (
    AwasAction,
    AwasAuth,
    AwasAuthType,
    AwasManifest,
    AwasParameter,
    HttpMethod,
    ParameterLocation,
)


class TestAwasExecutor:
    """Tests for AwasExecutor."""

    @pytest.fixture
    def executor(self):
        """Create executor instance."""
        return AwasExecutor()

    @pytest.fixture
    def sample_manifest(self):
        """Create sample manifest for testing."""
        return AwasManifest(
            app_name="Test API",
            base_url="https://api.example.com",
            actions=[
                AwasAction(
                    id="get_users",
                    name="Get Users",
                    method=HttpMethod.GET,
                    endpoint="/users",
                    intent="Retrieve all users",
                    parameters=[
                        AwasParameter(
                            name="limit",
                            location=ParameterLocation.QUERY,
                            type="integer",
                            required=False,
                            default=10,
                        ),
                        AwasParameter(
                            name="active",
                            location=ParameterLocation.QUERY,
                            type="boolean",
                            required=False,
                        ),
                    ],
                ),
                AwasAction(
                    id="get_user",
                    name="Get User",
                    method=HttpMethod.GET,
                    endpoint="/users/{user_id}",
                    intent="Get specific user",
                    parameters=[
                        AwasParameter(
                            name="user_id",
                            location=ParameterLocation.PATH,
                            type="string",
                            required=True,
                        ),
                    ],
                ),
                AwasAction(
                    id="create_user",
                    name="Create User",
                    method=HttpMethod.POST,
                    endpoint="/users",
                    intent="Create a new user",
                    side_effect=True,
                    parameters=[
                        AwasParameter(
                            name="name",
                            location=ParameterLocation.BODY,
                            type="string",
                            required=True,
                        ),
                        AwasParameter(
                            name="email",
                            location=ParameterLocation.BODY,
                            type="string",
                            required=True,
                        ),
                        AwasParameter(
                            name="role",
                            location=ParameterLocation.BODY,
                            type="string",
                            required=False,
                            enum=["admin", "user", "guest"],
                        ),
                    ],
                ),
            ],
            auth=AwasAuth(type=AwasAuthType.BEARER_TOKEN),
        )

    def test_build_url_simple(self, executor, sample_manifest):
        """Test building URL for simple endpoint."""
        action = sample_manifest.get_action("get_users")
        url = executor._build_url(sample_manifest, action, {})

        assert url == "https://api.example.com/users"

    def test_build_url_with_query_params(self, executor, sample_manifest):
        """Test building URL with query parameters."""
        action = sample_manifest.get_action("get_users")
        url = executor._build_url(sample_manifest, action, {"limit": 20, "active": True})

        assert "limit=20" in url
        assert "active=true" in url

    def test_build_url_with_path_params(self, executor, sample_manifest):
        """Test building URL with path parameters."""
        action = sample_manifest.get_action("get_user")
        url = executor._build_url(sample_manifest, action, {"user_id": "abc123"})

        assert url == "https://api.example.com/users/abc123"

    def test_build_body(self, executor, sample_manifest):
        """Test building request body."""
        action = sample_manifest.get_action("create_user")
        body = executor._build_body(
            action, {"name": "John", "email": "john@example.com", "role": "admin"}
        )

        assert body == {"name": "John", "email": "john@example.com", "role": "admin"}

    def test_build_body_get_request(self, executor, sample_manifest):
        """Test that GET requests have no body."""
        action = sample_manifest.get_action("get_users")
        body = executor._build_body(action, {"limit": 10})

        assert body is None

    def test_build_headers(self, executor, sample_manifest):
        """Test building request headers."""
        action = sample_manifest.get_action("get_users")
        headers = executor._build_headers(action, {}, sample_manifest, {})

        assert "Content-Type" in headers
        assert "Accept" in headers
        assert "User-Agent" in headers

    def test_build_auth_headers_bearer(self, executor, sample_manifest):
        """Test building bearer token auth headers."""
        headers = executor._build_auth_headers(sample_manifest, {"token": "my-secret-token"})

        assert headers.get("Authorization") == "Bearer my-secret-token"

    def test_build_auth_headers_api_key(self, executor):
        """Test building API key auth headers."""
        manifest = AwasManifest(
            app_name="Test",
            base_url="https://test.com",
            auth=AwasAuth(
                type=AwasAuthType.API_KEY,
                header_name="X-Custom-Key",
            ),
        )

        headers = executor._build_auth_headers(manifest, {"api_key": "secret123"})

        assert headers.get("X-Custom-Key") == "secret123"

    def test_build_auth_headers_basic(self, executor):
        """Test building basic auth headers."""
        manifest = AwasManifest(
            app_name="Test",
            base_url="https://test.com",
            auth=AwasAuth(type=AwasAuthType.BASIC),
        )

        headers = executor._build_auth_headers(manifest, {"username": "user", "password": "pass"})

        # Basic auth should be base64 encoded
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Basic ")

    def test_validate_params_missing_required(self, executor, sample_manifest):
        """Test validation catches missing required parameters."""
        action = sample_manifest.get_action("create_user")
        errors = executor.validate_params(action, {"name": "John"})  # Missing email

        assert len(errors) > 0
        assert any("email" in e for e in errors)

    def test_validate_params_invalid_enum(self, executor, sample_manifest):
        """Test validation catches invalid enum values."""
        action = sample_manifest.get_action("create_user")
        errors = executor.validate_params(
            action,
            {"name": "John", "email": "john@example.com", "role": "superadmin"},
        )

        assert len(errors) > 0
        assert any("role" in e and "superadmin" in e for e in errors)

    def test_validate_params_valid(self, executor, sample_manifest):
        """Test validation passes for valid parameters."""
        action = sample_manifest.get_action("create_user")
        errors = executor.validate_params(
            action,
            {"name": "John", "email": "john@example.com", "role": "admin"},
        )

        assert len(errors) == 0


class TestAwasExecutorAsync:
    """Async tests for AwasExecutor."""

    @pytest.fixture
    def executor(self):
        """Create executor instance."""
        return AwasExecutor(timeout_seconds=5)

    @pytest.fixture
    def sample_manifest(self):
        """Create sample manifest."""
        return AwasManifest(
            app_name="Test",
            base_url="https://httpbin.org",
            actions=[
                AwasAction(
                    id="test_get",
                    name="Test GET",
                    method=HttpMethod.GET,
                    endpoint="/get",
                    intent="Test GET request",
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_execute_action_not_found(self, executor, sample_manifest):
        """Test executing non-existent action."""
        result = await executor.execute(sample_manifest, "nonexistent_action")

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_timeout(self, executor):
        """Test request timeout handling."""
        # Create manifest with slow endpoint
        manifest = AwasManifest(
            app_name="Test",
            base_url="https://httpbin.org",
            actions=[
                AwasAction(
                    id="slow",
                    name="Slow",
                    method=HttpMethod.GET,
                    endpoint="/delay/10",  # 10 second delay
                    intent="Test timeout",
                ),
            ],
        )

        # Use very short timeout
        result = await executor.execute(manifest, "slow", timeout_seconds=0.1)

        assert result.success is False
        assert "timed out" in result.error.lower()
