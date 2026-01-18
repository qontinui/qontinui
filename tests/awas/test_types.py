"""Tests for AWAS type definitions."""

import pytest

from qontinui.awas.types import (
    AwasAction,
    AwasAuth,
    AwasAuthType,
    AwasElement,
    AwasManifest,
    AwasParameter,
    ConformanceLevel,
    HttpMethod,
    ParameterLocation,
)


class TestAwasManifest:
    """Tests for AwasManifest model."""

    def test_minimal_manifest(self):
        """Test creating manifest with minimal required fields."""
        manifest = AwasManifest(
            app_name="Test App",
            base_url="https://example.com",
        )
        assert manifest.app_name == "Test App"
        assert manifest.base_url == "https://example.com"
        assert manifest.schema_version == "1.0"
        assert manifest.conformance_level == ConformanceLevel.L1
        assert manifest.actions == []

    def test_full_manifest(self):
        """Test creating manifest with all fields."""
        manifest = AwasManifest(
            schemaVersion="1.1",
            appName="Full App",
            description="A fully configured app",
            baseUrl="https://api.example.com",
            conformanceLevel=ConformanceLevel.L3,
            openapiUrl="/openapi.json",
            mcpManifestUrl="/.well-known/mcp-manifest.json",
            actions=[
                AwasAction(
                    id="test_action",
                    name="Test Action",
                    method=HttpMethod.GET,
                    endpoint="/test",
                    intent="Test something",
                )
            ],
            auth=AwasAuth(type=AwasAuthType.BEARER_TOKEN),
        )
        assert manifest.schema_version == "1.1"
        assert manifest.conformance_level == ConformanceLevel.L3
        assert len(manifest.actions) == 1

    def test_manifest_from_json(self):
        """Test parsing manifest from JSON-like dict."""
        data = {
            "schemaVersion": "1.0",
            "appName": "JSON App",
            "baseUrl": "https://json.example.com",
            "conformanceLevel": "L2",
            "actions": [
                {
                    "id": "get_users",
                    "name": "Get Users",
                    "method": "GET",
                    "endpoint": "/users",
                    "intent": "Retrieve users",
                }
            ],
        }
        manifest = AwasManifest.model_validate(data)
        assert manifest.app_name == "JSON App"
        assert manifest.conformance_level == ConformanceLevel.L2
        assert len(manifest.actions) == 1

    def test_get_action(self):
        """Test finding action by ID."""
        manifest = AwasManifest(
            app_name="Test",
            base_url="https://test.com",
            actions=[
                AwasAction(
                    id="action1",
                    name="Action 1",
                    method=HttpMethod.GET,
                    endpoint="/a1",
                    intent="First action",
                ),
                AwasAction(
                    id="action2",
                    name="Action 2",
                    method=HttpMethod.POST,
                    endpoint="/a2",
                    intent="Second action",
                ),
            ],
        )

        action = manifest.get_action("action1")
        assert action is not None
        assert action.name == "Action 1"

        action = manifest.get_action("nonexistent")
        assert action is None

    def test_get_read_only_actions(self):
        """Test filtering read-only actions."""
        manifest = AwasManifest(
            app_name="Test",
            base_url="https://test.com",
            actions=[
                AwasAction(
                    id="read1",
                    name="Read 1",
                    method=HttpMethod.GET,
                    endpoint="/r1",
                    intent="Read data",
                    side_effect=False,
                ),
                AwasAction(
                    id="write1",
                    name="Write 1",
                    method=HttpMethod.POST,
                    endpoint="/w1",
                    intent="Write data",
                    side_effect=True,
                ),
                AwasAction(
                    id="read2",
                    name="Read 2",
                    method=HttpMethod.GET,
                    endpoint="/r2",
                    intent="Read more data",
                    side_effect=False,
                ),
            ],
        )

        read_only = manifest.get_read_only_actions()
        assert len(read_only) == 2
        assert all(a.method == HttpMethod.GET and not a.side_effect for a in read_only)


class TestAwasAction:
    """Tests for AwasAction model."""

    def test_action_with_parameters(self):
        """Test action with various parameter types."""
        action = AwasAction(
            id="create_user",
            name="Create User",
            method=HttpMethod.POST,
            endpoint="/users/{org_id}",
            intent="Create a new user in organization",
            side_effect=True,
            parameters=[
                AwasParameter(
                    name="org_id",
                    location=ParameterLocation.PATH,
                    type="string",
                    required=True,
                ),
                AwasParameter(
                    name="name",
                    location=ParameterLocation.BODY,
                    type="string",
                    required=True,
                ),
                AwasParameter(
                    name="active",
                    location=ParameterLocation.QUERY,
                    type="boolean",
                    required=False,
                    default=True,
                ),
            ],
        )

        assert len(action.parameters) == 3
        assert action.is_read_only is False

    def test_is_read_only(self):
        """Test is_read_only property."""
        read_action = AwasAction(
            id="get",
            name="Get",
            method=HttpMethod.GET,
            endpoint="/data",
            intent="Read",
            side_effect=False,
        )
        assert read_action.is_read_only is True

        write_action = AwasAction(
            id="post",
            name="Post",
            method=HttpMethod.POST,
            endpoint="/data",
            intent="Write",
            side_effect=True,
        )
        assert write_action.is_read_only is False

        # GET with side effect (unusual but possible)
        get_with_effect = AwasAction(
            id="get_trigger",
            name="Get Trigger",
            method=HttpMethod.GET,
            endpoint="/trigger",
            intent="Trigger something",
            side_effect=True,
        )
        assert get_with_effect.is_read_only is False


class TestAwasElement:
    """Tests for AwasElement model."""

    def test_element_creation(self):
        """Test creating AWAS element."""
        element = AwasElement(
            id="submit-btn",
            action_id="submit_form",
            selector='[data-awas-action="submit_form"]',
            param_bindings={"form_id": "main-form"},
            trigger="click",
        )

        assert element.id == "submit-btn"
        assert element.action_id == "submit_form"
        assert element.param_bindings["form_id"] == "main-form"

    def test_element_with_bounds(self):
        """Test element with bounding box."""
        element = AwasElement(
            id="btn",
            selector="#btn",
            bounds={"x": 100, "y": 200, "width": 80, "height": 30},
        )

        assert element.bounds is not None
        assert element.bounds["x"] == 100


class TestConformanceLevel:
    """Tests for ConformanceLevel enum."""

    def test_conformance_levels(self):
        """Test all conformance levels exist."""
        assert ConformanceLevel.L1.value == "L1"
        assert ConformanceLevel.L2.value == "L2"
        assert ConformanceLevel.L3.value == "L3"


class TestHttpMethod:
    """Tests for HttpMethod enum."""

    def test_http_methods(self):
        """Test all HTTP methods exist."""
        assert HttpMethod.GET.value == "GET"
        assert HttpMethod.POST.value == "POST"
        assert HttpMethod.PUT.value == "PUT"
        assert HttpMethod.PATCH.value == "PATCH"
        assert HttpMethod.DELETE.value == "DELETE"
