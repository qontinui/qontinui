"""
AWAS (AI Web Action Standard) type definitions.

These Pydantic models represent the AWAS manifest schema and related types
for discovering and executing AI-accessible web actions.

Reference: https://github.com/TamTunnel/AWAS
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ConformanceLevel(str, Enum):
    """AWAS conformance levels indicate implementation depth."""

    L1 = "L1"  # Basic: Actions only
    L2 = "L2"  # Typed I/O: Input/output schemas
    L3 = "L3"  # Full: Rate limits, scopes, side effects


class HttpMethod(str, Enum):
    """HTTP methods supported by AWAS actions."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class ParameterLocation(str, Enum):
    """Where a parameter should be placed in the HTTP request."""

    PATH = "path"
    QUERY = "query"
    BODY = "body"
    HEADER = "header"


class AwasAuthType(str, Enum):
    """Authentication types supported by AWAS."""

    BEARER_TOKEN = "bearer_token"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BASIC = "basic"
    NONE = "none"


class AwasScope(BaseModel):
    """OAuth2 scope definition."""

    name: str = Field(..., description="Scope identifier")
    description: str = Field(..., description="Human-readable scope description")


class AwasAuth(BaseModel):
    """Authentication configuration for AWAS manifest."""

    type: AwasAuthType = Field(default=AwasAuthType.NONE, description="Authentication type")
    token_endpoint: str | None = Field(default=None, description="OAuth2 token endpoint")
    authorization_url: str | None = Field(default=None, description="OAuth2 authorization URL")
    header_name: str | None = Field(
        default=None, description="Header name for API key authentication"
    )
    scopes: list[AwasScope] = Field(default_factory=list, description="Available OAuth2 scopes")


class AwasParameter(BaseModel):
    """Parameter definition for an AWAS action."""

    name: str = Field(..., description="Parameter name")
    location: ParameterLocation = Field(..., description="Where to place the parameter")
    type: str = Field(default="string", description="Parameter type (string, integer, boolean, etc.)")
    required: bool = Field(default=False, description="Whether the parameter is required")
    description: str | None = Field(default=None, description="Parameter description")
    default: Any | None = Field(default=None, description="Default value if not provided")
    enum: list[str] | None = Field(default=None, description="Allowed values for the parameter")


class AwasAction(BaseModel):
    """AWAS action definition representing an AI-accessible endpoint."""

    id: str = Field(..., description="Unique action identifier")
    name: str = Field(..., description="Human-readable action name")
    method: HttpMethod = Field(..., description="HTTP method")
    endpoint: str = Field(..., description="API endpoint path (relative to base_url)")
    intent: str = Field(..., description="Description of what this action does")
    side_effect: bool = Field(default=False, description="Whether this action modifies data")
    parameters: list[AwasParameter] = Field(
        default_factory=list, description="Action parameters"
    )
    input_schema: dict[str, Any] | None = Field(
        default=None, description="JSON Schema for request body validation"
    )
    output_schema: dict[str, Any] | None = Field(
        default=None, description="JSON Schema for response structure"
    )
    required_scopes: list[str] = Field(
        default_factory=list, description="Required OAuth2 scopes"
    )
    rate_limit: int | None = Field(
        default=None, description="Rate limit in requests per minute"
    )

    @property
    def is_read_only(self) -> bool:
        """Check if this action is read-only (no side effects)."""
        return not self.side_effect and self.method == HttpMethod.GET


class AwasManifest(BaseModel):
    """
    AWAS manifest representing a website's AI-accessible capabilities.

    This is the main structure served at /.well-known/ai-actions.json.
    """

    schema_version: str = Field(
        alias="schemaVersion",
        default="1.0",
        description="AWAS schema version",
    )
    app_name: str = Field(
        alias="appName",
        description="Application name",
    )
    description: str | None = Field(default=None, description="Application description")
    base_url: str = Field(
        alias="baseUrl",
        description="Base URL for all action endpoints",
    )
    actions: list[AwasAction] = Field(
        default_factory=list, description="Available actions"
    )
    auth: AwasAuth | None = Field(default=None, description="Authentication configuration")
    conformance_level: ConformanceLevel = Field(
        alias="conformanceLevel",
        default=ConformanceLevel.L1,
        description="AWAS conformance level",
    )
    openapi_url: str | None = Field(
        alias="openapiUrl",
        default=None,
        description="Link to OpenAPI/Swagger specification",
    )
    mcp_manifest_url: str | None = Field(
        alias="mcpManifestUrl",
        default=None,
        description="Link to MCP manifest",
    )

    model_config = {"populate_by_name": True}

    def get_action(self, action_id: str) -> AwasAction | None:
        """Find an action by its ID."""
        for action in self.actions:
            if action.id == action_id:
                return action
        return None

    def get_read_only_actions(self) -> list[AwasAction]:
        """Get all read-only (safe) actions."""
        return [action for action in self.actions if action.is_read_only]

    def get_actions_by_method(self, method: HttpMethod) -> list[AwasAction]:
        """Get all actions with a specific HTTP method."""
        return [action for action in self.actions if action.method == method]


class AwasElement(BaseModel):
    """
    AWAS element extracted from HTML data attributes.

    Elements are DOM elements with data-awas-* attributes that map to actions.
    """

    id: str = Field(..., description="Element identifier from data-awas-element")
    action_id: str | None = Field(
        default=None, description="Associated action ID from data-awas-action"
    )
    selector: str = Field(..., description="CSS selector to locate this element")
    param_bindings: dict[str, str] = Field(
        default_factory=dict, description="Parameter bindings from data-awas-param-*"
    )
    trigger: str | None = Field(
        default=None, description="Trigger type from data-awas-trigger (click, submit, etc.)"
    )
    bounds: dict[str, int] | None = Field(
        default=None, description="Element bounding box {x, y, width, height}"
    )


class AwasActionResult(BaseModel):
    """Result of executing an AWAS action."""

    success: bool = Field(..., description="Whether the action succeeded")
    action_id: str = Field(..., description="ID of the executed action")
    status_code: int = Field(..., description="HTTP status code")
    response_body: Any | None = Field(default=None, description="Response body (parsed JSON)")
    response_time_ms: int = Field(default=0, description="Response time in milliseconds")
    error: str | None = Field(default=None, description="Error message if failed")
    headers: dict[str, str] = Field(default_factory=dict, description="Response headers")
