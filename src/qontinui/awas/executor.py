"""
AWAS action executor.

Handles executing AWAS actions via HTTP requests, including parameter
substitution, authentication, and response handling.
"""

import logging
import time
from typing import Any
from urllib.parse import urljoin

import httpx

from .types import (
    AwasAction,
    AwasActionResult,
    AwasAuthType,
    AwasManifest,
    HttpMethod,
    ParameterLocation,
)

logger = logging.getLogger(__name__)


class AwasExecutor:
    """
    Executor for AWAS actions.

    Handles:
    - URL building with path/query parameters
    - Request body construction
    - Authentication header injection
    - Response parsing
    """

    DEFAULT_TIMEOUT_SECONDS = 30.0

    def __init__(self, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS):
        """
        Initialize the executor.

        Args:
            timeout_seconds: Default timeout for HTTP requests
        """
        self._timeout = timeout_seconds

    async def execute(
        self,
        manifest: AwasManifest,
        action_id: str,
        params: dict[str, Any] | None = None,
        credentials: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> AwasActionResult:
        """
        Execute an AWAS action.

        Args:
            manifest: AWAS manifest containing action definitions
            action_id: ID of the action to execute
            params: Parameters to pass to the action
            credentials: Authentication credentials (token, api_key, etc.)
            timeout_seconds: Override default timeout

        Returns:
            AwasActionResult with response data or error
        """
        params = params or {}
        credentials = credentials or {}
        timeout = timeout_seconds or self._timeout

        # Find the action
        action = manifest.get_action(action_id)
        if action is None:
            return AwasActionResult(
                success=False,
                action_id=action_id,
                status_code=0,
                error=f"Action '{action_id}' not found in manifest",
            )

        # Build request components
        url = self._build_url(manifest, action, params)
        body = self._build_body(action, params)
        headers = self._build_headers(action, params, manifest, credentials)

        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(
                    method=action.method.value,
                    url=url,
                    json=body if body else None,
                    headers=headers,
                )

            response_time_ms = int((time.time() - start_time) * 1000)

            # Parse response body
            response_body = None
            if response.content:
                try:
                    response_body = response.json()
                except Exception:
                    # Not JSON, store as string
                    response_body = response.text

            result = AwasActionResult(
                success=response.is_success,
                action_id=action_id,
                status_code=response.status_code,
                response_body=response_body,
                response_time_ms=response_time_ms,
                headers=dict(response.headers),
            )

            if not response.is_success:
                result.error = f"HTTP {response.status_code}: {response.reason_phrase}"

            logger.info(
                f"AWAS action '{action_id}' completed: {response.status_code} "
                f"in {response_time_ms}ms"
            )
            return result

        except httpx.TimeoutException:
            response_time_ms = int((time.time() - start_time) * 1000)
            logger.warning(f"AWAS action '{action_id}' timed out after {response_time_ms}ms")
            return AwasActionResult(
                success=False,
                action_id=action_id,
                status_code=0,
                response_time_ms=response_time_ms,
                error=f"Request timed out after {timeout}s",
            )

        except httpx.RequestError as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            logger.warning(f"AWAS action '{action_id}' failed: {e}")
            return AwasActionResult(
                success=False,
                action_id=action_id,
                status_code=0,
                response_time_ms=response_time_ms,
                error=str(e),
            )

    def execute_sync(
        self,
        manifest: AwasManifest,
        action_id: str,
        params: dict[str, Any] | None = None,
        credentials: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> AwasActionResult:
        """
        Synchronous version of execute().

        Args:
            manifest: AWAS manifest
            action_id: Action to execute
            params: Action parameters
            credentials: Authentication credentials
            timeout_seconds: Override timeout

        Returns:
            AwasActionResult
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.execute(manifest, action_id, params, credentials, timeout_seconds)
        )

    def _build_url(
        self,
        manifest: AwasManifest,
        action: AwasAction,
        params: dict[str, Any],
    ) -> str:
        """Build the full URL for an action, substituting path and query parameters."""
        # Start with endpoint
        endpoint = action.endpoint

        # Substitute path parameters
        for param in action.parameters:
            if param.location == ParameterLocation.PATH:
                placeholder = f"{{{param.name}}}"
                if placeholder in endpoint and param.name in params:
                    value = str(params[param.name])
                    endpoint = endpoint.replace(placeholder, value)

        # Build full URL
        url = urljoin(manifest.base_url.rstrip("/") + "/", endpoint.lstrip("/"))

        # Add query parameters
        query_params = []
        for param in action.parameters:
            if param.location == ParameterLocation.QUERY:
                if param.name in params:
                    value = params[param.name]
                    if isinstance(value, bool):
                        value = str(value).lower()
                    query_params.append(f"{param.name}={value}")
                elif param.required and param.default is not None:
                    query_params.append(f"{param.name}={param.default}")

        if query_params:
            url = f"{url}?{'&'.join(query_params)}"

        return url

    def _build_body(
        self,
        action: AwasAction,
        params: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Build the request body from body parameters."""
        if action.method == HttpMethod.GET:
            return None

        body: dict[str, Any] = {}

        for param in action.parameters:
            if param.location == ParameterLocation.BODY:
                if param.name in params:
                    body[param.name] = params[param.name]
                elif param.required and param.default is not None:
                    body[param.name] = param.default

        return body if body else None

    def _build_headers(
        self,
        action: AwasAction,
        params: dict[str, Any],
        manifest: AwasManifest,
        credentials: dict[str, Any],
    ) -> dict[str, str]:
        """Build request headers including auth and parameter headers."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Qontinui/1.0 (AWAS-compatible)",
        }

        # Add parameter headers
        for param in action.parameters:
            if param.location == ParameterLocation.HEADER:
                if param.name in params:
                    headers[param.name] = str(params[param.name])

        # Add authentication headers
        if manifest.auth and credentials:
            auth_headers = self._build_auth_headers(manifest, credentials)
            headers.update(auth_headers)

        return headers

    def _build_auth_headers(
        self,
        manifest: AwasManifest,
        credentials: dict[str, Any],
    ) -> dict[str, str]:
        """Build authentication headers based on manifest auth config."""
        if not manifest.auth:
            return {}

        auth = manifest.auth
        headers: dict[str, str] = {}

        if auth.type == AwasAuthType.BEARER_TOKEN:
            token = credentials.get("token") or credentials.get("bearer_token")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        elif auth.type == AwasAuthType.API_KEY:
            api_key = credentials.get("api_key")
            header_name = auth.header_name or "X-API-Key"
            if api_key:
                headers[header_name] = api_key

        elif auth.type == AwasAuthType.BASIC:
            username = credentials.get("username")
            password = credentials.get("password")
            if username and password:
                import base64

                encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers["Authorization"] = f"Basic {encoded}"

        return headers

    async def execute_batch(
        self,
        manifest: AwasManifest,
        actions: list[tuple[str, dict[str, Any] | None]],
        credentials: dict[str, Any] | None = None,
        stop_on_error: bool = False,
    ) -> list[AwasActionResult]:
        """
        Execute multiple AWAS actions in sequence.

        Args:
            manifest: AWAS manifest
            actions: List of (action_id, params) tuples
            credentials: Authentication credentials
            stop_on_error: Whether to stop execution on first error

        Returns:
            List of AwasActionResult for each action
        """
        results: list[AwasActionResult] = []

        for action_id, params in actions:
            result = await self.execute(manifest, action_id, params, credentials)
            results.append(result)

            if not result.success and stop_on_error:
                logger.warning(f"Batch execution stopped due to error in action '{action_id}'")
                break

        return results

    def validate_params(
        self,
        action: AwasAction,
        params: dict[str, Any],
    ) -> list[str]:
        """
        Validate parameters against action definition.

        Args:
            action: Action to validate against
            params: Parameters to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []

        # Check required parameters
        for param in action.parameters:
            if param.required and param.name not in params:
                if param.default is None:
                    errors.append(f"Missing required parameter: {param.name}")

        # Check enum constraints
        for param in action.parameters:
            if param.enum and param.name in params:
                value = params[param.name]
                if str(value) not in param.enum:
                    errors.append(
                        f"Invalid value for {param.name}: '{value}' "
                        f"(allowed: {', '.join(param.enum)})"
                    )

        return errors
