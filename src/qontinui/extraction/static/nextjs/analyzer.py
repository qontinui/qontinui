"""
Next.js static code analyzer.

Extends ReactStaticAnalyzer with Next.js specific features including:
- App Router (app/) and Pages Router (pages/) detection
- Server Components and Server Actions
- File-system based routing
- Special Next.js files (layout, loading, error, etc.)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from qontinui.extraction.static.react.analyzer import ReactStaticAnalyzer

if TYPE_CHECKING:
    from qontinui.extraction.config import FrameworkType, StaticConfig
    from qontinui.extraction.models.static import (
        APICallDefinition,
        ComponentDefinition,
        RouteDefinition,
        StaticAnalysisResult,
    )


class NextJSStaticAnalyzer(ReactStaticAnalyzer):
    """
    Extended React analyzer with Next.js specific features.

    Handles both App Router (app/) and Pages Router (pages/).
    """

    def __init__(self):
        """Initialize the Next.js analyzer."""
        super().__init__()
        self._router_type: str | None = None
        self._server_components: list[ComponentDefinition] = []
        self._server_actions: list[APICallDefinition] = []

    async def analyze(self, config: StaticConfig) -> StaticAnalysisResult:
        """
        Analyze Next.js source code.

        Workflow:
        1. Call parent analyze for React features
        2. Detect router type (app/ vs pages/)
        3. Extract routes from file system
        4. Extract server components
        5. Extract server actions

        Args:
            config: Analysis configuration

        Returns:
            Analysis results with Next.js specific features
        """
        # Step 1: Get base React analysis
        result = await super().analyze(config)

        # Step 2: Detect router type
        self._router_type = self._detect_router_type(config.source_root)

        # Step 3: Extract routes based on router type
        if self._router_type == "app":
            from qontinui.extraction.static.nextjs.app_router import extract_app_routes

            app_dir = config.source_root / "app"
            if app_dir.exists():
                app_routes = extract_app_routes(app_dir)
                self._routes.extend(app_routes)

        elif self._router_type == "pages":
            from qontinui.extraction.static.nextjs.pages_router import (
                extract_pages_routes,
            )

            pages_dir = config.source_root / "pages"
            if pages_dir.exists():
                pages_routes = extract_pages_routes(pages_dir)
                self._routes.extend(pages_routes)

        # Step 4: Extract server components (App Router only)
        if self._router_type == "app":
            from qontinui.extraction.static.nextjs.app_router import (
                extract_server_components,
            )

            app_dir = config.source_root / "app"
            if app_dir.exists():
                # Parse results would come from actual AST parsing
                parse_results: dict[str, str] = {}  # Placeholder
                server_components = extract_server_components(app_dir, parse_results)
                self._server_components.extend(server_components)
                self._components.extend(server_components)

        # Step 5: Extract server actions
        from qontinui.extraction.static.nextjs.app_router import extract_server_actions

        parse_results = {}  # Placeholder
        server_actions = extract_server_actions(parse_results)
        self._server_actions.extend(server_actions)
        self._api_calls.extend(server_actions)

        # Extract Pages Router data fetching methods
        if self._router_type == "pages":
            from qontinui.extraction.static.nextjs.pages_router import (
                extract_get_server_side_props,
                extract_get_static_props,
            )

            parse_results = {}  # Placeholder
            gssp = extract_get_server_side_props(parse_results)
            gsp = extract_get_static_props(parse_results)
            self._api_calls.extend(gssp)
            self._api_calls.extend(gsp)

        # Extract Next.js specific hooks
        from qontinui.extraction.static.nextjs.hooks import (
            extract_use_params,
            extract_use_pathname,
            extract_use_router,
            extract_use_search_params,
        )

        parse_results = {}  # Placeholder
        router_state = extract_use_router(parse_results)
        search_params_state = extract_use_search_params(parse_results)
        pathname_state = extract_use_pathname(parse_results)
        params_state = extract_use_params(parse_results)

        self._state_variables.extend(router_state)
        self._state_variables.extend(search_params_state)
        self._state_variables.extend(pathname_state)
        self._state_variables.extend(params_state)

        # Update result with new data
        result.routes = self._routes
        result.components = self._components
        result.state_variables = self._state_variables
        result.api_calls = self._api_calls

        # Note: metadata fields would be added here if StaticAnalysisResult supported them
        # For now, the router type and counts are implicit from the result data

        return result

    def get_routes(self) -> list[RouteDefinition]:
        """
        Extract routes from pages/ or app/ directories.

        Returns:
            List of route definitions based on file system structure
        """
        return self._routes

    def _detect_router_type(self, source_root: Path) -> str:
        """
        Detect if using App Router or Pages Router.

        Args:
            source_root: Root directory of the Next.js project

        Returns:
            "app" for App Router, "pages" for Pages Router, or "unknown"
        """
        app_dir = source_root / "app"
        pages_dir = source_root / "pages"

        has_app_dir = app_dir.exists() and app_dir.is_dir()
        has_pages_dir = pages_dir.exists() and pages_dir.is_dir()

        # Check for page files in app directory
        if has_app_dir:
            app_pages = list(app_dir.glob("**/page.{tsx,ts,jsx,js}"))
            if app_pages:
                return "app"

        # Check for pages directory
        if has_pages_dir:
            # Exclude pages/api directory from check
            page_files = [
                f
                for f in pages_dir.glob("**/*.{tsx,ts,jsx,js}")
                if not f.is_relative_to(pages_dir / "api")
            ]
            if page_files:
                return "pages"

        # Default to app if app directory exists, otherwise pages
        if has_app_dir:
            return "app"
        if has_pages_dir:
            return "pages"

        return "unknown"

    @classmethod
    def supports_framework(cls, framework: FrameworkType) -> bool:
        """Check if this analyzer supports Next.js."""
        # Support both enum naming conventions (NEXT_JS from config, NEXT from base)
        framework_value = (
            framework.value if hasattr(framework, "value") else str(framework)
        )
        return framework_value in ("next_js", "next")
