"""
Route Analyzer for React.

Handles route extraction and API call detection.
"""

from __future__ import annotations

import logging
from pathlib import Path

from qontinui.extraction.static.models import (
    APICallDefinition,
    APICallType,
    RouteDefinition,
    RouteType,
)

logger = logging.getLogger(__name__)


class RouteAnalyzer:
    """Analyzer for extracting routes and API calls."""

    def __init__(self):
        """Initialize the route analyzer."""
        self.routes: list[RouteDefinition] = []
        self.api_calls: list[APICallDefinition] = []
        self.navigation_links: list[dict] = []

    def extract_routes(self, file_path: Path, parse_result: dict) -> list[RouteDefinition]:
        """
        Extract route definitions (Next.js App Router, Pages Router, etc.).

        Args:
            file_path: Source file path
            parse_result: File parse result

        Returns:
            List of RouteDefinition objects
        """
        routes: list[RouteDefinition] = []

        # Check if this is a Next.js page/route file
        path_str = str(file_path)

        # Next.js App Router: app/path/page.tsx
        if "/app/" in path_str and (
            path_str.endswith("page.tsx")
            or path_str.endswith("page.ts")
            or path_str.endswith("page.jsx")
            or path_str.endswith("page.js")
        ):
            route_path = self._extract_app_router_path(file_path)
            routes.append(
                RouteDefinition(
                    id=f"{file_path.stem}:{route_path}",
                    path=route_path,
                    file_path=file_path,
                    route_type=RouteType.PAGE,
                    metadata={"router": "app", "is_server_component": True},
                )
            )

        # Next.js Pages Router: pages/path.tsx
        elif (
            "/pages/" in path_str
            and not path_str.endswith("/_app.tsx")
            and not path_str.endswith("/_document.tsx")
        ):
            route_path = self._extract_pages_router_path(file_path)
            routes.append(
                RouteDefinition(
                    id=f"{file_path.stem}:{route_path}",
                    path=route_path,
                    file_path=file_path,
                    route_type=RouteType.PAGE,
                    metadata={"router": "pages"},
                )
            )

        self.routes.extend(routes)
        return routes

    def extract_api_calls(self, parse_result: dict, file_path: Path) -> list[APICallDefinition]:
        """
        Extract API calls from the file.

        Args:
            parse_result: File parse result
            file_path: Source file path

        Returns:
            List of APICallDefinition objects
        """
        api_calls: list[APICallDefinition] = []

        # Look for common API call patterns
        function_declarations = parse_result.get("function_declarations", [])

        for func in function_declarations:
            func_name = func.get("name", "")

            # Check if this looks like a data fetching function
            if any(
                keyword in func_name.lower()
                for keyword in [
                    "fetch",
                    "get",
                    "post",
                    "put",
                    "delete",
                    "api",
                    "query",
                    "mutation",
                ]
            ):
                line_num = func.get("line", 0)
                api_calls.append(
                    APICallDefinition(
                        id=f"{file_path.stem}:{line_num}:{func_name}",
                        file_path=file_path,
                        line_number=line_num,
                        method="GET",  # Default, could be inferred from name
                        endpoint="",  # Would need to trace the actual API endpoint
                        call_type=self._infer_api_call_type(func_name),
                        metadata={
                            "name": func_name,
                            "is_async": func.get("async", False),
                            "parameters": self._extract_parameter_names(func),
                        },
                    )
                )

        self.api_calls.extend(api_calls)
        return api_calls

    def extract_navigation_links(self, parse_result, file_path: Path) -> None:
        """
        Extract navigation links from parse result.

        Args:
            parse_result: File parse result
            file_path: Source file path
        """
        nav_links = getattr(parse_result, "navigation_links", [])
        for link in nav_links:
            self.navigation_links.append(
                {
                    "type": link.type,
                    "target": link.target,
                    "line": link.line,
                    "component": link.component,
                    "file": str(file_path),
                }
            )

    def _extract_app_router_path(self, file_path: Path) -> str:
        """
        Extract route path from Next.js App Router file path.

        Args:
            file_path: File path

        Returns:
            Route path
        """
        path_str = str(file_path)
        app_index = path_str.rfind("/app/")

        if app_index == -1:
            return "/"

        # Get everything after /app/ and before /page.*
        route_part = path_str[app_index + 5 :]
        route_part = route_part.split("/page.")[0]

        # Convert to route path
        if not route_part:
            return "/"

        # Handle dynamic segments
        route_part = route_part.replace("[", ":").replace("]", "")

        return "/" + route_part

    def _extract_pages_router_path(self, file_path: Path) -> str:
        """
        Extract route path from Next.js Pages Router file path.

        Args:
            file_path: File path

        Returns:
            Route path
        """
        path_str = str(file_path)
        pages_index = path_str.rfind("/pages/")

        if pages_index == -1:
            return "/"

        # Get everything after /pages/
        route_part = path_str[pages_index + 7 :]

        # Remove file extension
        for ext in [".tsx", ".ts", ".jsx", ".js"]:
            if route_part.endswith(ext):
                route_part = route_part[: -len(ext)]
                break

        # Handle index files
        if route_part.endswith("/index"):
            route_part = route_part[:-6]

        # Convert to route path
        if not route_part:
            return "/"

        # Handle dynamic segments
        route_part = route_part.replace("[", ":").replace("]", "")

        return "/" + route_part

    def _infer_api_call_type(self, func_name: str) -> APICallType:
        """
        Infer API call type from function name.

        Args:
            func_name: Function name

        Returns:
            API call type
        """
        lower_name = func_name.lower()

        if "fetch" in lower_name:
            return APICallType.FETCH
        elif "query" in lower_name or "mutation" in lower_name:
            return APICallType.REACT_QUERY
        elif any(method in lower_name for method in ["get", "post", "put", "delete", "patch"]):
            return APICallType.AXIOS
        else:
            return APICallType.FETCH

    def _extract_parameter_names(self, func_node: dict) -> list[str]:
        """
        Extract parameter names from a function.

        Args:
            func_node: Function AST node

        Returns:
            List of parameter names
        """
        params = func_node.get("parameters", [])
        param_names = []

        for param in params:
            if param.get("type") == "identifier":
                param_names.append(param.get("name", ""))
            elif param.get("type") == "object_pattern":
                # For destructured params, just note it's an object
                param_names.append("{...}")

        return param_names

    def get_routes(self) -> list[RouteDefinition]:
        """Get all extracted routes."""
        return self.routes

    def get_api_calls(self) -> list[APICallDefinition]:
        """Get all extracted API calls."""
        return self.api_calls

    def get_navigation_links(self) -> list[dict]:
        """Get all extracted navigation links."""
        return self.navigation_links

    def reset(self) -> None:
        """Reset the analyzer state."""
        self.routes = []
        self.api_calls = []
        self.navigation_links = []
