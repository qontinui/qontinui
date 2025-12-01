"""
React Static Analyzer.

Main analyzer class for extracting UI structure and state from React codebases.
Supports: React, Next.js, Remix, Tauri (React), Electron (React)
"""

import logging
from pathlib import Path

from qontinui.extraction.config import FrameworkType
from qontinui.extraction.static.base import StaticAnalyzer
from qontinui.extraction.static.models import (
    APICallDefinition,
    ComponentDefinition,
    ConditionalRender,
    EventHandler,
    RouteDefinition,
    StateVariable,
    StaticAnalysisResult,
    StaticConfig,
)
from qontinui.extraction.static.typescript import TypeScriptParser

from . import components as comp_module
from . import handlers as handler_module
from . import hooks as hook_module
from . import jsx as jsx_module

logger = logging.getLogger(__name__)


class ReactStaticAnalyzer(StaticAnalyzer):
    """
    Static analyzer for React-based codebases.

    Supports: React, Next.js, Remix, Tauri (React), Electron (React)

    This analyzer:
    1. Uses TypeScriptParser to parse React/TypeScript/JavaScript files
    2. Extracts components (function and class components)
    3. Extracts state variables (useState, useReducer, useContext, custom hooks)
    4. Extracts conditional rendering patterns (logical AND, ternary, early returns, switch)
    5. Extracts event handlers and traces state mutations
    6. Builds component hierarchies
    7. Correlates state variables with conditional renders
    """

    def __init__(self, parser: TypeScriptParser | None = None):
        """
        Initialize the React analyzer.

        Args:
            parser: Optional TypeScriptParser instance. If not provided, a new one will be created.
        """
        self.parser = parser
        self._components: list[ComponentDefinition] = []
        self._state_variables: list[StateVariable] = []
        self._conditional_renders: list[ConditionalRender] = []
        self._event_handlers: list[EventHandler] = []
        self._routes: list[RouteDefinition] = []
        self._api_calls: list[APICallDefinition] = []
        self._errors: list[str] = []
        self._warnings: list[str] = []

    async def analyze(self, config: StaticConfig) -> StaticAnalysisResult:
        """
        Analyze React source code.

        Steps:
        1. Find all files matching patterns
        2. Parse with TypeScript parser
        3. Extract components, state, conditionals, handlers
        4. Build relationships between components
        5. Return StaticAnalysisResult

        Args:
            config: Configuration specifying source paths and analysis options

        Returns:
            StaticAnalysisResult containing all extracted information
        """
        logger.info("Starting React static analysis")

        # Reset state
        self._components = []
        self._state_variables = []
        self._conditional_renders = []
        self._event_handlers = []
        self._routes = []
        self._api_calls = []
        self._errors = []
        self._warnings = []

        # Initialize parser if not provided
        if self.parser is None:
            from qontinui.extraction.static.typescript import create_parser

            self.parser = create_parser()

        # Find all files to analyze
        source_root = config.source_root
        files_to_analyze = self._find_files(
            source_root, config.include_patterns, config.exclude_patterns
        )

        logger.info(f"Found {len(files_to_analyze)} files to analyze")

        # Parse and extract from each file
        for file_path in files_to_analyze:
            try:
                await self._analyze_file(file_path)
            except Exception as e:
                error_msg = f"Error analyzing {file_path}: {str(e)}"
                logger.error(error_msg)
                self._errors.append(error_msg)

        # Build component relationships
        self._build_relationships()

        logger.info(
            f"Analysis complete: {len(self._components)} components, "
            f"{len(self._state_variables)} state variables, "
            f"{len(self._conditional_renders)} conditionals, "
            f"{len(self._event_handlers)} handlers"
        )

        return StaticAnalysisResult(
            config=config,
            routes=self._routes,
            components=self._components,
            state_variables=self._state_variables,
            api_calls=self._api_calls,
            conditional_renders=self._conditional_renders,
            event_handlers=self._event_handlers,
            errors=self._errors,
            warnings=self._warnings,
            metadata={
                "framework": "react",
                "files_analyzed": len(files_to_analyze),
            },
        )

    async def _analyze_file(self, file_path: Path) -> None:
        """
        Analyze a single file.

        Args:
            file_path: Path to the file to analyze
        """
        logger.debug(f"Analyzing {file_path}")

        # Parse the file
        parse_result = await self.parser.parse_file(str(file_path))

        # Extract components
        function_components = comp_module.extract_function_components(
            parse_result.to_dict(), file_path
        )
        class_components = comp_module.extract_class_components(
            parse_result.to_dict(), file_path
        )

        all_components = function_components + class_components
        self._components.extend(all_components)

        # For each component, extract state, conditionals, and handlers
        for component in all_components:
            component_parse = self._get_component_parse_result(
                parse_result.to_dict(), component.name
            )

            # Extract state variables
            state_vars = self._extract_state_for_component(
                component_parse, component.name, file_path
            )
            self._state_variables.extend(state_vars)

            # Extract conditional renders
            conditionals = self._extract_conditionals_for_component(
                component_parse, component.name, file_path
            )
            self._conditional_renders.extend(conditionals)

            # Extract event handlers
            handlers = handler_module.extract_event_handlers(
                component_parse, component.name, file_path, state_vars
            )
            self._event_handlers.extend(handlers)

        # Extract API calls
        api_calls = self._extract_api_calls(parse_result.to_dict(), file_path)
        self._api_calls.extend(api_calls)

        # Extract routes (for Next.js App Router, Pages Router, etc.)
        routes = self._extract_routes(file_path, parse_result.to_dict())
        self._routes.extend(routes)

    def _find_files(
        self,
        source_root: Path,
        include_patterns: list[str],
        exclude_patterns: list[str],
    ) -> list[Path]:
        """
        Find all files matching the include/exclude patterns.

        Args:
            source_root: Root directory to search
            include_patterns: Glob patterns to include
            exclude_patterns: Glob patterns to exclude

        Returns:
            List of file paths to analyze
        """
        import fnmatch

        files: list[Path] = []

        for include_pattern in include_patterns:
            # Use glob to find matching files
            matching_files = source_root.glob(include_pattern)

            for file_path in matching_files:
                # Check if it matches any exclude pattern
                should_exclude = False
                relative_path = str(file_path.relative_to(source_root))

                for exclude_pattern in exclude_patterns:
                    if fnmatch.fnmatch(relative_path, exclude_pattern):
                        should_exclude = True
                        break

                if not should_exclude and file_path.is_file():
                    files.append(file_path)

        # Remove duplicates
        return list(set(files))

    def _get_component_parse_result(
        self, parse_result: dict, component_name: str
    ) -> dict:
        """
        Get parse result scoped to a specific component.

        Args:
            parse_result: Full file parse result
            component_name: Name of component to scope to

        Returns:
            Parse result containing only nodes within the component
        """
        # For now, return the full parse result
        # A more sophisticated implementation would filter to only nodes
        # within the component's function/class body
        return parse_result

    def _extract_state_for_component(
        self, component_parse: dict, component_name: str, file_path: Path
    ) -> list[StateVariable]:
        """
        Extract all state variables for a component.

        Args:
            component_parse: Parse result for the component
            component_name: Component name
            file_path: Source file path

        Returns:
            List of StateVariable objects
        """
        state_vars: list[StateVariable] = []

        # Extract from different hook types
        state_vars.extend(
            hook_module.extract_use_state(component_parse, component_name, file_path)
        )
        state_vars.extend(
            hook_module.extract_use_reducer(component_parse, component_name, file_path)
        )
        state_vars.extend(
            hook_module.extract_use_context(component_parse, component_name, file_path)
        )
        state_vars.extend(
            hook_module.extract_custom_hooks(component_parse, component_name, file_path)
        )

        return state_vars

    def _extract_conditionals_for_component(
        self, component_parse: dict, component_name: str, file_path: Path
    ) -> list[ConditionalRender]:
        """
        Extract all conditional rendering patterns for a component.

        Args:
            component_parse: Parse result for the component
            component_name: Component name
            file_path: Source file path

        Returns:
            List of ConditionalRender objects
        """
        conditionals: list[ConditionalRender] = []

        # Extract different conditional patterns
        conditionals.extend(
            jsx_module.extract_logical_and(component_parse, component_name, file_path)
        )
        conditionals.extend(
            jsx_module.extract_ternary(component_parse, component_name, file_path)
        )
        conditionals.extend(
            jsx_module.extract_early_returns(component_parse, component_name, file_path)
        )
        conditionals.extend(
            jsx_module.extract_switch_render(component_parse, component_name, file_path)
        )

        return conditionals

    def _extract_api_calls(
        self, parse_result: dict, file_path: Path
    ) -> list[APICallDefinition]:
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
                api_calls.append(
                    APICallDefinition(
                        name=func_name,
                        file_path=file_path,
                        call_type=self._infer_api_call_type(func_name),
                        is_async=func.get("async", False),
                        parameters=self._extract_parameter_names(func),
                        metadata={"line": func.get("line", 0)},
                    )
                )

        return api_calls

    def _extract_routes(
        self, file_path: Path, parse_result: dict
    ) -> list[RouteDefinition]:
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
                    path=route_path,
                    file_path=file_path,
                    route_type="page",
                    is_server_component=True,
                    metadata={"router": "app"},
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
                    path=route_path,
                    file_path=file_path,
                    route_type="page",
                    metadata={"router": "pages"},
                )
            )

        return routes

    def _build_relationships(self) -> None:
        """Build component parent-child relationships."""
        # This would require access to JSX rendering information
        # For now, we'll leave component hierarchies to be built separately
        pass

    def _infer_api_call_type(self, func_name: str) -> str:
        """
        Infer API call type from function name.

        Args:
            func_name: Function name

        Returns:
            API call type
        """
        lower_name = func_name.lower()

        if "fetch" in lower_name:
            return "fetch"
        elif "query" in lower_name:
            return "query"
        elif "mutation" in lower_name:
            return "mutation"
        elif any(
            method in lower_name for method in ["get", "post", "put", "delete", "patch"]
        ):
            return "rest"
        else:
            return "unknown"

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

    # Implement abstract methods from StaticAnalyzer

    def get_components(self) -> list[ComponentDefinition]:
        """Get extracted components."""
        return self._components

    def get_state_variables(self) -> list[StateVariable]:
        """Get extracted state variables."""
        return self._state_variables

    def get_conditional_renders(self) -> list[ConditionalRender]:
        """Get extracted conditional rendering patterns."""
        return self._conditional_renders

    def get_event_handlers(self) -> list[EventHandler]:
        """Get extracted event handlers."""
        return self._event_handlers

    def get_routes(self) -> list[RouteDefinition]:
        """Get extracted routes."""
        return self._routes

    def get_api_calls(self) -> list[APICallDefinition]:
        """Get extracted API calls."""
        return self._api_calls

    @classmethod
    def supports_framework(cls, framework: FrameworkType) -> bool:
        """
        Check if this analyzer supports the given framework.

        Args:
            framework: Framework type to check

        Returns:
            True if this analyzer supports the framework
        """
        supported_frameworks = {
            FrameworkType.REACT,
            FrameworkType.NEXT_JS,
            FrameworkType.REMIX,
            FrameworkType.TAURI,  # When using React
            FrameworkType.ELECTRON,  # When using React
        }

        return framework in supported_frameworks
