"""
React Static Analyzer.

Main analyzer class for extracting UI structure and state from React codebases.
Supports: React, Next.js, Remix, Tauri (React), Electron (React)
"""

from __future__ import annotations

import logging
from pathlib import Path

from qontinui.extraction.config import FrameworkType
from qontinui.extraction.static.base import StaticAnalyzer
from qontinui.extraction.static.models import (
    APICallDefinition,
    APICallType,
    ComponentCategory,
    ComponentDefinition,
    ConditionalRender,
    EventHandler,
    RouteDefinition,
    StateHint,
    StateImageHint,
    StateVariable,
    StaticAnalysisResult,
    StaticConfig,
    TransitionHint,
    VisibilityState,
)
from qontinui.extraction.static.typescript import FileParseResult, TypeScriptParser

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
        self._navigation_links: list[dict] = (
            []
        )  # Navigation links from JSX Link elements
        self._visibility_states: list[VisibilityState] = []
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
        self._navigation_links = []
        self._visibility_states = []
        self._errors = []
        self._warnings = []

        # Initialize parser if not provided
        if self.parser is None:
            from qontinui.extraction.static.typescript import create_parser

            self.parser = create_parser()

        assert self.parser is not None  # For type checking

        # Find all files to analyze
        source_root = config.source_root
        files_to_analyze = self._find_files(
            source_root, config.include_patterns, config.exclude_patterns
        )

        logger.info(f"Found {len(files_to_analyze)} files to analyze")

        # Parse files in batches to avoid memory issues
        # Using file-based output, so we can use larger batches
        batch_size = 100  # Process 100 files at a time
        total_files = len(files_to_analyze)

        for batch_start in range(0, total_files, batch_size):
            batch_end = min(batch_start + batch_size, total_files)
            batch_files = files_to_analyze[batch_start:batch_end]

            logger.info(
                f"Parsing batch {batch_start // batch_size + 1}/{(total_files + batch_size - 1) // batch_size} ({len(batch_files)} files)"
            )

            try:
                batch_result = await self.parser.parse_files(batch_files)

                # Process parse results for each file in the batch
                for file_path in batch_files:
                    try:
                        file_key = str(file_path.resolve())
                        if file_key in batch_result.files:
                            parse_result = batch_result.files[file_key]
                            self._process_parse_result(parse_result, file_path)
                        else:
                            # Try with just the file path string
                            for key in batch_result.files:
                                if key.endswith(file_path.name):
                                    parse_result = batch_result.files[key]
                                    self._process_parse_result(parse_result, file_path)
                                    break
                    except Exception as e:
                        error_msg = f"Error processing {file_path}: {str(e)}"
                        logger.error(error_msg)
                        self._errors.append(error_msg)
            except Exception as e:
                error_msg = (
                    f"Error parsing batch {batch_start // batch_size + 1}: {str(e)}"
                )
                logger.error(error_msg)
                self._errors.append(error_msg)

        # Build component relationships
        self._build_relationships()

        # Classify components as states (page-level) or widgets (UI elements)
        comp_module.classify_components(self._components)

        # Count states vs widgets
        state_count = sum(
            1 for c in self._components if c.category == ComponentCategory.STATE
        )
        widget_count = sum(
            1 for c in self._components if c.category == ComponentCategory.WIDGET
        )

        logger.info(
            f"Analysis complete: {len(self._components)} total components "
            f"({state_count} page-level states, {widget_count} UI widgets), "
            f"{len(self._state_variables)} state variables, "
            f"{len(self._conditional_renders)} conditionals, "
            f"{len(self._event_handlers)} handlers, "
            f"{len(self._visibility_states)} visibility sub-states"
        )

        # Generate hints for runtime state discovery
        state_hints, state_image_hints, transition_hints = self._generate_hints()

        logger.info(
            f"Generated hints: {len(state_hints)} state hints, "
            f"{len(state_image_hints)} state image hints, "
            f"{len(transition_hints)} transition hints"
        )

        return StaticAnalysisResult(
            components=self._components,
            state_variables=self._state_variables,
            conditional_renders=self._conditional_renders,
            routes=self._routes,
            event_handlers=self._event_handlers,
            api_calls=self._api_calls,
            visibility_states=self._visibility_states,
            # Hints for runtime state discovery
            state_hints=state_hints,
            state_image_hints=state_image_hints,
            transition_hints=transition_hints,
        )

    async def _analyze_file(self, file_path: Path) -> None:
        """
        Analyze a single file.

        Args:
            file_path: Path to the file to analyze
        """
        logger.debug(f"Analyzing {file_path}")

        assert self.parser is not None  # Ensured by analyze() method

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

            # Extract visibility-based sub-states
            visibility_states = self._extract_visibility_states(
                component, state_vars, conditionals, handlers
            )
            self._visibility_states.extend(visibility_states)

        # Extract API calls
        api_calls = self._extract_api_calls(parse_result.to_dict(), file_path)
        self._api_calls.extend(api_calls)

        # Extract routes (for Next.js App Router, Pages Router, etc.)
        routes = self._extract_routes(file_path, parse_result.to_dict())
        self._routes.extend(routes)

    def _process_parse_result(
        self,
        parse_result: FileParseResult,  # type: ignore
        file_path: Path,
    ) -> None:
        """
        Process a parse result for a single file.

        This method extracts components, state, conditionals, handlers,
        API calls, and routes from the parse result.

        Args:
            parse_result: The FileParseResult from the batch parser
            file_path: Path to the file being processed
        """
        logger.debug(f"Processing {file_path}")

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

            # Extract visibility-based sub-states
            visibility_states = self._extract_visibility_states(
                component, state_vars, conditionals, handlers
            )
            self._visibility_states.extend(visibility_states)

        # Extract API calls
        api_calls = self._extract_api_calls(parse_result.to_dict(), file_path)
        self._api_calls.extend(api_calls)

        # Extract routes (for Next.js App Router, Pages Router, etc.)
        routes = self._extract_routes(file_path, parse_result.to_dict())
        self._routes.extend(routes)

        # Extract navigation links (Link elements with href)
        nav_links = getattr(parse_result, "navigation_links", [])
        for link in nav_links:
            self._navigation_links.append(
                {
                    "type": link.type,
                    "target": link.target,
                    "line": link.line,
                    "component": link.component,
                    "file": str(file_path),
                }
            )

    def _find_files(
        self,
        source_root: Path,
        include_patterns: list[str],
        exclude_patterns: list[str],
    ) -> list[Path]:
        """
        Find all files matching the include/exclude patterns.

        Uses os.walk with directory pruning to avoid traversing into excluded
        directories like node_modules, which is much faster than using glob
        and then filtering.

        Args:
            source_root: Root directory to search
            include_patterns: Glob patterns to include
            exclude_patterns: Glob patterns to exclude

        Returns:
            List of file paths to analyze
        """
        import fnmatch
        import os

        # Extract directory names to skip from exclude patterns
        excluded_dirs = set()
        for pattern in exclude_patterns:
            # Extract directory names from patterns like "**/node_modules/**"
            parts = pattern.replace("**", "").strip("/").split("/")
            for part in parts:
                if part and part != "*":
                    excluded_dirs.add(part)

        # Extract file extensions from include patterns
        include_extensions = set()
        for pattern in include_patterns:
            if "*." in pattern:
                ext = pattern.split("*.")[-1]
                include_extensions.add(f".{ext}")

        files: list[Path] = []

        for root, dirs, filenames in os.walk(source_root):
            # Prune excluded directories IN PLACE to prevent os.walk from descending
            dirs[:] = [d for d in dirs if d not in excluded_dirs]

            for filename in filenames:
                # Check extension
                file_ext = Path(filename).suffix
                if file_ext not in include_extensions:
                    continue

                file_path = Path(root) / filename

                # Double-check with pattern matching for any remaining patterns
                relative_path = str(file_path.relative_to(source_root))
                should_exclude = False

                for exclude_pattern in exclude_patterns:
                    # Check if any excluded directory is in the path
                    path_parts = relative_path.split(os.sep)
                    for part in path_parts:
                        if part in excluded_dirs:
                            should_exclude = True
                            break

                    if should_exclude:
                        break

                    # Also check with fnmatch for non-directory patterns
                    if fnmatch.fnmatch(relative_path, exclude_pattern):
                        should_exclude = True
                        break

                if not should_exclude:
                    files.append(file_path)

        return files

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

    def _extract_visibility_states(
        self,
        component: ComponentDefinition,
        state_vars: list[StateVariable],
        conditionals: list[ConditionalRender],
        handlers: list[EventHandler],
    ) -> list[VisibilityState]:
        """
        Extract visibility-based sub-states from a component.

        Analyzes conditional rendering patterns and state variables to detect
        different UI configurations of the same page (e.g., modal open/closed,
        sidebar expanded/collapsed, dropdown visible/hidden).

        Args:
            component: The component to analyze
            state_vars: State variables in this component
            conditionals: Conditional rendering patterns in this component
            handlers: Event handlers in this component

        Returns:
            List of VisibilityState objects representing sub-states
        """
        visibility_states: list[VisibilityState] = []

        # Build a map of state variable names to their IDs for quick lookup
        {var.name: var for var in state_vars}

        # Identify visibility-controlling state variables
        # These typically have names like: isOpen, showModal, menuExpanded, etc.
        visibility_vars = self._identify_visibility_variables(state_vars)

        if not visibility_vars:
            # No visibility-controlling variables found
            return visibility_states

        # For each visibility variable, create sub-states
        for var in visibility_vars:
            # Find conditionals that use this variable
            related_conditionals = [
                cond
                for cond in conditionals
                if var.id in cond.controlling_variables or var.name in cond.condition
            ]

            if not related_conditionals:
                continue

            # Find event handlers that toggle this variable
            toggle_handlers = self._find_toggle_handlers(var, handlers)

            # For boolean visibility variables, create two states: visible and hidden
            if (
                var.initial_value is False
                or var.initial_value is True
                or self._is_boolean_var(var)
            ):
                # State 1: Variable is False (default/closed/hidden)
                default_state = VisibilityState(
                    id=f"{component.id}:{var.name}_false",
                    name=f"{component.name}_{var.name}_false",
                    parent_component=component.id,
                    parent_route=component.route_path,
                    controlling_variable=var.id,
                    variable_value=False,
                    rendered_components=[],  # Nothing extra rendered
                    hidden_components=self._extract_rendered_components(
                        related_conditionals, True
                    ),
                    toggle_handlers=[h.id for h in toggle_handlers],
                    conditional_render_id=(
                        related_conditionals[0].id if related_conditionals else None
                    ),
                    file_path=component.file_path,
                    line_number=var.line_number,
                    metadata={
                        "variable_name": var.name,
                        "is_default": var.initial_value is False,
                    },
                )
                visibility_states.append(default_state)

                # State 2: Variable is True (visible/open/expanded)
                visible_state = VisibilityState(
                    id=f"{component.id}:{var.name}_true",
                    name=f"{component.name}_{var.name}_true",
                    parent_component=component.id,
                    parent_route=component.route_path,
                    controlling_variable=var.id,
                    variable_value=True,
                    rendered_components=self._extract_rendered_components(
                        related_conditionals, True
                    ),
                    hidden_components=self._extract_rendered_components(
                        related_conditionals, False
                    ),
                    toggle_handlers=[h.id for h in toggle_handlers],
                    conditional_render_id=(
                        related_conditionals[0].id if related_conditionals else None
                    ),
                    file_path=component.file_path,
                    line_number=var.line_number,
                    metadata={
                        "variable_name": var.name,
                        "is_default": var.initial_value is True,
                    },
                )
                visibility_states.append(visible_state)

        return visibility_states

    def _identify_visibility_variables(
        self, state_vars: list[StateVariable]
    ) -> list[StateVariable]:
        """
        Identify state variables that likely control visibility.

        Common patterns:
        - is*, show*, *Open, *Visible, *Expanded, *Active, *Hidden, *Collapsed
        - Boolean variables used in conditional rendering

        Args:
            state_vars: List of state variables to filter

        Returns:
            List of state variables that appear to control visibility
        """
        visibility_vars = []

        # Common visibility-related prefixes and suffixes
        visibility_patterns = [
            "is",
            "show",
            "hide",
            "visible",
            "hidden",
            "open",
            "closed",
            "expanded",
            "collapsed",
            "active",
            "inactive",
            "enabled",
            "disabled",
            "toggle",
            "display",
            "render",
            "mounted",
        ]

        for var in state_vars:
            var_name_lower = var.name.lower()

            # Check if name contains visibility patterns
            for pattern in visibility_patterns:
                if (
                    var_name_lower.startswith(pattern)
                    or var_name_lower.endswith(pattern)
                    or pattern in var_name_lower
                ):
                    visibility_vars.append(var)
                    break
            else:
                # Also check if initial value is boolean (strong indicator)
                if var.initial_value is True or var.initial_value is False:
                    visibility_vars.append(var)

        return visibility_vars

    def _is_boolean_var(self, var: StateVariable) -> bool:
        """Check if a state variable is boolean-typed."""
        if var.value_type and "boolean" in var.value_type.lower():
            return True
        if var.initial_value is True or var.initial_value is False:
            return True
        return False

    def _find_toggle_handlers(
        self, var: StateVariable, handlers: list[EventHandler]
    ) -> list[EventHandler]:
        """
        Find event handlers that toggle or modify a state variable.

        Args:
            var: The state variable to find handlers for
            handlers: List of all event handlers

        Returns:
            List of handlers that modify this variable
        """
        toggle_handlers = []

        for handler in handlers:
            # Check if this handler modifies the state variable
            if var.id in handler.state_changes:
                toggle_handlers.append(handler)

        return toggle_handlers

    def _extract_rendered_components(
        self, conditionals: list[ConditionalRender], when_true: bool
    ) -> list[str]:
        """
        Extract component names that are rendered based on conditional state.

        Args:
            conditionals: List of conditional renders to analyze
            when_true: If True, extract components rendered when condition is true;
                      if False, extract components rendered when condition is false

        Returns:
            List of component/element names rendered in this state
        """
        components = []

        for cond in conditionals:
            if when_true:
                components.extend(cond.renders_when_true)
            else:
                components.extend(cond.renders_when_false)

        # Remove duplicates while preserving order
        seen = set()
        unique_components = []
        for comp in components:
            if comp and comp not in seen:
                seen.add(comp)
                unique_components.append(comp)

        return unique_components

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
            from qontinui.extraction.static.models import RouteType

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
            from qontinui.extraction.static.models import RouteType

            routes.append(
                RouteDefinition(
                    id=f"{file_path.stem}:{route_path}",
                    path=route_path,
                    file_path=file_path,
                    route_type=RouteType.PAGE,
                    metadata={"router": "pages"},
                )
            )

        return routes

    def _build_relationships(self) -> None:
        """Build component parent-child relationships."""
        # This would require access to JSX rendering information
        # For now, we'll leave component hierarchies to be built separately
        pass

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
        elif any(
            method in lower_name for method in ["get", "post", "put", "delete", "patch"]
        ):
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

    def _generate_hints(
        self,
    ) -> tuple[list[StateHint], list[StateImageHint], list[TransitionHint]]:
        """
        Generate hints for runtime state discovery.

        This method analyzes the extracted components, routes, conditional renders,
        event handlers, and visibility states to produce hints that guide runtime
        state discovery.

        Returns:
            Tuple of (state_hints, state_image_hints, transition_hints)
        """
        state_hints: list[StateHint] = []
        state_image_hints: list[StateImageHint] = []
        transition_hints: list[TransitionHint] = []

        # Build lookup maps
        component_by_id = {c.id: c for c in self._components}
        {s.id: s for s in self._state_variables}
        {h.id: h for h in self._event_handlers}

        # 1. Generate StateHints from routes (each route is a potential state)
        for route in self._routes:
            state_hint = StateHint(
                id=f"state_hint_{route.id}",
                name=self._route_to_state_name(route.path),
                source_type="route",
                file_path=route.file_path,
                line_number=0,
                route_path=route.path,
                route_params=[p.name for p in route.params],
                metadata={"route_type": route.route_type.value},
            )
            state_hints.append(state_hint)

        # 2. Generate StateHints from visibility states (sub-states within pages)
        for vis_state in self._visibility_states:
            parent_hint_id = None
            # Find parent route state hint
            parent_comp = component_by_id.get(vis_state.parent_component)
            if parent_comp and parent_comp.route_path:
                parent_hint_id = f"state_hint_route_{parent_comp.route_path}"

            state_hint = StateHint(
                id=f"state_hint_{vis_state.id}",
                name=vis_state.name,
                source_type="conditional_render",
                file_path=vis_state.file_path,
                line_number=vis_state.line_number,
                parent_state_hint_id=parent_hint_id,
                controlling_variable=vis_state.controlling_variable,
                condition_value=vis_state.variable_value,
                metadata={
                    "rendered_components": vis_state.rendered_components,
                    "hidden_components": vis_state.hidden_components,
                },
            )
            state_hints.append(state_hint)

        # 3. Generate StateImageHints from interactive components
        for component in self._components:
            # Skip non-interactive widgets
            if component.category != ComponentCategory.WIDGET:
                continue

            # Find event handlers attached to this component
            component_handlers = [
                h for h in self._event_handlers if h.trigger_element == component.id
            ]

            if not component_handlers:
                continue

            # Determine interaction type
            interaction_types = {h.event_type for h in component_handlers}
            primary_interaction = (
                "click" if "click" in interaction_types else list(interaction_types)[0]
            )

            # Check if conditionally rendered
            is_conditional = any(
                component.id in cr.renders_when_true
                or component.id in cr.renders_when_false
                for cr in self._conditional_renders
            )

            state_image_hint = StateImageHint(
                id=f"state_image_hint_{component.id}",
                name=component.name,
                component_id=component.id,
                file_path=component.file_path,
                line_number=component.line_number,
                element_type=self._infer_element_type(component.name),
                jsx_element_name=component.name,
                is_interactive=True,
                interaction_type=primary_interaction,
                conditionally_rendered=is_conditional,
                metadata={"handlers": [h.id for h in component_handlers]},
            )
            state_image_hints.append(state_image_hint)

        # 4. Generate TransitionHints from navigation links
        for nav_link in self._navigation_links:
            target_path = nav_link.get("target", "")
            if not target_path or target_path.startswith("http"):
                continue  # Skip external links

            # Find source state hint (based on file/component)
            from_state = None
            source_file = nav_link.get("file", "")
            for sh in state_hints:
                if sh.file_path and str(sh.file_path) == source_file:
                    from_state = sh.id
                    break

            # Find target state hint
            to_state = None
            for sh in state_hints:
                if sh.route_path == target_path:
                    to_state = sh.id
                    break

            transition_hint = TransitionHint(
                id=f"transition_hint_nav_{len(transition_hints)}",
                from_state_hint=from_state,
                to_state_hint=to_state,
                trigger_type="navigation",
                navigation_path=target_path,
                file_path=Path(source_file) if source_file else None,
                line_number=nav_link.get("line", 0),
                confidence=0.8 if to_state else 0.5,
            )
            transition_hints.append(transition_hint)

        # 5. Generate TransitionHints from event handlers that modify state
        for handler in self._event_handlers:
            # Check if handler navigates
            if handler.navigation:
                to_state = None
                for sh in state_hints:
                    if sh.route_path == handler.navigation:
                        to_state = sh.id
                        break

                transition_hint = TransitionHint(
                    id=f"transition_hint_handler_{handler.id}",
                    from_state_hint=None,  # Would need component context
                    to_state_hint=to_state,
                    trigger_type=handler.event_type,
                    event_handler_id=handler.id,
                    navigation_path=handler.navigation,
                    file_path=handler.file_path,
                    line_number=handler.line_number,
                    confidence=0.7,
                )
                transition_hints.append(transition_hint)

            # Check if handler changes visibility state
            for state_change in handler.state_changes:
                # Find visibility states controlled by this variable
                for vis_state in self._visibility_states:
                    if vis_state.controlling_variable == state_change:
                        transition_hint = TransitionHint(
                            id=f"transition_hint_vis_{handler.id}_{vis_state.id}",
                            from_state_hint=None,
                            to_state_hint=f"state_hint_{vis_state.id}",
                            trigger_type="state_change",
                            event_handler_id=handler.id,
                            file_path=handler.file_path,
                            line_number=handler.line_number,
                            confidence=0.6,
                            metadata={"state_variable": state_change},
                        )
                        transition_hints.append(transition_hint)

        return state_hints, state_image_hints, transition_hints

    def _route_to_state_name(self, route_path: str) -> str:
        """Convert a route path to a readable state name."""
        if route_path == "/" or route_path == "":
            return "HomePage"

        # Remove leading slash and convert to PascalCase
        parts = route_path.strip("/").split("/")
        name_parts = []
        for part in parts:
            if part.startswith(":"):
                # Dynamic segment
                name_parts.append(part[1:].title() + "Dynamic")
            elif part.startswith("[") and part.endswith("]"):
                # Next.js dynamic segment
                name_parts.append(part[1:-1].title() + "Dynamic")
            else:
                name_parts.append(part.title().replace("-", "").replace("_", ""))

        return "".join(name_parts) + "Page"

    def _infer_element_type(self, component_name: str) -> str:
        """Infer the element type from the component name."""
        name_lower = component_name.lower()

        if "button" in name_lower or "btn" in name_lower:
            return "button"
        elif "input" in name_lower or "field" in name_lower or "text" in name_lower:
            return "input"
        elif "icon" in name_lower:
            return "icon"
        elif "image" in name_lower or "img" in name_lower or "avatar" in name_lower:
            return "image"
        elif "link" in name_lower or "anchor" in name_lower:
            return "link"
        elif "modal" in name_lower or "dialog" in name_lower:
            return "modal"
        elif "menu" in name_lower or "dropdown" in name_lower:
            return "menu"
        elif "card" in name_lower:
            return "card"
        elif "list" in name_lower or "table" in name_lower:
            return "list"
        else:
            return "component"

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

    def get_navigation_links(self) -> list[dict]:
        """Get extracted navigation links."""
        return self._navigation_links

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
