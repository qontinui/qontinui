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
    ComponentDefinition,
    ConditionalRender,
    EventHandler,
    RouteDefinition,
    StateVariable,
    StaticAnalysisResult,
    StaticConfig,
)
from qontinui.extraction.static.typescript import FileParseResult, TypeScriptParser

from .component_analyzer import ComponentAnalyzer
from .event_analyzer import EventAnalyzer
from .hint_generator import HintGenerator
from .route_analyzer import RouteAnalyzer
from .state_analyzer import StateAnalyzer

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

        # Specialized analyzers
        self.component_analyzer = ComponentAnalyzer()
        self.state_analyzer = StateAnalyzer()
        self.event_analyzer = EventAnalyzer()
        self.route_analyzer = RouteAnalyzer()
        self.hint_generator = HintGenerator()

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
        self.component_analyzer.reset()
        self.state_analyzer.reset()
        self.event_analyzer.reset()
        self.route_analyzer.reset()
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
                error_msg = f"Error parsing batch {batch_start // batch_size + 1}: {str(e)}"
                logger.error(error_msg)
                self._errors.append(error_msg)

        # Build component relationships
        self.component_analyzer.build_relationships()

        # Classify components as states (page-level) or widgets (UI elements)
        self.component_analyzer.classify_components()

        # Get statistics
        stats = self.component_analyzer.get_stats()
        components = self.component_analyzer.get_components()
        state_variables = self.state_analyzer.get_state_variables()
        conditional_renders = self.event_analyzer.get_conditional_renders()
        event_handlers = self.event_analyzer.get_event_handlers()
        visibility_states = self.state_analyzer.get_visibility_states()

        logger.info(
            f"Analysis complete: {stats['total']} total components "
            f"({stats['states']} page-level states, {stats['widgets']} UI widgets), "
            f"{len(state_variables)} state variables, "
            f"{len(conditional_renders)} conditionals, "
            f"{len(event_handlers)} handlers, "
            f"{len(visibility_states)} visibility sub-states"
        )

        # Generate hints for runtime state discovery
        state_hints, state_image_hints, transition_hints = self.hint_generator.generate_hints(
            components=components,
            state_variables=state_variables,
            conditional_renders=conditional_renders,
            event_handlers=event_handlers,
            routes=self.route_analyzer.get_routes(),
            visibility_states=visibility_states,
            navigation_links=self.route_analyzer.get_navigation_links(),
        )

        logger.info(
            f"Generated hints: {len(state_hints)} state hints, "
            f"{len(state_image_hints)} state image hints, "
            f"{len(transition_hints)} transition hints"
        )

        return StaticAnalysisResult(
            components=components,
            state_variables=state_variables,
            conditional_renders=conditional_renders,
            routes=self.route_analyzer.get_routes(),
            event_handlers=event_handlers,
            api_calls=self.route_analyzer.get_api_calls(),
            visibility_states=visibility_states,
            # Hints for runtime state discovery
            state_hints=state_hints,
            state_image_hints=state_image_hints,
            transition_hints=transition_hints,
        )

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
        all_components = self.component_analyzer.extract_components(
            parse_result.to_dict(), file_path
        )

        # For each component, extract state, conditionals, and handlers
        for component in all_components:
            component_parse = self._get_component_parse_result(
                parse_result.to_dict(), component.name
            )

            # Extract state variables
            state_vars = self.state_analyzer.extract_state_for_component(
                component_parse, component.name, file_path
            )

            # Extract conditional renders
            conditionals = self.event_analyzer.extract_conditionals_for_component(
                component_parse, component.name, file_path
            )

            # Extract event handlers
            handlers = self.event_analyzer.extract_event_handlers_for_component(
                component_parse, component.name, file_path, state_vars
            )

            # Extract visibility-based sub-states
            self.state_analyzer.extract_visibility_states(
                component, state_vars, conditionals, handlers
            )

        # Extract API calls
        self.route_analyzer.extract_api_calls(parse_result.to_dict(), file_path)

        # Extract routes (for Next.js App Router, Pages Router, etc.)
        self.route_analyzer.extract_routes(file_path, parse_result.to_dict())

        # Extract navigation links (Link elements with href)
        self.route_analyzer.extract_navigation_links(parse_result, file_path)

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

    def _get_component_parse_result(self, parse_result: dict, component_name: str) -> dict:
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

    # Implement abstract methods from StaticAnalyzer

    def get_components(self) -> list[ComponentDefinition]:
        """Get extracted components."""
        return self.component_analyzer.get_components()

    def get_state_variables(self) -> list[StateVariable]:
        """Get extracted state variables."""
        return self.state_analyzer.get_state_variables()

    def get_conditional_renders(self) -> list[ConditionalRender]:
        """Get extracted conditional rendering patterns."""
        return self.event_analyzer.get_conditional_renders()

    def get_event_handlers(self) -> list[EventHandler]:
        """Get extracted event handlers."""
        return self.event_analyzer.get_event_handlers()

    def get_routes(self) -> list[RouteDefinition]:
        """Get extracted routes."""
        return self.route_analyzer.get_routes()

    def get_api_calls(self) -> list[APICallDefinition]:
        """Get extracted API calls."""
        return self.route_analyzer.get_api_calls()

    def get_navigation_links(self) -> list[dict]:
        """Get extracted navigation links."""
        return self.route_analyzer.get_navigation_links()

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
