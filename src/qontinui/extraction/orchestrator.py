"""
Extraction orchestrator that coordinates the extraction process.

The ExtractionOrchestrator manages the full extraction pipeline:
- Framework detection
- Static code analysis
- Runtime extraction
- State correlation and verification
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from .config import FrameworkType as ConfigFrameworkType
from .models.base import (
    ConfigError,
    CorrelatedState,
    ExtractionConfig,
    ExtractionMode,
    ExtractionResult,
    ExtractionTarget,
    FrameworkType,
    InferredTransition,
    RuntimeExtractionResult,
    RuntimeExtractor,
    StateMatcher,
    StaticAnalysisResult,
    StaticAnalyzer,
)

logger = logging.getLogger(__name__)


class ExtractionOrchestrator:
    """Main orchestrator that coordinates extraction components."""

    def __init__(self) -> None:
        """Initialize the orchestrator with empty registries."""
        self.static_analyzers: dict[FrameworkType, type[StaticAnalyzer]] = {}
        self.runtime_extractors: list[type[RuntimeExtractor]] = []
        self.matchers: dict[FrameworkType, type[StateMatcher]] = {}
        self._register_defaults()

    def register_static_analyzer(
        self, framework: FrameworkType, analyzer_class: type[StaticAnalyzer]
    ) -> None:
        """
        Register a static analyzer for a framework.

        Args:
            framework: The framework this analyzer supports
            analyzer_class: The analyzer class (not an instance)
        """
        logger.info(f"Registering static analyzer for {framework.value}: {analyzer_class.__name__}")
        self.static_analyzers[framework] = analyzer_class

    def register_runtime_extractor(self, extractor_class: type[RuntimeExtractor]) -> None:
        """
        Register a runtime extractor.

        Args:
            extractor_class: The extractor class (not an instance)
        """
        logger.info(f"Registering runtime extractor: {extractor_class.__name__}")
        self.runtime_extractors.append(extractor_class)

    def register_matcher(self, framework: FrameworkType, matcher_class: type[StateMatcher]) -> None:
        """
        Register a state matcher for a framework.

        Args:
            framework: The framework this matcher supports
            matcher_class: The matcher class (not an instance)
        """
        logger.info(f"Registering state matcher for {framework.value}: {matcher_class.__name__}")
        self.matchers[framework] = matcher_class

    async def extract(self, config: ExtractionConfig) -> ExtractionResult:
        """
        Execute extraction based on configuration.

        This is the main entry point that orchestrates the entire extraction process:
        1. Validate configuration
        2. Detect framework (if not specified)
        3. Run static analysis (if applicable)
        4. Run runtime extraction (if applicable)
        5. Correlate results (if WHITE_BOX mode)

        Args:
            config: Extraction configuration

        Returns:
            Complete extraction results

        Raises:
            ConfigError: If configuration is invalid
            RuntimeError: If extraction fails
        """
        # Validate configuration
        self._validate_config(config)

        # Create result
        extraction_id = str(uuid.uuid4())
        result = ExtractionResult(
            extraction_id=extraction_id,
            framework=FrameworkType.UNKNOWN,
            mode=config.mode,
        )

        try:
            # Detect framework
            framework = await self._detect_framework(config.target)
            result.framework = framework
            logger.info(f"Detected framework: {framework.value}")

            # Phase 0: HYBRID mode uses tech stack-specific extractors
            if config.mode == ExtractionMode.HYBRID:
                logger.info("Using hybrid extraction mode...")
                await self._run_hybrid_extraction(config, result)
                # Mark completion
                from qontinui_schemas.common import utc_now

                result.completed_at = utc_now()
                logger.info(
                    f"Hybrid extraction complete. States: {len(result.states)}, "
                    f"Transitions: {len(result.transitions)}"
                )
                return result

            # Phase 1: Static Analysis (STATIC_ONLY or WHITE_BOX)
            if config.mode in (ExtractionMode.STATIC_ONLY, ExtractionMode.WHITE_BOX):
                logger.info("Starting static analysis phase...")
                result.static_analysis = await self._run_static_analysis(config, framework)

                if result.static_analysis.errors:
                    logger.warning(
                        f"Static analysis completed with {len(result.static_analysis.errors)} errors"
                    )
                    result.warnings.extend(result.static_analysis.warnings)
                    result.errors.extend(result.static_analysis.errors)

            # Phase 2: Runtime Extraction (BLACK_BOX or WHITE_BOX)
            if config.mode in (ExtractionMode.BLACK_BOX, ExtractionMode.WHITE_BOX):
                logger.info("=" * 50)
                logger.info("PHASE 2: RUNTIME EXTRACTION")
                logger.info("=" * 50)
                start_runtime = time.time()
                logger.info("[PERF_DEBUG] Starting _run_runtime_extraction()")
                result.runtime_extraction = await self._run_runtime_extraction(config)
                duration_runtime = time.time() - start_runtime
                logger.info(
                    f"[PERF_DEBUG] _run_runtime_extraction() finished in {duration_runtime:.2f}s"
                )
                logger.info(
                    f"Runtime extraction returned {len(result.runtime_extraction.states)} states, "
                    f"{len(result.runtime_extraction.elements)} elements"
                )

                # Use runtime extractor's extraction_id if available (for screenshot retrieval)
                if result.runtime_extraction.extraction_id:
                    result.extraction_id = result.runtime_extraction.extraction_id
                    logger.info(f"Using runtime extraction_id: {result.extraction_id}")
                # Log state details
                for i, state in enumerate(result.runtime_extraction.states):
                    state_name = getattr(state, "name", "Unknown")
                    state_id = getattr(state, "id", f"state_{i}")
                    state_type = getattr(state, "state_type", "Unknown")
                    logger.info(
                        f"  Runtime state {i}: id={state_id}, name={state_name}, type={state_type}"
                    )

                if result.runtime_extraction.errors:
                    logger.warning(
                        f"Runtime extraction completed with {len(result.runtime_extraction.errors)} errors"
                    )
                    result.errors.extend(result.runtime_extraction.errors)

            # Phase 3: Correlation (WHITE_BOX only)
            if config.mode == ExtractionMode.WHITE_BOX:
                if result.static_analysis and result.runtime_extraction:
                    logger.info("Starting correlation phase...")
                    await self._correlate_results(config, result)
                else:
                    error_msg = "WHITE_BOX mode requires both static and runtime results"
                    logger.error(error_msg)
                    result.errors.append(error_msg)

            # Handle STATIC_ONLY mode - convert static results to states
            elif config.mode == ExtractionMode.STATIC_ONLY and result.static_analysis:
                logger.info("Converting static analysis to preliminary states...")
                result.states = self._states_from_static(result.static_analysis)
                result.transitions = self._transitions_from_static(result.static_analysis)

            # Handle BLACK_BOX mode - convert runtime results to CorrelatedState
            elif config.mode == ExtractionMode.BLACK_BOX and result.runtime_extraction:
                logger.info("=" * 50)
                logger.info("BLACK_BOX MODE: CONVERTING RUNTIME STATES AND TRANSITIONS")
                logger.info("=" * 50)
                logger.info(
                    f"Converting {len(result.runtime_extraction.states)} runtime states to CorrelatedState..."
                )
                result.states = self._states_from_runtime(result.runtime_extraction)
                logger.info(
                    f"Conversion complete: {len(result.states)} CorrelatedState objects created"
                )
                for i, state in enumerate(result.states):
                    logger.info(
                        f"  CorrelatedState {i}: id={state.id}, name={state.name}, confidence={state.confidence}"
                    )

                # Convert runtime transitions
                if result.runtime_extraction.transitions:
                    logger.info(
                        f"Converting {len(result.runtime_extraction.transitions)} runtime transitions..."
                    )
                    result.transitions = self._transitions_from_runtime(result.runtime_extraction)
                    logger.info(
                        f"Conversion complete: {len(result.transitions)} transitions created"
                    )

            # Mark completion
            from qontinui_schemas.common import utc_now

            result.completed_at = utc_now()

            logger.info(
                f"Extraction complete. States: {len(result.states)}, "
                f"Transitions: {len(result.transitions)}"
            )

            return result

        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)
            result.errors.append(str(e))
            raise

    def _validate_config(self, config: ExtractionConfig) -> None:
        """
        Validate configuration is complete for the requested mode.

        Args:
            config: Configuration to validate

        Raises:
            ConfigError: If configuration is invalid
        """
        try:
            config.validate()
        except ConfigError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

        # Additional orchestrator-specific validation
        if config.mode in (ExtractionMode.STATIC_ONLY, ExtractionMode.WHITE_BOX):
            if not config.target.project_path:
                raise ConfigError(f"{config.mode.value} mode requires target.project_path")

        if config.mode in (ExtractionMode.BLACK_BOX, ExtractionMode.WHITE_BOX):
            if not (config.target.url or config.target.executable_path or config.target.app_id):
                raise ConfigError(
                    f"{config.mode.value} mode requires target.url, executable_path, or app_id"
                )

    async def _detect_framework(self, target: ExtractionTarget) -> FrameworkType:
        """
        Auto-detect framework from project files.

        Checks for framework-specific files and configuration:
        - package.json for JavaScript/TypeScript projects
        - pubspec.yaml for Flutter
        - tauri.conf.json for Tauri
        - etc.

        Args:
            target: The extraction target

        Returns:
            Detected framework type
        """
        # If framework is explicitly specified, use it
        if target.framework:
            logger.info(f"Using explicitly specified framework: {target.framework.value}")
            return target.framework

        # Otherwise, auto-detect from project files
        if not target.project_path:
            # No project path - can't detect, default to generic
            if target.url:
                logger.info("No project path, assuming generic web application")
                return FrameworkType.WEB
            logger.warning("Cannot detect framework without project_path or framework hint")
            return FrameworkType.UNKNOWN

        project_path = target.project_path

        # Check for Tauri (desktop)
        tauri_config = project_path / "src-tauri" / "tauri.conf.json"
        if tauri_config.exists():
            logger.info("Detected Tauri project")
            return FrameworkType.TAURI

        # Check for Flutter (mobile/desktop)
        pubspec = project_path / "pubspec.yaml"
        if pubspec.exists():
            logger.info("Detected Flutter project")
            return FrameworkType.FLUTTER

        # Check for JavaScript/TypeScript projects
        package_json = project_path / "package.json"
        if package_json.exists():
            framework = self._detect_js_framework(package_json)
            if framework != FrameworkType.UNKNOWN:
                return framework

        # Default to generic based on target type
        if target.url:
            logger.info("Defaulting to generic web framework")
            return FrameworkType.WEB
        elif target.executable_path:
            logger.info("Defaulting to generic desktop framework")
            return FrameworkType.DESKTOP
        elif target.app_id:
            logger.info("Defaulting to generic mobile framework")
            return FrameworkType.MOBILE

        logger.warning("Could not detect framework, using UNKNOWN")
        return FrameworkType.UNKNOWN

    def _detect_js_framework(self, package_json_path: Path) -> FrameworkType:
        """
        Detect JavaScript framework from package.json.

        Args:
            package_json_path: Path to package.json

        Returns:
            Detected framework type
        """
        try:
            with open(package_json_path) as f:
                package_data = json.load(f)

            dependencies = {
                **package_data.get("dependencies", {}),
                **package_data.get("devDependencies", {}),
            }

            # Check for meta-frameworks first (they include base frameworks)
            if "next" in dependencies:
                logger.info("Detected Next.js framework")
                return FrameworkType.NEXT
            if "@remix-run/react" in dependencies:
                logger.info("Detected Remix framework")
                return FrameworkType.REMIX
            if "nuxt" in dependencies:
                logger.info("Detected Nuxt framework")
                return FrameworkType.NUXT
            if "@sveltejs/kit" in dependencies:
                logger.info("Detected SvelteKit framework")
                return FrameworkType.SVELTE_KIT
            if "astro" in dependencies:
                logger.info("Detected Astro framework")
                return FrameworkType.ASTRO

            # Check for base frameworks
            if "react" in dependencies:
                # Could be Electron with React
                if "electron" in dependencies:
                    logger.info("Detected Electron framework")
                    return FrameworkType.ELECTRON
                logger.info("Detected React framework")
                return FrameworkType.REACT
            if "vue" in dependencies:
                logger.info("Detected Vue framework")
                return FrameworkType.VUE
            if "svelte" in dependencies:
                logger.info("Detected Svelte framework")
                return FrameworkType.SVELTE
            if "@angular/core" in dependencies:
                logger.info("Detected Angular framework")
                return FrameworkType.ANGULAR
            if "solid-js" in dependencies:
                logger.info("Detected Solid framework")
                return FrameworkType.SOLID

            # Electron without React
            if "electron" in dependencies:
                logger.info("Detected Electron framework")
                return FrameworkType.ELECTRON

            logger.info("JavaScript project detected but no specific framework found")
            return FrameworkType.WEB

        except Exception as e:
            logger.warning(f"Error reading package.json: {e}")
            return FrameworkType.UNKNOWN

    def _get_static_analyzer(self, framework: FrameworkType) -> StaticAnalyzer | None:
        """
        Get appropriate static analyzer for framework.

        Args:
            framework: The framework to analyze

        Returns:
            Analyzer instance or None if not available
        """
        analyzer_class = self.static_analyzers.get(framework)
        if analyzer_class:
            return analyzer_class()

        # Try to find a generic analyzer that supports this framework
        for fw, analyzer_class in self.static_analyzers.items():
            analyzer = analyzer_class()
            if analyzer.supports_framework(framework):
                logger.info(f"Using {fw.value} analyzer for {framework.value}")
                return analyzer

        logger.warning(f"No static analyzer available for {framework.value}")
        return None

    def _get_runtime_extractor(self, target: ExtractionTarget) -> RuntimeExtractor | None:
        """
        Get appropriate runtime extractor for target.

        Args:
            target: The extraction target

        Returns:
            Extractor instance or None if not available
        """
        for extractor_class in self.runtime_extractors:
            extractor = extractor_class()
            if extractor.supports_target(target):
                logger.info(f"Using {extractor_class.__name__} for extraction")
                return extractor

        logger.warning("No runtime extractor available for target")
        return None

    def _get_matcher(self, framework: FrameworkType) -> StateMatcher | None:
        """
        Get appropriate matcher for framework.

        Args:
            framework: The framework to match

        Returns:
            Matcher instance or None if not available
        """
        matcher_class = self.matchers.get(framework)
        if matcher_class:
            return matcher_class()

        # Try to find a generic matcher that supports this framework
        for fw, matcher_class in self.matchers.items():
            matcher = matcher_class()
            if matcher.supports_framework(framework):
                logger.info(f"Using {fw.value} matcher for {framework.value}")
                return matcher

        logger.warning(f"No state matcher available for {framework.value}")
        return None

    async def _run_static_analysis(
        self, config: ExtractionConfig, framework: FrameworkType
    ) -> StaticAnalysisResult:
        """
        Run the static analysis phase.

        Args:
            config: Extraction configuration
            framework: Detected framework

        Returns:
            Static analysis results

        Raises:
            RuntimeError: If no analyzer is available
        """
        analyzer = self._get_static_analyzer(framework)
        if not analyzer:
            raise RuntimeError(
                f"No static analyzer available for {framework.value}. "
                "Static analysis cannot be performed."
            )

        if not config.target.project_path:
            raise RuntimeError("Project path is required for static analysis")

        start_time = time.time()
        try:
            # Pass the project path directly to analyzer.analyze()
            # StaticAnalyzer.analyze() expects a Path, not a StaticConfig
            analyzer_result = await analyzer.analyze(config.target.project_path)

            # Convert analyzer result to orchestrator's StaticAnalysisResult format
            # The analyzers return models.static.StaticAnalysisResult
            # The orchestrator expects models.base.StaticAnalysisResult
            result = StaticAnalysisResult(framework=framework)

            # Convert components
            components = getattr(analyzer_result, "components", [])
            result.components = [
                (
                    {
                        "id": c.id,
                        "name": c.name,
                        "file_path": str(c.file_path),
                        "line": c.line_number,
                    }
                    if hasattr(c, "id")
                    else c
                )
                for c in components
            ]

            # Convert routes
            routes = getattr(analyzer_result, "routes", [])
            result.routes = [
                ({"id": r.id, "path": r.path, "component": r.component} if hasattr(r, "id") else r)
                for r in routes
            ]

            # Convert state variables to state definitions
            state_vars = getattr(analyzer_result, "state_variables", [])
            result.state_definitions = [
                ({"id": s.id, "name": s.name, "type": s.var_type} if hasattr(s, "id") else s)
                for s in state_vars
            ]

            # Convert event handlers
            handlers = getattr(analyzer_result, "event_handlers", [])
            result.event_handlers = [
                (
                    {"id": h.id, "event": h.event_type, "handler": h.handler_name}
                    if hasattr(h, "id")
                    else h
                )
                for h in handlers
            ]

            # Convert navigation links to navigation flows
            # Navigation links come from Link elements in JSX
            if hasattr(analyzer, "get_navigation_links"):
                nav_links = analyzer.get_navigation_links()
                # Build route path to state mapping for resolving from/to
                route_map = {}
                for r in routes:
                    path = r.path if hasattr(r, "path") else r.get("path", "")
                    state_id = f"state_{path.replace('/', '_').strip('_')}" if path else ""
                    route_map[path] = state_id
                    # Also store normalized versions
                    normalized = path.rstrip("/") or "/"
                    route_map[normalized] = state_id

                for link in nav_links:
                    target = link.get("target", "")
                    source_file = link.get("file", "")
                    # Try to find source state from the file path
                    from_state = ""
                    for r in routes:
                        route_file = str(r.file_path) if hasattr(r, "file_path") else ""
                        if route_file and source_file and route_file in source_file:
                            from_path = r.path if hasattr(r, "path") else r.get("path", "")
                            from_state = route_map.get(from_path, "")
                            break

                    # Find target state from the href target
                    to_state = route_map.get(target, "")
                    # Try normalized version
                    if not to_state:
                        normalized_target = target.rstrip("/") or "/"
                        to_state = route_map.get(normalized_target, "")

                    if to_state:  # Only add if we can resolve the target
                        result.navigation_flows.append(
                            {
                                "from": from_state,
                                "to": to_state,
                                "target": target,
                                "type": link.get("type", "link"),
                                "source_file": source_file,
                                "line": link.get("line", 0),
                            }
                        )

            # Track file count
            result.analyzed_files = len(components) if components else 0

            result.analysis_duration_ms = (time.time() - start_time) * 1000

            # Count page components vs widgets if available
            page_count = 0
            widget_count = 0
            if hasattr(analyzer_result, "count_page_components") and hasattr(
                analyzer_result, "count_widget_components"
            ):
                page_count = analyzer_result.count_page_components()
                widget_count = analyzer_result.count_widget_components()
                logger.info(
                    f"Static analysis complete: {result.analyzed_files} files, "
                    f"{len(result.components)} components ({page_count} page-level states, {widget_count} UI widgets), "
                    f"{len(result.routes)} routes, {len(result.navigation_flows)} navigation links"
                )
            else:
                logger.info(
                    f"Static analysis complete: {result.analyzed_files} files, "
                    f"{len(result.components)} components, {len(result.routes)} routes, "
                    f"{len(result.navigation_flows)} navigation links"
                )
            return result
        except Exception as e:
            logger.error(f"Static analysis failed: {e}", exc_info=True)
            # Return empty result with error
            result = StaticAnalysisResult(framework=framework)
            result.errors.append(str(e))
            result.analysis_duration_ms = (time.time() - start_time) * 1000
            return result

    async def _run_runtime_extraction(self, config: ExtractionConfig) -> RuntimeExtractionResult:
        """
        Run the runtime extraction phase.

        Args:
            config: Extraction configuration

        Returns:
            Runtime extraction results

        Raises:
            RuntimeError: If no extractor is available
        """
        extractor = self._get_runtime_extractor(config.target)
        if not extractor:
            raise RuntimeError(
                "No runtime extractor available for target. "
                "Runtime extraction cannot be performed."
            )

        start_time = time.time()
        try:
            result = await extractor.extract(config.target, config)
            result.extraction_duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Runtime extraction complete: {result.pages_visited} pages, "
                f"{len(result.states)} states, {len(result.elements)} elements"
            )
            return result
        except Exception as e:
            logger.error(f"Runtime extraction failed: {e}", exc_info=True)
            # Return empty result with error
            result = RuntimeExtractionResult()
            result.errors.append(str(e))
            result.extraction_duration_ms = (time.time() - start_time) * 1000
            return result

    async def _correlate_results(self, config: ExtractionConfig, result: ExtractionResult) -> None:
        """
        Correlate static and runtime results (WHITE_BOX mode).

        Updates the result object with correlated states and transitions.

        Args:
            config: Extraction configuration
            result: Extraction result to update
        """
        if not result.static_analysis or not result.runtime_extraction:
            logger.error("Cannot correlate: missing static or runtime results")
            return

        matcher = self._get_matcher(result.framework)
        if not matcher:
            warning = f"No matcher available for {result.framework.value}, skipping correlation"
            logger.warning(warning)
            result.warnings.append(warning)

            # Fallback: use static results as preliminary states
            result.states = self._states_from_static(result.static_analysis)
            result.transitions = self._transitions_from_static(result.static_analysis)
            return

        try:
            # Match states
            correlated_states = await matcher.match(
                result.static_analysis,
                result.runtime_extraction,
                threshold=config.correlation_threshold,
            )

            result.states = correlated_states
            logger.info(f"Correlated {len(correlated_states)} states")

            # Calculate average correlation score
            if correlated_states:
                avg_score = sum(s.correlation_score for s in correlated_states) / len(
                    correlated_states
                )
                logger.info(f"Average correlation score: {avg_score:.2f}")

                # Warn if correlation is low
                if avg_score < config.correlation_threshold:
                    warning = (
                        f"Low correlation score ({avg_score:.2f} < {config.correlation_threshold}). "
                        "Results may be unreliable."
                    )
                    logger.warning(warning)
                    result.warnings.append(warning)

                    if config.require_correlation:
                        error = "Correlation threshold not met and require_correlation=True"
                        logger.error(error)
                        result.errors.append(error)

            # Extract transitions
            result.transitions = self._transitions_from_static(result.static_analysis)

        except Exception as e:
            error = f"Correlation failed: {e}"
            logger.error(error, exc_info=True)
            result.errors.append(error)

            # Fallback to static results
            result.states = self._states_from_static(result.static_analysis)
            result.transitions = self._transitions_from_static(result.static_analysis)

    def _states_from_static(self, static: StaticAnalysisResult) -> list[CorrelatedState]:
        """
        Convert static analysis to preliminary states.

        States represent navigable screens/views (routes/pages), NOT individual
        components. Only page-level components (category=STATE) are converted to states.
        Widget components (category=WIDGET) are UI elements within states.

        This method now also includes visibility-based sub-states (e.g., modal open/closed,
        sidebar expanded/collapsed) as distinct states in the state machine.

        Args:
            static: Static analysis results

        Returns:
            List of preliminary states (one per route or page-level component, plus visibility sub-states)
        """
        states: list[CorrelatedState] = []

        # Build a map of components by name for lookup
        component_map = {c.get("name"): c for c in static.components if c.get("name")}

        # Create states from routes - routes ARE states (navigable screens)
        for i, route in enumerate(static.routes):
            route_path = route.get("path", f"/route_{i}")
            route_component = route.get("component")

            # Get component details if available
            component_info = component_map.get(route_component, {}) if route_component else {}

            state = CorrelatedState(
                id=f"state_{i:04d}",
                name=route_path,  # Use route path as state name
                confidence=0.8,
                route_path=route_path,
                component_name=route_component,
                source_file=component_info.get("file_path") or route.get("file"),
                line_number=component_info.get("line"),
                state_variables=component_info.get("state_vars", []),
                correlation_method="static_only",
                metadata={
                    "route": route,
                    "component": component_info if component_info else None,
                },
            )
            states.append(state)

        # Add visibility-based sub-states as distinct states
        # These represent different UI configurations of the same page
        visibility_states = getattr(static, "visibility_states", [])
        for i, vis_state in enumerate(visibility_states):
            # Create a CorrelatedState from VisibilityState
            state = CorrelatedState(
                id=f"vis_state_{i:04d}",
                name=vis_state.name,
                confidence=0.7,  # Slightly lower confidence for inferred sub-states
                route_path=vis_state.parent_route,
                component_name=vis_state.parent_component,
                source_file=str(vis_state.file_path) if vis_state.file_path else None,
                line_number=vis_state.line_number,
                state_variables=(
                    [vis_state.controlling_variable] if vis_state.controlling_variable else []
                ),
                correlation_method="visibility_state",
                metadata={
                    "visibility_state": True,
                    "parent_component": vis_state.parent_component,
                    "controlling_variable": vis_state.controlling_variable,
                    "variable_value": vis_state.variable_value,
                    "rendered_components": vis_state.rendered_components,
                    "hidden_components": vis_state.hidden_components,
                    "toggle_handlers": vis_state.toggle_handlers,
                    "conditional_render_id": vis_state.conditional_render_id,
                },
            )
            states.append(state)

        # Note: Widget components (category=WIDGET) are NOT converted to states.
        # They are UI elements within states. Only page-level components (category=STATE)
        # that don't have routes should be considered as additional states.

        logger.info(
            f"Created {len(states)} states: {len(static.routes)} route-based states + "
            f"{len(visibility_states)} visibility sub-states"
        )
        return states

    def _states_from_runtime(self, runtime: RuntimeExtractionResult) -> list[CorrelatedState]:
        """
        Convert runtime extraction results to CorrelatedState objects.

        This is used for BLACK_BOX mode where we only have runtime extraction
        (no static analysis to correlate with).

        Args:
            runtime: Runtime extraction results

        Returns:
            List of CorrelatedState objects derived from runtime states
        """
        states: list[CorrelatedState] = []

        logger.info(
            f"Converting runtime results: {len(runtime.states)} states, "
            f"{len(runtime.elements)} elements"
        )

        # Convert each ExtractedState from runtime to CorrelatedState
        for i, extracted_state in enumerate(runtime.states):
            # ExtractedState has: id, name, bbox, state_type, element_ids, screenshot_id,
            # detection_method, confidence, semantic_role, aria_label, source_url, metadata

            # Get state attributes safely
            state_id = getattr(extracted_state, "id", f"runtime_state_{i:04d}")
            state_name = getattr(extracted_state, "name", f"State {i}")
            state_confidence = getattr(extracted_state, "confidence", 0.7)
            source_url = getattr(extracted_state, "source_url", None)
            state_type = getattr(extracted_state, "state_type", None)
            state_type_value = state_type.value if state_type else "unknown"
            element_ids = getattr(extracted_state, "element_ids", [])
            screenshot_id = getattr(extracted_state, "screenshot_id", None)
            detection_method = getattr(extracted_state, "detection_method", "runtime")
            metadata = getattr(extracted_state, "metadata", {}) or {}

            # Get bbox for state visualization
            bbox = getattr(extracted_state, "bbox", None)
            bbox_dict = None
            if bbox:
                # BoundingBox may have a to_dict() method or be a dataclass
                if hasattr(bbox, "to_dict"):
                    bbox_dict = bbox.to_dict()
                elif hasattr(bbox, "x"):
                    bbox_dict = {
                        "x": bbox.x,
                        "y": bbox.y,
                        "width": bbox.width,
                        "height": bbox.height,
                    }

            # Build visible elements list from element_ids
            visible_elements = element_ids if isinstance(element_ids, list) else []

            # Build metadata with bbox included for visualization
            state_metadata = {
                "state_type": state_type_value,
                "detection_method": detection_method,
                "element_count": len(visible_elements),
                **metadata,
            }
            if bbox_dict:
                state_metadata["bbox"] = bbox_dict

            state = CorrelatedState(
                id=state_id,
                name=state_name,
                confidence=state_confidence,
                route_path=None,  # No route info in BLACK_BOX mode
                component_name=None,  # No component info in BLACK_BOX mode
                source_file=None,
                line_number=None,
                state_variables=[],
                runtime_state_id=state_id,
                screenshot_id=screenshot_id,
                url=source_url,
                visible_elements=visible_elements,
                correlation_method=detection_method,
                correlation_score=state_confidence,
                metadata=state_metadata,
            )
            states.append(state)

        logger.info(f"Converted {len(states)} runtime states to CorrelatedState")
        return states

    def _transitions_from_runtime(
        self, runtime: RuntimeExtractionResult
    ) -> list[InferredTransition]:
        """
        Convert runtime extraction transitions to InferredTransition objects.

        This is used for BLACK_BOX mode where transitions are discovered
        during page crawling.

        Args:
            runtime: Runtime extraction results

        Returns:
            List of InferredTransition objects derived from runtime transitions
        """
        transitions: list[InferredTransition] = []

        logger.info(f"Converting runtime transitions: {len(runtime.transitions)} transitions")

        for i, extracted_trans in enumerate(runtime.transitions):
            # Handle both InferredTransition objects and dict-like objects
            if isinstance(extracted_trans, InferredTransition):
                # Already the right type, just add it
                transitions.append(extracted_trans)
            else:
                # Extract transition attributes
                trans_id = getattr(extracted_trans, "id", f"runtime_trans_{i:04d}")
                from_state_id = getattr(extracted_trans, "from_state_id", "")
                to_state_id = getattr(extracted_trans, "to_state_id", "")
                trigger_type = getattr(extracted_trans, "trigger_type", "click")
                target_element = getattr(extracted_trans, "target_element", None)
                confidence = getattr(extracted_trans, "confidence", 0.8)
                metadata = getattr(extracted_trans, "metadata", {}) or {}

                transition = InferredTransition(
                    id=trans_id,
                    from_state_id=from_state_id,
                    to_state_id=to_state_id,
                    trigger_type=trigger_type,
                    target_element=target_element,
                    confidence=confidence,
                    metadata=metadata,
                )
                transitions.append(transition)

        logger.info(f"Converted {len(transitions)} runtime transitions to InferredTransition")
        return transitions

    def _transitions_from_static(self, static: StaticAnalysisResult) -> list[InferredTransition]:
        """
        Extract transitions from static analysis.

        This now includes:
        1. Navigation transitions (between routes)
        2. Visibility state transitions (e.g., opening/closing modals, toggling sidebars)

        Args:
            static: Static analysis results

        Returns:
            List of inferred transitions
        """
        transitions: list[InferredTransition] = []

        # Create transitions from navigation flows
        for i, flow in enumerate(static.navigation_flows):
            transition = InferredTransition(
                id=f"trans_{i:04d}",
                from_state_id=flow.get("from", ""),
                to_state_id=flow.get("to", ""),
                trigger_type=flow.get("trigger", "navigation"),
                event_handler=flow.get("handler"),
                source_location=flow.get("location"),
                confidence=0.6,  # Lower confidence for inferred transitions
                metadata=flow,
            )
            transitions.append(transition)

        # Create transitions from event handlers
        for i, handler in enumerate(static.event_handlers):
            if handler.get("navigates_to"):
                transition = InferredTransition(
                    id=f"handler_trans_{i:04d}",
                    from_state_id=handler.get("component", ""),
                    to_state_id=handler.get("navigates_to", ""),
                    trigger_type=handler.get("event_type", "click"),
                    event_handler=handler.get("name"),
                    source_location=f"{handler.get('file')}:{handler.get('line')}",
                    confidence=0.7,
                    metadata=handler,
                )
                transitions.append(transition)

        # Create transitions between visibility states
        # Each visibility state pair (e.g., modal_closed <-> modal_open) has bidirectional transitions
        visibility_states = getattr(static, "visibility_states", [])

        # Group visibility states by parent component and controlling variable
        state_groups: dict[tuple[Any, Any], list[Any]] = {}
        for vis_state in visibility_states:
            key = (vis_state.parent_component, vis_state.controlling_variable)
            if key not in state_groups:
                state_groups[key] = []
            state_groups[key].append(vis_state)

        # Create transitions between states in each group
        trans_idx = 0
        for (_parent_comp, control_var), states_in_group in state_groups.items():
            if len(states_in_group) < 2:
                continue  # Need at least 2 states to have a transition

            # For each state, create transitions to other states via toggle handlers
            for state in states_in_group:
                for other_state in states_in_group:
                    if state.id == other_state.id:
                        continue

                    # Create transition for each toggle handler
                    for handler_id in state.toggle_handlers:
                        transition = InferredTransition(
                            id=f"vis_trans_{trans_idx:04d}",
                            from_state_id=f"vis_state_{visibility_states.index(state):04d}",
                            to_state_id=f"vis_state_{visibility_states.index(other_state):04d}",
                            trigger_type="toggle",
                            event_handler=handler_id,
                            source_location=(
                                f"{state.file_path}:{state.line_number}" if state.file_path else ""
                            ),
                            confidence=0.75,
                            metadata={
                                "transition_type": "visibility_toggle",
                                "controlling_variable": control_var,
                                "from_value": state.variable_value,
                                "to_value": other_state.variable_value,
                            },
                        )
                        transitions.append(transition)
                        trans_idx += 1

        nav_count = sum(
            1 for t in transitions if t.metadata.get("transition_type") != "visibility_toggle"
        )
        vis_count = sum(
            1 for t in transitions if t.metadata.get("transition_type") == "visibility_toggle"
        )

        logger.info(
            f"Created {len(transitions)} transitions: {nav_count} navigation transitions + "
            f"{vis_count} visibility state transitions"
        )
        return transitions

    async def _run_hybrid_extraction(
        self, config: ExtractionConfig, result: ExtractionResult
    ) -> None:
        """
        Run hybrid extraction using tech stack-specific extractors.

        Hybrid extraction combines static code analysis with runtime screenshot
        capture to produce States, StateImages with precise bounding boxes, and
        Transitions.

        Args:
            config: Extraction configuration
            result: Extraction result to update
        """
        if not config.target.project_path:
            raise RuntimeError("HYBRID mode requires project_path")

        try:
            from .hybrid import HybridExtractionConfig, HybridExtractor

            # Create hybrid extraction config
            hybrid_config = HybridExtractionConfig(
                project_path=config.target.project_path,
                viewport=(1920, 1080),
                headless=True,
                timeout_seconds=config.timeout_seconds,
            )

            # Run hybrid extraction
            hybrid_extractor = HybridExtractor()
            hybrid_result = await hybrid_extractor.extract(
                config.target.project_path, hybrid_config
            )

            # Convert hybrid States to CorrelatedState
            for state in hybrid_result.states:
                correlated = CorrelatedState(
                    id=state.id,
                    name=state.name,
                    confidence=state.confidence,
                    route_path=state.route_path,
                    component_name=state.component_name,
                    source_file=str(state.source_file) if state.source_file else None,
                    line_number=state.source_line,
                    state_variables=(
                        [state.controlling_variable] if state.controlling_variable else []
                    ),
                    correlation_method="hybrid",
                    metadata={
                        "state_type": state.state_type.value,
                        "url": state.url,
                        "viewport": state.viewport,
                        "screenshot_path": (
                            str(state.screenshot_path) if state.screenshot_path else None
                        ),
                        "state_images": [img.id for img in state.state_images],
                    },
                )
                result.states.append(correlated)

            # Convert hybrid Transitions to InferredTransition
            for trans in hybrid_result.transitions:
                inferred = InferredTransition(
                    id=trans.id,
                    from_state_id=trans.from_state_id,
                    to_state_id=trans.to_state_id,
                    trigger_type=trans.trigger.value,
                    event_handler=trans.event_handler_name,
                    source_location=(
                        f"{trans.source_file}:{trans.source_line}" if trans.source_file else None
                    ),
                    confidence=trans.confidence,
                    metadata={
                        "trigger_image_id": trans.trigger_image_id,
                        "trigger_selector": trans.trigger_selector,
                        "navigation_path": trans.navigation_path,
                    },
                )
                result.transitions.append(inferred)

            # Store hybrid result in metadata for access to StateImages
            result.metadata = result.metadata or {}
            result.metadata["hybrid_result"] = hybrid_result.to_dict()
            result.metadata["tech_stack"] = hybrid_result.tech_stack
            result.metadata["screenshots_dir"] = (
                str(hybrid_result.screenshots_dir) if hybrid_result.screenshots_dir else None
            )

            # Copy errors and warnings
            result.errors.extend(hybrid_result.errors)
            result.warnings.extend(hybrid_result.warnings)

            logger.info(
                f"Hybrid extraction complete: {len(result.states)} states, "
                f"{len(result.transitions)} transitions, "
                f"{len(hybrid_result.state_images)} state images"
            )

        except ImportError as e:
            error = f"Hybrid extraction not available: {e}"
            logger.error(error)
            result.errors.append(error)
        except Exception as e:
            error = f"Hybrid extraction failed: {e}"
            logger.error(error, exc_info=True)
            result.errors.append(error)

    def _register_defaults(self) -> None:
        """
        Register built-in implementations.

        This will register default analyzers, extractors, and matchers
        as they are implemented.
        """
        # Register static analyzers
        try:
            from .static.react.analyzer import ReactStaticAnalyzer

            self.register_static_analyzer(FrameworkType.REACT, ReactStaticAnalyzer)  # type: ignore[arg-type]
        except ImportError:
            logger.warning("ReactStaticAnalyzer not available")

        try:
            from .static.nextjs.analyzer import NextJSStaticAnalyzer

            self.register_static_analyzer(ConfigFrameworkType.NEXT_JS, NextJSStaticAnalyzer)  # type: ignore[arg-type]
        except ImportError:
            logger.warning("NextJSStaticAnalyzer not available")

        # Register runtime extractors
        try:
            from .runtime.playwright.extractor import PlaywrightExtractor

            self.register_runtime_extractor(PlaywrightExtractor)  # type: ignore[arg-type]
        except ImportError:
            logger.warning("PlaywrightExtractor not available")

        try:
            from .runtime.tauri.extractor import TauriExtractor

            self.register_runtime_extractor(TauriExtractor)  # type: ignore[arg-type]
        except ImportError:
            logger.warning("TauriExtractor not available")

        try:
            from .runtime.awas.extractor import AwasRuntimeExtractor

            self.register_runtime_extractor(AwasRuntimeExtractor)  # type: ignore[arg-type]
        except ImportError:
            logger.warning("AwasRuntimeExtractor not available")

        # Register UI-TARS explorer for desktop/native app exploration
        try:
            from .runtime.uitars import HAS_UITARS

            if HAS_UITARS:
                from .runtime.uitars.explorer import UITARSExplorer

                self.register_runtime_extractor(UITARSExplorer)  # type: ignore[arg-type]
                logger.info("UITARSExplorer registered for native/desktop exploration")
            else:
                logger.info("UI-TARS dependencies not available, UITARSExplorer not registered")
        except ImportError:
            logger.warning("UITARSExplorer not available")

        # Register state matchers
        try:
            from .matching.matcher import DefaultStateMatcher

            # DefaultStateMatcher works for all frameworks
            for framework in [FrameworkType.REACT, ConfigFrameworkType.NEXT_JS]:
                self.register_matcher(framework, DefaultStateMatcher)  # type: ignore[arg-type]
        except ImportError:
            logger.warning("DefaultStateMatcher not available")
