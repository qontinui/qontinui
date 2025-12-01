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

    def __init__(self):
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
        logger.info(
            f"Registering static analyzer for {framework.value}: {analyzer_class.__name__}"
        )
        self.static_analyzers[framework] = analyzer_class

    def register_runtime_extractor(
        self, extractor_class: type[RuntimeExtractor]
    ) -> None:
        """
        Register a runtime extractor.

        Args:
            extractor_class: The extractor class (not an instance)
        """
        logger.info(f"Registering runtime extractor: {extractor_class.__name__}")
        self.runtime_extractors.append(extractor_class)

    def register_matcher(
        self, framework: FrameworkType, matcher_class: type[StateMatcher]
    ) -> None:
        """
        Register a state matcher for a framework.

        Args:
            framework: The framework this matcher supports
            matcher_class: The matcher class (not an instance)
        """
        logger.info(
            f"Registering state matcher for {framework.value}: {matcher_class.__name__}"
        )
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

            # Phase 1: Static Analysis (STATIC_ONLY or WHITE_BOX)
            if config.mode in (ExtractionMode.STATIC_ONLY, ExtractionMode.WHITE_BOX):
                logger.info("Starting static analysis phase...")
                result.static_analysis = await self._run_static_analysis(
                    config, framework
                )

                if result.static_analysis.errors:
                    logger.warning(
                        f"Static analysis completed with {len(result.static_analysis.errors)} errors"
                    )
                    result.warnings.extend(result.static_analysis.warnings)
                    result.errors.extend(result.static_analysis.errors)

            # Phase 2: Runtime Extraction (BLACK_BOX or WHITE_BOX)
            if config.mode in (ExtractionMode.BLACK_BOX, ExtractionMode.WHITE_BOX):
                logger.info("Starting runtime extraction phase...")
                result.runtime_extraction = await self._run_runtime_extraction(config)

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
                    error_msg = (
                        "WHITE_BOX mode requires both static and runtime results"
                    )
                    logger.error(error_msg)
                    result.errors.append(error_msg)

            # Handle STATIC_ONLY mode - convert static results to states
            elif config.mode == ExtractionMode.STATIC_ONLY and result.static_analysis:
                logger.info("Converting static analysis to preliminary states...")
                result.states = self._states_from_static(result.static_analysis)
                result.transitions = self._transitions_from_static(
                    result.static_analysis
                )

            # Handle BLACK_BOX mode - use runtime results directly
            elif config.mode == ExtractionMode.BLACK_BOX and result.runtime_extraction:
                logger.info("Using runtime extraction results directly...")
                # Runtime states are already in result.runtime_extraction
                # Could convert them to CorrelatedState format if needed
                pass

            # Mark completion
            import datetime

            result.completed_at = datetime.datetime.now()

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
                raise ConfigError(
                    f"{config.mode.value} mode requires target.project_path"
                )

        if config.mode in (ExtractionMode.BLACK_BOX, ExtractionMode.WHITE_BOX):
            if not (
                config.target.url
                or config.target.executable_path
                or config.target.app_id
            ):
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
            logger.info(
                f"Using explicitly specified framework: {target.framework.value}"
            )
            return target.framework

        # Otherwise, auto-detect from project files
        if not target.project_path:
            # No project path - can't detect, default to generic
            if target.url:
                logger.info("No project path, assuming generic web application")
                return FrameworkType.WEB
            logger.warning(
                "Cannot detect framework without project_path or framework hint"
            )
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

    def _get_runtime_extractor(
        self, target: ExtractionTarget
    ) -> RuntimeExtractor | None:
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

        start_time = time.time()
        try:
            result = await analyzer.analyze(config.target.project_path)
            result.analysis_duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Static analysis complete: {result.analyzed_files} files, "
                f"{len(result.components)} components, {len(result.routes)} routes"
            )
            return result
        except Exception as e:
            logger.error(f"Static analysis failed: {e}", exc_info=True)
            # Return empty result with error
            result = StaticAnalysisResult(framework=framework)
            result.errors.append(str(e))
            result.analysis_duration_ms = (time.time() - start_time) * 1000
            return result

    async def _run_runtime_extraction(
        self, config: ExtractionConfig
    ) -> RuntimeExtractionResult:
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

    async def _correlate_results(
        self, config: ExtractionConfig, result: ExtractionResult
    ) -> None:
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
                        error = (
                            "Correlation threshold not met and require_correlation=True"
                        )
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

    def _states_from_static(
        self, static: StaticAnalysisResult
    ) -> list[CorrelatedState]:
        """
        Convert static analysis to preliminary states.

        Args:
            static: Static analysis results

        Returns:
            List of preliminary states
        """
        states: list[CorrelatedState] = []

        # Create states from components
        for i, component in enumerate(static.components):
            state = CorrelatedState(
                id=f"state_{i:04d}",
                name=component.get("name", f"Component {i}"),
                confidence=0.7,  # Lower confidence for static-only
                component_name=component.get("name"),
                source_file=component.get("file"),
                line_number=component.get("line"),
                state_variables=component.get("state_vars", []),
                correlation_method="static_only",
                metadata=component,
            )
            states.append(state)

        # Create states from routes
        for i, route in enumerate(static.routes):
            state = CorrelatedState(
                id=f"route_state_{i:04d}",
                name=route.get("path", f"Route {i}"),
                confidence=0.8,  # Higher confidence for routes
                route_path=route.get("path"),
                component_name=route.get("component"),
                source_file=route.get("file"),
                correlation_method="static_only",
                metadata=route,
            )
            states.append(state)

        logger.info(f"Created {len(states)} preliminary states from static analysis")
        return states

    def _transitions_from_static(
        self, static: StaticAnalysisResult
    ) -> list[InferredTransition]:
        """
        Extract transitions from static analysis.

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

        logger.info(f"Created {len(transitions)} transitions from static analysis")
        return transitions

    def _register_defaults(self) -> None:
        """
        Register built-in implementations.

        This will register default analyzers, extractors, and matchers
        as they are implemented.
        """
        # Register static analyzers
        try:
            from .static.react.analyzer import ReactStaticAnalyzer

            self.register_static_analyzer(FrameworkType.REACT, ReactStaticAnalyzer)
        except ImportError:
            logger.warning("ReactStaticAnalyzer not available")

        try:
            from .static.nextjs.analyzer import NextJSStaticAnalyzer

            self.register_static_analyzer(FrameworkType.NEXTJS, NextJSStaticAnalyzer)
        except ImportError:
            logger.warning("NextJSStaticAnalyzer not available")

        # Register runtime extractors
        try:
            from .runtime.playwright.extractor import PlaywrightExtractor

            self.register_runtime_extractor(PlaywrightExtractor)
        except ImportError:
            logger.warning("PlaywrightExtractor not available")

        try:
            from .runtime.tauri.extractor import TauriExtractor

            self.register_runtime_extractor(TauriExtractor)
        except ImportError:
            logger.warning("TauriExtractor not available")

        # Register state matchers
        try:
            from .matching.matcher import DefaultStateMatcher

            # DefaultStateMatcher works for all frameworks
            for framework in [FrameworkType.REACT, FrameworkType.NEXTJS]:
                self.register_matcher(framework, DefaultStateMatcher)
        except ImportError:
            logger.warning("DefaultStateMatcher not available")
