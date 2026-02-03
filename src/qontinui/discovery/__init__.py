"""State Discovery system for automated state and StateImage detection.

This module provides comprehensive functionality for discovering, detecting, and
constructing application states through automated analysis of screenshots and
UI elements.

Quick Start:
    The simplest way to use state discovery is through the facade:

    >>> from qontinui.discovery import StateDiscoveryFacade, discover_states
    >>>
    >>> # Quick one-liner
    >>> result = discover_states(screenshots)
    >>>
    >>> # Or with more control
    >>> facade = StateDiscoveryFacade()
    >>> result = facade.discover_states(screenshots)
    >>> print(f"Found {len(result.states)} states")

Submodules:
    - element_detection: UI element identification and classification
    - region_analysis: Screen region segmentation and analysis
    - state_detection: State identification and matching
    - state_construction: State object creation from detected elements
    - experimental: Research and experimental detection features
    - pixel_analysis: Low-level pixel stability analysis
    - background_removal: Remove dynamic backgrounds for robust comparison

The discovery system forms the foundation for automated testing and exploration
by enabling the system to understand and navigate application states without
manual state definition.

Advanced Example:
    >>> from qontinui.discovery.state_detection import DifferentialConsistencyDetector
    >>> from qontinui.discovery.state_construction import StateBuilder
    >>>
    >>> # Detect consistent regions across screenshots
    >>> detector = DifferentialConsistencyDetector()
    >>> result = detector.analyze_screenshots(screenshots)
    >>>
    >>> # Build state objects from detected elements
    >>> builder = StateBuilder()
    >>> state = builder.build_state(elements, screenshot)
"""

# Background removal
from .background_removal import (
    BackgroundRemovalAnalyzer,
    BackgroundRemovalConfig,
    create_default_config,
    decode_base64_image,
    encode_image_to_base64,
    remove_backgrounds_from_base64,
    remove_backgrounds_simple,
)

# Base classes
from .base_detector import BaseDetector

# Click analysis - import key classes for convenience
from .click_analysis import (
    ClickBoundingBoxInferrer,
    ClickContextAnalyzer,
    DetectionStrategy,
    ElementBoundaryFinder,
    ElementType,
    InferenceConfig,
    InferenceResult,
    InferredBoundingBox,
    infer_bbox_from_click,
)

# Discovery facade - primary entry point
from .discovery_facade import (
    DiscoveryAlgorithm,
    DiscoveryConfig,
    DiscoveryResult,
    StateDiscoveryFacade,
    discover_states,
)
from .models import AnalysisResult, DiscoveredState, StateImage
from .multi_screenshot_detector import MultiScreenshotDetector

# Pixel analysis
from .pixel_analysis.analyzers import PixelStabilityAnalyzer
from .pixel_analysis.extractor import StableRegionExtractor
from .pixel_stability_matrix_analyzer import PixelStabilityMatrixAnalyzer

# State construction - import key classes for convenience
from .state_construction import (
    ElementIdentifier,
    OCRNameGenerator,
    StateBuilder,
    TransitionInfo,
)

# State detection - import key classes for convenience
from .state_detection import (
    DifferentialConsistencyDetector,
    StateDetector,
    TransitionDetector,
)

# Unified state discovery service
from .state_discovery import (
    DiscoveredElement,
    DiscoveredTransition,
    DiscoveryStrategyType,
    FingerprintStrategy,
    LegacyStrategy,
    StateDiscoveryInput,
    StateDiscoveryResult,
    StateDiscoveryService,
    StateDiscoveryStrategy,
)
from .state_discovery import (
    DiscoveredState as UnifiedDiscoveredState,
)
from .state_discovery import (
    discover_states as discover_states_unified,
)

# UI Bridge exploration - automatic application exploration
from .target_connection import (
    ActionResult,
    ActionType,
    BoundingBox,
    DesktopTargetConnection,
    DOMSnapshot,
    Element,
    ElementRole,
    ExplorationConfig,
    MobileTargetConnection,
    TargetConnection,
    WebTargetConnection,
    create_connection,
)

# UI Bridge adapter - state discovery from semantic render data (legacy interface)
from .ui_bridge_adapter import (
    UIBridgeElement,
    UIBridgeRender,
    UIBridgeStateDiscoveryResult,
    discover_states_from_renders,
    extract_elements_from_render,
    get_active_states_for_render,
    get_elements_by_render,
    get_state_elements,
)
from .ui_bridge_explorer import (
    ElementPrioritizer,
    ExplorationResult,
    ExplorationStep,
    SafetyFilter,
    UIBridgeExplorer,
    explore_application,
)

# Visual context for AI consumption
from .visual_context import (
    AnnotatedSnapshot,
    ElementColorScheme,
    InteractionHeatmap,
    VisualContextGenerator,
    VisualDiff,
)

# Submodules available for import
# from qontinui.discovery import element_detection
# from qontinui.discovery import region_analysis
# from qontinui.discovery import state_detection
# from qontinui.discovery import state_construction
# from qontinui.discovery import experimental
# from qontinui.discovery import click_analysis

__all__ = [
    # Unified state discovery service - NEW primary entry point
    "StateDiscoveryService",
    "StateDiscoveryInput",
    "StateDiscoveryResult",
    "StateDiscoveryStrategy",
    "DiscoveryStrategyType",
    "LegacyStrategy",
    "FingerprintStrategy",
    "DiscoveredElement",
    "UnifiedDiscoveredState",
    "DiscoveredTransition",
    "discover_states_unified",
    # Discovery facade - legacy entry point for pixel analysis
    "StateDiscoveryFacade",
    "DiscoveryConfig",
    "DiscoveryResult",
    "DiscoveryAlgorithm",
    "discover_states",
    # Background removal
    "BackgroundRemovalAnalyzer",
    "BackgroundRemovalConfig",
    "create_default_config",
    "remove_backgrounds_simple",
    "remove_backgrounds_from_base64",
    "decode_base64_image",
    "encode_image_to_base64",
    # Base classes
    "BaseDetector",
    "MultiScreenshotDetector",
    # Pixel analysis
    "PixelStabilityAnalyzer",
    "PixelStabilityMatrixAnalyzer",
    "StableRegionExtractor",
    # Models
    "StateImage",
    "DiscoveredState",
    "AnalysisResult",
    # State detection (convenience exports)
    "DifferentialConsistencyDetector",
    "StateDetector",
    "TransitionDetector",
    # State construction (convenience exports)
    "StateBuilder",
    "ElementIdentifier",
    "OCRNameGenerator",
    "TransitionInfo",
    # Click analysis (convenience exports)
    "ClickBoundingBoxInferrer",
    "ElementBoundaryFinder",
    "ClickContextAnalyzer",
    "InferredBoundingBox",
    "InferenceConfig",
    "InferenceResult",
    "ElementType",
    "DetectionStrategy",
    "infer_bbox_from_click",
    # Submodules
    "element_detection",
    "region_analysis",
    "state_detection",
    "state_construction",
    "experimental",
    "click_analysis",
    "background_removal",
    # UI Bridge adapter
    "UIBridgeElement",
    "UIBridgeRender",
    "UIBridgeStateDiscoveryResult",
    "discover_states_from_renders",
    "extract_elements_from_render",
    "get_state_elements",
    "get_elements_by_render",
    "get_active_states_for_render",
    # UI Bridge exploration
    "UIBridgeExplorer",
    "ExplorationConfig",
    "ExplorationResult",
    "ExplorationStep",
    "SafetyFilter",
    "ElementPrioritizer",
    "explore_application",
    # Target connections
    "TargetConnection",
    "WebTargetConnection",
    "DesktopTargetConnection",
    "MobileTargetConnection",
    "create_connection",
    # Target connection models
    "Element",
    "ElementRole",
    "ActionResult",
    "ActionType",
    "DOMSnapshot",
    "BoundingBox",
    # Visual context for AI
    "VisualContextGenerator",
    "AnnotatedSnapshot",
    "VisualDiff",
    "InteractionHeatmap",
    "ElementColorScheme",
]
