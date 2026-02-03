"""State Machine Integration for UI Bridge.

This package provides state machine integration between UI Bridge and multistate,
enabling model-based GUI automation with states, transitions, and pathfinding.

Key components:
- UIBridgeRuntime: Implements multistate StateSpaceRuntime for UI Bridge
- StateDiscovery: Discovers states from UI Bridge element co-occurrence
- FingerprintStateDiscovery: Enhanced discovery using element fingerprints
- TransitionDetector: Detects transitions from action/state-change relationships
- StatePersistence: Stores states and transitions in runner database

Example:
    from ui_bridge import UIBridgeClient
    from qontinui.state_machine import (
        UIBridgeRuntime,
        UIBridgeState,
        UIBridgeTransition,
        StatePersistence,
    )

    # Create runtime with UI Bridge client
    client = UIBridgeClient("http://localhost:9876")
    runtime = UIBridgeRuntime(client)

    # Register states and transitions
    runtime.register_state(UIBridgeState(
        id="dashboard",
        name="Dashboard",
        element_ids=["nav-menu", "dashboard-content"],
    ))

    runtime.register_transition(UIBridgeTransition(
        id="open_settings",
        name="Open Settings",
        from_states=["dashboard"],
        activate_states=["settings_panel"],
        exit_states=[],
        actions=[{"type": "click", "elementId": "settings-btn"}],
    ))

    # Navigate using pathfinding
    result = runtime.navigate_to(["settings_panel"])

    # Persist for later use
    persistence = StatePersistence("path/to/db.sqlite")
    persistence.save_states(list(runtime._ui_states.values()))

Fingerprint-enhanced discovery example:
    from qontinui.state_machine import (
        FingerprintStateDiscovery,
        FingerprintStateDiscoveryConfig,
    )

    # Configure fingerprint-aware discovery
    config = FingerprintStateDiscoveryConfig(
        treat_header_footer_as_global=True,
        dedupe_repeating_elements=True,
        use_size_weighting=True,
    )

    discovery = FingerprintStateDiscovery(config)

    # Load co-occurrence export from UI Bridge
    discovery.load_cooccurrence_export(cooccurrence_data)

    # Get enhanced states
    states = discovery.get_discovered_states()
"""

from .fingerprint_state_discovery import (
    DiscoveredFingerprintState,
    FingerprintStateDiscovery,
    FingerprintStateDiscoveryConfig,
)
from .fingerprint_types import (
    ARIA_LANDMARKS,
    BLOCKING_POSITION_ZONES,
    GLOBAL_POSITION_ZONES,
    POSITION_ZONES,
    SIZE_CATEGORIES,
    CaptureRecord,
    CooccurrenceExport,
    ElementFingerprint,
    FingerprintStats,
    PresenceMatrixEntry,
    RepeatPattern,
    StateCandidate,
    TransitionRecord,
)
from .persistence import (
    StateGroupRecord,
    StatePersistence,
)
from .state_discovery import (
    DiscoveredUIState,
    StateDiscoveryConfig,
    StateGraph,
    StateGraphEdge,
    StateGraphFormat,
    StateGraphNode,
    UIBridgeStateDiscovery,
)
from .transition_detector import (
    DetectedTransition,
    TransitionDetector,
    TransitionReliability,
)
from .ui_bridge_runtime import (
    TransitionExecutionResult,
    UIBridgeRuntime,
    UIBridgeRuntimeConfig,
    UIBridgeState,
    UIBridgeTransition,
    generate_state_id,
)

__all__ = [
    # Runtime
    "UIBridgeRuntime",
    "UIBridgeRuntimeConfig",
    "UIBridgeState",
    "UIBridgeTransition",
    "TransitionExecutionResult",
    "generate_state_id",
    # ID-based Discovery (legacy)
    "UIBridgeStateDiscovery",
    "DiscoveredUIState",
    "StateDiscoveryConfig",
    # Fingerprint-enhanced Discovery
    "FingerprintStateDiscovery",
    "FingerprintStateDiscoveryConfig",
    "DiscoveredFingerprintState",
    # Fingerprint Types
    "ElementFingerprint",
    "RepeatPattern",
    "CaptureRecord",
    "TransitionRecord",
    "CooccurrenceExport",
    "FingerprintStats",
    "PresenceMatrixEntry",
    "StateCandidate",
    # Fingerprint Constants
    "POSITION_ZONES",
    "GLOBAL_POSITION_ZONES",
    "BLOCKING_POSITION_ZONES",
    "SIZE_CATEGORIES",
    "ARIA_LANDMARKS",
    # State Graph
    "StateGraph",
    "StateGraphNode",
    "StateGraphEdge",
    "StateGraphFormat",
    # Transition Detection
    "TransitionDetector",
    "DetectedTransition",
    "TransitionReliability",
    # Persistence
    "StatePersistence",
    "StateGroupRecord",
]
