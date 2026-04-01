"""Unified State Discovery Package.

This package provides a unified interface for state discovery with
automatic strategy selection based on available input data.

Quick Start:
    from qontinui.discovery.state_discovery import discover_states

    # From render logs (uses fingerprint strategy with ID fallback)
    result = discover_states(renders=my_renders)

    # From co-occurrence export with fingerprints (enhanced approach)
    result = discover_states(cooccurrence_export=my_export)

    # Auto-detect: uses fingerprints if available, falls back to ID synthesis
    result = discover_states(
        renders=my_renders,
        cooccurrence_export=my_export,
    )

Using the Service:
    from qontinui.discovery.state_discovery import (
        StateDiscoveryService,
        DiscoveryStrategyType,
    )

    service = StateDiscoveryService()

    # Fingerprint-enhanced discovery
    result = service.discover_from_export(export_data)

    # Render-based discovery (ID fallback)
    result = service.discover_from_renders(renders)

Available Strategies:
    - FINGERPRINT: Enhanced discovery with element fingerprints (supports ID fallback)
    - AUTO: Automatic selection based on available data
"""

from .base import (
                   DiscoveredElement,
                   DiscoveredState,
                   DiscoveredTransition,
                   DiscoveryStrategyType,
                   StateDiscoveryInput,
                   StateDiscoveryResult,
                   StateDiscoveryStrategy,
)
from .service import StateDiscoveryService, discover_states
from .strategies import FingerprintStrategy

__all__ = [
    # Main service
    "StateDiscoveryService",
    "discover_states",
    # Strategy types
    "DiscoveryStrategyType",
    "StateDiscoveryStrategy",
    "FingerprintStrategy",
    # Input/Output types
    "StateDiscoveryInput",
    "StateDiscoveryResult",
    # Data types
    "DiscoveredState",
    "DiscoveredElement",
    "DiscoveredTransition",
]
