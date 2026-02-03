"""Unified State Discovery Service.

This module provides the main entry point for state discovery with
automatic strategy selection based on available input data.

Example:
    from qontinui.discovery.state_discovery import StateDiscoveryService

    service = StateDiscoveryService()

    # From render logs (legacy)
    result = service.discover_from_renders(renders)

    # From cooccurrence export with fingerprints
    result = service.discover_from_export(export_data)

    # Auto-detect based on input
    result = service.discover(StateDiscoveryInput(
        renders=renders,
        cooccurrence_export=export_data,
    ))
"""

from __future__ import annotations

import logging
from typing import Any

from qontinui.state_machine.fingerprint_state_discovery import FingerprintStateDiscoveryConfig

from .base import (
    DiscoveryStrategyType,
    StateDiscoveryInput,
    StateDiscoveryResult,
    StateDiscoveryStrategy,
)
from .strategies import FingerprintStrategy, LegacyStrategy

logger = logging.getLogger(__name__)


class StateDiscoveryService:
    """Unified service for state discovery.

    Automatically selects the best strategy based on available input data:
    - If fingerprint data is available, uses FingerprintStrategy
    - Otherwise falls back to LegacyStrategy

    You can also explicitly request a specific strategy.
    """

    def __init__(
        self,
        fingerprint_config: FingerprintStateDiscoveryConfig | None = None,
    ) -> None:
        """Initialize the service.

        Args:
            fingerprint_config: Optional configuration for fingerprint strategy
        """
        self._strategies: dict[DiscoveryStrategyType, StateDiscoveryStrategy] = {
            DiscoveryStrategyType.LEGACY: LegacyStrategy(),
            DiscoveryStrategyType.FINGERPRINT: FingerprintStrategy(fingerprint_config),
        }

    def discover(
        self,
        input_data: StateDiscoveryInput,
        strategy: DiscoveryStrategyType = DiscoveryStrategyType.AUTO,
    ) -> StateDiscoveryResult:
        """Discover states from input data.

        Args:
            input_data: Input data for discovery
            strategy: Strategy to use (AUTO for automatic selection)

        Returns:
            Discovery result with states, elements, and transitions
        """
        # Select strategy
        selected_strategy = self._select_strategy(input_data, strategy)

        if selected_strategy is None:
            logger.warning("No suitable strategy found for input data")
            return StateDiscoveryResult(
                states=[],
                elements=[],
                element_to_renders={},
                render_count=0,
                unique_element_count=0,
                strategy_used=DiscoveryStrategyType.LEGACY,
                strategy_metadata={"error": "no_suitable_strategy"},
            )

        logger.info(f"Using {selected_strategy.strategy_type.value} strategy for discovery")

        # Run discovery
        return selected_strategy.discover(input_data)

    def discover_from_renders(
        self,
        renders: list[dict[str, Any]],
        include_html_ids: bool = False,
        strategy: DiscoveryStrategyType = DiscoveryStrategyType.LEGACY,
    ) -> StateDiscoveryResult:
        """Discover states from render log entries.

        This is the primary method for UI Bridge render-based discovery.
        Uses the legacy strategy by default.

        Args:
            renders: List of render log entries
            include_html_ids: Whether to include HTML id attributes
            strategy: Strategy to use

        Returns:
            Discovery result
        """
        input_data = StateDiscoveryInput(
            renders=renders,
            include_html_ids=include_html_ids,
        )

        return self.discover(input_data, strategy)

    def discover_from_export(
        self,
        cooccurrence_export: dict[str, Any],
        strategy: DiscoveryStrategyType = DiscoveryStrategyType.AUTO,
    ) -> StateDiscoveryResult:
        """Discover states from a co-occurrence export.

        This is the primary method for fingerprint-enhanced discovery.
        Automatically uses fingerprint strategy if fingerprint data is present.

        Args:
            cooccurrence_export: Export data from UI Bridge
            strategy: Strategy to use (AUTO recommended)

        Returns:
            Discovery result with enhanced state information
        """
        input_data = StateDiscoveryInput(
            cooccurrence_export=cooccurrence_export,
        )

        return self.discover(input_data, strategy)

    def _select_strategy(
        self,
        input_data: StateDiscoveryInput,
        requested: DiscoveryStrategyType,
    ) -> StateDiscoveryStrategy | None:
        """Select the best strategy for the input data.

        Args:
            input_data: Input data to analyze
            requested: Requested strategy type

        Returns:
            Selected strategy or None if no suitable strategy
        """
        # If specific strategy requested, check if it can process
        if requested != DiscoveryStrategyType.AUTO:
            strategy = self._strategies.get(requested)
            if strategy and strategy.can_process(input_data):
                return strategy
            logger.warning(
                f"Requested strategy {requested.value} cannot process input, "
                "falling back to auto-detection"
            )

        # Auto-detect: prefer fingerprint if available
        if input_data.has_fingerprint_data():
            fingerprint_strategy = self._strategies.get(DiscoveryStrategyType.FINGERPRINT)
            if fingerprint_strategy and fingerprint_strategy.can_process(input_data):
                return fingerprint_strategy

        # Fall back to legacy
        if input_data.has_render_data():
            legacy_strategy = self._strategies.get(DiscoveryStrategyType.LEGACY)
            if legacy_strategy and legacy_strategy.can_process(input_data):
                return legacy_strategy

        return None

    def get_available_strategies(self) -> list[DiscoveryStrategyType]:
        """Get list of available strategy types.

        Returns:
            List of available strategy types
        """
        return list(self._strategies.keys())


# Convenience function for simple usage
def discover_states(
    renders: list[dict[str, Any]] | None = None,
    cooccurrence_export: dict[str, Any] | None = None,
    include_html_ids: bool = False,
    strategy: DiscoveryStrategyType = DiscoveryStrategyType.AUTO,
) -> StateDiscoveryResult:
    """Discover states from UI Bridge data.

    This is a convenience function that creates a service and runs discovery.
    For repeated use, create a StateDiscoveryService instance instead.

    Args:
        renders: Optional list of render log entries
        cooccurrence_export: Optional co-occurrence export with fingerprints
        include_html_ids: Whether to include HTML id attributes
        strategy: Strategy to use (AUTO for automatic selection)

    Returns:
        Discovery result with states, elements, and transitions

    Example:
        # From renders
        result = discover_states(renders=my_renders)

        # From export with fingerprints
        result = discover_states(cooccurrence_export=my_export)

        # Both (fingerprints preferred if available)
        result = discover_states(
            renders=my_renders,
            cooccurrence_export=my_export,
        )
    """
    service = StateDiscoveryService()

    input_data = StateDiscoveryInput(
        renders=renders or [],
        include_html_ids=include_html_ids,
        cooccurrence_export=cooccurrence_export,
    )

    return service.discover(input_data, strategy)
