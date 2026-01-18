"""
AWAS-based runtime extractor.

Extracts UI elements and states from AWAS manifests rather than visual/DOM analysis.
This provides high-confidence, semantically-rich extraction for websites that
implement the AWAS standard.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qontinui.awas.discovery import AwasDiscoveryService
from qontinui.awas.executor import AwasExecutor
from qontinui.awas.types import AwasAction, AwasElement, AwasManifest, HttpMethod

from ...models.base import BoundingBox, Screenshot, Viewport
from ...web.models import BoundingBox as WebBoundingBox
from ...web.models import (
    ExtractedElement,
    ExtractedState,
    StateType,
)
from ..base import DetectedRegion, InteractionAction, RuntimeExtractor, StateChange
from ..types import ExtractionTarget, RuntimeExtractionSession, RuntimeStateCapture

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class AwasRuntimeExtractor(RuntimeExtractor):
    """
    Runtime extractor that uses AWAS manifests for semantic extraction.

    Unlike DOM or vision-based extractors, AWAS extraction uses the website's
    declared manifest to understand available actions and elements. This provides:
    - High confidence (1.0) for all extracted elements
    - Semantic intent descriptions
    - Type-safe parameter definitions
    - Rate limiting awareness
    - Authentication requirements

    This extractor is best used in conjunction with DOM/vision extractors
    as a "semantic overlay" that enriches extracted elements with AWAS data.
    """

    def __init__(self) -> None:
        """Initialize the AWAS runtime extractor."""
        super().__init__()
        self.discovery = AwasDiscoveryService()
        self.executor = AwasExecutor()

        # State
        self._manifest: AwasManifest | None = None
        self._target: ExtractionTarget | None = None
        self._elements: list[AwasElement] = []
        self._capture_counter = 0
        self.is_connected = False
        self.session: RuntimeExtractionSession | None = None

    async def connect(self, target: ExtractionTarget) -> None:
        """
        Connect to web target by discovering its AWAS manifest.

        Args:
            target: ExtractionTarget with URL.

        Raises:
            ConnectionError: If no AWAS manifest is found.
        """
        if not target.url:
            raise ValueError("ExtractionTarget must have a URL for AwasRuntimeExtractor")

        try:
            self._manifest = await self.discovery.discover(target.url)

            if self._manifest is None:
                raise ConnectionError(
                    f"No AWAS manifest found at {target.url}. " "The website does not support AWAS."
                )

            self._target = target
            self.is_connected = True

            logger.info(
                f"Connected to AWAS-enabled site: {self._manifest.app_name} "
                f"({len(self._manifest.actions)} actions, "
                f"level {self._manifest.conformance_level.value})"
            )

        except ConnectionError:
            raise
        except Exception as e:
            logger.error(f"Failed to connect to {target.url}: {e}")
            raise ConnectionError(f"Failed to discover AWAS manifest at {target.url}") from e

    async def disconnect(self) -> None:
        """Disconnect and clean up."""
        self._manifest = None
        self._target = None
        self._elements = []
        self.is_connected = False
        logger.info("Disconnected from AWAS target")

    async def extract_current_state(self) -> RuntimeStateCapture:
        """
        Extract current state from AWAS manifest.

        Since AWAS is manifest-based (not live DOM), this returns the
        manifest's declared actions as extractable elements.

        Returns:
            RuntimeStateCapture with AWAS-derived elements.
        """
        if not self.is_connected or not self._manifest:
            raise RuntimeError("Not connected to AWAS target")

        self._capture_counter += 1
        capture_id = f"awas_capture_{self._capture_counter:04d}"

        # Convert AWAS actions to ExtractedElements
        elements = await self.extract_elements()

        # Create a state representing the AWAS-enabled page
        state = ExtractedState(
            id=f"awas_state_{self._capture_counter:04d}",
            name=f"{self._manifest.app_name} (AWAS)",
            bbox=WebBoundingBox(x=0, y=0, width=1920, height=1080),
            state_type=StateType.PAGE,
            element_ids=[e.id for e in elements],
            screenshot_id=None,  # AWAS doesn't capture screenshots
            detection_method="awas_manifest",
            confidence=1.0,
            source_url=self._target.url if self._target else None,
            metadata={
                "awas_app_name": self._manifest.app_name,
                "awas_conformance": self._manifest.conformance_level.value,
                "awas_action_count": len(self._manifest.actions),
            },
        )

        return RuntimeStateCapture(
            capture_id=capture_id,
            timestamp=None,
            url=self._target.url if self._target else None,
            elements=elements,
            states=[state],
            viewport=(1920, 1080),
            scroll_position=(0, 0),
            metadata={
                "extraction_method": "awas",
                "manifest": self._manifest.model_dump(),
            },
        )

    async def extract_elements(self) -> list[ExtractedElement]:
        """
        Extract elements from AWAS manifest.

        Converts AWAS actions to ExtractedElement objects with high confidence.

        Returns:
            List of ExtractedElement objects derived from AWAS actions.
        """
        if not self._manifest:
            return []

        elements: list[ExtractedElement] = []

        for i, action in enumerate(self._manifest.actions):
            element = self._action_to_element(action, i)
            elements.append(element)

        logger.info(f"Extracted {len(elements)} elements from AWAS manifest")
        return elements

    def _action_to_element(self, action: AwasAction, index: int) -> ExtractedElement:
        """Convert an AWAS action to an ExtractedElement."""
        # Determine element type based on action method
        element_type = self._infer_element_type(action)

        # Build selector from action ID
        selector = f'[data-awas-action="{action.id}"]'

        return ExtractedElement(
            id=f"awas_elem_{action.id}",
            tag_name="button" if action.side_effect else "a",
            element_type=element_type,
            text=action.name,
            bbox=WebBoundingBox(x=0, y=index * 50, width=200, height=40),  # Placeholder bounds
            is_visible=True,
            is_interactive=True,
            selector=selector,
            computed_role="button" if action.side_effect else "link",
            confidence=1.0,  # AWAS provides definitive info
            extraction_method="awas_manifest",
            metadata={
                "awas_action_id": action.id,
                "awas_intent": action.intent,
                "awas_method": action.method.value,
                "awas_endpoint": action.endpoint,
                "awas_side_effect": action.side_effect,
                "awas_parameters": [p.model_dump() for p in action.parameters],
                "awas_required_scopes": action.required_scopes,
                "awas_rate_limit": action.rate_limit,
            },
        )

    def _infer_element_type(self, action: AwasAction) -> str:
        """Infer HTML element type from AWAS action."""
        if action.method == HttpMethod.GET and not action.side_effect:
            return "link"
        if action.method == HttpMethod.DELETE:
            return "button"  # Destructive action
        if action.method in (HttpMethod.POST, HttpMethod.PUT, HttpMethod.PATCH):
            return "button"  # Modifying action
        return "button"

    async def detect_regions(self) -> list[DetectedRegion]:
        """
        Detect UI regions from AWAS manifest.

        AWAS doesn't define regions explicitly, so this returns a single
        region representing the API surface.

        Returns:
            List with a single region for the AWAS API surface.
        """
        if not self._manifest:
            return []

        # Group actions by intent/type
        regions: list[DetectedRegion] = []

        # Create a region for all actions
        regions.append(
            DetectedRegion(
                id="awas_api_surface",
                name=f"{self._manifest.app_name} API",
                bbox=BoundingBox(x=0, y=0, width=1920, height=1080),
                region_type="api_surface",
                confidence=1.0,
                metadata={
                    "action_count": len(self._manifest.actions),
                    "conformance_level": self._manifest.conformance_level.value,
                },
            )
        )

        return regions

    async def capture_screenshot(self, region: BoundingBox | None = None) -> Screenshot:
        """
        AWAS doesn't capture screenshots (it's API-based).

        Returns a placeholder Screenshot object.
        """
        return Screenshot(
            id=f"awas_placeholder_{uuid.uuid4().hex[:8]}",
            path=Path("/dev/null"),  # No actual file
            viewport=Viewport(width=1920, height=1080),
            metadata={"note": "AWAS extraction does not capture screenshots"},
        )

    async def navigate_to_route(self, route: str) -> None:
        """
        AWAS navigation is handled through action execution.

        For AWAS targets, "navigation" means executing a specific action.

        Args:
            route: Could be an action ID or endpoint path.
        """
        if not self._manifest:
            raise RuntimeError("Not connected to AWAS target")

        # Check if route matches an action ID
        action = self._manifest.get_action(route)
        if action:
            logger.info(f"Executing AWAS action: {action.name}")
            result = await self.executor.execute(self._manifest, route)
            if not result.success:
                raise RuntimeError(f"AWAS action failed: {result.error}")
        else:
            logger.warning(f"No AWAS action found for route: {route}")

    async def simulate_interaction(self, action: InteractionAction) -> StateChange:
        """
        Simulate interaction by executing AWAS action.

        For AWAS targets, interactions are API calls rather than UI clicks.

        Args:
            action: InteractionAction with awas_action_id in metadata.

        Returns:
            StateChange with results of the action execution.
        """
        if not self._manifest:
            raise RuntimeError("Not connected to AWAS target")

        # Extract AWAS action ID from metadata or target element
        action_id = action.metadata.get("awas_action_id")
        if not action_id and action.target_element_id:
            # Try to extract from element ID format "awas_elem_{action_id}"
            if action.target_element_id.startswith("awas_elem_"):
                action_id = action.target_element_id[len("awas_elem_") :]

        if not action_id:
            return StateChange(metadata={"error": "No AWAS action ID provided"})

        # Extract parameters from action value or metadata
        params = action.metadata.get("awas_params", {})

        # Execute the AWAS action
        result = await self.executor.execute(
            self._manifest,
            action_id,
            params,
        )

        return StateChange(
            url_changed=False,  # AWAS actions don't change URL
            metadata={
                "awas_result": result.model_dump(),
                "success": result.success,
                "status_code": result.status_code,
                "response_time_ms": result.response_time_ms,
            },
        )

    @classmethod
    def supports_target(cls, target: ExtractionTarget) -> bool:
        """
        Check if target is AWAS-enabled.

        This performs a synchronous check for AWAS manifest availability.

        Args:
            target: ExtractionTarget to check.

        Returns:
            True if target has an AWAS manifest, False otherwise.
        """
        if not target.url:
            return False

        # Quick check - don't block on full discovery
        # The actual manifest will be fetched in connect()
        discovery = AwasDiscoveryService()
        cached = discovery.get_cached_manifest(target.url)
        if cached is not None:
            return True

        # For now, return False and let connect() do the actual check
        # This avoids blocking the extractor selection process
        return False

    @classmethod
    def get_priority(cls) -> int:
        """
        AWAS has medium-high priority.

        Higher than vision (lowest), lower than Playwright (highest for web).
        AWAS should be used when available but falls back to DOM extraction.

        Returns:
            Priority value (higher = more preferred).
        """
        return 5  # Playwright is typically ~10, Vision is ~1

    def get_manifest(self) -> AwasManifest | None:
        """Get the currently loaded AWAS manifest."""
        return self._manifest

    async def execute_action(
        self,
        action_id: str,
        params: dict[str, Any] | None = None,
        credentials: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute an AWAS action directly.

        Convenience method for executing actions without going through
        the interaction simulation interface.

        Args:
            action_id: ID of the action to execute.
            params: Parameters for the action.
            credentials: Authentication credentials.

        Returns:
            Dict with execution result.
        """
        if not self._manifest:
            return {"success": False, "error": "Not connected to AWAS target"}

        result = await self.executor.execute(
            self._manifest,
            action_id,
            params,
            credentials,
        )

        return result.model_dump()

    def list_actions(self) -> list[dict[str, Any]]:
        """
        List all available AWAS actions.

        Returns:
            List of action summaries.
        """
        if not self._manifest:
            return []

        return [
            {
                "id": action.id,
                "name": action.name,
                "method": action.method.value,
                "endpoint": action.endpoint,
                "intent": action.intent,
                "side_effect": action.side_effect,
                "parameters": [p.name for p in action.parameters],
            }
            for action in self._manifest.actions
        ]
