"""Visual State Learner for GUI Environment Discovery.

Learns visual states of UI elements through observation:
- Button states (enabled, disabled, hover, pressed)
- Input states (focused, unfocused)
- Toggle states (checked, unchecked)
- Expansion states (expanded, collapsed)

Uses before/after comparisons from user interactions.
"""

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.environment import (
    ColorProfile,
    ElementState,
    ElementStateType,
    ElementTypeStates,
    StateDetectionMethod,
    VisualSignature,
    VisualStates,
)

from qontinui.vision.environment.analyzers.base import BaseAnalyzer

logger = logging.getLogger(__name__)


class VisualStateLearner(BaseAnalyzer[VisualStates]):
    """Learns visual states of UI elements through interaction observation.

    Observes click-and-observe, focus tracking, and toggle recording
    to learn how different element states appear visually.
    """

    def __init__(
        self,
        min_observations: int = 3,
        saturation_threshold: float = 0.3,
        brightness_threshold: float = 0.2,
    ) -> None:
        """Initialize the visual state learner.

        Args:
            min_observations: Minimum observations to learn a state.
            saturation_threshold: Threshold for saturation-based detection.
            brightness_threshold: Threshold for brightness-based detection.
        """
        super().__init__("VisualStateLearner")
        self.min_observations = min_observations
        self.saturation_threshold = saturation_threshold
        self.brightness_threshold = brightness_threshold

    async def analyze(
        self,
        screenshots: list[NDArray[np.uint8]],
        click_observations: list[dict[str, Any]] | None = None,
        focus_observations: list[dict[str, Any]] | None = None,
        toggle_observations: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> VisualStates:
        """Learn visual states from interaction observations.

        Args:
            screenshots: Base screenshots (not directly used, but for consistency).
            click_observations: List of click observations with before/after.
                Each should have: element_type, bbox, before_image, after_image
            focus_observations: List of focus observations.
            toggle_observations: List of toggle observations.

        Returns:
            VisualStates with learned state information.
        """
        self.reset()

        element_states: dict[str, ElementTypeStates] = {}
        total_observations = 0

        # Learn button states from click observations
        if click_observations:
            button_states = self._learn_button_states(click_observations)
            if button_states.states:
                element_states["button"] = button_states
                total_observations += button_states.total_observations

        # Learn input states from focus observations
        if focus_observations:
            input_states = self._learn_input_states(focus_observations)
            if input_states.states:
                element_states["input"] = input_states
                total_observations += input_states.total_observations

        # Learn toggle states from toggle observations
        if toggle_observations:
            checkbox_states = self._learn_toggle_states(toggle_observations, "checkbox")
            if checkbox_states.states:
                element_states["checkbox"] = checkbox_states
                total_observations += checkbox_states.total_observations

        self.confidence = self._calculate_confidence(
            total_observations,
            min_samples=5,
            optimal_samples=50,
        )

        return VisualStates(
            element_states=element_states,
            actions_observed=total_observations,
            confidence=self.confidence,
        )

    def _learn_button_states(
        self,
        observations: list[dict[str, Any]],
    ) -> ElementTypeStates:
        """Learn button visual states from click observations.

        Args:
            observations: Click observations with before/after images.

        Returns:
            ElementTypeStates for buttons.
        """
        states: dict[str, ElementState] = {}
        enabled_samples: list[dict[str, Any]] = []
        disabled_samples: list[dict[str, Any]] = []
        hover_samples: list[dict[str, Any]] = []

        for obs in observations:
            before = obs.get("before_image")
            _after = obs.get("after_image")  # noqa: F841 - kept for future state comparison
            element_type = obs.get("element_type", "button")

            if element_type != "button":
                continue

            if before is not None:
                before = self._ensure_bgr(before)

                # Analyze the before state
                signature = self._extract_visual_signature(before)

                # Determine if this was enabled or disabled
                if obs.get("click_succeeded", True):
                    enabled_samples.append({"image": before, "signature": signature})
                else:
                    disabled_samples.append({"image": before, "signature": signature})

            # If there's a hover state captured
            if obs.get("hover_image") is not None:
                hover_img = self._ensure_bgr(obs["hover_image"])
                hover_sig = self._extract_visual_signature(hover_img)
                hover_samples.append({"image": hover_img, "signature": hover_sig})

        # Create enabled state
        if len(enabled_samples) >= self.min_observations:
            states["enabled"] = self._aggregate_state(enabled_samples, ElementStateType.ENABLED)

        # Create disabled state
        if len(disabled_samples) >= self.min_observations:
            states["disabled"] = self._aggregate_state(disabled_samples, ElementStateType.DISABLED)

        # Create hover state
        if len(hover_samples) >= self.min_observations:
            states["hover"] = self._aggregate_state(hover_samples, ElementStateType.HOVER)

        # Determine best detection method
        detection_method = self._determine_detection_method(states)

        total = len(enabled_samples) + len(disabled_samples) + len(hover_samples)

        return ElementTypeStates(
            element_type="button",
            states=states,
            detection_method=detection_method,
            total_observations=total,
        )

    def _learn_input_states(
        self,
        observations: list[dict[str, Any]],
    ) -> ElementTypeStates:
        """Learn input field visual states from focus observations.

        Args:
            observations: Focus observations with focused/unfocused images.

        Returns:
            ElementTypeStates for inputs.
        """
        states: dict[str, ElementState] = {}
        focused_samples: list[dict[str, Any]] = []
        unfocused_samples: list[dict[str, Any]] = []

        for obs in observations:
            focused = obs.get("focused_image")
            unfocused = obs.get("unfocused_image")

            if focused is not None:
                focused = self._ensure_bgr(focused)
                signature = self._extract_visual_signature(focused)
                focused_samples.append({"image": focused, "signature": signature})

            if unfocused is not None:
                unfocused = self._ensure_bgr(unfocused)
                signature = self._extract_visual_signature(unfocused)
                unfocused_samples.append({"image": unfocused, "signature": signature})

        if len(focused_samples) >= self.min_observations:
            states["focused"] = self._aggregate_state(focused_samples, ElementStateType.FOCUSED)

        if len(unfocused_samples) >= self.min_observations:
            states["unfocused"] = self._aggregate_state(
                unfocused_samples, ElementStateType.UNFOCUSED
            )

        detection_method = StateDetectionMethod.BORDER_ANALYSIS
        if states:
            # Check if focus is detected by glow
            focused_state = states.get("focused")
            if focused_state and focused_state.visual_signature.has_glow:
                detection_method = StateDetectionMethod.BORDER_ANALYSIS

        total = len(focused_samples) + len(unfocused_samples)

        return ElementTypeStates(
            element_type="input",
            states=states,
            detection_method=detection_method,
            total_observations=total,
        )

    def _learn_toggle_states(
        self,
        observations: list[dict[str, Any]],
        element_type: str = "checkbox",
    ) -> ElementTypeStates:
        """Learn toggle visual states from toggle observations.

        Args:
            observations: Toggle observations with checked/unchecked images.
            element_type: Type of toggle element (checkbox, switch, etc.).

        Returns:
            ElementTypeStates for toggles.
        """
        states: dict[str, ElementState] = {}
        checked_samples: list[dict[str, Any]] = []
        unchecked_samples: list[dict[str, Any]] = []

        for obs in observations:
            checked = obs.get("checked_image")
            unchecked = obs.get("unchecked_image")

            if checked is not None:
                checked = self._ensure_bgr(checked)
                signature = self._extract_visual_signature(checked)
                # Add checkbox-specific analysis
                signature = self._analyze_checkbox_interior(checked, signature)
                checked_samples.append({"image": checked, "signature": signature})

            if unchecked is not None:
                unchecked = self._ensure_bgr(unchecked)
                signature = self._extract_visual_signature(unchecked)
                signature = self._analyze_checkbox_interior(unchecked, signature)
                unchecked_samples.append({"image": unchecked, "signature": signature})

        if len(checked_samples) >= self.min_observations:
            states["checked"] = self._aggregate_state(checked_samples, ElementStateType.CHECKED)

        if len(unchecked_samples) >= self.min_observations:
            states["unchecked"] = self._aggregate_state(
                unchecked_samples, ElementStateType.UNCHECKED
            )

        total = len(checked_samples) + len(unchecked_samples)

        return ElementTypeStates(
            element_type=element_type,
            states=states,
            detection_method=StateDetectionMethod.INTERIOR_FILL,
            total_observations=total,
        )

    def _extract_visual_signature(
        self,
        image: NDArray[np.uint8],
    ) -> VisualSignature:
        """Extract visual signature from an element image.

        Args:
            image: BGR image of element.

        Returns:
            VisualSignature with extracted characteristics.
        """
        h, w = image.shape[:2]

        # Get color profile
        color_profile = self._extract_color_profile(image)

        # Detect border
        border_width = self._detect_border_width(image)

        # Detect glow
        has_glow, glow_color = self._detect_glow(image)

        # Calculate fill percentage
        fill_pct = self._calculate_fill_percentage(image)

        return VisualSignature(
            color_profile=color_profile,
            fill_percentage=fill_pct,
            border_width=border_width,
            has_glow=has_glow,
            glow_color=glow_color,
        )

    def _extract_color_profile(
        self,
        image: NDArray[np.uint8],
    ) -> ColorProfile:
        """Extract color profile from element image.

        Args:
            image: BGR element image.

        Returns:
            ColorProfile with extracted colors.
        """
        h, w = image.shape[:2]

        # Background: sample corners
        corner_size = max(2, min(h, w) // 4)
        corners = [
            image[:corner_size, :corner_size],
            image[:corner_size, -corner_size:],
            image[-corner_size:, :corner_size],
            image[-corner_size:, -corner_size:],
        ]
        corner_pixels = np.vstack([c.reshape(-1, 3) for c in corners])
        bg_color = corner_pixels.mean(axis=0).astype(int)
        b, g, r = bg_color
        background = self._rgb_to_hex(int(r), int(g), int(b))

        # Foreground: center region (excluding corners)
        center_h, center_w = h // 3, w // 3
        center = image[center_h : 2 * center_h, center_w : 2 * center_w]
        if center.size > 0:
            fg_color = center.reshape(-1, 3).mean(axis=0).astype(int)
            b, g, r = fg_color
            foreground = self._rgb_to_hex(int(r), int(g), int(b))
        else:
            foreground = None

        # Calculate relative saturation and brightness
        saturation = self._calculate_saturation(image)
        brightness = image.mean() / 255.0

        return ColorProfile(
            background=background,
            foreground=foreground,
            saturation=saturation,
            brightness=brightness,
        )

    def _calculate_saturation(
        self,
        image: NDArray[np.uint8],
    ) -> float:
        """Calculate average saturation of image.

        Args:
            image: BGR image.

        Returns:
            Saturation value (0.0-1.0).
        """
        try:
            import cv2

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            return float(hsv[:, :, 1].mean() / 255.0)
        except ImportError:
            # Simplified calculation
            image_float = image.astype(float)
            max_c = image_float.max(axis=2)
            min_c = image_float.min(axis=2)
            mask = max_c > 0
            sat = np.zeros_like(max_c)
            sat[mask] = (max_c[mask] - min_c[mask]) / max_c[mask]
            return float(sat.mean())

    def _detect_border_width(
        self,
        image: NDArray[np.uint8],
    ) -> int | None:
        """Detect border width of element.

        Args:
            image: BGR element image.

        Returns:
            Border width in pixels or None.
        """
        try:
            import cv2

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Count edge pixels at borders
            h, w = edges.shape
            top_edges = edges[0, :].sum()
            bottom_edges = edges[-1, :].sum()
            left_edges = edges[:, 0].sum()
            right_edges = edges[:, -1].sum()

            total_border_edges = top_edges + bottom_edges + left_edges + right_edges
            perimeter = 2 * (h + w)

            if total_border_edges > perimeter * 0.5:
                # Estimate width by counting consecutive edge rows
                for border_w in range(1, min(h, w) // 4):
                    inner_edges = edges[border_w:-border_w, border_w:-border_w].sum()
                    outer_edges = edges.sum() - inner_edges
                    if outer_edges > inner_edges * 2:
                        return border_w
                return 1
            return None

        except ImportError:
            return None

    def _detect_glow(
        self,
        image: NDArray[np.uint8],
    ) -> tuple[bool, str | None]:
        """Detect if element has a glow effect.

        Args:
            image: BGR element image.

        Returns:
            Tuple of (has_glow, glow_color_hex).
        """
        try:
            import cv2

            h, w = image.shape[:2]

            # Check if outer edges are brighter/different from background
            # Sample outer ring
            outer_pixels = np.vstack(
                [
                    image[0, :],  # Top
                    image[-1, :],  # Bottom
                    image[:, 0],  # Left
                    image[:, -1],  # Right
                ]
            )

            # Sample inner area (skip potential border)
            margin = max(3, min(h, w) // 8)
            inner = image[margin:-margin, margin:-margin]
            if inner.size == 0:
                return False, None

            inner_avg = inner.reshape(-1, 3).mean(axis=0)
            outer_avg = outer_pixels.mean(axis=0)

            # Check if outer is brighter (glow)
            outer_brightness = outer_avg.mean()
            inner_brightness = inner_avg.mean()

            if outer_brightness > inner_brightness * 1.2:
                b, g, r = outer_avg.astype(int)
                return True, self._rgb_to_hex(int(r), int(g), int(b))

            return False, None

        except ImportError:
            return False, None

    def _calculate_fill_percentage(
        self,
        image: NDArray[np.uint8],
    ) -> float:
        """Calculate interior fill percentage.

        Args:
            image: BGR element image.

        Returns:
            Fill percentage (0.0-1.0).
        """
        # Use brightness threshold to separate filled/empty
        gray = image.mean(axis=2) if len(image.shape) == 3 else image
        threshold = gray.mean()

        # Assume darker = filled for checkboxes
        filled_pixels = (gray < threshold).sum()
        total_pixels = gray.size

        return filled_pixels / total_pixels if total_pixels > 0 else 0.0

    def _analyze_checkbox_interior(
        self,
        image: NDArray[np.uint8],
        signature: VisualSignature,
    ) -> VisualSignature:
        """Add checkbox-specific analysis to signature.

        Args:
            image: BGR checkbox image.
            signature: Existing visual signature.

        Returns:
            Updated signature with checkmark detection.
        """
        # Detect checkmark by looking for diagonal lines
        try:
            import cv2

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Look for diagonal edges
            kernel_diag1 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype=np.float32)
            kernel_diag2 = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]], dtype=np.float32)

            diag1 = cv2.filter2D(gray, -1, kernel_diag1)
            diag2 = cv2.filter2D(gray, -1, kernel_diag2)

            diagonal_response = (np.abs(diag1) + np.abs(diag2)).mean()

            # High diagonal response suggests checkmark
            signature.has_checkmark = diagonal_response > 10

        except ImportError:
            signature.has_checkmark = (
                signature.fill_percentage is not None and signature.fill_percentage > 0.5
            )

        return signature

    def _aggregate_state(
        self,
        samples: list[dict[str, Any]],
        state_type: ElementStateType,
    ) -> ElementState:
        """Aggregate multiple samples into a single state definition.

        Args:
            samples: List of samples with signatures.
            state_type: Type of state being created.

        Returns:
            ElementState with aggregated signature.
        """
        # Average the color profiles
        backgrounds = []
        foregrounds = []
        saturations = []
        brightnesses = []
        border_widths = []
        has_glows = []
        fill_percentages = []
        has_checkmarks = []

        for sample in samples:
            sig = sample["signature"]
            if sig.color_profile:
                if sig.color_profile.background:
                    backgrounds.append(sig.color_profile.background)
                if sig.color_profile.foreground:
                    foregrounds.append(sig.color_profile.foreground)
                if sig.color_profile.saturation is not None:
                    saturations.append(sig.color_profile.saturation)
                if sig.color_profile.brightness is not None:
                    brightnesses.append(sig.color_profile.brightness)

            if sig.border_width is not None:
                border_widths.append(sig.border_width)
            if sig.has_glow is not None:
                has_glows.append(sig.has_glow)
            if sig.fill_percentage is not None:
                fill_percentages.append(sig.fill_percentage)
            if sig.has_checkmark is not None:
                has_checkmarks.append(sig.has_checkmark)

        # Create aggregated signature
        color_profile = ColorProfile()
        if backgrounds:
            color_profile.background = self._mode_color(backgrounds)
        if foregrounds:
            color_profile.foreground = self._mode_color(foregrounds)
        if saturations:
            color_profile.saturation = sum(saturations) / len(saturations)
        if brightnesses:
            color_profile.brightness = sum(brightnesses) / len(brightnesses)

        signature = VisualSignature(
            color_profile=color_profile,
            border_width=int(sum(border_widths) / len(border_widths)) if border_widths else None,
            has_glow=sum(has_glows) > len(has_glows) / 2 if has_glows else None,
            fill_percentage=(
                sum(fill_percentages) / len(fill_percentages) if fill_percentages else None
            ),
            has_checkmark=sum(has_checkmarks) > len(has_checkmarks) / 2 if has_checkmarks else None,
        )

        return ElementState(
            state_type=state_type,
            observed_samples=len(samples),
            visual_signature=signature,
            confidence=min(1.0, len(samples) / 10),
        )

    def _mode_color(self, colors: list[str]) -> str:
        """Get mode (most common) color from list.

        Args:
            colors: List of hex color strings.

        Returns:
            Most common color.
        """
        from collections import Counter

        counts = Counter(colors)
        return counts.most_common(1)[0][0]

    def _determine_detection_method(
        self,
        states: dict[str, ElementState],
    ) -> StateDetectionMethod:
        """Determine best detection method for differentiating states.

        Args:
            states: Learned states.

        Returns:
            Best StateDetectionMethod.
        """
        if len(states) < 2:
            return StateDetectionMethod.COMBINED

        state_list = list(states.values())

        # Check saturation difference
        saturations = [
            s.visual_signature.color_profile.saturation
            for s in state_list
            if s.visual_signature.color_profile and s.visual_signature.color_profile.saturation
        ]
        if len(saturations) >= 2:
            sat_diff = max(saturations) - min(saturations)
            if sat_diff > self.saturation_threshold:
                return StateDetectionMethod.COLOR_SATURATION

        # Check brightness difference
        brightnesses = [
            s.visual_signature.color_profile.brightness
            for s in state_list
            if s.visual_signature.color_profile and s.visual_signature.color_profile.brightness
        ]
        if len(brightnesses) >= 2:
            bright_diff = max(brightnesses) - min(brightnesses)
            if bright_diff > self.brightness_threshold:
                return StateDetectionMethod.OPACITY

        # Check fill percentage (for checkboxes)
        fills = [
            s.visual_signature.fill_percentage
            for s in state_list
            if s.visual_signature.fill_percentage is not None
        ]
        if len(fills) >= 2:
            fill_diff = max(fills) - min(fills)
            if fill_diff > 0.3:
                return StateDetectionMethod.INTERIOR_FILL

        # Check border differences
        borders = [
            s.visual_signature.border_width
            for s in state_list
            if s.visual_signature.border_width is not None
        ]
        if len(set(borders)) > 1:
            return StateDetectionMethod.BORDER_ANALYSIS

        return StateDetectionMethod.COMBINED
