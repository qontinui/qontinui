"""
Screenshot fixtures for testing detection components.

This module provides synthetic screenshot generation utilities for testing
element detection, region analysis, and state detection components.

Example usage:
    >>> import pytest
    >>> from tests.fixtures.screenshot_fixtures import synthetic_screenshot
    >>>
    >>> def test_button_detection(synthetic_screenshot):
    ...     screenshot = synthetic_screenshot(
    ...         width=800, height=600,
    ...         elements=[
    ...             {"type": "button", "text": "Submit", "x": 100, "y": 100}
    ...         ]
    ...     )
    ...     # Use screenshot in your test
    ...     assert screenshot.shape == (600, 800, 3)
"""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pytest


@dataclass
class ElementSpec:
    """Specification for a UI element to be drawn on synthetic screenshot."""

    element_type: str  # "button", "text", "icon", "input", "checkbox", etc.
    x: int
    y: int
    width: int = 100
    height: int = 40
    text: str | None = None
    color: tuple[int, int, int] = (200, 200, 200)
    text_color: tuple[int, int, int] = (0, 0, 0)
    border_color: tuple[int, int, int] | None = (100, 100, 100)
    border_width: int = 2
    metadata: dict[str, Any] | None = None


class SyntheticScreenshotGenerator:
    """
    Generate synthetic screenshots with known UI elements for testing.

    This class creates realistic-looking UI screenshots with precisely positioned
    elements, enabling deterministic testing of detection algorithms.

    Example usage:
        >>> generator = SyntheticScreenshotGenerator()
        >>> screenshot = generator.generate(
        ...     width=1024, height=768,
        ...     background_color=(240, 240, 240),
        ...     elements=[
        ...         ElementSpec("button", x=100, y=100, text="Click Me"),
        ...         ElementSpec("text", x=100, y=200, text="Label", width=80),
        ...     ]
        ... )
        >>> cv2.imwrite("test_screenshot.png", screenshot)
    """

    def __init__(self):
        """Initialize the screenshot generator."""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2

    def generate(
        self,
        width: int = 800,
        height: int = 600,
        background_color: tuple[int, int, int] = (255, 255, 255),
        elements: list[ElementSpec] | None = None,
        noise_level: float = 0.0,
    ) -> np.ndarray:
        """
        Generate a synthetic screenshot with specified elements.

        Args:
            width: Screenshot width in pixels
            height: Screenshot height in pixels
            background_color: Background color as (B, G, R) tuple
            elements: List of ElementSpec objects to draw
            noise_level: Amount of gaussian noise to add (0.0 to 1.0)

        Returns:
            NumPy array of shape (height, width, 3) in BGR format

        Example:
            >>> generator = SyntheticScreenshotGenerator()
            >>> img = generator.generate(
            ...     width=640, height=480,
            ...     elements=[ElementSpec("button", x=50, y=50, text="OK")]
            ... )
            >>> assert img.shape == (480, 640, 3)
        """
        # Create blank image
        image = np.full((height, width, 3), background_color, dtype=np.uint8)

        # Draw each element
        if elements:
            for element in elements:
                self._draw_element(image, element)

        # Add noise if requested
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 255, image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return image

    def _draw_element(self, image: np.ndarray, element: ElementSpec) -> None:
        """
        Draw a single UI element on the image.

        Args:
            image: Image to draw on (modified in-place)
            element: Element specification
        """
        if element.element_type == "button":
            self._draw_button(image, element)
        elif element.element_type == "text":
            self._draw_text(image, element)
        elif element.element_type == "icon":
            self._draw_icon(image, element)
        elif element.element_type == "input":
            self._draw_input(image, element)
        elif element.element_type == "checkbox":
            self._draw_checkbox(image, element)
        elif element.element_type == "rectangle":
            self._draw_rectangle(image, element)
        else:
            # Default: draw as rectangle
            self._draw_rectangle(image, element)

    def _draw_button(self, image: np.ndarray, element: ElementSpec) -> None:
        """Draw a button element."""
        # Draw rounded rectangle
        x1, y1 = element.x, element.y
        x2, y2 = element.x + element.width, element.y + element.height

        # Background
        cv2.rectangle(image, (x1, y1), (x2, y2), element.color, -1)

        # Border
        if element.border_color:
            cv2.rectangle(image, (x1, y1), (x2, y2), element.border_color, element.border_width)

        # Text
        if element.text:
            text_size = cv2.getTextSize(
                element.text, self.font, self.font_scale, self.font_thickness
            )[0]
            text_x = x1 + (element.width - text_size[0]) // 2
            text_y = y1 + (element.height + text_size[1]) // 2
            cv2.putText(
                image,
                element.text,
                (text_x, text_y),
                self.font,
                self.font_scale,
                element.text_color,
                self.font_thickness,
            )

    def _draw_text(self, image: np.ndarray, element: ElementSpec) -> None:
        """Draw a text label element."""
        if element.text:
            cv2.putText(
                image,
                element.text,
                (element.x, element.y + element.height // 2),
                self.font,
                self.font_scale,
                element.text_color,
                self.font_thickness,
            )

    def _draw_icon(self, image: np.ndarray, element: ElementSpec) -> None:
        """Draw a simple icon (circle with inner shape)."""
        center_x = element.x + element.width // 2
        center_y = element.y + element.height // 2
        radius = min(element.width, element.height) // 2 - 2

        # Draw circle
        cv2.circle(image, (center_x, center_y), radius, element.color, -1)
        if element.border_color:
            cv2.circle(
                image, (center_x, center_y), radius, element.border_color, element.border_width
            )

        # Draw inner symbol (simplified)
        inner_radius = radius // 2
        cv2.circle(image, (center_x, center_y), inner_radius, element.text_color, -1)

    def _draw_input(self, image: np.ndarray, element: ElementSpec) -> None:
        """Draw a text input field."""
        x1, y1 = element.x, element.y
        x2, y2 = element.x + element.width, element.y + element.height

        # Background (usually white)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)

        # Border
        cv2.rectangle(image, (x1, y1), (x2, y2), element.border_color or (150, 150, 150), 1)

        # Text if provided
        if element.text:
            cv2.putText(
                image,
                element.text,
                (x1 + 5, y1 + element.height // 2 + 5),
                self.font,
                self.font_scale * 0.8,
                element.text_color,
                1,
            )

    def _draw_checkbox(self, image: np.ndarray, element: ElementSpec) -> None:
        """Draw a checkbox element."""
        size = min(element.width, element.height)
        x1, y1 = element.x, element.y
        x2, y2 = x1 + size, y1 + size

        # Box background
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), element.border_color or (100, 100, 100), 2)

        # Check mark if metadata indicates checked
        if element.metadata and element.metadata.get("checked", False):
            cv2.line(image, (x1 + 3, y1 + size // 2), (x1 + size // 3, y2 - 3), (0, 0, 0), 2)
            cv2.line(image, (x1 + size // 3, y2 - 3), (x2 - 3, y1 + 3), (0, 0, 0), 2)

    def _draw_rectangle(self, image: np.ndarray, element: ElementSpec) -> None:
        """Draw a simple rectangle."""
        x1, y1 = element.x, element.y
        x2, y2 = element.x + element.width, element.y + element.height
        cv2.rectangle(image, (x1, y1), (x2, y2), element.color, -1)
        if element.border_color:
            cv2.rectangle(image, (x1, y1), (x2, y2), element.border_color, element.border_width)

    def generate_with_known_elements(
        self,
        width: int = 800,
        height: int = 600,
        num_buttons: int = 3,
        num_inputs: int = 2,
        num_icons: int = 2,
    ) -> tuple[np.ndarray, list[ElementSpec]]:
        """
        Generate a screenshot with randomly positioned known elements.

        Args:
            width: Screenshot width
            height: Screenshot height
            num_buttons: Number of buttons to generate
            num_inputs: Number of input fields to generate
            num_icons: Number of icons to generate

        Returns:
            Tuple of (image, list of ElementSpec objects)

        Example:
            >>> generator = SyntheticScreenshotGenerator()
            >>> img, elements = generator.generate_with_known_elements(
            ...     width=640, height=480, num_buttons=2
            ... )
            >>> assert len(elements) >= 2
            >>> assert all(e.element_type in ["button", "input", "icon"] for e in elements)
        """
        elements = []

        # Generate buttons
        for i in range(num_buttons):
            elements.append(
                ElementSpec(
                    element_type="button",
                    x=50 + i * 150,
                    y=50 + (i % 2) * 100,
                    width=120,
                    height=40,
                    text=f"Button {i+1}",
                    color=(200, 200, 200),
                    border_color=(100, 100, 100),
                )
            )

        # Generate input fields
        for i in range(num_inputs):
            elements.append(
                ElementSpec(
                    element_type="input",
                    x=50,
                    y=200 + i * 60,
                    width=200,
                    height=35,
                    text=f"Input {i+1}",
                )
            )

        # Generate icons
        for i in range(num_icons):
            elements.append(
                ElementSpec(
                    element_type="icon",
                    x=300 + i * 80,
                    y=250,
                    width=50,
                    height=50,
                    color=(100, 150, 200),
                )
            )

        image = self.generate(width=width, height=height, elements=elements)
        return image, elements


# Pytest fixtures


@pytest.fixture
def screenshot_generator() -> SyntheticScreenshotGenerator:
    """
    Pytest fixture providing a screenshot generator instance.

    Example:
        >>> def test_detection(screenshot_generator):
        ...     img = screenshot_generator.generate(width=640, height=480)
        ...     assert img.shape == (480, 640, 3)
    """
    return SyntheticScreenshotGenerator()


@pytest.fixture
def synthetic_screenshot(screenshot_generator):
    """
    Pytest fixture that returns a function to generate synthetic screenshots.

    Example:
        >>> def test_with_buttons(synthetic_screenshot):
        ...     img = synthetic_screenshot(
        ...         width=800, height=600,
        ...         elements=[ElementSpec("button", x=100, y=100, text="OK")]
        ...     )
        ...     # Test your detection code with img
    """

    def _generate(**kwargs):
        return screenshot_generator.generate(**kwargs)

    return _generate


@pytest.fixture
def screenshot_with_elements(screenshot_generator):
    """
    Pytest fixture that generates a screenshot with random known elements.

    Returns a tuple of (image, elements_list) for testing.

    Example:
        >>> def test_element_detection(screenshot_with_elements):
        ...     img, elements = screenshot_with_elements
        ...     # Test that your detector finds all elements
        ...     assert len(elements) > 0
    """
    return screenshot_generator.generate_with_known_elements()


@pytest.fixture
def empty_screenshot(screenshot_generator):
    """
    Pytest fixture providing a blank screenshot for testing.

    Example:
        >>> def test_no_elements(empty_screenshot):
        ...     # Test detector behavior on empty screenshot
        ...     assert empty_screenshot.shape == (600, 800, 3)
    """
    return screenshot_generator.generate(width=800, height=600, elements=[])


# Helper functions for creating specific test scenarios


def generate_synthetic_screenshot(
    width: int = 800,
    height: int = 600,
    background_color: tuple[int, int, int] = (255, 255, 255),
    elements: list[ElementSpec] | None = None,
    noise_level: float = 0.0,
) -> np.ndarray:
    """
    Generate a synthetic screenshot with specified parameters.

    This is a convenience function that creates a generator and produces
    a single screenshot. Use this for simple test cases.

    Args:
        width: Screenshot width in pixels
        height: Screenshot height in pixels
        background_color: Background color as (B, G, R) tuple
        elements: List of ElementSpec objects to draw
        noise_level: Amount of gaussian noise to add (0.0 to 1.0)

    Returns:
        NumPy array of shape (height, width, 3) in BGR format

    Example:
        >>> screenshot = generate_synthetic_screenshot(
        ...     width=640,
        ...     height=480,
        ...     elements=[ElementSpec("button", x=100, y=100, text="Click Me")]
        ... )
        >>> assert screenshot.shape == (480, 640, 3)
    """
    generator = SyntheticScreenshotGenerator()
    return generator.generate(
        width=width,
        height=height,
        background_color=background_color,
        elements=elements or [],
        noise_level=noise_level,
    )


def create_test_element(
    element_type: str,
    x: int,
    y: int,
    width: int = 100,
    height: int = 40,
    text: str | None = None,
    **kwargs,
) -> ElementSpec:
    """
    Create a test element specification.

    Convenience function for creating ElementSpec objects with sensible defaults.

    Args:
        element_type: Type of element ("button", "input", "text", "icon", etc.)
        x: X coordinate
        y: Y coordinate
        width: Element width (default: 100)
        height: Element height (default: 40)
        text: Optional text label
        **kwargs: Additional ElementSpec parameters

    Returns:
        ElementSpec object

    Example:
        >>> button = create_test_element("button", x=100, y=100, text="Submit")
        >>> assert button.element_type == "button"
        >>> assert button.text == "Submit"
    """
    return ElementSpec(
        element_type=element_type, x=x, y=y, width=width, height=height, text=text, **kwargs
    )


def create_menu_transition_pair() -> tuple[np.ndarray, np.ndarray]:
    """
    Create a pair of screenshots showing a menu transition.

    Returns before/after screenshots demonstrating a menu being opened.
    Useful for testing state transition detection.

    Returns:
        Tuple of (before_screenshot, after_screenshot)

    Example:
        >>> before, after = create_menu_transition_pair()
        >>> assert before.shape == after.shape
        >>> # Test that your detector can identify the transition
    """
    generator = SyntheticScreenshotGenerator()

    # Before: Main screen with menu button
    before_elements = [
        ElementSpec("button", x=50, y=10, width=80, height=30, text="Menu"),
        ElementSpec("text", x=200, y=200, width=200, height=40, text="Main Content"),
    ]

    before_screenshot = generator.generate(
        width=800, height=600, background_color=(240, 240, 240), elements=before_elements
    )

    # After: Menu opened with options
    after_elements = [
        ElementSpec("button", x=50, y=10, width=80, height=30, text="Menu"),
        ElementSpec("text", x=200, y=200, width=200, height=40, text="Main Content"),
        # Menu dropdown
        ElementSpec(
            "rectangle",
            x=50,
            y=45,
            width=150,
            height=200,
            color=(255, 255, 255),
            border_color=(100, 100, 100),
        ),
        ElementSpec("button", x=55, y=50, width=140, height=35, text="New"),
        ElementSpec("button", x=55, y=90, width=140, height=35, text="Open"),
        ElementSpec("button", x=55, y=130, width=140, height=35, text="Save"),
        ElementSpec("button", x=55, y=170, width=140, height=35, text="Exit"),
    ]

    after_screenshot = generator.generate(
        width=800, height=600, background_color=(240, 240, 240), elements=after_elements
    )

    return before_screenshot, after_screenshot


def create_button_screenshot(
    num_buttons: int = 3,
    button_text: list[str] | None = None,
    width: int = 800,
    height: int = 600,
) -> tuple[np.ndarray, list[ElementSpec]]:
    """
    Create a screenshot with multiple buttons.

    Convenient for testing button detection algorithms.

    Args:
        num_buttons: Number of buttons to create
        button_text: Optional list of button labels
        width: Screenshot width
        height: Screenshot height

    Returns:
        Tuple of (screenshot, list of ElementSpec for buttons)

    Example:
        >>> screenshot, buttons = create_button_screenshot(
        ...     num_buttons=3,
        ...     button_text=["OK", "Cancel", "Apply"]
        ... )
        >>> assert len(buttons) == 3
        >>> assert buttons[0].text == "OK"
    """
    generator = SyntheticScreenshotGenerator()

    if button_text is None:
        button_text = [f"Button {i+1}" for i in range(num_buttons)]
    elif len(button_text) < num_buttons:
        button_text.extend([f"Button {i+1}" for i in range(len(button_text), num_buttons)])

    buttons = []
    for i in range(num_buttons):
        button = ElementSpec(
            element_type="button",
            x=100 + (i % 3) * 200,
            y=100 + (i // 3) * 80,
            width=150,
            height=50,
            text=button_text[i],
            color=(220, 220, 220),
            border_color=(100, 100, 100),
        )
        buttons.append(button)

    screenshot = generator.generate(
        width=width, height=height, background_color=(240, 240, 240), elements=buttons
    )

    return screenshot, buttons


def create_dialog_screenshot(
    dialog_title: str = "Dialog",
    show_ok: bool = True,
    show_cancel: bool = True,
    width: int = 800,
    height: int = 600,
) -> tuple[np.ndarray, list[ElementSpec]]:
    """
    Create a screenshot with a dialog box.

    Useful for testing dialog detection and region analysis.

    Args:
        dialog_title: Title text for the dialog
        show_ok: Include OK button
        show_cancel: Include Cancel button
        width: Screenshot width
        height: Screenshot height

    Returns:
        Tuple of (screenshot, list of ElementSpec for dialog elements)

    Example:
        >>> screenshot, elements = create_dialog_screenshot(
        ...     dialog_title="Confirm Action",
        ...     show_ok=True,
        ...     show_cancel=True
        ... )
        >>> assert len(elements) >= 2  # At least dialog box and title
    """
    generator = SyntheticScreenshotGenerator()

    dialog_x = 200
    dialog_y = 150
    dialog_width = 400
    dialog_height = 300

    elements = [
        # Dialog background
        ElementSpec(
            "rectangle",
            x=dialog_x,
            y=dialog_y,
            width=dialog_width,
            height=dialog_height,
            color=(240, 240, 240),
            border_color=(100, 100, 100),
            border_width=2,
        ),
        # Title bar
        ElementSpec(
            "rectangle",
            x=dialog_x,
            y=dialog_y,
            width=dialog_width,
            height=40,
            color=(200, 200, 200),
            border_color=(100, 100, 100),
        ),
        # Title text
        ElementSpec(
            "text",
            x=dialog_x + 10,
            y=dialog_y + 10,
            width=dialog_width - 20,
            height=30,
            text=dialog_title,
            text_color=(0, 0, 0),
        ),
        # Content area
        ElementSpec(
            "text",
            x=dialog_x + 20,
            y=dialog_y + 80,
            width=dialog_width - 40,
            height=100,
            text="Dialog content goes here",
            text_color=(50, 50, 50),
        ),
    ]

    # Add buttons
    button_y = dialog_y + dialog_height - 60
    button_x_start = dialog_x + dialog_width - 200

    if show_cancel:
        elements.append(
            ElementSpec(
                "button",
                x=button_x_start,
                y=button_y,
                width=80,
                height=35,
                text="Cancel",
                color=(220, 220, 220),
                border_color=(100, 100, 100),
            )
        )

    if show_ok:
        ok_x = button_x_start + (90 if show_cancel else 0)
        elements.append(
            ElementSpec(
                "button",
                x=ok_x,
                y=button_y,
                width=80,
                height=35,
                text="OK",
                color=(100, 180, 100),
                border_color=(80, 120, 80),
            )
        )

    screenshot = generator.generate(
        width=width, height=height, background_color=(200, 200, 200), elements=elements
    )

    return screenshot, elements


def create_login_form_screenshot(
    width: int = 800, height: int = 600
) -> tuple[np.ndarray, list[ElementSpec]]:
    """
    Create a screenshot with a login form.

    Useful for testing login state detection.

    Args:
        width: Screenshot width
        height: Screenshot height

    Returns:
        Tuple of (screenshot, list of ElementSpec for form elements)

    Example:
        >>> screenshot, elements = create_login_form_screenshot()
        >>> # Use to test login state detection
        >>> assert len(elements) >= 5  # Labels, inputs, button
    """
    generator = SyntheticScreenshotGenerator()

    form_x = 250
    form_y = 150

    elements = [
        # Form background
        ElementSpec(
            "rectangle",
            x=form_x,
            y=form_y,
            width=300,
            height=300,
            color=(255, 255, 255),
            border_color=(150, 150, 150),
            border_width=1,
        ),
        # Title
        ElementSpec("text", x=form_x + 100, y=form_y + 30, text="Login", text_color=(50, 50, 50)),
        # Username label
        ElementSpec(
            "text", x=form_x + 30, y=form_y + 80, text="Username:", text_color=(70, 70, 70)
        ),
        # Username input
        ElementSpec("input", x=form_x + 30, y=form_y + 110, width=240, height=35),
        # Password label
        ElementSpec(
            "text", x=form_x + 30, y=form_y + 160, text="Password:", text_color=(70, 70, 70)
        ),
        # Password input
        ElementSpec("input", x=form_x + 30, y=form_y + 190, width=240, height=35),
        # Login button
        ElementSpec(
            "button",
            x=form_x + 90,
            y=form_y + 245,
            width=120,
            height=40,
            text="Login",
            color=(100, 180, 100),
            border_color=(80, 140, 80),
        ),
    ]

    screenshot = generator.generate(
        width=width, height=height, background_color=(230, 230, 230), elements=elements
    )

    return screenshot, elements


def create_multi_region_screenshot(
    include_toolbar: bool = True,
    include_sidebar: bool = True,
    include_content: bool = True,
    width: int = 1024,
    height: int = 768,
) -> tuple[np.ndarray, dict[str, list[ElementSpec]]]:
    """
    Create a screenshot with multiple distinct regions.

    Useful for testing region analysis and detection.

    Args:
        include_toolbar: Include toolbar region at top
        include_sidebar: Include sidebar region on left
        include_content: Include main content region
        width: Screenshot width
        height: Screenshot height

    Returns:
        Tuple of (screenshot, dict mapping region names to ElementSpecs)

    Example:
        >>> screenshot, regions = create_multi_region_screenshot(
        ...     include_toolbar=True,
        ...     include_sidebar=True,
        ...     include_content=True
        ... )
        >>> assert "toolbar" in regions
        >>> assert "sidebar" in regions
        >>> assert "content" in regions
    """
    generator = SyntheticScreenshotGenerator()

    regions_dict: dict[str, list[ElementSpec]] = {}
    all_elements = []

    # Toolbar region
    if include_toolbar:
        toolbar_elements = [
            ElementSpec(
                "rectangle",
                x=0,
                y=0,
                width=width,
                height=60,
                color=(220, 220, 220),
                border_color=(150, 150, 150),
            ),
            ElementSpec(
                "button", x=10, y=10, width=80, height=40, text="File", color=(200, 200, 200)
            ),
            ElementSpec(
                "button", x=100, y=10, width=80, height=40, text="Edit", color=(200, 200, 200)
            ),
            ElementSpec(
                "button", x=190, y=10, width=80, height=40, text="View", color=(200, 200, 200)
            ),
        ]
        regions_dict["toolbar"] = toolbar_elements
        all_elements.extend(toolbar_elements)

    # Sidebar region
    sidebar_width = 200
    sidebar_y = 60 if include_toolbar else 0
    if include_sidebar:
        sidebar_elements = [
            ElementSpec(
                "rectangle",
                x=0,
                y=sidebar_y,
                width=sidebar_width,
                height=height - sidebar_y,
                color=(240, 240, 240),
                border_color=(180, 180, 180),
            ),
            ElementSpec("text", x=10, y=sidebar_y + 20, text="Navigation", text_color=(70, 70, 70)),
        ]
        # Add navigation items
        for i in range(5):
            sidebar_elements.append(
                ElementSpec(
                    "button",
                    x=10,
                    y=sidebar_y + 60 + i * 50,
                    width=180,
                    height=40,
                    text=f"Item {i+1}",
                    color=(230, 230, 230),
                )
            )
        regions_dict["sidebar"] = sidebar_elements
        all_elements.extend(sidebar_elements)

    # Content region
    content_x = sidebar_width if include_sidebar else 0
    content_y = 60 if include_toolbar else 0
    if include_content:
        content_elements = [
            ElementSpec(
                "rectangle",
                x=content_x,
                y=content_y,
                width=width - content_x,
                height=height - content_y,
                color=(255, 255, 255),
                border_color=(200, 200, 200),
            ),
            ElementSpec(
                "text",
                x=content_x + 20,
                y=content_y + 30,
                text="Main Content Area",
                text_color=(50, 50, 50),
            ),
        ]
        regions_dict["content"] = content_elements
        all_elements.extend(content_elements)

    screenshot = generator.generate(
        width=width, height=height, background_color=(200, 200, 200), elements=all_elements
    )

    return screenshot, regions_dict
