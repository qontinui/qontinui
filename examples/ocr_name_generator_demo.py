"""Demo script for OCR Name Generator.

This script demonstrates how to use the OCR Name Generator to create
meaningful names for GUI elements and states.

Requirements:
    - pytesseract (pip install pytesseract) or
    - easyocr (pip install easyocr)
    - opencv-python (pip install opencv-python)
    - numpy (pip install numpy)

Usage:
    python examples/ocr_name_generator_demo.py
"""

import cv2
import numpy as np

from qontinui.discovery.state_construction import (
    OCRNameGenerator,
    NameValidator,
    generate_element_name,
    generate_state_name_from_screenshot,
)


def create_demo_button_image(text: str = "Save") -> np.ndarray:
    """Create a synthetic button image with text.

    Args:
        text: Button text

    Returns:
        Button image as numpy array
    """
    # Create white background
    img = np.ones((60, 200, 3), dtype=np.uint8) * 255

    # Add button border
    cv2.rectangle(img, (5, 5), (195, 55), (100, 100, 100), 2)

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (200 - text_size[0]) // 2
    text_y = (60 + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, (0, 0, 0), 2)

    return img


def create_demo_screenshot(title: str = "Main Menu") -> np.ndarray:
    """Create a synthetic screenshot with title.

    Args:
        title: Screen title

    Returns:
        Screenshot as numpy array
    """
    # Create gray background
    img = np.ones((600, 800, 3), dtype=np.uint8) * 200

    # Add title bar
    cv2.rectangle(img, (0, 0), (800, 60), (50, 50, 50), -1)

    # Add title text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, title, (20, 40), font, 1.5, (255, 255, 255), 2)

    # Add some UI elements
    cv2.rectangle(img, (50, 100), (750, 200), (255, 255, 255), -1)
    cv2.putText(img, "Content Area", (60, 160), font, 1, (0, 0, 0), 2)

    return img


def demo_basic_usage():
    """Demonstrate basic OCR name generation."""
    print("=" * 70)
    print("Demo 1: Basic OCR Name Generation")
    print("=" * 70)

    try:
        # Create generator
        generator = OCRNameGenerator(engine="auto")
        print(f"Using OCR engine: {generator.engine}\n")

        # Generate name for a button
        button_img = create_demo_button_image("Save File")
        button_name = generator.generate_name_from_image(button_img, context="button")
        print(f"Button image -> name: '{button_name}'")

        # Generate name for a state
        screenshot = create_demo_screenshot("Inventory Screen")
        state_name = generator.generate_state_name(screenshot)
        print(f"Screenshot -> state name: '{state_name}'")

        print()

    except ValueError as e:
        print(f"Error: {e}")
        print("Please install pytesseract or easyocr to run this demo.")


def demo_text_sanitization():
    """Demonstrate text sanitization."""
    print("=" * 70)
    print("Demo 2: Text Sanitization")
    print("=" * 70)

    try:
        generator = OCRNameGenerator(engine="auto")

        test_cases = [
            "Save File!",
            "Player-1 Inventory",
            "Click Me!!!",
            "File/Path",
            "Test:Value",
            "123 Main Street",
            "Hello   World",
            "Café Menu",
        ]

        for text in test_cases:
            sanitized = generator._sanitize_text(text)
            print(f"{text:25} -> {sanitized}")

        print()

    except ValueError as e:
        print(f"Skipping: {e}\n")


def demo_fallback_strategies():
    """Demonstrate fallback naming strategies."""
    print("=" * 70)
    print("Demo 3: Fallback Strategies")
    print("=" * 70)

    try:
        generator = OCRNameGenerator(engine="auto")

        # Blank image (OCR will fail)
        blank = np.zeros((100, 200, 3), dtype=np.uint8)
        fallback_name = generator.generate_name_from_image(blank, context="icon")
        print(f"Blank image -> fallback name: '{fallback_name}'")

        # Random noise (OCR will likely fail)
        noise = np.random.randint(0, 255, (150, 300, 3), dtype=np.uint8)
        noise_name = generator.generate_name_from_image(noise, context="panel")
        print(f"Noise image -> fallback name: '{noise_name}'")

        print()

    except ValueError as e:
        print(f"Skipping: {e}\n")


def demo_name_validation():
    """Demonstrate name validation utilities."""
    print("=" * 70)
    print("Demo 4: Name Validation")
    print("=" * 70)

    test_names = [
        "valid_name",
        "button_123",
        "123invalid",
        "has-dash",
        "class",  # Python keyword
        "element_12345678",
        "state_999",
        "save_button",
    ]

    for name in test_names:
        is_valid = NameValidator.is_valid_identifier(name)
        is_meaningful = NameValidator.is_meaningful(name)
        print(f"{name:20} - Valid: {is_valid:5} Meaningful: {is_meaningful}")

    print()


def demo_conflict_resolution():
    """Demonstrate name conflict resolution."""
    print("=" * 70)
    print("Demo 5: Conflict Resolution")
    print("=" * 70)

    existing_names = {"button", "button_2", "save_button"}

    test_names = ["button", "button_2", "new_button", "save_button"]

    for name in test_names:
        alternative = NameValidator.suggest_alternative(name, existing_names)
        conflict = "conflict" if name != alternative else "no conflict"
        print(f"{name:15} -> {alternative:15} ({conflict})")

    print()


def demo_convenience_functions():
    """Demonstrate convenience functions."""
    print("=" * 70)
    print("Demo 6: Convenience Functions")
    print("=" * 70)

    try:
        # Quick element name generation
        button = create_demo_button_image("Click Me")
        name = generate_element_name(button, "button")
        print(f"Quick element name: '{name}'")

        # Quick state name generation
        screenshot = create_demo_screenshot("Settings")
        state_name = generate_state_name_from_screenshot(screenshot)
        print(f"Quick state name: '{state_name}'")

        print()

    except ValueError as e:
        print(f"Skipping: {e}\n")


def demo_batch_processing():
    """Demonstrate batch processing of multiple elements."""
    print("=" * 70)
    print("Demo 7: Batch Processing")
    print("=" * 70)

    try:
        generator = OCRNameGenerator(engine="auto")
        existing_names = set()

        button_texts = ["Save", "Cancel", "OK", "Apply", "Close"]

        print("Generating unique names for multiple buttons:")
        for text in button_texts:
            img = create_demo_button_image(text)
            name = generator.generate_name_from_image(img, context="button")

            # Ensure uniqueness
            unique_name = NameValidator.suggest_alternative(name, existing_names)
            existing_names.add(unique_name)

            print(f"  {text:10} -> {unique_name}")

        print(f"\nTotal unique names: {len(existing_names)}")
        print()

    except ValueError as e:
        print(f"Skipping: {e}\n")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "OCR Name Generator Demo" + " " * 30 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    # Run all demos
    demo_basic_usage()
    demo_text_sanitization()
    demo_fallback_strategies()
    demo_name_validation()
    demo_conflict_resolution()
    demo_convenience_functions()
    demo_batch_processing()

    print("=" * 70)
    print("Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
