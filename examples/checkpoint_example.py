"""Example usage of the CheckpointService.

This example demonstrates how to use the CheckpointService to capture
screenshots with OCR at key points during automation.

Prerequisites:
    - qontinui library installed
    - Screen capture dependencies (mss)
    - Optional: OCR dependencies (easyocr, opencv-python)
"""

from pathlib import Path

from qontinui.checkpointing import CheckpointService, CheckpointTrigger
from qontinui.hal.implementations import MSSScreenCapture

# Optional: Import OCR engine if available
try:
    from qontinui.hal.implementations import EasyOCREngine

    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("Note: EasyOCR not available, checkpoints will not include OCR data")


def main():
    """Demonstrate checkpoint service usage."""
    # Initialize screen capture
    screen_capture = MSSScreenCapture()

    # Initialize OCR engine if available
    ocr_engine = EasyOCREngine() if HAS_OCR else None

    # Create checkpoint service with custom output directory
    output_dir = Path.home() / "qontinui_checkpoints_demo"
    service = CheckpointService(
        screen_capture=screen_capture,
        ocr_engine=ocr_engine,
        output_dir=output_dir,
    )

    print("Checkpoint service initialized")
    print(f"Output directory: {service.output_dir}")
    print(f"OCR enabled: {service.has_ocr}\n")

    # Example 1: Capture a manual checkpoint
    print("Capturing manual checkpoint...")
    checkpoint1 = service.capture_checkpoint(
        name="example_manual",
        trigger=CheckpointTrigger.MANUAL,
        action_context="user_requested",
        metadata={"example": "manual_checkpoint"},
    )

    print(f"✓ Checkpoint captured: {checkpoint1.name}")
    print(f"  Screenshot: {checkpoint1.screenshot_path}")
    print(f"  Timestamp: {checkpoint1.timestamp}")
    print(f"  Has OCR: {bool(checkpoint1.ocr_text)}")
    if checkpoint1.ocr_text:
        print(f"  OCR text (first 100 chars): {checkpoint1.ocr_text[:100]}")
        print(f"  Text regions found: {checkpoint1.region_count}")

    # Example 2: Capture a specific region
    print("\nCapturing region checkpoint...")
    # Capture a 400x300 region starting at (100, 100)
    checkpoint2 = service.capture_checkpoint(
        name="example_region",
        trigger=CheckpointTrigger.TRANSITION_COMPLETE,
        action_context="after_click",
        region=(100, 100, 400, 300),
    )

    print(f"✓ Region checkpoint captured: {checkpoint2.name}")
    print(f"  Screenshot: {checkpoint2.screenshot_path}")

    # Example 3: Capture without OCR (faster)
    print("\nCapturing checkpoint without OCR...")
    checkpoint3 = service.capture_checkpoint(
        name="example_no_ocr",
        trigger=CheckpointTrigger.TERMINAL_FAILURE,
        action_context="error_recovery",
        run_ocr=False,
    )

    print(f"✓ No-OCR checkpoint captured: {checkpoint3.name}")
    print(f"  Screenshot: {checkpoint3.screenshot_path}")

    # Example 4: Search for text in checkpoint
    if checkpoint1.text_regions:
        print("\nSearching for text in checkpoint...")
        # Search for any region containing "the" (common word)
        matching_regions = checkpoint1.get_regions_containing("the", case_sensitive=False)
        print(f"Found {len(matching_regions)} regions containing 'the'")

        if matching_regions:
            region = matching_regions[0]
            print(f"  First match: '{region.text}' at {region.bounds}")
            print(f"  Confidence: {region.confidence:.2f}")

    # Clean up old checkpoints (older than 1 hour)
    print("\nCleaning up old checkpoints...")
    deleted = service.clear_checkpoints(older_than_hours=1)
    print(f"Deleted {deleted} old checkpoint files")

    print(f"\n✅ Demo complete! Checkpoints saved to: {service.output_dir}")


if __name__ == "__main__":
    main()
