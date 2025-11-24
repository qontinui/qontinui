# Element Detection Usage Guide

## Quick Start

The element detection module provides specialized detectors for identifying UI components in screenshots. All detectors inherit from `BaseAnalyzer` and follow a consistent interface.

## Installation Requirements

```bash
pip install opencv-python numpy pillow
```

## Basic Usage Pattern

```python
from qontinui.discovery.element_detection import (
    InputFieldDetector,
    AnalysisInput,
)
from uuid import uuid4

# 1. Create detector instance
detector = InputFieldDetector()

# 2. Prepare input data
input_data = AnalysisInput(
    annotation_set_id=uuid4(),
    screenshots=[{"id": "screen1", "timestamp": "2024-11-24"}],
    screenshot_data=[image_bytes],  # List of bytes
    parameters={}  # Optional: override default parameters
)

# 3. Run analysis (async)
result = await detector.analyze(input_data)

# 4. Process results
for element in result.elements:
    bbox = element.bounding_box
    print(f"Found {element.label} at ({bbox.x}, {bbox.y}, {bbox.width}, {bbox.height})")
    print(f"Confidence: {element.confidence:.2f}")
    print(f"Metadata: {element.metadata}")
```

## Available Detectors

### Button Detectors

#### ButtonColorDetector
Detects buttons based on color characteristics and common button color patterns.

```python
from qontinui.discovery.element_detection import ButtonColorDetector

detector = ButtonColorDetector()
params = {
    "n_colors": 8,  # Number of color clusters
    "min_size": 50,  # Minimum button size (pixels)
    "max_size": 300,
}
```

#### ButtonShapeDetector
Detects buttons by analyzing rectangular shapes and rounded corners.

```python
from qontinui.discovery.element_detection import ButtonShapeDetector

detector = ButtonShapeDetector()
params = {
    "min_width": 60,
    "max_width": 300,
    "min_height": 30,
    "max_height": 60,
}
```

#### ButtonShadowDetector
Detects buttons by identifying shadow patterns and elevation effects.

```python
from qontinui.discovery.element_detection import ButtonShadowDetector

detector = ButtonShadowDetector()
params = {
    "shadow_threshold": 30,
    "min_shadow_size": 5,
}
```

#### ButtonHoverDetector
Detects buttons using multi-screenshot hover state analysis.

```python
from qontinui.discovery.element_detection import ButtonHoverDetector

# Requires 2+ screenshots (hover states)
detector = ButtonHoverDetector()
```

#### ButtonEnsembleDetector
Combines multiple button detection strategies using ensemble methods.

```python
from qontinui.discovery.element_detection import ButtonEnsembleDetector

detector = ButtonEnsembleDetector()
# Automatically combines color, shape, and shadow detectors
```

#### ButtonFusionDetector
Fuses button detection results from multiple methods.

```python
from qontinui.discovery.element_detection import ButtonFusionDetector

detector = ButtonFusionDetector()
# Uses weighted fusion of detection strategies
```

### UI Component Detectors

#### InputFieldDetector
Detects text input fields using shape analysis and color characteristics.

```python
from qontinui.discovery.element_detection import InputFieldDetector

detector = InputFieldDetector()
params = {
    "min_aspect_ratio": 3.0,  # Width/height ratio
    "max_aspect_ratio": 15.0,
    "min_width": 100,
    "max_width": 600,
    "min_height": 20,
    "max_height": 60,
    "light_bg_threshold": 200,  # For detecting light backgrounds
}
```

#### IconButtonDetector
Detects icon-based buttons (small buttons with icons/symbols).

```python
from qontinui.discovery.element_detection import IconButtonDetector

detector = IconButtonDetector()
params = {
    "min_size": 20,
    "max_size": 64,
    "icon_detection": True,
}
```

#### DropdownDetector
Detects dropdown menu components.

```python
from qontinui.discovery.element_detection import DropdownDetector

detector = DropdownDetector()
params = {
    "arrow_detection": True,  # Look for dropdown arrows
    "min_width": 100,
}
```

#### ModalDialogDetector
Detects modal dialogs and popup windows.

```python
from qontinui.discovery.element_detection import ModalDialogDetector

detector = ModalDialogDetector()
params = {
    "min_width": 200,
    "min_height": 150,
    "centered_detection": True,  # Expect centered modals
}
```

#### SidebarDetector
Detects sidebar navigation components.

```python
from qontinui.discovery.element_detection import SidebarDetector

detector = SidebarDetector()
params = {
    "side": "left",  # or "right"
    "min_height_ratio": 0.5,  # Minimum height relative to screen
}
```

#### MenuBarDetector
Detects menu bar components.

```python
from qontinui.discovery.element_detection import MenuBarDetector

detector = MenuBarDetector()
params = {
    "position": "top",  # Expected menu position
    "min_items": 2,  # Minimum menu items
}
```

#### TypographyDetector
Detects and analyzes text/typography elements.

```python
from qontinui.discovery.element_detection import TypographyDetector

detector = TypographyDetector()
params = {
    "font_size_detection": True,
    "text_hierarchy": True,  # Detect headers, body text, etc.
}
```

## Understanding Results

### AnalysisResult Structure

```python
@dataclass
class AnalysisResult:
    analyzer_type: AnalysisType  # SINGLE_SHOT, STABLE_REGION, etc.
    analyzer_name: str           # Name of the detector
    elements: List[DetectedElement]  # List of detected elements
    confidence: float            # Overall confidence (0.0-1.0)
    metadata: Dict[str, Any]     # Additional information
```

### DetectedElement Structure

```python
@dataclass
class DetectedElement:
    bounding_box: BoundingBox    # Location and size
    confidence: float            # Detection confidence (0.0-1.0)
    label: Optional[str]         # Human-readable label
    element_type: Optional[str]  # button, input, image, etc.
    screenshot_index: int        # Which screenshot (for multi-shot)
    metadata: Dict[str, Any]     # Detector-specific data
```

### BoundingBox Structure

```python
@dataclass
class BoundingBox:
    x: int      # Left coordinate
    y: int      # Top coordinate
    width: int  # Width in pixels
    height: int # Height in pixels

    # Utility methods
    def iou(self, other: BoundingBox) -> float: ...
    def overlaps(self, other: BoundingBox, threshold: float = 0.5) -> bool: ...
```

## Advanced Usage

### Custom Parameters

Each detector has default parameters that can be overridden:

```python
# Get default parameters
defaults = detector.get_default_parameters()
print(defaults)

# Override specific parameters
custom_params = {
    **defaults,
    "min_width": 150,
    "confidence_threshold": 0.8,
}

input_data = AnalysisInput(
    annotation_set_id=uuid4(),
    screenshots=screenshots,
    screenshot_data=screenshot_bytes,
    parameters=custom_params,
)

result = await detector.analyze(input_data)
```

### Multi-Screenshot Detection

Some detectors support multiple screenshots for improved accuracy:

```python
from qontinui.discovery.element_detection import ButtonHoverDetector

detector = ButtonHoverDetector()

# Provide multiple screenshots
input_data = AnalysisInput(
    annotation_set_id=uuid4(),
    screenshots=[
        {"id": "screen1", "state": "normal"},
        {"id": "screen2", "state": "hover"},
    ],
    screenshot_data=[bytes1, bytes2],
    parameters={},
)

result = await detector.analyze(input_data)

# Elements include screenshot_index to identify source
for element in result.elements:
    print(f"Found in screenshot {element.screenshot_index}")
```

### Filtering Results

```python
# Filter by confidence threshold
high_confidence = [
    elem for elem in result.elements
    if elem.confidence >= 0.8
]

# Filter by size
large_elements = [
    elem for elem in result.elements
    if elem.bounding_box.width > 100
]

# Filter by location
top_half_elements = [
    elem for elem in result.elements
    if elem.bounding_box.y < screen_height / 2
]
```

### Combining Multiple Detectors

```python
from qontinui.discovery.element_detection import (
    InputFieldDetector,
    ButtonColorDetector,
    DropdownDetector,
)

# Create multiple detectors
detectors = [
    InputFieldDetector(),
    ButtonColorDetector(),
    DropdownDetector(),
]

# Run all detectors
all_elements = []
for detector in detectors:
    result = await detector.analyze(input_data)
    all_elements.extend(result.elements)

# Remove overlapping detections
from .analysis_base import BoundingBox

def remove_overlaps(elements, iou_threshold=0.5):
    """Remove overlapping detections, keeping highest confidence"""
    filtered = []
    elements = sorted(elements, key=lambda e: e.confidence, reverse=True)

    for elem in elements:
        overlaps = any(
            elem.bounding_box.overlaps(other.bounding_box, iou_threshold)
            for other in filtered
        )
        if not overlaps:
            filtered.append(elem)

    return filtered

unique_elements = remove_overlaps(all_elements)
```

## Error Handling

```python
try:
    result = await detector.analyze(input_data)
except ValueError as e:
    print(f"Invalid input data: {e}")
except Exception as e:
    print(f"Detection failed: {e}")
```

## Best Practices

1. **Use appropriate detectors** - Choose specialized detectors for specific UI elements
2. **Tune parameters** - Adjust parameters based on your specific UI
3. **Combine detectors** - Use ensemble methods for better accuracy
4. **Filter results** - Apply confidence thresholds to reduce false positives
5. **Multi-screenshot analysis** - Provide multiple screenshots when available for better detection
6. **Validate results** - Always check confidence scores and bounding boxes

## Performance Tips

1. **Process images in batches** when analyzing multiple screenshots
2. **Cache detector instances** instead of creating new ones
3. **Resize large images** before processing to improve speed
4. **Use appropriate parameters** - larger search ranges increase computation time
5. **Consider memory usage** when processing many screenshots

## Troubleshooting

### Low Detection Rates
- Adjust confidence thresholds
- Tune size parameters (min/max width/height)
- Try different detectors or ensemble methods
- Ensure image quality is sufficient

### False Positives
- Increase confidence threshold
- Use more restrictive size constraints
- Combine with other detectors
- Filter by location or other metadata

### Performance Issues
- Reduce image resolution
- Limit parameter search ranges
- Use faster detectors for initial pass
- Process screenshots in parallel

## Examples

See `/Users/jspinak/Documents/qontinui/qontinui/src/qontinui/discovery/example_detector_usage.py` for complete working examples.

## API Reference

For detailed API documentation, refer to the docstrings in each detector class and the base classes in `analysis_base.py`.
