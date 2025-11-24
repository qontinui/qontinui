# OCR Name Generator

Automatically generates meaningful names for GUI states and elements using OCR (Optical Character Recognition).

## Overview

The OCR Name Generator implements Phase 3.2 of the DETECTION_MIGRATION_PLAN, providing intelligent naming strategies for automatically detected GUI elements. It uses a priority-based approach to create meaningful, valid Python identifiers from visual content.

## Features

### Multi-Engine OCR Support
- **pytesseract**: Fast, requires external binary installation
- **easyocr**: Pure Python, includes models (slower but easier to install)
- **Auto-selection**: Automatically uses best available engine

### Naming Strategies (Priority Order)

1. **OCR Text Extraction**
   - Window titles: "Inventory - Player 1" → `inventory_player_1`
   - Button labels: "Save File" → `save_file_button`
   - Headings: "Main Menu" → `main_menu`

2. **Semantic Context**
   - Title bars, buttons, panels, icons, headers
   - Context-aware suffixes: `_button`, `_panel`, `_icon`

3. **Position-Based Fallback**
   - Format: `{context}_{dimensions}_{hash}`
   - Example: `button_200x100_457`

### Text Sanitization
- Converts to lowercase
- Replaces spaces/separators with underscores
- Removes special characters
- Handles numeric prefixes (prepends `n_`)
- Truncates long names at word boundaries
- Ensures valid Python identifiers

### Name Validation
- Python identifier compliance
- Keyword avoidance
- Meaningfulness detection
- Conflict resolution with automatic numbering

## Installation

### Option 1: Tesseract (Recommended for Speed)

```bash
# Install tesseract binary
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki

# Install Python wrapper
pip install pytesseract
```

### Option 2: EasyOCR (Easier Setup)

```bash
# Pure Python installation (includes models)
pip install easyocr
```

### Dependencies

```bash
pip install numpy opencv-python
```

## Quick Start

### Basic Usage

```python
from qontinui.discovery.state_construction import OCRNameGenerator

# Create generator (auto-selects best engine)
generator = OCRNameGenerator(engine='auto')

# Generate name from button image
button_name = generator.generate_name_from_image(
    button_image,
    context='button'
)
print(button_name)  # 'save_file_button'

# Generate state name from screenshot
state_name = generator.generate_state_name(screenshot)
print(state_name)  # 'main_menu'
```

### Convenience Functions

```python
from qontinui.discovery.state_construction import (
    generate_element_name,
    generate_state_name_from_screenshot
)

# One-line element naming
name = generate_element_name(image, 'button')

# One-line state naming
state = generate_state_name_from_screenshot(screenshot)
```

## API Reference

### OCRNameGenerator

#### Constructor

```python
OCRNameGenerator(engine: str = 'auto')
```

**Parameters:**
- `engine`: OCR engine to use
  - `'auto'`: Try easyocr, then tesseract (recommended)
  - `'tesseract'`: Use pytesseract
  - `'easyocr'`: Use easyocr

**Raises:**
- `ValueError`: If no OCR engine is available

#### generate_name_from_image()

```python
generate_name_from_image(
    image: np.ndarray,
    context: str = 'generic'
) -> str
```

Generate name from an image region.

**Parameters:**
- `image`: Image as numpy array (BGR or grayscale)
- `context`: Element type hint
  - `'button'`, `'icon'`, `'panel'`, `'title_bar'`, `'header'`, etc.

**Returns:**
- Valid Python identifier

**Example:**
```python
button = cv2.imread('save_button.png')
name = generator.generate_name_from_image(button, 'button')
# Returns: 'save_button' or 'button_120x50_789' (fallback)
```

#### generate_state_name()

```python
generate_state_name(
    screenshot: np.ndarray,
    detected_text_regions: Optional[List[Dict]] = None
) -> str
```

Generate name for a state from its screenshot.

**Parameters:**
- `screenshot`: Full screen capture
- `detected_text_regions`: Optional pre-detected text regions
  - Each dict should have: `text`, `x`, `y`, `width`, `height`, `area`

**Returns:**
- State name as valid identifier

**Strategies (in order):**
1. Title bar text (top 10%)
2. Prominent headings (large text near top)
3. Pre-detected text regions
4. Hash-based fallback

**Example:**
```python
screenshot = capture_screen()
name = generator.generate_state_name(screenshot)
# Returns: 'inventory_screen' or 'state_800x600_1234' (fallback)

# With pre-detected regions
regions = [
    {'text': 'Main Menu', 'x': 100, 'y': 50, 'width': 200, 'height': 40, 'area': 8000}
]
name = generator.generate_state_name(screenshot, regions)
# Returns: 'main_menu'
```

### NameValidator

#### is_valid_identifier()

```python
@staticmethod
is_valid_identifier(name: str) -> bool
```

Check if name is a valid Python identifier.

**Example:**
```python
NameValidator.is_valid_identifier('valid_name')  # True
NameValidator.is_valid_identifier('123invalid')  # False
NameValidator.is_valid_identifier('class')       # False (keyword)
```

#### is_meaningful()

```python
@staticmethod
is_meaningful(name: str, min_length: int = 3) -> bool
```

Check if name is meaningful (not just hash/position).

**Example:**
```python
NameValidator.is_meaningful('save_button')    # True
NameValidator.is_meaningful('element_12345')  # False (fallback pattern)
NameValidator.is_meaningful('ab')             # False (too short)
```

#### suggest_alternative()

```python
@staticmethod
suggest_alternative(name: str, existing_names: Set[str]) -> str
```

Suggest alternative name to avoid conflicts.

**Example:**
```python
existing = {'button', 'button_2'}
NameValidator.suggest_alternative('button', existing)
# Returns: 'button_3'
```

## Advanced Usage

### Batch Processing with Uniqueness

```python
generator = OCRNameGenerator()
existing_names = set()

for element_image in element_images:
    # Generate name
    name = generator.generate_name_from_image(element_image, 'button')

    # Ensure uniqueness
    unique_name = NameValidator.suggest_alternative(name, existing_names)
    existing_names.add(unique_name)

    print(f"Generated: {unique_name}")
```

### Custom Context Types

```python
contexts = {
    'top_bar': 'title_bar',
    'side_panel': 'sidebar',
    'action_button': 'button',
    'icon_small': 'icon'
}

for region, context_type in zip(regions, contexts.values()):
    name = generator.generate_name_from_image(region.image, context_type)
```

### Pre-Processing for Better OCR

```python
import cv2

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding for better contrast
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Generate name from enhanced image
name = generator.generate_name_from_image(binary, 'text')
```

### Using with Pre-Detected Text Regions

```python
# From another detector (e.g., OCRProcessor)
from qontinui.semantic.processors import OCRProcessor

ocr_processor = OCRProcessor()
scene = ocr_processor.process(screenshot)

# Extract text regions
text_regions = [
    {
        'text': obj.text,
        'x': obj.location.x,
        'y': obj.location.y,
        'width': obj.location.width,
        'height': obj.location.height,
        'area': obj.location.width * obj.location.height
    }
    for obj in scene.objects
    if obj.object_type.value == 'text'
]

# Use for state naming
state_name = generator.generate_state_name(screenshot, text_regions)
```

## Best Practices

### Image Quality
- **Resolution**: Minimum 100x100 pixels for reliable OCR
- **Contrast**: High contrast improves accuracy
- **Preprocessing**: Apply grayscale, thresholding if needed

### Context Selection
- Use specific contexts: `'button'` over `'element'`
- Consistent naming: Same context for similar elements
- Hierarchical contexts: `'main_button'`, `'toolbar_button'`

### Performance
- **EasyOCR**: Slower first run (loads models), fast after
- **Tesseract**: Consistently fast if binary installed
- **Caching**: Cache generated names for repeated elements

### Error Handling

```python
try:
    generator = OCRNameGenerator()
    name = generator.generate_name_from_image(image, 'button')
except ValueError as e:
    # No OCR engine available
    print(f"OCR unavailable: {e}")
    # Use fallback naming strategy
    name = f"button_{image.shape[1]}x{image.shape[0]}"
```

## Examples

See `examples/ocr_name_generator_demo.py` for comprehensive examples:
- Basic usage
- Text sanitization
- Fallback strategies
- Name validation
- Conflict resolution
- Batch processing

## Testing

Run the test suite:

```bash
cd qontinui
pytest tests/discovery/state_construction/test_ocr_name_generator.py -v
```

## Integration with State Construction

The OCR Name Generator is designed to integrate with the StateBuilder:

```python
from qontinui.discovery.state_construction import StateBuilder

# StateBuilder uses OCRNameGenerator internally
builder = StateBuilder()
state = builder.build_state_from_screenshots(
    screenshot_sequence=screenshots,
    transitions_to_state=transitions
)

# All elements and regions have meaningful names
print(f"State: {state.name}")
for img in state.state_images:
    print(f"  Image: {img.name}")
```

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'pytesseract'`

**Solution:**
```bash
pip install pytesseract
# OR
pip install easyocr
```

### Tesseract Not Found

**Problem:** `TesseractNotFoundError`

**Solution:**
Install tesseract binary:
```bash
# macOS
brew install tesseract

# Ubuntu
sudo apt-get install tesseract-ocr
```

### Poor OCR Quality

**Problem:** Nonsensical names or empty text

**Solutions:**
1. Improve image quality (resolution, contrast)
2. Apply preprocessing (grayscale, threshold)
3. Use context-based fallback (automatic)
4. Try different OCR engine

### Memory Issues with EasyOCR

**Problem:** High memory usage

**Solution:**
Use tesseract for lightweight deployment:
```python
generator = OCRNameGenerator(engine='tesseract')
```

## Performance Characteristics

| Engine     | First Run | Subsequent | Accuracy | Memory | Installation |
|------------|-----------|------------|----------|--------|--------------|
| Tesseract  | Fast      | Fast       | Good     | Low    | External     |
| EasyOCR    | Slow      | Fast       | Better   | High   | Pure Python  |

## Limitations

1. **Language Support**: Currently English-only
2. **Text Orientation**: Best with horizontal text
3. **Font Variations**: May struggle with decorative fonts
4. **Image Quality**: Requires reasonable resolution (>100px)
5. **Contextual Understanding**: Cannot understand semantic meaning

## Future Enhancements

- [ ] Multi-language support
- [ ] Custom context templates
- [ ] ML-based name suggestion
- [ ] Integration with element classifiers
- [ ] Name quality scoring
- [ ] Interactive name refinement

## Related Modules

- `state_builder.py`: Uses OCRNameGenerator for state construction
- `element_identifier.py`: Classifies elements for context hints
- `semantic/processors/ocr_processor.py`: Lower-level OCR processing

## References

- [DETECTION_MIGRATION_PLAN.md](../../../DETECTION_MIGRATION_PLAN.md) - Phase 3.2
- [pytesseract documentation](https://github.com/madmaze/pytesseract)
- [EasyOCR documentation](https://github.com/JaidedAI/EasyOCR)
