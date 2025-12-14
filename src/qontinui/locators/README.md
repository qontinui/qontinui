# Self-Healing Locator System

A multi-strategy locator system for qontinui that reduces test brittleness by providing multiple methods to find UI elements and automatically recovering when primary locators fail.

## Overview

Traditional test automation tools break when UI elements change slightly. The self-healing locator system solves this by:

1. **Multiple Finding Strategies**: Uses 5 different strategies to locate elements
2. **Automatic Fallback**: Tries alternatives when primary strategy fails
3. **Learning**: Tracks which strategies work for each pattern
4. **Self-Healing**: Optionally updates patterns after successful healing
5. **Transparency**: Reports which strategy found the element

## Architecture

```
┌─────────────────────────────────────────────┐
│          HealingManager                     │
│  ┌───────────────────────────────────────┐ │
│  │   MultiStrategyLocator                │ │
│  │  ┌─────────────────────────────────┐ │ │
│  │  │  Strategy 1: VisualPattern      │ │ │
│  │  │  Strategy 2: SemanticText       │ │ │
│  │  │  Strategy 3: RelativePosition   │ │ │
│  │  │  Strategy 4: ColorRegion        │ │ │
│  │  │  Strategy 5: Structural         │ │ │
│  │  └─────────────────────────────────┘ │ │
│  └───────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

## Components

### 1. Locator Strategies

#### VisualPatternStrategy
Template matching using OpenCV (primary strategy)
- **Use when**: Element has stable visual appearance
- **Pros**: Fast, accurate for unchanged elements
- **Cons**: Brittle when element changes

#### SemanticTextStrategy
Finds elements by text content using OCR
- **Use when**: Element contains unique text
- **Pros**: Resilient to visual changes
- **Cons**: Requires pytesseract, slower

#### RelativePositionStrategy
Finds elements relative to stable anchor elements
- **Use when**: Element position relative to another is stable
- **Pros**: Good for dynamic content
- **Cons**: Requires stable anchor

#### ColorRegionStrategy
Finds elements by color patterns
- **Use when**: Element has unique color
- **Pros**: Resilient to shape changes
- **Cons**: Sensitive to theme changes

#### StructuralStrategy
Finds elements by visual structure (edges, contours)
- **Use when**: Element type is known (button, input, etc.)
- **Pros**: Very resilient
- **Cons**: May find wrong element

### 2. MultiStrategyLocator

Orchestrates multiple strategies, trying each in sequence until one succeeds.

```python
from qontinui.locators import MultiStrategyLocator

# Create with all strategies
locator = MultiStrategyLocator.create_default()

# Or custom set
locator = MultiStrategyLocator.create_with_strategies("visual", "text", "relative")

# Find element
result = locator.find(pattern, context, min_confidence=0.8)

if result.found:
    print(f"Found using {result.successful_strategy}")
```

### 3. HealingManager

Manages self-healing, learning, and pattern updates.

```python
from qontinui.locators import HealingManager, HealingConfig

# Create with configuration
config = HealingConfig(
    auto_heal=True,
    confidence_threshold=0.7,
    update_on_heal=True,
    max_healing_attempts=5,
)

manager = HealingManager(config)

# Find with healing
result = manager.find_with_healing(pattern, context)

if result.found:
    if result.metadata['healed']:
        print(f"Healed using {result.successful_strategy}")
    else:
        print("Found with primary strategy")
```

## Configuration

### HealingConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto_heal` | bool | True | Enable automatic healing |
| `confidence_threshold` | float | 0.7 | Minimum confidence to accept |
| `update_on_heal` | bool | False | Update pattern after healing |
| `max_healing_attempts` | int | 5 | Max alternative strategies |
| `learn_successful_strategies` | bool | True | Track successful strategies |
| `emit_events` | bool | True | Emit healing events |
| `fallback_strategies` | list[str] | All | Strategy names to use |

## Usage Examples

### Basic Usage

```python
from qontinui.locators import HealingManager, HealingConfig
from qontinui.model.element import Pattern
from qontinui.locators.strategies import ScreenContext
import numpy as np

# Load pattern
pattern = Pattern.from_file("button.png")

# Create healing manager
manager = HealingManager()

# Capture screenshot (from your capture system)
screenshot = np.array(...)  # Your screenshot as numpy array

# Create context
context = ScreenContext(screenshot=screenshot, timestamp=time.time())

# Find with healing
result = manager.find_with_healing(pattern, context, min_confidence=0.8)

if result.found:
    region = result.match_result.region
    print(f"Found at ({region.x}, {region.y})")

    if result.metadata['healed']:
        print(f"Used healing strategy: {result.successful_strategy}")
```

### Aggressive Healing (Updates Patterns)

```python
# Create aggressive config
manager = HealingManager.create_aggressive()

result = manager.find_with_healing(pattern, context)

# Pattern will be updated if healed
if result.metadata.get('updated_pattern'):
    print("Pattern updated for future use")
```

### Conservative Healing (No Updates)

```python
# Create conservative config
manager = HealingManager.create_conservative()

result = manager.find_with_healing(pattern, context)

# Pattern NOT updated, but healing still attempted
```

### Custom Strategy Set

```python
config = HealingConfig(
    fallback_strategies=["visual", "text", "color"]  # Only these 3
)

manager = HealingManager(config)
result = manager.find_with_healing(pattern, context)
```

### Getting Statistics

```python
manager = HealingManager()

# Perform multiple finds...

# Get overall stats
stats = manager.get_healing_stats()
print(f"Healing rate: {stats['healing_rate']:.2%}")
print(f"Patterns updated: {stats['patterns_updated']}")

# Get strategy preferences for a pattern
preferences = manager.get_strategy_preferences(pattern.id)
print(f"Best strategy for this pattern: {max(preferences, key=preferences.get)}")
```

## Integration with Actions

### Drop-in Replacement

```python
from qontinui.locators.integration_example import SelfHealingActions

# Use like standard Actions
actions = SelfHealingActions()

result = actions.find(pattern)

if result.success:
    # Check if healing was used
    if result.metadata.get('healed'):
        print("Element found via healing")
```

### Gradual Adoption

```python
from qontinui.actions import Actions
from qontinui.locators import HealingManager

actions = Actions()
healing_manager = HealingManager()

# Try standard first
result = actions.find(pattern)

if not result.success:
    # Fall back to healing
    healing_result = healing_manager.find_with_healing(pattern, context)
    # Use healing_result...
```

## Advanced Features

### Pattern Update Callbacks

```python
def on_pattern_updated(pattern, new_pixel_data):
    print(f"Pattern {pattern.id} updated")
    # Save to database, file, etc.

manager = HealingManager()
manager.register_update_callback(pattern.id, on_pattern_updated)
```

### Healing Events

```python
from qontinui.reporting.events import subscribe_to_event, EventType

def on_healing_succeeded(event_data):
    print(f"Healed pattern {event_data['pattern_id']}")
    print(f"Strategy: {event_data['strategy']}")

# Subscribe to events
subscribe_to_event(EventType.DIAGNOSTIC, on_healing_succeeded)

# Events are emitted automatically when healing occurs
```

### Strategy Statistics

```python
# Get per-strategy stats
strategy_stats = locator.get_strategy_stats()

for strategy_name, stats in strategy_stats.items():
    print(f"{strategy_name}:")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print(f"  Avg confidence: {stats['avg_confidence']:.3f}")
    print(f"  Avg duration: {stats['avg_duration']:.3f}s")
```

## Best Practices

### 1. Start Conservative

Begin with conservative healing (no pattern updates) to understand healing behavior:

```python
manager = HealingManager.create_conservative()
```

### 2. Monitor Healing Rate

Track healing statistics to understand when healing is needed:

```python
stats = manager.get_healing_stats()
if stats['healing_rate'] > 0.3:  # >30% healing
    print("High healing rate - consider updating patterns")
```

### 3. Use Appropriate Strategies

Choose strategies based on your UI:

```python
# For text-heavy UIs
config = HealingConfig(fallback_strategies=["visual", "text"])

# For colorful UIs
config = HealingConfig(fallback_strategies=["visual", "color"])

# For stable layouts
config = HealingConfig(fallback_strategies=["visual", "relative"])
```

### 4. Update Patterns Carefully

Only enable `update_on_heal` when you trust the healing process:

```python
# Start without updates
config = HealingConfig(update_on_heal=False)

# Monitor healing for a while...

# Enable updates once confident
config = HealingConfig(update_on_heal=True)
```

### 5. Set Appropriate Thresholds

Balance between false positives and brittleness:

```python
# High precision (fewer false positives)
config = HealingConfig(confidence_threshold=0.9)

# High recall (fewer missed elements)
config = HealingConfig(confidence_threshold=0.6)

# Balanced
config = HealingConfig(confidence_threshold=0.75)
```

## Performance Considerations

### Strategy Order Matters

Strategies are tried in order. Put fastest/most reliable first:

```python
# Visual is fastest - always first
config = HealingConfig(
    fallback_strategies=["visual", "text", "color", "structural"]
)
```

### Stop on First Match

By default, searching stops after first successful match:

```python
# Default behavior (fast)
result = locator.find(pattern, context, stop_on_first=True)

# Try all strategies (slow, for diagnostics)
result = locator.find(pattern, context, stop_on_first=False)
```

### Cache Screenshots

Reuse screenshots when finding multiple elements:

```python
context = ScreenContext(screenshot=screenshot, timestamp=time.time())

# Use same context for multiple finds
result1 = manager.find_with_healing(pattern1, context)
result2 = manager.find_with_healing(pattern2, context)
```

## Troubleshooting

### Healing Always Fails

1. Check confidence threshold is not too high
2. Verify fallback strategies are appropriate
3. Enable debug logging to see strategy attempts

```python
import logging
logging.getLogger("qontinui.locators").setLevel(logging.DEBUG)
```

### Too Many False Positives

1. Increase confidence threshold
2. Use fewer/more specific strategies
3. Disable structural strategy (most prone to false positives)

```python
config = HealingConfig(
    confidence_threshold=0.85,
    fallback_strategies=["visual", "text"]
)
```

### Performance Issues

1. Reduce number of fallback strategies
2. Put fastest strategies first
3. Increase confidence threshold to fail faster

```python
config = HealingConfig(
    confidence_threshold=0.8,
    max_healing_attempts=3,
    fallback_strategies=["visual", "text"]
)
```

## API Reference

See individual module docstrings for detailed API documentation:

- `strategies.py` - Locator strategy implementations
- `multi_strategy.py` - Multi-strategy orchestrator
- `healing.py` - Self-healing manager
- `integration_example.py` - Integration examples

## Testing

```python
# Test with mock screenshot
import numpy as np

screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)
context = ScreenContext(screenshot=screenshot)

pattern = Pattern.from_file("test_pattern.png")

manager = HealingManager()
result = manager.find_with_healing(pattern, context)

assert result.found or not result.found  # Check behavior
```

## Future Enhancements

Potential future improvements:

1. **ML-based Strategy Selection**: Learn optimal strategy order per pattern
2. **Fuzzy Template Matching**: More resilient visual matching
3. **Semantic Element Detection**: AI-powered element classification
4. **Region Proposals**: Generate candidate regions before strategy application
5. **Confidence Calibration**: Learn optimal thresholds per pattern/strategy
6. **Pattern Versioning**: Track pattern evolution over time
7. **A/B Testing**: Compare healing strategies effectiveness

## Contributing

When adding new strategies:

1. Inherit from `LocatorStrategy`
2. Implement `find()`, `get_name()`, `can_handle()`
3. Add to `create_default()` if generally useful
4. Document use cases and limitations
5. Add tests with various input types
