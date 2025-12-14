# Self-Healing Locators - Quick Start Guide

## 5-Minute Setup

### 1. Install (if needed)
```bash
# OCR support (optional, for SemanticTextStrategy)
pip install pytesseract
```

### 2. Basic Usage

```python
from qontinui.locators import HealingManager
from qontinui.locators.strategies import ScreenContext
from qontinui.model.element import Pattern
import numpy as np
import time

# Load your pattern
pattern = Pattern.from_file("button.png")

# Capture screenshot (replace with your capture method)
screenshot = np.array(...)  # BGR format, numpy array

# Create context
context = ScreenContext(screenshot=screenshot, timestamp=time.time())

# Create healing manager
manager = HealingManager()

# Find with automatic healing
result = manager.find_with_healing(pattern, context, min_confidence=0.8)

if result.found:
    region = result.match_result.region
    print(f"Found at ({region.x}, {region.y})")

    if result.metadata['healed']:
        print(f"Used fallback strategy: {result.successful_strategy}")
    else:
        print("Found with primary strategy (no healing needed)")
else:
    print("Not found even after trying all strategies")
```

## Common Scenarios

### Scenario 1: Drop-In Replacement for Actions

```python
from qontinui.locators.integration_example import SelfHealingActions

# Replace:
# actions = Actions()

# With:
actions = SelfHealingActions()

# Use exactly as before
result = actions.find(pattern)
```

### Scenario 2: Gradual Adoption

```python
from qontinui.actions import Actions
from qontinui.locators import HealingManager

actions = Actions()
healing_manager = HealingManager()

# Try standard first
result = actions.find(pattern)

if not result.success:
    # Only use healing if standard fails
    healing_result = healing_manager.find_with_healing(pattern, context)
```

### Scenario 3: Aggressive Healing

```python
from qontinui.locators import HealingManager

# Auto-updates patterns when healed
manager = HealingManager.create_aggressive()

result = manager.find_with_healing(pattern, context)

if result.metadata.get('updated_pattern'):
    print("Pattern was updated - save it!")
    # pattern.save() or pattern.to_database() etc.
```

### Scenario 4: Conservative Healing

```python
from qontinui.locators import HealingManager

# Higher threshold, no updates, fewer strategies
manager = HealingManager.create_conservative()

result = manager.find_with_healing(pattern, context)
# Pattern never updated, but healing still attempted
```

### Scenario 5: Custom Strategy Set

```python
from qontinui.locators import HealingConfig, HealingManager

# Only visual and text strategies
config = HealingConfig(
    fallback_strategies=["visual", "text"]
)

manager = HealingManager(config)
result = manager.find_with_healing(pattern, context)
```

### Scenario 6: Monitoring & Stats

```python
from qontinui.locators import HealingManager

manager = HealingManager()

# Find multiple patterns...
for pattern in patterns:
    manager.find_with_healing(pattern, context)

# Get statistics
stats = manager.get_healing_stats()

print(f"Healing rate: {stats['healing_rate']:.1%}")
print(f"Total attempts: {stats['total_attempts']}")
print(f"Successes: {stats['healing_successes']}")
print(f"Patterns updated: {stats['patterns_updated']}")
```

## Configuration Presets

### Default (Balanced)
```python
manager = HealingManager()
# auto_heal=True, threshold=0.7, update_on_heal=False
```

### Aggressive
```python
manager = HealingManager.create_aggressive()
# auto_heal=True, threshold=0.6, update_on_heal=True
```

### Conservative
```python
manager = HealingManager.create_conservative()
# auto_heal=True, threshold=0.85, update_on_heal=False
```

### Custom
```python
from qontinui.locators import HealingConfig, HealingManager

config = HealingConfig(
    auto_heal=True,                      # Enable healing
    confidence_threshold=0.75,            # Balanced threshold
    update_on_heal=False,                 # Don't update patterns
    max_healing_attempts=3,               # Try up to 3 strategies
    learn_successful_strategies=True,     # Learn what works
    fallback_strategies=["visual", "text", "color"]  # Custom set
)

manager = HealingManager(config)
```

## Understanding Results

### Result Properties
```python
result = manager.find_with_healing(pattern, context)

# Did we find the element?
if result.found:
    # What strategy worked?
    strategy = result.successful_strategy  # "VisualPattern", "SemanticText", etc.

    # How confident are we?
    confidence = result.confidence  # 0.0-1.0

    # Where is it?
    region = result.match_result.region
    location = result.match_result.to_location()

    # Was healing used?
    healed = result.metadata['healed']  # True if non-primary strategy used

    # How many strategies were tried?
    attempts = len(result.attempts)

    # How long did it take?
    duration = result.total_duration  # seconds
```

### Interpreting Results

```python
if result.found:
    if not result.metadata['healed']:
        # Best case: primary strategy worked
        print("✓ Found immediately")
    elif result.successful_strategy == "SemanticText":
        # Element appearance changed, but text stayed same
        print("! Element appearance changed - consider updating pattern")
    elif result.successful_strategy == "RelativePosition":
        # Element moved, but anchor stayed stable
        print("! Element position changed - verify result")
    elif result.successful_strategy == "ColorRegion":
        # Found by color - may need verification
        print("! Found by color - verify it's the right element")
    elif result.successful_strategy == "Structural":
        # Found by shape - high chance of false positive
        print("! Found by structure - HIGH PRIORITY: verify result")
```

## Troubleshooting

### Problem: Always fails to find

**Solution 1**: Lower confidence threshold
```python
config = HealingConfig(confidence_threshold=0.6)
manager = HealingManager(config)
```

**Solution 2**: Add more strategies
```python
config = HealingConfig(
    fallback_strategies=["visual", "text", "relative", "color", "structural"]
)
```

**Solution 3**: Enable debug logging
```python
import logging
logging.getLogger("qontinui.locators").setLevel(logging.DEBUG)
```

### Problem: Too many false positives

**Solution 1**: Increase confidence threshold
```python
config = HealingConfig(confidence_threshold=0.9)
```

**Solution 2**: Use fewer strategies
```python
config = HealingConfig(
    fallback_strategies=["visual", "text"]  # Skip structural
)
```

**Solution 3**: Disable structural strategy
```python
# Structural is most prone to false positives
config = HealingConfig(
    fallback_strategies=["visual", "text", "relative", "color"]
)
```

### Problem: Too slow

**Solution 1**: Use fewer strategies
```python
config = HealingConfig(
    fallback_strategies=["visual", "text"]  # Skip slower strategies
)
```

**Solution 2**: Increase threshold to fail faster
```python
config = HealingConfig(confidence_threshold=0.85)
```

**Solution 3**: Reduce max attempts
```python
config = HealingConfig(max_healing_attempts=2)
```

## Best Practices

### 1. Start Simple
```python
# Begin with default config
manager = HealingManager()

# Monitor healing rate
stats = manager.get_healing_stats()
if stats['healing_rate'] > 0.3:
    print("High healing rate - may need pattern updates")
```

### 2. Use Appropriate Strategies
```python
# Text-heavy UI
config = HealingConfig(fallback_strategies=["visual", "text"])

# Colorful UI
config = HealingConfig(fallback_strategies=["visual", "color"])

# Stable layouts
config = HealingConfig(fallback_strategies=["visual", "relative"])
```

### 3. Monitor and Adapt
```python
manager = HealingManager()

# After some time...
stats = manager.get_healing_stats()

if stats['healing_rate'] > 0.5:
    # Over 50% healing - patterns are outdated
    print("URGENT: Update patterns or enable update_on_heal")

preferences = manager.get_strategy_preferences(pattern.id)
if preferences.get('SemanticText', 0) > 5:
    # Text strategy worked 5+ times for this pattern
    print("Consider switching to text-based primary strategy")
```

### 4. Pattern Updates
```python
# Start conservative
manager = HealingManager.create_conservative()

# Monitor for a while...

if stats['healing_rate'] > 0.3 and stats['healing_successes'] > 50:
    # High healing rate, many successes - safe to enable updates
    manager = HealingManager.create_aggressive()
```

## Next Steps

1. **Read full documentation**: See `README.md` for complete guide
2. **Check examples**: See `integration_example.py` for more scenarios
3. **Run tests**: See `test_locators.py` for test examples
4. **Customize**: Create your own strategies or configs
5. **Monitor**: Track healing stats and adjust thresholds

## Quick Reference

### Imports
```python
from qontinui.locators import (
    HealingManager,
    HealingConfig,
    MultiStrategyLocator,
)
from qontinui.locators.strategies import ScreenContext
```

### Create Manager
```python
manager = HealingManager()                    # Default
manager = HealingManager.create_aggressive()  # Aggressive
manager = HealingManager.create_conservative()  # Conservative
```

### Find with Healing
```python
result = manager.find_with_healing(pattern, context, min_confidence=0.8)
```

### Check Results
```python
if result.found:
    region = result.match_result.region
    healed = result.metadata['healed']
    strategy = result.successful_strategy
```

### Get Stats
```python
stats = manager.get_healing_stats()
healing_rate = stats['healing_rate']
```

## Support

- **Documentation**: See `README.md`
- **Examples**: See `integration_example.py`
- **Tests**: See `test_locators.py`
- **Source**: See `strategies.py`, `multi_strategy.py`, `healing.py`
