# State Detection

## Overview

The `state_detection` module provides core functionality for identifying application states from screenshots. This is the intelligence layer that enables automated state discovery and recognition.

## Purpose

State detection is critical for understanding application behavior and enabling automated testing by:

- Identifying the current state of an application from visual information
- Matching screenshots against known state signatures
- Detecting state transitions
- Validating state stability over time
- Providing probabilistic state inference
- Supporting multi-frame state validation

## Key Components

### Planned Modules

- **State Matcher**: Match screenshots against known state signatures
- **Transition Detector**: Identify when state changes occur
- **Stability Analyzer**: Determine if a state is stable
- **Signature Extractor**: Extract distinctive features that define a state
- **Probability Engine**: Calculate state match probabilities
- **Multi-frame Validator**: Validate states across multiple frames

## Usage Pattern

```python
from qontinui.discovery.state_detection import StateDetector

# Initialize detector with known states
detector = StateDetector(known_states=state_repository)

# Detect current state
result = detector.detect(screenshot)

print(f"State: {result.state_name}")
print(f"Confidence: {result.confidence}")
print(f"Matches: {result.matching_features}")

# Detect state transition
if detector.is_transition(previous_frame, current_frame):
    transition = detector.get_transition(previous_frame, current_frame)
    print(f"Transition: {transition.from_state} -> {transition.to_state}")
```

## Detection Strategies

### Signature-Based Detection
Uses distinctive visual features to identify states:
- Presence/absence of specific elements
- Layout patterns
- Color distributions
- Text content

### Probabilistic Detection
Calculates likelihood of being in each state:
- Feature matching scores
- Historical transition probabilities
- Temporal consistency

### Multi-Frame Detection
Validates states across multiple frames:
- Reduces false positives from transient visual noise
- Confirms state stability
- Detects animation states

## State Signatures

A state signature consists of:

1. **Defining Features**: Elements that must be present
2. **Optional Features**: Elements that may be present
3. **Negative Features**: Elements that should NOT be present
4. **Layout Constraints**: Spatial relationships between elements
5. **Visual Properties**: Colors, textures, patterns

## Migration Status

This module represents a major refactoring of state detection logic from the original Brobot library, with improvements for:
- Better separation of concerns
- Enhanced probability models
- Multi-frame validation
- More flexible matching strategies

## Related Modules

- `element_detection`: Provides elements used in state signatures
- `region_analysis`: Provides regional context for state detection
- `state_construction`: Builds state objects from detection results
- `state_management`: Manages the lifecycle of detected states
