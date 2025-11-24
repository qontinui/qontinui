# State Construction

## Overview

The `state_construction` module provides functionality for building complete state objects from detected elements and regions. It transforms raw detection data into structured, actionable state representations.

## Purpose

State construction is the bridge between detection and application automation, enabling:

- Creation of state objects from detected elements
- Identification of defining features that characterize states
- Establishment of relationships between states
- Validation and refinement of constructed states
- Merging of duplicate or similar states
- Enrichment of states with metadata and context

## Key Components

### Planned Modules

- **State Builder**: Construct state objects from detection data
- **Feature Identifier**: Identify defining vs. auxiliary features
- **Relationship Mapper**: Map transitions and relationships between states
- **State Validator**: Validate constructed states for completeness
- **State Merger**: Merge duplicate or similar states
- **Metadata Enricher**: Add semantic information to states

## Usage Pattern

```python
from qontinui.discovery.state_construction import StateBuilder

# Initialize builder
builder = StateBuilder()

# Detect elements and regions
elements = element_detector.detect(screenshot)
regions = region_analyzer.analyze(screenshot)

# Construct state
state = builder.construct(
    screenshot=screenshot,
    elements=elements,
    regions=regions,
    name="LoginScreen"
)

# Access constructed state
print(f"State: {state.name}")
print(f"Defining elements: {len(state.defining_elements)}")
print(f"Optional elements: {len(state.optional_elements)}")
print(f"Transitions: {state.possible_transitions}")
```

## State Object Structure

A constructed state includes:

```python
class State:
    name: str
    defining_elements: List[Element]      # Must be present
    optional_elements: List[Element]       # May be present
    regions: List[Region]                  # Screen regions
    transitions: List[Transition]          # Possible next states
    metadata: Dict[str, Any]              # Additional context
    screenshot: Optional[Image]            # Reference screenshot
    signature: StateSignature             # Visual signature
```

## Construction Process

1. **Element Analysis**: Categorize elements by importance
2. **Feature Selection**: Identify which elements define the state
3. **Relationship Discovery**: Determine spatial/logical relationships
4. **Signature Creation**: Build visual signature for matching
5. **Validation**: Ensure state is complete and valid
6. **Enrichment**: Add metadata, transitions, semantic info

## State Refinement

After initial construction, states can be refined through:

- **User Feedback**: Manual corrections and annotations
- **Usage Data**: Learn from actual application behavior
- **Similarity Analysis**: Merge or split states based on similarity
- **Temporal Analysis**: Refine based on observed transitions

## Best Practices

### Defining Elements
Choose elements that are:
- Unique to the state
- Always present when state is active
- Stable (don't change appearance frequently)
- Easy to detect reliably

### Optional Elements
Include elements that:
- May appear conditionally
- Provide additional context
- Help disambiguate similar states

### Negative Elements
Specify elements that:
- Should NOT be present in this state
- Help distinguish from similar states

## Migration Status

This module consolidates state construction logic from multiple locations in the original Brobot codebase, providing a cleaner, more maintainable architecture.

## Related Modules

- `element_detection`: Source of elements for construction
- `region_analysis`: Provides regional context
- `state_detection`: Uses constructed states for matching
- `state_management`: Manages constructed state lifecycle
