# Element Detection

## Overview

The `element_detection` module provides core functionality for identifying and classifying UI elements in screenshots. This is the foundational layer of the discovery pipeline, enabling automated identification of buttons, text fields, icons, and other interactive components.

## Purpose

Element detection serves as the first step in understanding the structure of an application's UI. By identifying individual elements and their properties, we can:

- Locate interactive components for automated testing
- Build element inventories for state construction
- Enable template-based element matching
- Support OCR-based text element detection
- Facilitate machine learning-based element classification

## Key Components

### Planned Modules

- **Template Matching**: Match known UI patterns against screenshots
- **Feature Detection**: Identify distinct visual features (corners, edges, blobs)
- **OCR Integration**: Detect and extract text-based elements
- **ML Classifiers**: Use machine learning models to classify element types
- **Element Clustering**: Group related elements together

## Usage Pattern

```python
from qontinui.discovery.element_detection import ElementDetector

# Initialize detector with configuration
detector = ElementDetector(
    methods=['template', 'feature', 'ocr'],
    confidence_threshold=0.7
)

# Detect elements in screenshot
elements = detector.detect(screenshot)

# Process detected elements
for element in elements:
    print(f"Element: {element.type}")
    print(f"Location: {element.bounds}")
    print(f"Confidence: {element.confidence}")
```

## Migration Status

This module is part of the detection code migration from the Java Brobot library. Components will be migrated and refactored to follow Python best practices and the new qontinui architecture.

## Related Modules

- `region_analysis`: Analyzes regions containing detected elements
- `state_detection`: Uses element detection for state identification
- `state_construction`: Builds states from detected elements
