# Region Analysis

## Overview

The `region_analysis` module provides functionality for dividing screenshots into meaningful regions, analyzing their properties, and understanding spatial relationships between different areas of the screen.

## Purpose

Region analysis bridges the gap between low-level element detection and high-level state understanding by:

- Segmenting screens into logical functional areas
- Analyzing region properties (size, position, content type)
- Detecting spatial relationships between regions
- Clustering related regions
- Tracking regions across multiple frames
- Identifying stable vs. dynamic regions

## Key Components

### Planned Modules

- **Region Extractor**: Segment screens into distinct regions
- **Region Analyzer**: Analyze properties of individual regions
- **Spatial Analyzer**: Understand spatial relationships
- **Region Tracker**: Track regions across frames
- **Region Classifier**: Classify region types (navigation, content, toolbar, etc.)

## Usage Pattern

```python
from qontinui.discovery.region_analysis import RegionAnalyzer

# Initialize analyzer
analyzer = RegionAnalyzer()

# Analyze screenshot regions
regions = analyzer.analyze(screenshot)

# Examine regions
for region in regions:
    print(f"Region: {region.type}")
    print(f"Bounds: {region.bounds}")
    print(f"Elements: {region.element_count}")
    print(f"Stability: {region.stability_score}")
```

## Region Types

Regions can be classified into various types:

- **Navigation Regions**: Menus, toolbars, navigation bars
- **Content Regions**: Main content areas, panels
- **Interactive Regions**: Button groups, form fields
- **Informational Regions**: Status bars, notifications
- **Decorative Regions**: Backgrounds, borders, spacers

## Analysis Techniques

1. **Segmentation**: Divide screen using edge detection, color analysis
2. **Property Extraction**: Size, position, density, color distribution
3. **Relationship Detection**: Containment, adjacency, alignment
4. **Temporal Analysis**: Stability, change frequency across frames
5. **Classification**: Machine learning-based region type classification

## Migration Status

This module consolidates region-related functionality from multiple sources in the original Brobot codebase, providing a unified interface for region analysis.

## Related Modules

- `element_detection`: Provides elements contained within regions
- `state_detection`: Uses region analysis for state identification
- `pixel_analysis`: Low-level pixel stability analysis (existing)
