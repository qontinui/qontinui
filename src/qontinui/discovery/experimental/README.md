# Experimental Detection Features

## Overview

The `experimental` module contains cutting-edge detection algorithms, research implementations, and experimental features being evaluated for inclusion in the main detection pipeline.

## Warning

**This module is experimental and unstable:**
- APIs may change without notice
- Features may be removed or significantly modified
- Performance characteristics are not guaranteed
- Not suitable for production use
- Documentation may be incomplete
- No backward compatibility guarantees

## Recent Migrations (2024)

The following experimental detectors have been migrated from `qontinui-web/research_env/detectors/` to this module:

**Migrated Files:**
- `sam3_detector.py` - SAM3 model integration
- `hybrid_detector.py` - Multi-method consensus detection
- `consistency_detector.py` - Multi-screenshot consistency analysis
- `edge_detector.py` - Canny edge-based detection
- `contour_detector.py` - Adaptive threshold contour detection
- `mser_detector.py` - MSER region detection
- `color_detector.py` - K-means color clustering detection
- `template_detector.py` - Corner detection and pattern matching
- `base_detector.py` - Base classes for all detectors

**Migration Changes:**
- Updated imports to use qontinui library paths (`from .types import BBox`)
- Added comprehensive EXPERIMENTAL warnings in all docstrings
- Standardized module documentation format
- Created `types.py` for shared type definitions
- Updated `__init__.py` with proper exports

**Stability Status:**
All migrated detectors are marked as **EXPERIMENTAL** and **UNSTABLE**. They are research implementations that may have:
- Inconsistent performance across different UI types
- High computational costs
- Incomplete error handling
- Limited testing coverage
- Dependency on external models or checkpoints

## Purpose

This module serves as:

- Testing ground for new detection approaches
- Integration point for research implementations
- Performance optimization experiments
- Proof-of-concept implementations
- Evaluation of new ML models and techniques

## Current Experimental Features

### Advanced ML Models

#### Segment Anything Model 3 (SAM3)
**Status**: Migrated from research_env
**Module**: `sam3_detector.py`
- Zero-shot image segmentation
- Automatic mask generation via grid-based prompting
- Text-guided concept-based segmentation
- Interactive segmentation from point prompts
- Requires SAM3 checkpoint files and PyTorch 2.7+

### Traditional Computer Vision Detectors

#### Edge-Based Detector
**Status**: Migrated from research_env
**Module**: `edge_detector.py`
- Canny edge detection with configurable thresholds
- Morphological operations for edge connection
- Best for high-contrast UIs with well-defined boundaries

#### Contour Detector
**Status**: Migrated from research_env
**Module**: `contour_detector.py`
- Adaptive thresholding for varying lighting conditions
- Otsu thresholding option
- Effective for text-heavy interfaces and complex backgrounds

#### MSER Detector
**Status**: Migrated from research_env
**Module**: `mser_detector.py`
- Maximally Stable Extremal Regions detection
- Configurable stability parameters
- Ideal for text, icons, and buttons with distinct boundaries

#### Color Cluster Detector
**Status**: Migrated from research_env
**Module**: `color_detector.py`
- K-means color clustering (RGB or HSV)
- Connected component analysis per cluster
- Useful for color-coded UI elements

#### Template/Corner Detector
**Status**: Migrated from research_env
**Module**: `template_detector.py`
- Good features to track (corner detection)
- Morphological clustering of corners
- Works well for rectangular UI elements

### Multi-Method Detectors

#### Hybrid Detector
**Status**: Migrated from research_env
**Module**: `hybrid_detector.py`
- Combines edge, contour, and MSER detection
- Consensus voting mechanism
- Configurable agreement threshold
- Reduces false positives through multi-method validation

#### Consistency Detector
**Status**: Migrated from research_env
**Module**: `consistency_detector.py`
- Analyzes multiple screenshots to find persistent elements
- Pixel-wise consistency scoring with edge weighting
- Optional screenshot alignment (simple or feature-based)
- Identifies static navigation, headers, and UI chrome

### Novel Algorithms

- **Perceptual Hashing**: Fast visual similarity matching
- **Structural Similarity**: Layout-aware state comparison
- **Temporal Coherence**: Multi-frame detection smoothing
- **Uncertainty Quantification**: Confidence estimation improvements

### Performance Optimizations

- GPU acceleration experiments
- Parallel processing strategies
- Caching and memoization approaches
- Approximate algorithms for speed

## Usage Pattern

### Basic Single-Method Detection

```python
from qontinui.discovery.experimental import EdgeBasedDetector

# Initialize detector
detector = EdgeBasedDetector()

# Run detection with custom parameters
boxes = detector.detect(
    "screenshot.png",
    canny_low=50,
    canny_high=150,
    min_area=100
)

# Results are BBox objects with x1, y1, x2, y2, label, confidence
for box in boxes:
    print(f"Detected element at ({box.x1}, {box.y1}) - ({box.x2}, {box.y2})")
```

### Hybrid Multi-Method Detection

```python
from qontinui.discovery.experimental import HybridDetector

# Initialize hybrid detector (combines multiple methods)
detector = HybridDetector()

# Run with consensus voting
boxes = detector.detect(
    "screenshot.png",
    use_edge=True,
    use_contour=True,
    use_mser=True,
    consensus_threshold=2  # Require 2+ methods to agree
)
```

### Multi-Screenshot Consistency Detection

```python
from qontinui.discovery.experimental import ConsistencyDetector, MultiScreenshotDataset, ScreenshotInfo

# Prepare dataset with multiple screenshots
screenshots = [
    ScreenshotInfo(screenshot_id=1, path="screen1.png"),
    ScreenshotInfo(screenshot_id=2, path="screen2.png"),
    ScreenshotInfo(screenshot_id=3, path="screen3.png"),
]
dataset = MultiScreenshotDataset(screenshots=screenshots)

# Detect consistent elements across all screenshots
detector = ConsistencyDetector()
results = detector.detect_multi(
    dataset,
    consistency_threshold=0.9,
    edge_weight=0.3
)

# Results map screenshot_id -> List[BBox]
for screenshot_id, boxes in results.items():
    print(f"Screenshot {screenshot_id}: {len(boxes)} consistent elements")
```

### SAM3 Advanced Detection

```python
from qontinui.discovery.experimental import SAM3Detector

# Initialize SAM3 (requires checkpoint)
detector = SAM3Detector()

# Automatic grid-based segmentation
boxes = detector.detect(
    "screenshot.png",
    grid_points=32,
    min_area=100
)

# Or text-guided concept segmentation
boxes = detector.detect(
    "screenshot.png",
    text_prompt="button",
    min_area=100
)
```

## Experimental Projects

### 1. SAM Integration
**Status**: Active development
**Goal**: Leverage SAM for automatic UI element segmentation
**Files**: `sam_detector.py`, `sam_integration.py`

### 2. Semantic State Understanding
**Status**: Research phase
**Goal**: Use NLP and vision models for semantic state labeling
**Files**: `semantic_analyzer.py`, `clip_integration.py`

### 3. Few-Shot Learning
**Status**: Prototype
**Goal**: Enable state detection with minimal training examples
**Files**: `few_shot_detector.py`, `meta_learning.py`

### 4. Active Learning Pipeline
**Status**: Planning
**Goal**: Intelligently select examples for labeling
**Files**: `active_learning.py`, `uncertainty_sampling.py`

## Migration Path

When experimental features mature:

1. **Evaluation**: Thorough testing and benchmarking
2. **Stabilization**: API design and stabilization
3. **Documentation**: Complete documentation
4. **Testing**: Comprehensive test coverage
5. **Migration**: Move to appropriate stable module
6. **Deprecation**: Mark experimental version as deprecated

## Contributing Experimental Features

When adding experimental features:

- Document the research goal and approach
- Include performance characteristics (if known)
- Provide example usage
- Note any special dependencies
- Mark stability level clearly

## Dependencies

Experimental features may require additional dependencies:

```python
# SAM integration
pip install segment-anything torch torchvision

# YOLO integration
pip install ultralytics

# Transformer models
pip install transformers accelerate

# Research tools
pip install opencv-contrib-python scikit-image
```

## Related Modules

- `element_detection`: Stable element detection
- `state_detection`: Stable state detection
- `vision`: Core vision utilities
- `config`: Configuration for experimental features

## Future Directions

Potential areas for future experimental work:

- Reinforcement learning for state exploration
- Generative models for synthetic training data
- Multi-modal fusion (vision + accessibility tree)
- Graph neural networks for UI structure
- Self-supervised learning approaches
