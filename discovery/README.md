# Discovery Module

Automated discovery of UI states and elements from screenshots.

## Overview

The Discovery module analyzes screenshots to automatically identify:
- **State Images**: UI elements (icons, buttons, menu items) that appear across multiple screenshots
- **States**: Collections of State Images that appear together (representing application states)
- **Transitions**: Changes between states

## Modules

### Background Removal
**File**: `background_removal.py`

Removes dynamic backgrounds from screenshots before pixel comparison. Essential for robust State Discovery when UI elements remain fixed but backgrounds vary.

**Key Features**:
- Three detection strategies: temporal variance, edge density, uniformity
- Configurable thresholds for each strategy
- Outputs RGBA images with transparent backgrounds
- Includes visualization tools for debugging

**Usage**:
```python
from discovery.background_removal import remove_backgrounds_simple

screenshots = [cv2.imread(f) for f in files]
masked = remove_backgrounds_simple(screenshots)
```

See `docs/background_removal_summary.md` for details.

### Pixel Stability Analyzer
**File**: `pixel_stability_analyzer.py`

Grid-based approach for finding stable regions across screenshots.

**How it works**:
1. Divides screen into overlapping grids
2. Compares regions using MD5 hash and pixel similarity
3. Groups regions by screenshot presence
4. Creates State Images from stable regions

**Best for**: Quick analysis, rectangular UI elements

### Pixel Stability Matrix Analyzer
**File**: `pixel_stability_matrix_analyzer.py`

Pixel-level approach for finding non-rectangular UI elements.

**How it works**:
1. Builds stability matrix tracking pixel presence
2. Groups pixels by presence patterns
3. Finds connected components
4. Creates masked State Images

**Best for**: Non-rectangular elements, complex shapes, precise detection

### Models
**File**: `models.py`

Data models for State Discovery:
- `AnalysisConfig`: Configuration parameters
- `StateImage`: Detected UI element
- `DiscoveredState`: Collection of State Images
- `AnalysisResult`: Complete analysis output

## Recommended Workflow

### For Static UI with Dynamic Backgrounds

1. **Remove backgrounds first**:
```python
from discovery.background_removal import BackgroundRemovalAnalyzer, BackgroundRemovalConfig

config = BackgroundRemovalConfig(
    use_temporal_variance=True,
    variance_threshold=20.0,
)
analyzer = BackgroundRemovalAnalyzer(config)
masked_screenshots, stats = analyzer.remove_backgrounds(screenshots)
```

2. **Run State Discovery on masked screenshots**:
```python
from discovery.pixel_stability_matrix_analyzer import PixelStabilityMatrixAnalyzer
from discovery.models import AnalysisConfig

discovery_config = AnalysisConfig(
    min_screenshots_present=2,
    color_tolerance=5,
)
discovery_analyzer = PixelStabilityMatrixAnalyzer(discovery_config)
result = discovery_analyzer.analyze_screenshots(masked_screenshots)
```

3. **Review results**:
```python
print(f"Found {len(result.state_images)} State Images")
print(f"Found {len(result.states)} States")
for state in result.states:
    print(f"State {state.name}: {len(state.state_image_ids)} images")
```

### For General UI Discovery

Use the analyzer directly without background removal:
```python
from discovery.pixel_stability_matrix_analyzer import PixelStabilityMatrixAnalyzer

analyzer = PixelStabilityMatrixAnalyzer(config)
result = analyzer.analyze_screenshots(screenshots)
```

## Configuration Tips

### Background Removal

**For subtle backgrounds** (solid colors, gentle gradients):
```python
BackgroundRemovalConfig(
    use_temporal_variance=False,
    use_edge_density=True,
    edge_density_threshold=0.03,
    use_uniformity=True,
    uniformity_threshold=15.0,
)
```

**For dynamic backgrounds** (animations, video):
```python
BackgroundRemovalConfig(
    use_temporal_variance=True,
    variance_threshold=10.0,  # Lower = more sensitive
    min_screenshots_for_variance=3,
)
```

**For mixed backgrounds**:
```python
BackgroundRemovalConfig(
    use_temporal_variance=True,
    use_edge_density=True,
    use_uniformity=True,
    # Use default thresholds
)
```

### State Discovery

**For high precision** (fewer false positives):
```python
AnalysisConfig(
    color_tolerance=3,  # Stricter pixel matching
    min_screenshots_present=3,  # Must appear in more screenshots
    min_region_size=(30, 30),  # Larger minimum size
)
```

**For high recall** (find more elements):
```python
AnalysisConfig(
    color_tolerance=10,  # More tolerant matching
    min_screenshots_present=2,  # Can appear in fewer screenshots
    min_region_size=(10, 10),  # Smaller minimum size
)
```

## Debugging

### Visualize Background Removal
```python
analyzer = BackgroundRemovalAnalyzer()
masked, stats = analyzer.remove_backgrounds(screenshots, debug=True)

# Visualize detected background
background_mask = stats['background_mask']
vis = analyzer.visualize_mask(screenshots[0], background_mask)
cv2.imshow('Background Detection', vis)
cv2.waitKey(0)
```

### Check State Discovery Results
```python
# After running analysis
for si in result.state_images:
    print(f"{si.name}: {si.width}x{si.height} at ({si.x}, {si.y})")
    print(f"  Appears in: {len(si.screenshots)} screenshots")
    print(f"  Stability: {si.stability_score:.2f}")
```

## Performance

### Typical Times (on 1920x1080 screenshots)

- **Background Removal**: ~0.5-2s per screenshot (depends on strategies enabled)
- **Pixel Stability Analyzer**: ~1-5s for 10 screenshots
- **Pixel Stability Matrix Analyzer**: ~5-30s for 10 screenshots (more thorough)

### Optimization Tips

1. **Downsample large screenshots** before analysis
2. **Limit analysis region** if you know where UI elements are
3. **Disable unused strategies** in background removal
4. **Use grid analyzer** for quick results, matrix analyzer for precision

## Common Issues

### Issue: Too much foreground detected after background removal
**Solution**: Increase `variance_threshold`, decrease `edge_density_threshold`

### Issue: Icons being removed as background
**Solution**: Verify icons don't actually change between screenshots; adjust thresholds

### Issue: No State Images found
**Solution**: Lower `color_tolerance`, decrease `min_screenshots_present`

### Issue: Too many duplicate State Images
**Solution**: Increase `color_tolerance` to merge similar regions

## Documentation

- **Full Analysis**: `docs/state_discovery_improvement_analysis.md`
- **Background Removal Summary**: `docs/background_removal_summary.md`
- **API Reference**: See docstrings in each module

## Testing

Run tests:
```bash
# Background removal
python tests/discovery/test_background_removal_standalone.py

# Full test suite (requires pytest)
pytest tests/discovery/
```

## Future Enhancements

- Web UI for interactive parameter tuning
- Machine learning for automatic background detection
- Transition detection between states
- Semantic grouping of State Images
- Multi-resolution analysis
