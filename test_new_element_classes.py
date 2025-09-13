#!/usr/bin/env python3
"""Test the newly implemented model/element classes."""

import sys
sys.path.insert(0, 'src')

# Test new model/element imports
from qontinui.model.element import (
    Grid, GridBuilder,
    Scene,
    Text, 
    Anchors,
    Positions, PositionName,
    Region, Location, Anchor
)

print("✓ All new model/element classes imported successfully")

# Test Grid creation
grid = GridBuilder() \
    .set_region(Region(0, 0, 700, 500)) \
    .set_rows(5) \
    .set_columns(7) \
    .build()
print(f"✓ Grid created: {grid.rows}x{grid.cols} = {len(grid.grid_regions)} cells")

# Test Scene creation
scene = Scene(filename="test_screenshot.png")
print(f"✓ Scene created: {scene}")

# Test Text with OCR variations
text = Text()
text.add("Submit")
text.add("Subrnit")  # OCR error
text.add("Submit")
print(f"✓ Text created: most common = '{text.get_most_common()}', confidence = {text.get_confidence('Submit'):.1%}")

# Test Anchors collection
anchors = Anchors()
anchors.add(Anchor.TOP_LEFT)
anchors.add(Anchor.BOTTOM_RIGHT)
print(f"✓ Anchors created: {anchors.size()} anchors")

# Test Positions
coords = Positions.get_coordinates(PositionName.MIDDLEMIDDLE)
print(f"✓ Positions: center = {coords}")

print("\n✅ All new model/element classes working correctly!")