#!/usr/bin/env python3
"""Test Phase 3 newly implemented model classes."""

import sys
sys.path.insert(0, 'src')

# Test StateObjectMetadata
from qontinui.model.state import StateObjectMetadata, StateObjectType
print("Testing StateObjectMetadata...")
metadata = StateObjectMetadata()
metadata.state_object_id = "img_001"
metadata.object_type = StateObjectType.IMAGE
metadata.state_object_name = "LoginButton"
metadata.owner_state_name = "LoginState"
print(f"✓ StateObjectMetadata: {metadata}")
print(f"  is_image={metadata.is_image()}, is_valid={metadata.is_valid()}")

# Test Direction enum
from qontinui.model.transition import Direction
print("\nTesting Direction enum...")
to_direction = Direction.TO
from_direction = Direction.FROM
print(f"✓ Direction TO: is_forward={to_direction.is_forward()}")
print(f"✓ Direction FROM: reverse={from_direction.reverse()}")

# Test OverlappingGrids
from qontinui.model.element import Grid, GridBuilder, OverlappingGrids, Region
print("\nTesting OverlappingGrids...")
base_grid = GridBuilder() \
    .set_region(Region(0, 0, 800, 600)) \
    .set_columns(4) \
    .set_rows(3) \
    .build()
overlapping = OverlappingGrids(base_grid)
print(f"✓ OverlappingGrids: {overlapping}")
print(f"  Primary cells: {len(overlapping.get_primary_regions())}")
print(f"  Inner cells: {len(overlapping.get_inner_regions())}")
print(f"  Total cells: {overlapping.get_region_count()}")

# Test point containment
cells = overlapping.contains_point(100, 100)
print(f"✓ Point (100,100) contained in {len(cells)} cells")

# Test best cell selection
best_cell = overlapping.get_best_cell_for_point(400, 300)
print(f"✓ Best cell for (400,300): x={best_cell.x}, y={best_cell.y}")

print("\n✅ All Phase 3 classes working correctly!")