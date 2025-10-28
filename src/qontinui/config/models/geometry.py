"""
Geometric primitives for screen coordinates and regions.

This module provides basic geometric types used for positioning and
area definition in screen-based automation.
"""

from pydantic import BaseModel


class Region(BaseModel):
    """Rectangular region on screen."""

    x: int
    y: int
    width: int
    height: int


class Coordinates(BaseModel):
    """X,Y coordinates on screen."""

    x: int
    y: int
