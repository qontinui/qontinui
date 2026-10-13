"""Make docker/imagegen importable so the tests load qontinui_imagegen without installing it.

This directory deliberately has no __init__.py: as a package it would be named ``tests``
and collide with the repository's top-level ``tests`` package when pytest collects both.
"""

import sys
from pathlib import Path

IMAGEGEN_ROOT = Path(__file__).resolve().parent.parent
if str(IMAGEGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(IMAGEGEN_ROOT))
