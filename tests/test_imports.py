"""Test that modules can be imported without errors."""

import pytest
import sys
from unittest.mock import MagicMock


def test_import_state_management():
    """Test importing state management modules."""
    from qontinui.state_management import QontinuiStateManager, State, Element
    assert QontinuiStateManager is not None
    assert State is not None
    assert Element is not None


def test_import_perception():
    """Test importing perception modules."""
    # Mock heavy dependencies
    sys.modules['segment_anything'] = MagicMock()
    sys.modules['transformers'] = MagicMock()
    sys.modules['faiss'] = MagicMock()
    
    from qontinui.perception import ScreenSegmenter, ObjectVectorizer
    assert ScreenSegmenter is not None
    assert ObjectVectorizer is not None


def test_import_dsl():
    """Test importing DSL modules."""
    from qontinui.dsl import QontinuiDSLParser
    assert QontinuiDSLParser is not None


def test_import_migrations():
    """Test importing migration modules."""
    from qontinui.migrations import BrobotConverter
    assert BrobotConverter is not None