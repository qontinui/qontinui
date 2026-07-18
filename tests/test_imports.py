"""Test that modules can be imported without errors."""

import sys
from unittest.mock import MagicMock

import pytest


def test_import_state_management():
    """Test importing state management modules."""
    from qontinui.state_management import Element, QontinuiStateManager, State

    assert QontinuiStateManager is not None
    assert State is not None
    assert Element is not None


def test_import_perception():
    """Test importing perception modules."""
    # Mock heavy dependencies
    sys.modules["segment_anything"] = MagicMock()
    sys.modules["transformers"] = MagicMock()
    sys.modules["faiss"] = MagicMock()

    from qontinui.perception import ObjectVectorizer, ScreenSegmenter

    assert ScreenSegmenter is not None
    assert ObjectVectorizer is not None


def test_import_dsl():
    """Test importing DSL modules."""
    from qontinui.dsl import QontinuiDSLParser

    assert QontinuiDSLParser is not None


@pytest.mark.skip(reason="Migrations module requires AI dependencies (faiss, torch, transformers)")
def test_import_migrations():
    """Test importing migration modules."""
    from qontinui.migrations import BrobotConverter

    assert BrobotConverter is not None


def test_ml_lazy_imports_stay_torch_free():
    """The [ml]-dependent modules must import without loading torch.

    Regression guard for the extras-split: a future top-level torch /
    transformers / torchvision import in these modules would re-break the
    torch-free default install (they must import lazily inside the classes).
    """
    torch_preloaded = "torch" in sys.modules

    import qontinui.perception.embeddings  # noqa: F401
    from qontinui.rag.embeddings.image import (  # noqa: F401
        CLIPEmbedder,
        DINOv2Embedder,
        HybridImageEmbedder,
    )

    if not torch_preloaded:
        assert "torch" not in sys.modules, (
            "importing qontinui.rag.embeddings.image / "
            "qontinui.perception.embeddings must not load torch — "
            "a module-level ML import was reintroduced"
        )


def test_ml_instantiation_error_names_the_extra():
    """Without torch, instantiating the embedders raises an actionable error."""
    try:
        import torch  # noqa: F401

        pytest.skip("torch installed ([ml] env) — the error path is untestable here")
    except ImportError:
        pass

    from qontinui.perception.embeddings import EmbeddingGenerator
    from qontinui.rag.embeddings.image import CLIPEmbedder

    for cls in (CLIPEmbedder, EmbeddingGenerator):
        with pytest.raises(ImportError, match=r"qontinui\[ml\]"):
            cls()
