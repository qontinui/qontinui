"""Modules that touch torch must import cleanly when the ``ml`` extra is absent.

The macOS x86_64 bridge executor is built without torch/torchvision (no wheel
exists for that platform), so every torch-touching module has to import with
torch blocked and only fail, with an actionable message, when the feature is used.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Modules in src/qontinui that reference torch/torchvision (audit, Phase 1).
TORCH_MODULES = [
    "qontinui._ml_deps",
    "qontinui.find.backends.omniparser_config",
    "qontinui.find.backends.qatm_backend",
    "qontinui.find.backends.qatm_matcher",
    "qontinui.find.backends.qatm_config",
    "qontinui.find.backends.scale_adaptive_backend",
    "qontinui.rag.embeddings.text",
    "qontinui.rag.embeddings.image",
    "qontinui.hal.implementations.easyocr_engine",
    "qontinui.embeddings.sentence_transformers_provider",
    "qontinui.discovery.element_detection.omniparser_detector",
    "qontinui.semantic.processors.sam3_processor",
    "qontinui.semantic.description.clip_generator",
    "qontinui.extraction.runtime.uitars.provider",
    "qontinui.perception.vectorization",
    "qontinui.perception.embeddings",
]

BLOCKED = [
    "torch",
    "torchvision",
    "transformers",
    "sentence_transformers",
    "easyocr",
    "ultralytics",
    "timm",
]

_SCRIPT = textwrap.dedent(
    """
    import importlib, sys

    class _Block:
        # A meta-path finder (not sys.modules[name] = None): third-party code such
        # as scipy probes sys.modules["torch"] and chokes on a None placeholder.
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in {blocked!r}:
                raise ImportError("blocked for test: " + name)

    sys.meta_path.insert(0, _Block())
    importlib.import_module({module!r})
    """
)


@pytest.mark.parametrize("module", TORCH_MODULES)
def test_module_imports_without_torch(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(blocked=BLOCKED, module=module)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_require_torch_names_the_extra() -> None:
    code = textwrap.dedent(
        """
        import sys

        class _Block:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] == "torch":
                    raise ImportError("blocked for test")

        sys.meta_path.insert(0, _Block())
        from qontinui._ml_deps import require_torch
        try:
            require_torch("X")
        except ImportError as e:
            assert "qontinui[ml]" in str(e)
        else:
            raise SystemExit("no ImportError")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr[-2000:]
