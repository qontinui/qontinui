"""Visual GUI Automation (VGA) product surface.

VGA targets external apps that cannot be instrumented (commercial/legacy
Win32, games, kiosks). It consumes the fine-tuned ``qontinui-grounding-v5``
VLM model served through llama-swap, wraps it in an element-proposal /
state-machine builder, and drives the target app via the existing HAL
(mouse / keyboard / screen capture) abstractions.

This module is the Python half of the VGA product surface. The Rust runner
invokes :mod:`qontinui.vga.worker` as a subprocess; the web UI talks to
the same client via the Next.js ``/api/vga/*`` routes.
"""

from __future__ import annotations

from .client import GroundResult, VgaClient, VgaClientError
from .runtime import VgaRunResult, VgaRuntime, VgaStepEvent
from .state_machine import BBox, VgaElement, VgaState, VgaStateMachine, VgaTransition
from .worker import main as worker_main

__all__ = [
    "BBox",
    "GroundResult",
    "VgaClient",
    "VgaClientError",
    "VgaElement",
    "VgaRunResult",
    "VgaRuntime",
    "VgaState",
    "VgaStateMachine",
    "VgaStepEvent",
    "VgaTransition",
    "VgaWorker",
    "worker_main",
]


# Public alias for worker entrypoint — the runner calls ``python -m
# qontinui.vga.worker`` but Python-side callers can use ``VgaWorker`` as
# a named import to construct the worker via code.
VgaWorker = worker_main
