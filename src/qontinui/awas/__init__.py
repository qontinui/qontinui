"""
AWAS (AI Web Action Standard) integration for Qontinui.

This module provides support for discovering and executing AWAS actions,
enabling AI agents to interact with web applications through standardized
manifest files rather than vision-based detection.

Reference: https://github.com/TamTunnel/AWAS
"""

from .discovery import AwasDiscoveryService
from .executor import AwasExecutor
from .types import (
    AwasAction,
    AwasActionResult,
    AwasAuth,
    AwasAuthType,
    AwasElement,
    AwasManifest,
    AwasParameter,
    ConformanceLevel,
    HttpMethod,
    ParameterLocation,
)

__all__ = [
    # Types
    "AwasManifest",
    "AwasAction",
    "AwasParameter",
    "AwasAuth",
    "AwasAuthType",
    "AwasElement",
    "AwasActionResult",
    "ConformanceLevel",
    "HttpMethod",
    "ParameterLocation",
    # Services
    "AwasDiscoveryService",
    "AwasExecutor",
]
