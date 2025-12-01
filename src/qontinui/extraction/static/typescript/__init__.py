"""
TypeScript/JavaScript parser for QontinUI.

This module provides static analysis capabilities for TypeScript and JavaScript
source code, extracting React components, hooks, conditional rendering patterns,
event handlers, and import/export relationships.
"""

from .parser import (
    ComponentInfo,
    ConditionalRenderInfo,
    EventHandlerInfo,
    ExportInfo,
    FileParseResult,
    ImportInfo,
    JSXElementInfo,
    ParseResult,
    PropInfo,
    StateVariableInfo,
    TypeScriptParser,
    create_parser,
)

__all__ = [
    "TypeScriptParser",
    "create_parser",
    "ParseResult",
    "FileParseResult",
    "ComponentInfo",
    "PropInfo",
    "StateVariableInfo",
    "ConditionalRenderInfo",
    "EventHandlerInfo",
    "ImportInfo",
    "ExportInfo",
    "JSXElementInfo",
]
