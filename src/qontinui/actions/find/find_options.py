"""FindOptions - Configuration for find operations."""

from dataclasses import dataclass

from ...model.element import Region


@dataclass
class FindOptions:
    """Options for find operations.

    Used by all find operations regardless of mock/real execution.
    """

    similarity: float = 0.8
    find_all: bool = False
    search_region: Region | None = None
    timeout: float = 0.0
    collect_debug: bool = False
