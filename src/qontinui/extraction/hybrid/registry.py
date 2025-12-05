"""
Registry for tech stack extractors.

Manages the registration and discovery of TechStackExtractor implementations.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import TechStackExtractor

logger = logging.getLogger(__name__)


class TechStackRegistry:
    """
    Registry for tech stack-specific extractors.

    Provides automatic discovery and selection of the appropriate extractor
    based on project structure.
    """

    def __init__(self):
        self._extractors: list[type[TechStackExtractor]] = []
        self._load_default_extractors()

    def _load_default_extractors(self) -> None:
        """Load built-in extractors."""
        try:
            from .tauri_typescript import TauriTypeScriptExtractor

            self.register(TauriTypeScriptExtractor)
        except ImportError as e:
            logger.debug(f"TauriTypeScriptExtractor not available: {e}")

        try:
            from .nextjs import NextJSExtractor

            self.register(NextJSExtractor)
        except ImportError as e:
            logger.debug(f"NextJSExtractor not available: {e}")

    def register(self, extractor_class: type["TechStackExtractor"]) -> None:
        """
        Register a tech stack extractor.

        Args:
            extractor_class: TechStackExtractor subclass to register
        """
        if extractor_class not in self._extractors:
            self._extractors.append(extractor_class)
            logger.debug(f"Registered extractor: {extractor_class.tech_stack_name}")

    def unregister(self, extractor_class: type["TechStackExtractor"]) -> None:
        """
        Unregister a tech stack extractor.

        Args:
            extractor_class: TechStackExtractor subclass to unregister
        """
        if extractor_class in self._extractors:
            self._extractors.remove(extractor_class)
            logger.debug(f"Unregistered extractor: {extractor_class.tech_stack_name}")

    def get_extractor_for(self, project_path: Path) -> type["TechStackExtractor"] | None:
        """
        Find the appropriate extractor for a project.

        Args:
            project_path: Root directory of the project

        Returns:
            TechStackExtractor class that can handle the project, or None
        """
        for extractor_class in self._extractors:
            if extractor_class.can_handle(project_path):
                logger.info(
                    f"Selected extractor {extractor_class.tech_stack_name} "
                    f"for project at {project_path}"
                )
                return extractor_class

        return None

    def list_extractors(self) -> list[str]:
        """
        List all registered extractor names.

        Returns:
            List of tech stack names
        """
        return [e.tech_stack_name for e in self._extractors]

    def get_all_extractors(self) -> list[type["TechStackExtractor"]]:
        """
        Get all registered extractor classes.

        Returns:
            List of TechStackExtractor classes
        """
        return list(self._extractors)
