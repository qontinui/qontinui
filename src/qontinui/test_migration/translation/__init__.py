"""
Java-to-Python translation and migration engine components.
"""

from .java_to_python_translator import JavaToPythonTranslator
from .assertion_converter import AssertionConverter
from .spring_test_adapter import SpringTestAdapter, DependencyContainer
from .integration_test_environment import (
    IntegrationTestEnvironment,
    IntegrationTestGenerator,
    ComponentConfiguration,
    DatabaseConfiguration,
    ExternalServiceConfiguration
)

__all__ = [
    'JavaToPythonTranslator',
    'AssertionConverter',
    'SpringTestAdapter',
    'DependencyContainer',
    'IntegrationTestEnvironment',
    'IntegrationTestGenerator',
    'ComponentConfiguration',
    'DatabaseConfiguration',
    'ExternalServiceConfiguration'
]

# Note: LLM and Hybrid translators are available as separate modules
# Import them directly to avoid circular import issues:
# from qontinui.test_migration.translation.llm_test_translator import LLMTestTranslator
# from qontinui.test_migration.translation.hybrid_test_translator import HybridTestTranslator