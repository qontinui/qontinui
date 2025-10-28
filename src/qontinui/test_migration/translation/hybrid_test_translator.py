"""
Hybrid test translator that combines utility-based and LLM-based translation.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ..core.interfaces import TestTranslator
    from ..core.models import TestFile, TestType
    from .llm_test_translator import LLMTestTranslator
    from .python_test_generator import PythonTestGenerator
else:
    try:
        from ..core.interfaces import TestTranslator
        from ..core.models import TestFile, TestType
        from .llm_test_translator import LLMTestTranslator
        from .python_test_generator import PythonTestGenerator
    except ImportError:
        # For standalone testing
        from core.interfaces import TestTranslator
        from core.models import TestFile, TestType
        from llm_test_translator import LLMTestTranslator
        from python_test_generator import PythonTestGenerator


class TranslationStrategy(Enum):
    """Translation strategy options."""

    UTILITY_ONLY = "utility_only"
    LLM_ONLY = "llm_only"
    HYBRID_UTILITY_FIRST = "hybrid_utility_first"
    HYBRID_LLM_FIRST = "hybrid_llm_first"
    UTILITY_WITH_LLM_VALIDATION = "utility_with_llm_validation"


@dataclass
class TranslationResult:
    """Result of hybrid translation with metadata."""

    translated_code: str
    strategy_used: TranslationStrategy
    utility_attempted: bool
    llm_attempted: bool
    confidence_score: float
    translation_time: float
    errors: list[str]
    warnings: list[str]
    improvement_suggestions: list[str]


class TranslationException(Exception):
    """Exception raised during translation process."""

    pass


class HybridTestTranslator(TestTranslator):
    """
    Hybrid translator that intelligently combines utility-based and LLM-based translation.

    This translator provides multiple strategies:
    1. Utility-first: Try utility translator, fall back to LLM if needed
    2. LLM-first: Try LLM translator, fall back to utility if needed
    3. Utility with LLM validation: Use utility then validate/enhance with LLM
    4. Pure utility or LLM: Use only one approach

    The translator automatically selects the best strategy based on test complexity.
    """

    def __init__(
        self,
        llm_client=None,
        default_strategy: TranslationStrategy = TranslationStrategy.HYBRID_UTILITY_FIRST,
        enable_caching: bool = True,
    ) -> None:
        """
        Initialize the hybrid translator.

        Args:
            llm_client: LLM client for AI-based translation
            default_strategy: Default translation strategy to use
            enable_caching: Whether to cache translation results
        """
        self.utility_translator = PythonTestGenerator()
        self.llm_translator = LLMTestTranslator(llm_client) if llm_client else None
        self.default_strategy = default_strategy
        self.enable_caching = enable_caching
        self.translation_cache: dict[str, Any] | None = {} if enable_caching else None

        # Configure logging
        self.logger = logging.getLogger(__name__)

        # Translation statistics
        self.stats = {
            "total_translations": 0,
            "utility_successes": 0,
            "llm_successes": 0,
            "hybrid_successes": 0,
            "failures": 0,
        }

    def translate_test_file(self, test_file: TestFile) -> str:
        """
        Translate a test file using the configured strategy.

        Args:
            test_file: The TestFile object to translate

        Returns:
            Translated Python test code
        """
        result = self.translate_with_metadata(test_file)
        return result.translated_code

    def translate_with_metadata(
        self, test_file: TestFile, strategy: TranslationStrategy | None = None
    ) -> TranslationResult:
        """
        Translate test file and return detailed results with metadata.

        Args:
            test_file: The TestFile object to translate
            strategy: Override default strategy for this translation

        Returns:
            TranslationResult with code and metadata
        """
        import time

        start_time = time.time()
        strategy = strategy or self.default_strategy

        # Check cache first
        if self.enable_caching and self.translation_cache is not None:
            cache_key = self._generate_cache_key(test_file, strategy)
            if cache_key in self.translation_cache:
                cached_result = self.translation_cache[cache_key]
                self.logger.debug(f"Using cached translation for {test_file.class_name}")
                return cast(TranslationResult, cached_result)

        # Determine optimal strategy based on test complexity
        optimal_strategy = self._determine_optimal_strategy(test_file, strategy)

        try:
            # Execute translation based on strategy
            result = self._execute_translation_strategy(test_file, optimal_strategy)
            result.translation_time = time.time() - start_time

            # Cache the result
            if self.enable_caching and self.translation_cache is not None:
                self.translation_cache[cache_key] = result

            # Update statistics
            self._update_stats(result)

            return result

        except Exception as e:
            self.stats["failures"] += 1
            self.logger.error(f"Translation failed for {test_file.class_name}: {str(e)}")

            return TranslationResult(
                translated_code="# Translation failed\npass",
                strategy_used=optimal_strategy,
                utility_attempted=True,
                llm_attempted=bool(self.llm_translator),
                confidence_score=0.0,
                translation_time=time.time() - start_time,
                errors=[str(e)],
                warnings=[],
                improvement_suggestions=[],
            )

    def translate_test_method(self, method_code: str) -> str:
        """
        Translate a single test method using hybrid approach.

        Args:
            method_code: Java test method code

        Returns:
            Translated Python method code
        """
        # Try utility first for method-level translation
        try:
            return cast(str, self.utility_translator.translate_test_method(method_code))
        except Exception as e:
            if self.llm_translator:
                return self.llm_translator.translate_test_method(method_code)
            else:
                raise TranslationException("Both utility and LLM translation failed") from e

    def translate_assertions(self, assertion_code: str) -> str:
        """
        Translate assertions using hybrid approach.

        Args:
            assertion_code: Java assertion code

        Returns:
            Translated Python assertion code
        """
        # Try utility first for assertions
        try:
            return cast(str, self.utility_translator.translate_assertions(assertion_code))
        except Exception as e:
            if self.llm_translator:
                return self.llm_translator.translate_assertions(assertion_code)
            else:
                raise TranslationException(
                    "Both utility and LLM assertion translation failed"
                ) from e

    def _determine_optimal_strategy(
        self, test_file: TestFile, requested_strategy: TranslationStrategy
    ) -> TranslationStrategy:
        """
        Determine the optimal translation strategy based on test complexity.

        Args:
            test_file: The test file to analyze
            requested_strategy: The requested strategy

        Returns:
            Optimal strategy to use
        """
        # If LLM not available, force utility-only
        if not self.llm_translator and requested_strategy != TranslationStrategy.UTILITY_ONLY:
            self.logger.warning("LLM not available, falling back to utility-only strategy")
            return TranslationStrategy.UTILITY_ONLY

        # Analyze test complexity
        complexity_score = self._calculate_complexity_score(test_file)

        # Auto-select strategy based on complexity
        if requested_strategy in [
            TranslationStrategy.HYBRID_UTILITY_FIRST,
            TranslationStrategy.HYBRID_LLM_FIRST,
        ]:
            if complexity_score < 0.3:  # Simple tests
                return TranslationStrategy.UTILITY_ONLY
            elif complexity_score > 0.7:  # Complex tests
                return TranslationStrategy.LLM_ONLY
            else:  # Medium complexity
                return requested_strategy

        return requested_strategy

    def _calculate_complexity_score(self, test_file: TestFile) -> float:
        """
        Calculate complexity score for a test file (0.0 = simple, 1.0 = complex).

        Args:
            test_file: The test file to analyze

        Returns:
            Complexity score between 0.0 and 1.0
        """
        score = 0.0

        # Mock usage increases complexity
        if test_file.mock_usage:
            score += 0.2 + (len(test_file.mock_usage) * 0.1)

        # Integration tests are more complex
        if test_file.test_type == TestType.INTEGRATION:
            score += 0.3

        # Many test methods increase complexity
        if len(test_file.test_methods) > 5:
            score += 0.2

        # Complex method bodies increase complexity
        for method in test_file.test_methods:
            if method.body and len(method.body.split("\n")) > 10:
                score += 0.1

            # Complex assertions
            if any(
                keyword in method.body for keyword in ["when(", "verify(", "thenReturn", "doThrow"]
            ):
                score += 0.2

        # Dependencies increase complexity
        if len(test_file.dependencies) > 5:
            score += 0.1

        return min(score, 1.0)  # Cap at 1.0

    def _execute_translation_strategy(
        self, test_file: TestFile, strategy: TranslationStrategy
    ) -> TranslationResult:
        """Execute the specified translation strategy."""

        if strategy == TranslationStrategy.UTILITY_ONLY:
            return self._translate_utility_only(test_file)

        elif strategy == TranslationStrategy.LLM_ONLY:
            return self._translate_llm_only(test_file)

        elif strategy == TranslationStrategy.HYBRID_UTILITY_FIRST:
            return self._translate_hybrid_utility_first(test_file)

        elif strategy == TranslationStrategy.HYBRID_LLM_FIRST:
            return self._translate_hybrid_llm_first(test_file)

        elif strategy == TranslationStrategy.UTILITY_WITH_LLM_VALIDATION:
            return self._translate_utility_with_llm_validation(test_file)

        else:
            raise ValueError(f"Unknown translation strategy: {strategy}")

    def _translate_utility_only(self, test_file: TestFile) -> TranslationResult:
        """Translate using only utility-based translator."""
        try:
            translated_code = self.utility_translator.translate_test_file(test_file)
            validation_errors = self.utility_translator.validate_generated_file(translated_code)

            return TranslationResult(
                translated_code=translated_code,
                strategy_used=TranslationStrategy.UTILITY_ONLY,
                utility_attempted=True,
                llm_attempted=False,
                confidence_score=0.9 if not validation_errors else 0.7,
                translation_time=0.0,  # Will be set by caller
                errors=validation_errors,
                warnings=[],
                improvement_suggestions=[],
            )
        except Exception as e:
            raise TranslationException(f"Utility translation failed: {str(e)}") from e

    def _translate_llm_only(self, test_file: TestFile) -> TranslationResult:
        """Translate using only LLM-based translator."""
        if not self.llm_translator:
            raise TranslationException("LLM translator not available")

        try:
            llm_result = self.llm_translator.translate_with_detailed_result(test_file)

            return TranslationResult(
                translated_code=llm_result.translated_code,
                strategy_used=TranslationStrategy.LLM_ONLY,
                utility_attempted=False,
                llm_attempted=True,
                confidence_score=llm_result.confidence_score,
                translation_time=0.0,  # Will be set by caller
                errors=[],
                warnings=[],
                improvement_suggestions=llm_result.suggested_improvements,
            )
        except Exception as e:
            raise TranslationException(f"LLM translation failed: {str(e)}") from e

    def _translate_hybrid_utility_first(self, test_file: TestFile) -> TranslationResult:
        """Translate using utility first, fall back to LLM if needed."""
        errors = []
        warnings = []

        # Try utility translation first
        try:
            utility_result = self._translate_utility_only(test_file)

            # Check if utility translation is good enough
            if utility_result.confidence_score >= 0.8 and not utility_result.errors:
                utility_result.strategy_used = TranslationStrategy.HYBRID_UTILITY_FIRST
                return utility_result
            else:
                warnings.append("Utility translation had issues, trying LLM")
                errors.extend(utility_result.errors)

        except Exception as e:
            warnings.append(f"Utility translation failed: {str(e)}")

        # Fall back to LLM
        if self.llm_translator:
            try:
                llm_result = self._translate_llm_only(test_file)
                llm_result.strategy_used = TranslationStrategy.HYBRID_UTILITY_FIRST
                llm_result.utility_attempted = True
                llm_result.warnings = warnings
                return llm_result

            except Exception as e:
                errors.append(f"LLM fallback failed: {str(e)}")

        raise TranslationException(f"Both utility and LLM translation failed: {errors}")

    def _translate_hybrid_llm_first(self, test_file: TestFile) -> TranslationResult:
        """Translate using LLM first, fall back to utility if needed."""
        errors = []
        warnings = []

        # Try LLM translation first
        if self.llm_translator:
            try:
                llm_result = self._translate_llm_only(test_file)

                # Check if LLM translation is good enough
                if llm_result.confidence_score >= 0.8:
                    llm_result.strategy_used = TranslationStrategy.HYBRID_LLM_FIRST
                    return llm_result
                else:
                    warnings.append("LLM translation had low confidence, trying utility")

            except Exception as e:
                warnings.append(f"LLM translation failed: {str(e)}")

        # Fall back to utility
        try:
            utility_result = self._translate_utility_only(test_file)
            utility_result.strategy_used = TranslationStrategy.HYBRID_LLM_FIRST
            utility_result.llm_attempted = True
            utility_result.warnings = warnings
            return utility_result

        except Exception as e:
            errors.append(f"Utility fallback failed: {str(e)}")

        raise TranslationException(f"Both LLM and utility translation failed: {errors}")

    def _translate_utility_with_llm_validation(self, test_file: TestFile) -> TranslationResult:
        """Translate with utility then validate and enhance with LLM."""
        # Get utility translation
        utility_result = self._translate_utility_only(test_file)

        # If no LLM available, return utility result
        if not self.llm_translator:
            utility_result.strategy_used = TranslationStrategy.UTILITY_WITH_LLM_VALIDATION
            utility_result.warnings = ["LLM validation not available"]
            return utility_result

        # Enhance with LLM
        try:
            enhanced_result = self.llm_translator.enhance_utility_translation(
                utility_result.translated_code, test_file
            )

            return TranslationResult(
                translated_code=enhanced_result.translated_code,
                strategy_used=TranslationStrategy.UTILITY_WITH_LLM_VALIDATION,
                utility_attempted=True,
                llm_attempted=True,
                confidence_score=enhanced_result.confidence_score,
                translation_time=0.0,  # Will be set by caller
                errors=utility_result.errors,
                warnings=[],
                improvement_suggestions=enhanced_result.suggested_improvements,
            )

        except Exception as e:
            # Return utility result with warning if LLM enhancement fails
            utility_result.strategy_used = TranslationStrategy.UTILITY_WITH_LLM_VALIDATION
            utility_result.llm_attempted = True
            utility_result.warnings = [f"LLM enhancement failed: {str(e)}"]
            return utility_result

    def _generate_cache_key(self, test_file: TestFile, strategy: TranslationStrategy) -> str:
        """Generate cache key for translation result."""
        import hashlib

        # Create hash from test file content and strategy
        content = f"{test_file.class_name}_{test_file.package}_{len(test_file.test_methods)}_{strategy.value}"
        return hashlib.md5(content.encode()).hexdigest()

    def _update_stats(self, result: TranslationResult) -> None:
        """Update translation statistics."""
        self.stats["total_translations"] += 1

        if result.strategy_used == TranslationStrategy.UTILITY_ONLY:
            self.stats["utility_successes"] += 1
        elif result.strategy_used == TranslationStrategy.LLM_ONLY:
            self.stats["llm_successes"] += 1
        else:
            self.stats["hybrid_successes"] += 1

    def get_translation_stats(self) -> dict[str, Any]:
        """Get translation statistics."""
        total = self.stats["total_translations"]
        if total == 0:
            return self.stats

        return {
            **self.stats,
            "utility_success_rate": self.stats["utility_successes"] / total,
            "llm_success_rate": self.stats["llm_successes"] / total,
            "hybrid_success_rate": self.stats["hybrid_successes"] / total,
            "failure_rate": self.stats["failures"] / total,
            "cache_size": len(self.translation_cache) if self.translation_cache else 0,
        }

    def clear_cache(self) -> None:
        """Clear translation cache."""
        if self.translation_cache:
            self.translation_cache.clear()

    def configure_strategy(self, strategy: TranslationStrategy) -> None:
        """Configure default translation strategy."""
        self.default_strategy = strategy
        self.logger.info(f"Default translation strategy set to: {strategy.value}")

    def is_llm_available(self) -> bool:
        """Check if LLM translator is available."""
        return self.llm_translator is not None
