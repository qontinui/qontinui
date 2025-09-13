"""
Hybrid test translator that combines utility-based and LLM-based translation.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    from ..core.interfaces import TestTranslator
    from ..core.models import TestFile, TestMethod, TestType
    from .python_test_generator import PythonTestGenerator
    from .llm_test_translator import LLMTestTranslator
except ImportError:
    # For standalone testing
    from core.interfaces import TestTranslator
    from core.models import TestFile, TestMethod, TestType
    from .python_test_generator import PythonTestGenerator
    from .llm_test_translator import LLMTestTranslator


class TranslationResult:
    """Result of a translation attempt."""
    
    def __init__(self, 
                 content: str, 
                 method: str, 
                 confidence: float, 
                 errors: List[str] = None,
                 execution_time: float = 0.0):
        self.content = content
        self.method = method  # 'utility', 'llm', or 'hybrid'
        self.confidence = confidence
        self.errors = errors or []
        self.execution_time = execution_time
        self.success = len(self.errors) == 0


class HybridTestTranslator(TestTranslator):
    """
    Hybrid translator that uses utility-based translation first, 
    then falls back to LLM for complex cases.
    
    This approach provides:
    - Fast, deterministic translation for common patterns
    - Intelligent handling of complex edge cases
    - Cost optimization by using LLM selectively
    - High reliability through fallback mechanisms
    """
    
    def __init__(self, 
                 llm_client=None, 
                 utility_confidence_threshold: float = 0.8,
                 llm_confidence_threshold: float = 0.7,
                 enable_llm_validation: bool = True):
        """
        Initialize the hybrid translator.
        
        Args:
            llm_client: LLM client for complex translations
            utility_confidence_threshold: Minimum confidence to accept utility translation
            llm_confidence_threshold: Minimum confidence to accept LLM translation
            enable_llm_validation: Whether to use LLM to validate utility translations
        """
        self.utility_translator = PythonTestGenerator()
        self.llm_translator = LLMTestTranslator(llm_client=llm_client)
        
        self.utility_confidence_threshold = utility_confidence_threshold
        self.llm_confidence_threshold = llm_confidence_threshold
        self.enable_llm_validation = enable_llm_validation and llm_client is not None
        
        self.logger = logging.getLogger(__name__)
        
        # Statistics tracking
        self.stats = {
            'utility_success': 0,
            'llm_fallback': 0,
            'hybrid_success': 0,
            'total_translations': 0,
            'total_time': 0.0
        }
    
    def translate_test_file(self, test_file: TestFile) -> str:
        """
        Translate a test file using the hybrid approach.
        
        Args:
            test_file: The TestFile object to translate
            
        Returns:
            Translated Python test file content
        """
        start_time = time.time()
        
        self.stats['total_translations'] += 1
        
        try:
            # Phase 1: Try utility-based translation
            utility_result = self._try_utility_translation(test_file)
            
            # Phase 2: Evaluate utility result
            if utility_result.success and utility_result.confidence >= self.utility_confidence_threshold:
                self.logger.info(f"Utility translation successful for {test_file.class_name}")
                self.stats['utility_success'] += 1
                
                # Optional: LLM validation of utility result
                if self.enable_llm_validation:
                    validated_result = self._validate_with_llm(utility_result, test_file)
                    if validated_result:
                        return validated_result.content
                
                return utility_result.content
            
            # Phase 3: Fall back to LLM translation
            self.logger.info(f"Falling back to LLM for {test_file.class_name} (utility confidence: {utility_result.confidence:.2f})")
            llm_result = self._try_llm_translation(test_file)
            
            if llm_result.success and llm_result.confidence >= self.llm_confidence_threshold:
                self.logger.info(f"LLM translation successful for {test_file.class_name}")
                self.stats['llm_fallback'] += 1
                return llm_result.content
            
            # Phase 4: Hybrid approach - combine both results
            self.logger.info(f"Using hybrid approach for {test_file.class_name}")
            hybrid_result = self._create_hybrid_translation(utility_result, llm_result, test_file)
            self.stats['hybrid_success'] += 1
            
            return hybrid_result.content
            
        except Exception as e:
            self.logger.error(f"Translation failed for {test_file.class_name}: {str(e)}")
            # Return best effort result
            return self._create_fallback_translation(test_file)
            
        finally:
            execution_time = time.time() - start_time
            self.stats['total_time'] += execution_time
            self.logger.debug(f"Translation completed in {execution_time:.2f}s")
    
    def translate_test_method(self, method_code: str) -> str:
        """
        Translate a single test method using hybrid approach.
        
        Args:
            method_code: Java test method code
            
        Returns:
            Python test method code
        """
        # Try utility first
        try:
            utility_result = self.utility_translator.translate_test_method(method_code)
            
            # Simple validation - check if it looks reasonable
            if (utility_result and 
                'def test_' in utility_result and 
                'assert' in utility_result):
                return utility_result
        except Exception as e:
            self.logger.debug(f"Utility method translation failed: {e}")
        
        # Fall back to LLM
        try:
            return self.llm_translator.translate_test_method(method_code)
        except Exception as e:
            self.logger.error(f"LLM method translation failed: {e}")
            return f"# TODO: Failed to translate method\n# Original: {method_code}\npass"
    
    def translate_assertions(self, assertion_code: str) -> str:
        """
        Translate assertions using hybrid approach.
        
        Args:
            assertion_code: Java assertion code
            
        Returns:
            Python assertion code
        """
        # Try utility first (it's good at simple assertions)
        utility_result = self.utility_translator.translate_assertions(assertion_code)
        
        # If utility produced a reasonable result, use it
        if utility_result and 'assert' in utility_result:
            return utility_result
        
        # Fall back to LLM for complex assertions
        try:
            return self.llm_translator.translate_assertions(assertion_code)
        except Exception as e:
            self.logger.error(f"Assertion translation failed: {e}")
            return f"# TODO: Failed to translate assertion: {assertion_code}"
    
    def _try_utility_translation(self, test_file: TestFile) -> TranslationResult:
        """Try utility-based translation and evaluate the result."""
        start_time = time.time()
        
        try:
            content = self.utility_translator.generate_python_test_file(test_file)
            errors = self.utility_translator.validate_generated_file(content)
            confidence = self._calculate_utility_confidence(test_file, content, errors)
            
            execution_time = time.time() - start_time
            
            return TranslationResult(
                content=content,
                method='utility',
                confidence=confidence,
                errors=errors,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TranslationResult(
                content="",
                method='utility',
                confidence=0.0,
                errors=[f"Utility translation failed: {str(e)}"],
                execution_time=execution_time
            )
    
    def _try_llm_translation(self, test_file: TestFile) -> TranslationResult:
        """Try LLM-based translation and evaluate the result."""
        start_time = time.time()
        
        try:
            content = self.llm_translator.translate_test_file(test_file)
            errors = self.llm_translator.validate_translation(content, test_file)
            confidence = self.llm_translator.get_translation_confidence(test_file)
            
            execution_time = time.time() - start_time
            
            return TranslationResult(
                content=content,
                method='llm',
                confidence=confidence,
                errors=errors,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TranslationResult(
                content="",
                method='llm',
                confidence=0.0,
                errors=[f"LLM translation failed: {str(e)}"],
                execution_time=execution_time
            )
    
    def _calculate_utility_confidence(self, test_file: TestFile, content: str, errors: List[str]) -> float:
        """Calculate confidence score for utility translation."""
        if errors:
            return 0.0
        
        confidence = 1.0
        
        # Reduce confidence for complex patterns that utility might miss
        complexity_factors = [
            (len(test_file.test_methods) > 10, 0.1),  # Many methods
            (test_file.mock_usage and len(test_file.mock_usage) > 3, 0.2),  # Complex mocking
            (test_file.test_type == TestType.INTEGRATION, 0.1),  # Integration tests
            (any(len(m.body.split('\n')) > 15 for m in test_file.test_methods if m.body), 0.2),  # Long methods
        ]
        
        for condition, penalty in complexity_factors:
            if condition:
                confidence -= penalty
        
        # Check if content looks complete
        content_checks = [
            ('import pytest' in content, 0.1),
            ('class Test' in content, 0.2),
            ('def test_' in content, 0.2),
            ('assert' in content, 0.1),
        ]
        
        for check, penalty in content_checks:
            if not check:
                confidence -= penalty
        
        return max(0.0, confidence)
    
    def _validate_with_llm(self, utility_result: TranslationResult, test_file: TestFile) -> Optional[TranslationResult]:
        """Use LLM to validate and potentially improve utility translation."""
        try:
            validation_prompt = f"""
Review this Python test translation and suggest improvements:

## Original Java Test Info
- Class: {test_file.class_name}
- Test Type: {test_file.test_type.value}
- Methods: {len(test_file.test_methods)}

## Generated Python Code
```python
{utility_result.content}
```

## Review Criteria
1. Correctness of pytest syntax
2. Proper assertion conversions
3. Mock usage accuracy
4. Missing imports or setup
5. Code style and conventions

If the code is good as-is, respond with "APPROVED".
If improvements are needed, provide the corrected Python code.
"""
            
            response = self.llm_translator._call_llm(validation_prompt)
            
            if "APPROVED" in response.upper():
                return utility_result
            
            # Extract improved code
            improved_code = self.llm_translator._extract_python_code(response)
            if improved_code and improved_code != utility_result.content:
                errors = self.llm_translator.validate_translation(improved_code, test_file)
                if not errors:
                    return TranslationResult(
                        content=improved_code,
                        method='utility_llm_validated',
                        confidence=min(1.0, utility_result.confidence + 0.1),
                        errors=[],
                        execution_time=utility_result.execution_time
                    )
            
        except Exception as e:
            self.logger.warning(f"LLM validation failed: {e}")
        
        return None
    
    def _create_hybrid_translation(self, 
                                 utility_result: TranslationResult, 
                                 llm_result: TranslationResult, 
                                 test_file: TestFile) -> TranslationResult:
        """Create a hybrid translation combining both approaches."""
        
        # If one result is clearly better, use it
        if utility_result.success and not llm_result.success:
            return utility_result
        if llm_result.success and not utility_result.success:
            return llm_result
        
        # Both failed - create minimal working version
        if not utility_result.success and not llm_result.success:
            content = self._create_fallback_translation(test_file)
            return TranslationResult(
                content=content,
                method='hybrid_fallback',
                confidence=0.3,
                errors=["Both utility and LLM translation failed"],
                execution_time=utility_result.execution_time + llm_result.execution_time
            )
        
        # Both succeeded - use the result with higher confidence
        if utility_result.confidence >= llm_result.confidence:
            return utility_result
        else:
            return llm_result
    
    def _create_fallback_translation(self, test_file: TestFile) -> str:
        """Create a minimal working Python test as fallback."""
        class_name = f"Test{test_file.class_name.replace('Test', '')}"
        
        content = f'''"""
Fallback translation for {test_file.class_name}.
Manual review and completion required.
"""

import pytest
from unittest.mock import Mock
from qontinui.core import QontinuiCore


class {class_name}:
    """Migrated from {test_file.class_name}."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.qontinui = QontinuiCore()
    
'''
        
        # Add placeholder test methods
        for test_method in test_file.test_methods:
            method_name = f"test_{test_method.name.replace('test', '').replace('Test', '').lower()}"
            content += f'''
    def {method_name}(self):
        """TODO: Implement test for {test_method.name}."""
        # Original Java method: {test_method.name}
        # TODO: Translate the following Java code:
        # {test_method.body[:200] + '...' if test_method.body and len(test_method.body) > 200 else test_method.body or 'No body available'}
        pytest.skip("Translation incomplete - manual review required")
    
'''
        
        return content
    
    def get_translation_stats(self) -> Dict:
        """Get translation statistics."""
        total = self.stats['total_translations']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'utility_success_rate': self.stats['utility_success'] / total,
            'llm_fallback_rate': self.stats['llm_fallback'] / total,
            'hybrid_success_rate': self.stats['hybrid_success'] / total,
            'average_time': self.stats['total_time'] / total
        }
    
    def configure_thresholds(self, 
                           utility_threshold: float = None, 
                           llm_threshold: float = None):
        """Configure confidence thresholds."""
        if utility_threshold is not None:
            self.utility_confidence_threshold = utility_threshold
        if llm_threshold is not None:
            self.llm_confidence_threshold = llm_threshold
    
    def reset_stats(self):
        """Reset translation statistics."""
        self.stats = {
            'utility_success': 0,
            'llm_fallback': 0,
            'hybrid_success': 0,
            'total_translations': 0,
            'total_time': 0.0
        }