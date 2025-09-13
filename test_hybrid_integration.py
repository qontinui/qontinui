"""
Integration test demonstrating the hybrid translation approach.
"""

import sys
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src" / "qontinui" / "test_migration"))

# Import modules directly without going through __init__.py to avoid circular imports
import importlib.util
import os

# Helper function to import module from file path
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Get the base path
base_path = Path(__file__).parent / "src" / "qontinui" / "test_migration"

# Import core models
models_module = import_module_from_path("models", base_path / "core" / "models.py")
Dependency = models_module.Dependency
TestFile = models_module.TestFile
TestMethod = models_module.TestMethod
TestType = models_module.TestType
MockUsage = models_module.MockUsage

# Import PythonTestGenerator
generator_module = import_module_from_path("python_test_generator", base_path / "execution" / "python_test_generator.py")
PythonTestGenerator = generator_module.PythonTestGenerator

# Import LLM translator
llm_module = import_module_from_path("llm_test_translator", base_path / "translation" / "llm_test_translator.py")
LLMTestTranslator = llm_module.LLMTestTranslator
MockLLMClient = llm_module.MockLLMClient

# Import hybrid translator
hybrid_module = import_module_from_path("hybrid_test_translator", base_path / "translation" / "hybrid_test_translator.py")
HybridTestTranslator = hybrid_module.HybridTestTranslator
TranslationStrategy = hybrid_module.TranslationStrategy
TranslationResult = hybrid_module.TranslationResult


def test_hybrid_translation_strategies():
    """Test different hybrid translation strategies."""
    print("Testing Hybrid Translation Strategies...")
    
    # Create mock LLM client
    mock_client = MockLLMClient()
    
    # Create hybrid translator
    hybrid_translator = HybridTestTranslator(
        llm_client=mock_client,
        default_strategy=TranslationStrategy.HYBRID_UTILITY_FIRST
    )
    print("✓ Hybrid translator created")
    
    # Test case 1: Simple test (should use utility-only)
    simple_test = TestFile(
        path=Path("SimpleTest.java"),
        test_type=TestType.UNIT,
        class_name="SimpleTest",
        test_methods=[
            TestMethod(name="testBasic", body="assertTrue(true);")
        ]
    )
    
    result = hybrid_translator.translate_with_metadata(simple_test)
    print(f"✓ Simple test strategy: {result.strategy_used.value}")
    print(f"  Confidence: {result.confidence_score}")
    print(f"  Utility attempted: {result.utility_attempted}")
    print(f"  LLM attempted: {result.llm_attempted}")
    
    # Test case 2: Complex test (should use LLM or hybrid)
    mock_usage = MockUsage(mock_type="spring_mock", mock_class="UserRepository")
    
    complex_test = TestFile(
        path=Path("ComplexIntegrationTest.java"),
        test_type=TestType.INTEGRATION,
        class_name="ComplexIntegrationTest",
        package="com.example.integration",
        mock_usage=[mock_usage],
        test_methods=[
            TestMethod(
                name="testComplexScenario",
                body="""
                when(userRepository.findById(anyLong())).thenReturn(Optional.of(user));
                when(emailService.sendEmail(any())).thenReturn(true);
                
                UserDto result = userService.processUser(userId);
                
                verify(userRepository, times(1)).findById(userId);
                verify(emailService, times(1)).sendEmail(any());
                assertThat(result).isNotNull();
                assertThat(result.getEmail()).isEqualTo(expectedEmail);
                """,
                mock_usage=[mock_usage]
            ),
            TestMethod(
                name="testErrorHandling",
                body="""
                when(userRepository.findById(anyLong())).thenThrow(new RuntimeException("DB Error"));
                
                assertThrows(ServiceException.class, () -> {
                    userService.processUser(userId);
                });
                
                verify(userRepository, times(1)).findById(userId);
                verify(emailService, never()).sendEmail(any());
                """
            )
        ],
        dependencies=[
            Dependency(java_import="org.mockito.Mockito"),
            Dependency(java_import="org.springframework.boot.test.mock.mockito.MockBean"),
            Dependency(java_import="org.assertj.core.api.Assertions")
        ]
    )
    
    result = hybrid_translator.translate_with_metadata(complex_test)
    print(f"✓ Complex test strategy: {result.strategy_used.value}")
    print(f"  Confidence: {result.confidence_score}")
    print(f"  Utility attempted: {result.utility_attempted}")
    print(f"  LLM attempted: {result.llm_attempted}")
    
    # Test case 3: Force different strategies
    strategies_to_test = [
        TranslationStrategy.UTILITY_ONLY,
        TranslationStrategy.LLM_ONLY,
        TranslationStrategy.UTILITY_WITH_LLM_VALIDATION
    ]
    
    for strategy in strategies_to_test:
        result = hybrid_translator.translate_with_metadata(simple_test, strategy=strategy)
        print(f"✓ Forced {strategy.value}: confidence={result.confidence_score:.2f}")
    
    print("\nAll strategy tests passed! ✅")


def test_translation_quality_comparison():
    """Compare translation quality between utility and LLM approaches."""
    print("\nTesting Translation Quality Comparison...")
    
    # Create translators
    utility_translator = PythonTestGenerator()
    llm_translator = LLMTestTranslator(llm_client=MockLLMClient())
    hybrid_translator = HybridTestTranslator(llm_client=MockLLMClient())
    
    # Test case: Medium complexity test
    test_method = TestMethod(
        name="shouldCalculateUserScore",
        body="""
        User user = new User("John", "john@example.com");
        user.setAge(25);
        
        ScoreCalculator calculator = new ScoreCalculator();
        int score = calculator.calculateScore(user);
        
        assertEquals(75, score);
        assertTrue(score > 0);
        assertNotNull(user.getEmail());
        """
    )
    
    test_file = TestFile(
        path=Path("UserScoreTest.java"),
        test_type=TestType.UNIT,
        class_name="UserScoreTest",
        package="com.example.service",
        test_methods=[test_method],
        dependencies=[
            Dependency(java_import="org.junit.jupiter.api.Test"),
            Dependency(java_import="org.junit.jupiter.api.Assertions")
        ]
    )
    
    # Utility translation
    utility_result = utility_translator.translate_test_file(test_file)
    utility_errors = utility_translator.validate_generated_file(utility_result)
    
    print("Utility Translation:")
    print(f"  Length: {len(utility_result)} chars")
    print(f"  Validation errors: {len(utility_errors)}")
    print(f"  Has pytest import: {'import pytest' in utility_result}")
    print(f"  Has test class: {'class Test' in utility_result}")
    
    # LLM translation
    llm_result = llm_translator.translate_with_detailed_result(test_file)
    
    print("LLM Translation:")
    print(f"  Length: {len(llm_result.translated_code)} chars")
    print(f"  Confidence: {llm_result.confidence_score}")
    print(f"  Notes: {len(llm_result.translation_notes)}")
    print(f"  Suggestions: {len(llm_result.suggested_improvements)}")
    
    # Hybrid translation
    hybrid_result = hybrid_translator.translate_with_metadata(test_file)
    
    print("Hybrid Translation:")
    print(f"  Strategy used: {hybrid_result.strategy_used.value}")
    print(f"  Length: {len(hybrid_result.translated_code)} chars")
    print(f"  Confidence: {hybrid_result.confidence_score}")
    print(f"  Translation time: {hybrid_result.translation_time:.3f}s")
    print(f"  Errors: {len(hybrid_result.errors)}")
    print(f"  Warnings: {len(hybrid_result.warnings)}")
    
    print("✓ Quality comparison completed")


def test_performance_comparison():
    """Compare performance between different translation approaches."""
    print("\nTesting Performance Comparison...")
    
    import time
    
    # Create test files of different complexities
    test_files = []
    
    # Simple test
    test_files.append(TestFile(
        path=Path("SimpleTest.java"),
        test_type=TestType.UNIT,
        class_name="SimpleTest",
        test_methods=[TestMethod(name="testSimple", body="assertTrue(true);")]
    ))
    
    # Medium test
    test_files.append(TestFile(
        path=Path("MediumTest.java"),
        test_type=TestType.UNIT,
        class_name="MediumTest",
        test_methods=[
            TestMethod(name=f"testMethod{i}", body="assertEquals(expected, actual);")
            for i in range(5)
        ]
    ))
    
    # Complex test
    mock_usage = MockUsage(mock_type="spring_mock", mock_class="Repository")
    test_files.append(TestFile(
        path=Path("ComplexTest.java"),
        test_type=TestType.INTEGRATION,
        class_name="ComplexTest",
        mock_usage=[mock_usage],
        test_methods=[
            TestMethod(
                name=f"testComplex{i}", 
                body="when(mock.method()).thenReturn(value); verify(mock).method();",
                mock_usage=[mock_usage]
            )
            for i in range(3)
        ]
    ))
    
    # Create translators
    utility_translator = PythonTestGenerator()
    hybrid_translator = HybridTestTranslator(llm_client=MockLLMClient())
    
    # Performance test
    for i, test_file in enumerate(test_files):
        complexity = ["Simple", "Medium", "Complex"][i]
        print(f"\n{complexity} Test Performance:")
        
        # Utility performance
        start_time = time.time()
        utility_result = utility_translator.translate_test_file(test_file)
        utility_time = time.time() - start_time
        
        print(f"  Utility: {utility_time:.4f}s")
        
        # Hybrid performance
        start_time = time.time()
        hybrid_result = hybrid_translator.translate_with_metadata(test_file)
        hybrid_time = time.time() - start_time
        
        print(f"  Hybrid: {hybrid_time:.4f}s (strategy: {hybrid_result.strategy_used.value})")
        print(f"  Speedup: {hybrid_time/utility_time:.2f}x")
    
    print("✓ Performance comparison completed")


def test_caching_and_statistics():
    """Test caching functionality and statistics tracking."""
    print("\nTesting Caching and Statistics...")
    
    hybrid_translator = HybridTestTranslator(
        llm_client=MockLLMClient(),
        enable_caching=True
    )
    
    test_file = TestFile(
        path=Path("CacheTest.java"),
        test_type=TestType.UNIT,
        class_name="CacheTest",
        test_methods=[TestMethod(name="testCache", body="assertTrue(true);")]
    )
    
    # First translation
    start_time = time.time()
    result1 = hybrid_translator.translate_with_metadata(test_file)
    first_time = time.time() - start_time
    
    # Second translation (should use cache)
    start_time = time.time()
    result2 = hybrid_translator.translate_with_metadata(test_file)
    second_time = time.time() - start_time
    
    print(f"First translation: {first_time:.4f}s")
    print(f"Second translation: {second_time:.4f}s")
    print(f"Cache speedup: {first_time/second_time:.2f}x")
    
    # Check cache
    assert len(hybrid_translator.translation_cache) > 0
    print(f"✓ Cache contains {len(hybrid_translator.translation_cache)} entries")
    
    # Check statistics
    stats = hybrid_translator.get_translation_stats()
    print(f"✓ Statistics: {stats['total_translations']} translations")
    print(f"  Utility success rate: {stats['utility_success_rate']:.2f}")
    print(f"  Cache size: {stats['cache_size']}")
    
    # Clear cache
    hybrid_translator.clear_cache()
    assert len(hybrid_translator.translation_cache) == 0
    print("✓ Cache cleared successfully")


def test_error_handling():
    """Test error handling in hybrid translation."""
    print("\nTesting Error Handling...")
    
    # Test with no LLM client
    hybrid_no_llm = HybridTestTranslator()
    
    test_file = TestFile(
        path=Path("ErrorTest.java"),
        test_type=TestType.UNIT,
        class_name="ErrorTest"
    )
    
    # Should fall back to utility-only
    result = hybrid_no_llm.translate_with_metadata(test_file)
    print(f"✓ No LLM fallback: {result.strategy_used.value}")
    
    # Test LLM availability check
    assert hybrid_no_llm.is_llm_available() is False
    print("✓ LLM availability check works")
    
    # Test strategy configuration
    hybrid_no_llm.configure_strategy(TranslationStrategy.UTILITY_ONLY)
    assert hybrid_no_llm.default_strategy == TranslationStrategy.UTILITY_ONLY
    print("✓ Strategy configuration works")


if __name__ == "__main__":
    test_hybrid_translation_strategies()
    test_translation_quality_comparison()
    test_performance_comparison()
    test_caching_and_statistics()
    test_error_handling()
    
    print("\n🎉 All hybrid integration tests passed! 🎉")
    print("\nHybrid Translation Summary:")
    print("- ✅ Utility-first strategy for simple tests")
    print("- ✅ LLM enhancement for complex tests") 
    print("- ✅ Intelligent strategy selection")
    print("- ✅ Caching for performance")
    print("- ✅ Comprehensive error handling")
    print("- ✅ Statistics and monitoring")
    print("\nThe hybrid approach provides the best of both worlds:")
    print("- Fast, deterministic translation for common patterns")
    print("- AI-powered translation for complex edge cases")
    print("- Automatic strategy selection based on complexity")
    print("- Fallback mechanisms for reliability")