#!/usr/bin/env python3
"""
Example demonstrating the hybrid test translation approach.

This script shows how to use the HybridTestTranslator to convert Java tests
to Python tests using a combination of utility-based and LLM-based translation.
"""

import sys
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qontinui.test_migration.core.models import (
    MockUsage,
    TestFile,
    TestMethod,
    TestType,
)
from qontinui.test_migration.execution.hybrid_test_translator import (
    HybridTestTranslator,
)
from qontinui.test_migration.execution.llm_test_translator import LLMTestTranslator
from qontinui.test_migration.execution.python_test_generator import PythonTestGenerator


def create_sample_test_file() -> TestFile:
    """Create a sample Java test file for demonstration."""
    return TestFile(
        path=Path("CalculatorTest.java"),
        test_type=TestType.UNIT,
        class_name="CalculatorTest",
        package="com.example.calculator",
        test_methods=[
            TestMethod(
                name="testAddition",
                body="""
                Calculator calculator = new Calculator();
                int result = calculator.add(2, 3);
                assertEquals(5, result);
                """,
            ),
            TestMethod(
                name="testSubtraction",
                body="""
                Calculator calculator = new Calculator();
                int result = calculator.subtract(10, 4);
                assertEquals(6, result);
                """,
            ),
            TestMethod(
                name="testDivision",
                body="""
                Calculator calculator = new Calculator();
                double result = calculator.divide(10.0, 2.0);
                assertEquals(5.0, result, 0.001);
                """,
            ),
        ],
    )


def create_complex_test_file() -> TestFile:
    """Create a complex test file with mocks for demonstration."""
    mock_usage = MockUsage(mock_type="spring_mock", mock_class="UserRepository")

    return TestFile(
        path=Path("UserServiceTest.java"),
        test_type=TestType.INTEGRATION,
        class_name="UserServiceTest",
        package="com.example.service",
        mock_usage=[mock_usage],
        test_methods=[
            TestMethod(
                name="testFindUserByEmail",
                body="""
                // Setup mock
                User mockUser = new User("john@example.com", "John Doe");
                when(userRepository.findByEmail("john@example.com")).thenReturn(mockUser);

                // Execute test
                User result = userService.findUserByEmail("john@example.com");

                // Verify results
                assertNotNull(result);
                assertEquals("john@example.com", result.getEmail());
                assertEquals("John Doe", result.getName());

                // Verify mock interactions
                verify(userRepository, times(1)).findByEmail("john@example.com");
                """,
                mock_usage=[mock_usage],
            ),
            TestMethod(
                name="testCreateUser",
                body="""
                // Setup
                User newUser = new User("jane@example.com", "Jane Smith");
                when(userRepository.save(any(User.class))).thenReturn(newUser);

                // Execute
                User result = userService.createUser("jane@example.com", "Jane Smith");

                // Verify
                assertNotNull(result);
                assertEquals("jane@example.com", result.getEmail());
                verify(userRepository).save(any(User.class));
                """,
                mock_usage=[mock_usage],
            ),
        ],
    )


def demonstrate_utility_translation():
    """Demonstrate utility-based translation."""
    print("=" * 60)
    print("UTILITY-BASED TRANSLATION DEMO")
    print("=" * 60)

    translator = PythonTestGenerator()
    test_file = create_sample_test_file()

    print(f"Translating: {test_file.class_name}")
    print(f"Test Type: {test_file.test_type.value}")
    print(f"Methods: {len(test_file.test_methods)}")
    print()

    result = translator.translate_test_file(test_file)

    print("Generated Python Test:")
    print("-" * 40)
    print(result)
    print()

    # Validate the result
    errors = translator.validate_generated_file(result)
    if errors:
        print("Validation Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✓ Validation passed - no errors found")
    print()


def demonstrate_llm_translation():
    """Demonstrate LLM-based translation (without actual LLM client)."""
    print("=" * 60)
    print("LLM-BASED TRANSLATION DEMO")
    print("=" * 60)

    translator = LLMTestTranslator()  # No LLM client - will use mock responses
    test_file = create_complex_test_file()

    print(f"Translating: {test_file.class_name}")
    print(f"Test Type: {test_file.test_type.value}")
    print(f"Methods: {len(test_file.test_methods)}")
    print(f"Has Mocks: {bool(test_file.mock_usage)}")
    print()

    result = translator.translate_test_file(test_file)

    print("Generated Python Test:")
    print("-" * 40)
    print(result)
    print()

    # Get confidence score
    confidence = translator.get_translation_confidence(test_file)
    print(f"Translation Confidence: {confidence:.2f}")
    print()


def demonstrate_hybrid_translation():
    """Demonstrate hybrid translation approach."""
    print("=" * 60)
    print("HYBRID TRANSLATION DEMO")
    print("=" * 60)

    translator = HybridTestTranslator(
        utility_confidence_threshold=0.8, llm_confidence_threshold=0.7
    )

    # Test with simple file (should use utility)
    simple_test = create_sample_test_file()
    print(f"Translating Simple Test: {simple_test.class_name}")

    simple_result = translator.translate_test_file(simple_test)
    print("Simple Test Result (first 200 chars):")
    print(simple_result[:200] + "..." if len(simple_result) > 200 else simple_result)
    print()

    # Test with complex file (might use LLM)
    complex_test = create_complex_test_file()
    print(f"Translating Complex Test: {complex_test.class_name}")

    complex_result = translator.translate_test_file(complex_test)
    print("Complex Test Result (first 200 chars):")
    print(complex_result[:200] + "..." if len(complex_result) > 200 else complex_result)
    print()

    # Show statistics
    stats = translator.get_translation_stats()
    print("Translation Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print()


def demonstrate_method_translation():
    """Demonstrate method-level translation."""
    print("=" * 60)
    print("METHOD-LEVEL TRANSLATION DEMO")
    print("=" * 60)

    java_method = """
    @Test
    public void testComplexCalculation() {
        // Setup
        Calculator calc = new Calculator();
        double x = 10.5;
        double y = 3.2;

        // Execute
        double result = calc.multiply(x, y);

        // Verify
        assertEquals(33.6, result, 0.001);
        assertTrue(result > 0);
        assertNotNull(calc);
    }
    """

    print("Original Java Method:")
    print("-" * 30)
    print(java_method)
    print()

    translator = HybridTestTranslator()
    result = translator.translate_test_method(java_method)

    print("Translated Python Method:")
    print("-" * 30)
    print(result)
    print()


def demonstrate_assertion_translation():
    """Demonstrate assertion-level translation."""
    print("=" * 60)
    print("ASSERTION TRANSLATION DEMO")
    print("=" * 60)

    java_assertions = [
        "assertEquals(expected, actual);",
        "assertTrue(condition);",
        "assertFalse(flag);",
        "assertNull(value);",
        "assertNotNull(object);",
        "assertEquals(5.0, result, 0.001);",
    ]

    translator = HybridTestTranslator()

    print("Java Assertion -> Python Assertion")
    print("-" * 50)

    for assertion in java_assertions:
        result = translator.translate_assertions(assertion)
        print(f"{assertion:<30} -> {result}")

    print()


def main():
    """Run all demonstration examples."""
    print("HYBRID TEST TRANSLATION SYSTEM DEMONSTRATION")
    print("=" * 60)
    print()

    try:
        demonstrate_utility_translation()
        demonstrate_llm_translation()
        demonstrate_hybrid_translation()
        demonstrate_method_translation()
        demonstrate_assertion_translation()

        print("=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print()
        print("Key Benefits of the Hybrid Approach:")
        print("✓ Fast utility-based translation for simple cases")
        print("✓ LLM-powered translation for complex scenarios")
        print("✓ Intelligent strategy selection based on complexity")
        print("✓ Robust fallback mechanisms")
        print("✓ Confidence scoring and validation")
        print("✓ Comprehensive statistics tracking")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
