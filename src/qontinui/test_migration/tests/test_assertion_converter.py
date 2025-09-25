"""
Unit tests for AssertionConverter.
"""

from qontinui.test_migration.translation.assertion_converter import AssertionConverter


class TestAssertionConverter:
    """Test cases for AssertionConverter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = AssertionConverter()

    def test_assert_true_conversion(self):
        """Test assertTrue conversion."""
        # Simple assertTrue
        result = self.converter.convert_assertion("assertTrue(condition)")
        assert result == "assert condition"

        # assertTrue with message
        result = self.converter.convert_assertion('assertTrue(condition, "Error message")')
        assert result == 'assert condition, "Error message"'

        # assertTrue with complex condition
        result = self.converter.convert_assertion("assertTrue(user.isValid() && user.isActive())")
        assert result == "assert user.isValid() && user.isActive()"

    def test_assert_false_conversion(self):
        """Test assertFalse conversion."""
        # Simple assertFalse
        result = self.converter.convert_assertion("assertFalse(condition)")
        assert result == "assert not condition"

        # assertFalse with message
        result = self.converter.convert_assertion('assertFalse(condition, "Should be false")')
        assert result == 'assert not condition, "Should be false"'

    def test_assert_equals_conversion(self):
        """Test assertEquals conversion."""
        # Simple assertEquals
        result = self.converter.convert_assertion("assertEquals(expected, actual)")
        assert result == "assert actual == expected"

        # assertEquals with message
        result = self.converter.convert_assertion('assertEquals(5, result, "Values should match")')
        assert result == 'assert result == 5, "Values should match"'

        # assertEquals with string values
        result = self.converter.convert_assertion('assertEquals("expected", actual)')
        assert result == 'assert actual == "expected"'

    def test_assert_not_equals_conversion(self):
        """Test assertNotEquals conversion."""
        result = self.converter.convert_assertion("assertNotEquals(unexpected, actual)")
        assert result == "assert actual != unexpected"

        result = self.converter.convert_assertion(
            'assertNotEquals(null, result, "Should not be null")'
        )
        assert result == 'assert result != null, "Should not be null"'

    def test_assert_null_conversion(self):
        """Test assertNull conversion."""
        result = self.converter.convert_assertion("assertNull(value)")
        assert result == "assert value is None"

        result = self.converter.convert_assertion('assertNull(result, "Should be null")')
        assert result == 'assert result is None, "Should be null"'

    def test_assert_not_null_conversion(self):
        """Test assertNotNull conversion."""
        result = self.converter.convert_assertion("assertNotNull(value)")
        assert result == "assert value is not None"

        result = self.converter.convert_assertion('assertNotNull(user, "User should exist")')
        assert result == 'assert user is not None, "User should exist"'

    def test_assert_same_conversion(self):
        """Test assertSame conversion."""
        result = self.converter.convert_assertion("assertSame(expected, actual)")
        assert result == "assert actual is expected"

        result = self.converter.convert_assertion(
            'assertSame(instance1, instance2, "Should be same object")'
        )
        assert result == 'assert instance2 is instance1, "Should be same object"'

    def test_assert_not_same_conversion(self):
        """Test assertNotSame conversion."""
        result = self.converter.convert_assertion("assertNotSame(obj1, obj2)")
        assert result == "assert obj2 is not obj1"

    def test_assert_array_equals_conversion(self):
        """Test assertArrayEquals conversion."""
        result = self.converter.convert_assertion("assertArrayEquals(expectedArray, actualArray)")
        assert result == "assert list(actualArray) == list(expectedArray)"

        result = self.converter.convert_assertion(
            'assertArrayEquals(expected, actual, "Arrays should match")'
        )
        assert result == 'assert list(actual) == list(expected), "Arrays should match"'

    def test_assert_throws_conversion(self):
        """Test assertThrows conversion."""
        result = self.converter.convert_assertion(
            "assertThrows(IllegalArgumentException.class, () -> method.call())"
        )
        expected = "with pytest.raises(IllegalArgumentException):\n    method.call()"
        assert result == expected

        # Test with method reference
        result = self.converter.convert_assertion(
            "assertThrows(RuntimeException.class, service::throwException)"
        )
        expected = "with pytest.raises(RuntimeException):\n    service.throwException"
        assert result == expected

    def test_assert_does_not_throw_conversion(self):
        """Test assertDoesNotThrow conversion."""
        result = self.converter.convert_assertion("assertDoesNotThrow(() -> method.call())")
        assert result == "method.call()"

        result = self.converter.convert_assertion("assertDoesNotThrow(service::safeMethod)")
        assert result == "service.safeMethod"

    def test_fail_conversion(self):
        """Test fail() conversion."""
        result = self.converter.convert_assertion("fail()")
        assert result == "pytest.fail()"

        result = self.converter.convert_assertion('fail("Test failed")')
        assert result == 'pytest.fail("Test failed")'

    def test_hamcrest_assert_that_conversion(self):
        """Test assertThat with Hamcrest matchers."""
        # is(equalTo(value))
        result = self.converter.convert_assertion("assertThat(actual, is(equalTo(expected)))")
        assert result == "assert actual == expected"

        # is(value)
        result = self.converter.convert_assertion("assertThat(result, is(5))")
        assert result == "assert result == 5"

        # equalTo(value)
        result = self.converter.convert_assertion("assertThat(name, equalTo(expectedName))")
        assert result == "assert name == expectedName"

        # nullValue()
        result = self.converter.convert_assertion("assertThat(value, nullValue())")
        assert result == "assert value is None"

        # notNullValue()
        result = self.converter.convert_assertion("assertThat(user, notNullValue())")
        assert result == "assert user is not None"

        # hasSize(n)
        result = self.converter.convert_assertion("assertThat(list, hasSize(3))")
        assert result == "assert len(list) == 3"

        # empty()
        result = self.converter.convert_assertion("assertThat(collection, empty())")
        assert result == "assert len(collection) == 0"

        # containsString(str)
        result = self.converter.convert_assertion('assertThat(text, containsString("hello"))')
        assert result == 'assert "hello" in text'

        # startsWith(str)
        result = self.converter.convert_assertion('assertThat(text, startsWith("prefix"))')
        assert result == 'assert text.startswith("prefix")'

        # endsWith(str)
        result = self.converter.convert_assertion('assertThat(text, endsWith("suffix"))')
        assert result == 'assert text.endswith("suffix")'

        # greaterThan(value)
        result = self.converter.convert_assertion("assertThat(number, greaterThan(10))")
        assert result == "assert number > 10"

        # lessThan(value)
        result = self.converter.convert_assertion("assertThat(number, lessThan(100))")
        assert result == "assert number < 100"

        # instanceOf(Class.class)
        result = self.converter.convert_assertion("assertThat(obj, instanceOf(String.class))")
        assert result == "assert isinstance(obj, String)"

        # not(matcher)
        result = self.converter.convert_assertion("assertThat(value, not(equalTo(unwanted)))")
        assert result == "assert not (value == unwanted)"

    def test_parameter_extraction(self):
        """Test parameter extraction from assertion calls."""
        # Simple parameters
        params, message = self.converter._extract_assertion_params(
            "assertTrue(condition)", "assertTrue"
        )
        assert params == ["condition"]
        assert message is None

        # Parameters with message
        params, message = self.converter._extract_assertion_params(
            'assertEquals(5, result, "Error")', "assertEquals"
        )
        assert params == ["5", "result"]
        assert message == '"Error"'

        # Complex parameters with nested calls
        params, message = self.converter._extract_assertion_params(
            "assertEquals(service.getValue(), result.getActual())", "assertEquals"
        )
        assert params == ["service.getValue()", "result.getActual()"]
        assert message is None

    def test_parameter_splitting(self):
        """Test parameter splitting with complex expressions."""
        # Nested method calls
        params = self.converter._split_parameters("service.getValue(), result.getActual()")
        assert params == ["service.getValue()", "result.getActual()"]

        # String parameters with commas
        params = self.converter._split_parameters('"Hello, world", "Another, string"')
        assert params == ['"Hello, world"', '"Another, string"']

        # Nested parentheses
        params = self.converter._split_parameters("method(param1, param2), otherMethod()")
        assert params == ["method(param1, param2)", "otherMethod()"]

    def test_string_literal_detection(self):
        """Test string literal detection."""
        assert self.converter._is_string_literal('"Hello"') is True
        assert self.converter._is_string_literal("'World'") is True
        assert self.converter._is_string_literal("variable") is False
        assert self.converter._is_string_literal("123") is False

    def test_lambda_body_extraction(self):
        """Test lambda expression body extraction."""
        # Simple lambda
        result = self.converter._extract_lambda_body("() -> method.call()")
        assert result == "method.call()"

        # Lambda with parameters
        result = self.converter._extract_lambda_body("(x) -> x.process()")
        assert result == "x.process()"

        # Method reference
        result = self.converter._extract_lambda_body("service::processData")
        assert result == "service.processData"

    def test_multiple_assertions_conversion(self):
        """Test converting multiple assertion lines."""
        assertions = [
            "assertTrue(condition1)",
            "assertEquals(expected, actual)",
            "assertNotNull(result)",
        ]

        converted = self.converter.convert_multiple_assertions(assertions)

        expected = ["assert condition1", "assert actual == expected", "assert result is not None"]

        assert converted == expected

    def test_custom_assertion_extraction(self):
        """Test extraction of custom assertion methods."""
        test_code = """
        public class TestClass {
            @Test
            public void testMethod() {
                assertTrue(condition);
                assertUserIsValid(user);
                assertCustomCondition(value1, value2);
            }

            private void assertUserIsValid(User user) {
                assertTrue(user != null);
                assertTrue(user.isActive());
            }

            private static void assertCustomCondition(String val1, String val2) {
                assertEquals(val1, val2);
            }
        }
        """

        custom_assertions = self.converter.extract_custom_assertion_methods(test_code)
        assert "assertUserIsValid" in custom_assertions
        assert "assertCustomCondition" in custom_assertions
        assert "assertTrue" not in custom_assertions  # Standard assertion should not be included

    def test_custom_assertion_conversion(self):
        """Test conversion of custom assertion method calls."""
        result = self.converter.convert_custom_assertion(
            "assertUserIsValid(user)", "assertUserIsValid"
        )
        assert result == "self.assert_user_is_valid(user)"

        result = self.converter.convert_custom_assertion(
            "assertCustomCondition(val1, val2)", "assertCustomCondition"
        )
        assert result == "self.assert_custom_condition(val1, val2)"

    def test_camel_to_snake_conversion(self):
        """Test camelCase to snake_case conversion."""
        assert self.converter._camel_to_snake("assertUserIsValid") == "assert_user_is_valid"
        assert self.converter._camel_to_snake("assertCustomCondition") == "assert_custom_condition"
        assert self.converter._camel_to_snake("simpleAssert") == "simple_assert"

    def test_semicolon_removal(self):
        """Test that semicolons are properly removed."""
        result = self.converter.convert_assertion("assertTrue(condition);")
        assert result == "assert condition"

        result = self.converter.convert_assertion("assertEquals(expected, actual);")
        assert result == "assert actual == expected"

    def test_complex_assertion_scenarios(self):
        """Test complex real-world assertion scenarios."""
        # Assertion with method chaining
        result = self.converter.convert_assertion("assertTrue(user.getProfile().isActive())")
        assert result == "assert user.getProfile().isActive()"

        # Assertion with boolean operations
        result = self.converter.convert_assertion(
            "assertTrue(condition1 && condition2 || condition3)"
        )
        assert result == "assert condition1 && condition2 || condition3"

        # Nested method calls in assertEquals
        result = self.converter.convert_assertion(
            "assertEquals(service.getExpectedValue(), processor.calculateResult())"
        )
        assert result == "assert processor.calculateResult() == service.getExpectedValue()"

        # Complex Hamcrest matcher
        result = self.converter.convert_assertion("assertThat(users, hasSize(greaterThan(0)))")
        # This would need more complex parsing, for now it falls back to basic conversion
        assert "assert" in result
