"""
LLM-based test translator for handling complex Java to Python test conversions.
"""

import json
import re
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from ..core.interfaces import TestTranslator
    from ..core.models import TestFile, TestMethod, TestType
except ImportError:
    # For standalone testing
    from core.interfaces import TestTranslator
    from core.models import TestFile, TestMethod, TestType


class LLMTestTranslator(TestTranslator):
    """
    LLM-powered test translator for complex Java to Python test conversions.
    
    This class handles cases that are too complex for rule-based translation:
    - Complex mock interactions
    - Custom assertion patterns
    - Business logic embedded in tests
    - Framework-specific patterns not covered by utility translator
    """
    
    def __init__(self, llm_client=None, model_name: str = "gpt-4"):
        """
        Initialize the LLM test translator.
        
        Args:
            llm_client: LLM client instance (OpenAI, Anthropic, etc.)
            model_name: Name of the model to use
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.brobot_context = self._load_brobot_context()
        self.qontinui_context = self._load_qontinui_context()
        
    def _load_brobot_context(self) -> Dict[str, Any]:
        """Load Brobot framework context for better translation."""
        return {
            "framework_name": "Brobot",
            "language": "Java",
            "test_framework": "JUnit 5",
            "mock_framework": "Mockito",
            "key_concepts": {
                "StateObject": "Represents GUI states and elements",
                "ActionOptions": "Configuration for GUI actions",
                "Find": "Element location and interaction",
                "Mock": "Test doubles for GUI components",
                "Transitions": "State changes in GUI workflow"
            },
            "common_patterns": {
                "state_setup": "@BeforeEach setup with StateObject initialization",
                "mock_injection": "@MockBean for Spring components",
                "action_verification": "verify() calls on action mocks",
                "state_assertions": "assertEquals on state properties"
            }
        }
    
    def _load_qontinui_context(self) -> Dict[str, Any]:
        """Load Qontinui framework context for accurate translation."""
        return {
            "framework_name": "Qontinui",
            "language": "Python",
            "test_framework": "pytest",
            "mock_framework": "unittest.mock",
            "key_concepts": {
                "State": "Python equivalent of StateObject",
                "ActionConfig": "Python equivalent of ActionOptions",
                "Find": "Element location with Python syntax",
                "Mock": "Python mock objects",
                "StateTransition": "State change management"
            },
            "common_patterns": {
                "fixture_setup": "@pytest.fixture for test setup",
                "mock_patching": "@patch decorators or Mock() objects",
                "assertion_style": "assert statements instead of assertEquals",
                "state_validation": "Direct property assertions"
            }
        }
    
    def translate_test_file(self, test_file: TestFile) -> str:
        """
        Translate a complete test file using LLM.
        
        Args:
            test_file: The TestFile object containing Java test information
            
        Returns:
            Complete Python test file content as a string
        """
        if not self.llm_client:
            # Use mock response for testing/fallback
            return self._generate_mock_response_for_file(test_file)
        
        # Build comprehensive prompt
        prompt = self._build_translation_prompt(test_file)
        
        # Get LLM response
        response = self._call_llm(prompt)
        
        # Extract and clean Python code
        python_code = self._extract_python_code(response)
        
        # Post-process and validate
        python_code = self._post_process_translation(python_code, test_file)
        
        return python_code
    
    def translate_test_method(self, method_code: str) -> str:
        """
        Translate a single test method using LLM.
        
        Args:
            method_code: Java test method code
            
        Returns:
            Python test method code
        """
        if not self.llm_client:
            return self._generate_mock_method_response(method_code)
            
        prompt = f"""
Translate this Java test method to Python pytest format:

```java
{method_code}
```

Context:
- Convert from Brobot (Java) to Qontinui (Python)
- Use pytest conventions
- Convert JUnit assertions to pytest assertions
- Preserve test logic and intent
- Handle mock objects appropriately

Return only the Python method code:
"""
        
        response = self._call_llm(prompt)
        return self._extract_python_code(response)
    
    def translate_assertions(self, assertion_code: str) -> str:
        """
        Translate complex assertions using LLM context understanding.
        
        Args:
            assertion_code: Java assertion code
            
        Returns:
            Python assertion code
        """
        if not self.llm_client:
            return self._generate_mock_assertion_response(assertion_code)
            
        prompt = f"""
Translate these Java assertions to Python pytest format:

```java
{assertion_code}
```

Rules:
- assertEquals(a, b) -> assert a == b
- assertTrue(x) -> assert x
- assertFalse(x) -> assert not x
- assertNull(x) -> assert x is None
- assertNotNull(x) -> assert x is not None
- Custom Brobot assertions should map to appropriate Qontinui equivalents

Return only the Python assertion code:
"""
        
        response = self._call_llm(prompt)
        return self._extract_python_code(response)
    
    def _build_translation_prompt(self, test_file: TestFile) -> str:
        """Build a comprehensive prompt for test file translation."""
        
        # Get original Java content if available
        java_content = self._get_java_content(test_file)
        
        prompt = f"""
You are an expert in translating Java tests to Python tests. Translate the following Java test file from the Brobot framework to the Qontinui framework.

## Source Framework Context (Brobot - Java)
{json.dumps(self.brobot_context, indent=2)}

## Target Framework Context (Qontinui - Python)
{json.dumps(self.qontinui_context, indent=2)}

## Test File Information
- File: {test_file.path.name}
- Class: {test_file.class_name}
- Package: {test_file.package or 'default'}
- Test Type: {test_file.test_type.value}
- Number of test methods: {len(test_file.test_methods)}
- Uses mocks: {bool(test_file.mock_usage)}

## Java Test Content
```java
{java_content}
```

## Translation Requirements
1. **Framework Migration**: Convert from Brobot (Java) to Qontinui (Python)
2. **Test Framework**: Convert from JUnit 5 to pytest
3. **Assertions**: Convert JUnit assertions to pytest assertions
4. **Mocks**: Convert Mockito mocks to unittest.mock
5. **Imports**: Use appropriate Python imports for Qontinui
6. **Naming**: Follow Python naming conventions
7. **Structure**: Maintain test class structure with proper pytest fixtures
8. **Comments**: Preserve and translate comments explaining test intent

## Specific Mappings
- `@Test` -> remove (pytest auto-discovers test methods)
- `@BeforeEach` -> `@pytest.fixture(autouse=True)` or `setup_method`
- `@AfterEach` -> `teardown_method`
- `@MockBean` -> `@patch` or `Mock()`
- `assertEquals(a, b)` -> `assert a == b`
- `assertTrue(x)` -> `assert x`
- `assertFalse(x)` -> `assert not x`

## Expected Output Format
Provide a complete Python test file with:
1. Proper docstring explaining the migration
2. All necessary imports
3. Test class with converted methods
4. Proper pytest fixtures if needed
5. Converted assertions and mock usage

Return only the Python code, no explanations:
"""
        
        return prompt
    
    def _get_java_content(self, test_file: TestFile) -> str:
        """Get the original Java content for the test file."""
        if hasattr(test_file, 'original_content') and test_file.original_content:
            return test_file.original_content
        
        # Reconstruct from test file model
        java_content = f"""
package {test_file.package or 'com.example'};

// Imports (reconstructed)
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

public class {test_file.class_name} {{
"""
        
        # Add setup methods if any
        if test_file.setup_methods:
            for setup_method in test_file.setup_methods:
                java_content += f"""
    @BeforeEach
    public void {setup_method.name}() {{
        {setup_method.body or '// Setup logic'}
    }}
"""
        
        # Add test methods
        for test_method in test_file.test_methods:
            java_content += f"""
    @Test
    public void {test_method.name}() {{
        {test_method.body or '// Test logic'}
    }}
"""
        
        java_content += "\n}"
        return java_content
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        try:
            # This would be the actual LLM call
            # Implementation depends on the LLM client (OpenAI, Anthropic, etc.)
            response = self.llm_client.complete(
                prompt=prompt,
                model=self.model_name,
                max_tokens=2000,
                temperature=0.1  # Low temperature for consistent code generation
            )
            return response.text if hasattr(response, 'text') else str(response)
            
        except Exception as e:
            raise RuntimeError(f"LLM translation failed: {str(e)}")
    
    def _generate_mock_response_for_file(self, test_file: TestFile) -> str:
        """Generate a mock response for testing purposes."""
        python_class_name = f"Test{test_file.class_name.replace('Test', '')}"
        
        return f'''"""
Migrated test file from Java Brobot to Python Qontinui.
Generated by LLM translator.
"""

import pytest
from unittest.mock import Mock, patch
from qontinui.core import QontinuiCore
from qontinui.state_management import State
from qontinui.actions import ActionConfig


class {python_class_name}:
    """Migrated from {test_file.class_name}."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.qontinui = QontinuiCore()
        self.test_state = State()
    
    def test_example_method(self):
        """Example test method - replace with actual translated methods."""
        # TODO: Replace with actual translated test logic
        assert True
        
    def teardown_method(self):
        """Clean up after tests."""
        pass
'''
    
    def _generate_mock_method_response(self, method_code: str) -> str:
        """Generate mock method response."""
        return """
    def test_example_method(self):
        \"\"\"Translated test method.\"\"\"
        # TODO: Implement actual test logic
        assert True
"""
    
    def _generate_mock_assertion_response(self, assertion_code: str) -> str:
        """Generate mock assertion response."""
        return "assert True  # TODO: Translate assertion"
    
    def _extract_python_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Look for code blocks
        code_block_pattern = r'```python\s*\n(.*?)\n```'
        match = re.search(code_block_pattern, response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # If no code block, try to extract everything after a common prefix
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            # Start collecting when we see imports or class definition
            if (line.strip().startswith('import ') or 
                line.strip().startswith('from ') or
                line.strip().startswith('class ') or
                line.strip().startswith('"""')):
                in_code = True
            
            if in_code:
                code_lines.append(line)
        
        return '\n'.join(code_lines).strip() if code_lines else response.strip()
    
    def _post_process_translation(self, python_code: str, test_file: TestFile) -> str:
        """Post-process the translated Python code."""
        # Fix common issues
        python_code = self._fix_common_issues(python_code)
        
        # Add missing imports if needed
        python_code = self._ensure_required_imports(python_code, test_file)
        
        # Format and clean up
        python_code = self._format_code(python_code)
        
        return python_code
    
    def _fix_common_issues(self, code: str) -> str:
        """Fix common translation issues."""
        # Fix indentation issues
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Convert tabs to spaces
            line = line.replace('\t', '    ')
            fixed_lines.append(line)
        
        code = '\n'.join(fixed_lines)
        
        # Fix common syntax issues
        fixes = {
            # Java-style method calls
            r'\.assertEquals\(': '.assert_equal(',
            r'\.assertTrue\(': '.assert_true(',
            r'\.assertFalse\(': '.assert_false(',
            
            # Java-style variable declarations
            r'String\s+(\w+)\s*=': r'\1 =',
            r'int\s+(\w+)\s*=': r'\1 =',
            r'boolean\s+(\w+)\s*=': r'\1 =',
            
            # Java-style comments
            r'//': '#',
        }
        
        for pattern, replacement in fixes.items():
            code = re.sub(pattern, replacement, code)
        
        return code
    
    def _ensure_required_imports(self, code: str, test_file: TestFile) -> str:
        """Ensure all required imports are present."""
        required_imports = ['import pytest']
        
        # Add mock imports if mocks are used
        if test_file.mock_usage:
            required_imports.append('from unittest.mock import Mock, patch, MagicMock')
        
        # Add Qontinui imports
        required_imports.extend([
            'from qontinui.core import QontinuiCore',
        ])
        
        # Add integration test imports if needed
        if test_file.test_type == TestType.INTEGRATION:
            required_imports.extend([
                'from qontinui.config import ConfigurationManager',
                'from qontinui.startup import QontinuiStartup'
            ])
        
        # Check which imports are missing
        missing_imports = []
        for import_stmt in required_imports:
            if import_stmt not in code:
                missing_imports.append(import_stmt)
        
        # Add missing imports at the top
        if missing_imports:
            lines = code.split('\n')
            
            # Find where to insert imports (after docstring)
            insert_index = 0
            in_docstring = False
            
            for i, line in enumerate(lines):
                if line.strip().startswith('"""'):
                    if not in_docstring:
                        in_docstring = True
                    else:
                        insert_index = i + 1
                        break
                elif not in_docstring and (line.strip().startswith('import ') or 
                                         line.strip().startswith('from ')):
                    insert_index = i
                    break
            
            # Insert missing imports
            for import_stmt in reversed(missing_imports):
                lines.insert(insert_index, import_stmt)
            
            code = '\n'.join(lines)
        
        return code
    
    def _format_code(self, code: str) -> str:
        """Basic code formatting."""
        # Remove excessive blank lines
        lines = code.split('\n')
        formatted_lines = []
        prev_blank = False
        
        for line in lines:
            is_blank = not line.strip()
            
            if is_blank and prev_blank:
                continue  # Skip multiple consecutive blank lines
            
            formatted_lines.append(line)
            prev_blank = is_blank
        
        return '\n'.join(formatted_lines)
    
    def validate_translation(self, python_code: str, test_file: TestFile) -> List[str]:
        """
        Validate the LLM translation for common issues.
        
        Args:
            python_code: The translated Python code
            test_file: Original test file information
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check for basic Python syntax
        try:
            compile(python_code, '<string>', 'exec')
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
        
        # Check for required components
        if 'import pytest' not in python_code:
            errors.append("Missing pytest import")
        
        if 'class Test' not in python_code:
            errors.append("Missing test class")
        
        if 'def test_' not in python_code:
            errors.append("No test methods found")
        
        # Check that all original test methods are translated
        for test_method in test_file.test_methods:
            expected_method = f"test_{test_method.name.replace('test', '').replace('Test', '').lower()}"
            if expected_method not in python_code.lower():
                errors.append(f"Missing translated method for {test_method.name}")
        
        # Check for mock usage if original had mocks
        if test_file.mock_usage and 'Mock' not in python_code:
            errors.append("Original test used mocks but translation doesn't include mock imports")
        
        return errors
    
    def get_translation_confidence(self, test_file: TestFile) -> float:
        """
        Estimate confidence in LLM translation based on test complexity.
        
        Args:
            test_file: The test file to analyze
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 1.0
        
        # Reduce confidence for complex patterns
        if len(test_file.test_methods) > 10:
            confidence -= 0.1  # Many methods
        
        if test_file.mock_usage and len(test_file.mock_usage) > 5:
            confidence -= 0.2  # Complex mocking
        
        if test_file.test_type == TestType.INTEGRATION:
            confidence -= 0.1  # Integration tests are more complex
        
        # Check for complex method bodies
        for method in test_file.test_methods:
            if method.body and len(method.body.split('\n')) > 20:
                confidence -= 0.1  # Long methods
                break
        
        return max(0.0, confidence)