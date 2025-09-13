"""
Integration tests for FixSuggestionEngine with DiagnosticReporter.
"""

import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import Mock

from qontinui.test_migration.core.models import (
    FailureAnalysis,
    FailureType,
    SuspectedCause,
    TestFile,
    TestMethod,
    TestType,
)
from qontinui.test_migration.validation.diagnostic_reporter import DiagnosticReporterImpl
from qontinui.test_migration.validation.fix_suggestion_engine import (
    FixComplexity,
    FixSuggestion,
    FixSuggestionEngine,
    FixType,
)


class TestFixSuggestionIntegration:
    """Integration test cases for FixSuggestionEngine with DiagnosticReporter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fix_engine = FixSuggestionEngine()
        self.diagnostic_reporter = DiagnosticReporterImpl()

    def test_end_to_end_brobot_migration_fix(self):
        """Test end-to-end fix suggestion for Brobot migration issue."""
        # Create Java test file
        java_test = TestFile(
            path=Path("BrobotTest.java"),
            test_type=TestType.UNIT,
            class_name="BrobotTest",
            test_methods=[
                TestMethod(
                    name="testAction",
                    annotations=["@Test"],
                    assertions=["assertEquals(expected, actual)"]
                )
            ]
        )
        
        # Create Python test file with Brobot import issue
        python_content = """
from brobot.library import Action
import pytest

def test_action():
    action = Action()
    assertEquals(expected, actual)
"""
        
        with NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(python_content)
            python_path = Path(f.name)
        
        try:
            # Create failure analysis
            failure_analysis = FailureAnalysis(
                is_migration_issue=True,
                is_code_issue=False,
                confidence=0.9,
                diagnostic_info={
                    "error_message": "ModuleNotFoundError: No module named 'brobot.library'",
                    "stack_trace": "from brobot.library import Action",
                    "test_name": "test_action",
                    "test_file": str(python_path)
                }
            )
            
            # Generate comprehensive diagnostic report
            diagnostic_report = self.diagnostic_reporter.generate_comprehensive_report(
                java_test, python_path, failure_analysis
            )
            
            # Generate fix suggestions
            fix_suggestions = self.fix_engine.suggest_fixes(
                failure_analysis, java_test, python_path
            )
            
            # Verify we have relevant suggestions
            assert len(fix_suggestions) > 0
            
            # Should have import fix for Brobot
            import_fixes = [s for s in fix_suggestions if s.fix_type == FixType.IMPORT_FIX]
            assert len(import_fixes) > 0
            
            brobot_fix = next((s for s in import_fixes if "brobot" in s.original_code.lower()), None)
            assert brobot_fix is not None
            assert "qontinui" in brobot_fix.suggested_code.lower()
            
            # Should have assertion fix
            assertion_fixes = [s for s in fix_suggestions if s.fix_type == FixType.ASSERTION_FIX]
            assert len(assertion_fixes) > 0
            
            # Apply simple fixes
            applied_fixes = self.fix_engine.apply_simple_fixes(fix_suggestions, python_path)
            
            # Verify fixes were applied
            assert len(applied_fixes) > 0
            
            # Check that diagnostic report includes useful information
            assert diagnostic_report.migration_completeness >= 0.0
            assert diagnostic_report.overall_confidence >= 0.0
            assert len(diagnostic_report.recommendations) > 0
            
        finally:
            python_path.unlink()

    def test_junit_annotation_and_assertion_fixes(self):
        """Test fix suggestions for JUnit annotations and assertions."""
        # Create Java test file
        java_test = TestFile(
            path=Path("JUnitTest.java"),
            test_type=TestType.UNIT,
            class_name="JUnitTest",
            test_methods=[
                TestMethod(
                    name="testCalculation",
                    annotations=["@Test", "@BeforeEach"],
                    assertions=["assertEquals(5, result)", "assertTrue(isValid)"]
                )
            ]
        )
        
        # Create Python test file with JUnit issues
        python_content = """
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach

@Test
def testCalculation():
    result = 2 + 3
    assertEquals(5, result)
    assertTrue(isValid)

@BeforeEach
def setUp():
    pass
"""
        
        with NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(python_content)
            python_path = Path(f.name)
        
        try:
            # Create failure analysis for multiple issues
            failure_analysis = FailureAnalysis(
                is_migration_issue=True,
                is_code_issue=False,
                confidence=0.8,
                diagnostic_info={
                    "error_message": "NameError: name '@Test' is not defined. AttributeError: module has no attribute 'assertEquals'",
                    "stack_trace": "@Test\ndef testCalculation():\n    assertEquals(5, result)",
                    "test_name": "testCalculation",
                    "test_file": str(python_path)
                }
            )
            
            # Generate fix suggestions
            fix_suggestions = self.fix_engine.suggest_fixes(
                failure_analysis, java_test, python_path
            )
            
            # Verify we have multiple types of fixes
            fix_types = {s.fix_type for s in fix_suggestions}
            assert FixType.ANNOTATION_FIX in fix_types
            assert FixType.ASSERTION_FIX in fix_types
            
            # Check annotation fixes
            annotation_fixes = [s for s in fix_suggestions if s.fix_type == FixType.ANNOTATION_FIX]
            assert len(annotation_fixes) > 0
            
            test_annotation_fix = next((s for s in annotation_fixes if "@Test" in s.original_code), None)
            assert test_annotation_fix is not None
            assert "def test_" in test_annotation_fix.suggested_code
            
            # Check assertion fixes
            assertion_fixes = [s for s in fix_suggestions if s.fix_type == FixType.ASSERTION_FIX]
            assert len(assertion_fixes) > 0
            
            equals_fix = next((s for s in assertion_fixes if "assertEquals" in s.original_code), None)
            assert equals_fix is not None
            assert "assert" in equals_fix.suggested_code
            
            # Generate diagnostic differences
            dependency_diffs = self.diagnostic_reporter.detect_dependency_differences(java_test, python_path)
            setup_diffs = self.diagnostic_reporter.detect_setup_differences(java_test, python_path)
            assertion_diffs = self.diagnostic_reporter.compare_assertion_logic(java_test, python_path)
            
            # Should detect issues in all categories
            assert len(dependency_diffs) > 0  # JUnit imports
            assert len(setup_diffs) > 0      # @Test, @BeforeEach annotations
            assert len(assertion_diffs) > 0  # assertEquals, assertTrue
            
        finally:
            python_path.unlink()

    def test_spring_boot_integration_fixes(self):
        """Test fix suggestions for Spring Boot integration test issues."""
        # Create Java test file
        java_test = TestFile(
            path=Path("SpringIntegrationTest.java"),
            test_type=TestType.INTEGRATION,
            class_name="SpringIntegrationTest",
            test_methods=[
                TestMethod(
                    name="testServiceIntegration",
                    annotations=["@Test", "@SpringBootTest"],
                    assertions=["assertNotNull(service)"]
                )
            ]
        )
        
        # Create Python test file with Spring Boot issues
        python_content = """
from org.springframework.boot.test.context import SpringBootTest
from org.springframework.beans.factory.annotation import Autowired

@SpringBootTest
class TestServiceIntegration:
    
    @Autowired
    private Service service
    
    @Test
    def testServiceIntegration(self):
        assertNotNull(service)
"""
        
        with NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(python_content)
            python_path = Path(f.name)
        
        try:
            # Create failure analysis for Spring Boot issues
            failure_analysis = FailureAnalysis(
                is_migration_issue=True,
                is_code_issue=False,
                confidence=0.7,
                diagnostic_info={
                    "error_message": "AttributeError: 'TestClass' has no attribute 'SpringBootTest'",
                    "stack_trace": "@SpringBootTest\nclass TestServiceIntegration:",
                    "test_name": "testServiceIntegration",
                    "test_file": str(python_path)
                }
            )
            
            # Generate fix suggestions
            fix_suggestions = self.fix_engine.suggest_fixes(
                failure_analysis, java_test, python_path
            )
            
            # Should have Spring-specific fixes
            spring_fixes = [s for s in fix_suggestions 
                           if "spring" in s.description.lower() or "autowired" in s.description.lower()]
            assert len(spring_fixes) > 0
            
            # Check for SpringBootTest fix
            springboot_fix = next((s for s in fix_suggestions 
                                 if "@SpringBootTest" in s.original_code), None)
            assert springboot_fix is not None
            assert "pytest.fixture" in springboot_fix.suggested_code
            
            # Generate comprehensive report
            diagnostic_report = self.diagnostic_reporter.generate_comprehensive_report(
                java_test, python_path, failure_analysis
            )
            
            # Should have recommendations for Spring migration
            recommendations_text = " ".join(diagnostic_report.recommendations)
            assert any(keyword in recommendations_text.lower() 
                      for keyword in ["spring", "fixture", "dependency"])
            
        finally:
            python_path.unlink()

    def test_fix_suggestion_confidence_correlation(self):
        """Test that fix suggestion confidence correlates with diagnostic confidence."""
        # Create failure analysis with high confidence
        high_confidence_analysis = FailureAnalysis(
            is_migration_issue=True,
            is_code_issue=False,
            confidence=0.95,
            diagnostic_info={
                "error_message": "ModuleNotFoundError: No module named 'brobot.library'",
                "stack_trace": "from brobot.library import Action",
                "test_name": "test_brobot"
            }
        )
        
        high_confidence_suggestions = self.fix_engine.suggest_fixes(high_confidence_analysis)
        
        # Create failure analysis with low confidence
        low_confidence_analysis = FailureAnalysis(
            is_migration_issue=True,
            is_code_issue=False,
            confidence=0.3,
            diagnostic_info={
                "error_message": "Some generic error",
                "stack_trace": "generic stack trace",
                "test_name": "test_generic"
            }
        )
        
        low_confidence_suggestions = self.fix_engine.suggest_fixes(low_confidence_analysis)
        
        # High confidence analysis should produce more specific, higher confidence suggestions
        if high_confidence_suggestions and low_confidence_suggestions:
            avg_high_confidence = sum(s.confidence for s in high_confidence_suggestions) / len(high_confidence_suggestions)
            avg_low_confidence = sum(s.confidence for s in low_confidence_suggestions) / len(low_confidence_suggestions)
            
            # High confidence analysis should generally produce higher confidence suggestions
            assert avg_high_confidence >= avg_low_confidence

    def test_fix_application_with_diagnostic_validation(self):
        """Test that applied fixes improve diagnostic scores."""
        # Create Java test file
        java_test = TestFile(
            path=Path("TestClass.java"),
            test_type=TestType.UNIT,
            class_name="TestClass",
            test_methods=[
                TestMethod(
                    name="testMethod",
                    annotations=["@Test"],
                    assertions=["assertEquals(expected, actual)"]
                )
            ]
        )
        
        # Create Python test file with fixable issues
        original_content = """
from brobot.library import Action

@Test
def testMethod():
    assertEquals(expected, actual)
"""
        
        with NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(original_content)
            python_path = Path(f.name)
        
        try:
            # Generate initial diagnostic report
            initial_report = self.diagnostic_reporter.generate_comprehensive_report(
                java_test, python_path
            )
            initial_completeness = initial_report.migration_completeness
            
            # Create failure analysis
            failure_analysis = FailureAnalysis(
                is_migration_issue=True,
                is_code_issue=False,
                confidence=0.9,
                diagnostic_info={
                    "error_message": "ModuleNotFoundError: No module named 'brobot.library'",
                    "stack_trace": "from brobot.library import Action",
                    "test_name": "testMethod"
                }
            )
            
            # Generate and apply fixes
            fix_suggestions = self.fix_engine.suggest_fixes(
                failure_analysis, java_test, python_path
            )
            
            applied_fixes = self.fix_engine.apply_simple_fixes(fix_suggestions, python_path)
            
            if applied_fixes:
                # Generate post-fix diagnostic report
                post_fix_report = self.diagnostic_reporter.generate_comprehensive_report(
                    java_test, python_path
                )
                post_fix_completeness = post_fix_report.migration_completeness
                
                # Migration completeness should improve or stay the same
                assert post_fix_completeness >= initial_completeness
                
                # Should have fewer issues after fixes
                initial_issues = (len(initial_report.dependency_differences) + 
                                len(initial_report.setup_differences) + 
                                len(initial_report.assertion_differences))
                
                post_fix_issues = (len(post_fix_report.dependency_differences) + 
                                 len(post_fix_report.setup_differences) + 
                                 len(post_fix_report.assertion_differences))
                
                # Should have same or fewer issues after applying fixes
                assert post_fix_issues <= initial_issues
            
        finally:
            python_path.unlink()

    def test_complex_migration_scenario(self):
        """Test a complex migration scenario with multiple issue types."""
        # Create comprehensive Java test file
        java_test = TestFile(
            path=Path("ComplexTest.java"),
            test_type=TestType.INTEGRATION,
            class_name="ComplexTest",
            test_methods=[
                TestMethod(
                    name="testComplexScenario",
                    annotations=["@Test", "@SpringBootTest"],
                    assertions=["assertEquals(expected, actual)", "assertTrue(condition)", "assertNotNull(object)"]
                )
            ]
        )
        
        # Create Python test file with multiple issues
        python_content = """
from brobot.library import Action
from java.util import List
import org.junit.jupiter.api.Test
from org.springframework.boot.test.context import SpringBootTest

@SpringBootTest
@Test
def testComplexScenario():
    action = Action()
    list_obj = List()
    
    assertEquals(expected, actual)
    assertTrue(condition)
    assertNotNull(object)
    
    if (condition) {
        statement;
    }
"""
        
        with NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(python_content)
            python_path = Path(f.name)
        
        try:
            # Create failure analysis for multiple issues
            failure_analysis = FailureAnalysis(
                is_migration_issue=True,
                is_code_issue=False,
                confidence=0.8,
                diagnostic_info={
                    "error_message": "Multiple errors: ModuleNotFoundError, SyntaxError, AttributeError",
                    "stack_trace": "Complex stack trace with multiple issues",
                    "test_name": "testComplexScenario",
                    "test_file": str(python_path)
                }
            )
            
            # Generate comprehensive diagnostic report
            diagnostic_report = self.diagnostic_reporter.generate_comprehensive_report(
                java_test, python_path, failure_analysis
            )
            
            # Generate fix suggestions
            fix_suggestions = self.fix_engine.suggest_fixes(
                failure_analysis, java_test, python_path
            )
            
            # Should have multiple types of fixes
            fix_types = {s.fix_type for s in fix_suggestions}
            expected_types = {FixType.IMPORT_FIX, FixType.ANNOTATION_FIX, FixType.ASSERTION_FIX}
            
            # Should have at least some of the expected fix types
            assert len(fix_types.intersection(expected_types)) > 0
            
            # Should have high-confidence fixes for clear issues
            high_confidence_fixes = [s for s in fix_suggestions if s.confidence > 0.8]
            assert len(high_confidence_fixes) > 0
            
            # Diagnostic report should identify multiple issue categories
            assert len(diagnostic_report.dependency_differences) > 0
            assert len(diagnostic_report.setup_differences) > 0
            assert len(diagnostic_report.assertion_differences) > 0
            
            # Should have comprehensive recommendations
            assert len(diagnostic_report.recommendations) > 0
            
            # Migration completeness should reflect the complexity
            assert 0.0 <= diagnostic_report.migration_completeness <= 1.0
            assert 0.0 <= diagnostic_report.overall_confidence <= 1.0
            
        finally:
            python_path.unlink()

    def test_pattern_recognition_integration(self):
        """Test integration between pattern recognition and diagnostic reporting."""
        # Test various error patterns
        test_cases = [
            {
                "error": "ModuleNotFoundError: No module named 'brobot.library'",
                "stack": "from brobot.library import Action",
                "expected_pattern": "brobot_import_error"
            },
            {
                "error": "ImportError: cannot import name 'java.util.List'",
                "stack": "from java.util import List",
                "expected_pattern": "java_import_error"
            },
            {
                "error": "NameError: name '@Test' is not defined",
                "stack": "@Test\ndef testMethod():",
                "expected_pattern": "junit_annotation_error"
            },
            {
                "error": "AttributeError: module has no attribute 'assertEquals'",
                "stack": "assertEquals(expected, actual)",
                "expected_pattern": "junit_assertion_error"
            }
        ]
        
        for test_case in test_cases:
            # Test pattern recognition
            patterns = self.fix_engine.recognize_common_patterns(
                test_case["error"], test_case["stack"]
            )
            assert test_case["expected_pattern"] in patterns
            
            # Test fix suggestion generation
            failure_analysis = FailureAnalysis(
                is_migration_issue=True,
                is_code_issue=False,
                confidence=0.8,
                diagnostic_info={
                    "error_message": test_case["error"],
                    "stack_trace": test_case["stack"],
                    "test_name": "test_pattern"
                }
            )
            
            suggestions = self.fix_engine.suggest_fixes(failure_analysis)
            assert len(suggestions) > 0
            
            # Test diagnostic report generation
            report = self.diagnostic_reporter.generate_failure_report(failure_analysis)
            assert test_case["expected_pattern"].replace("_", " ") in report.lower() or "migration issue: yes" in report.lower()

    def test_fix_suggestion_sorting_and_prioritization(self):
        """Test that fix suggestions are properly sorted and prioritized."""
        # Create failure analysis that matches multiple patterns
        failure_analysis = FailureAnalysis(
            is_migration_issue=True,
            is_code_issue=False,
            confidence=0.8,
            diagnostic_info={
                "error_message": "ModuleNotFoundError: No module named 'brobot'. SyntaxError: invalid syntax. AttributeError: assertEquals",
                "stack_trace": "from brobot.library import Action\nif (condition) {\n    assertEquals(a, b)",
                "test_name": "test_multiple_issues"
            }
        )
        
        suggestions = self.fix_engine.suggest_fixes(failure_analysis)
        
        if len(suggestions) > 1:
            # Should be sorted by confidence (descending)
            for i in range(len(suggestions) - 1):
                assert suggestions[i].confidence >= suggestions[i + 1].confidence
            
            # High confidence suggestions should be first
            assert suggestions[0].confidence >= 0.7
            
            # Simple fixes should be prioritized over complex ones for same confidence
            simple_fixes = [s for s in suggestions if s.complexity == FixComplexity.SIMPLE]
            if simple_fixes:
                # Simple fixes should have reasonable confidence
                assert any(s.confidence >= 0.7 for s in simple_fixes)