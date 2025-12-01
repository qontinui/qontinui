"""
Unit tests for TestClassifier.
"""

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.models import Dependency, TestFile, TestType
    from ..discovery.classifier import TestClassifier
else:
    try:
        from ..core.models import Dependency, TestFile, TestType
        from ..discovery.classifier import TestClassifier
    except ImportError:
        from core.models import Dependency, TestFile, TestType
        from discovery.classifier import TestClassifier


class TestTestClassifier:
    """Test cases for TestClassifier."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = TestClassifier()

    def test_classifier_initialization(self):
        """Test classifier initializes with correct patterns."""
        assert "integration" in self.classifier.integration_indicators["path_indicators"]
        assert "@SpringBootTest" in self.classifier.integration_indicators["spring_annotations"]
        assert "BrobotMock" in self.classifier.mock_patterns["brobot_mocks"]
        assert "junit" in self.classifier.java_library_patterns

    def test_categorize_unit_test_by_default(self):
        """Test that tests are categorized as unit tests by default."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file_path = temp_path / "UserServiceTest.java"

            test_content = """
package com.example.service;

import org.junit.Test;
import org.mockito.Mock;

public class UserServiceTest {
    @Mock
    private UserRepository userRepository;

    @Test
    public void testFindUser() {
        // unit test code
    }
}
"""
            test_file_path.write_text(test_content)

            test_file = TestFile(
                path=test_file_path,
                test_type=TestType.UNKNOWN,
                class_name="UserServiceTest",
                package="com.example.service",
            )

            result = self.classifier.categorize_test(test_file)
            assert result == TestType.UNIT

    def test_categorize_integration_test_by_path(self):
        """Test categorization of integration test by path indicators."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            integration_dir = temp_path / "integration"
            integration_dir.mkdir()
            test_file_path = integration_dir / "UserServiceTest.java"

            test_content = """
package com.example.integration;

import org.junit.Test;

public class UserServiceTest {
    @Test
    public void testUserWorkflow() {
        // integration test code
    }
}
"""
            test_file_path.write_text(test_content)

            test_file = TestFile(
                path=test_file_path,
                test_type=TestType.UNKNOWN,
                class_name="UserServiceTest",
                package="com.example.integration",
            )

            result = self.classifier.categorize_test(test_file)
            assert result == TestType.INTEGRATION

    def test_categorize_integration_test_by_class_name(self):
        """Test categorization of integration test by class name indicators."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file_path = temp_path / "UserServiceIntegrationTest.java"

            test_content = """
package com.example;

import org.junit.Test;

public class UserServiceIntegrationTest {
    @Test
    public void testUserWorkflow() {
        // integration test code
    }
}
"""
            test_file_path.write_text(test_content)

            test_file = TestFile(
                path=test_file_path,
                test_type=TestType.UNKNOWN,
                class_name="UserServiceIntegrationTest",
                package="com.example",
            )

            result = self.classifier.categorize_test(test_file)
            assert result == TestType.INTEGRATION

    def test_categorize_integration_test_by_spring_import(self):
        """Test categorization of integration test by Spring imports."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file_path = temp_path / "UserControllerTest.java"

            test_content = """
package com.example.controller;

import org.junit.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
public class UserControllerTest {
    @Test
    public void testCreateUser() {
        // integration test code
    }
}
"""
            test_file_path.write_text(test_content)

            spring_dependency = Dependency(
                java_import="org.springframework.boot.test.context.SpringBootTest"
            )

            test_file = TestFile(
                path=test_file_path,
                test_type=TestType.UNKNOWN,
                class_name="UserControllerTest",
                package="com.example.controller",
                dependencies=[spring_dependency],
            )

            result = self.classifier.categorize_test(test_file)
            assert result == TestType.INTEGRATION

    def test_categorize_integration_test_by_spring_annotation(self):
        """Test categorization of integration test by Spring annotations in content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file_path = temp_path / "UserControllerTest.java"

            test_content = """
package com.example.controller;

import org.junit.Test;

@SpringBootTest
public class UserControllerTest {
    @Test
    public void testCreateUser() {
        // integration test code
    }
}
"""
            test_file_path.write_text(test_content)

            test_file = TestFile(
                path=test_file_path,
                test_type=TestType.UNKNOWN,
                class_name="UserControllerTest",
                package="com.example.controller",
            )

            result = self.classifier.categorize_test(test_file)
            assert result == TestType.INTEGRATION

    def test_detect_brobot_mock_usage(self):
        """Test detection of Brobot mock usage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file_path = temp_path / "GuiTest.java"

            test_content = """
package com.example.gui;

import org.junit.Test;
import com.brobot.BrobotMock;

public class GuiTest {
    private BrobotMock brobotMock;

    @Test
    public void testGuiInteraction() {
        brobotMock.findElement("loginButton");
        brobotMock.click("loginButton");
        brobotMock.type("username", "testuser");
    }
}
"""
            test_file_path.write_text(test_content)

            test_file = TestFile(
                path=test_file_path,
                test_type=TestType.UNIT,
                class_name="GuiTest",
                package="com.example.gui",
            )

            mock_usages = self.classifier.detect_mock_usage(test_file)

            assert len(mock_usages) > 0
            brobot_mocks = [m for m in mock_usages if m.mock_type == "brobot_mock"]
            assert len(brobot_mocks) > 0

            # Check GUI model extraction
            brobot_mock = brobot_mocks[0]
            assert brobot_mock.gui_model is not None
            assert "loginButton" in brobot_mock.gui_model.elements
            assert "click" in brobot_mock.gui_model.actions

    def test_detect_spring_mock_usage(self):
        """Test detection of Spring mock usage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file_path = temp_path / "ServiceTest.java"

            test_content = """
package com.example.service;

import org.junit.Test;
import org.springframework.boot.test.mock.mockito.MockBean;

public class ServiceTest {
    @MockBean
    private UserRepository userRepository;

    @Test
    public void testService() {
        // test code
    }
}
"""
            test_file_path.write_text(test_content)

            test_file = TestFile(
                path=test_file_path,
                test_type=TestType.UNIT,
                class_name="ServiceTest",
                package="com.example.service",
            )

            mock_usages = self.classifier.detect_mock_usage(test_file)

            spring_mocks = [m for m in mock_usages if m.mock_type == "spring_mock"]
            assert len(spring_mocks) > 0
            assert spring_mocks[0].mock_class == "@MockBean"

    def test_detect_mockito_mock_usage(self):
        """Test detection of Mockito mock usage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file_path = temp_path / "ServiceTest.java"

            test_content = """
package com.example.service;

import org.junit.Test;
import org.mockito.Mock;
import org.mockito.Mockito;

public class ServiceTest {
    @Mock
    private UserRepository userRepository;

    @Test
    public void testService() {
        Mockito.when(userRepository.findById(1)).thenReturn(user);
    }
}
"""
            test_file_path.write_text(test_content)

            test_file = TestFile(
                path=test_file_path,
                test_type=TestType.UNIT,
                class_name="ServiceTest",
                package="com.example.service",
            )

            mock_usages = self.classifier.detect_mock_usage(test_file)

            mockito_mocks = [m for m in mock_usages if m.mock_type == "mockito_mock"]
            assert len(mockito_mocks) > 0

    def test_analyze_dependencies_testing_frameworks(self):
        """Test analysis of testing framework dependencies."""
        junit_dependency = Dependency(java_import="org.junit.Test")
        testng_dependency = Dependency(java_import="org.testng.annotations.Test")

        test_file = TestFile(
            path=Path("test.java"),
            test_type=TestType.UNIT,
            class_name="Test",
            dependencies=[junit_dependency, testng_dependency],
        )

        analysis = self.classifier.analyze_dependencies(test_file)

        assert "org.junit.Test" in analysis["testing_frameworks"]
        assert "org.testng.annotations.Test" in analysis["testing_frameworks"]

    def test_analyze_dependencies_mocking_frameworks(self):
        """Test analysis of mocking framework dependencies."""
        mockito_dependency = Dependency(java_import="org.mockito.Mock")
        powermock_dependency = Dependency(java_import="org.powermock.api.mockito.PowerMockito")

        test_file = TestFile(
            path=Path("test.java"),
            test_type=TestType.UNIT,
            class_name="Test",
            dependencies=[mockito_dependency, powermock_dependency],
        )

        analysis = self.classifier.analyze_dependencies(test_file)

        assert "org.mockito.Mock" in analysis["mocking_frameworks"]
        assert "org.powermock.api.mockito.PowerMockito" in analysis["mocking_frameworks"]

    def test_analyze_dependencies_spring_testing(self):
        """Test analysis of Spring testing dependencies."""
        spring_dependency = Dependency(
            java_import="org.springframework.boot.test.context.SpringBootTest"
        )

        test_file = TestFile(
            path=Path("test.java"),
            test_type=TestType.INTEGRATION,
            class_name="Test",
            dependencies=[spring_dependency],
        )

        analysis = self.classifier.analyze_dependencies(test_file)

        assert "org.springframework.boot.test.context.SpringBootTest" in analysis["spring_testing"]

    def test_analyze_dependencies_custom_libraries(self):
        """Test analysis of custom library dependencies."""
        custom_dependency = Dependency(java_import="com.example.custom.CustomLibrary")

        test_file = TestFile(
            path=Path("test.java"),
            test_type=TestType.UNIT,
            class_name="Test",
            dependencies=[custom_dependency],
        )

        analysis = self.classifier.analyze_dependencies(test_file)

        assert "com.example.custom.CustomLibrary" in analysis["custom_libraries"]

    def test_has_integration_path_indicators(self):
        """Test detection of integration indicators in file path."""
        integration_path = Path("/project/src/test/integration/UserServiceTest.java")
        e2e_path = Path("/project/src/test/e2e/WorkflowTest.java")
        unit_path = Path("/project/src/test/unit/UserServiceTest.java")

        integration_file = TestFile(
            path=integration_path,
            test_type=TestType.UNKNOWN,
            class_name="UserServiceTest",
        )

        e2e_file = TestFile(path=e2e_path, test_type=TestType.UNKNOWN, class_name="WorkflowTest")

        unit_file = TestFile(
            path=unit_path, test_type=TestType.UNKNOWN, class_name="UserServiceTest"
        )

        assert self.classifier._has_integration_path_indicators(integration_file) is True
        assert self.classifier._has_integration_path_indicators(e2e_file) is True
        assert self.classifier._has_integration_path_indicators(unit_file) is False

    def test_has_integration_class_indicators(self):
        """Test detection of integration indicators in class name."""
        integration_file = TestFile(
            path=Path("test.java"),
            test_type=TestType.UNKNOWN,
            class_name="UserServiceIntegrationTest",
        )

        e2e_file = TestFile(
            path=Path("test.java"),
            test_type=TestType.UNKNOWN,
            class_name="WorkflowE2ETest",
        )

        unit_file = TestFile(
            path=Path("test.java"),
            test_type=TestType.UNKNOWN,
            class_name="UserServiceTest",
        )

        assert self.classifier._has_integration_class_indicators(integration_file) is True
        assert self.classifier._has_integration_class_indicators(e2e_file) is True
        assert self.classifier._has_integration_class_indicators(unit_file) is False

    def test_extract_gui_model_from_content(self):
        """Test extraction of GUI model from Brobot mock content."""
        content = """
        brobotMock.findElement("loginButton");
        brobotMock.click("loginButton");
        brobotMock.type("username", "testuser");
        brobotMock.waitFor("welcomeMessage");
        brobotMock.verify("successMessage");
        """

        gui_model = self.classifier._extract_gui_model_from_content(content, "BrobotMock")

        assert gui_model.model_name == "BrobotMock_model"
        assert "loginButton" in gui_model.elements
        assert "username" in gui_model.elements
        assert "welcomeMessage" in gui_model.elements
        assert "click" in gui_model.actions
        assert "type" in gui_model.actions
        assert "waitFor" in gui_model.actions
        assert "verify" in gui_model.actions

    def test_determine_simulation_scope(self):
        """Test determination of mock simulation scope."""
        class_scope_content = """
        @BeforeClass
        public static void setup() {
            // class setup
        }
        """

        method_scope_content = """
        @BeforeEach
        public void setup() {
            // method setup
        }
        """

        default_content = """
        @Test
        public void testSomething() {
            // test code
        }
        """

        assert self.classifier._determine_simulation_scope(class_scope_content, "Mock") == "class"
        assert self.classifier._determine_simulation_scope(method_scope_content, "Mock") == "method"
        assert self.classifier._determine_simulation_scope(default_content, "Mock") == "method"
