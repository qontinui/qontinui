"""
Unit tests for SpringTestAdapter.
"""

from unittest.mock import Mock, patch

import pytest

from qontinui.test_migration.core.models import TestFile, TestMethod, TestType
from qontinui.test_migration.translation.spring_test_adapter import (
    DependencyContainer,
    SpringTestAdapter,
)


class TestSpringTestAdapter:
    """Test cases for SpringTestAdapter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = SpringTestAdapter()

        # Create a sample test file with Spring annotations
        self.test_file = TestFile(
            path="test_example.py",
            test_type=TestType.INTEGRATION,
            class_name="ExampleTest",
            package="com.example.test",
        )

        # Add test method with Spring annotations
        self.test_method = TestMethod(
            name="testExample",
            annotations=["@Test", "@Transactional"],
            body="// test implementation",
        )
        self.test_file.test_methods.append(self.test_method)

    def test_handle_spring_boot_test_annotation(self):
        """Test handling of @SpringBootTest annotation."""
        annotation = "@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)"
        config = {
            "imports": set(),
            "class_decorators": [],
            "setup_code": [],
            "fixtures": [],
            "mock_configurations": [],
            "environment_setup": [],
            "dependency_injection": {},
        }

        self.adapter._handle_spring_boot_test(annotation, config, "class")

        assert "import pytest" in config["imports"]
        assert "from unittest.mock import Mock, patch" in config["imports"]
        assert any("SpringBootTest equivalent setup" in line for line in config["setup_code"])

    def test_handle_mock_bean_annotation(self):
        """Test handling of @MockBean annotation."""
        annotation = "@MockBean(UserService.class)"
        config = {
            "imports": set(),
            "class_decorators": [],
            "setup_code": [],
            "fixtures": [],
            "mock_configurations": [],
            "environment_setup": [],
            "dependency_injection": {},
        }

        self.adapter._handle_mock_bean(annotation, config, "field")

        assert "from unittest.mock import Mock, patch" in config["imports"]
        assert any("Mock bean" in line for line in config["setup_code"])

    def test_handle_autowired_annotation(self):
        """Test handling of @Autowired annotation."""
        annotation = "@Autowired"
        config = {
            "imports": set(),
            "class_decorators": [],
            "setup_code": [],
            "fixtures": [],
            "mock_configurations": [],
            "environment_setup": [],
            "dependency_injection": {},
        }

        self.adapter._handle_autowired(annotation, config, "field")

        assert "import pytest" in config["imports"]

    def test_handle_value_injection_annotation(self):
        """Test handling of @Value annotation."""
        annotation = '@Value("${app.test.property}")'
        config = {
            "imports": set(),
            "class_decorators": [],
            "setup_code": [],
            "fixtures": [],
            "mock_configurations": [],
            "environment_setup": [],
            "dependency_injection": {},
        }

        self.adapter._handle_value_injection(annotation, config, "field")

        assert "import os" in config["imports"]
        assert any("APP_TEST_PROPERTY" in line for line in config["environment_setup"])

    def test_handle_test_property_source_annotation(self):
        """Test handling of @TestPropertySource annotation."""
        annotation = '@TestPropertySource(properties = {"test.prop1=value1", "test.prop2=value2"})'
        config = {
            "imports": set(),
            "class_decorators": [],
            "setup_code": [],
            "fixtures": [],
            "mock_configurations": [],
            "environment_setup": [],
            "dependency_injection": {},
        }

        self.adapter._handle_test_property_source(annotation, config, "class")

        assert "import os" in config["imports"]
        assert any("TEST_PROP1" in line for line in config["environment_setup"])
        assert any("TEST_PROP2" in line for line in config["environment_setup"])

    def test_handle_active_profiles_annotation(self):
        """Test handling of @ActiveProfiles annotation."""
        annotation = '@ActiveProfiles("test")'
        config = {
            "imports": set(),
            "class_decorators": [],
            "setup_code": [],
            "fixtures": [],
            "mock_configurations": [],
            "environment_setup": [],
            "dependency_injection": {},
        }

        self.adapter._handle_active_profiles(annotation, config, "class")

        assert "import os" in config["imports"]
        assert any("ACTIVE_PROFILES" in line for line in config["environment_setup"])

    def test_handle_transactional_annotation(self):
        """Test handling of @Transactional annotation."""
        annotation = "@Transactional"
        config = {
            "imports": set(),
            "class_decorators": [],
            "setup_code": [],
            "fixtures": [],
            "mock_configurations": [],
            "environment_setup": [],
            "dependency_injection": {},
        }

        method = TestMethod(name="testTransactional", annotations=["@Transactional"])
        self.adapter._handle_transactional(annotation, config, "method", method)

        assert "import pytest" in config["imports"]
        assert any("transaction_for_testTransactional" in line for line in config["fixtures"])

    def test_handle_dirties_context_annotation(self):
        """Test handling of @DirtiesContext annotation."""
        annotation = "@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)"
        config = {
            "imports": set(),
            "class_decorators": [],
            "setup_code": [],
            "fixtures": [],
            "mock_configurations": [],
            "environment_setup": [],
            "dependency_injection": {},
        }

        self.adapter._handle_dirties_context(annotation, config, "class")

        assert "import pytest" in config["imports"]
        assert any("reset_context_after_class" in line for line in config["fixtures"])

    def test_extract_annotation_parameter(self):
        """Test extraction of annotation parameters."""
        annotation = "@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT, classes = TestConfig.class)"

        web_env = self.adapter._extract_annotation_parameter(annotation, "webEnvironment")
        classes = self.adapter._extract_annotation_parameter(annotation, "classes")

        assert web_env == "SpringBootTest.WebEnvironment.RANDOM_PORT"
        assert classes == "TestConfig.class"

    def test_extract_property_name(self):
        """Test extraction of property names from @Value expressions."""
        value_expression = "${app.test.property}"

        property_name = self.adapter._extract_property_name(value_expression)

        assert property_name == "app.test.property"

    def test_convert_properties_to_env_vars(self):
        """Test conversion of Spring properties to environment variables."""
        properties = '{"test.prop1=value1", "test.prop2=value2"}'

        env_vars = self.adapter._convert_properties_to_env_vars(properties)

        assert len(env_vars) == 2
        assert "os.environ['TEST_PROP1'] = 'value1'" in env_vars
        assert "os.environ['TEST_PROP2'] = 'value2'" in env_vars

    def test_create_application_context_setup(self):
        """Test creation of application context setup code."""
        # Add SpringBootTest annotation to trigger context setup
        spring_method = TestMethod(
            name="testWithSpringBoot",
            annotations=["@SpringBootTest"],
            body="// test with spring boot",
        )
        self.test_file.test_methods.append(spring_method)

        setup_lines = self.adapter.create_application_context_setup(self.test_file)

        assert len(setup_lines) > 0
        assert any("@pytest.fixture" in line for line in setup_lines)
        assert any("application_context" in line for line in setup_lines)
        assert any("DependencyContainer" in line for line in setup_lines)

    def test_create_dependency_injection_setup(self):
        """Test creation of dependency injection setup."""
        # Mock the _extract_autowired_fields method to return test data
        with patch.object(self.adapter, "_extract_autowired_fields") as mock_extract:
            mock_extract.return_value = {
                "userService": "UserService",
                "dataRepository": "DataRepository",
            }

            injection_setup = self.adapter.create_dependency_injection_setup(self.test_file)

            assert "fixtures" in injection_setup
            assert "setup_methods" in injection_setup
            assert "field_initializers" in injection_setup

            fixtures = injection_setup["fixtures"]
            assert any("userService" in line for line in fixtures)
            assert any("dataRepository" in line for line in fixtures)

    def test_handle_spring_annotations_integration(self):
        """Test the main handle_spring_annotations method."""
        # Create a test file with various Spring annotations
        test_file = TestFile(
            path="integration_test.py",
            test_type=TestType.INTEGRATION,
            class_name="IntegrationTest",
            package="com.example.integration",
        )

        # Add method with multiple annotations
        method = TestMethod(
            name="testIntegration",
            annotations=["@Test", "@Transactional", "@Rollback"],
            body="// integration test",
        )
        test_file.test_methods.append(method)

        # Mock the extraction methods to return test data
        with (
            patch.object(self.adapter, "_extract_class_annotations") as mock_class_ann,
            patch.object(self.adapter, "_extract_field_annotations") as mock_field_ann,
        ):

            mock_class_ann.return_value = ["@SpringBootTest", '@ActiveProfiles("test")']
            mock_field_ann.return_value = {
                "userService": ["@Autowired"],
                "mockRepository": ["@MockBean"],
            }

            spring_config = self.adapter.handle_spring_annotations(test_file)

            assert "imports" in spring_config
            assert "class_decorators" in spring_config
            assert "setup_code" in spring_config
            assert "fixtures" in spring_config
            assert "mock_configurations" in spring_config
            assert "environment_setup" in spring_config
            assert "dependency_injection" in spring_config

            # Verify imports were added
            assert len(spring_config["imports"]) > 0
            assert "import pytest" in spring_config["imports"]


class TestDependencyContainer:
    """Test cases for DependencyContainer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.container = DependencyContainer()

    def test_register_and_get_singleton_component(self):
        """Test registering and retrieving singleton components."""
        mock_service = Mock()

        self.container.register_component("UserService", mock_service, singleton=True)

        retrieved_service = self.container.get_component("UserService")

        assert retrieved_service is mock_service

        # Verify singleton behavior - should return same instance
        retrieved_again = self.container.get_component("UserService")
        assert retrieved_again is mock_service

    def test_register_and_get_non_singleton_component(self):
        """Test registering and retrieving non-singleton components."""

        def service_factory():
            return Mock()

        self.container.register_component("UserService", service_factory, singleton=False)

        service1 = self.container.get_component("UserService")
        service2 = self.container.get_component("UserService")

        # Should be different instances for non-singleton
        assert service1 is not service2

    def test_has_component(self):
        """Test checking if component is registered."""
        mock_service = Mock()

        assert not self.container.has_component("UserService")

        self.container.register_component("UserService", mock_service)

        assert self.container.has_component("UserService")

    def test_get_unregistered_component_raises_error(self):
        """Test that getting unregistered component raises ValueError."""
        with pytest.raises(ValueError, match="Component 'UnknownService' not registered"):
            self.container.get_component("UnknownService")

    def test_register_non_callable_non_singleton(self):
        """Test registering non-callable instance as non-singleton."""
        mock_service = Mock()

        self.container.register_component("UserService", mock_service, singleton=False)

        retrieved_service = self.container.get_component("UserService")

        assert retrieved_service is mock_service
