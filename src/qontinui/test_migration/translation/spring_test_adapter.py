"""
SpringBoot test annotation mapper and adapter for Python migration.

This module handles the migration of SpringBoot test patterns to Python equivalents,
including dependency injection, application context setup, and Spring-specific annotations.
"""

import re
from typing import Any

from ..core.models import TestFile, TestMethod


class SpringTestAdapter:
    """
    Handles SpringBoot test annotations and patterns for Python migration.

    Maps Spring-specific test annotations and configurations to appropriate
    Python dependency injection or configuration patterns.
    """

    def __init__(self):
        """Initialize the Spring test adapter with annotation mappings."""
        self._spring_annotation_mappings = {
            "@SpringBootTest": self._handle_spring_boot_test,
            "@TestConfiguration": self._handle_test_configuration,
            "@MockBean": self._handle_mock_bean,
            "@SpyBean": self._handle_spy_bean,
            "@Autowired": self._handle_autowired,
            "@Value": self._handle_value_injection,
            "@TestPropertySource": self._handle_test_property_source,
            "@ActiveProfiles": self._handle_active_profiles,
            "@DirtiesContext": self._handle_dirties_context,
            "@Transactional": self._handle_transactional,
            "@Rollback": self._handle_rollback,
            "@Sql": self._handle_sql_annotation,
            "@DataJpaTest": self._handle_data_jpa_test,
            "@WebMvcTest": self._handle_web_mvc_test,
            "@JsonTest": self._handle_json_test,
            "@TestMethodOrder": self._handle_test_method_order,
            "@TestInstance": self._handle_test_instance,
        }

        self._dependency_injection_patterns = {
            "field_injection": self._convert_field_injection,
            "constructor_injection": self._convert_constructor_injection,
            "setter_injection": self._convert_setter_injection,
        }

        # Python equivalents for Spring testing dependencies
        self._python_test_dependencies = {
            "org.springframework.boot.test.context.SpringBootTest": "pytest",
            "org.springframework.test.context.junit4.SpringJUnit4ClassRunner": "pytest",
            "org.springframework.boot.test.mock.mockito.MockBean": "from unittest.mock import Mock, patch",
            "org.springframework.boot.test.mock.mockito.SpyBean": "from unittest.mock import Mock, patch",
            "org.springframework.beans.factory.annotation.Autowired": "import pytest",
            "org.springframework.test.context.TestPropertySource": "import os",
            "org.springframework.test.context.ActiveProfiles": "import os",
            "org.springframework.transaction.annotation.Transactional": "import pytest",
            "org.springframework.test.annotation.Rollback": "import pytest",
            "org.springframework.test.context.jdbc.Sql": "import pytest",
            "org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest": "import pytest",
            "org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest": "import pytest",
            "org.springframework.boot.test.autoconfigure.json.JsonTest": "import pytest",
        }

    def handle_spring_annotations(self, test_file: TestFile) -> dict[str, Any]:
        """
        Process Spring annotations in a test file and generate Python equivalents.

        Args:
            test_file: The test file containing Spring annotations

        Returns:
            Dictionary containing Python configuration and setup code
        """
        spring_config: dict[str, Any] = {
            "imports": set(),
            "class_decorators": [],
            "setup_code": [],
            "fixtures": [],
            "mock_configurations": [],
            "environment_setup": [],
            "dependency_injection": {},
        }

        # Process class-level annotations
        class_annotations = self._extract_class_annotations(test_file)
        for annotation in class_annotations:
            self._process_annotation(annotation, spring_config, "class")

        # Process method-level annotations
        for method in test_file.test_methods + test_file.setup_methods + test_file.teardown_methods:
            for annotation in method.annotations:
                self._process_annotation(annotation, spring_config, "method", method)

        # Process field annotations (dependency injection)
        field_annotations = self._extract_field_annotations(test_file)
        for field_name, annotations in field_annotations.items():
            for annotation in annotations:
                self._process_field_annotation(annotation, field_name, spring_config)

        return spring_config

    def create_application_context_setup(self, test_file: TestFile) -> list[str]:
        """
        Create Python application context setup equivalent to Spring's ApplicationContext.

        Args:
            test_file: The test file requiring application context

        Returns:
            List of Python code lines for application context setup
        """
        setup_lines = []

        # Check if SpringBootTest is present
        has_spring_boot_test = any(
            "@SpringBootTest" in str(method.annotations)
            for method in test_file.test_methods + test_file.setup_methods
        )

        if has_spring_boot_test:
            setup_lines.extend(
                [
                    "@pytest.fixture(scope='class')",
                    "def application_context(self):",
                    '    """',
                    "    Python equivalent of Spring ApplicationContext.",
                    "    Sets up dependency injection container and configuration.",
                    '    """',
                    "    # Initialize dependency injection container",
                    "    container = DependencyContainer()",
                    "    ",
                    "    # Register components and services",
                    "    self._register_components(container)",
                    "    ",
                    "    # Configure test environment",
                    "    self._configure_test_environment(container)",
                    "    ",
                    "    return container",
                    "",
                    "def _register_components(self, container):",
                    '    """Register application components in the DI container."""',
                    "    # Register your application components here",
                    "    pass",
                    "",
                    "def _configure_test_environment(self, container):",
                    '    """Configure test-specific environment settings."""',
                    "    # Set up test configuration",
                    "    pass",
                ]
            )

        return setup_lines

    def create_dependency_injection_setup(self, test_file: TestFile) -> dict[str, list[str]]:
        """
        Create Python dependency injection patterns from Spring's @Autowired and similar.

        Args:
            test_file: The test file with dependency injection

        Returns:
            Dictionary with injection patterns and setup code
        """
        injection_setup: dict[str, Any] = {
            "fixtures": [],
            "setup_methods": [],
            "field_initializers": [],
        }

        # Extract autowired fields
        autowired_fields = self._extract_autowired_fields(test_file)

        for field_name, field_type in autowired_fields.items():
            # Create pytest fixture for each autowired dependency
            fixture_code = [
                "@pytest.fixture",
                f"def {field_name}(self, application_context):",
                f'    """Inject {field_type} dependency."""',
                f"    return application_context.get_component('{field_type}')",
            ]
            injection_setup["fixtures"].extend(fixture_code)
            injection_setup["fixtures"].append("")

        return injection_setup

    def _process_annotation(
        self, annotation: str, config: dict[str, Any], scope: str, method: TestMethod | None = None
    ):
        """Process a single Spring annotation."""
        annotation_name = annotation.split("(")[0].strip()

        if annotation_name in self._spring_annotation_mappings:
            handler = self._spring_annotation_mappings[annotation_name]
            handler(annotation, config, scope, method)

    def _process_field_annotation(self, annotation: str, field_name: str, config: dict[str, Any]):
        """Process field-level annotations like @Autowired, @MockBean."""
        annotation_name = annotation.split("(")[0].strip()

        if annotation_name == "@Autowired":
            config["dependency_injection"][field_name] = {
                "type": "autowired",
                "annotation": annotation,
            }
        elif annotation_name == "@MockBean":
            config["mock_configurations"].append(
                {"field_name": field_name, "type": "mock_bean", "annotation": annotation}
            )
        elif annotation_name == "@SpyBean":
            config["mock_configurations"].append(
                {"field_name": field_name, "type": "spy_bean", "annotation": annotation}
            )

    def _handle_spring_boot_test(
        self, annotation: str, config: dict[str, Any], scope: str, method: TestMethod | None = None
    ):
        """Handle @SpringBootTest annotation."""
        config["imports"].add("import pytest")
        config["imports"].add("from unittest.mock import Mock, patch")

        # Extract configuration from annotation
        web_environment = self._extract_annotation_parameter(annotation, "webEnvironment")
        classes = self._extract_annotation_parameter(annotation, "classes")
        properties = self._extract_annotation_parameter(annotation, "properties")

        setup_code = ["# SpringBootTest equivalent setup", "# Configure test application context"]

        if web_environment:
            setup_code.append(f"# Web environment: {web_environment}")

        if classes:
            setup_code.append(f"# Test configuration classes: {classes}")

        if properties:
            setup_code.append(f"# Test properties: {properties}")
            config["environment_setup"].extend(self._convert_properties_to_env_vars(properties))

        config["setup_code"].extend(setup_code)

    def _handle_test_configuration(
        self, annotation: str, config: dict[str, Any], scope: str, method: TestMethod | None = None
    ):
        """Handle @TestConfiguration annotation."""
        config["imports"].add("import pytest")
        config["setup_code"].append("# Test configuration setup")

    def _handle_mock_bean(
        self, annotation: str, config: dict[str, Any], scope: str, method: TestMethod | None = None
    ):
        """Handle @MockBean annotation."""
        config["imports"].add("from unittest.mock import Mock, patch")

        bean_class = self._extract_annotation_parameter(
            annotation, "value"
        ) or self._extract_annotation_parameter(annotation, "classes")

        if bean_class:
            mock_setup = f"# Mock bean for {bean_class}"
            config["setup_code"].append(mock_setup)

    def _handle_spy_bean(
        self, annotation: str, config: dict[str, Any], scope: str, method: TestMethod | None = None
    ):
        """Handle @SpyBean annotation."""
        config["imports"].add("from unittest.mock import Mock, patch")
        config["setup_code"].append("# Spy bean setup")

    def _handle_autowired(
        self, annotation: str, config: dict[str, Any], scope: str, method: TestMethod | None = None
    ):
        """Handle @Autowired annotation."""
        config["imports"].add("import pytest")
        # Autowired fields will be handled in dependency injection setup

    def _handle_value_injection(
        self, annotation: str, config: dict[str, Any], scope: str, method: TestMethod | None = None
    ):
        """Handle @Value annotation for property injection."""
        config["imports"].add("import os")

        value_expression = self._extract_annotation_parameter(annotation, "value")
        if value_expression:
            # Convert ${property.name} to os.environ.get('PROPERTY_NAME')
            property_name = self._extract_property_name(value_expression)
            if property_name:
                env_var = property_name.upper().replace(".", "_")
                config["environment_setup"].append(f"os.environ['{env_var}'] = 'test_value'")

    def _handle_test_property_source(
        self, annotation: str, config: dict[str, Any], scope: str, method: TestMethod | None = None
    ):
        """Handle @TestPropertySource annotation."""
        config["imports"].add("import os")

        self._extract_annotation_parameter(annotation, "locations")
        properties = self._extract_annotation_parameter(annotation, "properties")

        if properties:
            config["environment_setup"].extend(self._convert_properties_to_env_vars(properties))

    def _handle_active_profiles(
        self, annotation: str, config: dict[str, Any], scope: str, method: TestMethod | None = None
    ):
        """Handle @ActiveProfiles annotation."""
        config["imports"].add("import os")

        profiles = self._extract_annotation_parameter(
            annotation, "profiles"
        ) or self._extract_annotation_parameter(annotation, "value")

        if profiles:
            config["environment_setup"].append(f"os.environ['ACTIVE_PROFILES'] = '{profiles}'")

    def _handle_dirties_context(
        self, annotation: str, config: dict[str, Any], scope: str, method: TestMethod | None = None
    ):
        """Handle @DirtiesContext annotation."""
        config["imports"].add("import pytest")

        self._extract_annotation_parameter(annotation, "classMode")
        self._extract_annotation_parameter(annotation, "methodMode")

        if scope == "class":
            config["fixtures"].append("@pytest.fixture(scope='class', autouse=True)")
            config["fixtures"].append("def reset_context_after_class(self):")
            config["fixtures"].append("    yield")
            config["fixtures"].append("    # Reset application context")
        elif scope == "method":
            config["fixtures"].append("@pytest.fixture(autouse=True)")
            config["fixtures"].append("def reset_context_after_method(self):")
            config["fixtures"].append("    yield")
            config["fixtures"].append("    # Reset application context")

    def _handle_transactional(
        self, annotation: str, config: dict[str, Any], scope: str, method: TestMethod | None = None
    ):
        """Handle @Transactional annotation."""
        config["imports"].add("import pytest")

        if scope == "method" and method:
            # Add transaction fixture to method
            config["fixtures"].append("@pytest.fixture")
            config["fixtures"].append(f"def transaction_for_{method.name}(self):")
            config["fixtures"].append("    # Begin transaction")
            config["fixtures"].append("    yield")
            config["fixtures"].append("    # Rollback transaction")

    def _handle_rollback(
        self, annotation: str, config: dict[str, Any], scope: str, method: TestMethod | None = None
    ):
        """Handle @Rollback annotation."""
        config["imports"].add("import pytest")
        config["setup_code"].append("# Rollback configuration")

    def _handle_sql_annotation(
        self, annotation: str, config: dict[str, Any], scope: str, method: TestMethod | None = None
    ):
        """Handle @Sql annotation."""
        config["imports"].add("import pytest")

        scripts = self._extract_annotation_parameter(
            annotation, "scripts"
        ) or self._extract_annotation_parameter(annotation, "value")

        if scripts:
            config["setup_code"].append(f"# Execute SQL scripts: {scripts}")

    def _handle_data_jpa_test(
        self, annotation: str, config: dict[str, Any], scope: str, method: TestMethod | None = None
    ):
        """Handle @DataJpaTest annotation."""
        config["imports"].add("import pytest")
        config["imports"].add("from unittest.mock import Mock")
        config["setup_code"].append("# JPA test configuration")

    def _handle_web_mvc_test(
        self, annotation: str, config: dict[str, Any], scope: str, method: TestMethod | None = None
    ):
        """Handle @WebMvcTest annotation."""
        config["imports"].add("import pytest")
        config["imports"].add("from unittest.mock import Mock")
        config["setup_code"].append("# Web MVC test configuration")

    def _handle_json_test(
        self, annotation: str, config: dict[str, Any], scope: str, method: TestMethod | None = None
    ):
        """Handle @JsonTest annotation."""
        config["imports"].add("import pytest")
        config["imports"].add("import json")
        config["setup_code"].append("# JSON test configuration")

    def _handle_test_method_order(
        self, annotation: str, config: dict[str, Any], scope: str, method: TestMethod | None = None
    ):
        """Handle @TestMethodOrder annotation."""
        config["imports"].add("import pytest")

        order_type = self._extract_annotation_parameter(annotation, "value")
        if order_type:
            config["class_decorators"].append(f"# Test method order: {order_type}")

    def _handle_test_instance(
        self, annotation: str, config: dict[str, Any], scope: str, method: TestMethod | None = None
    ):
        """Handle @TestInstance annotation."""
        config["imports"].add("import pytest")

        lifecycle = self._extract_annotation_parameter(annotation, "value")
        if lifecycle:
            config["class_decorators"].append(f"# Test instance lifecycle: {lifecycle}")

    def _extract_class_annotations(self, test_file: TestFile) -> list[str]:
        """Extract class-level annotations from test file."""
        # This would need to be implemented based on how annotations are stored
        # For now, return empty list as placeholder
        return []

    def _extract_field_annotations(self, test_file: TestFile) -> dict[str, list[str]]:
        """Extract field-level annotations from test file."""
        # This would need to be implemented based on how field annotations are stored
        # For now, return empty dict as placeholder
        return {}

    def _extract_autowired_fields(self, test_file: TestFile) -> dict[str, str]:
        """Extract @Autowired fields and their types."""
        # This would need to be implemented based on how field information is stored
        # For now, return empty dict as placeholder
        return {}

    def _extract_annotation_parameter(self, annotation: str, parameter_name: str) -> str | None:
        """Extract a parameter value from an annotation."""
        # Pattern to match parameter=value in annotation
        pattern = rf"{parameter_name}\s*=\s*([^,)]+)"
        match = re.search(pattern, annotation)

        if match:
            value = match.group(1).strip()
            # Remove quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            return value

        return None

    def _extract_property_name(self, value_expression: str) -> str | None:
        """Extract property name from @Value expression like ${property.name}."""
        pattern = r"\$\{([^}]+)\}"
        match = re.search(pattern, value_expression)

        if match:
            return match.group(1)

        return None

    def _convert_properties_to_env_vars(self, properties: str) -> list[str]:
        """Convert Spring properties to environment variable assignments."""
        env_vars = []

        # Handle array of properties: {"key1=value1", "key2=value2"}
        if properties.startswith("{") and properties.endswith("}"):
            properties = properties[1:-1]

        # Split by comma and process each property
        for prop in properties.split(","):
            prop = prop.strip().strip('"').strip("'")
            if "=" in prop:
                key, value = prop.split("=", 1)
                env_key = key.strip().upper().replace(".", "_")
                env_vars.append(f"os.environ['{env_key}'] = '{value.strip()}'")

        return env_vars

    def _convert_field_injection(self, field_info: dict[str, Any]) -> list[str]:
        """Convert field injection to Python equivalent."""
        return [
            f"# Field injection for {field_info.get('name', 'unknown')}",
            "# Use pytest fixture or dependency injection container",
        ]

    def _convert_constructor_injection(self, constructor_info: dict[str, Any]) -> list[str]:
        """Convert constructor injection to Python equivalent."""
        return ["# Constructor injection", "# Use __init__ method with injected dependencies"]

    def _convert_setter_injection(self, setter_info: dict[str, Any]) -> list[str]:
        """Convert setter injection to Python equivalent."""
        return ["# Setter injection", "# Use property setters with injected dependencies"]


class DependencyContainer:
    """
    Simple dependency injection container for Python tests.

    Provides basic dependency injection functionality to replace
    Spring's ApplicationContext in test scenarios.
    """

    def __init__(self):
        """Initialize the dependency container."""
        self._components = {}
        self._singletons = {}

    def register_component(
        self, component_type: str, component_instance: Any, singleton: bool = True
    ):
        """
        Register a component in the container.

        Args:
            component_type: The type/name of the component
            component_instance: The component instance or factory
            singleton: Whether to treat as singleton
        """
        if singleton:
            self._singletons[component_type] = component_instance
        else:
            self._components[component_type] = component_instance

    def get_component(self, component_type: str) -> Any:
        """
        Get a component from the container.

        Args:
            component_type: The type/name of the component to retrieve

        Returns:
            The component instance
        """
        if component_type in self._singletons:
            return self._singletons[component_type]
        elif component_type in self._components:
            component_factory = self._components[component_type]
            if callable(component_factory):
                return component_factory()
            return component_factory
        else:
            raise ValueError(f"Component '{component_type}' not registered")

    def has_component(self, component_type: str) -> bool:
        """
        Check if a component is registered.

        Args:
            component_type: The type/name of the component

        Returns:
            True if component is registered, False otherwise
        """
        return component_type in self._singletons or component_type in self._components
