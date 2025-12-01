"""
Unit tests for IntegrationTestEnvironment.
"""

from pathlib import Path

from qontinui.test_migration.core.models import (
    MockUsage,
    TestFile,
    TestMethod,
    TestType,
)
from qontinui.test_migration.translation.integration_test_environment import (
    ComponentConfiguration,
    DatabaseConfiguration,
    ExternalServiceConfiguration,
    IntegrationTestEnvironment,
    IntegrationTestGenerator,
)


class TestIntegrationTestEnvironment:
    """Test cases for IntegrationTestEnvironment."""

    def setup_method(self):
        """Set up test fixtures."""
        self.environment = IntegrationTestEnvironment()

        # Create a sample test file
        self.test_file = TestFile(
            path=Path("integration_test.py"),
            test_type=TestType.INTEGRATION,
            class_name="IntegrationTest",
            package="com.example.integration",
        )

        # Add mock usage
        self.test_file.mock_usage.append(
            MockUsage(mock_type="spring_mock", mock_class="UserService")
        )

        # Add test method
        self.test_method = TestMethod(
            name="testUserServiceIntegration",
            annotations=["@Test", "@Transactional"],
            body="userService.createUser(user); repository.save(user);",
        )
        self.test_file.test_methods.append(self.test_method)

    def test_configure_component_wiring(self):
        """Test component wiring configuration."""
        wiring_config = self.environment.configure_component_wiring(self.test_file)

        assert "imports" in wiring_config
        assert "fixtures" in wiring_config
        assert "setup_methods" in wiring_config
        assert "component_registry" in wiring_config

        # Verify imports
        assert "import pytest" in wiring_config["imports"]
        assert (
            "from unittest.mock import Mock, MagicMock, patch"
            in wiring_config["imports"]
        )

        # Verify component registry is generated
        registry_code = "\n".join(wiring_config["component_registry"])
        assert "class ComponentRegistry:" in registry_code
        assert "def register_component" in registry_code
        assert "def get_component" in registry_code
        assert "def wire_dependencies" in registry_code

        # Verify integration environment fixture is generated
        fixtures_code = "\n".join(wiring_config["fixtures"])
        assert "@pytest.fixture(scope='class')" in fixtures_code
        assert "def integration_environment" in fixtures_code

    def test_configure_database_environment_in_memory(self):
        """Test in-memory database configuration."""
        db_config = DatabaseConfiguration(
            database_type="in_memory",
            schema_files=["schema.sql"],
            data_files=["test_data.sql"],
        )

        setup_code = self.environment.configure_database_environment(db_config)

        assert len(setup_code) > 0
        setup_text = "\n".join(setup_code)
        assert "@pytest.fixture(scope='class')" in setup_text
        assert "def database_connection" in setup_text
        assert "sqlite3.connect(':memory:')" in setup_text
        assert "schema_files" in setup_text
        assert "data_files" in setup_text

    def test_configure_database_environment_testcontainer(self):
        """Test testcontainer database configuration."""
        db_config = DatabaseConfiguration(database_type="testcontainer")

        setup_code = self.environment.configure_database_environment(db_config)

        assert len(setup_code) > 0
        setup_text = "\n".join(setup_code)
        assert "testcontainers.postgres" in setup_text
        assert "PostgresContainer" in setup_text
        assert "get_connection_url" in setup_text

    def test_configure_database_environment_mock(self):
        """Test mock database configuration."""
        db_config = DatabaseConfiguration(database_type="mock")

        setup_code = self.environment.configure_database_environment(db_config)

        assert len(setup_code) > 0
        setup_text = "\n".join(setup_code)
        assert "@pytest.fixture" in setup_text
        assert "def database_mock" in setup_text
        assert "Mock()" in setup_text

    def test_configure_external_service_mocking_rest_api(self):
        """Test REST API service mocking configuration."""
        service_config = ExternalServiceConfiguration(
            service_name="user_api",
            service_type="rest_api",
            endpoints=["http://api.example.com/users"],
            mock_responses={"http://api.example.com/users": {"users": []}},
        )

        setup_code = self.environment.configure_external_service_mocking(
            [service_config]
        )

        assert len(setup_code) > 0
        setup_text = "\n".join(setup_code)
        assert "import requests_mock" in setup_text
        assert "def user_api_mock" in setup_text
        assert "requests_mock.Mocker()" in setup_text
        assert "m.get('http://api.example.com/users'" in setup_text

    def test_configure_external_service_mocking_message_queue(self):
        """Test message queue service mocking configuration."""
        service_config = ExternalServiceConfiguration(
            service_name="notification_queue", service_type="message_queue"
        )

        setup_code = self.environment.configure_external_service_mocking(
            [service_config]
        )

        assert len(setup_code) > 0
        setup_text = "\n".join(setup_code)
        assert "def notification_queue_mock" in setup_text
        assert "Mock()" in setup_text

    def test_configure_external_service_mocking_file_system(self):
        """Test file system service mocking configuration."""
        service_config = ExternalServiceConfiguration(
            service_name="file_storage", service_type="file_system"
        )

        setup_code = self.environment.configure_external_service_mocking(
            [service_config]
        )

        assert len(setup_code) > 0
        setup_text = "\n".join(setup_code)
        assert "def file_storage_mock" in setup_text
        assert "patch('builtins.open')" in setup_text

    def test_configure_external_service_mocking_cache(self):
        """Test cache service mocking configuration."""
        service_config = ExternalServiceConfiguration(
            service_name="redis_cache", service_type="cache"
        )

        setup_code = self.environment.configure_external_service_mocking(
            [service_config]
        )

        assert len(setup_code) > 0
        setup_text = "\n".join(setup_code)
        assert "def redis_cache_mock" in setup_text
        assert "mock_cache.get.return_value = None" in setup_text
        assert "mock_cache.set.return_value = True" in setup_text

    def test_create_multi_component_test_scenario(self):
        """Test multi-component test scenario creation."""
        # Add multiple mock usages to make it a multi-component test
        self.test_file.mock_usage.append(
            MockUsage(mock_type="spring_mock", mock_class="DataRepository")
        )

        scenario_config = self.environment.create_multi_component_test_scenario(
            self.test_file
        )

        assert "imports" in scenario_config
        assert "fixtures" in scenario_config
        assert "test_methods" in scenario_config
        assert "helper_methods" in scenario_config

        # Verify imports
        assert "import pytest" in scenario_config["imports"]
        assert "import asyncio" in scenario_config["imports"]

        # Verify scenario fixture
        fixtures_text = "\n".join(scenario_config["fixtures"])
        assert "def multi_component_scenario" in fixtures_text
        assert "MultiComponentScenario" in fixtures_text

        # Verify helper class
        helpers_text = "\n".join(scenario_config["helper_methods"])
        assert "class MultiComponentScenario:" in helpers_text
        assert "def record_interaction" in helpers_text
        assert "def verify_interaction_sequence" in helpers_text

    def test_extract_components_from_test_file(self):
        """Test component extraction from test file."""
        components = self.environment._extract_components_from_test_file(self.test_file)

        assert len(components) > 0

        # Should extract UserService from mock usage
        user_service_component = next(
            (c for c in components if c.component_type == "UserService"), None
        )
        assert user_service_component is not None
        assert user_service_component.mock_type == "mock"

    def test_generate_component_fixture_mock(self):
        """Test component fixture generation for mock components."""
        component = ComponentConfiguration(
            component_name="user_service",
            component_type="UserService",
            mock_type="mock",
        )

        fixture_code = self.environment._generate_component_fixture(component)

        assert len(fixture_code) > 0
        fixture_text = "\n".join(fixture_code)
        assert "@pytest.fixture" in fixture_text
        assert "def user_service" in fixture_text
        assert "Mock(spec=UserService)" in fixture_text

    def test_generate_component_fixture_real(self):
        """Test component fixture generation for real components."""
        component = ComponentConfiguration(
            component_name="user_service",
            component_type="UserService",
            mock_type="real",
        )

        fixture_code = self.environment._generate_component_fixture(component)

        assert len(fixture_code) > 0
        fixture_text = "\n".join(fixture_code)
        assert "@pytest.fixture" in fixture_text
        assert "def user_service" in fixture_text
        assert "UserService()" in fixture_text


class TestIntegrationTestGenerator:
    """Test cases for IntegrationTestGenerator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = IntegrationTestGenerator()

        # Create a sample test file
        self.test_file = TestFile(
            path=Path("integration_test.py"),
            test_type=TestType.INTEGRATION,
            class_name="IntegrationTest",
            package="com.example.integration",
        )

        # Add mock usage and test method
        self.test_file.mock_usage.append(
            MockUsage(mock_type="spring_mock", mock_class="UserService")
        )

        self.test_method = TestMethod(
            name="testUserServiceIntegration",
            annotations=["@Test", "@Transactional"],
            body="userService.createUser(user);",
        )
        self.test_file.test_methods.append(self.test_method)

    def test_generate_integration_test_file_basic(self):
        """Test basic integration test file generation."""
        result = self.generator.generate_integration_test_file(self.test_file)

        assert len(result) > 0
        assert "Integration tests migrated from" in result
        assert "import pytest" in result
        assert "class ComponentRegistry:" in result
        assert "class IntegrationTest:" in result

    def test_generate_integration_test_file_with_database(self):
        """Test integration test file generation with database configuration."""
        db_config = DatabaseConfiguration(
            database_type="in_memory", schema_files=["schema.sql"]
        )

        result = self.generator.generate_integration_test_file(
            self.test_file, database_config=db_config
        )

        assert "import sqlite3" in result
        assert "def database_connection" in result
        assert "sqlite3.connect(':memory:')" in result

    def test_generate_integration_test_file_with_external_services(self):
        """Test integration test file generation with external services."""
        service_config = ExternalServiceConfiguration(
            service_name="user_api",
            service_type="rest_api",
            endpoints=["http://api.example.com/users"],
        )

        result = self.generator.generate_integration_test_file(
            self.test_file, external_services=[service_config]
        )

        assert "import requests_mock" in result
        assert "def user_api_mock" in result
        assert "requests_mock.Mocker()" in result

    def test_generate_integration_test_file_complete(self):
        """Test complete integration test file generation with all features."""
        db_config = DatabaseConfiguration(database_type="testcontainer")

        service_config = ExternalServiceConfiguration(
            service_name="notification_service",
            service_type="rest_api",
            endpoints=["http://notifications.example.com/send"],
        )

        result = self.generator.generate_integration_test_file(
            self.test_file,
            database_config=db_config,
            external_services=[service_config],
        )

        # Verify all components are included
        assert "import testcontainers.postgres" in result
        assert "import requests_mock" in result
        assert "class ComponentRegistry:" in result
        assert "def database_container" in result
        assert "def notification_service_mock" in result
        assert "class IntegrationTest:" in result


class TestComponentConfiguration:
    """Test cases for ComponentConfiguration."""

    def test_component_configuration_creation(self):
        """Test ComponentConfiguration creation."""
        config = ComponentConfiguration(
            component_name="user_service",
            component_type="UserService",
            dependencies=["data_repository"],
            mock_type="mock",
        )

        assert config.component_name == "user_service"
        assert config.component_type == "UserService"
        assert config.dependencies == ["data_repository"]
        assert config.mock_type == "mock"


class TestDatabaseConfiguration:
    """Test cases for DatabaseConfiguration."""

    def test_database_configuration_defaults(self):
        """Test DatabaseConfiguration with default values."""
        config = DatabaseConfiguration()

        assert config.database_type == "in_memory"
        assert config.connection_url is None
        assert config.schema_files == []
        assert config.data_files == []
        assert config.cleanup_strategy == "rollback"

    def test_database_configuration_custom(self):
        """Test DatabaseConfiguration with custom values."""
        config = DatabaseConfiguration(
            database_type="testcontainer",
            connection_url="jdbc:postgresql://localhost:5432/test",
            schema_files=["schema.sql", "indexes.sql"],
            data_files=["test_data.sql"],
            cleanup_strategy="truncate",
        )

        assert config.database_type == "testcontainer"
        assert config.connection_url == "jdbc:postgresql://localhost:5432/test"
        assert config.schema_files == ["schema.sql", "indexes.sql"]
        assert config.data_files == ["test_data.sql"]
        assert config.cleanup_strategy == "truncate"


class TestExternalServiceConfiguration:
    """Test cases for ExternalServiceConfiguration."""

    def test_external_service_configuration_creation(self):
        """Test ExternalServiceConfiguration creation."""
        config = ExternalServiceConfiguration(
            service_name="user_api",
            service_type="rest_api",
            mock_strategy="wiremock",
            endpoints=["http://api.example.com/users", "http://api.example.com/auth"],
            mock_responses={
                "http://api.example.com/users": {"users": []},
                "http://api.example.com/auth": {"token": "test_token"},
            },
        )

        assert config.service_name == "user_api"
        assert config.service_type == "rest_api"
        assert config.mock_strategy == "wiremock"
        assert len(config.endpoints) == 2
        assert len(config.mock_responses) == 2
