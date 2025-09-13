"""
Integration test environment setup for Python migration.

This module handles the setup of Python integration test environments,
including component wiring, dependency injection, database mocking,
and external service mocking patterns.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union
from unittest.mock import Mock, MagicMock, patch

from ..core.models import TestFile, TestMethod, MockUsage


@dataclass
class ComponentConfiguration:
    """Configuration for a component in the integration test environment."""
    component_name: str
    component_type: str
    dependencies: List[str] = field(default_factory=list)
    mock_type: Optional[str] = None  # 'mock', 'spy', 'real'
    configuration: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatabaseConfiguration:
    """Configuration for database setup in integration tests."""
    database_type: str = "in_memory"  # 'in_memory', 'testcontainer', 'mock'
    connection_url: Optional[str] = None
    schema_files: List[str] = field(default_factory=list)
    data_files: List[str] = field(default_factory=list)
    cleanup_strategy: str = "rollback"  # 'rollback', 'truncate', 'recreate'


@dataclass
class ExternalServiceConfiguration:
    """Configuration for external service mocking."""
    service_name: str
    service_type: str  # 'rest_api', 'message_queue', 'file_system', 'cache'
    mock_strategy: str = "mock"  # 'mock', 'wiremock', 'testcontainer'
    endpoints: List[str] = field(default_factory=list)
    mock_responses: Dict[str, Any] = field(default_factory=dict)


class IntegrationTestEnvironment:
    """
    Manages the setup and configuration of Python integration test environments.
    
    Provides functionality to:
    - Set up component wiring for dependency injection
    - Configure database connections and test data
    - Mock external services and APIs
    - Manage test isolation and cleanup
    """
    
    def __init__(self):
        """Initialize the integration test environment."""
        self._components: Dict[str, ComponentConfiguration] = {}
        self._database_config: Optional[DatabaseConfiguration] = None
        self._external_services: Dict[str, ExternalServiceConfiguration] = {}
        self._environment_variables: Dict[str, str] = {}
        self._test_fixtures: List[str] = []
        self._cleanup_handlers: List[callable] = []
    
    def configure_component_wiring(self, test_file: TestFile) -> Dict[str, List[str]]:
        """
        Configure component wiring for Python dependency injection.
        
        Args:
            test_file: The test file requiring component wiring
            
        Returns:
            Dictionary with wiring configuration and setup code
        """
        wiring_config = {
            'imports': [],
            'fixtures': [],
            'setup_methods': [],
            'component_registry': []
        }
        
        # Add basic dependency injection imports
        wiring_config['imports'].extend([
            "import pytest",
            "from unittest.mock import Mock, MagicMock, patch",
            "from typing import Dict, Any, Optional"
        ])
        
        # Extract components from test file
        components = self._extract_components_from_test_file(test_file)
        
        # Generate component registry
        wiring_config['component_registry'].extend([
            "class ComponentRegistry:",
            "    \"\"\"Registry for managing test components and their dependencies.\"\"\"",
            "    ",
            "    def __init__(self):",
            "        self._components = {}",
            "        self._mocks = {}",
            "    ",
            "    def register_component(self, name: str, component: Any, is_mock: bool = False):",
            "        \"\"\"Register a component in the registry.\"\"\"",
            "        if is_mock:",
            "            self._mocks[name] = component",
            "        else:",
            "            self._components[name] = component",
            "    ",
            "    def get_component(self, name: str) -> Any:",
            "        \"\"\"Get a component from the registry.\"\"\"",
            "        if name in self._mocks:",
            "            return self._mocks[name]",
            "        return self._components.get(name)",
            "    ",
            "    def wire_dependencies(self, component: Any, dependencies: Dict[str, str]):",
            "        \"\"\"Wire dependencies into a component.\"\"\"",
            "        for attr_name, dep_name in dependencies.items():",
            "            dependency = self.get_component(dep_name)",
            "            if dependency:",
            "                setattr(component, attr_name, dependency)"
        ])
        
        # Generate component fixtures
        for component in components:
            fixture_code = self._generate_component_fixture(component)
            wiring_config['fixtures'].extend(fixture_code)
            wiring_config['fixtures'].append("")
        
        # Generate main integration test fixture
        wiring_config['fixtures'].extend([
            "@pytest.fixture(scope='class')",
            "def integration_environment(self):",
            "    \"\"\"Set up the integration test environment.\"\"\"",
            "    registry = ComponentRegistry()",
            "    ",
            "    # Register all components",
            "    self._register_components(registry)",
            "    ",
            "    # Wire dependencies",
            "    self._wire_component_dependencies(registry)",
            "    ",
            "    yield registry",
            "    ",
            "    # Cleanup",
            "    self._cleanup_environment(registry)"
        ])
        
        return wiring_config
    
    def configure_database_environment(self, database_config: DatabaseConfiguration) -> List[str]:
        """
        Configure database environment for integration tests.
        
        Args:
            database_config: Database configuration settings
            
        Returns:
            List of Python code lines for database setup
        """
        self._database_config = database_config
        
        setup_code = []
        
        if database_config.database_type == "in_memory":
            setup_code.extend(self._setup_in_memory_database(database_config))
        elif database_config.database_type == "testcontainer":
            setup_code.extend(self._setup_testcontainer_database(database_config))
        elif database_config.database_type == "mock":
            setup_code.extend(self._setup_mock_database(database_config))
        
        return setup_code
    
    def configure_external_service_mocking(self, services: List[ExternalServiceConfiguration]) -> List[str]:
        """
        Configure external service mocking patterns.
        
        Args:
            services: List of external service configurations
            
        Returns:
            List of Python code lines for service mocking setup
        """
        setup_code = []
        
        # Add imports for external service mocking
        setup_code.extend([
            "import requests_mock",
            "from unittest.mock import patch, Mock",
            "import json"
        ])
        
        for service in services:
            self._external_services[service.service_name] = service
            
            if service.service_type == "rest_api":
                setup_code.extend(self._setup_rest_api_mock(service))
            elif service.service_type == "message_queue":
                setup_code.extend(self._setup_message_queue_mock(service))
            elif service.service_type == "file_system":
                setup_code.extend(self._setup_file_system_mock(service))
            elif service.service_type == "cache":
                setup_code.extend(self._setup_cache_mock(service))
        
        return setup_code
    
    def create_multi_component_test_scenario(self, test_file: TestFile) -> Dict[str, List[str]]:
        """
        Create integration tests for multi-component scenarios.
        
        Args:
            test_file: The test file with multi-component tests
            
        Returns:
            Dictionary with test scenario setup and execution code
        """
        scenario_config = {
            'imports': [],
            'fixtures': [],
            'test_methods': [],
            'helper_methods': []
        }
        
        # Add imports for multi-component testing
        scenario_config['imports'].extend([
            "import pytest",
            "import asyncio",
            "from unittest.mock import Mock, patch, call",
            "from typing import List, Dict, Any"
        ])
        
        # Generate scenario fixtures
        scenario_config['fixtures'].extend([
            "@pytest.fixture(scope='function')",
            "def multi_component_scenario(self, integration_environment):",
            "    \"\"\"Set up a multi-component test scenario.\"\"\"",
            "    scenario = MultiComponentScenario(integration_environment)",
            "    scenario.setup()",
            "    yield scenario",
            "    scenario.cleanup()"
        ])
        
        # Generate scenario helper class
        scenario_config['helper_methods'].extend([
            "class MultiComponentScenario:",
            "    \"\"\"Helper class for managing multi-component test scenarios.\"\"\"",
            "    ",
            "    def __init__(self, environment):",
            "        self.environment = environment",
            "        self.interactions = []",
            "        self.state_snapshots = {}",
            "    ",
            "    def setup(self):",
            "        \"\"\"Set up the scenario.\"\"\"",
            "        # Initialize component interactions",
            "        pass",
            "    ",
            "    def record_interaction(self, component: str, method: str, args: tuple, result: Any):",
            "        \"\"\"Record a component interaction for verification.\"\"\"",
            "        self.interactions.append({",
            "            'component': component,",
            "            'method': method,",
            "            'args': args,",
            "            'result': result",
            "        })",
            "    ",
            "    def verify_interaction_sequence(self, expected_sequence: List[Dict]):",
            "        \"\"\"Verify that interactions occurred in the expected sequence.\"\"\"",
            "        assert len(self.interactions) == len(expected_sequence)",
            "        for actual, expected in zip(self.interactions, expected_sequence):",
            "            assert actual['component'] == expected['component']",
            "            assert actual['method'] == expected['method']",
            "    ",
            "    def cleanup(self):",
            "        \"\"\"Clean up the scenario.\"\"\"",
            "        self.interactions.clear()",
            "        self.state_snapshots.clear()"
        ])
        
        # Generate example multi-component test methods
        for method in test_file.test_methods:
            if self._is_multi_component_test(method):
                test_code = self._generate_multi_component_test_method(method)
                scenario_config['test_methods'].extend(test_code)
                scenario_config['test_methods'].append("")
        
        return scenario_config
    
    def _extract_components_from_test_file(self, test_file: TestFile) -> List[ComponentConfiguration]:
        """Extract component configurations from test file."""
        components = []
        
        # Extract from mock usage
        for mock_usage in test_file.mock_usage:
            if mock_usage.mock_type in ['spring_mock', 'autowired']:
                component = ComponentConfiguration(
                    component_name=mock_usage.mock_class.lower().replace('service', '').replace('repository', ''),
                    component_type=mock_usage.mock_class,
                    mock_type='mock' if mock_usage.mock_type == 'spring_mock' else 'real'
                )
                components.append(component)
        
        # Add default components if none found
        if not components:
            components.extend([
                ComponentConfiguration(
                    component_name="user_service",
                    component_type="UserService",
                    mock_type="mock"
                ),
                ComponentConfiguration(
                    component_name="data_repository",
                    component_type="DataRepository",
                    mock_type="mock"
                )
            ])
        
        return components
    
    def _generate_component_fixture(self, component: ComponentConfiguration) -> List[str]:
        """Generate pytest fixture for a component."""
        fixture_code = []
        
        if component.mock_type == "mock":
            fixture_code.extend([
                f"@pytest.fixture",
                f"def {component.component_name}(self):",
                f"    \"\"\"Mock {component.component_type} for testing.\"\"\"",
                f"    mock = Mock(spec={component.component_type})",
                f"    # Configure mock behavior",
                f"    return mock"
            ])
        else:
            fixture_code.extend([
                f"@pytest.fixture",
                f"def {component.component_name}(self):",
                f"    \"\"\"Real {component.component_type} instance for testing.\"\"\"",
                f"    # Create and configure real component",
                f"    component = {component.component_type}()",
                f"    return component"
            ])
        
        return fixture_code
    
    def _setup_in_memory_database(self, config: DatabaseConfiguration) -> List[str]:
        """Set up in-memory database configuration."""
        return [
            "@pytest.fixture(scope='class')",
            "def database_connection(self):",
            "    \"\"\"Set up in-memory database connection.\"\"\"",
            "    import sqlite3",
            "    conn = sqlite3.connect(':memory:')",
            "    ",
            "    # Execute schema files",
            f"    schema_files = {config.schema_files}",
            "    for schema_file in schema_files:",
            "        with open(schema_file, 'r') as f:",
            "            conn.executescript(f.read())",
            "    ",
            "    # Load test data",
            f"    data_files = {config.data_files}",
            "    for data_file in data_files:",
            "        with open(data_file, 'r') as f:",
            "            conn.executescript(f.read())",
            "    ",
            "    yield conn",
            "    conn.close()"
        ]
    
    def _setup_testcontainer_database(self, config: DatabaseConfiguration) -> List[str]:
        """Set up testcontainer database configuration."""
        return [
            "import testcontainers.postgres",
            "",
            "@pytest.fixture(scope='class')",
            "def database_container(self):",
            "    \"\"\"Set up database test container.\"\"\"",
            "    with testcontainers.postgres.PostgresContainer() as postgres:",
            "        connection_url = postgres.get_connection_url()",
            "        yield connection_url"
        ]
    
    def _setup_mock_database(self, config: DatabaseConfiguration) -> List[str]:
        """Set up mock database configuration."""
        return [
            "@pytest.fixture",
            "def database_mock(self):",
            "    \"\"\"Set up database mock.\"\"\"",
            "    mock_db = Mock()",
            "    # Configure mock database behavior",
            "    return mock_db"
        ]
    
    def _setup_rest_api_mock(self, service: ExternalServiceConfiguration) -> List[str]:
        """Set up REST API mocking."""
        setup_code = [
            f"@pytest.fixture",
            f"def {service.service_name}_mock(self):",
            f"    \"\"\"Mock {service.service_name} REST API.\"\"\"",
            f"    with requests_mock.Mocker() as m:"
        ]
        
        # Add mock responses for each endpoint
        for endpoint in service.endpoints:
            if endpoint in service.mock_responses:
                response = service.mock_responses[endpoint]
                setup_code.append(f"        m.get('{endpoint}', json={response})")
        
        setup_code.extend([
            "        yield m"
        ])
        
        return setup_code
    
    def _setup_message_queue_mock(self, service: ExternalServiceConfiguration) -> List[str]:
        """Set up message queue mocking."""
        return [
            f"@pytest.fixture",
            f"def {service.service_name}_mock(self):",
            f"    \"\"\"Mock {service.service_name} message queue.\"\"\"",
            f"    mock_queue = Mock()",
            f"    # Configure message queue mock behavior",
            f"    return mock_queue"
        ]
    
    def _setup_file_system_mock(self, service: ExternalServiceConfiguration) -> List[str]:
        """Set up file system mocking."""
        return [
            f"@pytest.fixture",
            f"def {service.service_name}_mock(self):",
            f"    \"\"\"Mock {service.service_name} file system.\"\"\"",
            f"    with patch('builtins.open') as mock_open:",
            f"        # Configure file system mock behavior",
            f"        yield mock_open"
        ]
    
    def _setup_cache_mock(self, service: ExternalServiceConfiguration) -> List[str]:
        """Set up cache mocking."""
        return [
            f"@pytest.fixture",
            f"def {service.service_name}_mock(self):",
            f"    \"\"\"Mock {service.service_name} cache.\"\"\"",
            f"    mock_cache = Mock()",
            f"    # Configure cache mock behavior",
            f"    mock_cache.get.return_value = None",
            f"    mock_cache.set.return_value = True",
            f"    return mock_cache"
        ]
    
    def _is_multi_component_test(self, method: TestMethod) -> bool:
        """Check if a test method involves multiple components."""
        # Simple heuristic: check if method has multiple mock usages or mentions multiple services
        return len(method.mock_usage) > 1 or any(
            keyword in method.body.lower() 
            for keyword in ['service', 'repository', 'controller', 'component']
        )
    
    def _generate_multi_component_test_method(self, method: TestMethod) -> List[str]:
        """Generate a multi-component test method."""
        method_name = method.name.lower().replace('test', 'test_')
        if not method_name.startswith('test_'):
            method_name = f"test_{method_name}"
        
        return [
            f"def {method_name}(self, multi_component_scenario):",
            f"    \"\"\"",
            f"    Multi-component integration test: {method.name}",
            f"    \"\"\"",
            f"    # Arrange",
            f"    scenario = multi_component_scenario",
            f"    ",
            f"    # Act",
            f"    # Execute the multi-component interaction",
            f"    result = scenario.environment.get_component('main_service').execute_operation()",
            f"    ",
            f"    # Assert",
            f"    assert result is not None",
            f"    # Verify component interactions",
            f"    scenario.verify_interaction_sequence([",
            f"        {{'component': 'service1', 'method': 'method1'}},",
            f"        {{'component': 'service2', 'method': 'method2'}}",
            f"    ])"
        ]


class IntegrationTestGenerator:
    """
    Generates complete integration test files with environment setup.
    """
    
    def __init__(self):
        """Initialize the integration test generator."""
        self.environment = IntegrationTestEnvironment()
    
    def generate_integration_test_file(self, test_file: TestFile, 
                                     database_config: Optional[DatabaseConfiguration] = None,
                                     external_services: Optional[List[ExternalServiceConfiguration]] = None) -> str:
        """
        Generate a complete integration test file with environment setup.
        
        Args:
            test_file: The original test file to migrate
            database_config: Database configuration for the test
            external_services: External service configurations
            
        Returns:
            Complete Python integration test file content
        """
        lines = []
        
        # Add file header
        lines.extend([
            '"""',
            f'Integration tests migrated from {test_file.path}',
            '"""',
            ''
        ])
        
        # Add imports
        imports = set([
            "import pytest",
            "from unittest.mock import Mock, MagicMock, patch",
            "from typing import Dict, Any, List, Optional"
        ])
        
        if database_config:
            if database_config.database_type == "testcontainer":
                imports.add("import testcontainers.postgres")
            elif database_config.database_type == "in_memory":
                imports.add("import sqlite3")
        
        if external_services:
            for service in external_services:
                if service.service_type == "rest_api":
                    imports.add("import requests_mock")
        
        lines.extend(sorted(imports))
        lines.append("")
        
        # Generate component wiring
        wiring_config = self.environment.configure_component_wiring(test_file)
        lines.extend(wiring_config['component_registry'])
        lines.append("")
        
        # Generate database setup if configured
        if database_config:
            db_setup = self.environment.configure_database_environment(database_config)
            lines.extend(db_setup)
            lines.append("")
        
        # Generate external service mocking if configured
        if external_services:
            service_setup = self.environment.configure_external_service_mocking(external_services)
            lines.extend(service_setup)
            lines.append("")
        
        # Generate component fixtures
        lines.extend(wiring_config['fixtures'])
        lines.append("")
        
        # Generate multi-component test scenarios
        scenario_config = self.environment.create_multi_component_test_scenario(test_file)
        lines.extend(scenario_config['fixtures'])
        lines.append("")
        lines.extend(scenario_config['helper_methods'])
        lines.append("")
        
        # Generate test class
        lines.extend([
            f"class {test_file.class_name}:",
            f'    """Integration tests for {test_file.class_name}."""',
            ""
        ])
        
        # Generate test methods
        lines.extend(scenario_config['test_methods'])
        
        return "\n".join(lines)