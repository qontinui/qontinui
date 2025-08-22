"""
Pytest configuration and fixtures for the test migration system.
"""

import tempfile
from pathlib import Path
from typing import Generator

import pytest

from qontinui.test_migration.core.models import (
    Dependency,
    GuiModel,
    MigrationConfig,
    MockUsage,
    TestFile,
    TestMethod,
    TestType,
)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_java_test_content() -> str:
    """Sample Java test file content for testing."""
    return '''
package com.example.test;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

public class SampleTest {
    
    private SampleClass sampleClass;
    
    @BeforeEach
    void setUp() {
        sampleClass = new SampleClass();
    }
    
    @Test
    void testBasicFunctionality() {
        String result = sampleClass.process("input");
        assertEquals("expected", result);
    }
    
    @Test
    void testEdgeCase() {
        assertThrows(IllegalArgumentException.class, () -> {
            sampleClass.process(null);
        });
    }
}
'''


@pytest.fixture
def sample_test_file(temp_dir: Path) -> TestFile:
    """Create a sample TestFile object for testing."""
    test_path = temp_dir / "SampleTest.java"
    
    return TestFile(
        path=test_path,
        test_type=TestType.UNIT,
        class_name="SampleTest",
        package="com.example.test",
        dependencies=[
            Dependency(
                java_import="org.junit.jupiter.api.Test",
                python_equivalent="pytest",
                requires_adaptation=True
            ),
            Dependency(
                java_import="org.junit.jupiter.api.Assertions",
                python_equivalent="assert",
                requires_adaptation=True
            )
        ],
        test_methods=[
            TestMethod(
                name="testBasicFunctionality",
                annotations=["@Test"],
                body="String result = sampleClass.process(\"input\");\nassertEquals(\"expected\", result);",
                assertions=["assertEquals(\"expected\", result)"]
            ),
            TestMethod(
                name="testEdgeCase",
                annotations=["@Test"],
                body="assertThrows(IllegalArgumentException.class, () -> {\n    sampleClass.process(null);\n});",
                assertions=["assertThrows(IllegalArgumentException.class, () -> { sampleClass.process(null); })"]
            )
        ],
        setup_methods=[
            TestMethod(
                name="setUp",
                annotations=["@BeforeEach"],
                body="sampleClass = new SampleClass();"
            )
        ]
    )


@pytest.fixture
def sample_mock_usage() -> MockUsage:
    """Create a sample MockUsage object for testing."""
    gui_model = GuiModel(
        model_name="TestWindow",
        elements={"button1": {"type": "button", "text": "Click Me"}},
        actions=["click", "hover"],
        state_properties={"visible": True, "enabled": True}
    )
    
    return MockUsage(
        mock_type="brobot_mock",
        mock_class="BrobotMock",
        gui_model=gui_model,
        simulation_scope="method",
        configuration={"mock_gui": True, "simulate_actions": True}
    )


@pytest.fixture
def migration_config(temp_dir: Path) -> MigrationConfig:
    """Create a sample MigrationConfig for testing."""
    source_dir = temp_dir / "java_tests"
    target_dir = temp_dir / "python_tests"
    source_dir.mkdir()
    target_dir.mkdir()
    
    return MigrationConfig(
        source_directories=[source_dir],
        target_directory=target_dir,
        preserve_structure=True,
        enable_mock_migration=True,
        diagnostic_level="detailed",
        parallel_execution=False,  # Disable for testing
        comparison_mode="behavioral"
    )


@pytest.fixture
def java_test_files(temp_dir: Path) -> list[Path]:
    """Create sample Java test files for testing."""
    java_dir = temp_dir / "java_tests"
    java_dir.mkdir()
    
    # Create unit test
    unit_test = java_dir / "UnitTest.java"
    unit_test.write_text('''
package com.example;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class UnitTest {
    @Test
    void testMethod() {
        assertTrue(true);
    }
}
''')
    
    # Create integration test
    integration_test = java_dir / "IntegrationTest.java"
    integration_test.write_text('''
package com.example;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
public class IntegrationTest {
    @Test
    void testIntegration() {
        assertTrue(true);
    }
}
''')
    
    return [unit_test, integration_test]