"""
Unit tests for BrobotTestScanner.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ..core.models import TestType
from ..discovery.scanner import BrobotTestScanner


class TestBrobotTestScanner:
    """Test cases for BrobotTestScanner."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scanner = BrobotTestScanner()
        self.temp_dir = None
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir:
            # Cleanup is handled by tempfile context manager
            pass
    
    def test_scanner_initialization(self):
        """Test scanner initializes with correct default patterns."""
        assert self.scanner.java_test_patterns == ["*Test.java", "*Tests.java"]
        assert "**/target/**" in self.scanner.exclude_patterns
        assert "org.junit.Test" in self.scanner.junit_imports
    
    def test_scan_nonexistent_directory(self):
        """Test scanning a non-existent directory returns empty list."""
        non_existent = Path("/non/existent/path")
        result = self.scanner.scan_directory(non_existent)
        assert result == []
    
    def test_scan_empty_directory(self):
        """Test scanning an empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            result = self.scanner.scan_directory(temp_path)
            assert result == []
    
    def test_is_test_file_with_junit_import(self):
        """Test identification of test file with JUnit import."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "ExampleTest.java"
            
            test_content = """
package com.example;

import org.junit.Test;
import org.junit.Assert;

public class ExampleTest {
    @Test
    public void testSomething() {
        Assert.assertTrue(true);
    }
}
"""
            test_file.write_text(test_content)
            
            assert self.scanner._is_test_file(test_file) is True
    
    def test_is_test_file_with_test_annotation(self):
        """Test identification of test file with @Test annotation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "ExampleTest.java"
            
            test_content = """
package com.example;

public class ExampleTest {
    @Test
    public void shouldDoSomething() {
        // test code
    }
}
"""
            test_file.write_text(test_content)
            
            assert self.scanner._is_test_file(test_file) is True
    
    def test_is_test_file_with_test_method_pattern(self):
        """Test identification of test file with test method pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "ExampleTest.java"
            
            test_content = """
package com.example;

public class ExampleTest {
    public void testSomething() {
        // test code
    }
}
"""
            test_file.write_text(test_content)
            
            assert self.scanner._is_test_file(test_file) is True
    
    def test_is_not_test_file(self):
        """Test identification of non-test file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            regular_file = temp_path / "RegularClass.java"
            
            regular_content = """
package com.example;

public class RegularClass {
    public void doSomething() {
        // regular code
    }
}
"""
            regular_file.write_text(regular_content)
            
            assert self.scanner._is_test_file(regular_file) is False
    
    def test_extract_package(self):
        """Test package extraction from Java file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "ExampleTest.java"
            
            test_content = """
package com.example.test;

import org.junit.Test;

public class ExampleTest {
    @Test
    public void testSomething() {
        // test code
    }
}
"""
            test_file.write_text(test_content)
            
            package = self.scanner._extract_package(test_file)
            assert package == "com.example.test"
    
    def test_extract_package_no_package(self):
        """Test package extraction when no package declaration exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "ExampleTest.java"
            
            test_content = """
import org.junit.Test;

public class ExampleTest {
    @Test
    public void testSomething() {
        // test code
    }
}
"""
            test_file.write_text(test_content)
            
            package = self.scanner._extract_package(test_file)
            assert package == ""
    
    def test_classify_unit_test(self):
        """Test classification of unit test."""
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
            
            test_file = self.scanner._create_test_file(test_file_path)
            test_type = self.scanner.classify_test_type(test_file)
            
            assert test_type == TestType.UNIT
    
    def test_classify_integration_test_by_path(self):
        """Test classification of integration test by path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            integration_dir = temp_path / "integration"
            integration_dir.mkdir()
            test_file_path = integration_dir / "UserServiceIntegrationTest.java"
            
            test_content = """
package com.example.integration;

import org.junit.Test;

public class UserServiceIntegrationTest {
    @Test
    public void testUserWorkflow() {
        // integration test code
    }
}
"""
            test_file_path.write_text(test_content)
            
            test_file = self.scanner._create_test_file(test_file_path)
            test_type = self.scanner.classify_test_type(test_file)
            
            assert test_type == TestType.INTEGRATION
    
    def test_classify_integration_test_by_spring_annotation(self):
        """Test classification of integration test by Spring Boot annotation."""
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
            
            test_file = self.scanner._create_test_file(test_file_path)
            test_type = self.scanner.classify_test_type(test_file)
            
            assert test_type == TestType.INTEGRATION
    
    def test_extract_dependencies(self):
        """Test dependency extraction from test file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file_path = temp_path / "ExampleTest.java"
            
            test_content = """
package com.example;

import org.junit.Test;
import org.junit.Assert;
import org.mockito.Mock;
import java.util.List;
import com.example.service.UserService;

public class ExampleTest {
    @Test
    public void testSomething() {
        Assert.assertTrue(true);
    }
}
"""
            test_file_path.write_text(test_content)
            
            test_file = self.scanner._create_test_file(test_file_path)
            dependencies = self.scanner.extract_dependencies(test_file)
            
            # Check that dependencies were extracted
            assert len(dependencies) > 0
            
            # Check specific dependencies
            import_names = [dep.java_import for dep in dependencies]
            assert "org.junit.Test" in import_names
            assert "org.junit.Assert" in import_names
            assert "org.mockito.Mock" in import_names
            assert "java.util.List" in import_names
            assert "com.example.service.UserService" in import_names
    
    def test_map_to_python_equivalent(self):
        """Test mapping of Java imports to Python equivalents."""
        assert self.scanner._map_to_python_equivalent("org.junit.Test") == "pytest"
        assert self.scanner._map_to_python_equivalent("org.mockito.Mock") == "unittest.mock.Mock"
        assert self.scanner._map_to_python_equivalent("java.util.List") == "typing.List"
        assert self.scanner._map_to_python_equivalent("unknown.import") == "unknown.import"
    
    def test_requires_adaptation(self):
        """Test identification of imports requiring adaptation."""
        assert self.scanner._requires_adaptation("org.junit.Test") is True
        assert self.scanner._requires_adaptation("org.mockito.Mock") is True
        assert self.scanner._requires_adaptation("java.util.List") is False
        assert self.scanner._requires_adaptation("com.example.CustomClass") is False
    
    def test_is_excluded(self):
        """Test exclusion pattern matching."""
        target_path = Path("/project/target/classes/Test.java")
        build_path = Path("/project/build/Test.java")
        git_path = Path("/project/.git/Test.java")
        regular_path = Path("/project/src/Test.java")
        
        assert self.scanner._is_excluded(target_path) is True
        assert self.scanner._is_excluded(build_path) is True
        assert self.scanner._is_excluded(git_path) is True
        assert self.scanner._is_excluded(regular_path) is False
    
    def test_scan_directory_with_test_files(self):
        """Test scanning directory with actual test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test directory structure
            src_dir = temp_path / "src" / "test" / "java" / "com" / "example"
            src_dir.mkdir(parents=True)
            
            # Create a valid test file
            test_file = src_dir / "UserServiceTest.java"
            test_content = """
package com.example;

import org.junit.Test;
import org.junit.Assert;

public class UserServiceTest {
    @Test
    public void testFindUser() {
        Assert.assertTrue(true);
    }
}
"""
            test_file.write_text(test_content)
            
            # Create a non-test file (should be ignored)
            non_test_file = src_dir / "UserService.java"
            non_test_content = """
package com.example;

public class UserService {
    public void findUser() {
        // regular code
    }
}
"""
            non_test_file.write_text(non_test_content)
            
            # Scan the directory
            result = self.scanner.scan_directory(temp_path)
            
            # Should find only the test file
            assert len(result) == 1
            assert result[0].class_name == "UserServiceTest"
            assert result[0].package == "com.example"
            assert result[0].test_type == TestType.UNIT
    
    def test_create_test_file_success(self):
        """Test successful creation of TestFile object."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file_path = temp_path / "ExampleTest.java"
            
            test_content = """
package com.example;

import org.junit.Test;

public class ExampleTest {
    @Test
    public void testSomething() {
        // test code
    }
}
"""
            test_file_path.write_text(test_content)
            
            test_file = self.scanner._create_test_file(test_file_path)
            
            assert test_file is not None
            assert test_file.class_name == "ExampleTest"
            assert test_file.package == "com.example"
            assert test_file.path == test_file_path
            assert len(test_file.dependencies) > 0
    
    def test_create_test_file_with_invalid_file(self):
        """Test TestFile creation with invalid file."""
        non_existent = Path("/non/existent/file.java")
        
        # Should handle gracefully and return None
        test_file = self.scanner._create_test_file(non_existent)
        assert test_file is None