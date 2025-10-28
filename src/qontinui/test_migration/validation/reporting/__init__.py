"""
Diagnostic reporting components for test migration validation.
"""

from .comprehensive_analyzer import ComprehensiveAnalyzer
from .diagnostic_reporter_impl import DiagnosticReporterImpl
from .error_analyzer import ErrorAnalyzer
from .report_data_collector import ReportDataCollector
from .report_formatter import ReportFormatter, ReportFormatterFactory, TextReportFormatter
from .report_models import (
    AssertionDifference,
    DependencyDifference,
    DiagnosticReport,
    SetupDifference,
)

__all__ = [
    "DiagnosticReporterImpl",
    "ReportDataCollector",
    "ErrorAnalyzer",
    "ComprehensiveAnalyzer",
    "ReportFormatter",
    "TextReportFormatter",
    "ReportFormatterFactory",
    "DependencyDifference",
    "SetupDifference",
    "AssertionDifference",
    "DiagnosticReport",
]
