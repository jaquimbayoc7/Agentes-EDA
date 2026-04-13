"""Skills reutilizables: encoding, tests estadísticos, timeseries, reportes."""

from src.skills.encoding import encode_column, reencode_column, encode_all
from src.skills.statistical_tests import (
    compute_correlations,
    detect_outliers_iqr,
    test_normality,
    compute_vif,
    breusch_pagan_test,
    suggest_heteroscedasticity_correction,
)
from src.skills.timeseries import (
    test_stationarity,
    determine_differencing_order,
    diagnose_residuals,
    select_ts_model,
)
from src.skills.report_builder import (
    build_report_markdown,
    build_report_sections,
    convert_to_pdf,
    build_decision,
    serialize_state,
)

__all__ = [
    # encoding
    "encode_column",
    "reencode_column",
    "encode_all",
    # statistical_tests
    "compute_correlations",
    "detect_outliers_iqr",
    "test_normality",
    "compute_vif",
    "breusch_pagan_test",
    "suggest_heteroscedasticity_correction",
    # timeseries
    "test_stationarity",
    "determine_differencing_order",
    "diagnose_residuals",
    "select_ts_model",
    # report_builder
    "build_report_markdown",
    "build_report_sections",
    "convert_to_pdf",
    "build_decision",
    "serialize_state",
]
