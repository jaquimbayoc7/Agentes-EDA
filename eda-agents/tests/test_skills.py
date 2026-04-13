"""Tests para src/skills/ — encoding, statistical_tests, timeseries, report_builder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------
from src.skills.encoding import encode_column, reencode_column, encode_all


@pytest.fixture
def cat_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test con columnas categóricas para tests de encoding."""
    train = pd.DataFrame({
        "binary": ["si", "no", "si", "si", "no", "si", "no", "no"],
        "nominal_low": ["A", "B", "A", "B", "A", "B", "A", "B"],
        "nominal_high": ["x", "y", "z", "w", "x", "y", "z", "w"],
        "alta_card": [f"cat_{i}" for i in range(8)],
        "ordinal": ["bajo", "medio", "alto", "bajo", "medio", "alto", "bajo", "medio"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
    })
    test = pd.DataFrame({
        "binary": ["si", "no"],
        "nominal_low": ["A", "B"],
        "nominal_high": ["x", "y"],
        "alta_card": ["cat_0", "cat_1"],
        "ordinal": ["bajo", "alto"],
        "target": [0, 1],
    })
    return train, test


class TestEncodeColumn:
    def test_binary_label(self, cat_frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        train, test = cat_frames
        tr, te, log = encode_column(train, test, "binary", "BINARIA")
        assert log["encoding"] == "label"
        assert tr["binary"].dtype in (np.int64, np.float64, int, float, np.intp)
        assert te["binary"].notna().all()

    def test_nominal_low_onehot(self, cat_frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        train, test = cat_frames
        tr, te, log = encode_column(train, test, "nominal_low", "NOMINAL", ohe_max_categories=5)
        assert log["encoding"] == "onehot"
        assert "nominal_low" not in tr.columns
        assert any(c.startswith("nominal_low_") for c in tr.columns)

    def test_nominal_high_tree_label(self, cat_frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        train, test = cat_frames
        tr, te, log = encode_column(
            train, test, "nominal_high", "NOMINAL",
            model_family="tree", ohe_max_categories=2,
        )
        assert log["encoding"] == "label"

    def test_nominal_high_linear_frequency(self, cat_frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        train, test = cat_frames
        tr, te, log = encode_column(
            train, test, "nominal_high", "NOMINAL",
            model_family="linear", ohe_max_categories=2,
        )
        assert log["encoding"] == "frequency"
        assert tr["nominal_high"].dtype == float

    def test_alta_card_frequency(self, cat_frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        train, test = cat_frames
        tr, te, log = encode_column(train, test, "alta_card", "ALTA_CARD")
        assert log["encoding"] == "frequency"

    def test_ordinal_explicit_order(self, cat_frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        train, test = cat_frames
        tr, te, log = encode_column(
            train, test, "ordinal", "ORDINAL",
            ordinal_order=["bajo", "medio", "alto"],
        )
        assert log["encoding"] == "ordinal"
        assert float(tr["ordinal"].iloc[0]) == 0.0  # "bajo" → 0
        assert float(tr["ordinal"].iloc[2]) == 2.0  # "alto" → 2

    def test_unknown_flag_noop(self, cat_frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        train, test = cat_frames
        tr, te, log = encode_column(train, test, "binary", "DESCONOCIDO")
        assert log["encoding"] == "none"


class TestReencodeColumn:
    def test_label_to_frequency(self) -> None:
        train = pd.DataFrame({"col": [0, 1, 0, 1, 0, 0]})
        test = pd.DataFrame({"col": [0, 1]})
        tr, te, enc = reencode_column(train, test, "col", "label", "linear")
        assert enc == "frequency"
        # Frecuencia de 0 en train = 4/6
        assert abs(tr["col"].iloc[0] - 4 / 6) < 1e-6

    def test_label_to_tree_noop(self) -> None:
        train = pd.DataFrame({"col": [0, 1, 2]})
        test = pd.DataFrame({"col": [0, 1]})
        tr, te, enc = reencode_column(train, test, "col", "label", "tree")
        assert enc == "label"
        assert list(tr["col"]) == [0, 1, 2]


class TestEncodeAll:
    def test_encodes_all_flagged_columns(self, cat_frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        train, test = cat_frames
        flags = {
            "binary": "BINARIA",
            "nominal_low": "NOMINAL",
            "alta_card": "ALTA_CARD",
            "target": "TARGET",
        }
        tr, te, log = encode_all(train, test, flags, target="target", ohe_max_categories=5)
        assert "binary" in log
        assert "nominal_low" in log
        assert "alta_card" in log
        assert "target" not in log  # TARGET must be skipped


# ---------------------------------------------------------------------------
# statistical_tests
# ---------------------------------------------------------------------------
from src.skills.statistical_tests import (
    compute_correlations,
    detect_outliers_iqr,
    test_normality as _test_normality,
    compute_vif,
    breusch_pagan_test,
    suggest_heteroscedasticity_correction,
)


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(5, 2, n),
        "x3": rng.uniform(0, 10, n),
        "target": rng.normal(10, 3, n),
    })


class TestComputeCorrelations:
    def test_returns_spearman_default(self, numeric_df: pd.DataFrame) -> None:
        res = compute_correlations(numeric_df, ["x1", "x2", "x3"])
        assert "spearman" in res
        assert "x1" in res["spearman"]

    def test_explicit_pearson(self, numeric_df: pd.DataFrame) -> None:
        res = compute_correlations(numeric_df, ["x1", "x2"], methods=["pearson"])
        assert "pearson" in res

    def test_multiple_methods(self, numeric_df: pd.DataFrame) -> None:
        res = compute_correlations(numeric_df, ["x1", "x2"], methods=["pearson", "spearman"])
        assert "spearman" in res

    def test_empty_features(self, numeric_df: pd.DataFrame) -> None:
        res = compute_correlations(numeric_df, [])
        assert res == {}


class TestDetectOutliersIqr:
    def test_detects_outliers(self, numeric_df: pd.DataFrame) -> None:
        res = detect_outliers_iqr(numeric_df, ["x1", "x2", "x3"])
        assert "x1" in res
        assert "n_outliers" in res["x1"]
        assert "pct" in res["x1"]
        assert res["x1"]["n_outliers"] >= 0

    def test_missing_col_skipped(self, numeric_df: pd.DataFrame) -> None:
        res = detect_outliers_iqr(numeric_df, ["nonexistent"])
        assert res == {}


class TestTestNormality:
    def test_shapiro_for_small_n(self, numeric_df: pd.DataFrame) -> None:
        res = _test_normality(numeric_df, ["x1"], shapiro_threshold=5000)
        assert res["x1"]["test"] == "shapiro"
        assert "p_value" in res["x1"]

    def test_anderson_for_large_n(self) -> None:
        rng = np.random.RandomState(42)
        big = pd.DataFrame({"col": rng.normal(0, 1, 6000)})
        res = _test_normality(big, ["col"], shapiro_threshold=5000)
        assert res["col"]["test"] == "anderson"

    def test_max_cols_respected(self, numeric_df: pd.DataFrame) -> None:
        res = _test_normality(numeric_df, ["x1", "x2", "x3", "target"], max_cols=2)
        assert len(res) <= 2


class TestComputeVif:
    def test_returns_flagged_and_all(self, numeric_df: pd.DataFrame) -> None:
        flagged, all_vif = compute_vif(numeric_df, ["x1", "x2", "x3"], threshold=10.0)
        assert isinstance(flagged, list)
        assert isinstance(all_vif, dict)
        assert "x1" in all_vif

    def test_single_feature_returns_empty(self, numeric_df: pd.DataFrame) -> None:
        flagged, all_vif = compute_vif(numeric_df, ["x1"])
        assert flagged == []
        assert all_vif == {}


class TestBreuschPagan:
    def test_regression_result(self, numeric_df: pd.DataFrame) -> None:
        res = breusch_pagan_test(numeric_df, "target", ["x1", "x2", "x3"])
        assert "bp_statistic" in res
        assert "bp_pvalue" in res
        assert "heteroscedastic" in res

    def test_insufficient_cols(self, numeric_df: pd.DataFrame) -> None:
        res = breusch_pagan_test(numeric_df, "target", [])
        assert "error" in res


class TestSuggestCorrection:
    def test_no_hetero_returns_none(self) -> None:
        assert suggest_heteroscedasticity_correction({"heteroscedastic": False}, []) is None

    def test_hetero_no_vif_returns_wls(self) -> None:
        assert suggest_heteroscedasticity_correction({"heteroscedastic": True}, []) == "WLS"

    def test_hetero_with_vif_returns_gls(self) -> None:
        assert (
            suggest_heteroscedasticity_correction(
                {"heteroscedastic": True}, [{"column": "x", "vif": 15.0}]
            )
            == "GLS"
        )


# ---------------------------------------------------------------------------
# timeseries
# ---------------------------------------------------------------------------
from src.skills.timeseries import (
    test_stationarity as _test_stationarity,
    determine_differencing_order,
    diagnose_residuals,
    select_ts_model,
)


@pytest.fixture
def ts_series() -> pd.Series:
    rng = np.random.RandomState(42)
    return pd.Series(rng.normal(0, 1, 100))


class TestTestStationarity:
    def test_stationary_series(self, ts_series: pd.Series) -> None:
        res = _test_stationarity(ts_series)
        assert "adf_statistic" in res
        assert "kpss_statistic" in res
        assert "conclusion" in res

    def test_short_series(self) -> None:
        res = _test_stationarity(pd.Series([1, 2, 3]))
        assert res["error"] == "insufficient_data"


class TestDetermineDifferencingOrder:
    def test_stationary_returns_0(self, ts_series: pd.Series) -> None:
        d = determine_differencing_order(ts_series)
        assert d == 0  # White noise is already stationary

    def test_random_walk(self) -> None:
        rng = np.random.RandomState(42)
        walk = pd.Series(rng.normal(0, 1, 200).cumsum())
        d = determine_differencing_order(walk)
        assert d >= 1


class TestDiagnoseResiduals:
    def test_white_noise_residuals(self, ts_series: pd.Series) -> None:
        res = diagnose_residuals(ts_series)
        assert "ljung_box" in res
        assert "jarque_bera" in res
        assert res["n_residuals"] == 100

    def test_short_residuals(self) -> None:
        res = diagnose_residuals(pd.Series([1, 2, 3]))
        assert "error" in res

    def test_numpy_array_input(self) -> None:
        arr = np.random.RandomState(42).normal(0, 1, 50)
        res = diagnose_residuals(arr)
        assert res["n_residuals"] == 50


class TestSelectTsModel:
    def test_non_seasonal(self) -> None:
        modelo, params = select_ts_model(pd.Series([1, 2, 3]), d=1, seasonal=False)
        assert modelo["type"] == "ARIMA"
        assert params["d"] == 1
        assert params["seasonal"] is False

    def test_seasonal(self) -> None:
        modelo, params = select_ts_model(pd.Series([1, 2, 3]), d=1, seasonal=True, m=12)
        assert modelo["type"] == "SARIMA"
        assert params["seasonal"] is True
        assert params["m"] == 12


# ---------------------------------------------------------------------------
# report_builder
# ---------------------------------------------------------------------------
from src.skills.report_builder import (
    build_report_markdown,
    build_report_sections,
    build_decision,
    serialize_state,
)


@pytest.fixture
def sample_state() -> dict:
    return {
        "research_question": "¿Qué factores?",
        "context": "Café colombiano",
        "data_type": "tabular",
        "target": "produccion",
        "refs": [{"title": "Paper1", "doi": "10.1234"}],
        "search_equations": ["coffee AND yield"],
        "hipotesis": {"h1": "H1 text", "h2": "H2 text", "h3": "H3 text"},
        "dataset_size": 1000,
        "desbalance_ratio": 1.5,
        "flag_timeseries": False,
        "encoding_log": {"col1": {"encoding": "label"}},
        "features_nuevas": ["feat_new"],
        "balanceo_log": {"applied": False},
        "hallazgos_eda": {"correlations": {}},
        "tarea_sugerida": "classification",
        "model_family": "tree",
        "modelos_recomendados": [{"name": "XGBoost", "reason": "Best for trees"}],
        "hyperparams_technique": "Optuna",
        "metrica_principal": "f1_macro",
        "advertencias": ["Datos desbalanceados"],
    }


class TestBuildReportSections:
    def test_returns_12_sections(self, sample_state: dict) -> None:
        sections = build_report_sections(sample_state)
        assert len(sections) == 12

    def test_each_section_starts_with_heading(self, sample_state: dict) -> None:
        sections = build_report_sections(sample_state)
        for s in sections:
            assert s.startswith("# §")


class TestBuildReportMarkdown:
    def test_contains_question(self, sample_state: dict) -> None:
        md = build_report_markdown(sample_state)
        assert "¿Qué factores?" in md

    def test_timeseries_section_when_flag_true(self, sample_state: dict) -> None:
        sample_state["flag_timeseries"] = True
        sample_state["modelo_ts"] = {"type": "ARIMA"}
        md = build_report_markdown(sample_state)
        assert "ARIMA" in md

    def test_timeseries_section_no_aplica(self, sample_state: dict) -> None:
        md = build_report_markdown(sample_state)
        assert "No aplica" in md


class TestBuildDecision:
    def test_has_required_keys(self, sample_state: dict) -> None:
        dec = build_decision(sample_state)
        assert dec["tarea"] == "classification"
        assert dec["model_family"] == "tree"
        assert len(dec["modelos_recomendados"]) == 1


class TestSerializeState:
    def test_handles_non_serializable(self) -> None:
        state = {"key": "value", "array": np.array([1, 2, 3])}
        result = serialize_state(state)
        assert result["key"] == "value"
        # numpy array se convierte a string
        assert isinstance(result["array"], str)

    def test_preserves_serializable(self) -> None:
        state = {"a": 1, "b": [1, 2], "c": {"nested": True}}
        result = serialize_state(state)
        assert result == state


# ---------------------------------------------------------------------------
# html_report
# ---------------------------------------------------------------------------
from src.skills.html_report import build_html_report


class TestBuildHtmlReport:
    def test_generates_html_file(self, sample_state: dict, tmp_path: Path) -> None:
        sample_state["run_id"] = "test_html"
        sample_state["figures"] = []
        sample_state["agent_status"] = {"ag1": "ok", "ag2": "ok"}
        sample_state["perfil_columnas"] = {"col1": {"dtype": "float64", "n_unique": 10, "null_pct": 0.0}}
        html_path = build_html_report(sample_state, tmp_path)
        assert html_path.exists()
        assert html_path.suffix == ".html"
        assert html_path.parent.name == "reportesFinales"

    def test_html_contains_question(self, sample_state: dict, tmp_path: Path) -> None:
        sample_state["run_id"] = "test_html2"
        sample_state["figures"] = []
        sample_state["agent_status"] = {}
        sample_state["perfil_columnas"] = {}
        html_path = build_html_report(sample_state, tmp_path)
        content = html_path.read_text(encoding="utf-8")
        assert sample_state["research_question"] in content

    def test_html_contains_sections(self, sample_state: dict, tmp_path: Path) -> None:
        sample_state["run_id"] = "test_html3"
        sample_state["figures"] = []
        sample_state["agent_status"] = {"ag1": "ok"}
        sample_state["perfil_columnas"] = {}
        html_path = build_html_report(sample_state, tmp_path)
        content = html_path.read_text(encoding="utf-8")
        assert "Resumen Ejecutivo" in content
        assert "Hipotesis" in content
        assert "Proximos Pasos" in content

    def test_html_has_theme_toggle(self, sample_state: dict, tmp_path: Path) -> None:
        sample_state["run_id"] = "test_html4"
        sample_state["figures"] = []
        sample_state["agent_status"] = {}
        sample_state["perfil_columnas"] = {}
        html_path = build_html_report(sample_state, tmp_path)
        content = html_path.read_text(encoding="utf-8")
        assert "toggleTheme" in content

    def test_html_embeds_figures(self, sample_state: dict, tmp_path: Path) -> None:
        # Create a small PNG
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        fig_path = figures_dir / "test_fig.png"
        # Minimal 1x1 white PNG
        import struct, zlib
        def _mini_png():
            sig = b'\x89PNG\r\n\x1a\n'
            ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
            ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
            raw = b'\x00\x00\xff\xff\xff'
            idat_data = zlib.compress(raw)
            idat_crc = zlib.crc32(b'IDAT' + idat_data) & 0xffffffff
            idat = struct.pack('>I', len(idat_data)) + b'IDAT' + idat_data + struct.pack('>I', idat_crc)
            iend_crc = zlib.crc32(b'IEND') & 0xffffffff
            iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
            return sig + ihdr + idat + iend
        fig_path.write_bytes(_mini_png())

        sample_state["run_id"] = "test_html5"
        sample_state["figures"] = [{"name": "test_fig.png", "path": str(fig_path), "description": "Test figure"}]
        sample_state["agent_status"] = {}
        sample_state["perfil_columnas"] = {}
        html_path = build_html_report(sample_state, tmp_path)
        content = html_path.read_text(encoding="utf-8")
        assert "data:image/png;base64," in content


# ---------------------------------------------------------------------------
# notebook_builder
# ---------------------------------------------------------------------------
from src.skills.notebook_builder import build_notebook


class TestBuildNotebook:
    def test_generates_ipynb_file(self, sample_state: dict, tmp_path: Path) -> None:
        sample_state["run_id"] = "test_nb"
        sample_state["dataset_path"] = "tests/fixtures/sample_100.csv"
        sample_state["train_path"] = "outputs/test_nb/train.csv"
        sample_state["test_path"] = "outputs/test_nb/test.csv"
        sample_state["dataset_train_final"] = "outputs/test_nb/train_final.csv"
        sample_state["dataset_test_final"] = "outputs/test_nb/test_final.csv"
        sample_state["figures"] = []
        sample_state["random_seed"] = 42
        nb_path = build_notebook(sample_state, tmp_path)
        assert nb_path.exists()
        assert nb_path.suffix == ".ipynb"
        assert nb_path.parent.name == "notebooksFinales"

    def test_notebook_is_valid_json(self, sample_state: dict, tmp_path: Path) -> None:
        sample_state["run_id"] = "test_nb2"
        sample_state["dataset_path"] = "data/test.csv"
        sample_state["train_path"] = ""
        sample_state["test_path"] = ""
        sample_state["dataset_train_final"] = ""
        sample_state["dataset_test_final"] = ""
        sample_state["figures"] = []
        sample_state["random_seed"] = 42
        nb_path = build_notebook(sample_state, tmp_path)
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        assert "cells" in nb
        assert len(nb["cells"]) > 5

    def test_notebook_contains_question(self, sample_state: dict, tmp_path: Path) -> None:
        sample_state["run_id"] = "test_nb3"
        sample_state["dataset_path"] = "data/test.csv"
        sample_state["train_path"] = ""
        sample_state["test_path"] = ""
        sample_state["dataset_train_final"] = ""
        sample_state["dataset_test_final"] = ""
        sample_state["figures"] = []
        sample_state["random_seed"] = 42
        nb_path = build_notebook(sample_state, tmp_path)
        content = nb_path.read_text(encoding="utf-8")
        assert sample_state["research_question"] in content

    def test_notebook_has_code_and_markdown_cells(self, sample_state: dict, tmp_path: Path) -> None:
        sample_state["run_id"] = "test_nb4"
        sample_state["dataset_path"] = "data/test.csv"
        sample_state["train_path"] = ""
        sample_state["test_path"] = ""
        sample_state["dataset_train_final"] = ""
        sample_state["dataset_test_final"] = ""
        sample_state["figures"] = []
        sample_state["random_seed"] = 42
        nb_path = build_notebook(sample_state, tmp_path)
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        cell_types = {c["cell_type"] for c in nb["cells"]}
        assert "markdown" in cell_types
        assert "code" in cell_types

    def test_notebook_regression_model_example(self, sample_state: dict, tmp_path: Path) -> None:
        sample_state["run_id"] = "test_nb5"
        sample_state["tarea_sugerida"] = "regression"
        sample_state["dataset_path"] = "data/test.csv"
        sample_state["train_path"] = ""
        sample_state["test_path"] = ""
        sample_state["dataset_train_final"] = ""
        sample_state["dataset_test_final"] = ""
        sample_state["figures"] = []
        sample_state["random_seed"] = 42
        nb_path = build_notebook(sample_state, tmp_path)
        content = nb_path.read_text(encoding="utf-8")
        assert "Ridge" in content
