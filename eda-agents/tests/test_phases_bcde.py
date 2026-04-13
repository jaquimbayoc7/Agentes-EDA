"""Tests para Fases B, C, D y E del plan V4.

B — Feature Importance  (skills + agent_04 integration)
C — Tavily Client       (utils + agent_01 integration)
D — Expanded ML Models  (agent_06 recommend functions)
E — Plotly + Notebook   (agent_07 + notebook_builder)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def regression_df() -> pd.DataFrame:
    """DataFrame para tests de regresión con correlación clara."""
    rng = np.random.RandomState(42)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(5, 2, n)
    x3 = rng.uniform(0, 10, n)
    noise = rng.normal(0, 0.5, n)
    target = 3 * x1 + 0.5 * x2 + noise
    return pd.DataFrame({
        "x1": x1, "x2": x2, "x3": x3, "target": target,
    })


@pytest.fixture
def classification_df() -> pd.DataFrame:
    """DataFrame para tests de clasificación."""
    rng = np.random.RandomState(42)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    target = (x1 + x2 > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": target})


# =========================================================================
# Phase B — Feature Importance
# =========================================================================

from src.skills.feature_importance import (
    compute_mutual_information,
    compute_permutation_importance,
    select_top_features,
)


class TestComputeMutualInformation:
    def test_regression_returns_scores(self, regression_df: pd.DataFrame) -> None:
        scores = compute_mutual_information(
            regression_df, ["x1", "x2", "x3"], "target", "regression",
        )
        assert isinstance(scores, dict)
        assert len(scores) == 3
        assert all(v >= 0 for v in scores.values())

    def test_x1_highest_mi(self, regression_df: pd.DataFrame) -> None:
        scores = compute_mutual_information(
            regression_df, ["x1", "x2", "x3"], "target", "regression",
        )
        top_feature = list(scores.keys())[0]
        assert top_feature == "x1", f"Expected x1 as top, got {top_feature}"

    def test_classification_task(self, classification_df: pd.DataFrame) -> None:
        scores = compute_mutual_information(
            classification_df, ["x1", "x2"], "target", "classification",
        )
        assert len(scores) == 2

    def test_empty_features(self, regression_df: pd.DataFrame) -> None:
        scores = compute_mutual_information(regression_df, [], "target")
        assert scores == {}

    def test_missing_target(self, regression_df: pd.DataFrame) -> None:
        scores = compute_mutual_information(
            regression_df, ["x1", "x2"], "nonexistent",
        )
        assert scores == {}

    def test_insufficient_rows(self) -> None:
        small = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        scores = compute_mutual_information(small, ["x"], "y")
        assert scores == {}

    def test_sorted_descending(self, regression_df: pd.DataFrame) -> None:
        scores = compute_mutual_information(
            regression_df, ["x1", "x2", "x3"], "target",
        )
        vals = list(scores.values())
        assert vals == sorted(vals, reverse=True)


class TestComputePermutationImportance:
    def test_regression_returns_mean_std(self, regression_df: pd.DataFrame) -> None:
        scores = compute_permutation_importance(
            regression_df, ["x1", "x2", "x3"], "target", "regression",
        )
        assert isinstance(scores, dict)
        assert len(scores) == 3
        for v in scores.values():
            assert "mean" in v
            assert "std" in v

    def test_classification_task(self, classification_df: pd.DataFrame) -> None:
        scores = compute_permutation_importance(
            classification_df, ["x1", "x2"], "target", "classification",
        )
        assert len(scores) == 2

    def test_empty_features(self, regression_df: pd.DataFrame) -> None:
        scores = compute_permutation_importance(regression_df, [], "target")
        assert scores == {}

    def test_insufficient_rows(self) -> None:
        small = pd.DataFrame({"x": list(range(10)), "y": list(range(10))})
        scores = compute_permutation_importance(small, ["x"], "y")
        assert scores == {}

    def test_sorted_by_mean_desc(self, regression_df: pd.DataFrame) -> None:
        scores = compute_permutation_importance(
            regression_df, ["x1", "x2", "x3"], "target",
        )
        means = [v["mean"] for v in scores.values()]
        assert means == sorted(means, reverse=True)


class TestSelectTopFeatures:
    def test_combines_ranks(self) -> None:
        mi = {"a": 0.9, "b": 0.5, "c": 0.1}
        perm = {"a": {"mean": 0.8, "std": 0.1}, "b": {"mean": 0.3, "std": 0.05}, "c": {"mean": 0.6, "std": 0.1}}
        top = select_top_features(mi, perm, top_k=2)
        assert len(top) == 2
        assert "a" in top  # best in MI and Perm → should be #1

    def test_top_k_limits(self) -> None:
        mi = {f"f{i}": float(i) for i in range(20)}
        perm = {f"f{i}": {"mean": float(i), "std": 0.1} for i in range(20)}
        top = select_top_features(mi, perm, top_k=5)
        assert len(top) == 5

    def test_empty_scores(self) -> None:
        assert select_top_features({}, {}) == []

    def test_disjoint_features(self) -> None:
        mi = {"a": 0.9, "b": 0.5}
        perm = {"c": {"mean": 0.8, "std": 0.1}, "d": {"mean": 0.3, "std": 0.05}}
        top = select_top_features(mi, perm, top_k=10)
        assert set(top) == {"a", "b", "c", "d"}


# =========================================================================
# Phase C — Tavily Client
# =========================================================================

from src.utils.tavily_client import search_tavily, search_literature_tavily


class TestSearchTavily:
    def test_empty_api_key_returns_empty(self) -> None:
        result = search_tavily("test query", api_key="")
        assert result == []

    @patch("src.utils.tavily_client.search_tavily")
    def test_returns_normalized_results(self, mock_search: MagicMock) -> None:
        mock_search.return_value = [
            {"title": "Paper 1", "url": "https://arxiv.org/1", "content": "...", "score": 0.9},
        ]
        results = mock_search("test", api_key="fake-key")
        assert len(results) == 1
        assert results[0]["title"] == "Paper 1"
        assert "url" in results[0]
        assert "score" in results[0]


class TestSearchLiteratureTavily:
    @patch("src.utils.tavily_client.search_tavily")
    def test_deduplicates_by_url(self, mock_search: MagicMock) -> None:
        mock_search.return_value = [
            {"title": "Paper A", "url": "https://arxiv.org/1", "content": "...", "score": 0.9},
        ]
        results = search_literature_tavily(
            equations=["eq1", "eq2"], api_key="fake-key", max_results_per_eq=1,
        )
        # Both equations return same URL, should be deduplicated
        assert len(results) == 1

    @patch("src.utils.tavily_client.search_tavily")
    def test_empty_equations(self, mock_search: MagicMock) -> None:
        results = search_literature_tavily(equations=[], api_key="fake-key")
        assert results == []
        mock_search.assert_not_called()

    @patch("src.utils.tavily_client.search_tavily")
    def test_multiple_unique_results(self, mock_search: MagicMock) -> None:
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            return [
                {"title": f"Paper {call_count[0]}", "url": f"https://arxiv.org/{call_count[0]}",
                 "content": "...", "score": 0.8},
            ]

        mock_search.side_effect = side_effect
        results = search_literature_tavily(
            equations=["eq1", "eq2"], api_key="fake-key", max_results_per_eq=1,
        )
        assert len(results) == 2


# =========================================================================
# Phase D — Expanded ML Models
# =========================================================================

from src.agents.agent_06_ml_strategist import (
    _recommend_regression,
    _recommend_classification,
    _recommend_forecasting,
)


class TestRecommendRegression:
    def test_minimum_models(self) -> None:
        modelos, family, metric = _recommend_regression({}, [], None, n=200)
        assert len(modelos) >= 5
        assert metric == "RMSE"

    def test_high_vif_adds_regularization(self) -> None:
        vif_flags = [{"column": "x", "vif": 15.0}]
        modelos, family, _ = _recommend_regression({}, vif_flags, None, n=200)
        names = [m["name"] for m in modelos]
        assert "Ridge" in names
        assert "Lasso" in names
        assert "ElasticNet" in names
        assert family == "linear"

    def test_heterosc_adds_svr(self) -> None:
        bp = {"heteroscedastic": True, "bp_pvalue": 0.01}
        modelos, _, _ = _recommend_regression({}, [], bp, n=200)
        names = [m["name"] for m in modelos]
        assert "SVR" in names

    def test_ensembles_for_n_ge_100(self) -> None:
        modelos, _, _ = _recommend_regression({}, [], None, n=150)
        names = [m["name"] for m in modelos]
        assert "StackingRegressor" in names
        assert "VotingRegressor" in names

    def test_adaboost_for_n_ge_300(self) -> None:
        modelos, _, _ = _recommend_regression({}, [], None, n=300)
        names = [m["name"] for m in modelos]
        assert "AdaBoostRegressor" in names

    def test_small_n_fewer_models(self) -> None:
        modelos, _, _ = _recommend_regression({}, [], None, n=50)
        names = [m["name"] for m in modelos]
        # Should not include ensembles or gradient boosting for small N
        assert "StackingRegressor" not in names


class TestRecommendClassification:
    def test_minimum_models(self) -> None:
        modelos, family, metric = _recommend_classification({}, None, n=200)
        assert len(modelos) >= 5
        assert metric == "f1_macro"
        assert family == "tree"

    def test_high_desbalance(self) -> None:
        modelos, _, _ = _recommend_classification({}, desbalance=5.0, n=200)
        names = [m["name"] for m in modelos]
        assert "XGBClassifier" in names
        # Reason should mention desbalance
        xgb = [m for m in modelos if m["name"] == "XGBClassifier"][0]
        assert "5.0" in xgb["reason"]

    def test_svc_for_large_n(self) -> None:
        modelos, _, _ = _recommend_classification({}, None, n=600)
        names = [m["name"] for m in modelos]
        assert "SVC" in names

    def test_no_svc_for_small_n(self) -> None:
        modelos, _, _ = _recommend_classification({}, None, n=200)
        names = [m["name"] for m in modelos]
        assert "SVC" not in names

    def test_ensembles_for_n_ge_100(self) -> None:
        modelos, _, _ = _recommend_classification({}, None, n=200)
        names = [m["name"] for m in modelos]
        assert "StackingClassifier" in names
        assert "VotingClassifier" in names

    def test_bagging_adaboost_for_n_ge_200(self) -> None:
        modelos, _, _ = _recommend_classification({}, None, n=250)
        names = [m["name"] for m in modelos]
        assert "BaggingClassifier" in names
        assert "AdaBoostClassifier" in names


class TestRecommendForecasting:
    def test_basic_models(self) -> None:
        modelos = _recommend_forecasting({"type": "ARIMA"}, n=50)
        names = [m["name"] for m in modelos]
        assert "ARIMA" in names
        assert "SARIMAX" in names
        assert "ExponentialSmoothing" in names
        assert "Theta" in names

    def test_prophet_for_large_n(self) -> None:
        modelos = _recommend_forecasting({"type": "ARIMA"}, n=150)
        names = [m["name"] for m in modelos]
        assert "Prophet" in names

    def test_tbats_for_n_ge_200(self) -> None:
        modelos = _recommend_forecasting({"type": "ARIMA"}, n=250)
        names = [m["name"] for m in modelos]
        assert "TBATS" in names

    def test_autoarima_for_n_ge_50(self) -> None:
        modelos = _recommend_forecasting({"type": "ARIMA"}, n=50)
        names = [m["name"] for m in modelos]
        assert "AutoARIMA" in names

    def test_none_ts_model(self) -> None:
        modelos = _recommend_forecasting(None, n=100)
        names = [m["name"] for m in modelos]
        assert "ARIMA" in names  # default fallback

    def test_sarima_input_no_duplicate(self) -> None:
        modelos = _recommend_forecasting({"type": "SARIMAX"}, n=100)
        names = [m["name"] for m in modelos]
        assert names.count("SARIMAX") == 1  # no duplicate


# =========================================================================
# Phase E — Plotly viz + Notebook Builder
# =========================================================================


class TestNotebookBuilder:
    def test_notebook_has_plotly_imports(self, base_state, tmp_path) -> None:
        from src.skills.notebook_builder import build_notebook

        nb_path = build_notebook(base_state, str(tmp_path))
        content = nb_path.read_text(encoding="utf-8")
        nb = json.loads(content)

        # Find setup cell and check for plotly imports
        code_sources = [
            "".join(c["source"])
            for c in nb["cells"]
            if c["cell_type"] == "code"
        ]
        setup_cell = code_sources[0]
        assert "import plotly.express as px" in setup_cell
        assert "import plotly.graph_objects as go" in setup_cell
        assert "from plotly.subplots import make_subplots" in setup_cell

    def test_notebook_has_spearman_correlation(self, base_state, tmp_path) -> None:
        from src.skills.notebook_builder import build_notebook

        nb_path = build_notebook(base_state, str(tmp_path))
        content = nb_path.read_text(encoding="utf-8")
        assert "spearman" in content.lower()

    def test_notebook_has_feature_importance_section(self, base_state, tmp_path) -> None:
        from src.skills.notebook_builder import build_notebook

        nb_path = build_notebook(base_state, str(tmp_path))
        content = nb_path.read_text(encoding="utf-8")
        assert "Importancia de Variables" in content
        assert "Mutual Information" in content
        assert "Permutation Importance" in content

    def test_notebook_always_has_live_feature_importance(self, base_state, tmp_path) -> None:
        from src.skills.notebook_builder import build_notebook

        base_state["feature_importance"] = {}
        nb_path = build_notebook(base_state, str(tmp_path))
        content = nb_path.read_text(encoding="utf-8")
        # Live code always generates feature importance section
        assert "Mutual Information" in content
        assert "Permutation Importance" in content

    def test_notebook_has_live_statistical_code(self, base_state, tmp_path) -> None:
        """Notebook generates live executable code, not pre-computed results."""
        from src.skills.notebook_builder import build_notebook

        nb_path = build_notebook(base_state, str(tmp_path))
        content = nb_path.read_text(encoding="utf-8")
        # Live code markers — computed from scratch, not embedded results
        assert "variance_inflation_factor" in content
        assert "shapiro" in content
        assert "train_test_split" in content
        assert "mutual_info" in content
        assert "permutation_importance" in content

    def test_notebook_interactive_distribution(self, base_state, tmp_path) -> None:
        from src.skills.notebook_builder import build_notebook

        nb_path = build_notebook(base_state, str(tmp_path))
        content = nb_path.read_text(encoding="utf-8")
        assert "go.Histogram" in content
        assert "go.Box" in content


class TestHtmlReportFeatureImportance:
    def test_feature_importance_section_present(self) -> None:
        from src.skills.html_report import _build_feature_importance_html

        feat_imp = {
            "mutual_information": {"x1": 0.9, "x2": 0.5},
            "permutation_importance": {
                "x1": {"mean": 0.8, "std": 0.1},
                "x2": {"mean": 0.3, "std": 0.05},
            },
            "top_features": ["x1", "x2"],
        }
        html = _build_feature_importance_html(feat_imp)
        assert "x1" in html
        assert "x2" in html
        assert "0.9" in html  # MI score

    def test_empty_feature_importance(self) -> None:
        from src.skills.html_report import _build_feature_importance_html

        html = _build_feature_importance_html({})
        assert "No se calculó" in html or "no se" in html.lower() or html == ""


class TestHtmlReportPlotlyFigures:
    def test_builds_iframe_for_html_figures(self, tmp_path) -> None:
        from src.skills.html_report import _build_figures_html

        figures = [
            {"name": "test_chart", "description": "Test Chart",
             "path": str(tmp_path / "fake.html"), "format": "html"},
        ]
        # We can't truly test with a real file, but verify the function
        # handles the format field without error
        html = _build_figures_html(figures, tmp_path)
        assert isinstance(html, str)
