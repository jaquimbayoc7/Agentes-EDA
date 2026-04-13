"""Tests para los 8 agentes con llamadas LLM mockeadas."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config_no_key():
    """Mock PipelineConfig sin API keys (fallback path)."""
    with patch("src.utils.config.PipelineConfig.from_state") as mock_from_state:
        cfg = MagicMock()
        cfg.anthropic_api_key = ""
        cfg.model = "claude-sonnet-4-5"
        cfg.max_tokens = 4096
        cfg.random_seed = 42
        cfg.imbalance_thresholds = MagicMock(oversample=3, hybrid=10, undersample=30)
        cfg.encoding = MagicMock(ohe_max_categories=3)
        cfg.split = MagicMock(test_size=0.2, stratify=True)
        cfg.vif_threshold = 10
        cfg.bp_pvalue = 0.05
        mock_from_state.return_value = cfg
        yield cfg


@pytest.fixture
def mock_config_with_key():
    """Mock PipelineConfig con API keys (Claude path)."""
    with patch("src.utils.config.PipelineConfig.from_state") as mock_from_state:
        cfg = MagicMock()
        cfg.anthropic_api_key = "sk-test-key"
        cfg.model = "claude-sonnet-4-5"
        cfg.max_tokens = 4096
        cfg.random_seed = 42
        cfg.imbalance_thresholds = MagicMock(oversample=3, hybrid=10, undersample=30)
        cfg.encoding = MagicMock(ohe_max_categories=3)
        cfg.split = MagicMock(test_size=0.2, stratify=True)
        cfg.vif_threshold = 10
        cfg.bp_pvalue = 0.05
        mock_from_state.return_value = cfg
        yield cfg


# ---------------------------------------------------------------------------
# Agent 01 — Research Lead
# ---------------------------------------------------------------------------


class TestAgent01:
    def test_fallback_without_api_key(self, base_state, mock_config_no_key):
        from src.agents.agent_01_research_lead import research_lead

        result = research_lead(base_state)
        assert "ag1" in result["agent_status"]
        assert result["search_equations"]
        assert "h1" in result["hipotesis"]
        assert "h2" in result["hipotesis"]
        assert "h3" in result["hipotesis"]

    @patch("src.agents.agent_01_research_lead.call_claude_json")
    def test_with_claude(self, mock_claude, base_state, mock_config_with_key):
        mock_claude.side_effect = [
            {"equations": ["eq1 AND eq2", "eq3 OR eq4", "eq5"]},
            {"refs": [
                {"title": "Study on Target Prediction", "authors": "Smith et al.",
                 "year": "2020", "doi": "", "key_finding": "Positive correlation found",
                 "relevance": "Directly related"}
            ]},
            {"h1": "La relación principal confirmada por literatura es positiva para el target",
             "h2": "Existen relaciones no documentadas que podrían ser relevantes aquí",
             "h3": "El supuesto más común sobre esta pregunta podría ser incorrecto"},
            {"task": "classification"},
        ]

        from src.agents.agent_01_research_lead import research_lead

        result = research_lead(base_state)
        assert len(result["search_equations"]) == 3
        assert "confirmada" in result["hipotesis"]["h1"]
        assert result["tarea_sugerida"] == "classification"

    def test_task_inference_heuristic(self, base_state, mock_config_no_key):
        from src.agents.agent_01_research_lead import _infer_task

        assert _infer_task("predicting house prices", [], mock_config_no_key) == "regression"
        assert _infer_task("clasificar pacientes", [], mock_config_no_key) == "classification"
        assert _infer_task("forecast de ventas", [], mock_config_no_key) == "forecasting"


# ---------------------------------------------------------------------------
# Agent 02 — Data Steward
# ---------------------------------------------------------------------------


class TestAgent02:
    def test_steward_runs(self, base_state, mock_config_no_key):
        from src.agents.agent_02_data_steward import data_steward

        base_state["dataset_path"] = base_state["train_path"]
        result = data_steward(base_state)
        assert result["agent_status"]["ag2"] == "ok"
        assert result["perfil_columnas"]
        assert result["encoding_flags"]
        assert result["train_path"]
        assert result["test_path"]

    def test_encoding_flags_heuristic(self, sample_df, mock_config_no_key):
        from src.agents.agent_02_data_steward import _infer_encoding_flags

        flags = _infer_encoding_flags(sample_df, "target", "fecha")
        assert flags["target"] == "TARGET"
        assert flags["fecha"] == "FECHA"
        assert flags["feature_num"] == "NUMERICA"

    @patch("src.agents.agent_02_data_steward.call_claude_json")
    def test_encoding_flags_claude(self, mock_claude, sample_df, mock_config_with_key):
        expected_flags = {
            col: "NUMERICA" for col in sample_df.columns
        }
        expected_flags["target"] = "TARGET"
        expected_flags["fecha"] = "FECHA"
        mock_claude.return_value = {"flags": expected_flags}

        from src.agents.agent_02_data_steward import _infer_encoding_flags

        flags = _infer_encoding_flags(sample_df, "target", "fecha", mock_config_with_key)
        assert flags["target"] == "TARGET"


# ---------------------------------------------------------------------------
# Agent 03 — Data Engineer
# ---------------------------------------------------------------------------


class TestAgent03:
    def test_engineer_runs(self, base_state, mock_config_no_key):
        from src.agents.agent_03_data_engineer import data_engineer

        base_state["encoding_flags"] = {
            "feature_num": "NUMERICA",
            "feature_cat": "NOMINAL",
            "feature_ord": "ORDINAL",
            "fecha": "FECHA",
            "target": "TARGET",
        }
        result = data_engineer(base_state)
        assert result["agent_status"]["ag3"] == "ok"
        assert result["dataset_train_provisional"]

    @patch("src.agents.agent_03_data_engineer.call_claude_json")
    def test_feature_engineering_with_claude(self, mock_claude, base_state, mock_config_with_key):
        mock_claude.return_value = {
            "features": [
                {"name": "feat_sq", "expr": "feature_num ** 2", "reason": "test"},
            ]
        }

        from src.agents.agent_03_data_engineer import _feature_engineering

        df_train = pd.read_csv(base_state["train_path"])
        df_test = pd.read_csv(base_state["test_path"])
        df_train, df_test, new_feats = _feature_engineering(
            df_train, df_test, base_state, mock_config_with_key
        )
        assert "feat_sq" in new_feats
        assert "feat_sq" in df_train.columns

    def test_feature_engineering_no_key(self, base_state, mock_config_no_key):
        from src.agents.agent_03_data_engineer import _feature_engineering

        df_train = pd.read_csv(base_state["train_path"])
        df_test = pd.read_csv(base_state["test_path"])
        _, _, new_feats = _feature_engineering(
            df_train, df_test, base_state, mock_config_no_key
        )
        assert new_feats == []


# ---------------------------------------------------------------------------
# Agent 04 — Statistician
# ---------------------------------------------------------------------------


class TestAgent04:
    def test_statistician_runs(self, base_state, mock_config_no_key, tmp_path):
        from src.agents.agent_04_statistician import statistician

        # Prepare a provisional dataset
        df = pd.read_csv(base_state["train_path"])
        prov_path = str(tmp_path / "prov.csv")
        df.to_csv(prov_path, index=False)
        base_state["dataset_train_provisional"] = prov_path
        base_state["tarea_sugerida"] = "classification"

        result = statistician(base_state)
        assert result["agent_status"]["ag4"] == "ok"
        assert "outliers" in result["hallazgos_eda"]
        assert "normality" in result["hallazgos_eda"]

    @patch("src.agents.agent_04_statistician.call_claude_json")
    def test_interpretation_with_claude(self, mock_claude, base_state, mock_config_with_key, tmp_path):
        mock_claude.return_value = {"interpretation": "Los datos muestran normalidad."}

        from src.agents.agent_04_statistician import _interpret_findings

        result = _interpret_findings({}, [], None, mock_config_with_key)
        assert "normalidad" in result

    def test_interpretation_no_key(self, mock_config_no_key):
        from src.agents.agent_04_statistician import _interpret_findings

        result = _interpret_findings({}, [], None, mock_config_no_key)
        assert result == ""


# ---------------------------------------------------------------------------
# Agent 05 — TS Analyst
# ---------------------------------------------------------------------------


class TestAgent05:
    def test_ts_analyst_no_time_col(self, base_state, mock_config_no_key):
        from src.agents.agent_05_ts_analyst import ts_analyst

        base_state["time_col"] = None
        result = ts_analyst(base_state)
        assert result["agent_status"]["ag5"] == "ok"
        assert result["modelo_ts"] is None

    def test_ts_analyst_with_time_col(self, base_state, mock_config_no_key, tmp_path):
        from src.agents.agent_05_ts_analyst import ts_analyst

        df = pd.read_csv(base_state["train_path"])
        prov_path = str(tmp_path / "prov_ts.csv")
        df.to_csv(prov_path, index=False)
        base_state["dataset_train_provisional"] = prov_path
        base_state["time_col"] = "fecha"

        result = ts_analyst(base_state)
        assert result["agent_status"]["ag5"] == "ok"
        assert result["modelo_ts"] is not None


# ---------------------------------------------------------------------------
# Agent 06 — ML Strategist
# ---------------------------------------------------------------------------


class TestAgent06:
    def test_strategist_classification(self, base_state, mock_config_no_key):
        from src.agents.agent_06_ml_strategist import ml_strategist

        base_state["tarea_sugerida"] = "classification"
        base_state["hallazgos_eda"] = {}
        result = ml_strategist(base_state)
        assert result["agent_status"]["ag6"] == "ok"
        assert result["modelos_recomendados"]
        assert result["metrica_principal"] == "f1_macro"

    def test_strategist_regression(self, base_state, mock_config_no_key):
        from src.agents.agent_06_ml_strategist import ml_strategist

        base_state["tarea_sugerida"] = "regression"
        base_state["hallazgos_eda"] = {}
        result = ml_strategist(base_state)
        assert result["agent_status"]["ag6"] == "ok"
        assert result["metrica_principal"] == "RMSE"

    @patch("src.agents.agent_06_ml_strategist.call_claude_json")
    def test_strategist_with_claude(self, mock_claude, base_state, mock_config_with_key):
        mock_claude.return_value = {
            "models": [{"name": "XGBClassifier", "reason": "Claude says so"}],
            "model_family": "tree",
            "metric": "f1_weighted",
            "hp_technique": "Optuna",
            "warnings": ["test warning"],
        }

        from src.agents.agent_06_ml_strategist import ml_strategist

        base_state["tarea_sugerida"] = "classification"
        base_state["hallazgos_eda"] = {}
        result = ml_strategist(base_state)
        assert result["metrica_principal"] == "f1_weighted"
        assert result["hyperparams_technique"] == "Optuna"


# ---------------------------------------------------------------------------
# Agent 07 — Viz Designer
# ---------------------------------------------------------------------------


class TestAgent07:
    def test_viz_designer_runs(self, base_state, mock_config_no_key, tmp_path):
        from src.agents.agent_07_viz_designer import viz_designer

        df = pd.read_csv(base_state["train_path"])
        final_path = str(tmp_path / "train_final.csv")
        df.to_csv(final_path, index=False)
        base_state["dataset_train_final"] = final_path

        result = viz_designer(base_state)
        assert result["agent_status"]["ag7"] == "ok"
        assert len(result["figures"]) > 0

    def test_viz_includes_boxplots(self, base_state, mock_config_no_key, tmp_path):
        from src.agents.agent_07_viz_designer import viz_designer

        df = pd.read_csv(base_state["train_path"])
        final_path = str(tmp_path / "train_final.csv")
        df.to_csv(final_path, index=False)
        base_state["dataset_train_final"] = final_path

        result = viz_designer(base_state)
        fig_names = [f["name"] for f in result["figures"]]
        assert any(n.startswith("box_") for n in fig_names)

    def test_viz_includes_target_dist(self, base_state, mock_config_no_key, tmp_path):
        from src.agents.agent_07_viz_designer import viz_designer

        df = pd.read_csv(base_state["train_path"])
        final_path = str(tmp_path / "train_final.csv")
        df.to_csv(final_path, index=False)
        base_state["dataset_train_final"] = final_path

        result = viz_designer(base_state)
        fig_names = [f["name"] for f in result["figures"]]
        assert "target_dist.png" in fig_names

    def test_no_dataset_fallback(self, base_state, mock_config_no_key):
        from src.agents.agent_07_viz_designer import viz_designer

        base_state["dataset_train_final"] = ""
        base_state["dataset_train_provisional"] = ""
        result = viz_designer(base_state)
        assert result["agent_status"]["ag7"] == "fallback"


# ---------------------------------------------------------------------------
# Agent 08 — Technical Writer
# ---------------------------------------------------------------------------


class TestAgent08:
    def test_writer_runs(self, base_state, mock_config_no_key, tmp_path):
        from src.agents.agent_08_technical_writer import technical_writer

        base_state["run_id"] = "test_writer"
        # Need outputs dir writable
        result = technical_writer(base_state)
        assert result["agent_status"]["ag8"] == "ok"
        report_path = Path("outputs") / "test_writer" / "report.md"
        assert report_path.exists()

    def test_writer_generates_decision_json(self, base_state, mock_config_no_key):
        from src.agents.agent_08_technical_writer import technical_writer

        base_state["run_id"] = "test_writer_dec"
        base_state["modelos_recomendados"] = [{"name": "XGB", "reason": "test"}]
        result = technical_writer(base_state)
        assert result["agent_status"]["ag8"] == "ok"
        dec_path = Path("outputs") / "test_writer_dec" / "decision.json"
        assert dec_path.exists()
        dec = json.loads(dec_path.read_text(encoding="utf-8"))
        assert dec["modelos_recomendados"][0]["name"] == "XGB"

    @patch("src.agents.agent_08_technical_writer.call_claude")
    def test_writer_enrichment_with_claude(self, mock_claude, base_state, mock_config_with_key):
        mock_claude.return_value = "# §1 Pregunta enriquecida\n\nNarrativa generada por Claude."

        from src.agents.agent_08_technical_writer import _build_enriched_report

        report = _build_enriched_report(base_state, mock_config_with_key)
        assert "enriquecida" in report

    def test_writer_no_enrichment_without_key(self, base_state, mock_config_no_key):
        from src.agents.agent_08_technical_writer import _build_enriched_report

        report = _build_enriched_report(base_state, mock_config_no_key)
        assert "§1" in report
