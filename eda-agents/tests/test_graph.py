"""Tests para src/graph.py — Grafo LangGraph y re_encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.graph import (
    _abort_node,
    _route_after_engineer,
    build_graph,
    re_encoder,
)


# ---------------------------------------------------------------------------
# Tests de routing condicional
# ---------------------------------------------------------------------------


class TestRouteAfterEngineer:
    """Tests para la función de routing condicional post-data_engineer."""

    def test_timeseries_routes_both(self) -> None:
        state = {"flag_timeseries": True}
        result = _route_after_engineer(state)
        assert "statistician" in result
        assert "ts_analyst" in result

    def test_no_timeseries_routes_only_statistician(self) -> None:
        state = {"flag_timeseries": False}
        result = _route_after_engineer(state)
        assert result == ["statistician"]

    def test_missing_flag_defaults_no_ts(self) -> None:
        state = {}
        result = _route_after_engineer(state)
        assert result == ["statistician"]


# ---------------------------------------------------------------------------
# Tests del nodo re_encoder
# ---------------------------------------------------------------------------


class TestReEncoder:
    """Tests para el nodo re_encoder (Python puro)."""

    def test_tree_family_keeps_label(self, tmp_path: Path) -> None:
        """Si model_family == 'tree', mantiene LabelEncoder."""
        # Preparar datasets
        df_train = pd.DataFrame({"col_a": [0, 1, 2, 0, 1], "target": [1, 0, 1, 0, 1]})
        df_test = pd.DataFrame({"col_a": [1, 0, 2], "target": [0, 1, 0]})
        train_path = str(tmp_path / "train_prov.csv")
        test_path = str(tmp_path / "test_proc.csv")
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)

        # Crear output dir
        run_id = "test_re"
        out_dir = Path("outputs") / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "run_id": run_id,
            "model_family": "tree",
            "encoding_log": {"col_a": {"encoding": "label", "moment": 1}},
            "dataset_train_provisional": train_path,
            "dataset_test_procesado": test_path,
        }

        result = re_encoder(state)

        assert result["encoding_log"]["col_a"]["encoding_final"] == "label"
        assert Path(result["dataset_train_final"]).exists()
        assert Path(result["dataset_test_final"]).exists()

    def test_linear_family_converts_to_frequency(self, tmp_path: Path) -> None:
        """Si model_family == 'linear', convierte LabelEncoder a FrequencyEncoder."""
        df_train = pd.DataFrame({"col_a": [0, 1, 0, 0, 1], "target": [1, 0, 1, 0, 1]})
        df_test = pd.DataFrame({"col_a": [1, 0, 0], "target": [0, 1, 0]})
        train_path = str(tmp_path / "train_prov.csv")
        test_path = str(tmp_path / "test_proc.csv")
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)

        run_id = "test_re_lin"
        out_dir = Path("outputs") / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "run_id": run_id,
            "model_family": "linear",
            "encoding_log": {"col_a": {"encoding": "label", "moment": 1}},
            "dataset_train_provisional": train_path,
            "dataset_test_procesado": test_path,
        }

        result = re_encoder(state)

        assert result["encoding_log"]["col_a"]["encoding_final"] == "frequency"
        assert result["encoding_log"]["col_a"]["moment"] == 2

    def test_empty_datasets_fallback(self) -> None:
        """Si no hay datasets provisionales, retorna sin error."""
        state = {
            "run_id": "test_empty",
            "model_family": "tree",
            "encoding_log": {},
            "dataset_train_provisional": "",
            "dataset_test_procesado": "",
        }

        result = re_encoder(state)
        assert result["dataset_train_final"] == ""
        assert result["dataset_test_final"] == ""


# ---------------------------------------------------------------------------
# Tests del nodo abort
# ---------------------------------------------------------------------------


class TestAbortNode:
    """Tests para el nodo de aborto."""

    def test_abort_marks_status(self) -> None:
        state = {
            "run_id": "test_abort",
            "agent_status": {"ag2": "error"},
        }
        result = _abort_node(state)
        assert result["agent_status"]["orchestrator"] == "aborted"
        assert len(result["error_log"]) == 1


# ---------------------------------------------------------------------------
# Tests de construcción del grafo
# ---------------------------------------------------------------------------


class TestBuildGraph:
    """Tests para la construcción del grafo LangGraph."""

    def test_graph_compiles_without_checkpointer(self) -> None:
        """El grafo se compila correctamente sin checkpointer."""
        graph = build_graph(checkpointer=None)
        assert graph is not None

    def test_graph_has_all_nodes(self) -> None:
        """El grafo contiene todos los nodos esperados."""
        graph = build_graph(checkpointer=None)
        # LangGraph compiled graph has a .get_graph() method
        graph_obj = graph.get_graph()
        node_ids = set(graph_obj.nodes.keys())
        expected = {
            "research_lead", "data_steward", "data_engineer",
            "statistician", "ts_analyst", "ml_strategist",
            "re_encoder", "viz_designer", "technical_writer",
            "abort", "__start__", "__end__",
        }
        assert expected.issubset(node_ids), f"Missing nodes: {expected - node_ids}"

    def test_graph_end_to_end_tabular(self, base_state: dict, tmp_path: Path) -> None:
        """Ejecución end-to-end con datos tabulares (sin timeseries)."""
        graph = build_graph(checkpointer=None)

        # Ajustar state para usar tmp_path como dataset
        state = {**base_state, "flag_timeseries": False}

        result = graph.invoke(state)

        assert result is not None
        assert "agent_status" in result
        # Al menos research_lead y data_steward deben haber corrido
        agent_st = result.get("agent_status", {})
        assert "ag1" in agent_st, f"ag1 missing from status: {agent_st}"
        assert "ag2" in agent_st, f"ag2 missing from status: {agent_st}"

    def test_graph_end_to_end_timeseries(self, base_state: dict) -> None:
        """Ejecución end-to-end con flag_timeseries=True."""
        graph = build_graph(checkpointer=None)

        state = {**base_state, "flag_timeseries": True, "time_col": "fecha"}

        result = graph.invoke(state)

        assert result is not None
        agent_st = result.get("agent_status", {})
        # ts_analyst debe haber corrido
        assert "ag5" in agent_st, f"ag5 missing from status: {agent_st}"
