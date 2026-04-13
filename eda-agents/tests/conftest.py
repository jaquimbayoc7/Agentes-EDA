"""Fixtures compartidos para todos los tests del pipeline EDA."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame de prueba con 100 filas y columnas de distintos tipos."""
    np.random.seed(42)
    return pd.DataFrame({
        "feature_num": np.random.randn(100),
        "feature_cat": np.random.choice(["A", "B", "C"], 100),
        "feature_ord": np.random.choice(["bajo", "medio", "alto"], 100),
        "fecha": pd.date_range("2020-01-01", periods=100, freq="D").astype(str),
        "target": np.random.randn(100),
    })


@pytest.fixture
def base_state(sample_df: pd.DataFrame, tmp_path) -> dict:
    """Estado base para tests de agentes."""
    train_path = str(tmp_path / "train.csv")
    test_path = str(tmp_path / "test.csv")
    sample_df[:80].to_csv(train_path, index=False)
    sample_df[80:].to_csv(test_path, index=False)

    return {
        "research_question": "¿Qué factores predicen el target?",
        "dataset_path": train_path,
        "data_type": "tabular",
        "target": "target",
        "time_col": None,
        "context": "Test context for EDA pipeline",
        "run_id": "test001",
        "random_seed": 42,
        "config": {},
        "train_path": train_path,
        "test_path": test_path,
        "refs": [],
        "hipotesis": None,
        "tarea_sugerida": None,
        "search_equations": [],
        "perfil_columnas": {},
        "nulos_pct": {},
        "cardinalidad": {},
        "encoding_flags": {},
        "desbalance_ratio": None,
        "flag_timeseries": False,
        "dataset_size": 100,
        "encoding_log": {},
        "features_nuevas": [],
        "balanceo_log": {},
        "dataset_train_provisional": "",
        "dataset_test_procesado": "",
        "dataset_train_final": "",
        "dataset_test_final": "",
        "hallazgos_eda": {},
        "breusch_pagan_result": None,
        "modelo_correccion_heterosc": None,
        "vif_flags": [],
        "modelo_ts": None,
        "params_pdq": None,
        "diagnostico_residuos_ts": None,
        "modelos_recomendados": [],
        "model_family": None,
        "hyperparams_technique": None,
        "metrica_principal": None,
        "advertencias": [],
        "figures": [],
        "agent_status": {},
        "error_log": [],
    }
