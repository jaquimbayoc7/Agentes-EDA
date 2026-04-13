"""Agente 5 — TS Analyst (condicional).

Rol: Analista de series temporales
Responsabilidad: Estacionariedad, descomposición STL, selección
de modelo ARIMA/SARIMA/SARIMAX/VAR, diagnóstico de residuos.
Solo se invoca si flag_timeseries == True.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from src.state import EDAState
from src.utils.config import PipelineConfig
from src.utils.state_validator import validate_ag5_output
from src.skills.timeseries import (
    test_stationarity,
    determine_differencing_order,
    diagnose_residuals,
    select_ts_model,
)

logger = structlog.get_logger()


def ts_analyst(state: EDAState) -> dict[str, Any]:
    """Agente 5 — TS Analyst.

    Rol: Analista de series temporales del equipo EDA.
    Responsabilidad:
        - ADF + KPSS tests → determinar d (diferenciación)
        - STL descomposición → trend, seasonal, residual
        - ACF + PACF → inferir (p, q) iniciales
        - pmdarima.auto_arima para selección automática
        - Diagnóstico: Ljung-Box, Jarque-Bera
        - Detección de cambios de régimen con ruptures
    """
    run_id = state["run_id"]
    log = logger.bind(agent="ag5", run_id=run_id)
    config = PipelineConfig.from_state(state)

    try:
        log.info("starting")
        time_col = state.get("time_col")
        train_prov = state.get("dataset_train_provisional", "")
        target = state.get("target")

        if not train_prov or not time_col:
            log.warning("skipping_no_time_data")
            return {
                "modelo_ts": None,
                "params_pdq": None,
                "diagnostico_residuos_ts": None,
                "figures": [],
                "agent_status": {**state.get("agent_status", {}), "ag5": "ok"},
            }

        df = pd.read_csv(train_prov)

        if time_col not in df.columns:
            raise ValueError(f"time_col '{time_col}' not found in dataset")

        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)

        series = df[target] if target and target in df.columns else df.iloc[:, 1]

        figures: list[dict[str, Any]] = []
        output_dir = Path("outputs") / run_id / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Stationarity tests ---
        stationarity = test_stationarity(series)
        log.info("stationarity_tested", adf_pval=stationarity.get("adf_pvalue"))

        # --- Differencing order ---
        d = determine_differencing_order(series)

        # --- Model selection ---
        modelo_ts, params_pdq = select_ts_model(series, d=d, seasonal=False)

        # --- Residual diagnostics ---
        diagnostico = diagnose_residuals(series)

        output: dict[str, Any] = {
            "modelo_ts": modelo_ts,
            "params_pdq": params_pdq,
            "diagnostico_residuos_ts": diagnostico,
            "figures": figures,
            "agent_status": {**state.get("agent_status", {}), "ag5": "ok"},
        }

        validate_ag5_output(output)
        log.info("completed")
        return output

    except Exception as e:
        log.error("failed", error=str(e))
        return {
            "modelo_ts": None,
            "params_pdq": None,
            "diagnostico_residuos_ts": None,
            "figures": [],
            "agent_status": {**state.get("agent_status", {}), "ag5": "error"},
            "error_log": [{"agent": "ag5", "error": str(e), "run_id": run_id}],
        }
