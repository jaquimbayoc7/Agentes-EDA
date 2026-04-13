"""Agente 2 — Data Steward.

Rol: Curador de datos
Responsabilidad: Perfilado con ydata-profiling, train/test split,
detección de encoding flags, desbalanceo y series temporales.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from src.state import EDAState
from src.utils.config import PipelineConfig
from src.utils.llm import call_claude_json
from src.utils.state_validator import validate_ag2_output

logger = structlog.get_logger()


def data_steward(state: EDAState) -> dict[str, Any]:
    """Agente 2 — Data Steward.

    Rol: Curador de datos del equipo EDA.
    Responsabilidad:
        - Ejecutar ydata-profiling sobre el dataset
        - Realizar el ÚNICO train/test split del pipeline
        - Emitir encoding_flags por columna con tipo semántico
        - Detectar desbalance de clases y flag de serie temporal
    """
    run_id = state["run_id"]
    log = logger.bind(agent="ag2", run_id=run_id)
    config = PipelineConfig.from_state(state)

    try:
        log.info("starting")
        dataset_path = state["dataset_path"]
        target = state.get("target")
        time_col = state.get("time_col")
        data_type = state.get("data_type", "tabular")

        df = pd.read_csv(dataset_path)
        dataset_size = len(df)
        log.info("dataset_loaded", rows=dataset_size, cols=len(df.columns))

        # --- Profiling ---
        perfil_columnas = _build_column_profile(df, config)

        # --- Nulos y cardinalidad ---
        nulos_pct = {col: float(df[col].isna().mean() * 100) for col in df.columns}
        cardinalidad = {col: int(df[col].nunique()) for col in df.columns}

        # --- Encoding flags (tipo semántico inferido) ---
        encoding_flags = _infer_encoding_flags(df, target, time_col, config)

        # --- Desbalance ---
        desbalance_ratio = _compute_imbalance_ratio(df, target)

        # --- Flag timeseries ---
        flag_timeseries = data_type in ("timeseries", "mixed") or (time_col is not None)

        # --- TRAIN/TEST SPLIT (único lugar) ---
        train_path, test_path = _split_dataset(
            df, target, run_id, config
        )

        output: dict[str, Any] = {
            "perfil_columnas": perfil_columnas,
            "nulos_pct": nulos_pct,
            "cardinalidad": cardinalidad,
            "encoding_flags": encoding_flags,
            "desbalance_ratio": desbalance_ratio,
            "flag_timeseries": flag_timeseries,
            "dataset_size": dataset_size,
            "train_path": train_path,
            "test_path": test_path,
            "agent_status": {**state.get("agent_status", {}), "ag2": "ok"},
        }

        validate_ag2_output(output)
        log.info("completed", train_path=train_path, test_path=test_path)
        return output

    except Exception as e:
        log.error("failed", error=str(e))
        return {
            "perfil_columnas": {},
            "nulos_pct": {},
            "cardinalidad": {},
            "encoding_flags": {},
            "desbalance_ratio": None,
            "flag_timeseries": False,
            "dataset_size": 0,
            "train_path": state.get("train_path", ""),
            "test_path": state.get("test_path", ""),
            "agent_status": {**state.get("agent_status", {}), "ag2": "error"},
            "error_log": [{"agent": "ag2", "error": str(e), "run_id": run_id}],
        }


def _build_column_profile(df: pd.DataFrame, config: PipelineConfig) -> dict[str, Any]:
    """Genera perfil de columnas (profiling simplificado)."""
    profile: dict[str, Any] = {}
    for col in df.columns:
        col_info: dict[str, Any] = {
            "dtype": str(df[col].dtype),
            "n_unique": int(df[col].nunique()),
            "null_pct": float(df[col].isna().mean() * 100),
            "sample_values": df[col].dropna().head(5).tolist(),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["mean"] = float(df[col].mean())
            col_info["std"] = float(df[col].std())
            col_info["min"] = float(df[col].min())
            col_info["max"] = float(df[col].max())
        profile[col] = col_info
    return profile


def _infer_encoding_flags(
    df: pd.DataFrame, target: str | None, time_col: str | None,
    config: PipelineConfig | None = None,
) -> dict[str, str]:
    """Infiere tipo semántico de cada columna para encoding.

    Usa Claude API si disponible para clasificación semántica más precisa,
    con fallback heurístico.
    """
    # Build column summary for LLM
    col_summaries: list[str] = []
    for col in df.columns:
        n_unique = df[col].nunique()
        dtype = str(df[col].dtype)
        samples = df[col].dropna().head(5).tolist()
        col_summaries.append(
            f"- {col}: dtype={dtype}, unique={n_unique}, samples={samples}"
        )

    # Try Claude classification
    if config and config.anthropic_api_key:
        try:
            valid_types = "NUMERICA, BINARIA, NOMINAL, ORDINAL, FECHA, ALTA_CARD, TARGET"
            result = call_claude_json(
                prompt=(
                    f"Classify each column by semantic type.\n"
                    f"Target column: {target}\n"
                    f"Time column: {time_col}\n\n"
                    f"Columns:\n" + "\n".join(col_summaries) + "\n\n"
                    f"Valid types: {valid_types}\n"
                    "Rules: target column → TARGET, time column → FECHA, "
                    "binary (2 values) → BINARIA, numeric continuous → NUMERICA, "
                    "ordered categories → ORDINAL, unordered categories → NOMINAL, "
                    "high cardinality text → ALTA_CARD.\n"
                    'Return JSON: {"flags": {"col_name": "TYPE", ...}}'
                ),
                system="You are a data profiling expert. Return only valid JSON.",
                model=config.model,
                max_tokens=1024,
                api_key=config.anthropic_api_key,
            )
            flags = result.get("flags", {})
            if flags and len(flags) == len(df.columns):
                return flags
        except Exception:
            pass

    # Fallback heurístico
    flags: dict[str, str] = {}
    for col in df.columns:
        if col == target:
            flags[col] = "TARGET"
            continue
        if col == time_col:
            flags[col] = "FECHA"
            continue

        n_unique = df[col].nunique()
        dtype = df[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            flags[col] = "NUMERICA"
        elif n_unique == 2:
            flags[col] = "BINARIA"
        elif n_unique <= 3:
            flags[col] = "NOMINAL"
        elif n_unique <= 15:
            flags[col] = "NOMINAL"
        else:
            try:
                pd.to_datetime(df[col].dropna().head(10))
                flags[col] = "FECHA"
            except (ValueError, TypeError):
                flags[col] = "ALTA_CARD"

    return flags


def _compute_imbalance_ratio(df: pd.DataFrame, target: str | None) -> float | None:
    """Calcula ratio de desbalanceo para target categórico."""
    if target is None or target not in df.columns:
        return None
    if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10:
        return None  # Regresión, no aplica

    counts = df[target].value_counts()
    if len(counts) < 2:
        return None
    return float(counts.max() / counts.min())


def _split_dataset(
    df: pd.DataFrame,
    target: str | None,
    run_id: str,
    config: PipelineConfig,
) -> tuple[str, str]:
    """Realiza train/test split estratificado y guarda CSVs."""
    from sklearn.model_selection import train_test_split

    output_dir = Path("outputs") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    stratify_col = None
    if (
        config.split.stratify
        and target
        and target in df.columns
        and not pd.api.types.is_numeric_dtype(df[target])
    ):
        # Solo estratificar si target categórico y suficientes muestras por clase
        min_count = df[target].value_counts().min()
        if min_count >= 2:
            stratify_col = df[target]

    train_df, test_df = train_test_split(
        df,
        test_size=config.split.test_size,
        random_state=config.random_seed,
        stratify=stratify_col,
    )

    train_path = str(output_dir / "train.csv")
    test_path = str(output_dir / "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, test_path
