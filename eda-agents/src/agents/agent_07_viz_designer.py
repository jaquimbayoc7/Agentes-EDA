"""Agente 7 — Visualization Designer.

Rol: Diseñador de visualizaciones
Responsabilidad: Generar figuras PNG y Plotly según tipo de tarea.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from src.state import EDAState
from src.utils.config import PipelineConfig
from src.utils.state_validator import validate_ag7_output

logger = structlog.get_logger()


def viz_designer(state: EDAState) -> dict[str, Any]:
    """Agente 7 — Visualization Designer.

    Rol: Diseñador de visualizaciones del equipo EDA.
    Responsabilidad:
        - Generar figuras según tipo de tarea
        - Distribuciones, heatmap, missingno, pairplot top-6
        - 150 DPI PNG + versión interactiva Plotly HTML
        - Nombres semánticos para todas las figuras
    """
    run_id = state["run_id"]
    log = logger.bind(agent="ag7", run_id=run_id)
    config = PipelineConfig.from_state(state)

    try:
        log.info("starting")
        tarea = state.get("tarea_sugerida", "classification")
        train_final = state.get("dataset_train_final", "")

        if not train_final:
            train_final = state.get("dataset_train_provisional", "")

        figures: list[dict[str, Any]] = []
        output_dir = Path("outputs") / run_id / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)

        if not train_final:
            log.warning("no_dataset_for_visualization")
            return {
                "figures": [],
                "agent_status": {**state.get("agent_status", {}), "ag7": "fallback"},
            }

        df = pd.read_csv(train_final)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:6]

        # --- Generar figuras ---
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        target = state.get("target")

        # 1. Distribuciones
        for col in numeric_cols:
            fig_path = str(output_dir / f"dist_{col}.png")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df[col].dropna(), bins=30, edgecolor="black", alpha=0.7)
            ax.set_title(f"Distribución: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frecuencia")
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            figures.append({
                "name": f"dist_{col}.png",
                "path": fig_path,
                "description": f"Distribución de {col}",
                "agent": "ag7",
            })

        # 2. Boxplots
        for col in numeric_cols:
            fig_path = str(output_dir / f"box_{col}.png")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot(df[col].dropna(), orientation="vertical", patch_artist=True)
            ax.set_title(f"Boxplot: {col}")
            ax.set_ylabel(col)
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            figures.append({
                "name": f"box_{col}.png",
                "path": fig_path,
                "description": f"Boxplot de {col}",
                "agent": "ag7",
            })

        # 3. Heatmap de correlaciones
        if len(numeric_cols) >= 2:
            fig_path = str(output_dir / "corr_matrix.png")
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Matriz de Correlaciones")
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            figures.append({
                "name": "corr_matrix.png",
                "path": fig_path,
                "description": "Matriz de correlaciones Pearson",
                "agent": "ag7",
            })

        # 4. Pairplot top-6
        if len(numeric_cols) >= 2:
            try:
                pair_cols = numeric_cols[:6]
                fig_path = str(output_dir / "pairplot.png")
                pair_df = df[pair_cols].dropna()
                if len(pair_df) > 500:
                    pair_df = pair_df.sample(500, random_state=42)
                g = sns.pairplot(pair_df, diag_kind="kde", plot_kws={"alpha": 0.5, "s": 15})
                g.figure.suptitle("Pairplot Top-6 numéricas", y=1.02)
                g.figure.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close(g.figure)
                figures.append({
                    "name": "pairplot.png",
                    "path": fig_path,
                    "description": "Pairplot de las 6 columnas numéricas principales",
                    "agent": "ag7",
                })
            except Exception as pair_err:
                log.warning("pairplot_failed", error=str(pair_err))

        # 5. Target distribution
        if target and target in df.columns:
            fig_path = str(output_dir / "target_dist.png")
            fig, ax = plt.subplots(figsize=(8, 5))
            if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10:
                ax.hist(df[target].dropna(), bins=30, edgecolor="black", alpha=0.7, color="coral")
                ax.set_title(f"Distribución del target: {target}")
            else:
                df[target].value_counts().plot(kind="bar", ax=ax, color="coral", edgecolor="black")
                ax.set_title(f"Distribución de clases: {target}")
            ax.set_xlabel(target)
            ax.set_ylabel("Frecuencia")
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            figures.append({
                "name": "target_dist.png",
                "path": fig_path,
                "description": f"Distribución del target {target}",
                "agent": "ag7",
            })

        # 6. Time series plot (si aplica)
        time_col = state.get("time_col")
        if time_col and time_col in df.columns and target and target in df.columns:
            try:
                ts_df = df[[time_col, target]].copy()
                ts_df[time_col] = pd.to_datetime(ts_df[time_col], errors="coerce")
                ts_df = ts_df.dropna().sort_values(time_col)
                fig_path = str(output_dir / "timeseries_plot.png")
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(ts_df[time_col], ts_df[target], linewidth=0.8)
                ax.set_title(f"Serie temporal: {target}")
                ax.set_xlabel(time_col)
                ax.set_ylabel(target)
                fig.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                figures.append({
                    "name": "timeseries_plot.png",
                    "path": fig_path,
                    "description": f"Serie temporal de {target}",
                    "agent": "ag7",
                })
            except Exception as ts_err:
                log.warning("ts_plot_failed", error=str(ts_err))

        # 7. Missing data (missingno)
        try:
            import missingno as msno

            fig_path = str(output_dir / "missing_matrix.png")
            fig = msno.matrix(df, figsize=(12, 6))
            fig.figure.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig.figure)
            figures.append({
                "name": "missing_matrix.png",
                "path": fig_path,
                "description": "Matriz de datos faltantes",
                "agent": "ag7",
            })
        except Exception as miss_err:
            log.warning("missingno_failed", error=str(miss_err))

        log.info("figures_generated", n_figures=len(figures))

        output: dict[str, Any] = {
            "figures": figures,
            "agent_status": {**state.get("agent_status", {}), "ag7": "ok"},
        }

        validate_ag7_output(output)
        log.info("completed")
        return output

    except Exception as e:
        log.error("failed", error=str(e))
        return {
            "figures": [],
            "agent_status": {**state.get("agent_status", {}), "ag7": "error"},
            "error_log": [{"agent": "ag7", "error": str(e), "run_id": run_id}],
        }
