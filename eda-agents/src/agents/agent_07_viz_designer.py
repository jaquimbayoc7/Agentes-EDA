"""Agente 7 — Visualization Designer.

Rol: Diseñador de visualizaciones
Responsabilidad: Generar figuras Plotly interactivas (HTML) + PNG estáticos.
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

# ---------------------------------------------------------------------------
# Helpers Plotly
# ---------------------------------------------------------------------------

def _save_plotly_fig(fig: Any, output_dir: Path, name: str, description: str) -> list[dict]:
    """Guarda figura Plotly como HTML interactivo + PNG estático.

    Returns list of figure metadata dicts.
    """
    entries: list[dict] = []
    html_path = str(output_dir / f"{name}.html")
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=False)
    entries.append({
        "name": f"{name}.html",
        "path": html_path,
        "description": description,
        "agent": "ag7",
        "format": "html",
    })

    try:
        png_path = str(output_dir / f"{name}.png")
        fig.write_image(png_path, width=900, height=600, scale=2)
        entries.append({
            "name": f"{name}.png",
            "path": png_path,
            "description": description,
            "agent": "ag7",
            "format": "png",
        })
    except Exception:
        pass  # kaleido may not be available

    return entries


def viz_designer(state: EDAState) -> dict[str, Any]:
    """Agente 7 — Visualization Designer.

    Genera figuras Plotly interactivas + fallback PNG:
    - Distribuciones, boxplots, heatmap correlaciones
    - Pairplot top-6, target distribution
    - Feature importance bars (MI + Permutation)
    - Time series plot, missing data matrix
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
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target = state.get("target")
        plot_cols = numeric_cols[:8]  # limitar plots de distribuciones

        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # 1. Distribuciones (histogramas)
        if plot_cols:
            n_cols = len(plot_cols)
            rows_n = (n_cols + 1) // 2
            fig = make_subplots(
                rows=rows_n, cols=2,
                subplot_titles=[f"Dist: {c}" for c in plot_cols],
            )
            for i, col in enumerate(plot_cols):
                r, c = i // 2 + 1, i % 2 + 1
                fig.add_trace(
                    go.Histogram(x=df[col].dropna(), name=col, showlegend=False),
                    row=r, col=c,
                )
            fig.update_layout(
                title_text="Distribuciones de Variables Numéricas",
                height=300 * rows_n, template="plotly_white",
            )
            figures.extend(_save_plotly_fig(fig, output_dir, "distributions", "Distribuciones numéricas"))

        # 2. Boxplots
        if plot_cols:
            fig = go.Figure()
            for col in plot_cols:
                fig.add_trace(go.Box(y=df[col].dropna(), name=col))
            fig.update_layout(
                title="Boxplots — Variables Numéricas",
                height=500, template="plotly_white",
            )
            figures.extend(_save_plotly_fig(fig, output_dir, "boxplots", "Boxplots de variables numéricas"))

        # 3. Heatmap de correlaciones
        corr_cols = numeric_cols[:12]
        if len(corr_cols) >= 2:
            corr = df[corr_cols].corr(method="spearman")
            fig = px.imshow(
                corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                title="Matriz de Correlaciones (Spearman)",
                aspect="auto",
            )
            fig.update_layout(height=600, template="plotly_white")
            figures.extend(_save_plotly_fig(fig, output_dir, "corr_matrix", "Matriz de correlaciones Spearman"))

        # 4. Pairplot top-6 (scatter matrix)
        pair_cols = numeric_cols[:6]
        if len(pair_cols) >= 2:
            try:
                pair_df = df[pair_cols].dropna()
                if len(pair_df) > 500:
                    pair_df = pair_df.sample(500, random_state=42)
                fig = px.scatter_matrix(
                    pair_df, dimensions=pair_cols,
                    title="Scatter Matrix — Top Numéricas",
                )
                fig.update_traces(diagonal_visible=True, marker=dict(size=3, opacity=0.5))
                fig.update_layout(height=800, template="plotly_white")
                figures.extend(_save_plotly_fig(fig, output_dir, "pairplot", "Scatter matrix top numéricas"))
            except Exception as pair_err:
                log.warning("pairplot_failed", error=str(pair_err))

        # 5. Target distribution
        if target and target in df.columns:
            if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10:
                fig = px.histogram(
                    df, x=target, nbins=30,
                    title=f"Distribución del target: {target}",
                    color_discrete_sequence=["coral"],
                )
            else:
                vc = df[target].value_counts().reset_index()
                vc.columns = [target, "count"]
                fig = px.bar(
                    vc, x=target, y="count",
                    title=f"Distribución de clases: {target}",
                    color_discrete_sequence=["coral"],
                )
            fig.update_layout(template="plotly_white")
            figures.extend(_save_plotly_fig(fig, output_dir, "target_dist", f"Distribución del target {target}"))

        # 6. Feature Importance plots
        feat_imp = state.get("feature_importance", {})
        mi_scores = feat_imp.get("mutual_information", {})
        perm_scores = feat_imp.get("permutation_importance", {})

        if mi_scores:
            try:
                sorted_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)[:15]
                cols_mi = [x[0] for x in sorted_mi][::-1]
                vals_mi = [x[1] for x in sorted_mi][::-1]
                fig = go.Figure(go.Bar(
                    x=vals_mi, y=cols_mi, orientation="h",
                    marker_color="#3498db",
                ))
                fig.update_layout(
                    title="Feature Importance — Mutual Information",
                    xaxis_title="MI Score", template="plotly_white",
                    height=max(400, len(cols_mi) * 30),
                )
                figures.extend(_save_plotly_fig(fig, output_dir, "feat_imp_mi", "Feature Importance (MI)"))
            except Exception as mi_err:
                log.warning("mi_plot_failed", error=str(mi_err))

        if perm_scores:
            try:
                sorted_perm = sorted(perm_scores.items(), key=lambda x: x[1]["mean"], reverse=True)[:15]
                cols_p = [x[0] for x in sorted_perm][::-1]
                vals_p = [x[1]["mean"] for x in sorted_perm][::-1]
                errs_p = [x[1]["std"] for x in sorted_perm][::-1]
                fig = go.Figure(go.Bar(
                    x=vals_p, y=cols_p, orientation="h",
                    error_x=dict(type="data", array=errs_p),
                    marker_color="#e74c3c",
                ))
                fig.update_layout(
                    title="Feature Importance — Permutation Importance",
                    xaxis_title="Mean Importance", template="plotly_white",
                    height=max(400, len(cols_p) * 30),
                )
                figures.extend(_save_plotly_fig(fig, output_dir, "feat_imp_perm", "Feature Importance (Permutation)"))
            except Exception as perm_err:
                log.warning("perm_plot_failed", error=str(perm_err))

        # 7. Time series plot
        time_col = state.get("time_col")
        if time_col and time_col in df.columns and target and target in df.columns:
            try:
                ts_df = df[[time_col, target]].copy()
                ts_df[time_col] = pd.to_datetime(ts_df[time_col], errors="coerce")
                ts_df = ts_df.dropna().sort_values(time_col)
                fig = px.line(
                    ts_df, x=time_col, y=target,
                    title=f"Serie temporal: {target}",
                )
                fig.update_layout(template="plotly_white")
                figures.extend(_save_plotly_fig(fig, output_dir, "timeseries_plot", f"Serie temporal de {target}"))
            except Exception as ts_err:
                log.warning("ts_plot_failed", error=str(ts_err))

        # 8. Missing data matrix (matplotlib fallback — missingno doesn't support plotly)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import missingno as msno

            fig_path = str(output_dir / "missing_matrix.png")
            ax = msno.matrix(df, figsize=(12, 6))
            ax.figure.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(ax.figure)
            figures.append({
                "name": "missing_matrix.png",
                "path": fig_path,
                "description": "Matriz de datos faltantes",
                "agent": "ag7",
                "format": "png",
            })
        except Exception as miss_err:
            log.warning("missingno_failed", error=str(miss_err))

        # 9. VIF bar chart
        vif_all = state.get("vif_all", {})
        if not vif_all:
            # Try extracting from hallazgos_eda
            vif_all = state.get("hallazgos_eda", {}).get("vif_all", {})
        if vif_all:
            try:
                sorted_vif = sorted(vif_all.items(), key=lambda x: x[1], reverse=True)
                cols_v = [x[0] for x in sorted_vif][::-1]
                vals_v = [min(x[1], 100) for x in sorted_vif][::-1]  # Cap at 100 for readability
                colors = ["#e74c3c" if v > 10 else "#f39c12" if v > 5 else "#27ae60" for v in vals_v]
                fig = go.Figure(go.Bar(
                    x=vals_v, y=cols_v, orientation="h",
                    marker_color=colors,
                ))
                fig.add_vline(x=10, line_dash="dash", line_color="red",
                              annotation_text="Umbral VIF=10")
                fig.add_vline(x=5, line_dash="dot", line_color="orange",
                              annotation_text="VIF=5")
                fig.update_layout(
                    title="VIF — Factor de Inflación de Varianza",
                    xaxis_title="VIF", template="plotly_white",
                    height=max(400, len(cols_v) * 30),
                )
                figures.extend(_save_plotly_fig(fig, output_dir, "vif_chart", "VIF — Multicolinealidad"))
            except Exception as vif_err:
                log.warning("vif_chart_failed", error=str(vif_err))

        # 10. Normality Q-Q plots
        normality = state.get("hallazgos_eda", {}).get("normality", {})
        if normality:
            try:
                from scipy import stats as sp_stats

                norm_cols = list(normality.keys())[:6]
                n_norm = len(norm_cols)
                rows_n = (n_norm + 1) // 2
                fig = make_subplots(
                    rows=rows_n, cols=2,
                    subplot_titles=[f"Q-Q: {c}" for c in norm_cols],
                )
                for i, col in enumerate(norm_cols):
                    if col not in df.columns:
                        continue
                    clean = df[col].dropna().values
                    if len(clean) < 8:
                        continue
                    theoretical_q = sp_stats.norm.ppf(
                        np.linspace(0.01, 0.99, min(len(clean), 200))
                    )
                    sample_sorted = np.sort(
                        np.random.choice(clean, size=min(len(clean), 200), replace=False)
                    )
                    sample_q = (sample_sorted - np.mean(clean)) / (np.std(clean) + 1e-12)

                    r, c_ = i // 2 + 1, i % 2 + 1
                    fig.add_trace(
                        go.Scatter(x=theoretical_q, y=sample_q,
                                   mode="markers", name=col,
                                   marker=dict(size=4, opacity=0.6),
                                   showlegend=False),
                        row=r, col=c_,
                    )
                    # Reference line
                    fig.add_trace(
                        go.Scatter(x=[-3, 3], y=[-3, 3],
                                   mode="lines", line=dict(color="red", dash="dash"),
                                   showlegend=False),
                        row=r, col=c_,
                    )

                fig.update_layout(
                    title_text="Q-Q Plots — Test de Normalidad",
                    height=300 * rows_n, template="plotly_white",
                )
                figures.extend(_save_plotly_fig(fig, output_dir, "normality_qq", "Q-Q Plots de Normalidad"))
            except Exception as qq_err:
                log.warning("qq_plots_failed", error=str(qq_err))

        # 11. Heteroscedasticity — Residual scatter (Breusch-Pagan)
        bp_result = state.get("breusch_pagan_result")
        if bp_result and not bp_result.get("error") and target and target in df.columns:
            try:
                import statsmodels.api as sm

                features_for_bp = [c for c in numeric_cols if c != target][:10]
                clean = df[[target] + features_for_bp].dropna()
                if len(clean) > 20:
                    y = clean[target].values
                    X = sm.add_constant(clean[features_for_bp].values)
                    ols_model = sm.OLS(y, X).fit()
                    residuals = ols_model.resid
                    fitted = ols_model.fittedvalues

                    hetero_label = "Heteroscedástico" if bp_result.get("heteroscedastic") else "Homoscedástico"
                    bp_p = bp_result.get("bp_pvalue", 0)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=fitted, y=residuals,
                        mode="markers",
                        marker=dict(size=4, opacity=0.5, color="#3498db"),
                        name="Residuos",
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    fig.update_layout(
                        title=f"Residuos vs Ajustados — Breusch-Pagan (p={bp_p:.4f}, {hetero_label})",
                        xaxis_title="Valores ajustados",
                        yaxis_title="Residuos",
                        template="plotly_white", height=500,
                    )
                    figures.extend(_save_plotly_fig(
                        fig, output_dir, "heteroscedasticity",
                        f"Heteroscedasticidad — Breusch-Pagan ({hetero_label})"
                    ))
            except Exception as bp_err:
                log.warning("heteroscedasticity_plot_failed", error=str(bp_err))

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
