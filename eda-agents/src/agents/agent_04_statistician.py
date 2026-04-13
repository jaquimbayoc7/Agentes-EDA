"""Agente 4 — Statistician.

Rol: Estadístico senior
Responsabilidad: EDA tabular — distribuciones, correlaciones, outliers,
normalidad, VIF, Breusch-Pagan, corrección heteroscedasticidad.
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
from src.utils.state_validator import validate_ag4_output
from src.skills.statistical_tests import (
    compute_correlations,
    detect_outliers_iqr,
    test_normality,
    compute_vif,
    breusch_pagan_test,
    suggest_heteroscedasticity_correction,
)
from src.skills.feature_importance import (
    compute_mutual_information,
    compute_permutation_importance,
    select_top_features,
)

logger = structlog.get_logger()


def statistician(state: EDAState) -> dict[str, Any]:
    """Agente 4 — Statistician.

    Rol: Estadístico senior del equipo EDA.
    Responsabilidad:
        - Distribuciones: histogramas, KDE, Q-Q plots
        - Correlaciones: Pearson, Spearman, Kendall
        - Outliers: IQR + Isolation Forest
        - Normalidad: Shapiro-Wilk / Anderson-Darling
        - VIF para multicolinealidad
        - Breusch-Pagan + corrección heteroscedasticidad (si regresión)
    """
    run_id = state["run_id"]
    log = logger.bind(agent="ag4", run_id=run_id)
    config = PipelineConfig.from_state(state)

    try:
        log.info("starting")
        train_prov = state.get("dataset_train_provisional", "")
        target = state.get("target")
        tarea = state.get("tarea_sugerida", "classification")

        if not train_prov:
            raise ValueError("dataset_train_provisional is empty")

        df = pd.read_csv(train_prov)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target and target in numeric_cols:
            features = [c for c in numeric_cols if c != target]
        else:
            features = numeric_cols

        # Features incluyendo target para correlación target-aware
        features_with_target = features + ([target] if target and target in numeric_cols else [])

        hallazgos: dict[str, Any] = {}
        figures: list[dict[str, Any]] = []
        output_dir = Path("outputs") / run_id / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Correlaciones (Spearman, target-aware) ---
        if features_with_target:
            hallazgos["correlations"] = compute_correlations(df, features_with_target)
            log.info("correlations_computed", method="spearman", n_cols=len(features_with_target))

        # --- Outliers (IQR) ---
        hallazgos["outliers"] = detect_outliers_iqr(df, features)

        # --- Normalidad ---
        hallazgos["normality"] = test_normality(df, features)

        # --- VIF ---
        vif_flags, all_vif = compute_vif(df, features, threshold=config.vif_threshold)
        hallazgos["vif_summary"] = {"n_flagged": len(vif_flags)}
        hallazgos["vif_all"] = all_vif

        # --- Feature Importance ---
        feat_imp: dict[str, Any] = {}
        top_features: list[str] = []
        if target and target in numeric_cols and features:
            try:
                mi_scores = compute_mutual_information(
                    df, features, target, task=tarea, random_state=config.random_seed,
                )
                feat_imp["mutual_information"] = mi_scores
                log.info("mutual_information_computed", n_features=len(mi_scores))
            except Exception as mi_err:
                log.warning("mutual_information_failed", error=str(mi_err))
                mi_scores = {}

            try:
                perm_scores = compute_permutation_importance(
                    df, features, target, task=tarea, random_state=config.random_seed,
                )
                feat_imp["permutation_importance"] = perm_scores
                log.info("permutation_importance_computed", n_features=len(perm_scores))
            except Exception as pi_err:
                log.warning("permutation_importance_failed", error=str(pi_err))
                perm_scores = {}

            top_features = select_top_features(mi_scores, perm_scores, top_k=10)
            feat_imp["top_features"] = top_features

        hallazgos["feature_importance"] = feat_imp

        # --- Breusch-Pagan (solo regresión) ---
        bp_result: dict[str, Any] | None = None
        modelo_correccion: str | None = None

        if tarea == "regression" and target and target in df.columns and len(features) > 0:
            try:
                bp_result = breusch_pagan_test(df, target, features, alpha=config.bp_pvalue)
                modelo_correccion = suggest_heteroscedasticity_correction(bp_result, vif_flags)
            except Exception as bp_err:
                log.warning("breusch_pagan_failed", error=str(bp_err))

        # --- Interpretación con Claude ---
        hallazgos["interpretation"] = _interpret_findings(
            hallazgos, vif_flags, bp_result, config
        )

        output: dict[str, Any] = {
            "hallazgos_eda": hallazgos,
            "breusch_pagan_result": bp_result,
            "modelo_correccion_heterosc": modelo_correccion,
            "vif_flags": vif_flags,
            "vif_all": all_vif,
            "feature_importance": feat_imp,
            "figures": figures,
            "agent_status": {**state.get("agent_status", {}), "ag4": "ok"},
        }

        validate_ag4_output(output)
        log.info("completed", n_hallazgos=len(hallazgos))
        return output

    except Exception as e:
        log.error("failed", error=str(e))
        return {
            "hallazgos_eda": {},
            "breusch_pagan_result": None,
            "modelo_correccion_heterosc": None,
            "vif_flags": [],
            "vif_all": {},
            "feature_importance": {},
            "figures": [],
            "agent_status": {**state.get("agent_status", {}), "ag4": "error"},
            "error_log": [{"agent": "ag4", "error": str(e), "run_id": run_id}],
        }


def _interpret_findings(
    hallazgos: dict[str, Any],
    vif_flags: list,
    bp_result: dict | None,
    config: PipelineConfig,
) -> str:
    """Interpreta hallazgos estadísticos en lenguaje natural usando Claude."""
    if not config.anthropic_api_key:
        return ""

    try:
        import json as _json

        result = call_claude_json(
            prompt=(
                "Interpret these EDA statistical findings in Spanish. "
                "Be concise (max 300 words).\n\n"
                f"Correlations: {_json.dumps(hallazgos.get('correlations', {}), default=str)[:500]}\n"
                f"Outliers: {_json.dumps(hallazgos.get('outliers', {}), default=str)[:500]}\n"
                f"Normality: {_json.dumps(hallazgos.get('normality', {}), default=str)[:500]}\n"
                f"VIF flags: {vif_flags}\n"
                f"Breusch-Pagan: {_json.dumps(bp_result, default=str) if bp_result else 'N/A'}\n\n"
                'Return JSON: {"interpretation": "..."}'
            ),
            system="You are a senior statistician. Interpret findings clearly in Spanish.",
            model=config.model,
            max_tokens=1024,
            api_key=config.anthropic_api_key,
        )
        return result.get("interpretation", "")
    except Exception:
        return ""
