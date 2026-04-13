"""Agente 6 — ML Strategist.

Rol: Estratega de Machine Learning
Responsabilidad: Leer hallazgos del EDA y decidir modelos, métricas,
hiperparametrización y model_family.
"""

from __future__ import annotations

from typing import Any

import structlog

from src.state import EDAState
from src.utils.config import PipelineConfig
from src.utils.llm import call_claude_json
from src.utils.state_validator import validate_ag6_output

logger = structlog.get_logger()


def ml_strategist(state: EDAState) -> dict[str, Any]:
    """Agente 6 — ML Strategist.

    Rol: Estratega de Machine Learning del equipo EDA.
    Responsabilidad:
        - Leer hallazgos del estado (EDA, VIF, heteroscedasticidad, TS)
        - Recomendar modelos según señales estadísticas
        - Seleccionar métricas (NUNCA solo accuracy)
        - Decidir técnica de hiperparametrización
        - Emitir model_family ("tree" | "linear")
    """
    run_id = state["run_id"]
    log = logger.bind(agent="ag6", run_id=run_id)
    config = PipelineConfig.from_state(state)

    try:
        log.info("starting")
        tarea = state.get("tarea_sugerida", "classification")
        hallazgos = state.get("hallazgos_eda", {})
        vif_flags = state.get("vif_flags", [])
        bp_result = state.get("breusch_pagan_result")
        desbalance = state.get("desbalance_ratio")
        modelo_ts = state.get("modelo_ts")
        dataset_size = state.get("dataset_size", 100)

        modelos: list[dict[str, Any]] = []
        advertencias: list[str] = []
        model_family: str = "tree"  # default
        metrica: str | None = None
        hp_technique: str | None = None

        # Try Claude-based strategy first
        claude_ok = False
        if config.anthropic_api_key:
            try:
                import json as _json

                result = call_claude_json(
                    prompt=(
                        f"Task type: {tarea}\n"
                        f"Dataset size: {dataset_size} rows\n"
                        f"EDA findings: {_json.dumps(hallazgos, default=str)[:800]}\n"
                        f"VIF flagged columns: {vif_flags}\n"
                        f"Breusch-Pagan: {_json.dumps(bp_result, default=str) if bp_result else 'N/A'}\n"
                        f"Imbalance ratio: {desbalance}\n"
                        f"TS model: {modelo_ts}\n\n"
                        "Recommend ML models and strategy. Consider:\n"
                        "- Statistical signals (VIF, heteroscedasticity, normality)\n"
                        "- Dataset size appropriate methods\n"
                        "- Never use only accuracy for classification\n\n"
                        'Return JSON: {\n'
                        '  "models": [{"name": "...", "reason": "..."}],\n'
                        '  "model_family": "tree" or "linear",\n'
                        '  "metric": "...",\n'
                        '  "hp_technique": "GridSearchCV" or "RandomizedSearchCV" or "Optuna",\n'
                        '  "warnings": ["..."]\n'
                        '}'
                    ),
                    system="You are a senior ML engineer. Return only valid JSON.",
                    model=config.model,
                    max_tokens=1024,
                    api_key=config.anthropic_api_key,
                )
                if "models" in result and result["models"]:
                    modelos = result["models"]
                    model_family = result.get("model_family", "tree")
                    metrica = result.get("metric")
                    hp_technique = result.get("hp_technique")
                    advertencias = result.get("warnings", [])
                    claude_ok = True
            except Exception:
                pass

        if not claude_ok:
            if tarea == "regression":
                modelos, model_family, metrica = _recommend_regression(
                    hallazgos, vif_flags, bp_result, dataset_size
                )
            elif tarea == "classification":
                modelos, model_family, metrica = _recommend_classification(
                    hallazgos, desbalance, dataset_size
                )
            elif tarea == "forecasting" and modelo_ts:
                modelos = _recommend_forecasting(modelo_ts, dataset_size)
                model_family = "linear"
                metrica = "MAE"
            else:
                modelos = [{"name": "XGBClassifier", "reason": "default_fallback"}]
                metrica = "f1_macro"

        # --- Técnica de hiperparametrización (fallback si Claude no la dio) ---
        if not hp_technique:
            hp_technique = _select_hp_technique(modelos, dataset_size, config)

        # --- Advertencias ---
        if vif_flags:
            advertencias.append(f"VIF > {config.vif_threshold} en {len(vif_flags)} columnas")
        if bp_result and bp_result.get("heteroscedastic"):
            advertencias.append("Heteroscedasticidad detectada")
        if desbalance and desbalance > config.imbalance_thresholds.hybrid:
            advertencias.append(f"Desbalanceo alto: ratio {desbalance:.1f}")

        output: dict[str, Any] = {
            "modelos_recomendados": modelos,
            "model_family": model_family,
            "hyperparams_technique": hp_technique,
            "metrica_principal": metrica,
            "advertencias": advertencias,
            "agent_status": {**state.get("agent_status", {}), "ag6": "ok"},
        }

        validate_ag6_output(output)
        log.info("completed", model_family=model_family, n_models=len(modelos))
        return output

    except Exception as e:
        log.error("failed", error=str(e))
        return {
            "modelos_recomendados": [{"name": "XGBClassifier", "reason": "error_fallback"}],
            "model_family": "tree",
            "hyperparams_technique": None,
            "metrica_principal": None,
            "advertencias": [f"ML Strategist error: {e}"],
            "agent_status": {**state.get("agent_status", {}), "ag6": "error"},
            "error_log": [{"agent": "ag6", "error": str(e), "run_id": run_id}],
        }


def _recommend_regression(
    hallazgos: dict, vif_flags: list, bp_result: dict | None, n: int
) -> tuple[list[dict[str, Any]], str, str]:
    """Recomienda modelos para regresión según señales del EDA (5-7 modelos + ensembles)."""
    modelos: list[dict[str, Any]] = []
    model_family = "tree"

    has_high_vif = len(vif_flags) > 0
    has_heterosc = bp_result is not None and bp_result.get("heteroscedastic", False)

    # --- Modelos base ---
    if has_high_vif:
        modelos.append({"name": "Ridge", "reason": "VIF alto → regularización L2"})
        modelos.append({"name": "Lasso", "reason": "VIF alto → selección L1"})
        modelos.append({"name": "ElasticNet", "reason": "VIF alto → combinación L1+L2"})
        model_family = "linear"
    if has_heterosc:
        modelos.append({"name": "SVR", "reason": "Heteroscedasticidad → kernel RBF robusto"})
        if model_family != "linear":
            model_family = "linear"

    if n >= 200:
        modelos.append({"name": "XGBRegressor", "reason": "N suficiente + potencial no-lineal"})
        modelos.append({"name": "LGBMRegressor", "reason": "Alternativa eficiente a XGB"})
        modelos.append({"name": "RandomForestRegressor", "reason": "Baseline robusto tree-based"})
        modelos.append({"name": "GradientBoostingRegressor", "reason": "Boosting interpretable sklearn"})
    else:
        if not any(m["name"] in ("Ridge", "Lasso") for m in modelos):
            modelos.append({"name": "Ridge", "reason": "N bajo → regularización simple"})
        modelos.append({"name": "RandomForestRegressor", "reason": "Robusto con N bajo"})

    # --- Ensembles ---
    if n >= 100:
        modelos.append({"name": "StackingRegressor", "reason": "Ensemble: combina predicciones de base models"})
        modelos.append({"name": "VotingRegressor", "reason": "Ensemble: promedio de múltiples modelos"})
    if n >= 300:
        modelos.append({"name": "AdaBoostRegressor", "reason": "Boosting adaptativo"})

    if not modelos:
        modelos.append({"name": "XGBRegressor", "reason": "default"})

    return modelos, model_family, "RMSE"


def _recommend_classification(
    hallazgos: dict, desbalance: float | None, n: int
) -> tuple[list[dict[str, Any]], str, str]:
    """Recomienda modelos para clasificación según señales del EDA (5-7 modelos + ensembles)."""
    modelos: list[dict[str, Any]] = []
    model_family = "tree"

    # --- Modelos base ---
    if desbalance and desbalance > 3:
        modelos.append({
            "name": "XGBClassifier",
            "reason": f"Desbalanceo {desbalance:.1f} → scale_pos_weight",
        })
        modelos.append({
            "name": "RandomForestClassifier",
            "reason": "Desbalanceo → class_weight='balanced'",
        })
        modelos.append({
            "name": "LGBMClassifier",
            "reason": "Desbalanceo → is_unbalance=True",
        })
    else:
        modelos.append({"name": "XGBClassifier", "reason": "Versatilidad general"})
        modelos.append({"name": "RandomForestClassifier", "reason": "Baseline robusto"})
        modelos.append({"name": "LGBMClassifier", "reason": "Eficiente y competitivo"})

    modelos.append({"name": "LogisticRegression", "reason": "Interpretabilidad + L1/L2"})
    modelos.append({"name": "GradientBoostingClassifier", "reason": "Boosting sklearn interpretable"})

    if n >= 500:
        modelos.append({"name": "SVC", "reason": "N suficiente para kernel RBF"})

    # --- Ensembles ---
    if n >= 100:
        modelos.append({"name": "StackingClassifier", "reason": "Ensemble: meta-learner sobre base models"})
        modelos.append({"name": "VotingClassifier", "reason": "Ensemble: voto mayoritario/soft"})
    if n >= 200:
        modelos.append({"name": "BaggingClassifier", "reason": "Ensemble: reduce varianza"})
        modelos.append({"name": "AdaBoostClassifier", "reason": "Boosting adaptativo"})

    return modelos, model_family, "f1_macro"


def _select_hp_technique(
    modelos: list[dict], n: int, config: PipelineConfig
) -> str:
    """Selecciona técnica de hiperparametrización."""
    n_models = len(modelos)
    if n_models * 50 < 200:
        return "GridSearchCV"
    elif n > 5000:
        return "Optuna"
    else:
        return "RandomizedSearchCV"


def _recommend_forecasting(
    modelo_ts: dict | None, n: int,
) -> list[dict[str, Any]]:
    """Recomienda modelos de forecasting expandidos."""
    modelos: list[dict[str, Any]] = []

    ts_type = modelo_ts.get("type", "ARIMA") if modelo_ts else "ARIMA"
    modelos.append({"name": ts_type, "reason": "TS Analyst automatic selection"})

    if ts_type != "SARIMAX":
        modelos.append({"name": "SARIMAX", "reason": "ARIMA estacional con exógenas"})

    modelos.append({"name": "ExponentialSmoothing", "reason": "Holt-Winters: trend + seasonality"})
    modelos.append({"name": "Theta", "reason": "Theta method: robusto en M3/M4 competition"})

    if n >= 100:
        modelos.append({"name": "Prophet", "reason": "Facebook Prophet: seasonality automática"})
    if n >= 200:
        modelos.append({"name": "TBATS", "reason": "Multiple seasonal patterns"})
    if n >= 50:
        modelos.append({"name": "AutoARIMA", "reason": "pmdarima: búsqueda automática (p,d,q)"})

    return modelos
