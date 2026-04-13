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
                modelos = [{"name": modelo_ts.get("type", "ARIMA"), "reason": "TS Analyst selection"}]
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
    """Recomienda modelos para regresión según señales del EDA."""
    modelos: list[dict[str, Any]] = []
    model_family = "tree"

    has_high_vif = len(vif_flags) > 0
    has_heterosc = bp_result is not None and bp_result.get("heteroscedastic", False)

    if has_high_vif:
        modelos.append({"name": "Ridge", "reason": "VIF alto → regularización L2"})
        modelos.append({"name": "Lasso", "reason": "VIF alto → selección L1"})
        model_family = "linear"
    if has_heterosc:
        modelos.append({"name": "SVR", "reason": "Heteroscedasticidad persistente"})
        model_family = "linear"
    if n < 200:
        if not modelos:
            modelos.append({"name": "Ridge", "reason": "N bajo → regularización"})
            model_family = "linear"
    else:
        modelos.append({"name": "XGBRegressor", "reason": "N suficiente + potencial no-lineal"})
        modelos.append({"name": "LGBMRegressor", "reason": "Alternativa eficiente"})

    if not modelos:
        modelos.append({"name": "XGBRegressor", "reason": "default"})

    return modelos, model_family, "RMSE"


def _recommend_classification(
    hallazgos: dict, desbalance: float | None, n: int
) -> tuple[list[dict[str, Any]], str, str]:
    """Recomienda modelos para clasificación según señales del EDA."""
    modelos: list[dict[str, Any]] = []
    model_family = "tree"

    if desbalance and desbalance > 3:
        modelos.append({
            "name": "XGBClassifier",
            "reason": f"Desbalanceo {desbalance:.1f} → scale_pos_weight",
        })
        modelos.append({
            "name": "RandomForestClassifier",
            "reason": "Desbalanceo → class_weight='balanced'",
        })
    else:
        modelos.append({"name": "XGBClassifier", "reason": "Versatilidad general"})
        modelos.append({"name": "RandomForestClassifier", "reason": "Baseline robusto"})

    modelos.append({"name": "LogisticRegression", "reason": "Interpretabilidad + L1"})

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
