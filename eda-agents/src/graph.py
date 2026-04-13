"""Grafo LangGraph — Orquestador del pipeline EDA multi-agente.

Arquitectura:
    START → [research_lead, data_steward]   ← PARALELO
           ↓ barrera (ambos completos)
           data_engineer
           ↓ split condicional
           [statistician, ts_analyst*]       ← PARALELO (* si flag_timeseries)
           ↓ barrera
           ml_strategist
           ↓
           re_encoder                        ← nodo Python puro
           ↓
           viz_designer
           ↓
           technical_writer
           ↓ END
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import structlog
from langgraph.graph import END, START, StateGraph

from src.agents.agent_01_research_lead import research_lead
from src.agents.agent_02_data_steward import data_steward
from src.agents.agent_03_data_engineer import data_engineer
from src.agents.agent_04_statistician import statistician
from src.agents.agent_05_ts_analyst import ts_analyst
from src.agents.agent_06_ml_strategist import ml_strategist
from src.agents.agent_07_viz_designer import viz_designer
from src.agents.agent_08_technical_writer import technical_writer
from src.skills.encoding import reencode_column
from src.state import EDAState

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Nodo re_encoder (Python puro, NO usa LLM)
# ---------------------------------------------------------------------------


def re_encoder(state: EDAState) -> dict[str, Any]:
    """Re-aplica encoding definitivo según model_family del ML Strategist.

    Este nodo NO es un agente LLM. Es un nodo Python puro que:
    1. Lee model_family emitido por ML Strategist
    2. Recarga train y test provisionales
    3. Re-codifica columnas que usaban LabelEncoder provisional:
       - Si model_family == "linear" → FrequencyEncoder
       - Si model_family == "tree"  → mantiene LabelEncoder
    4. Guarda dataset_train_final.csv y dataset_test_final.csv
    """
    run_id = state["run_id"]
    log = logger.bind(agent="re_encoder", run_id=run_id)

    try:
        log.info("starting")
        model_family = state.get("model_family", "tree")
        encoding_log = dict(state.get("encoding_log", {}))

        train_prov = state.get("dataset_train_provisional", "")
        test_proc = state.get("dataset_test_procesado", "")

        if not train_prov or not test_proc:
            log.warning("no_provisional_datasets")
            return {
                "encoding_log": encoding_log,
                "dataset_train_final": train_prov,
                "dataset_test_final": test_proc,
            }

        df_train = pd.read_csv(train_prov)
        df_test = pd.read_csv(test_proc)

        # Re-encoding solo si model_family == "linear"
        if model_family == "linear":
            for col, info in list(encoding_log.items()):
                if info.get("encoding") == "label" and info.get("moment") == 1:
                    if col in df_train.columns:
                        df_train, df_test, new_enc = reencode_column(
                            df_train, df_test, col, "label", model_family
                        )
                        encoding_log[col] = {
                            **info,
                            "encoding_final": new_enc,
                            "moment": 2,
                            "reason": f"model_family={model_family}",
                        }
                    else:
                        log.warning("col_not_found_for_reencoding", col=col)
                else:
                    encoding_log[col] = {**info, "encoding_final": info.get("encoding")}
        else:
            # tree → mantener todo, solo marcar como final
            for col, info in encoding_log.items():
                encoding_log[col] = {**info, "encoding_final": info.get("encoding")}

        # Guardar datasets finales
        output_dir = Path("outputs") / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        train_final_path = str(output_dir / "dataset_train_final.csv")
        test_final_path = str(output_dir / "dataset_test_final.csv")
        df_train.to_csv(train_final_path, index=False)
        df_test.to_csv(test_final_path, index=False)

        log.info("completed", model_family=model_family, n_reencoded=sum(
            1 for v in encoding_log.values()
            if v.get("moment") == 2
        ))

        return {
            "encoding_log": encoding_log,
            "dataset_train_final": train_final_path,
            "dataset_test_final": test_final_path,
        }

    except Exception as e:
        log.error("failed", error=str(e))
        return {
            "encoding_log": state.get("encoding_log", {}),
            "dataset_train_final": state.get("dataset_train_provisional", ""),
            "dataset_test_final": state.get("dataset_test_procesado", ""),
            "error_log": [{"agent": "re_encoder", "error": str(e), "run_id": run_id}],
        }


# ---------------------------------------------------------------------------
# Routing condicional
# ---------------------------------------------------------------------------


def _route_after_engineer(state: EDAState) -> list[str]:
    """Decide si ejecutar ts_analyst en paralelo con statistician.

    Si flag_timeseries == True → ambos en paralelo.
    Si False → solo statistician.
    """
    if state.get("flag_timeseries", False):
        return ["statistician", "ts_analyst"]
    return ["statistician"]


def _should_continue_or_abort(state: EDAState) -> str:
    """Evalúa si el pipeline debe continuar o abortar.

    Verifica agent_status: si agentes críticos fallaron, aborta.
    """
    status = state.get("agent_status", {})
    critical_errors = [
        k for k, v in status.items()
        if v == "error" and k in ("ag2", "ag3")  # Data Steward y Engineer son críticos
    ]
    if critical_errors:
        return "abort"
    return "continue"


def _abort_node(state: EDAState) -> dict[str, Any]:
    """Nodo terminal de aborto — registra el error y sale."""
    log = logger.bind(agent="orchestrator", run_id=state.get("run_id", "unknown"))
    log.error("pipeline_aborted", agent_status=state.get("agent_status", {}))
    return {
        "agent_status": {**state.get("agent_status", {}), "orchestrator": "aborted"},
        "error_log": [{
            "agent": "orchestrator",
            "error": "Pipeline aborted due to critical agent failure",
            "run_id": state.get("run_id", "unknown"),
        }],
    }


# ---------------------------------------------------------------------------
# Construcción del grafo
# ---------------------------------------------------------------------------


def build_graph(checkpointer: Any = None) -> StateGraph:
    """Construye y compila el grafo LangGraph del pipeline EDA.

    Args:
        checkpointer: Checkpointer para persistencia (SqliteSaver, etc.).
            Si None, se compila sin checkpointing.

    Returns:
        Grafo compilado listo para invocar.
    """
    builder = StateGraph(EDAState)

    # --- Registrar nodos ---
    builder.add_node("research_lead", research_lead)
    builder.add_node("data_steward", data_steward)
    builder.add_node("data_engineer", data_engineer)
    builder.add_node("statistician", statistician)
    builder.add_node("ts_analyst", ts_analyst)
    builder.add_node("ml_strategist", ml_strategist)
    builder.add_node("re_encoder", re_encoder)
    builder.add_node("viz_designer", viz_designer)
    builder.add_node("technical_writer", technical_writer)
    builder.add_node("abort", _abort_node)

    # --- Paralelo inicial: research_lead + data_steward ---
    builder.add_edge(START, "research_lead")
    builder.add_edge(START, "data_steward")

    # --- Barrera 1: ambos → data_engineer ---
    builder.add_edge("research_lead", "data_engineer")
    builder.add_edge("data_steward", "data_engineer")

    # --- Condicional post-data_engineer ---
    # Si flag_timeseries → [statistician, ts_analyst] paralelo
    # Si no → solo [statistician]
    builder.add_conditional_edges(
        "data_engineer",
        _route_after_engineer,
        ["statistician", "ts_analyst"],
    )

    # --- Barrera 2: statistician (+ ts_analyst si aplica) → ml_strategist ---
    builder.add_edge("statistician", "ml_strategist")
    builder.add_edge("ts_analyst", "ml_strategist")

    # --- Secuencial: ml_strategist → re_encoder → viz_designer → writer → END ---
    builder.add_edge("ml_strategist", "re_encoder")
    builder.add_edge("re_encoder", "viz_designer")
    builder.add_edge("viz_designer", "technical_writer")
    builder.add_edge("technical_writer", END)

    # --- Abort va a END ---
    builder.add_edge("abort", END)

    # --- Compilar ---
    compiled = builder.compile(checkpointer=checkpointer)
    return compiled


def get_sqlite_checkpointer(run_id: str) -> Any:
    """Crea un SqliteSaver para checkpointing.

    Args:
        run_id: Identificador de la ejecución.

    Returns:
        SqliteSaver configurado, o None si no está disponible.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        checkpoint_dir = Path(".checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        db_path = checkpoint_dir / f"{run_id}.db"
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        return SqliteSaver(conn)
    except ImportError:
        logger.warning("sqlite_checkpointer_unavailable")
        return None
