"""Estado compartido del sistema multi-agente EDA.

EDAState es el TypedDict central que todos los agentes leen y escriben.
Cada agente retorna un dict parcial que LangGraph fusiona automáticamente.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional, TypedDict


def _merge_dicts(left: dict, right: dict) -> dict:
    """Reducer que fusiona dos dicts (para campos escritos en paralelo)."""
    merged = left.copy() if left else {}
    if right:
        merged.update(right)
    return merged


class HipotesisModel(TypedDict):
    """Tres hipótesis generadas por el Research Lead.

    h1: confirmatoria — relación esperada según literatura.
    h2: exploratoria — relación plausible no documentada.
    h3: alternativa — desafía el supuesto más común.
    """

    h1: str
    h2: str
    h3: str


class EDAState(TypedDict):
    """Estado compartido del pipeline EDA multi-agente.

    Todos los agentes leen de este estado y retornan dicts parciales
    que LangGraph fusiona. Los campos Annotated con operator.add son
    acumulativos (listas que se extienden).
    """

    # --- Entrada ---
    research_question: str
    dataset_path: str
    data_type: Literal["tabular", "timeseries", "mixed"]
    target: Optional[str]
    time_col: Optional[str]
    context: str

    # --- Infraestructura ---
    run_id: str
    random_seed: int
    config: dict

    # --- Splits ---
    train_path: str
    test_path: str

    # --- Ag.1 Research Lead ---
    refs: Annotated[list, operator.add]
    hipotesis: Optional[HipotesisModel]
    tarea_sugerida: Optional[str]
    task_override: Optional[str]
    search_equations: Annotated[list, operator.add]

    # --- Ag.2 Data Steward ---
    perfil_columnas: dict
    nulos_pct: dict
    cardinalidad: dict
    encoding_flags: dict
    desbalance_ratio: Optional[float]
    flag_timeseries: bool
    dataset_size: int

    # --- Ag.3 Data Engineer ---
    encoding_log: dict
    features_nuevas: list
    balanceo_log: dict
    sampling_variants: dict  # {oversample: {path, ratio_after, ...}, undersample: ..., hybrid: ...}
    dataset_train_provisional: str
    dataset_test_procesado: str
    dataset_train_final: str
    dataset_test_final: str

    # --- Ag.4 Statistician ---
    hallazgos_eda: dict
    breusch_pagan_result: Optional[dict]
    modelo_correccion_heterosc: Optional[str]
    vif_flags: list
    vif_all: dict
    feature_importance: dict

    # --- Ag.5 TS Analyst (condicional) ---
    modelo_ts: Optional[dict]
    params_pdq: Optional[dict]
    diagnostico_residuos_ts: Optional[dict]

    # --- Ag.6 ML Strategist ---
    modelos_recomendados: list
    model_family: Optional[Literal["tree", "linear"]]
    hyperparams_technique: Optional[str]
    metrica_principal: Optional[str]
    advertencias: list

    # --- Ag.7 Viz Designer ---
    figures: Annotated[list, operator.add]

    # --- Control de flujo ---
    agent_status: Annotated[dict, _merge_dicts]  # {ag1: "ok"|"fallback"|"error", ...}
    error_log: Annotated[list, operator.add]
