"""Validadores Pydantic para los outputs de cada agente.

Cada agente valida su output con el modelo correspondiente antes
de retornarlo al grafo. También se puede validar el estado completo
entre nodos o después de cada ejecución.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Modelos de validación por agente
# ---------------------------------------------------------------------------


class HipotesisOutput(BaseModel):
    """Validación para las hipótesis del Research Lead."""

    h1: str = Field(min_length=10, description="Hipótesis confirmatoria")
    h2: str = Field(min_length=10, description="Hipótesis exploratoria")
    h3: str = Field(min_length=10, description="Hipótesis alternativa")


class RefEntry(BaseModel):
    """Una referencia bibliográfica."""

    title: str = Field(min_length=3)
    doi: str = ""
    authors: str = ""
    year: int | str = ""
    relevance: str = ""


class Ag1Output(BaseModel):
    """Output del Research Lead (Agente 1)."""

    refs: list[RefEntry] = Field(default_factory=list)
    hipotesis: HipotesisOutput
    tarea_sugerida: Optional[str] = None
    search_equations: list[str] = Field(min_length=1)
    agent_status: dict


class Ag2Output(BaseModel):
    """Output del Data Steward (Agente 2)."""

    perfil_columnas: dict
    nulos_pct: dict
    cardinalidad: dict
    encoding_flags: dict
    desbalance_ratio: Optional[float] = None
    flag_timeseries: bool
    dataset_size: int = Field(gt=0)
    train_path: str
    test_path: str
    agent_status: dict

    @field_validator("train_path", "test_path")
    @classmethod
    def path_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Path cannot be empty")
        return v


class Ag3Output(BaseModel):
    """Output del Data Engineer (Agente 3)."""

    encoding_log: dict
    features_nuevas: list
    balanceo_log: dict
    dataset_train_provisional: str
    dataset_test_procesado: str
    agent_status: dict

    @field_validator("dataset_train_provisional", "dataset_test_procesado")
    @classmethod
    def path_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Path cannot be empty")
        return v


class Ag4Output(BaseModel):
    """Output del Statistician (Agente 4)."""

    hallazgos_eda: dict
    breusch_pagan_result: Optional[dict] = None
    modelo_correccion_heterosc: Optional[str] = None
    vif_flags: list
    vif_all: dict = Field(default_factory=dict)
    feature_importance: dict = Field(default_factory=dict)
    agent_status: dict


class Ag5Output(BaseModel):
    """Output del TS Analyst (Agente 5)."""

    modelo_ts: Optional[dict] = None
    params_pdq: Optional[dict] = None
    diagnostico_residuos_ts: Optional[dict] = None
    agent_status: dict


class Ag6Output(BaseModel):
    """Output del ML Strategist (Agente 6)."""

    modelos_recomendados: list = Field(min_length=1)
    model_family: Literal["tree", "linear"]
    hyperparams_technique: Optional[str] = None
    metrica_principal: Optional[str] = None
    advertencias: list = Field(default_factory=list)
    agent_status: dict


class ReEncoderOutput(BaseModel):
    """Output del nodo re_encoder."""

    encoding_log: dict
    dataset_train_final: str
    dataset_test_final: str

    @field_validator("dataset_train_final", "dataset_test_final")
    @classmethod
    def path_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Path cannot be empty")
        return v


class FigureEntry(BaseModel):
    """Una figura generada."""

    name: str
    path: str
    description: str = ""
    agent: str = ""


class Ag7Output(BaseModel):
    """Output del Viz Designer (Agente 7)."""

    figures: list[FigureEntry] = Field(min_length=1)
    agent_status: dict


class Ag8Output(BaseModel):
    """Output del Technical Writer (Agente 8)."""

    agent_status: dict


# ---------------------------------------------------------------------------
# Funciones de validación por agente
# ---------------------------------------------------------------------------

_VALIDATORS: dict[str, type[BaseModel]] = {
    "ag1": Ag1Output,
    "ag2": Ag2Output,
    "ag3": Ag3Output,
    "ag4": Ag4Output,
    "ag5": Ag5Output,
    "ag6": Ag6Output,
    "re_encoder": ReEncoderOutput,
    "ag7": Ag7Output,
    "ag8": Ag8Output,
}


def validate_agent_output(agent_key: str, output: dict[str, Any]) -> None:
    """Valida el output de un agente con su modelo Pydantic.

    Args:
        agent_key: Clave del agente (ej: "ag1", "ag2", ..., "re_encoder").
        output: Dict parcial retornado por el agente.

    Raises:
        ValueError: Si agent_key no existe.
        pydantic.ValidationError: Si el output no pasa la validación.
    """
    validator_cls = _VALIDATORS.get(agent_key)
    if validator_cls is None:
        raise ValueError(f"Unknown agent key: {agent_key}. Valid keys: {list(_VALIDATORS.keys())}")
    validator_cls.model_validate(output)


def validate_ag1_output(output: dict[str, Any]) -> None:
    """Valida output del Research Lead."""
    validate_agent_output("ag1", output)


def validate_ag2_output(output: dict[str, Any]) -> None:
    """Valida output del Data Steward."""
    validate_agent_output("ag2", output)


def validate_ag3_output(output: dict[str, Any]) -> None:
    """Valida output del Data Engineer."""
    validate_agent_output("ag3", output)


def validate_ag4_output(output: dict[str, Any]) -> None:
    """Valida output del Statistician."""
    validate_agent_output("ag4", output)


def validate_ag5_output(output: dict[str, Any]) -> None:
    """Valida output del TS Analyst."""
    validate_agent_output("ag5", output)


def validate_ag6_output(output: dict[str, Any]) -> None:
    """Valida output del ML Strategist."""
    validate_agent_output("ag6", output)


def validate_re_encoder_output(output: dict[str, Any]) -> None:
    """Valida output del nodo re_encoder."""
    validate_agent_output("re_encoder", output)


def validate_ag7_output(output: dict[str, Any]) -> None:
    """Valida output del Viz Designer."""
    validate_agent_output("ag7", output)


def validate_ag8_output(output: dict[str, Any]) -> None:
    """Valida output del Technical Writer."""
    validate_agent_output("ag8", output)


# ---------------------------------------------------------------------------
# Validación de estado completo (para hooks y CLI)
# ---------------------------------------------------------------------------


class FullStateValidator(BaseModel):
    """Validación del estado completo para auditoría."""

    run_id: str = Field(min_length=1)
    research_question: str = Field(min_length=5)
    dataset_path: str
    data_type: Literal["tabular", "timeseries", "mixed"]
    target: Optional[str] = None
    random_seed: int
    agent_status: dict


def validate_full_state(state: dict[str, Any]) -> None:
    """Valida campos críticos del estado completo.

    Args:
        state: Estado completo del pipeline.

    Raises:
        pydantic.ValidationError: Si falta algún campo requerido.
    """
    FullStateValidator.model_validate(state)


# ---------------------------------------------------------------------------
# CLI: python src/utils/state_validator.py <state.json>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python state_validator.py <state.json>")
        sys.exit(1)

    state_path = Path(sys.argv[1])
    if not state_path.exists():
        print(f"File not found: {state_path}")
        sys.exit(1)

    with open(state_path, "r", encoding="utf-8") as f:
        state_data = json.load(f)

    try:
        validate_full_state(state_data)
        print("State validation: OK")
    except Exception as e:
        print(f"State validation FAILED: {e}")
        sys.exit(1)
