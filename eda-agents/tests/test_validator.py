"""Tests para src/utils/state_validator.py — validadores Pydantic."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.utils.state_validator import (
    Ag1Output,
    Ag2Output,
    Ag3Output,
    Ag4Output,
    Ag5Output,
    Ag6Output,
    Ag7Output,
    Ag8Output,
    ReEncoderOutput,
    validate_agent_output,
    validate_full_state,
)


class TestAg1Validation:
    """Tests para validador del Research Lead."""

    def test_valid_output(self) -> None:
        output = {
            "refs": [{"title": "Test paper title", "doi": "10.1234/test"}],
            "hipotesis": {
                "h1": "Hipótesis confirmatoria con suficiente longitud",
                "h2": "Hipótesis exploratoria con suficiente longitud",
                "h3": "Hipótesis alternativa con suficiente longitud",
            },
            "tarea_sugerida": "clasificación",
            "search_equations": ["equation1 AND equation2"],
            "agent_status": {"ag1": "ok"},
        }
        validate_agent_output("ag1", output)

    def test_missing_hipotesis_fails(self) -> None:
        output = {
            "refs": [],
            "search_equations": ["eq1"],
            "agent_status": {"ag1": "ok"},
        }
        with pytest.raises(ValidationError):
            validate_agent_output("ag1", output)

    def test_empty_search_equations_fails(self) -> None:
        output = {
            "refs": [],
            "hipotesis": {
                "h1": "Suficiente longitud para h1",
                "h2": "Suficiente longitud para h2",
                "h3": "Suficiente longitud para h3",
            },
            "search_equations": [],
            "agent_status": {"ag1": "ok"},
        }
        with pytest.raises(ValidationError):
            validate_agent_output("ag1", output)


class TestAg2Validation:
    """Tests para validador del Data Steward."""

    def test_valid_output(self) -> None:
        output = {
            "perfil_columnas": {"col1": {"type": "numeric"}},
            "nulos_pct": {"col1": 0.0},
            "cardinalidad": {"col1": 50},
            "encoding_flags": {"col1": "NOMINAL"},
            "desbalance_ratio": 2.5,
            "flag_timeseries": False,
            "dataset_size": 1000,
            "train_path": "/tmp/train.csv",
            "test_path": "/tmp/test.csv",
            "agent_status": {"ag2": "ok"},
        }
        validate_agent_output("ag2", output)

    def test_empty_path_fails(self) -> None:
        output = {
            "perfil_columnas": {},
            "nulos_pct": {},
            "cardinalidad": {},
            "encoding_flags": {},
            "flag_timeseries": False,
            "dataset_size": 100,
            "train_path": "",
            "test_path": "/tmp/test.csv",
            "agent_status": {"ag2": "ok"},
        }
        with pytest.raises(ValidationError):
            validate_agent_output("ag2", output)

    def test_zero_dataset_size_fails(self) -> None:
        output = {
            "perfil_columnas": {},
            "nulos_pct": {},
            "cardinalidad": {},
            "encoding_flags": {},
            "flag_timeseries": False,
            "dataset_size": 0,
            "train_path": "/tmp/train.csv",
            "test_path": "/tmp/test.csv",
            "agent_status": {"ag2": "ok"},
        }
        with pytest.raises(ValidationError):
            validate_agent_output("ag2", output)


class TestAg6Validation:
    """Tests para validador del ML Strategist."""

    def test_valid_output(self) -> None:
        output = {
            "modelos_recomendados": ["XGBoost", "Ridge"],
            "model_family": "tree",
            "hyperparams_technique": "RandomizedSearchCV",
            "metrica_principal": "F1-macro",
            "advertencias": ["Desbalance alto"],
            "agent_status": {"ag6": "ok"},
        }
        validate_agent_output("ag6", output)

    def test_invalid_model_family_fails(self) -> None:
        output = {
            "modelos_recomendados": ["XGBoost"],
            "model_family": "neural",
            "agent_status": {"ag6": "ok"},
        }
        with pytest.raises(ValidationError):
            validate_agent_output("ag6", output)

    def test_empty_models_fails(self) -> None:
        output = {
            "modelos_recomendados": [],
            "model_family": "tree",
            "agent_status": {"ag6": "ok"},
        }
        with pytest.raises(ValidationError):
            validate_agent_output("ag6", output)


class TestReEncoderValidation:
    """Tests para validador del nodo re_encoder."""

    def test_valid_output(self) -> None:
        output = {
            "encoding_log": {"col1": {"encoding_final": "FrequencyEncoder"}},
            "dataset_train_final": "/tmp/train_final.csv",
            "dataset_test_final": "/tmp/test_final.csv",
        }
        validate_agent_output("re_encoder", output)

    def test_empty_path_fails(self) -> None:
        output = {
            "encoding_log": {},
            "dataset_train_final": "  ",
            "dataset_test_final": "/tmp/test.csv",
        }
        with pytest.raises(ValidationError):
            validate_agent_output("re_encoder", output)


class TestUnknownAgent:
    """Test para agentes desconocidos."""

    def test_unknown_agent_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown agent key"):
            validate_agent_output("ag99", {})


class TestFullStateValidation:
    """Tests para validación del estado completo."""

    def test_valid_full_state(self, base_state: dict) -> None:
        validate_full_state(base_state)

    def test_missing_run_id_fails(self) -> None:
        state = {
            "research_question": "Test question here",
            "dataset_path": "/tmp/data.csv",
            "data_type": "tabular",
            "random_seed": 42,
            "agent_status": {},
        }
        with pytest.raises(ValidationError):
            validate_full_state(state)

    def test_invalid_data_type_fails(self) -> None:
        state = {
            "run_id": "test001",
            "research_question": "Test question here",
            "dataset_path": "/tmp/data.csv",
            "data_type": "image",
            "random_seed": 42,
            "agent_status": {},
        }
        with pytest.raises(ValidationError):
            validate_full_state(state)
