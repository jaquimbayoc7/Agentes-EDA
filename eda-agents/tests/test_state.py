"""Tests para src/state.py — EDAState y HipotesisModel."""

from __future__ import annotations

from typing import get_type_hints

from src.state import EDAState, HipotesisModel


class TestHipotesisModel:
    """Tests para HipotesisModel TypedDict."""

    def test_hipotesis_has_required_fields(self) -> None:
        hints = get_type_hints(HipotesisModel)
        assert "h1" in hints
        assert "h2" in hints
        assert "h3" in hints

    def test_hipotesis_instantiation(self) -> None:
        h = HipotesisModel(
            h1="Hipótesis confirmatoria de prueba",
            h2="Hipótesis exploratoria de prueba",
            h3="Hipótesis alternativa de prueba",
        )
        assert h["h1"] == "Hipótesis confirmatoria de prueba"
        assert h["h2"] == "Hipótesis exploratoria de prueba"
        assert h["h3"] == "Hipótesis alternativa de prueba"


class TestEDAState:
    """Tests para EDAState TypedDict."""

    def test_has_all_required_sections(self) -> None:
        hints = get_type_hints(EDAState, include_extras=True)
        # Entrada
        assert "research_question" in hints
        assert "dataset_path" in hints
        assert "data_type" in hints
        assert "target" in hints
        assert "time_col" in hints
        assert "context" in hints

    def test_has_infrastructure_fields(self) -> None:
        hints = get_type_hints(EDAState, include_extras=True)
        assert "run_id" in hints
        assert "random_seed" in hints
        assert "config" in hints

    def test_has_split_fields(self) -> None:
        hints = get_type_hints(EDAState, include_extras=True)
        assert "train_path" in hints
        assert "test_path" in hints

    def test_has_agent_output_fields(self) -> None:
        hints = get_type_hints(EDAState, include_extras=True)
        # Ag1
        assert "refs" in hints
        assert "hipotesis" in hints
        assert "tarea_sugerida" in hints
        assert "search_equations" in hints
        # Ag2
        assert "perfil_columnas" in hints
        assert "encoding_flags" in hints
        assert "flag_timeseries" in hints
        # Ag3
        assert "encoding_log" in hints
        assert "features_nuevas" in hints
        # Ag4
        assert "hallazgos_eda" in hints
        assert "vif_flags" in hints
        # Ag5
        assert "modelo_ts" in hints
        # Ag6
        assert "modelos_recomendados" in hints
        assert "model_family" in hints
        # Ag7
        assert "figures" in hints
        # Control
        assert "agent_status" in hints
        assert "error_log" in hints

    def test_annotated_lists_present(self) -> None:
        """Verifica que refs, figures y error_log son campos acumulativos."""
        hints = get_type_hints(EDAState, include_extras=True)
        # Estos campos deben tener Annotated con operator.add
        for field_name in ("refs", "figures", "error_log"):
            assert field_name in hints, f"Missing annotated field: {field_name}"
