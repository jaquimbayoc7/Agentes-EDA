"""Tests para main.py — CLI del pipeline EDA."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from main import parse_args, main


FIXTURE_CSV = str(Path(__file__).resolve().parent / "fixtures" / "sample_100.csv")


# ---------------------------------------------------------------------------
# Tests de parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Tests para el parser de argumentos CLI."""

    def test_minimal_args(self) -> None:
        with patch("sys.argv", ["main.py", "--question", "test?", "--dataset", "d.csv"]):
            args = parse_args()
        assert args.question == "test?"
        assert args.dataset == "d.csv"
        assert args.data_type == "tabular"
        assert args.target is None
        assert args.time_col is None
        assert args.resume is None

    def test_all_args(self) -> None:
        with patch("sys.argv", [
            "main.py",
            "--question", "¿Qué factores?",
            "--dataset", "data/cafe.csv",
            "--data-type", "mixed",
            "--target", "produccion_kg",
            "--time-col", "fecha",
            "--context", "Café colombiano",
            "--resume", "abc123",
            "--config", "custom.yaml",
        ]):
            args = parse_args()
        assert args.question == "¿Qué factores?"
        assert args.data_type == "mixed"
        assert args.target == "produccion_kg"
        assert args.time_col == "fecha"
        assert args.context == "Café colombiano"
        assert args.resume == "abc123"
        assert args.config == "custom.yaml"

    def test_short_flags(self) -> None:
        with patch("sys.argv", ["main.py", "-q", "test?", "-d", "d.csv", "-t", "timeseries"]):
            args = parse_args()
        assert args.question == "test?"
        assert args.dataset == "d.csv"
        assert args.data_type == "timeseries"

    def test_invalid_data_type_exits(self) -> None:
        with patch("sys.argv", ["main.py", "-q", "x", "-d", "d.csv", "-t", "invalid"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_missing_required_args_exits(self) -> None:
        with patch("sys.argv", ["main.py"]):
            with pytest.raises(SystemExit):
                parse_args()


# ---------------------------------------------------------------------------
# Tests de main() end-to-end
# ---------------------------------------------------------------------------


class TestMainEndToEnd:
    """Tests de integración para main()."""

    def test_run_tabular_pipeline(self, tmp_path: Path) -> None:
        """Ejecuta el pipeline completo con dataset tabular de prueba."""
        with patch("sys.argv", [
            "main.py",
            "--question", "¿Qué factores predicen el target?",
            "--dataset", FIXTURE_CSV,
            "--data-type", "tabular",
            "--target", "target",
        ]):
            # Ejecutar en tmp_path context para no contaminar el workspace
            import os
            original_cwd = os.getcwd()
            os.chdir(str(tmp_path))
            try:
                # Copiar config/pipeline.yaml al tmp
                config_dir = tmp_path / "config"
                config_dir.mkdir()
                import shutil
                shutil.copy(
                    Path(original_cwd) / "config" / "pipeline.yaml",
                    config_dir / "pipeline.yaml",
                )
                main()
            finally:
                os.chdir(original_cwd)

        # Verificar que se crearon outputs
        output_dirs = list((tmp_path / "outputs").iterdir())
        assert len(output_dirs) == 1, f"Expected 1 run dir, got {len(output_dirs)}"

        run_dir = output_dirs[0]
        # Verificar artefactos clave
        assert (run_dir / "report.md").exists(), "report.md not generated"
        assert (run_dir / "decision.json").exists(), "decision.json not generated"
        assert (run_dir / "state_final.json").exists(), "state_final.json not generated"
        assert (run_dir / "run.log.jsonl").exists(), "run.log.jsonl not generated"

        # Verificar que decision.json es JSON válido
        decision = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
        assert "tarea" in decision
        assert "modelos_recomendados" in decision
        assert "model_family" in decision

        # Verificar state_final.json
        state = json.loads((run_dir / "state_final.json").read_text(encoding="utf-8"))
        assert state["research_question"] == "¿Qué factores predicen el target?"
        assert "agent_status" in state

    def test_missing_dataset_exits(self) -> None:
        """Sale con error si el dataset no existe."""
        with patch("sys.argv", [
            "main.py",
            "--question", "test?",
            "--dataset", "/nonexistent/data.csv",
        ]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_resume_uses_provided_run_id(self, tmp_path: Path) -> None:
        """--resume usa el run_id proporcionado."""
        with patch("sys.argv", [
            "main.py",
            "--question", "test?",
            "--dataset", FIXTURE_CSV,
            "--data-type", "tabular",
            "--target", "target",
            "--resume", "myrun42",
        ]):
            import os
            original_cwd = os.getcwd()
            os.chdir(str(tmp_path))
            try:
                config_dir = tmp_path / "config"
                config_dir.mkdir()
                import shutil
                shutil.copy(
                    Path(original_cwd) / "config" / "pipeline.yaml",
                    config_dir / "pipeline.yaml",
                )
                main()
            finally:
                os.chdir(original_cwd)

        run_dir = tmp_path / "outputs" / "myrun42"
        assert run_dir.exists(), "Resume run_id 'myrun42' output dir not found"
        assert (run_dir / "report.md").exists()

    def test_timeseries_pipeline(self, tmp_path: Path) -> None:
        """Ejecuta el pipeline con data-type mixed y time-col."""
        with patch("sys.argv", [
            "main.py",
            "--question", "¿Qué tendencia temporal existe?",
            "--dataset", FIXTURE_CSV,
            "--data-type", "mixed",
            "--target", "target",
            "--time-col", "fecha",
        ]):
            import os
            original_cwd = os.getcwd()
            os.chdir(str(tmp_path))
            try:
                config_dir = tmp_path / "config"
                config_dir.mkdir()
                import shutil
                shutil.copy(
                    Path(original_cwd) / "config" / "pipeline.yaml",
                    config_dir / "pipeline.yaml",
                )
                main()
            finally:
                os.chdir(original_cwd)

        output_dirs = list((tmp_path / "outputs").iterdir())
        assert len(output_dirs) == 1
        run_dir = output_dirs[0]

        state = json.loads((run_dir / "state_final.json").read_text(encoding="utf-8"))
        # ts_analyst should have run
        assert "ag5" in state.get("agent_status", {}), "TS Analyst should have run"
