"""Tests para src/utils/logger.py — configuración de structlog."""

from __future__ import annotations

from pathlib import Path

import structlog

from src.utils.logger import configure_logging, get_logger


class TestLogger:
    """Tests para configuración de logging."""

    def test_configure_creates_log_file(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "test_run"
        configure_logging("test001", output_dir=output_dir)
        log_path = output_dir / "run.log.jsonl"
        assert log_path.exists()

    def test_get_logger_returns_bound_logger(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "test_run"
        configure_logging("test001", output_dir=output_dir)
        log = get_logger(agent="ag1", run_id="test001")
        assert log is not None

    def test_log_writes_to_file(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "test_run"
        configure_logging("test001", output_dir=output_dir)
        log = get_logger(agent="ag1", run_id="test001")
        log.info("test_message", key="value")

        log_path = output_dir / "run.log.jsonl"
        content = log_path.read_text(encoding="utf-8")
        assert "test_message" in content
        assert "ag1" in content

    def test_output_dir_created_automatically(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "dir"
        configure_logging("test002", output_dir=nested)
        assert nested.exists()
        assert (nested / "run.log.jsonl").exists()
