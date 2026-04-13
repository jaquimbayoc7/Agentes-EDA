"""Tests para src/utils/config.py — PipelineConfig."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import (
    EncodingConfig,
    ImbalanceThresholds,
    PipelineConfig,
    SplitConfig,
)


class TestPipelineConfig:
    """Tests para carga y tipado de configuración."""

    def test_load_from_default_path(self) -> None:
        config = PipelineConfig.load()
        assert config.random_seed == 42
        assert config.model == "claude-sonnet-4-5"
        assert config.max_tokens == 4096

    def test_imbalance_thresholds(self) -> None:
        config = PipelineConfig.load()
        assert config.imbalance_thresholds.oversample == 3
        assert config.imbalance_thresholds.hybrid == 10
        assert config.imbalance_thresholds.undersample == 30

    def test_vif_and_bp(self) -> None:
        config = PipelineConfig.load()
        assert config.vif_threshold == 10
        assert config.bp_pvalue == 0.05

    def test_encoding_config(self) -> None:
        config = PipelineConfig.load()
        assert config.encoding.ohe_max_categories == 3

    def test_split_config(self) -> None:
        config = PipelineConfig.load()
        assert config.split.test_size == 0.2
        assert config.split.stratify is True

    def test_max_rows_profiling(self) -> None:
        config = PipelineConfig.load()
        assert config.max_rows_profiling == 100_000

    def test_load_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            PipelineConfig.load("/nonexistent/path.yaml")

    def test_from_state_overrides_seed(self) -> None:
        state = {
            "random_seed": 123,
            "run_id": "test",
            "config": {},
        }
        config = PipelineConfig.from_state(state)
        assert config.random_seed == 123

    def test_frozen_dataclass(self) -> None:
        config = PipelineConfig.load()
        with pytest.raises(AttributeError):
            config.random_seed = 99  # type: ignore[misc]

    def test_custom_yaml(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom.yaml"
        custom.write_text(
            "random_seed: 99\n"
            "model: test-model\n"
            "max_tokens: 2048\n"
            "vif_threshold: 5\n"
            "bp_pvalue: 0.01\n"
            "max_rows_profiling: 50000\n"
            "imbalance_thresholds:\n"
            "  oversample: 2\n"
            "  hybrid: 5\n"
            "  undersample: 20\n"
            "encoding:\n"
            "  ohe_max_categories: 5\n"
            "split:\n"
            "  test_size: 0.3\n"
            "  stratify: false\n",
            encoding="utf-8",
        )
        config = PipelineConfig.load(custom)
        assert config.random_seed == 99
        assert config.model == "test-model"
        assert config.vif_threshold == 5
        assert config.split.test_size == 0.3
        assert config.split.stratify is False
        assert config.encoding.ohe_max_categories == 5
