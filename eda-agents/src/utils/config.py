"""Carga y exposición tipada de config/pipeline.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from dotenv import load_dotenv

if TYPE_CHECKING:
    from src.state import EDAState

# Cargar variables de entorno desde .env
load_dotenv()

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "pipeline.yaml"


@dataclass(frozen=True)
class ImbalanceThresholds:
    """Umbrales para decisiones de balanceo de clases."""

    oversample: int = 3
    hybrid: int = 10
    undersample: int = 30


@dataclass(frozen=True)
class EncodingConfig:
    """Configuración de encoding categórico."""

    ohe_max_categories: int = 3


@dataclass(frozen=True)
class SplitConfig:
    """Configuración de train/test split."""

    test_size: float = 0.2
    stratify: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    """Configuración central del pipeline, tipada y validada."""

    random_seed: int = 42
    model: str = "claude-sonnet-4-5"
    max_tokens: int = 4096
    imbalance_thresholds: ImbalanceThresholds = field(default_factory=ImbalanceThresholds)
    vif_threshold: int = 10
    bp_pvalue: float = 0.05
    max_rows_profiling: int = 100_000
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    # API keys (cargadas desde .env)
    anthropic_api_key: str = ""
    tavily_api_key: str = ""

    @classmethod
    def load(cls, config_path: Path | str | None = None) -> PipelineConfig:
        """Carga la configuración desde YAML y variables de entorno."""
        path = Path(config_path) if config_path else _CONFIG_PATH

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        imbalance = ImbalanceThresholds(**raw.get("imbalance_thresholds", {}))
        encoding = EncodingConfig(**raw.get("encoding", {}))
        split = SplitConfig(**raw.get("split", {}))

        return cls(
            random_seed=raw.get("random_seed", 42),
            model=raw.get("model", "claude-sonnet-4-5"),
            max_tokens=raw.get("max_tokens", 4096),
            imbalance_thresholds=imbalance,
            vif_threshold=raw.get("vif_threshold", 10),
            bp_pvalue=raw.get("bp_pvalue", 0.05),
            max_rows_profiling=raw.get("max_rows_profiling", 100_000),
            encoding=encoding,
            split=split,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        )

    @classmethod
    def from_state(cls, state: EDAState) -> PipelineConfig:
        """Carga config y sobreescribe random_seed desde el estado."""
        config = cls.load()
        # Retornar nueva instancia con el seed del estado
        return PipelineConfig(
            random_seed=state.get("random_seed", config.random_seed),
            model=config.model,
            max_tokens=config.max_tokens,
            imbalance_thresholds=config.imbalance_thresholds,
            vif_threshold=config.vif_threshold,
            bp_pvalue=config.bp_pvalue,
            max_rows_profiling=config.max_rows_profiling,
            encoding=config.encoding,
            split=config.split,
            anthropic_api_key=config.anthropic_api_key,
            tavily_api_key=config.tavily_api_key,
        )
