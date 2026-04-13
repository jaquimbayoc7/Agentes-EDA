"""Configuración de structlog para logging estructurado por run_id."""

from __future__ import annotations

import sys
from pathlib import Path

import structlog


def configure_logging(run_id: str, output_dir: str | Path | None = None) -> None:
    """Configura structlog para escribir logs a outputs/{run_id}/run.log.jsonl.

    Args:
        run_id: Identificador único de la ejecución.
        output_dir: Directorio base de outputs. Si None, usa outputs/{run_id}/.
    """
    if output_dir is None:
        output_dir = Path("outputs") / run_id
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log.jsonl"

    # Abrir el archivo de log en modo append
    log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.WriteLoggerFactory(file=log_file),
        cache_logger_on_first_use=False,
    )


def get_logger(agent: str | None = None, run_id: str | None = None) -> structlog.BoundLogger:
    """Obtiene un logger con contexto de agente y run_id pre-bindeado.

    Args:
        agent: Nombre del agente (ej: "ag1", "data_steward").
        run_id: Identificador de la ejecución.

    Returns:
        Logger estructurado con contexto.
    """
    log = structlog.get_logger()
    if agent:
        log = log.bind(agent=agent)
    if run_id:
        log = log.bind(run_id=run_id)
    return log
