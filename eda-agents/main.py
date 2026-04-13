"""Punto de entrada CLI para el pipeline EDA multi-agente.

Uso:
    python main.py \\
      --question "¿Qué factores climáticos predicen la producción de café?" \\
      --dataset data/cafe.csv \\
      --data-type mixed \\
      --target produccion_kg \\
      --time-col fecha \\
      --resume {run_id}   ← opcional para reanudar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

import structlog

from src.graph import build_graph, get_sqlite_checkpointer
from src.state import EDAState
from src.utils.config import PipelineConfig
from src.utils.logger import configure_logging


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="EDA Agents — Pipeline multi-agente de análisis exploratorio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--question", "-q",
        required=True,
        help="Pregunta de investigación (ej: '¿Qué factores predicen X?')",
    )
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="Ruta al dataset CSV",
    )
    parser.add_argument(
        "--data-type", "-t",
        choices=["tabular", "timeseries", "mixed"],
        default="tabular",
        help="Tipo de datos (default: tabular)",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Nombre de la columna target",
    )
    parser.add_argument(
        "--time-col",
        default=None,
        help="Nombre de la columna temporal (para series de tiempo)",
    )
    parser.add_argument(
        "--context",
        default="",
        help="Contexto adicional del dominio",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="run_id para reanudar una ejecución previa",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Ruta a pipeline.yaml personalizado",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada principal."""
    args = parse_args()

    # --- Run ID ---
    run_id = args.resume if args.resume else str(uuid4())[:8]

    # --- Configurar logging ---
    configure_logging(run_id)
    log = structlog.get_logger().bind(run_id=run_id)
    log.info("pipeline_starting", question=args.question, dataset=args.dataset)

    # --- Cargar config ---
    config = PipelineConfig.load(args.config)

    # --- Validar dataset ---
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        log.error("dataset_not_found", path=str(dataset_path))
        print(f"Error: Dataset no encontrado: {dataset_path}")
        sys.exit(1)

    # --- Crear directorio de outputs ---
    output_dir = Path("outputs") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Inicializar estado ---
    initial_state: dict = {
        "research_question": args.question,
        "dataset_path": str(dataset_path.resolve()),
        "data_type": args.data_type,
        "target": args.target,
        "time_col": args.time_col,
        "context": args.context or "",
        "run_id": run_id,
        "random_seed": config.random_seed,
        "config": {},
        "train_path": "",
        "test_path": "",
        "refs": [],
        "hipotesis": None,
        "tarea_sugerida": None,
        "search_equations": [],
        "perfil_columnas": {},
        "nulos_pct": {},
        "cardinalidad": {},
        "encoding_flags": {},
        "desbalance_ratio": None,
        "flag_timeseries": False,
        "dataset_size": 0,
        "encoding_log": {},
        "features_nuevas": [],
        "balanceo_log": {},
        "dataset_train_provisional": "",
        "dataset_test_procesado": "",
        "dataset_train_final": "",
        "dataset_test_final": "",
        "hallazgos_eda": {},
        "breusch_pagan_result": None,
        "modelo_correccion_heterosc": None,
        "vif_flags": [],
        "modelo_ts": None,
        "params_pdq": None,
        "diagnostico_residuos_ts": None,
        "modelos_recomendados": [],
        "model_family": None,
        "hyperparams_technique": None,
        "metrica_principal": None,
        "advertencias": [],
        "figures": [],
        "agent_status": {},
        "error_log": [],
    }

    # --- Checkpointer ---
    checkpointer = get_sqlite_checkpointer(run_id)

    # --- Construir y ejecutar grafo ---
    print(f"\n{'='*60}")
    print(f"  EDA Agents — Run: {run_id}")
    print(f"  Pregunta: {args.question}")
    print(f"  Dataset: {args.dataset}")
    print(f"{'='*60}\n")

    graph = build_graph(checkpointer=checkpointer)

    graph_config = {"configurable": {"thread_id": run_id}}

    try:
        # Ejecutar nodo por nodo con streaming
        for event in graph.stream(initial_state, config=graph_config):
            for node_name, node_output in event.items():
                status = "?"
                if isinstance(node_output, dict):
                    agent_status = node_output.get("agent_status", {})
                    # Obtener el status del último agente que escribió
                    for k, v in agent_status.items():
                        status = v
                print(f"  [OK] {node_name:25s} [{status}]")

        # --- Guardar estado final ---
        print(f"\n{'='*60}")
        print(f"  Pipeline completado")
        print(f"  Outputs en: outputs/{run_id}/")
        print(f"{'='*60}\n")

        log.info("pipeline_completed", run_id=run_id)

    except Exception as e:
        log.error("pipeline_failed", error=str(e))
        print(f"\n  [FAIL] Pipeline fallo: {e}")
        # Guardar estado parcial
        error_state = {
            "run_id": run_id,
            "error": str(e),
            "agent_status": initial_state.get("agent_status", {}),
        }
        error_path = output_dir / "state_error.json"
        error_path.write_text(
            json.dumps(error_state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
