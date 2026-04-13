"""Agente 8 — Technical Writer.

Rol: Redactor técnico
Responsabilidad: Generar informe de 12 secciones en Markdown y PDF,
decision.json, state_final.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from src.state import EDAState
from src.utils.config import PipelineConfig
from src.utils.llm import call_claude
from src.utils.state_validator import validate_ag8_output
from src.skills.report_builder import (
    build_report_markdown,
    build_report_sections,
    convert_to_pdf,
    build_decision,
    serialize_state,
)

logger = structlog.get_logger()


def technical_writer(state: EDAState) -> dict[str, Any]:
    """Agente 8 — Technical Writer.

    Rol: Redactor técnico del equipo EDA.
    Responsabilidad:
        - Generar informe de 12 secciones en Markdown
        - Convertir a PDF con weasyprint
        - Generar decision.json
        - Guardar state_final.json para auditoría
    """
    run_id = state["run_id"]
    log = logger.bind(agent="ag8", run_id=run_id)
    config = PipelineConfig.from_state(state)

    try:
        log.info("starting")
        output_dir = Path("outputs") / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Generar report.md (enriquecido con Claude si disponible) ---
        report_md = _build_enriched_report(state, config)
        report_path = output_dir / "report.md"
        report_path.write_text(report_md, encoding="utf-8")
        log.info("report_md_generated", path=str(report_path))

        # --- Convertir a PDF ---
        pdf_path = output_dir / "report.pdf"
        if convert_to_pdf(report_md, pdf_path):
            log.info("report_pdf_generated", path=str(pdf_path))
        else:
            log.warning("pdf_generation_skipped_missing_deps")

        # --- decision.json ---
        decision = build_decision(state)
        decision_path = output_dir / "decision.json"
        decision_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")

        # --- state_final.json ---
        state_final = serialize_state(state)
        state_path = output_dir / "state_final.json"
        state_path.write_text(json.dumps(state_final, indent=2, ensure_ascii=False), encoding="utf-8")

        output: dict[str, Any] = {
            "agent_status": {**state.get("agent_status", {}), "ag8": "ok"},
        }

        validate_ag8_output(output)
        log.info("completed")
        return output

    except Exception as e:
        log.error("failed", error=str(e))
        return {
            "agent_status": {**state.get("agent_status", {}), "ag8": "error"},
            "error_log": [{"agent": "ag8", "error": str(e), "run_id": run_id}],
        }


def _build_enriched_report(state: dict, config: PipelineConfig) -> str:
    """Genera informe Markdown enriquecido con narrativa de Claude.

    Si la API está disponible, cada sección se enriquece con explicaciones
    naturales generadas por Claude. Si no, retorna el report base.
    """
    sections = build_report_sections(state)

    if not config.anthropic_api_key:
        return "\n\n".join(sections)

    enriched: list[str] = []
    for section in sections:
        try:
            narrative = call_claude(
                prompt=(
                    "You are writing a professional EDA report in Spanish. "
                    "Enrich the following section with a brief narrative paragraph "
                    "(2-4 sentences) that explains the findings in plain language. "
                    "Keep the original Markdown structure and add your narrative "
                    "right after the section title. Do not remove any existing content.\n\n"
                    f"{section}"
                ),
                system=(
                    "You are a senior data scientist writing a professional report. "
                    "Write in Spanish. Be concise and insightful."
                ),
                model=config.model,
                max_tokens=1024,
                api_key=config.anthropic_api_key,
                temperature=0.3,
            )
            enriched.append(narrative)
        except Exception:
            enriched.append(section)

    return "\n\n".join(enriched)
