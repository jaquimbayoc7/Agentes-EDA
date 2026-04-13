"""Agente 1 — Research Lead.

Rol: Investigador principal
Responsabilidad: Construir ecuaciones PICO, buscar literatura con Claude,
generar 3 hipótesis (H1 confirmatoria, H2 exploratoria, H3 alternativa).
"""

from __future__ import annotations

from typing import Any

import structlog

from src.state import EDAState
from src.utils.config import PipelineConfig
from src.utils.llm import call_claude, call_claude_json
from src.utils.state_validator import validate_ag1_output

logger = structlog.get_logger()


def research_lead(state: EDAState) -> dict[str, Any]:
    """Agente 1 — Research Lead.

    Rol: Investigador principal del equipo EDA.
    Responsabilidad:
        - Construir ecuaciones booleanas PICO desde research_question
        - Buscar literatura académica relevante con Claude
        - Extraer refs con DOI, variables reportadas, tarea sugerida
        - Generar 3 hipótesis: H1 confirmatoria, H2 exploratoria, H3 alternativa
    """
    run_id = state["run_id"]
    log = logger.bind(agent="ag1", run_id=run_id)
    config = PipelineConfig.from_state(state)

    try:
        log.info("starting")
        question = state["research_question"]
        context = state.get("context", "")

        # --- Construir ecuaciones PICO ---
        search_equations = _build_pico_equations(question, context, config)

        # --- Buscar literatura con Claude ---
        refs = []
        research_ok = True
        try:
            refs = _search_literature(search_equations, question, context, config)
        except Exception as res_err:
            log.warning("literature_search_failed", error=str(res_err))
            research_ok = False

        # --- Generar hipótesis ---
        hipotesis = _generate_hypotheses(question, context, refs, config)

        # --- Inferir tarea sugerida ---
        tarea_sugerida = _infer_task(question, refs, config)

        status = "ok" if research_ok else "fallback"
        output: dict[str, Any] = {
            "refs": refs,
            "hipotesis": hipotesis,
            "tarea_sugerida": tarea_sugerida,
            "search_equations": search_equations,
            "agent_status": {**state.get("agent_status", {}), "ag1": status},
        }

        validate_ag1_output(output)
        log.info("completed", status=status, n_refs=len(refs))
        return output

    except Exception as e:
        log.error("failed", error=str(e))
        return {
            "refs": [],
            "hipotesis": {"h1": "Error generating", "h2": "Error generating", "h3": "Error generating"},
            "tarea_sugerida": None,
            "search_equations": [state["research_question"]],
            "agent_status": {**state.get("agent_status", {}), "ag1": "error"},
            "error_log": [{"agent": "ag1", "error": str(e), "run_id": run_id}],
        }


def _build_pico_equations(
    question: str, context: str, config: PipelineConfig
) -> list[str]:
    """Construye 3 ecuaciones booleanas PICO en cascada desde la pregunta."""
    if not config.anthropic_api_key:
        return [question]

    try:
        result = call_claude_json(
            prompt=(
                f"Research question: {question}\n"
                f"Context: {context}\n\n"
                "Generate exactly 3 boolean PICO search equations for academic literature "
                "retrieval. Each equation should use AND/OR operators and focus on different "
                "aspects: broad, specific, and methodological.\n"
                "Return JSON: {\"equations\": [\"eq1\", \"eq2\", \"eq3\"]}"
            ),
            system="You are a research methodology expert. Return only valid JSON.",
            model=config.model,
            max_tokens=1024,
            api_key=config.anthropic_api_key,
        )
        equations = result.get("equations", [question])
        return equations if equations else [question]
    except Exception:
        return [question]


def _search_literature(
    equations: list[str],
    question: str,
    context: str,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    """Busca literatura académica relevante usando Claude.

    Claude actúa como asistente de investigación: dado las ecuaciones PICO
    y la pregunta de investigación, genera una revisión estructurada con
    referencias plausibles, autores, años y hallazgos clave.
    """
    if not config.anthropic_api_key:
        return []

    eq_text = "\n".join(f"- {eq}" for eq in equations)

    result = call_claude_json(
        prompt=(
            f"Research question: {question}\n"
            f"Context: {context}\n"
            f"Search equations:\n{eq_text}\n\n"
            "You are a senior research assistant conducting a literature review.\n"
            "Based on your training knowledge, identify 5-8 real and relevant "
            "academic papers, studies, or well-known references related to this "
            "research question. For each reference provide:\n"
            "- title: the paper/study title\n"
            "- authors: main authors\n"
            "- year: publication year (approximate if unsure)\n"
            "- doi: DOI if you know it, otherwise empty string\n"
            "- key_finding: one-sentence summary of the main finding\n"
            "- relevance: how it relates to the research question\n\n"
            "Focus on well-established, frequently-cited works in the field. "
            "If you are not sure about exact details, note it in the title. "
            "Do NOT fabricate DOIs — leave empty if unknown.\n\n"
            'Return JSON: {"refs": [{"title": ..., "authors": ..., '
            '"year": ..., "doi": ..., "key_finding": ..., "relevance": ...}]}'
        ),
        system=(
            "You are a scientific literature expert. Identify real, well-known "
            "academic works relevant to the query. Be honest about uncertainty. "
            "Return only valid JSON."
        ),
        model=config.model,
        max_tokens=2048,
        api_key=config.anthropic_api_key,
    )

    refs = result.get("refs", [])
    # Ensure each ref has required keys
    clean_refs: list[dict[str, Any]] = []
    for ref in refs:
        if isinstance(ref, dict) and ref.get("title"):
            clean_refs.append({
                "title": ref.get("title", ""),
                "authors": ref.get("authors", ""),
                "year": ref.get("year", ""),
                "doi": ref.get("doi", ""),
                "key_finding": ref.get("key_finding", ""),
                "relevance": ref.get("relevance", ""),
            })
    return clean_refs


def _generate_hypotheses(
    question: str,
    context: str,
    refs: list[dict[str, Any]],
    config: PipelineConfig,
) -> dict[str, str]:
    """Genera 3 hipótesis usando Claude API con contexto del dominio."""
    if not config.anthropic_api_key:
        return {
            "h1": f"Confirmatoria: La relación principal sugerida por la literatura para '{question}' se confirma.",
            "h2": f"Exploratoria: Existen relaciones no documentadas relevantes para '{question}'.",
            "h3": f"Alternativa: El supuesto más común sobre '{question}' podría no ser válido.",
        }

    refs_summary = "\n".join(
        f"- {r.get('title', 'N/A')}: {r.get('key_finding', '')}" for r in refs[:10]
    ) or "No references available."

    try:
        result = call_claude_json(
            prompt=(
                f"Research question: {question}\n"
                f"Context: {context}\n"
                f"Literature references:\n{refs_summary}\n\n"
                "Generate 3 hypotheses:\n"
                "- H1 (confirmatory): Tests the main relationship suggested by literature\n"
                "- H2 (exploratory): Proposes undocumented but plausible relationships\n"
                "- H3 (alternative): Challenges the most common assumption\n\n"
                'Return JSON: {"h1": "...", "h2": "...", "h3": "..."}'
            ),
            system="You are a senior researcher. Generate precise, testable hypotheses.",
            model=config.model,
            max_tokens=1024,
            api_key=config.anthropic_api_key,
        )
        return {
            "h1": result.get("h1", "Error generating h1"),
            "h2": result.get("h2", "Error generating h2"),
            "h3": result.get("h3", "Error generating h3"),
        }
    except Exception:
        return {
            "h1": f"Confirmatoria: La relación principal sugerida por la literatura para '{question}' se confirma.",
            "h2": f"Exploratoria: Existen relaciones no documentadas relevantes para '{question}'.",
            "h3": f"Alternativa: El supuesto más común sobre '{question}' podría no ser válido.",
        }


def _infer_task(
    question: str,
    refs: list[dict[str, Any]],
    config: PipelineConfig,
) -> str | None:
    """Infiere tipo de tarea sugerida desde la pregunta y refs."""
    if config.anthropic_api_key:
        try:
            refs_ctx = "\n".join(
                f"- {r.get('title', '')}" for r in refs[:5]
            ) or "None"
            result = call_claude_json(
                prompt=(
                    f"Research question: {question}\n"
                    f"References:\n{refs_ctx}\n\n"
                    "What ML task type best fits this question? "
                    "Choose exactly one: regression, classification, forecasting.\n"
                    'Return JSON: {"task": "..."}'
                ),
                system="You are an ML expert. Return only valid JSON.",
                model=config.model,
                max_tokens=256,
                api_key=config.anthropic_api_key,
            )
            task = result.get("task", "")
            if task in ("regression", "classification", "forecasting"):
                return task
        except Exception:
            pass

    # Fallback heurístico
    q_lower = question.lower()
    if any(w in q_lower for w in ("predic", "estim", "regres")):
        return "regression"
    if any(w in q_lower for w in ("clasific", "categor", "detect")):
        return "classification"
    if any(w in q_lower for w in ("forecast", "serie", "temporal", "tiempo")):
        return "forecasting"
    return "classification"
