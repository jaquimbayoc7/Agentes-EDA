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
from src.utils.tavily_client import search_literature_tavily

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

        # --- Inferir tarea sugerida (o usar override del CLI) ---
        task_override = state.get("task_override")
        if task_override:
            tarea_sugerida = task_override
            log.info("task_override_used", task=task_override)
        else:
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
    """Busca literatura académica usando Tavily + Claude.

    Estrategia:
    1. Si Tavily API key disponible → búsqueda web real por ecuaciones PICO
    2. Claude sintetiza/complementa con conocimiento propio
    3. Combina resultados deduplicados
    """
    refs: list[dict[str, Any]] = []

    # --- Paso 1: Búsqueda real con Tavily ---
    if config.tavily_api_key:
        try:
            tavily_results = search_literature_tavily(
                equations, api_key=config.tavily_api_key, max_results_per_eq=3,
            )
            for item in tavily_results:
                refs.append({
                    "title": item.get("title", ""),
                    "authors": "",
                    "year": "",
                    "doi": "",
                    "key_finding": item.get("content", "")[:300],
                    "relevance": f"Web search result (score: {item.get('score', 0):.2f})",
                    "url": item.get("url", ""),
                    "source": "tavily",
                })
        except Exception as tv_err:
            logger.warning("tavily_search_failed", error=str(tv_err))

    # --- Paso 2: Claude complementa con conocimiento académico ---
    if not config.anthropic_api_key:
        return refs

    existing_titles = {r.get("title", "").lower() for r in refs}
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

    claude_refs = result.get("refs", [])
    for ref in claude_refs:
        if isinstance(ref, dict) and ref.get("title"):
            title_lower = ref.get("title", "").lower()
            if title_lower not in existing_titles:
                existing_titles.add(title_lower)
                refs.append({
                    "title": ref.get("title", ""),
                    "authors": ref.get("authors", ""),
                    "year": ref.get("year", ""),
                    "doi": ref.get("doi", ""),
                    "key_finding": ref.get("key_finding", ""),
                    "relevance": ref.get("relevance", ""),
                    "source": "claude",
                })
    return refs


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
    if any(w in q_lower for w in (
        "predec", "predic", "predict", "regres", "estim", "valor", "cuant",
        "numer", "continu", "precio", "cost", "amount", "quantity",
    )):
        return "regression"
    if any(w in q_lower for w in (
        "clasific", "categor", "detect", "diagnos", "tipo", "clase", "binari",
    )):
        return "classification"
    if any(w in q_lower for w in (
        "forecast", "serie", "temporal", "tiempo", "pronostic", "tendencia",
    )):
        return "forecasting"
    return "classification"


# ---------------------------------------------------------------------------
# Nodo de refinamiento de ecuaciones post-EDA
# ---------------------------------------------------------------------------


def refine_search_equations(state: EDAState) -> dict[str, Any]:
    """Refina ecuaciones de búsqueda basándose en hallazgos del EDA.

    Después de que el Statistician produce hallazgos_eda (correlaciones,
    outliers, normalidad, VIF, Breusch-Pagan), este nodo usa Claude para
    generar ecuaciones PICO mejoradas y buscar literatura adicional.
    """
    run_id = state["run_id"]
    log = logger.bind(agent="refine_equations", run_id=run_id)
    config = PipelineConfig.from_state(state)

    try:
        log.info("starting")
        hallazgos = state.get("hallazgos_eda", {})
        question = state["research_question"]
        context = state.get("context", "")
        original_eqs = state.get("search_equations", [])
        existing_refs = state.get("refs", [])

        if not hallazgos or not config.anthropic_api_key:
            log.info("skipped", reason="no hallazgos or no API key")
            return {}

        # --- Generar ecuaciones refinadas ---
        refined_eqs = _build_refined_equations(
            question, context, hallazgos, original_eqs, config
        )

        if not refined_eqs:
            log.info("no_refined_equations")
            return {}

        # --- Buscar literatura con ecuaciones refinadas ---
        existing_titles = {r.get("title", "").lower() for r in existing_refs}
        new_refs = _search_literature(refined_eqs, question, context, config)

        # Filtrar refs duplicadas
        unique_refs = [
            r for r in new_refs
            if r.get("title", "").lower() not in existing_titles
        ]

        log.info(
            "completed",
            n_refined_eqs=len(refined_eqs),
            n_new_refs=len(unique_refs),
        )
        return {
            "search_equations": refined_eqs,
            "refs": unique_refs,
        }

    except Exception as e:
        log.error("failed", error=str(e))
        return {
            "error_log": [{
                "agent": "refine_equations",
                "error": str(e),
                "run_id": run_id,
            }],
        }


def _build_refined_equations(
    question: str,
    context: str,
    hallazgos: dict[str, Any],
    original_eqs: list[str],
    config: PipelineConfig,
) -> list[str]:
    """Genera ecuaciones PICO refinadas a partir de hallazgos estadísticos."""
    import json as _json

    correlations_text = _json.dumps(
        hallazgos.get("correlations", {}), default=str
    )[:600]
    outliers_text = _json.dumps(
        hallazgos.get("outliers", {}), default=str
    )[:400]
    normality_text = _json.dumps(
        hallazgos.get("normality", {}), default=str
    )[:400]
    vif_text = _json.dumps(
        hallazgos.get("vif_summary", {}), default=str
    )[:200]
    interpretation = hallazgos.get("interpretation", "")[:500]
    original_text = "\n".join(f"- {eq}" for eq in original_eqs)

    try:
        result = call_claude_json(
            prompt=(
                f"Research question: {question}\n"
                f"Context: {context}\n\n"
                f"Original search equations:\n{original_text}\n\n"
                "The following are ACTUAL statistical findings from the dataset:\n"
                f"- Correlations: {correlations_text}\n"
                f"- Outliers: {outliers_text}\n"
                f"- Normality tests: {normality_text}\n"
                f"- VIF (multicollinearity): {vif_text}\n"
                f"- Interpretation: {interpretation}\n\n"
                "Based on these real data findings, generate 3 NEW and IMPROVED "
                "boolean PICO search equations that:\n"
                "1. Focus on the specific variables/relationships found significant\n"
                "2. Investigate unexpected patterns (outliers, non-normality)\n"
                "3. Explore methodological approaches for issues found "
                "(multicollinearity, heteroscedasticity)\n\n"
                "The new equations must be DIFFERENT from the originals and more "
                "targeted based on actual data evidence.\n"
                'Return JSON: {"equations": ["eq1", "eq2", "eq3"]}'
            ),
            system=(
                "You are a research methodology expert. Generate refined PICO "
                "search equations informed by real statistical findings. "
                "Return only valid JSON."
            ),
            model=config.model,
            max_tokens=1024,
            api_key=config.anthropic_api_key,
        )
        equations = result.get("equations", [])
        return equations if equations else []
    except Exception:
        return []
