"""Skill reutilizable — Construcción de reportes.

Funciones puras para generar:
- Informe Markdown de 12 secciones
- Conversión a PDF (weasyprint)
- decision.json
- Serialización segura del estado

Usada por Agent 08 (Technical Writer).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def build_report_sections(state: dict[str, Any]) -> list[str]:
    """Genera las 12 secciones del informe EDA en Markdown.

    Returns
    -------
    Lista de strings, una por sección.
    """
    sections: list[str] = []

    # §1 Pregunta de investigación
    sections.append(
        f"# §1 Pregunta de investigación y contexto\n\n"
        f"**Pregunta:** {state.get('research_question', 'N/A')}\n\n"
        f"**Contexto:** {state.get('context', 'N/A')}\n\n"
        f"**Tipo de datos:** {state.get('data_type', 'N/A')}\n"
    )

    # §2 Revisión de literatura
    refs = state.get("refs", [])
    refs_text = (
        "\n".join(
            f"- {r.get('title', 'N/A')} (DOI: {r.get('doi', 'N/A')})"
            for r in refs
        )
        if refs
        else "No se encontraron referencias."
    )
    equations = state.get("search_equations", [])
    eq_text = "\n".join(f"- `{eq}`" for eq in equations) if equations else "N/A"
    sections.append(
        f"# §2 Revisión de literatura\n\n"
        f"## Ecuaciones PICO\n{eq_text}\n\n"
        f"## Referencias\n{refs_text}\n"
    )

    # §3 Hipótesis
    hip = state.get("hipotesis") or {}
    sections.append(
        f"# §3 Hipótesis\n\n"
        f"- **H1 (confirmatoria):** {hip.get('h1', 'N/A')}\n"
        f"- **H2 (exploratoria):** {hip.get('h2', 'N/A')}\n"
        f"- **H3 (alternativa):** {hip.get('h3', 'N/A')}\n"
    )

    # §4 Descripción del dataset
    sections.append(
        f"# §4 Descripción del dataset\n\n"
        f"- **Tamaño:** {state.get('dataset_size', 'N/A')} filas\n"
        f"- **Target:** {state.get('target', 'N/A')}\n"
        f"- **Desbalanceo:** {state.get('desbalance_ratio', 'N/A')}\n"
        f"- **Serie temporal:** {state.get('flag_timeseries', False)}\n"
    )

    # §5 Preprocesamiento
    sections.append(
        f"# §5 Preprocesamiento\n\n"
        f"## Encoding\n```json\n{json.dumps(state.get('encoding_log', {}), indent=2)}\n```\n\n"
        f"## Features nuevas\n{state.get('features_nuevas', [])}\n\n"
        f"## Balanceo\n```json\n{json.dumps(state.get('balanceo_log', {}), indent=2)}\n```\n"
    )

    # §6 Hallazgos EDA tabular
    hallazgos = state.get("hallazgos_eda", {})
    sections.append(
        f"# §6 Hallazgos EDA tabular\n\n"
        f"```json\n{json.dumps(hallazgos, indent=2, default=str)}\n```\n"
    )

    # §7 Hallazgos EDA series de tiempo
    if state.get("flag_timeseries"):
        sections.append(
            f"# §7 Hallazgos EDA series de tiempo\n\n"
            f"**Modelo TS:** {state.get('modelo_ts', 'N/A')}\n\n"
            f"**Parámetros:** {state.get('params_pdq', 'N/A')}\n"
        )
    else:
        sections.append("# §7 Hallazgos EDA series de tiempo\n\nNo aplica.\n")

    # §8 Decisión de tarea
    sections.append(
        f"# §8 Decisión de tarea\n\n"
        f"**Tarea:** {state.get('tarea_sugerida', 'N/A')}\n\n"
        f"**Model family:** {state.get('model_family', 'N/A')}\n"
    )

    # §9 Modelos recomendados
    modelos = state.get("modelos_recomendados", [])
    modelos_text = (
        "\n".join(
            f"- **{m.get('name', 'N/A')}:** {m.get('reason', 'N/A')}"
            for m in modelos
        )
        if modelos
        else "N/A"
    )
    sections.append(f"# §9 Modelos recomendados\n\n{modelos_text}\n")

    # §10 Hiperparametrización
    sections.append(
        f"# §10 Técnica de hiperparametrización\n\n"
        f"**Técnica:** {state.get('hyperparams_technique', 'N/A')}\n\n"
        f"**Métrica principal:** {state.get('metrica_principal', 'N/A')}\n"
    )

    # §11 Advertencias
    warns = state.get("advertencias", [])
    warns_text = "\n".join(f"- {w}" for w in warns) if warns else "Ninguna."
    sections.append(f"# §11 Advertencias y limitaciones\n\n{warns_text}\n")

    # §12 Próximos pasos
    sections.append(
        "# §12 Próximos pasos\n\n"
        "1. Entrenar modelos recomendados con los hiperparámetros sugeridos\n"
        "2. Validar con cross-validation sobre train set\n"
        "3. Evaluar en test set con métricas seleccionadas\n"
        "4. Iterar según resultados\n"
    )

    return sections


def build_report_markdown(state: dict[str, Any]) -> str:
    """Genera el informe completo como string Markdown."""
    return "\n\n".join(build_report_sections(state))


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


_DEFAULT_CSS = """
body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
pre { background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
th { background-color: #3498db; color: white; }
"""


def convert_to_pdf(markdown_text: str, pdf_path: Path, css: str = _DEFAULT_CSS) -> bool:
    """Convierte Markdown a PDF con weasyprint.

    Returns True si se generó exitosamente, False si falta dependencia.
    """
    try:
        import markdown
        from weasyprint import HTML

        html_content = markdown.markdown(
            markdown_text, extensions=["tables", "fenced_code"]
        )
        styled = f"<html><head><style>{css}</style></head><body>{html_content}</body></html>"
        HTML(string=styled).write_pdf(str(pdf_path))
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# decision.json
# ---------------------------------------------------------------------------


def build_decision(state: dict[str, Any]) -> dict[str, Any]:
    """Construye el dict de decisión final del pipeline."""
    return {
        "tarea": state.get("tarea_sugerida"),
        "modelos_recomendados": state.get("modelos_recomendados", []),
        "hyperparams_technique": state.get("hyperparams_technique"),
        "model_family": state.get("model_family"),
        "metrica_principal": state.get("metrica_principal"),
    }


# ---------------------------------------------------------------------------
# Serialización segura del estado
# ---------------------------------------------------------------------------


def serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    """Serializa el estado para guardado JSON, convirtiendo tipos no serializables."""
    serializable: dict[str, Any] = {}
    for key, value in state.items():
        try:
            json.dumps(value)
            serializable[key] = value
        except (TypeError, ValueError):
            serializable[key] = str(value)
    return serializable
