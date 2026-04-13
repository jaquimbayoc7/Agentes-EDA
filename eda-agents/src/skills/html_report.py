"""Skill — Generador de reporte HTML dinamico.

Produce un archivo HTML auto-contenido con:
- Navegacion lateral por secciones
- Figuras embebidas en base64
- Tablas interactivas (ordennbles)
- Tema claro/oscuro
- decision.json renderizado

Usada como post-procesamiento en main.py despues del pipeline.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any


def _encode_image_base64(path: str) -> str:
    """Lee una imagen y retorna su representacion base64."""
    p = Path(path)
    if not p.exists():
        return ""
    data = p.read_bytes()
    ext = p.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "svg": "image/svg+xml", "gif": "image/gif"}.get(ext, "image/png")
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _build_figures_html(figures: list[dict], output_dir: Path) -> str:
    """Genera HTML con todas las figuras embebidas."""
    if not figures:
        return "<p>No se generaron figuras.</p>"
    cards: list[str] = []
    for fig in figures:
        fig_path = fig.get("path", "")
        if not fig_path:
            continue
        # Intentar ruta absoluta o relativa al output_dir
        p = Path(fig_path)
        if not p.is_absolute():
            p = output_dir / fig_path
        if not p.exists():
            # Intentar dentro de figures/
            p = output_dir / "figures" / fig.get("name", "")
        b64 = _encode_image_base64(str(p))
        if not b64:
            continue
        desc = fig.get("description", fig.get("name", ""))
        cards.append(
            f'<div class="fig-card">'
            f'<img src="{b64}" alt="{desc}" loading="lazy">'
            f'<p class="fig-caption">{desc}</p>'
            f'</div>'
        )
    return "\n".join(cards) if cards else "<p>No se pudieron cargar las figuras.</p>"


def _build_json_table(data: dict) -> str:
    """Convierte un dict plano en una tabla HTML."""
    if not data:
        return "<p>N/A</p>"
    rows = []
    for k, v in data.items():
        val = json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else str(v)
        rows.append(f"<tr><td><strong>{k}</strong></td><td>{val}</td></tr>")
    return f'<table class="data-table"><thead><tr><th>Campo</th><th>Valor</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'


def _build_encoding_table(encoding_log: dict) -> str:
    """Tabla de encoding aplicado."""
    if not encoding_log:
        return "<p>Sin encoding aplicado.</p>"
    rows = []
    for col, info in encoding_log.items():
        if isinstance(info, dict):
            enc = info.get("encoding", "N/A")
            flag = info.get("flag", "")
            new_cols = ", ".join(info.get("new_cols", [])) or "-"
            rows.append(f"<tr><td>{col}</td><td>{flag}</td><td>{enc}</td><td>{new_cols}</td></tr>")
    if not rows:
        return "<p>Sin encoding aplicado.</p>"
    return (
        '<table class="data-table"><thead><tr><th>Columna</th><th>Flag</th>'
        f'<th>Encoding</th><th>Nuevas columnas</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def _build_models_table(models: list) -> str:
    """Tabla de modelos recomendados."""
    if not models:
        return "<p>N/A</p>"
    rows = []
    for m in models:
        name = m.get("name", "N/A") if isinstance(m, dict) else str(m)
        reason = m.get("reason", "") if isinstance(m, dict) else ""
        rows.append(f"<tr><td><strong>{name}</strong></td><td>{reason}</td></tr>")
    return (
        '<table class="data-table"><thead><tr><th>Modelo</th><th>Razon</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def _build_profile_table(perfil: dict) -> str:
    """Tabla de perfil de columnas."""
    if not perfil:
        return "<p>N/A</p>"
    rows = []
    for col, info in perfil.items():
        if isinstance(info, dict):
            dtype = info.get("dtype", "?")
            nunique = info.get("n_unique", "?")
            null_pct = info.get("null_pct", 0)
            rows.append(f"<tr><td>{col}</td><td>{dtype}</td><td>{nunique}</td><td>{null_pct:.1f}%</td></tr>")
    if not rows:
        return "<p>N/A</p>"
    return (
        '<table class="data-table sortable"><thead><tr><th>Columna</th><th>Tipo</th>'
        f'<th>Unicos</th><th>Nulos %</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


# ---------------------------------------------------------------------------
# CSS + JS
# ---------------------------------------------------------------------------

_CSS = """
:root {
    --bg: #f8f9fa; --bg2: #ffffff; --text: #212529; --accent: #3498db;
    --accent2: #2980b9; --sidebar-bg: #2c3e50; --sidebar-text: #ecf0f1;
    --border: #dee2e6; --card-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
[data-theme="dark"] {
    --bg: #1a1a2e; --bg2: #16213e; --text: #e0e0e0; --accent: #64b5f6;
    --accent2: #42a5f5; --sidebar-bg: #0f3460; --sidebar-text: #e0e0e0;
    --border: #333; --card-shadow: 0 2px 8px rgba(0,0,0,0.4);
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', Tahoma, Geneva, sans-serif; background: var(--bg);
       color: var(--text); display: flex; min-height: 100vh; }
.sidebar { width: 260px; background: var(--sidebar-bg); color: var(--sidebar-text);
           padding: 20px 0; position: fixed; height: 100vh; overflow-y: auto; z-index: 10; }
.sidebar h2 { padding: 0 20px 15px; font-size: 1.1rem; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 10px; }
.sidebar a { display: block; padding: 8px 20px; color: var(--sidebar-text); text-decoration: none;
             font-size: 0.9rem; transition: background 0.2s; }
.sidebar a:hover, .sidebar a.active { background: rgba(255,255,255,0.1); border-left: 3px solid var(--accent); }
.main { margin-left: 260px; padding: 30px 40px; flex: 1; max-width: 1100px; }
.header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px;
          padding-bottom: 15px; border-bottom: 2px solid var(--accent); }
.header h1 { font-size: 1.6rem; }
.theme-toggle { cursor: pointer; padding: 8px 16px; border: 1px solid var(--border);
                border-radius: 6px; background: var(--bg2); color: var(--text); font-size: 0.85rem; }
section { background: var(--bg2); border-radius: 10px; padding: 25px; margin-bottom: 25px;
          box-shadow: var(--card-shadow); }
section h2 { color: var(--accent); font-size: 1.25rem; margin-bottom: 15px;
             padding-bottom: 8px; border-bottom: 1px solid var(--border); }
section h3 { color: var(--accent2); margin: 15px 0 8px; }
.data-table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 0.9rem; }
.data-table th { background: var(--accent); color: white; padding: 10px 12px; text-align: left; cursor: pointer; }
.data-table th:hover { background: var(--accent2); }
.data-table td { padding: 8px 12px; border-bottom: 1px solid var(--border); }
.data-table tr:hover { background: rgba(52,152,219,0.05); }
.fig-card { display: inline-block; margin: 10px; text-align: center; vertical-align: top; }
.fig-card img { max-width: 480px; border-radius: 8px; box-shadow: var(--card-shadow); cursor: pointer;
                transition: transform 0.2s; }
.fig-card img:hover { transform: scale(1.02); }
.fig-caption { font-size: 0.85rem; color: var(--accent2); margin-top: 5px; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
.badge-ok { background: #d4edda; color: #155724; }
.badge-error { background: #f8d7da; color: #721c24; }
.badge-fallback { background: #fff3cd; color: #856404; }
.kpi-row { display: flex; gap: 15px; flex-wrap: wrap; margin: 15px 0; }
.kpi { background: var(--bg); border-radius: 8px; padding: 15px 20px; min-width: 150px;
       text-align: center; box-shadow: var(--card-shadow); }
.kpi .value { font-size: 1.8rem; font-weight: 700; color: var(--accent); }
.kpi .label { font-size: 0.8rem; color: var(--text); opacity: 0.7; }
pre { background: var(--bg); padding: 12px; border-radius: 6px; overflow-x: auto;
      font-size: 0.85rem; border: 1px solid var(--border); }
.modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
         background: rgba(0,0,0,0.8); z-index: 100; justify-content: center; align-items: center; }
.modal img { max-width: 90%; max-height: 90%; border-radius: 8px; }
.modal.show { display: flex; }
ul, ol { margin-left: 20px; }
li { margin-bottom: 5px; }
@media (max-width: 768px) {
    .sidebar { display: none; }
    .main { margin-left: 0; padding: 15px; }
    .fig-card img { max-width: 100%; }
}
"""

_JS = """
function toggleTheme() {
    const body = document.documentElement;
    const current = body.getAttribute('data-theme') || 'light';
    body.setAttribute('data-theme', current === 'light' ? 'dark' : 'light');
    localStorage.setItem('theme', current === 'light' ? 'dark' : 'light');
}
(function() {
    const saved = localStorage.getItem('theme');
    if (saved) document.documentElement.setAttribute('data-theme', saved);
})();

// Scroll spy for sidebar
document.addEventListener('DOMContentLoaded', function() {
    const sections = document.querySelectorAll('section[id]');
    const links = document.querySelectorAll('.sidebar a');
    window.addEventListener('scroll', function() {
        let current = '';
        sections.forEach(s => { if (window.scrollY >= s.offsetTop - 100) current = s.id; });
        links.forEach(a => {
            a.classList.remove('active');
            if (a.getAttribute('href') === '#' + current) a.classList.add('active');
        });
    });
});

// Sortable tables
document.addEventListener('click', function(e) {
    if (e.target.tagName === 'TH') {
        const th = e.target;
        const table = th.closest('table');
        const idx = Array.from(th.parentNode.children).indexOf(th);
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const asc = th.dataset.sort !== 'asc';
        th.dataset.sort = asc ? 'asc' : 'desc';
        rows.sort((a, b) => {
            const av = a.children[idx]?.textContent || '';
            const bv = b.children[idx]?.textContent || '';
            const an = parseFloat(av), bn = parseFloat(bv);
            if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
            return asc ? av.localeCompare(bv) : bv.localeCompare(av);
        });
        rows.forEach(r => tbody.appendChild(r));
    }
});

// Image modal
document.addEventListener('click', function(e) {
    if (e.target.matches('.fig-card img')) {
        const modal = document.getElementById('imgModal');
        document.getElementById('modalImg').src = e.target.src;
        modal.classList.add('show');
    }
});
function closeModal() { document.getElementById('imgModal').classList.remove('show'); }
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_html_report(state: dict[str, Any], output_dir: str | Path) -> Path:
    """Genera un reporte HTML dinamico completo.

    Parameters
    ----------
    state : dict
        Estado final del pipeline (o state_final.json cargado).
    output_dir : str or Path
        Directorio base de outputs del run (ej: outputs/<run_id>).

    Returns
    -------
    Path al archivo HTML generado.
    """
    output_dir = Path(output_dir)
    report_dir = output_dir / "reportesFinales"
    report_dir.mkdir(parents=True, exist_ok=True)

    run_id = state.get("run_id", "unknown")
    question = state.get("research_question", "N/A")
    data_type = state.get("data_type", "N/A")
    target = state.get("target", "N/A")
    dataset_size = state.get("dataset_size", 0)
    tarea = state.get("tarea_sugerida", "N/A")
    model_family = state.get("model_family", "N/A")
    metrica = state.get("metrica_principal", "N/A")
    hyper = state.get("hyperparams_technique", "N/A")
    figures = state.get("figures", [])
    agent_status = state.get("agent_status", {})

    # KPIs
    desbalance = state.get("desbalance_ratio")
    desbalance_str = f"{desbalance:.2f}" if desbalance else "N/A"
    n_features = len(state.get("perfil_columnas", {}))
    n_figures = len(figures)

    # Agent status badges
    status_html = " ".join(
        f'<span class="badge badge-{v}">{k}: {v}</span>'
        for k, v in sorted(agent_status.items())
    )

    # Hipotesis
    hip = state.get("hipotesis") or {}
    hip_html = (
        f"<ul>"
        f"<li><strong>H1 (confirmatoria):</strong> {hip.get('h1', 'N/A')}</li>"
        f"<li><strong>H2 (exploratoria):</strong> {hip.get('h2', 'N/A')}</li>"
        f"<li><strong>H3 (alternativa):</strong> {hip.get('h3', 'N/A')}</li>"
        f"</ul>"
    )

    # Referencias
    refs = state.get("refs", [])
    if refs:
        refs_html = "<ul>" + "".join(
            f"<li>{r.get('title', 'N/A')} (DOI: {r.get('doi', 'N/A')})</li>"
            for r in refs
        ) + "</ul>"
    else:
        refs_html = "<p>No se encontraron referencias.</p>"

    # Hallazgos
    hallazgos = state.get("hallazgos_eda", {})
    hallazgos_json = json.dumps(hallazgos, indent=2, default=str, ensure_ascii=False) if hallazgos else "N/A"

    # TS
    if state.get("flag_timeseries"):
        ts_html = (
            f"<p><strong>Modelo:</strong> {json.dumps(state.get('modelo_ts', {}), default=str)}</p>"
            f"<p><strong>Parametros:</strong> {json.dumps(state.get('params_pdq', {}), default=str)}</p>"
        )
    else:
        ts_html = "<p>No aplica (datos tabulares).</p>"

    # Advertencias
    warns = state.get("advertencias", [])
    warns_html = "<ul>" + "".join(f"<li>{w}</li>" for w in warns) + "</ul>" if warns else "<p>Ninguna.</p>"

    # Sidebar links
    nav_items = [
        ("resumen", "Resumen"),
        ("literatura", "Literatura"),
        ("hipotesis", "Hipotesis"),
        ("dataset", "Dataset"),
        ("perfil", "Perfil de Columnas"),
        ("preprocesamiento", "Preprocesamiento"),
        ("hallazgos", "Hallazgos EDA"),
        ("timeseries", "Series de Tiempo"),
        ("decision", "Decision"),
        ("modelos", "Modelos"),
        ("figuras", "Visualizaciones"),
        ("advertencias", "Advertencias"),
        ("pasos", "Proximos Pasos"),
        ("agentes", "Estado Agentes"),
    ]
    nav_html = "\n".join(f'<a href="#{sid}">{label}</a>' for sid, label in nav_items)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EDA Report - {run_id}</title>
<style>{_CSS}</style>
</head>
<body>

<nav class="sidebar">
<h2>EDA Agents</h2>
<p style="padding: 0 20px; font-size: 0.8rem; opacity: 0.7;">Run: {run_id}</p>
{nav_html}
</nav>

<div class="main">

<div class="header">
<h1>Reporte EDA - {run_id}</h1>
<button class="theme-toggle" onclick="toggleTheme()">Tema claro/oscuro</button>
</div>

<section id="resumen">
<h2>Resumen Ejecutivo</h2>
<p><strong>Pregunta:</strong> {question}</p>
<div class="kpi-row">
<div class="kpi"><div class="value">{dataset_size}</div><div class="label">Filas</div></div>
<div class="kpi"><div class="value">{n_features}</div><div class="label">Columnas</div></div>
<div class="kpi"><div class="value">{tarea}</div><div class="label">Tarea</div></div>
<div class="kpi"><div class="value">{model_family}</div><div class="label">Familia</div></div>
<div class="kpi"><div class="value">{metrica}</div><div class="label">Metrica</div></div>
<div class="kpi"><div class="value">{n_figures}</div><div class="label">Figuras</div></div>
</div>
<p><strong>Tipo de datos:</strong> {data_type} | <strong>Target:</strong> {target} | <strong>Desbalance:</strong> {desbalance_str}</p>
</section>

<section id="literatura">
<h2>Revision de Literatura</h2>
<h3>Ecuaciones PICO</h3>
<ul>{"".join(f"<li><code>{eq}</code></li>" for eq in state.get('search_equations', []))}</ul>
<h3>Referencias</h3>
{refs_html}
</section>

<section id="hipotesis">
<h2>Hipotesis</h2>
{hip_html}
</section>

<section id="dataset">
<h2>Descripcion del Dataset</h2>
{_build_json_table({"Filas": dataset_size, "Target": target, "Tipo": data_type,
                     "Desbalance": desbalance_str, "Serie temporal": state.get('flag_timeseries', False)})}
</section>

<section id="perfil">
<h2>Perfil de Columnas</h2>
{_build_profile_table(state.get('perfil_columnas', {}))}
</section>

<section id="preprocesamiento">
<h2>Preprocesamiento</h2>
<h3>Encoding</h3>
{_build_encoding_table(state.get('encoding_log', {}))}
<h3>Features nuevas</h3>
<p>{state.get('features_nuevas', []) or 'Ninguna'}</p>
<h3>Balanceo</h3>
<pre>{json.dumps(state.get('balanceo_log', {}), indent=2, ensure_ascii=False)}</pre>
</section>

<section id="hallazgos">
<h2>Hallazgos EDA Tabular</h2>
<pre>{hallazgos_json}</pre>
</section>

<section id="timeseries">
<h2>Series de Tiempo</h2>
{ts_html}
</section>

<section id="decision">
<h2>Decision de Tarea</h2>
{_build_json_table({"Tarea": tarea, "Model family": model_family,
                     "Tecnica hiperparametros": hyper, "Metrica principal": metrica})}
</section>

<section id="modelos">
<h2>Modelos Recomendados</h2>
{_build_models_table(state.get('modelos_recomendados', []))}
</section>

<section id="figuras">
<h2>Visualizaciones</h2>
<div>{_build_figures_html(figures, output_dir)}</div>
</section>

<section id="advertencias">
<h2>Advertencias y Limitaciones</h2>
{warns_html}
</section>

<section id="pasos">
<h2>Proximos Pasos</h2>
<ol>
<li>Entrenar modelos recomendados con los hiperparametros sugeridos</li>
<li>Validar con cross-validation sobre train set</li>
<li>Evaluar en test set con metricas seleccionadas</li>
<li>Iterar segun resultados</li>
</ol>
</section>

<section id="agentes">
<h2>Estado de Agentes</h2>
<p>{status_html}</p>
</section>

</div>

<div class="modal" id="imgModal" onclick="closeModal()">
<img id="modalImg" src="" alt="Zoom">
</div>

<script>{_JS}</script>
</body>
</html>"""

    html_path = report_dir / "reporte_eda.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path
