"""Skill — Generador de reporte HTML dinamico.

Produce un archivo HTML auto-contenido con:
- Navegacion lateral por secciones
- Figuras embebidas en base64 con export 400 DPI
- Tablas interactivas (ordennbles)
- Tema claro/oscuro
- Hallazgos EDA formateados (no JSON)
- VIF, Breusch-Pagan, Normalidad secciones dedicadas
- Botones de descarga para train/test datasets
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
    """Genera HTML con todas las figuras embebidas (PNG base64 + Plotly iframe) + export 400 DPI."""
    if not figures:
        return "<p>No se generaron figuras.</p>"
    cards: list[str] = []
    fig_idx = 0
    for fig in figures:
        fig_path = fig.get("path", "")
        fmt = fig.get("format", "png")
        if not fig_path:
            continue

        desc = fig.get("description", fig.get("name", ""))

        if fmt == "html":
            # Embed Plotly HTML inline via iframe srcdoc
            p = Path(fig_path)
            if not p.is_absolute():
                p = output_dir / fig_path
            if not p.exists():
                p = output_dir / "figures" / fig.get("name", "")
            if p.exists():
                html_content = p.read_text(encoding="utf-8")
                # Escape for srcdoc attribute
                escaped = html_content.replace("&", "&amp;").replace('"', "&quot;")
                cards.append(
                    f'<div class="fig-card fig-interactive">'
                    f'<iframe id="plotly-frame-{fig_idx}" srcdoc="{escaped}" '
                    f'style="width:100%;height:500px;border:none;" loading="lazy"></iframe>'
                    f'<div class="fig-actions">'
                    f'<p class="fig-caption">{desc} (interactivo)</p>'
                    f'<button class="btn-export" onclick="exportPlotly400DPI(\'plotly-frame-{fig_idx}\', \'{fig.get("name", "figure")}\')">Exportar PNG 400 DPI</button>'
                    f'</div></div>'
                )
                fig_idx += 1
            continue

        # PNG/JPEG: base64 embed
        p = Path(fig_path)
        if not p.is_absolute():
            p = output_dir / fig_path
        if not p.exists():
            p = output_dir / "figures" / fig.get("name", "")
        b64 = _encode_image_base64(str(p))
        if not b64:
            continue
        safe_name = fig.get("name", "figure").replace(".png", "")
        cards.append(
            f'<div class="fig-card">'
            f'<img src="{b64}" alt="{desc}" loading="lazy">'
            f'<div class="fig-actions">'
            f'<p class="fig-caption">{desc}</p>'
            f'<a class="btn-export" href="{b64}" download="{safe_name}_400dpi.png">Descargar PNG</a>'
            f'</div></div>'
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


def _build_feature_importance_html(feat_imp: dict) -> str:
    """Genera tabla HTML con ranking de feature importance."""
    if not feat_imp:
        return "<p>No se calculo feature importance.</p>"

    mi = feat_imp.get("mutual_information", {})
    perm = feat_imp.get("permutation_importance", {})
    top = feat_imp.get("top_features", [])

    parts: list[str] = []

    if top:
        parts.append("<h3>Top Features (ranking combinado)</h3><ol>")
        for f in top:
            parts.append(f"<li><strong>{f}</strong></li>")
        parts.append("</ol>")

    if mi:
        rows = []
        for col, score in mi.items():
            rows.append(f"<tr><td>{col}</td><td>{score:.6f}</td></tr>")
        parts.append(
            '<h3>Mutual Information</h3>'
            '<table class="data-table"><thead><tr><th>Feature</th><th>MI Score</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
        )

    if perm:
        rows = []
        for col, info in perm.items():
            if isinstance(info, dict):
                rows.append(
                    f"<tr><td>{col}</td><td>{info.get('mean', 0):.6f}</td>"
                    f"<td>{info.get('std', 0):.6f}</td></tr>"
                )
        parts.append(
            '<h3>Permutation Importance</h3>'
            '<table class="data-table"><thead><tr><th>Feature</th><th>Mean</th><th>Std</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
        )

    return "\n".join(parts) if parts else "<p>No data.</p>"


def _build_refs_html(refs: list) -> str:
    """Genera HTML para referencias con DOI y enlaces al artículo."""
    if not refs:
        return "<p>No se encontraron referencias.</p>"

    cards: list[str] = []
    for i, r in enumerate(refs, 1):
        title = r.get("title", "Sin título")
        authors = r.get("authors", "")
        year = r.get("year", "")
        doi = r.get("doi", "")
        url = r.get("url", "")
        key_finding = r.get("key_finding", "")
        relevance = r.get("relevance", "")
        source = r.get("source", "")

        links_parts: list[str] = []
        if doi:
            doi_url = f"https://doi.org/{doi}" if not doi.startswith("http") else doi
            links_parts.append(
                f'<a href="{doi_url}" target="_blank" rel="noopener">DOI: {doi}</a>'
            )
        if url:
            links_parts.append(
                f'<a href="{url}" target="_blank" rel="noopener">Ver artículo ↗</a>'
            )
        links_html = " &nbsp;|&nbsp; ".join(links_parts) if links_parts else "<em>Sin enlace disponible</em>"

        source_badge = ""
        if source == "tavily":
            source_badge = ' <span class="badge badge-ok">Web</span>'
        elif source == "claude":
            source_badge = ' <span class="badge badge-fallback">Académico</span>'

        meta_parts = []
        if authors:
            meta_parts.append(authors)
        if year:
            meta_parts.append(f"({year})")
        meta_line = " ".join(meta_parts)

        finding_html = f'<p class="ref-finding">{key_finding}</p>' if key_finding else ""
        relevance_html = f'<p class="ref-relevance"><em>{relevance}</em></p>' if relevance else ""
        meta_html = f'<p class="ref-meta">{meta_line}</p>' if meta_line else ""

        cards.append(
            f'<div class="ref-card">'
            f'<div class="ref-number">{i}</div>'
            f'<div class="ref-content">'
            f'<h4>{title}{source_badge}</h4>'
            f'{meta_html}'
            f'{finding_html}'
            f'{relevance_html}'
            f'<div class="ref-links">{links_html}</div>'
            f'</div></div>'
        )

    return "\n".join(cards)


def _build_hallazgos_html(hallazgos: dict) -> str:
    """Formatea hallazgos EDA en secciones legibles (no JSON)."""
    if not hallazgos:
        return "<p>No se generaron hallazgos.</p>"

    parts: list[str] = []

    # Interpretación (texto libre de Claude)
    interp = hallazgos.get("interpretation", "")
    if interp:
        parts.append(
            f'<div class="hallazgo-card hallazgo-interpretation">'
            f'<h3>🔍 Interpretación General</h3>'
            f'<p>{interp}</p></div>'
        )

    # KPI resumen rápido
    normality = hallazgos.get("normality", {})
    outliers_data = hallazgos.get("outliers", {})
    vif_summary = hallazgos.get("vif_summary", {})
    fi = hallazgos.get("feature_importance", {})
    top = fi.get("top_features", [])

    n_normal = sum(1 for v in normality.values() if isinstance(v, dict) and v.get("normal"))
    n_not_normal = sum(1 for v in normality.values() if isinstance(v, dict) and not v.get("normal"))
    n_outlier_vars = sum(1 for v in outliers_data.values() if isinstance(v, dict) and v.get("pct", 0) > 1)
    n_vif_flagged = vif_summary.get("n_flagged", 0)

    parts.append(
        '<div class="kpi-row">'
        f'<div class="kpi"><div class="value">{n_normal}</div><div class="label">Vars normales</div></div>'
        f'<div class="kpi"><div class="value">{n_not_normal}</div><div class="label">Vars no normales</div></div>'
        f'<div class="kpi"><div class="value">{n_outlier_vars}</div><div class="label">Vars con outliers &gt;1%</div></div>'
        f'<div class="kpi"><div class="value">{n_vif_flagged}</div><div class="label">Vars VIF alto</div></div>'
        f'<div class="kpi"><div class="value">{len(top)}</div><div class="label">Top features</div></div>'
        '</div>'
    )

    # Correlaciones
    corr = hallazgos.get("correlations", {})
    if corr:
        spearman = corr.get("spearman", {})
        if spearman:
            # Find strong correlations (|r| > 0.5, non-diagonal)
            strong: list[str] = []
            seen: set[tuple[str, str]] = set()
            for row_name, row_vals in spearman.items():
                for col_name, val in row_vals.items():
                    if row_name == col_name:
                        continue
                    pair = tuple(sorted((row_name, col_name)))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    if abs(val) > 0.5:
                        direction = "positiva" if val > 0 else "negativa"
                        strong.append(
                            f"<li><strong>{row_name}</strong> ↔ <strong>{col_name}</strong>: "
                            f"r = {val:.3f} (correlación {direction} {'fuerte' if abs(val) > 0.7 else 'moderada'})</li>"
                        )
            if strong:
                parts.append(
                    '<div class="hallazgo-card">'
                    '<h3>📊 Correlaciones Significativas (Spearman |r| &gt; 0.5)</h3>'
                    f'<ul>{"".join(strong[:15])}</ul></div>'
                )
            else:
                parts.append(
                    '<div class="hallazgo-card"><h3>📊 Correlaciones</h3>'
                    '<p>No se encontraron correlaciones fuertes (|r| &gt; 0.5) entre variables.</p></div>'
                )

    # Outliers
    outliers = hallazgos.get("outliers", {})
    if outliers:
        rows = []
        for col, info in outliers.items():
            n_out = info.get("n_outliers", 0)
            pct = info.get("pct", 0)
            severity = "alta" if pct > 5 else "moderada" if pct > 1 else "baja"
            color_class = "badge-error" if pct > 5 else "badge-fallback" if pct > 1 else "badge-ok"
            rows.append(
                f"<tr><td>{col}</td><td>{n_out}</td><td>{pct:.1f}%</td>"
                f'<td><span class="badge {color_class}">{severity}</span></td></tr>'
            )
        parts.append(
            '<div class="hallazgo-card">'
            '<h3>⚠️ Outliers (IQR 1.5×)</h3>'
            '<table class="data-table"><thead><tr><th>Variable</th><th>N Outliers</th>'
            f'<th>Porcentaje</th><th>Severidad</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        )

    # Feature importance summary
    if top:
        parts.append(
            '<div class="hallazgo-card">'
            '<h3>🏆 Top Features (Ranking Combinado MI + Permutation)</h3><ol>' +
            "".join(f"<li><strong>{f}</strong></li>" for f in top) +
            '</ol></div>'
        )

    if not parts:
        return "<p>No se generaron hallazgos relevantes.</p>"

    return "\n".join(parts)


def _build_normality_html(hallazgos: dict) -> str:
    """Genera tabla HTML con resultados de tests de normalidad."""
    normality = hallazgos.get("normality", {}) if hallazgos else {}
    if not normality:
        return "<p>No se ejecutaron tests de normalidad.</p>"

    rows = []
    for col, info in normality.items():
        test_name = info.get("test", "shapiro").capitalize()
        stat = info.get("statistic", 0)
        pval = info.get("p_value")
        is_normal = info.get("normal")

        if pval is not None:
            verdict = "Normal" if is_normal else "No normal"
            color_class = "badge-ok" if is_normal else "badge-error"
            rows.append(
                f"<tr><td>{col}</td><td>{test_name}</td><td>{stat:.4f}</td>"
                f'<td>{pval:.4f}</td><td><span class="badge {color_class}">{verdict}</span></td></tr>'
            )
        else:
            # Anderson-Darling (no p-value)
            rows.append(
                f"<tr><td>{col}</td><td>{test_name}</td><td>{stat:.4f}</td>"
                f"<td>N/A</td><td>Ver critical values</td></tr>"
            )

    return (
        '<table class="data-table sortable"><thead><tr>'
        '<th>Variable</th><th>Test</th><th>Estadístico</th>'
        f'<th>p-valor</th><th>Resultado</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def _build_vif_html(vif_all: dict, vif_flags: list) -> str:
    """Tabla HTML con valores VIF."""
    if not vif_all:
        return "<p>No se calcularon valores VIF.</p>"

    rows = []
    flagged_cols = {f["column"] for f in vif_flags} if vif_flags else set()
    for col, vif_val in sorted(vif_all.items(), key=lambda x: x[1], reverse=True):
        display_val = f"{vif_val:.2f}" if vif_val < 9999 else "∞ (colinealidad perfecta)"
        if vif_val > 10:
            severity = "alta"
            color_class = "badge-error"
        elif vif_val > 5:
            severity = "moderada"
            color_class = "badge-fallback"
        else:
            severity = "baja"
            color_class = "badge-ok"
        rows.append(
            f"<tr><td>{col}</td><td>{display_val}</td>"
            f'<td><span class="badge {color_class}">{severity}</span></td></tr>'
        )

    n_flagged = len(flagged_cols)
    summary = (
        f"<p><strong>{n_flagged} variable(s)</strong> con VIF &gt; 10 (multicolinealidad alta). "
        f"Considerar eliminación o reducción de dimensionalidad.</p>"
        if n_flagged > 0
        else "<p>Ninguna variable supera el umbral VIF = 10. No hay multicolinealidad crítica.</p>"
    )

    return (
        summary +
        '<table class="data-table sortable"><thead><tr>'
        '<th>Variable</th><th>VIF</th><th>Multicolinealidad</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def _build_bp_html(bp_result: dict | None, correccion: str | None) -> str:
    """Formatea resultados del test de Breusch-Pagan."""
    if not bp_result or bp_result.get("error"):
        return "<p>Test de Breusch-Pagan no aplicable (solo regresión) o no ejecutado.</p>"

    hetero = bp_result.get("heteroscedastic", False)
    bp_stat = bp_result.get("bp_statistic", 0)
    bp_p = bp_result.get("bp_pvalue", 0)
    f_stat = bp_result.get("f_statistic", 0)
    f_p = bp_result.get("f_pvalue", 0)

    verdict_class = "badge-error" if hetero else "badge-ok"
    verdict_text = "Heteroscedástico" if hetero else "Homoscedástico"

    html = (
        f'<div class="kpi-row">'
        f'<div class="kpi"><div class="value">{bp_stat:.2f}</div><div class="label">BP Statistic</div></div>'
        f'<div class="kpi"><div class="value">{bp_p:.4f}</div><div class="label">BP p-valor</div></div>'
        f'<div class="kpi"><div class="value">{f_stat:.2f}</div><div class="label">F Statistic</div></div>'
        f'<div class="kpi"><div class="value">{f_p:.4f}</div><div class="label">F p-valor</div></div>'
        f'<div class="kpi"><div class="value"><span class="badge {verdict_class}">{verdict_text}</span></div>'
        f'<div class="label">Resultado (α=0.05)</div></div>'
        f'</div>'
    )

    if hetero:
        html += (
            '<div class="alert alert-warning">'
            '<strong>Se detectó heteroscedasticidad.</strong> '
            'Los errores estándar OLS pueden ser sesgados. '
        )
        if correccion:
            html += f'Corrección sugerida: <strong>{correccion}</strong> '
            if correccion == "WLS":
                html += "(Weighted Least Squares)."
            elif correccion == "GLS":
                html += "(Generalized Least Squares — también hay multicolinealidad alta)."
            else:
                html += "(HC3 robust standard errors)."
        html += '</div>'
    else:
        html += (
            '<div class="alert alert-ok">'
            'No se detectó heteroscedasticidad significativa. '
            'Los errores estándar OLS son válidos.'
            '</div>'
        )

    return html


def _build_download_buttons(state: dict, output_dir: Path) -> str:
    """Genera botones de descarga para train y test datasets."""
    parts: list[str] = []

    train_final = state.get("dataset_train_final", "")
    test_final = state.get("dataset_test_final", "")
    train_prov = state.get("dataset_train_provisional", "")
    test_proc = state.get("dataset_test_procesado", "")

    datasets = []
    if train_final:
        datasets.append(("Train (Final)", train_final))
    elif train_prov:
        datasets.append(("Train (Provisional)", train_prov))

    if test_final:
        datasets.append(("Test (Final)", test_final))
    elif test_proc:
        datasets.append(("Test (Procesado)", test_proc))

    if not datasets:
        return "<p>No hay datasets disponibles para descarga.</p>"

    for label, path in datasets:
        p = Path(path)
        if not p.exists():
            continue
        try:
            csv_data = p.read_bytes()
            b64 = base64.b64encode(csv_data).decode("ascii")
            fname = p.name
            parts.append(
                f'<a class="btn-download" href="data:text/csv;base64,{b64}" '
                f'download="{fname}">'
                f'⬇ Descargar {label} ({fname})</a>'
            )
        except Exception:
            continue

    if not parts:
        return "<p>No se pudieron preparar los datasets para descarga.</p>"

    return '<div class="download-row">' + "\n".join(parts) + '</div>'


# ---------------------------------------------------------------------------
# CSS + JS
# ---------------------------------------------------------------------------

_CSS = """
:root {
    --bg: #f0f2f5; --bg2: #ffffff; --text: #212529; --accent: #2563eb;
    --accent2: #1d4ed8; --accent-light: #dbeafe; --sidebar-bg: #1e293b;
    --sidebar-text: #e2e8f0; --border: #e2e8f0; --card-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08);
    --success: #16a34a; --success-bg: #dcfce7; --warning: #d97706; --warning-bg: #fef3c7;
    --danger: #dc2626; --danger-bg: #fee2e2; --radius: 12px;
}
[data-theme="dark"] {
    --bg: #0f172a; --bg2: #1e293b; --text: #e2e8f0; --accent: #60a5fa;
    --accent2: #93c5fd; --accent-light: #1e3a5f; --sidebar-bg: #0f172a;
    --sidebar-text: #e2e8f0; --border: #334155; --card-shadow: 0 1px 3px rgba(0,0,0,0.4);
    --success: #4ade80; --success-bg: #14532d; --warning: #fbbf24; --warning-bg: #451a03;
    --danger: #f87171; --danger-bg: #450a0a;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
       background: var(--bg); color: var(--text); display: flex; min-height: 100vh;
       line-height: 1.6; }
.sidebar { width: 280px; background: var(--sidebar-bg); color: var(--sidebar-text);
           padding: 24px 0; position: fixed; height: 100vh; overflow-y: auto; z-index: 10;
           border-right: 1px solid rgba(255,255,255,0.05); }
.sidebar h2 { padding: 0 24px 16px; font-size: 1.1rem; letter-spacing: 0.5px;
              border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 12px; }
.sidebar a { display: block; padding: 10px 24px; color: var(--sidebar-text); text-decoration: none;
             font-size: 0.88rem; transition: all 0.2s; border-left: 3px solid transparent; }
.sidebar a:hover, .sidebar a.active { background: rgba(255,255,255,0.08);
    border-left-color: var(--accent); color: var(--accent); }
.main { margin-left: 280px; padding: 32px 48px; flex: 1; max-width: 1140px; }
.header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 32px;
          padding-bottom: 16px; border-bottom: 2px solid var(--accent); }
.header h1 { font-size: 1.5rem; font-weight: 700; }
.theme-toggle { cursor: pointer; padding: 8px 18px; border: 1px solid var(--border);
                border-radius: 8px; background: var(--bg2); color: var(--text);
                font-size: 0.85rem; transition: all 0.2s; }
.theme-toggle:hover { border-color: var(--accent); }
section { background: var(--bg2); border-radius: var(--radius); padding: 28px; margin-bottom: 24px;
          box-shadow: var(--card-shadow); border: 1px solid var(--border); }
section h2 { color: var(--accent); font-size: 1.2rem; margin-bottom: 18px; font-weight: 700;
             padding-bottom: 10px; border-bottom: 2px solid var(--accent-light); }
section h3 { color: var(--accent2); margin: 18px 0 10px; font-size: 1rem; font-weight: 600; }
.data-table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.88rem; }
.data-table th { background: var(--accent); color: white; padding: 10px 14px; text-align: left;
                 cursor: pointer; font-weight: 600; font-size: 0.82rem; text-transform: uppercase;
                 letter-spacing: 0.5px; }
.data-table th:hover { background: var(--accent2); }
.data-table td { padding: 10px 14px; border-bottom: 1px solid var(--border); }
.data-table tr:hover { background: var(--accent-light); }
.fig-card { display: inline-block; margin: 10px; text-align: center; vertical-align: top; }
.fig-card img { max-width: 480px; border-radius: 8px; box-shadow: var(--card-shadow); cursor: pointer;
                transition: transform 0.2s; }
.fig-card img:hover { transform: scale(1.02); }
.fig-interactive { width: 100%; margin: 14px 0; display: block; }
.fig-interactive iframe { border-radius: 8px; box-shadow: var(--card-shadow); }
.fig-actions { display: flex; align-items: center; justify-content: space-between; margin-top: 8px; gap: 10px; }
.fig-caption { font-size: 0.85rem; color: var(--accent2); }
.btn-export, .btn-download { display: inline-block; padding: 6px 16px; border-radius: 6px;
    font-size: 0.8rem; font-weight: 600; cursor: pointer; text-decoration: none;
    border: 1px solid var(--accent); color: var(--accent); background: transparent;
    transition: all 0.2s; white-space: nowrap; }
.btn-export:hover, .btn-download:hover { background: var(--accent); color: white; }
.download-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }
.btn-download { padding: 12px 24px; font-size: 0.9rem; border-radius: 8px;
    background: var(--accent); color: white; border: none; }
.btn-download:hover { background: var(--accent2); }
.badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; }
.badge-ok { background: var(--success-bg); color: var(--success); }
.badge-error { background: var(--danger-bg); color: var(--danger); }
.badge-fallback { background: var(--warning-bg); color: var(--warning); }
.kpi-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }
.kpi { background: var(--bg); border-radius: 10px; padding: 18px 22px; min-width: 140px;
       text-align: center; box-shadow: var(--card-shadow); border: 1px solid var(--border); }
.kpi .value { font-size: 1.6rem; font-weight: 700; color: var(--accent); }
.kpi .label { font-size: 0.78rem; color: var(--text); opacity: 0.65; margin-top: 4px; }
.hallazgo-card { background: var(--bg); border-radius: 10px; padding: 18px 22px;
    margin: 12px 0; border-left: 4px solid var(--accent); }
.hallazgo-card h3 { color: var(--accent); margin: 0 0 10px; font-size: 0.95rem; }
.alert { padding: 14px 20px; border-radius: 8px; margin: 12px 0; font-size: 0.9rem; }
.alert-warning { background: var(--warning-bg); border-left: 4px solid var(--warning); }
.alert-ok { background: var(--success-bg); border-left: 4px solid var(--success); }
.ref-card { display: flex; gap: 14px; background: var(--bg); border-radius: 10px; padding: 16px 20px;
    margin: 10px 0; border-left: 4px solid var(--accent); transition: box-shadow 0.2s; }
.ref-card:hover { box-shadow: var(--card-shadow); }
.ref-number { font-size: 1.3rem; font-weight: 700; color: var(--accent); min-width: 30px;
    display: flex; align-items: flex-start; justify-content: center; padding-top: 2px; }
.ref-content { flex: 1; }
.ref-content h4 { font-size: 0.95rem; margin-bottom: 4px; line-height: 1.4; }
.ref-meta { font-size: 0.82rem; color: var(--text); opacity: 0.7; margin-bottom: 6px; }
.ref-finding { font-size: 0.88rem; margin-bottom: 4px; }
.ref-relevance { font-size: 0.82rem; opacity: 0.75; margin-bottom: 6px; }
.ref-links { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 6px; }
.ref-links a { font-size: 0.82rem; color: var(--accent); text-decoration: none; font-weight: 600;
    padding: 3px 10px; border: 1px solid var(--accent); border-radius: 6px; transition: all 0.2s; }
.ref-links a:hover { background: var(--accent); color: white; }
.hallazgo-interpretation { border-left-color: var(--success); }
pre { background: var(--bg); padding: 14px; border-radius: 8px; overflow-x: auto;
      font-size: 0.85rem; border: 1px solid var(--border); }
.modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
         background: rgba(0,0,0,0.85); z-index: 100; justify-content: center; align-items: center; }
.modal img { max-width: 90%; max-height: 90%; border-radius: 8px; }
.modal.show { display: flex; }
ul, ol { margin-left: 22px; }
li { margin-bottom: 6px; }
@media (max-width: 768px) {
    .sidebar { display: none; }
    .main { margin-left: 0; padding: 16px; }
    .fig-card img { max-width: 100%; }
    .kpi-row { justify-content: center; }
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

// Export Plotly chart at 400 DPI
function exportPlotly400DPI(frameId, name) {
    try {
        var iframe = document.getElementById(frameId);
        var iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
        var plotDiv = iframeDoc.querySelector('.plotly-graph-div');
        if (plotDiv && iframe.contentWindow.Plotly) {
            iframe.contentWindow.Plotly.downloadImage(plotDiv, {
                format: 'png', width: 3200, height: 2400, scale: 3,
                filename: name.replace('.html', '') + '_400dpi'
            });
        } else {
            alert('Esperando carga de la gráfica. Intenta de nuevo en unos segundos.');
        }
    } catch(e) {
        alert('Error al exportar: ' + e.message);
    }
}

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
    refs_html = _build_refs_html(refs)

    # Hallazgos — formatted (no JSON)
    hallazgos = state.get("hallazgos_eda", {})

    # VIF data
    vif_all = state.get("vif_all", {})
    if not vif_all:
        vif_all = hallazgos.get("vif_all", {})
    vif_flags = state.get("vif_flags", [])

    # Breusch-Pagan data
    bp_result = state.get("breusch_pagan_result")
    correccion = state.get("modelo_correccion_heterosc")

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
        ("resumen", "Resumen Ejecutivo"),
        ("literatura", "Literatura"),
        ("hipotesis", "Hipotesis"),
        ("dataset", "Dataset"),
        ("descargas", "Descargar Datos"),
        ("perfil", "Perfil de Columnas"),
        ("preprocesamiento", "Preprocesamiento"),
        ("hallazgos", "Hallazgos EDA"),
        ("normalidad", "Normalidad"),
        ("vif", "VIF — Multicolinealidad"),
        ("breusch_pagan", "Breusch-Pagan"),
        ("importancia", "Feature Importance"),
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

<section id="descargas">
<h2>Descargar Datasets</h2>
<p>Datasets listos para modelamiento (con encoding y preprocesamiento aplicados):</p>
{_build_download_buttons(state, output_dir)}
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
{_build_hallazgos_html(hallazgos)}
</section>

<section id="normalidad">
<h2>Test de Normalidad</h2>
<p>Evaluacion de la distribucion normal de variables numericas (Shapiro-Wilk / Anderson-Darling, α = 0.05).</p>
{_build_normality_html(hallazgos)}
</section>

<section id="vif">
<h2>VIF — Factor de Inflacion de Varianza</h2>
<p>Analisis de multicolinealidad mediante VIF. Valores &gt; 10 indican multicolinealidad severa.</p>
{_build_vif_html(vif_all, vif_flags)}
</section>

<section id="breusch_pagan">
<h2>Test de Breusch-Pagan — Heteroscedasticidad</h2>
<p>Evalua si la varianza de los residuos es constante (homoscedasticidad) en el modelo de regresion.</p>
{_build_bp_html(bp_result, correccion)}
</section>

<section id="importancia">
<h2>Feature Importance</h2>
{_build_feature_importance_html(state.get('feature_importance', {}))}
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
