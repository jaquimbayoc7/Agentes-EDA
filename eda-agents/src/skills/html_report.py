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
    """Genera HTML grid de figuras con lightbox navigation + export 400 DPI."""
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
            p = Path(fig_path)
            if not p.is_absolute():
                p = output_dir / fig_path
            if not p.exists():
                p = output_dir / "figures" / fig.get("name", "")
            if p.exists():
                html_content = p.read_text(encoding="utf-8")
                escaped = html_content.replace("&", "&amp;").replace('"', "&quot;")
                cards.append(
                    f'<div class="fig-card fig-interactive">'
                    f'<iframe id="plotly-frame-{fig_idx}" srcdoc="{escaped}" '
                    f'style="width:100%;height:500px;border:none;" loading="lazy"></iframe>'
                    f'<div class="fig-actions">'
                    f'<p class="fig-caption">{desc} (interactivo)</p>'
                    f'<button class="btn-sm" onclick="exportPlotly400DPI(\'plotly-frame-{fig_idx}\', '
                    f'\'{fig.get("name", "figure")}\')">⬇ PNG 400 DPI</button>'
                    f'</div></div>'
                )
                fig_idx += 1
            continue

        # PNG/JPEG: base64 embed with lightbox support
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
            f'<a class="btn-sm" href="{b64}" download="{safe_name}.png">⬇ PNG</a>'
            f'</div></div>'
        )
    if not cards:
        return "<p>No se pudieron cargar las figuras.</p>"
    return f'<div class="fig-grid">{"".join(cards)}</div>'


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
    """Genera tabla HTML interactiva para referencias con busqueda y export Excel/CSV."""
    if not refs:
        return "<p>No se encontraron referencias.</p>"

    rows: list[str] = []
    for i, r in enumerate(refs, 1):
        title = r.get("title", "Sin título")
        authors = r.get("authors", "")
        year = r.get("year", "")
        doi = r.get("doi", "")
        url = r.get("url", "")
        key_finding = r.get("key_finding", "")
        source = r.get("source", "")

        source_badge = ""
        if source == "tavily":
            source_badge = '<span class="badge badge-ok">Web</span>'
        elif source == "claude":
            source_badge = '<span class="badge badge-fallback">Académico</span>'

        links_parts: list[str] = []
        if doi:
            doi_url = f"https://doi.org/{doi}" if not doi.startswith("http") else doi
            links_parts.append(
                f'<a href="{doi_url}" target="_blank" rel="noopener">DOI</a>'
            )
        if url:
            links_parts.append(
                f'<a href="{url}" target="_blank" rel="noopener">Ver ↗</a>'
            )
        links_html = " ".join(links_parts) if links_parts else "<em>-</em>"

        rows.append(
            f"<tr>"
            f"<td>{i}</td>"
            f'<td class="ref-title">{title}</td>'
            f"<td>{authors}</td>"
            f"<td>{year}</td>"
            f"<td>{source_badge}</td>"
            f'<td class="ref-finding-cell">{key_finding}</td>'
            f'<td class="ref-links-cell">{links_html}</td>'
            f"</tr>"
        )

    return (
        '<div class="table-toolbar">'
        '<input class="search-input" id="refsSearch" type="text" '
        'placeholder="Buscar en referencias..." '
        "oninput=\"filterTable('refsSearch', 'refsTable')\">"
        '<div class="table-actions">'
        '<button class="btn-sm" onclick="exportTableExcel(\'refsTable\', '
        "'referencias.xlsx')\">⬇ Excel</button>"
        '<button class="btn-sm" onclick="exportTableCSV(\'refsTable\', '
        "'referencias.csv')\">⬇ CSV</button>"
        '</div></div>'
        '<table class="data-table ref-table" id="refsTable"><thead><tr>'
        '<th>#</th><th>Título</th><th>Autores</th><th>Año</th><th>Fuente</th>'
        '<th>Hallazgo Clave</th><th>Enlaces</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def _build_hallazgos_html(hallazgos: dict) -> str:
    """Formatea hallazgos EDA en pestanas interactivas (tabs)."""
    if not hallazgos:
        return "<p>No se generaron hallazgos.</p>"

    interp = hallazgos.get("interpretation", "")
    normality = hallazgos.get("normality", {})
    outliers_data = hallazgos.get("outliers", {})
    vif_summary = hallazgos.get("vif_summary", {})
    fi = hallazgos.get("feature_importance", {})
    top = fi.get("top_features", [])
    corr = hallazgos.get("correlations", {})
    outliers = hallazgos.get("outliers", {})

    # KPI summary row
    n_normal = sum(1 for v in normality.values() if isinstance(v, dict) and v.get("normal"))
    n_not_normal = sum(1 for v in normality.values() if isinstance(v, dict) and not v.get("normal"))
    n_outlier_vars = sum(1 for v in outliers_data.values() if isinstance(v, dict) and v.get("pct", 0) > 1)
    n_vif_flagged = vif_summary.get("n_flagged", 0)

    kpi_html = (
        '<div class="kpi-row">'
        f'<div class="kpi"><div class="value">{n_normal}</div><div class="label">Vars normales</div></div>'
        f'<div class="kpi"><div class="value">{n_not_normal}</div><div class="label">Vars no normales</div></div>'
        f'<div class="kpi"><div class="value">{n_outlier_vars}</div><div class="label">Vars con outliers &gt;1%</div></div>'
        f'<div class="kpi"><div class="value">{n_vif_flagged}</div><div class="label">Vars VIF alto</div></div>'
        f'<div class="kpi"><div class="value">{len(top)}</div><div class="label">Top features</div></div>'
        '</div>'
    )

    # Tab 1: Interpretacion
    interp_html = f'<p>{interp}</p>' if interp else '<p>Sin interpretación disponible.</p>'

    # Tab 2: Correlaciones
    corr_html = ""
    spearman = corr.get("spearman", {}) if corr else {}
    if spearman:
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
                    strength = "fuerte" if abs(val) > 0.7 else "moderada"
                    strong.append(
                        f"<tr><td>{row_name}</td><td>{col_name}</td>"
                        f"<td>{val:.3f}</td><td>{direction} {strength}</td></tr>"
                    )
        if strong:
            corr_html = (
                '<table class="data-table"><thead><tr><th>Variable A</th><th>Variable B</th>'
                f'<th>Spearman r</th><th>Tipo</th></tr></thead><tbody>{"".join(strong[:20])}</tbody></table>'
            )
        else:
            corr_html = "<p>No se encontraron correlaciones fuertes (|r| &gt; 0.5).</p>"
    else:
        corr_html = "<p>Datos de correlación no disponibles.</p>"

    # Tab 3: Outliers
    outlier_html = ""
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
        outlier_html = (
            '<table class="data-table"><thead><tr><th>Variable</th><th>N Outliers</th>'
            f'<th>Porcentaje</th><th>Severidad</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
        )
    else:
        outlier_html = "<p>No se detectaron outliers significativos.</p>"

    # Tab 4: Top Features
    feat_html = ""
    if top:
        feat_html = '<ol>' + "".join(f"<li><strong>{f}</strong></li>" for f in top) + '</ol>'
    else:
        feat_html = "<p>No se calculó ranking de features.</p>"

    # Build tabs
    tabs = [
        ("hallTab1", "Resumen", interp_html),
        ("hallTab2", "Correlaciones", corr_html),
        ("hallTab3", "Outliers", outlier_html),
        ("hallTab4", "Top Features", feat_html),
    ]
    tab_btns = "".join(
        f'<button class="tab-btn{" active" if i == 0 else ""}" data-tab="{tid}" '
        f"onclick=\"switchTab('hallazgosTabs', '{tid}')\">{label}</button>"
        for i, (tid, label, _) in enumerate(tabs)
    )
    tab_panes = "".join(
        f'<div id="{tid}" class="tab-pane{" active" if i == 0 else ""}">{content}</div>'
        for i, (tid, _, content) in enumerate(tabs)
    )

    return kpi_html + f'<div id="hallazgosTabs"><div class="tabs">{tab_btns}</div>{tab_panes}</div>'


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


def _build_bp_html(bp_result: dict | None, correccion: str | None, tarea: str = "regression") -> str:
    """Formatea resultados del test de Breusch-Pagan."""
    if tarea != "regression":
        return "<p>Test de Breusch-Pagan no aplica para clasificación.</p>"
    if not bp_result or bp_result.get("error"):
        return "<p>Test de Breusch-Pagan no aplicable o no ejecutado.</p>"

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


def _build_sampling_variants_html(state: dict) -> str:
    """Genera HTML con la comparación de las 3 variantes de muestreo."""
    sampling_variants = state.get("sampling_variants", {})
    balanceo_log = state.get("balanceo_log", {})
    tarea = state.get("tarea_sugerida", "")

    if tarea != "classification" or not sampling_variants:
        return ""

    parts: list[str] = []

    # Justificación de la selección
    reason = balanceo_log.get("reason", "")
    selected_method = balanceo_log.get("method", "")
    if reason:
        parts.append(
            f'<div class="alert alert-ok">'
            f'<strong>Método seleccionado: {selected_method.capitalize()}</strong><br>'
            f'{reason}</div>'
        )

    # Tabla comparativa
    rows = []
    ratio_before = balanceo_log.get("ratio_before", "N/A")
    rows.append(
        f"<tr><td><strong>Original</strong></td>"
        f"<td>{ratio_before}</td>"
        f"<td>N/A</td>"
        f"<td>-</td></tr>"
    )

    for method in ["oversample", "hybrid", "undersample"]:
        info = sampling_variants.get(method, {})
        if info.get("error"):
            continue
        ratio = info.get("ratio_after", "N/A")
        n_rows = info.get("n_rows", "N/A")
        desc = info.get("description", "")
        selected = info.get("selected", False)
        badge = ' <span class="badge badge-ok">★ SELECCIONADO</span>' if selected else ""
        name = f"<strong>{method.capitalize()}</strong>{badge}"
        ratio_fmt = f"{ratio:.2f}" if isinstance(ratio, (int, float)) else ratio
        rows.append(
            f"<tr><td>{name}</td>"
            f"<td>{ratio_fmt}</td>"
            f"<td>{n_rows}</td>"
            f"<td>{desc}</td></tr>"
        )

    if rows:
        parts.append(
            '<table class="data-table"><thead><tr>'
            '<th>Variante</th><th>Ratio (max/min)</th><th>N filas</th><th>Descripción</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
        )

    # Distribución de clases por variante
    for method in ["oversample", "hybrid", "undersample"]:
        info = sampling_variants.get(method, {})
        dist = info.get("class_distribution", {})
        if dist and not info.get("error"):
            selected = info.get("selected", False)
            badge = " ★" if selected else ""
            dist_items = ", ".join(f"Clase {k}: {v}" for k, v in dist.items())
            parts.append(f"<p><strong>{method.capitalize()}{badge}:</strong> {dist_items}</p>")

    return "\n".join(parts) if parts else ""


def _build_download_buttons(state: dict, output_dir: Path) -> str:
    """Genera botones de descarga para train y test datasets + variantes de muestreo."""
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

    # Variantes de muestreo (solo clasificación)
    sampling_variants = state.get("sampling_variants", {})
    for method in ["oversample", "undersample", "hybrid"]:
        info = sampling_variants.get(method, {})
        path = info.get("path", "")
        if path and not info.get("error"):
            selected_tag = " ★" if info.get("selected") else ""
            datasets.append((f"Train {method.capitalize()}{selected_tag}", path))

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
    --sidebar-text: #e2e8f0; --border: #e2e8f0;
    --card-shadow: 0 2px 8px rgba(0,0,0,0.08);
    --success: #16a34a; --success-bg: #dcfce7; --warning: #d97706; --warning-bg: #fef3c7;
    --danger: #dc2626; --danger-bg: #fee2e2; --radius: 12px;
    --gradient-accent: linear-gradient(135deg, #2563eb, #7c3aed);
}
[data-theme="dark"] {
    --bg: #0f172a; --bg2: #1e293b; --text: #e2e8f0; --accent: #60a5fa;
    --accent2: #93c5fd; --accent-light: #1e3a5f; --sidebar-bg: #0f172a;
    --sidebar-text: #e2e8f0; --border: #334155;
    --card-shadow: 0 2px 8px rgba(0,0,0,0.4);
    --success: #4ade80; --success-bg: #14532d; --warning: #fbbf24; --warning-bg: #451a03;
    --danger: #f87171; --danger-bg: #450a0a;
    --gradient-accent: linear-gradient(135deg, #60a5fa, #a78bfa);
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
       background: var(--bg); color: var(--text); display: flex; min-height: 100vh;
       line-height: 1.6; }
/* Scroll progress */
.scroll-progress { position: fixed; top: 0; left: 0; width: 0%; height: 3px;
    background: var(--gradient-accent); z-index: 999; transition: width 0.05s linear; }
/* Sidebar */
.sidebar { width: 280px; background: var(--sidebar-bg); color: var(--sidebar-text);
           padding: 24px 0; position: fixed; height: 100vh; overflow-y: auto; z-index: 10;
           border-right: 1px solid rgba(255,255,255,0.05); }
.sidebar h2 { padding: 0 24px 16px; font-size: 1.1rem; letter-spacing: 0.5px;
              border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 12px; }
.sidebar a { display: flex; align-items: center; gap: 10px; padding: 10px 24px;
             color: var(--sidebar-text); text-decoration: none;
             font-size: 0.88rem; transition: all 0.2s; border-left: 3px solid transparent; }
.sidebar a:hover, .sidebar a.active { background: rgba(255,255,255,0.08);
    border-left-color: var(--accent); color: var(--accent); }
.nav-icon { font-size: 1rem; width: 22px; text-align: center; }
/* Main */
.main { margin-left: 280px; padding: 32px 48px; flex: 1; max-width: 1200px; }
.header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 32px;
          padding-bottom: 16px; border-bottom: 2px solid var(--accent); }
.header h1 { font-size: 1.5rem; font-weight: 700; }
.theme-toggle { cursor: pointer; padding: 8px 18px; border: 1px solid var(--border);
                border-radius: 8px; background: var(--bg2); color: var(--text);
                font-size: 0.85rem; transition: all 0.2s; }
.theme-toggle:hover { border-color: var(--accent); }
/* Sections — collapsible */
section { background: var(--bg2); border-radius: var(--radius); padding: 0; margin-bottom: 24px;
          box-shadow: var(--card-shadow); border: 1px solid var(--border); overflow: hidden;
          animation: fadeIn 0.3s ease; }
.section-header { display: flex; justify-content: space-between; align-items: center;
    cursor: pointer; user-select: none; padding: 22px 28px; transition: background 0.2s; }
.section-header:hover { background: var(--accent-light); }
.section-header h2 { color: var(--accent); font-size: 1.2rem; margin: 0; font-weight: 700;
    padding-bottom: 0; border-bottom: none; }
.section-header .chevron { transition: transform 0.3s ease; font-size: 1.1rem;
    color: var(--accent); opacity: 0.6; }
.section-header.collapsed .chevron { transform: rotate(-90deg); }
.section-content { padding: 0 28px 24px; transition: max-height 0.4s ease, opacity 0.3s ease,
    padding 0.3s ease; overflow: hidden; }
.section-content.collapsed { max-height: 0 !important; opacity: 0; padding-top: 0; padding-bottom: 0; }
section h3 { color: var(--accent2); margin: 18px 0 10px; font-size: 1rem; font-weight: 600; }
/* Tables */
.data-table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.88rem; }
.data-table th { background: var(--accent); color: white; padding: 10px 14px; text-align: left;
                 cursor: pointer; font-weight: 600; font-size: 0.82rem; text-transform: uppercase;
                 letter-spacing: 0.5px; position: sticky; top: 0; }
.data-table th:hover { background: var(--accent2); }
.data-table td { padding: 10px 14px; border-bottom: 1px solid var(--border); }
.data-table tr:hover { background: var(--accent-light); }
.table-toolbar { display: flex; justify-content: space-between; align-items: center; gap: 12px;
    flex-wrap: wrap; margin-bottom: 12px; }
.search-input { flex: 1; min-width: 200px; padding: 9px 16px; border: 1px solid var(--border);
    border-radius: 8px; font-size: 0.88rem; background: var(--bg); color: var(--text);
    transition: border-color 0.2s; }
.search-input:focus { outline: none; border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-light); }
.table-actions { display: flex; gap: 8px; }
.btn-sm { padding: 7px 14px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;
    cursor: pointer; text-decoration: none; border: 1px solid var(--accent); color: var(--accent);
    background: transparent; transition: all 0.2s; white-space: nowrap; }
.btn-sm:hover { background: var(--accent); color: white; }
/* Tabs */
.tabs { display: flex; gap: 0; border-bottom: 2px solid var(--border); margin-bottom: 16px;
    overflow-x: auto; }
.tab-btn { padding: 10px 20px; border: none; background: none; cursor: pointer; font-size: 0.88rem;
    font-weight: 600; color: var(--text); opacity: 0.55; border-bottom: 3px solid transparent;
    margin-bottom: -2px; transition: all 0.2s; white-space: nowrap; }
.tab-btn.active { opacity: 1; color: var(--accent); border-bottom-color: var(--accent); }
.tab-btn:hover { opacity: 0.85; background: var(--accent-light); }
.tab-pane { display: none; animation: fadeIn 0.3s ease; }
.tab-pane.active { display: block; }
/* Figure grid + cards */
.fig-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(440px, 1fr)); gap: 20px; }
.fig-card { background: var(--bg); border-radius: 10px; overflow: hidden;
    border: 1px solid var(--border); transition: box-shadow 0.2s, transform 0.2s; }
.fig-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.12); transform: translateY(-2px); }
.fig-card img { width: 100%; display: block; cursor: pointer; transition: opacity 0.2s; }
.fig-card img:hover { opacity: 0.92; }
.fig-interactive { grid-column: 1 / -1; }
.fig-interactive iframe { border-radius: 8px; }
.fig-actions { display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px; gap: 10px; }
.fig-caption { font-size: 0.85rem; color: var(--accent2); margin: 0; }
/* Buttons */
.btn-export, .btn-download { display: inline-block; padding: 6px 16px; border-radius: 6px;
    font-size: 0.8rem; font-weight: 600; cursor: pointer; text-decoration: none;
    border: 1px solid var(--accent); color: var(--accent); background: transparent;
    transition: all 0.2s; white-space: nowrap; }
.btn-export:hover, .btn-download:hover { background: var(--accent); color: white; }
.download-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; align-items: center; }
.btn-download { padding: 12px 24px; font-size: 0.88rem; border-radius: 8px;
    background: var(--accent); color: white; border: none; }
.btn-download:hover { background: var(--accent2); transform: translateY(-1px); }
/* KPI cards — enhanced */
.kpi-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }
.kpi { background: var(--bg); border-radius: 12px; padding: 20px 24px; min-width: 140px;
       text-align: center; box-shadow: var(--card-shadow); border: 1px solid var(--border);
       transition: transform 0.2s; position: relative; overflow: hidden; }
.kpi:hover { transform: translateY(-2px); }
.kpi::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--gradient-accent); }
.kpi .value { font-size: 1.6rem; font-weight: 700; color: var(--accent); }
.kpi .label { font-size: 0.78rem; color: var(--text); opacity: 0.65; margin-top: 4px; }
/* Badges */
.badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; }
.badge-ok { background: var(--success-bg); color: var(--success); }
.badge-error { background: var(--danger-bg); color: var(--danger); }
.badge-fallback { background: var(--warning-bg); color: var(--warning); }
/* Hallazgo cards */
.hallazgo-card { background: var(--bg); border-radius: 10px; padding: 18px 22px;
    margin: 12px 0; border-left: 4px solid var(--accent); }
.hallazgo-card h3 { color: var(--accent); margin: 0 0 10px; font-size: 0.95rem; }
.hallazgo-interpretation { border-left-color: var(--success); }
/* Alerts */
.alert { padding: 14px 20px; border-radius: 8px; margin: 12px 0; font-size: 0.9rem; }
.alert-warning { background: var(--warning-bg); border-left: 4px solid var(--warning); }
.alert-ok { background: var(--success-bg); border-left: 4px solid var(--success); }
/* Ref table */
.ref-table td { vertical-align: top; }
.ref-table .ref-title { font-weight: 600; color: var(--accent); }
.ref-table .ref-finding-cell { font-size: 0.85rem; max-width: 300px; }
.ref-links-cell a { font-size: 0.78rem; color: var(--accent); text-decoration: none; font-weight: 600;
    padding: 2px 8px; border: 1px solid var(--accent); border-radius: 4px; transition: all 0.2s;
    display: inline-block; margin: 2px 0; }
.ref-links-cell a:hover { background: var(--accent); color: white; }
/* Code blocks */
pre { background: var(--bg); padding: 14px; border-radius: 8px; overflow-x: auto;
      font-size: 0.85rem; border: 1px solid var(--border); }
/* Lightbox */
.lightbox { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.92); z-index: 200; justify-content: center; align-items: center;
    flex-direction: column; }
.lightbox.show { display: flex; }
.lightbox img { max-width: 88%; max-height: 78%; border-radius: 8px; object-fit: contain; }
.lb-nav { position: absolute; top: 50%; transform: translateY(-50%); padding: 14px 18px;
    background: rgba(255,255,255,0.12); color: white; border: none; cursor: pointer;
    font-size: 1.6rem; border-radius: 8px; transition: background 0.2s;
    backdrop-filter: blur(4px); }
.lb-nav:hover { background: rgba(255,255,255,0.25); }
.lb-prev { left: 20px; }
.lb-next { right: 20px; }
.lb-close { position: absolute; top: 16px; right: 24px; color: white; font-size: 2rem;
    cursor: pointer; background: none; border: none; opacity: 0.7; transition: opacity 0.2s; }
.lb-close:hover { opacity: 1; }
.lb-caption { color: rgba(255,255,255,0.85); margin-top: 14px; font-size: 0.9rem;
    text-align: center; max-width: 80%; }
.lb-counter { color: rgba(255,255,255,0.5); font-size: 0.8rem; margin-top: 6px; }
/* Toast */
.toast { position: fixed; bottom: 30px; right: 30px; padding: 14px 28px; border-radius: 10px;
    background: var(--success); color: white; font-weight: 600; font-size: 0.9rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25); z-index: 300;
    transform: translateY(100px); opacity: 0; transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    pointer-events: none; }
.toast.show { transform: translateY(0); opacity: 1; }
/* Lists */
ul, ol { margin-left: 22px; }
li { margin-bottom: 6px; }
/* Animations */
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); } }
/* Responsive */
@media (max-width: 900px) {
    .sidebar { display: none; }
    .main { margin-left: 0; padding: 16px; }
    .fig-grid { grid-template-columns: 1fr; }
    .kpi-row { justify-content: center; }
}
/* Print */
@media print {
    .sidebar, .theme-toggle, .btn-export, .btn-download, .btn-sm, .download-row,
    .scroll-progress, .lightbox, .toast, .search-input, .table-toolbar,
    .section-header .chevron, .fig-actions { display: none !important; }
    .main { margin-left: 0 !important; padding: 10px !important; max-width: 100% !important; }
    section { break-inside: avoid; box-shadow: none !important; border: 1px solid #ddd !important; }
    .section-content { max-height: none !important; opacity: 1 !important;
        padding: 12px 20px !important; }
    .fig-grid { grid-template-columns: 1fr 1fr; }
    body { font-size: 10pt; }
    .kpi::before { display: none; }
}
"""

_JS = """
/* === Theme === */
function toggleTheme() {
    var h = document.documentElement;
    var cur = h.getAttribute('data-theme') || 'light';
    var next = cur === 'light' ? 'dark' : 'light';
    h.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
}
(function(){ var s = localStorage.getItem('theme'); if(s) document.documentElement.setAttribute('data-theme', s); })();

/* === Scroll Progress === */
window.addEventListener('scroll', function() {
    var h = document.documentElement;
    var pct = (h.scrollTop / (h.scrollHeight - h.clientHeight)) * 100;
    var bar = document.getElementById('scrollProgress');
    if (bar) bar.style.width = Math.min(pct, 100) + '%';
});

/* === Collapsible Sections === */
function toggleSection(el) {
    var content = el.nextElementSibling;
    if (!content) return;
    el.classList.toggle('collapsed');
    if (el.classList.contains('collapsed')) {
        content.style.maxHeight = content.scrollHeight + 'px';
        content.offsetHeight;
        content.classList.add('collapsed');
    } else {
        content.classList.remove('collapsed');
        content.style.maxHeight = content.scrollHeight + 'px';
        setTimeout(function(){ content.style.maxHeight = 'none'; }, 400);
    }
}
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.section-content:not(.collapsed)').forEach(function(el) {
        el.style.maxHeight = 'none';
    });
});

/* === Tabs === */
function switchTab(groupId, tabId) {
    var group = document.getElementById(groupId);
    if (!group) return;
    group.querySelectorAll('.tab-btn').forEach(function(b){ b.classList.remove('active'); });
    group.querySelectorAll('.tab-pane').forEach(function(p){ p.classList.remove('active'); });
    var btn = group.querySelector('[data-tab=\"' + tabId + '\"]');
    if (btn) btn.classList.add('active');
    var pane = document.getElementById(tabId);
    if (pane) pane.classList.add('active');
}

/* === Table Search / Filter === */
function filterTable(inputId, tableId) {
    var q = document.getElementById(inputId).value.toLowerCase();
    document.querySelectorAll('#' + tableId + ' tbody tr').forEach(function(row) {
        row.style.display = row.textContent.toLowerCase().indexOf(q) > -1 ? '' : 'none';
    });
}

/* === Toast Notification === */
function showToast(msg) {
    var t = document.getElementById('toast');
    if (!t) return;
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(function(){ t.classList.remove('show'); }, 3000);
}

/* === Export Table as CSV === */
function exportTableCSV(tableId, filename) {
    var table = document.getElementById(tableId);
    if (!table) return;
    var csv = [];
    table.querySelectorAll('tr').forEach(function(r) {
        var row = [];
        r.querySelectorAll('th,td').forEach(function(c){ row.push('\"' + c.textContent.replace(/\"/g, '\"\"') + '\"'); });
        csv.push(row.join(','));
    });
    var blob = new Blob(['\\ufeff' + csv.join('\\n')], {type:'text/csv;charset=utf-8;'});
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a'); a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
    showToast('\\u2713 ' + filename + ' descargado');
}

/* === Export Table as Excel (SheetJS on demand) === */
function exportTableExcel(tableId, filename) {
    function run() {
        var table = document.getElementById(tableId);
        var wb = XLSX.utils.table_to_book(table, {sheet:'Datos'});
        XLSX.writeFile(wb, filename);
        showToast('\\u2713 ' + filename + ' descargado');
    }
    if (typeof XLSX !== 'undefined') { run(); return; }
    var s = document.createElement('script');
    s.src = 'https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js';
    s.crossOrigin = 'anonymous';
    s.onload = run;
    s.onerror = function(){ alert('No se pudo cargar SheetJS. Verifica tu conexion a internet.'); };
    document.head.appendChild(s);
}

/* === ZIP Download (JSZip on demand) === */
function downloadAllZip(zipName) {
    function run() {
        var zip = new JSZip();
        document.querySelectorAll('.btn-download[href^=\"data:\"]').forEach(function(a) {
            var fname = a.getAttribute('download');
            var raw = a.href.split(',')[1];
            try { zip.file(fname, atob(raw)); } catch(e) {}
        });
        zip.generateAsync({type:'blob'}).then(function(blob) {
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a'); a.href = url; a.download = zipName || 'datasets.zip'; a.click();
            URL.revokeObjectURL(url);
            showToast('\\u2713 ' + (zipName || 'datasets.zip') + ' descargado');
        });
    }
    if (typeof JSZip !== 'undefined') { run(); return; }
    var s = document.createElement('script');
    s.src = 'https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js';
    s.crossOrigin = 'anonymous';
    s.onload = run;
    s.onerror = function(){ alert('No se pudo cargar JSZip. Verifica tu conexion a internet.'); };
    document.head.appendChild(s);
}

/* === Lightbox with Navigation === */
var _lbImgs = [], _lbIdx = 0;
function openLightbox(idx) {
    _lbIdx = idx;
    var lb = document.getElementById('lightbox');
    if (!lb) return;
    document.getElementById('lbImg').src = _lbImgs[idx].src;
    document.getElementById('lbCaption').textContent = _lbImgs[idx].caption || '';
    document.getElementById('lbCounter').textContent = (idx+1) + ' / ' + _lbImgs.length;
    lb.classList.add('show');
    document.body.style.overflow = 'hidden';
}
function closeLightbox() {
    var lb = document.getElementById('lightbox');
    if (lb) lb.classList.remove('show');
    document.body.style.overflow = '';
}
function lbNav(dir) {
    _lbIdx = (_lbIdx + dir + _lbImgs.length) % _lbImgs.length;
    document.getElementById('lbImg').src = _lbImgs[_lbIdx].src;
    document.getElementById('lbCaption').textContent = _lbImgs[_lbIdx].caption || '';
    document.getElementById('lbCounter').textContent = (_lbIdx+1) + ' / ' + _lbImgs.length;
}
document.addEventListener('keydown', function(e) {
    var lb = document.getElementById('lightbox');
    if (!lb || !lb.classList.contains('show')) return;
    if (e.key === 'Escape') closeLightbox();
    if (e.key === 'ArrowLeft') lbNav(-1);
    if (e.key === 'ArrowRight') lbNav(1);
});

/* === Plotly 400 DPI Export === */
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
            showToast('Exportando ' + name + ' a 400 DPI...');
        } else {
            alert('Esperando carga de la grafica. Intenta de nuevo en unos segundos.');
        }
    } catch(e) {
        alert('Error al exportar: ' + e.message);
    }
}

/* === Scroll Spy === */
document.addEventListener('DOMContentLoaded', function() {
    var sections = document.querySelectorAll('section[id]');
    var links = document.querySelectorAll('.sidebar a');
    window.addEventListener('scroll', function() {
        var current = '';
        sections.forEach(function(s) { if (window.scrollY >= s.offsetTop - 120) current = s.id; });
        links.forEach(function(a) {
            a.classList.remove('active');
            if (a.getAttribute('href') === '#' + current) a.classList.add('active');
        });
    });
    /* Init lightbox image list */
    document.querySelectorAll('.fig-card img').forEach(function(img, i) {
        _lbImgs.push({src: img.src, caption: img.alt || ''});
        img.style.cursor = 'pointer';
        img.onclick = function() { openLightbox(i); };
    });
});

/* === Sortable Tables === */
document.addEventListener('click', function(e) {
    if (e.target.tagName === 'TH' && e.target.closest('.data-table')) {
        var th = e.target;
        var table = th.closest('table');
        var idx = Array.from(th.parentNode.children).indexOf(th);
        var tbody = table.querySelector('tbody');
        if (!tbody) return;
        var rows = Array.from(tbody.querySelectorAll('tr'));
        var asc = th.dataset.sort !== 'asc';
        th.parentNode.querySelectorAll('th').forEach(function(h){ delete h.dataset.sort; });
        th.dataset.sort = asc ? 'asc' : 'desc';
        rows.sort(function(a, b) {
            var av = (a.children[idx] || {}).textContent || '';
            var bv = (b.children[idx] || {}).textContent || '';
            var an = parseFloat(av), bn = parseFloat(bv);
            if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
            return asc ? av.localeCompare(bv) : bv.localeCompare(av);
        });
        rows.forEach(function(r){ tbody.appendChild(r); });
    }
});
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

    # Sidebar links — filtrados según tarea
    is_classification = tarea == "classification"
    is_regression = tarea == "regression"

    nav_items = [
        ("resumen", "Resumen Ejecutivo"),
        ("literatura", "Literatura"),
        ("hipotesis", "Hipotesis"),
        ("dataset", "Dataset"),
        ("descargas", "Descargar Datos"),
        ("perfil", "Perfil de Columnas"),
        ("preprocesamiento", "Preprocesamiento"),
    ]
    if is_classification:
        nav_items.append(("muestreo", "Variantes de Muestreo"))
    nav_items.extend([
        ("hallazgos", "Hallazgos EDA"),
        ("normalidad", "Normalidad"),
        ("vif", "VIF — Multicolinealidad"),
    ])
    if is_regression:
        nav_items.append(("breusch_pagan", "Breusch-Pagan"))
    nav_items.extend([
        ("importancia", "Feature Importance"),
    ])
    if state.get("flag_timeseries"):
        nav_items.append(("timeseries", "Series de Tiempo"))
    nav_items.extend([
        ("decision", "Decision"),
        ("modelos", "Modelos"),
        ("figuras", "Visualizaciones"),
        ("advertencias", "Advertencias"),
        ("pasos", "Proximos Pasos"),
        ("agentes", "Estado Agentes"),
    ])
    nav_icons = {
        "resumen": "📋", "literatura": "📚", "hipotesis": "🔬",
        "dataset": "📊", "descargas": "⬇", "perfil": "📐",
        "preprocesamiento": "⚙", "muestreo": "🔄", "hallazgos": "🔍",
        "normalidad": "📈", "vif": "🔗", "breusch_pagan": "📉",
        "importancia": "⭐", "timeseries": "⏰", "decision": "🎯",
        "modelos": "🤖", "figuras": "🖼", "advertencias": "⚠",
        "pasos": "👣", "agentes": "🔧",
    }
    nav_html = "\n".join(
        f'<a href="#{sid}"><span class="nav-icon">{nav_icons.get(sid, "•")}</span>{label}</a>'
        for sid, label in nav_items
    )

    # Helper for collapsible sections
    def _sec(sid: str, title: str, content: str) -> str:
        return (
            f'<section id="{sid}">'
            f'<div class="section-header" onclick="toggleSection(this)">'
            f'<h2>{title}</h2><span class="chevron">▼</span></div>'
            f'<div class="section-content">\n{content}\n</div></section>'
        )

    # Build section content blocks
    resumen_content = (
        f'<p><strong>Pregunta:</strong> {question}</p>'
        '<div class="kpi-row">'
        f'<div class="kpi"><div class="value">{dataset_size}</div><div class="label">Filas</div></div>'
        f'<div class="kpi"><div class="value">{n_features}</div><div class="label">Columnas</div></div>'
        f'<div class="kpi"><div class="value">{tarea}</div><div class="label">Tarea</div></div>'
        f'<div class="kpi"><div class="value">{model_family}</div><div class="label">Familia</div></div>'
        f'<div class="kpi"><div class="value">{metrica}</div><div class="label">Métrica</div></div>'
        f'<div class="kpi"><div class="value">{n_figures}</div><div class="label">Figuras</div></div>'
        '</div>'
        f'<p><strong>Tipo de datos:</strong> {data_type} | <strong>Target:</strong> {target} | <strong>Desbalance:</strong> {desbalance_str}</p>'
    )

    lit_content = (
        '<h3>Ecuaciones PICO</h3>'
        '<ul>' + "".join(f"<li><code>{eq}</code></li>" for eq in state.get('search_equations', [])) + '</ul>'
        '<h3>Referencias</h3>'
        + refs_html
    )

    dataset_content = _build_json_table({
        "Filas": dataset_size, "Target": target, "Tipo": data_type,
        "Desbalance": desbalance_str, "Serie temporal": state.get('flag_timeseries', False),
    })

    descargas_content = (
        '<p>Datasets listos para modelamiento (con encoding y preprocesamiento aplicados):</p>'
        + _build_download_buttons(state, output_dir)
        + '<div style="margin-top:12px;">'
        '<button class="btn-sm" onclick="downloadAllZip(\'datasets.zip\')">📦 Descargar todo (.zip)</button>'
        '</div>'
    )

    prepro_content = (
        '<h3>Encoding</h3>'
        + _build_encoding_table(state.get('encoding_log', {}))
        + '<h3>Features nuevas</h3>'
        f'<p>{state.get("features_nuevas", []) or "Ninguna"}</p>'
        '<h3>Balanceo</h3>'
        f'<pre>{json.dumps(state.get("balanceo_log", {}), indent=2, ensure_ascii=False)}</pre>'
    )

    muestreo_content = (
        '<p>Se generaron las 3 variantes de muestreo para comparar y seleccionar la mejor estrategia de balanceo.</p>'
        + _build_sampling_variants_html(state)
    )

    hallazgos_content = _build_hallazgos_html(hallazgos)

    normalidad_content = (
        '<p>Evaluación de la distribución normal de variables numéricas (Shapiro-Wilk / Anderson-Darling, α = 0.05).</p>'
        + _build_normality_html(hallazgos)
    )

    vif_content = (
        '<p>Análisis de multicolinealidad mediante VIF. Valores &gt; 10 indican multicolinealidad severa.</p>'
        + _build_vif_html(vif_all, vif_flags)
    )

    bp_content = (
        '<p>Evalúa si la varianza de los residuos es constante (homoscedasticidad) en el modelo de regresión.</p>'
        + _build_bp_html(bp_result, correccion, tarea)
    )

    fi_content = _build_feature_importance_html(state.get('feature_importance', {}))

    decision_content = _build_json_table({
        "Tarea": tarea, "Model family": model_family,
        "Técnica hiperparámetros": hyper, "Métrica principal": metrica,
    })

    modelos_content = _build_models_table(state.get('modelos_recomendados', []))

    figuras_content = _build_figures_html(figures, output_dir)

    pasos_content = (
        '<ol>'
        '<li>Entrenar modelos recomendados con los hiperparámetros sugeridos</li>'
        '<li>Validar con cross-validation sobre train set</li>'
        '<li>Evaluar en test set con métricas seleccionadas</li>'
        '<li>Iterar según resultados</li>'
        '</ol>'
    )

    # Assemble sections
    sections_html = "\n".join([
        _sec("resumen", "📋 Resumen Ejecutivo", resumen_content),
        _sec("literatura", "📚 Revisión de Literatura", lit_content),
        _sec("hipotesis", "🔬 Hipótesis", hip_html),
        _sec("dataset", "📊 Descripción del Dataset", dataset_content),
        _sec("descargas", "⬇ Descargar Datos", descargas_content),
        _sec("perfil", "📐 Perfil de Columnas", _build_profile_table(state.get('perfil_columnas', {}))),
        _sec("preprocesamiento", "⚙ Preprocesamiento", prepro_content),
    ])

    if is_classification:
        sections_html += "\n" + _sec("muestreo", "🔄 Variantes de Muestreo", muestreo_content)

    sections_html += "\n".join([
        "",
        _sec("hallazgos", "🔍 Hallazgos EDA", hallazgos_content),
        _sec("normalidad", "📈 Test de Normalidad", normalidad_content),
        _sec("vif", "🔗 VIF — Multicolinealidad", vif_content),
    ])

    if is_regression:
        sections_html += "\n" + _sec("breusch_pagan", "📉 Breusch-Pagan", bp_content)

    sections_html += "\n" + _sec("importancia", "⭐ Feature Importance", fi_content)

    if state.get('flag_timeseries'):
        sections_html += "\n" + _sec("timeseries", "⏰ Series de Tiempo", ts_html)

    sections_html += "\n".join([
        "",
        _sec("decision", "🎯 Decisión de Tarea", decision_content),
        _sec("modelos", "🤖 Modelos Recomendados", modelos_content),
        _sec("figuras", "🖼 Visualizaciones", figuras_content),
        _sec("advertencias", "⚠ Advertencias y Limitaciones", warns_html),
        _sec("pasos", "👣 Próximos Pasos", pasos_content),
        _sec("agentes", "🔧 Estado de Agentes", f'<p>{status_html}</p>'),
    ])

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EDA Report - {run_id}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="scroll-progress" id="scrollProgress"></div>

<nav class="sidebar">
<h2>EDA Agents</h2>
<p style="padding: 0 20px; font-size: 0.8rem; opacity: 0.7;">Run: {run_id}</p>
{nav_html}
</nav>

<div class="main">
<div class="header">
<h1>Reporte EDA - {run_id}</h1>
<button class="theme-toggle" onclick="toggleTheme()">🌓 Tema</button>
</div>

{sections_html}

</div>

<div class="lightbox" id="lightbox">
<button class="lb-close" onclick="closeLightbox()">&times;</button>
<button class="lb-nav lb-prev" onclick="lbNav(-1)">&#10094;</button>
<button class="lb-nav lb-next" onclick="lbNav(1)">&#10095;</button>
<img id="lbImg" src="" alt="">
<p class="lb-caption" id="lbCaption"></p>
<p class="lb-counter" id="lbCounter"></p>
</div>
<div class="toast" id="toast"></div>

<script>{_JS}</script>
</body>
</html>"""

    html_path = report_dir / "reporte_eda.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path
