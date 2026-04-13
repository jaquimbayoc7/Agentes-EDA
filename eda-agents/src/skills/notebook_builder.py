"""Skill — Generador de notebook Jupyter reproducible.

Produce un archivo .ipynb con el proceso completo del EDA:
1. Carga de datos y configuracion
2. Profiling basico
3. Preprocesamiento (encoding, balanceo)
4. EDA estadistico
5. Visualizaciones
6. Decision de modelos
7. Proximos pasos

Usada como post-procesamiento en main.py despues del pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _cell_markdown(source: str) -> dict:
    """Crea una celda Markdown."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")],
    }


def _cell_code(source: str) -> dict:
    """Crea una celda de codigo Python."""
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")],
    }


def build_notebook(state: dict[str, Any], output_dir: str | Path) -> Path:
    """Genera un notebook Jupyter reproducible con todo el proceso EDA.

    Parameters
    ----------
    state : dict
        Estado final del pipeline.
    output_dir : str or Path
        Directorio base de outputs del run.

    Returns
    -------
    Path al archivo .ipynb generado.
    """
    output_dir = Path(output_dir)
    nb_dir = output_dir / "notebooksFinales"
    nb_dir.mkdir(parents=True, exist_ok=True)

    run_id = state.get("run_id", "unknown")
    question = state.get("research_question", "N/A")
    target = state.get("target", "N/A")
    data_type = state.get("data_type", "tabular")
    dataset_path = state.get("dataset_path", "")
    train_path = state.get("train_path", "")
    test_path = state.get("test_path", "")
    dataset_train_final = state.get("dataset_train_final", "")
    dataset_test_final = state.get("dataset_test_final", "")

    # Relative paths for portability
    def _rel(p: str) -> str:
        if not p:
            return ""
        try:
            return str(Path(p).relative_to(Path.cwd()))
        except ValueError:
            return p

    dataset_rel = _rel(dataset_path)
    train_rel = _rel(train_path)
    test_rel = _rel(test_path)
    train_final_rel = _rel(dataset_train_final)
    test_final_rel = _rel(dataset_test_final)

    # Build figure paths relative
    figures = state.get("figures", [])
    fig_paths = []
    for f in figures:
        fp = f.get("path", "")
        if fp:
            fig_paths.append((_rel(fp), f.get("description", f.get("name", ""))))

    # Encoding log for reproducibility
    encoding_log = state.get("encoding_log", {})
    encoding_json = json.dumps(encoding_log, indent=2, ensure_ascii=False, default=str)

    # Hallazgos
    hallazgos = state.get("hallazgos_eda", {})
    hallazgos_json = json.dumps(hallazgos, indent=2, ensure_ascii=False, default=str)

    # Decision
    decision = {
        "tarea": state.get("tarea_sugerida"),
        "modelos_recomendados": state.get("modelos_recomendados", []),
        "hyperparams_technique": state.get("hyperparams_technique"),
        "model_family": state.get("model_family"),
        "metrica_principal": state.get("metrica_principal"),
    }
    decision_json = json.dumps(decision, indent=2, ensure_ascii=False, default=str)

    # Hipotesis
    hip = state.get("hipotesis") or {}

    # Perfil
    perfil = state.get("perfil_columnas", {})
    col_names = list(perfil.keys())

    # Numeric columns from perfil
    numeric_cols = [c for c, info in perfil.items()
                    if isinstance(info, dict) and info.get("dtype", "").startswith("float")]

    cells: list[dict] = []

    # ======================== TITULO ========================
    cells.append(_cell_markdown(
        f"# Notebook Reproducible - EDA Agents\n"
        f"\n"
        f"**Run ID:** `{run_id}`\n"
        f"\n"
        f"**Pregunta de investigacion:** {question}\n"
        f"\n"
        f"**Target:** `{target}` | **Tipo:** `{data_type}`\n"
        f"\n"
        f"Este notebook reproduce paso a paso el analisis exploratorio\n"
        f"generado por el sistema multi-agente EDA Agents."
    ))

    # ======================== 1. SETUP ========================
    cells.append(_cell_markdown("## 1. Configuracion e Importaciones"))
    cells.append(_cell_code(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import json\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "# Plotly para graficas interactivas\n"
        "import plotly.express as px\n"
        "import plotly.graph_objects as go\n"
        "from plotly.subplots import make_subplots\n"
        "\n"
        "# Configuracion de visualizacion\n"
        "plt.rcParams['figure.figsize'] = (10, 6)\n"
        "plt.rcParams['figure.dpi'] = 100\n"
        "sns.set_style('whitegrid')\n"
        "\n"
        f"RANDOM_SEED = {state.get('random_seed', 42)}\n"
        "np.random.seed(RANDOM_SEED)\n"
        "\n"
        f"print('Configuracion lista. Seed:', RANDOM_SEED)"
    ))

    # ======================== 2. CARGA ========================
    cells.append(_cell_markdown(
        "## 2. Carga del Dataset\n"
        f"\n"
        f"Dataset original: `{dataset_rel}`"
    ))
    cells.append(_cell_code(
        f"# Cargar dataset original\n"
        f"df = pd.read_csv(r'{dataset_rel}')\n"
        f"print(f'Shape: {{df.shape}}')\n"
        f"print(f'Columnas: {{list(df.columns)}}')\n"
        f"df.head()"
    ))
    cells.append(_cell_code(
        "# Informacion general\n"
        "df.info()"
    ))
    cells.append(_cell_code(
        "# Estadisticas descriptivas\n"
        "df.describe(include='all')"
    ))

    # ======================== 3. HIPOTESIS ========================
    cells.append(_cell_markdown(
        "## 3. Hipotesis de Investigacion\n"
        "\n"
        f"- **H1 (confirmatoria):** {hip.get('h1', 'N/A')}\n"
        f"- **H2 (exploratoria):** {hip.get('h2', 'N/A')}\n"
        f"- **H3 (alternativa):** {hip.get('h3', 'N/A')}"
    ))

    # ======================== 4. PERFIL ========================
    cells.append(_cell_markdown("## 4. Perfil de Datos"))
    cells.append(_cell_code(
        "# Valores nulos\n"
        "null_pct = df.isnull().mean() * 100\n"
        "print('Porcentaje de nulos por columna:')\n"
        "print(null_pct.to_string())"
    ))
    cells.append(_cell_code(
        "# Cardinalidad\n"
        "print('Valores unicos por columna:')\n"
        "print(df.nunique().to_string())"
    ))
    cells.append(_cell_code(
        "# Tipos de datos\n"
        "print(df.dtypes.to_string())"
    ))

    # ======================== 5. PREPROCESAMIENTO ========================
    cells.append(_cell_markdown(
        "## 5. Preprocesamiento\n"
        "\n"
        "### Encoding aplicado por el pipeline:"
    ))
    cells.append(_cell_code(
        f"encoding_log = json.loads('''{encoding_json}''')\n"
        f"\n"
        f"for col, info in encoding_log.items():\n"
        f"    if isinstance(info, dict):\n"
        f"        print(f'{{col}}: {{info.get(\"encoding\", \"N/A\")}} ({{info.get(\"flag\", \"\")}})')\n"
        f"        new_cols = info.get('new_cols', [])\n"
        f"        if new_cols:\n"
        f"            print(f'  -> Nuevas columnas: {{new_cols}}')"
    ))

    if train_final_rel:
        cells.append(_cell_markdown("### Cargar datos procesados"))
        cells.append(_cell_code(
            f"# Dataset de entrenamiento final (con encoding aplicado)\n"
            f"df_train = pd.read_csv(r'{train_final_rel}')\n"
            f"print(f'Train shape: {{df_train.shape}}')\n"
            f"df_train.head()"
        ))
        if test_final_rel:
            cells.append(_cell_code(
                f"# Dataset de test final\n"
                f"df_test = pd.read_csv(r'{test_final_rel}')\n"
                f"print(f'Test shape: {{df_test.shape}}')\n"
                f"df_test.head()"
            ))
    elif train_rel:
        cells.append(_cell_code(
            f"df_train = pd.read_csv(r'{train_rel}')\n"
            f"print(f'Train shape: {{df_train.shape}}')"
        ))

    # ======================== 6. EDA ESTADISTICO ========================
    cells.append(_cell_markdown("## 6. Analisis Exploratorio Estadistico"))

    cells.append(_cell_code(
        "# Correlaciones (Spearman - robusta a no-linealidad)\n"
        "numeric_df = df_train.select_dtypes(include=[np.number])\n"
        "if len(numeric_df.columns) >= 2:\n"
        "    corr = numeric_df.corr(method='spearman')\n"
        "    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',\n"
        "                    zmin=-1, zmax=1, title='Matriz de Correlaciones (Spearman)')\n"
        "    fig.update_layout(width=800, height=700)\n"
        "    fig.show()\n"
        "else:\n"
        "    print('Menos de 2 columnas numericas para correlacion.')"
    ))

    cells.append(_cell_code(
        "# Distribuciones de variables numericas (interactivo)\n"
        "numeric_cols = numeric_df.columns.tolist()\n"
        "n_cols = len(numeric_cols)\n"
        "if n_cols > 0:\n"
        "    cols_to_plot = numeric_cols[:8]\n"
        "    n = len(cols_to_plot)\n"
        "    rows = (n + 1) // 2\n"
        "    fig = make_subplots(rows=rows, cols=2, subplot_titles=cols_to_plot)\n"
        "    for i, col in enumerate(cols_to_plot):\n"
        "        r, c = i // 2 + 1, i % 2 + 1\n"
        "        fig.add_trace(go.Histogram(x=df_train[col].dropna(), name=col,\n"
        "                                    nbinsx=30, showlegend=False), row=r, col=c)\n"
        "    fig.update_layout(title='Distribuciones Numericas', height=300*rows, width=900)\n"
        "    fig.show()"
    ))

    cells.append(_cell_code(
        "# Boxplots interactivos\n"
        "if n_cols > 0:\n"
        "    cols_to_box = numeric_cols[:8]\n"
        "    fig = go.Figure()\n"
        "    for col in cols_to_box:\n"
        "        fig.add_trace(go.Box(y=df_train[col].dropna(), name=col))\n"
        "    fig.update_layout(title='Boxplots', height=500, width=900, showlegend=False)\n"
        "    fig.show()"
    ))

    if target and target != "N/A":
        cells.append(_cell_code(
            f"# Distribucion del target (interactivo)\n"
            f"if '{target}' in df_train.columns:\n"
            f"    if pd.api.types.is_numeric_dtype(df_train['{target}']) and df_train['{target}'].nunique() > 10:\n"
            f"        fig = px.histogram(df_train, x='{target}', nbins=30,\n"
            f"                            title='Distribucion del target: {target}',\n"
            f"                            color_discrete_sequence=['coral'])\n"
            f"    else:\n"
            f"        vc = df_train['{target}'].value_counts().reset_index()\n"
            f"        vc.columns = ['{target}', 'count']\n"
            f"        fig = px.bar(vc, x='{target}', y='count',\n"
            f"                     title='Distribucion de clases: {target}',\n"
            f"                     color_discrete_sequence=['coral'])\n"
            f"    fig.update_layout(height=500, width=800)\n"
            f"    fig.show()"
        ))

    cells.append(_cell_code(
        "# Scatter Matrix (top-6 numericas, interactivo)\n"
        "pair_cols = numeric_cols[:6]\n"
        "if len(pair_cols) >= 2:\n"
        "    pair_df = df_train[pair_cols].dropna()\n"
        "    if len(pair_df) > 500:\n"
        "        pair_df = pair_df.sample(500, random_state=RANDOM_SEED)\n"
        "    fig = px.scatter_matrix(pair_df, dimensions=pair_cols,\n"
        "                             title='Scatter Matrix Top-6')\n"
        "    fig.update_layout(height=800, width=900)\n"
        "    fig.update_traces(diagonal_visible=True, marker=dict(size=3, opacity=0.5))\n"
        "    fig.show()"
    ))

    # ======================== 6b. FEATURE IMPORTANCE ========================
    feat_imp = state.get("feature_importance", {})
    if feat_imp:
        cells.append(_cell_markdown(
            "## 6b. Feature Importance\n"
            "\n"
            "Importancia de variables calculada con Mutual Information y Permutation Importance."
        ))

        mi_scores = feat_imp.get("mutual_information", {})
        perm_scores = feat_imp.get("permutation_importance", {})
        top_feats = feat_imp.get("top_features", [])

        if mi_scores:
            mi_json = json.dumps(mi_scores, indent=2, ensure_ascii=False, default=str)
            cells.append(_cell_code(
                f"# Mutual Information\n"
                f"mi_scores = json.loads('''{mi_json}''')\n"
                f"mi_df = pd.DataFrame({{'feature': list(mi_scores.keys()), 'MI': list(mi_scores.values())}})\n"
                f"mi_df = mi_df.sort_values('MI', ascending=True)\n"
                f"\n"
                f"fig = go.Figure(go.Bar(x=mi_df['MI'], y=mi_df['feature'], orientation='h',\n"
                f"                        marker_color='steelblue'))\n"
                f"fig.update_layout(title='Mutual Information Scores', xaxis_title='MI Score',\n"
                f"                  height=max(300, len(mi_df)*25), width=800)\n"
                f"fig.show()"
            ))

        if perm_scores:
            perm_json = json.dumps(perm_scores, indent=2, ensure_ascii=False, default=str)
            cells.append(_cell_code(
                f"# Permutation Importance\n"
                f"perm_scores = json.loads('''{perm_json}''')\n"
                f"perm_df = pd.DataFrame([\n"
                f"    {{'feature': k, 'mean': v['mean'], 'std': v['std']}}\n"
                f"    for k, v in perm_scores.items()\n"
                f"])\n"
                f"perm_df = perm_df.sort_values('mean', ascending=True)\n"
                f"\n"
                f"fig = go.Figure(go.Bar(x=perm_df['mean'], y=perm_df['feature'], orientation='h',\n"
                f"                        error_x=dict(type='data', array=perm_df['std']),\n"
                f"                        marker_color='darkorange'))\n"
                f"fig.update_layout(title='Permutation Importance', xaxis_title='Importance',\n"
                f"                  height=max(300, len(perm_df)*25), width=800)\n"
                f"fig.show()"
            ))

        if top_feats:
            cells.append(_cell_code(
                f"# Top features seleccionadas\n"
                f"top_features = {top_feats}\n"
                f"print('Top features seleccionadas por average rank:')\n"
                f"for i, f in enumerate(top_features, 1):\n"
                f"    print(f'  {{i}}. {{f}}')"
            ))

    # ======================== 6c. VIF — Multicolinealidad ========================
    vif_all = state.get("vif_all", {})
    if not vif_all:
        vif_all = state.get("hallazgos_eda", {}).get("vif_all", {})
    if vif_all:
        vif_json = json.dumps(vif_all, indent=2, ensure_ascii=False, default=str)
        cells.append(_cell_markdown(
            "## 6c. VIF — Factor de Inflación de Varianza\n"
            "\n"
            "Valores VIF > 10 indican multicolinealidad severa."
        ))
        cells.append(_cell_code(
            f"# VIF (Factor de Inflacion de Varianza)\n"
            f"vif_data = json.loads('''{vif_json}''')\n"
            f"vif_df = pd.DataFrame({{'feature': list(vif_data.keys()), 'VIF': list(vif_data.values())}})\n"
            f"vif_df = vif_df.sort_values('VIF', ascending=True)\n"
            f"vif_df['VIF_capped'] = vif_df['VIF'].clip(upper=100)  # limitar para visualizacion\n"
            f"\n"
            f"colors = ['#e74c3c' if v > 10 else '#f39c12' if v > 5 else '#27ae60' for v in vif_df['VIF']]\n"
            f"fig = go.Figure(go.Bar(x=vif_df['VIF_capped'], y=vif_df['feature'], orientation='h',\n"
            f"                        marker_color=colors))\n"
            f"fig.add_vline(x=10, line_dash='dash', line_color='red', annotation_text='Umbral VIF=10')\n"
            f"fig.add_vline(x=5, line_dash='dot', line_color='orange', annotation_text='VIF=5')\n"
            f"fig.update_layout(title='VIF — Factor de Inflación de Varianza',\n"
            f"                  xaxis_title='VIF', template='plotly_white',\n"
            f"                  height=max(400, len(vif_df)*30), width=800)\n"
            f"fig.show()\n"
            f"\n"
            f"# Resumen\n"
            f"n_high = sum(1 for v in vif_data.values() if v > 10)\n"
            f"print(f'Variables con VIF > 10 (multicolinealidad alta): {{n_high}}')\n"
            f"for feat, vif_val in sorted(vif_data.items(), key=lambda x: x[1], reverse=True):\n"
            f"    flag = ' ⚠ ALTA' if vif_val > 10 else ' ⚡ moderada' if vif_val > 5 else ''\n"
            f"    print(f'  {{feat}}: {{vif_val:.2f}}{{flag}}')"
        ))

    # ======================== 6d. Normalidad (Q-Q Plots) ========================
    normality = state.get("hallazgos_eda", {}).get("normality", {})
    if normality:
        norm_json = json.dumps(normality, indent=2, ensure_ascii=False, default=str)
        cells.append(_cell_markdown(
            "## 6d. Test de Normalidad (Q-Q Plots)\n"
            "\n"
            "Evaluación de normalidad de variables numéricas con Shapiro-Wilk / Anderson-Darling."
        ))
        cells.append(_cell_code(
            f"from scipy import stats as sp_stats\n"
            f"\n"
            f"normality_results = json.loads('''{norm_json}''')\n"
            f"\n"
            f"# Tabla de resultados\n"
            f"print('=== Test de Normalidad ===')\n"
            f"for col, info in normality_results.items():\n"
            f"    test = info.get('test', 'shapiro')\n"
            f"    stat = info.get('statistic', 0)\n"
            f"    pval = info.get('p_value')\n"
            f"    normal = 'Si' if info.get('normal') else 'No'\n"
            f"    if pval is not None:\n"
            f"        print(f'  {{col}}: {{normal}} (test={{test}}, stat={{stat:.4f}}, p={{pval:.4f}})')\n"
            f"    else:\n"
            f"        print(f'  {{col}}: test={{test}}, stat={{stat:.4f}}')"
        ))
        cells.append(_cell_code(
            "# Q-Q Plots\n"
            "norm_cols = [c for c in normality_results.keys() if c in df_train.columns][:6]\n"
            "n_norm = len(norm_cols)\n"
            "if n_norm > 0:\n"
            "    rows_n = (n_norm + 1) // 2\n"
            "    fig = make_subplots(rows=rows_n, cols=2,\n"
            "                        subplot_titles=[f'Q-Q: {c}' for c in norm_cols])\n"
            "    for i, col in enumerate(norm_cols):\n"
            "        clean = df_train[col].dropna().values\n"
            "        if len(clean) < 8:\n"
            "            continue\n"
            "        theoretical_q = sp_stats.norm.ppf(np.linspace(0.01, 0.99, min(len(clean), 200)))\n"
            "        sample = np.sort(np.random.choice(clean, size=min(len(clean), 200), replace=False))\n"
            "        sample_q = (sample - np.mean(clean)) / (np.std(clean) + 1e-12)\n"
            "        r, c_ = i // 2 + 1, i % 2 + 1\n"
            "        fig.add_trace(go.Scatter(x=theoretical_q, y=sample_q, mode='markers',\n"
            "                                  name=col, marker=dict(size=4, opacity=0.6),\n"
            "                                  showlegend=False), row=r, col=c_)\n"
            "        fig.add_trace(go.Scatter(x=[-3, 3], y=[-3, 3], mode='lines',\n"
            "                                  line=dict(color='red', dash='dash'),\n"
            "                                  showlegend=False), row=r, col=c_)\n"
            "    fig.update_layout(title_text='Q-Q Plots — Test de Normalidad',\n"
            "                      height=300*rows_n, template='plotly_white')\n"
            "    fig.show()"
        ))

    # ======================== 6e. Breusch-Pagan ========================
    bp_result = state.get("breusch_pagan_result")
    if bp_result and not bp_result.get("error"):
        bp_json = json.dumps(bp_result, indent=2, ensure_ascii=False, default=str)
        correccion_sugerida = state.get("modelo_correccion_heterosc", "HC3")
        cells.append(_cell_markdown(
            "## 6e. Test de Breusch-Pagan — Heteroscedasticidad\n"
            "\n"
            "Evalúa si los residuos del modelo de regresión tienen varianza constante."
        ))
        cells.append(_cell_code(
            f"bp_result = json.loads('''{bp_json}''')\n"
            f"\n"
            f"print('=== Test de Breusch-Pagan ===')\n"
            "print(f'  BP Statistic: {bp_result[\"bp_statistic\"]:.4f}')\n"
            "print(f'  BP p-valor:   {bp_result[\"bp_pvalue\"]:.4f}')\n"
            "print(f'  F Statistic:  {bp_result[\"f_statistic\"]:.4f}')\n"
            "print(f'  F p-valor:    {bp_result[\"f_pvalue\"]:.4f}')\n"
            "print()\n"
            "if bp_result['heteroscedastic']:\n"
            "    print('⚠ Se detectó HETEROSCEDASTICIDAD (p < 0.05).')\n"
            "    print('  Los errores estándar OLS pueden ser sesgados.')\n"
            f"    print('  Corrección sugerida: {correccion_sugerida}')\n"
            "else:\n"
            "    print('✓ No se detectó heteroscedasticidad (homoscedasticidad).')\n"
            "    print('  Los errores estándar OLS son válidos.')"
        ))
        # Residual plot
        if target and target != "N/A":
            cells.append(_cell_code(
                f"# Gráfico de residuos vs ajustados\n"
                f"import statsmodels.api as sm_api\n"
                f"\n"
                f"features_bp = [c for c in df_train.select_dtypes(include=[np.number]).columns if c != '{target}'][:10]\n"
                f"clean = df_train[['{target}'] + features_bp].dropna()\n"
                f"if len(clean) > 20:\n"
                f"    y = clean['{target}'].values\n"
                f"    X = sm_api.add_constant(clean[features_bp].values)\n"
                f"    ols = sm_api.OLS(y, X).fit()\n"
                f"    residuals = ols.resid\n"
                f"    fitted = ols.fittedvalues\n"
                f"    hetero_lbl = 'Heteroscedástico' if bp_result['heteroscedastic'] else 'Homoscedástico'\n"
                f"    fig = go.Figure()\n"
                f"    fig.add_trace(go.Scatter(x=fitted, y=residuals, mode='markers',\n"
                f"                              marker=dict(size=4, opacity=0.5, color='#3498db'),\n"
                f"                              name='Residuos'))\n"
                f"    fig.add_hline(y=0, line_dash='dash', line_color='red')\n"
                f"    fig.update_layout(title=f'Residuos vs Ajustados ({{hetero_lbl}})',\n"
                f"                      xaxis_title='Valores ajustados', yaxis_title='Residuos',\n"
                f"                      template='plotly_white', height=500, width=800)\n"
                f"    fig.show()"
            ))

    # ======================== 7. HALLAZGOS ========================
    cells.append(_cell_markdown("## 7. Hallazgos del Pipeline"))

    # Interpretación
    interp = hallazgos.get("interpretation", "")
    if interp:
        cells.append(_cell_markdown(
            f"### Interpretación General\n\n{interp}"
        ))

    cells.append(_cell_code(
        f"hallazgos = json.loads('''{hallazgos_json}''')\n"
        f"\n"
        f"# Correlaciones significativas (|r| > 0.5)\n"
        f"if 'correlations' in hallazgos:\n"
        f"    spearman = hallazgos['correlations'].get('spearman', {{}})\n"
        f"    print('=== Correlaciones Significativas (|r| > 0.5) ===')\n"
        f"    seen = set()\n"
        f"    for row_name, row_vals in spearman.items():\n"
        f"        for col_name, val in row_vals.items():\n"
        f"            if row_name == col_name:\n"
        f"                continue\n"
        f"            pair = tuple(sorted((row_name, col_name)))\n"
        f"            if pair in seen:\n"
        f"                continue\n"
        f"            seen.add(pair)\n"
        f"            if abs(val) > 0.5:\n"
        f"                direction = 'positiva' if val > 0 else 'negativa'\n"
        f"                print(f'  {{row_name}} <-> {{col_name}}: r={{val:.3f}} ({{direction}})')\n"
        f"\n"
        f"# Outliers\n"
        f"if 'outliers' in hallazgos:\n"
        f"    print('\\n=== Outliers (IQR 1.5x) ===')\n"
        f"    for col, info in hallazgos['outliers'].items():\n"
        f"        n_out = info.get('n_outliers', 0)\n"
        f"        pct = info.get('pct', 0)\n"
        f"        flag = ' ⚠' if pct > 5 else ''\n"
        f"        print(f'  {{col}}: {{n_out}} outliers ({{pct:.1f}}%){{flag}}')"
    ))

    # ======================== 8. DECISION ========================
    cells.append(_cell_markdown("## 8. Decision del Pipeline"))
    cells.append(_cell_code(
        f"decision = json.loads('''{decision_json}''')\n"
        f"\n"
        f"print('Tarea inferida:', decision.get('tarea'))\n"
        f"print('Familia de modelo:', decision.get('model_family'))\n"
        f"print('Metrica principal:', decision.get('metrica_principal'))\n"
        f"print('Tecnica de hiperparametros:', decision.get('hyperparams_technique'))\n"
        f"print()\n"
        f"print('Modelos recomendados:')\n"
        f"for m in decision.get('modelos_recomendados', []):\n"
        f"    if isinstance(m, dict):\n"
        f"        print(f'  - {{m.get(\"name\", \"N/A\")}}: {{m.get(\"reason\", \"\")}}')"
    ))

    # ======================== 9. FIGURAS GENERADAS ========================
    if fig_paths:
        cells.append(_cell_markdown("## 9. Figuras Generadas por el Pipeline"))
        cells.append(_cell_code(
            "from IPython.display import Image, HTML, display\n"
            "from pathlib import Path as _Path\n"
            "\n"
            "# Mostrar figuras generadas por el pipeline\n"
            "figuras = [\n" +
            "".join(f"    (r'{fp}', '{desc}'),\n" for fp, desc in fig_paths) +
            "]\n"
            "\n"
            "for path, desc in figuras:\n"
            "    try:\n"
            "        print(f'--- {desc} ---')\n"
            "        p = _Path(path)\n"
            "        if p.suffix == '.html':\n"
            "            display(HTML(p.read_text(encoding='utf-8')))\n"
            "        else:\n"
            "            display(Image(filename=path))\n"
            "    except Exception as e:\n"
            "        print(f'No se pudo cargar {path}: {e}')"
        ))

    # ======================== 10. PROXIMOS PASOS ========================
    cells.append(_cell_markdown(
        "## 10. Proximos Pasos\n"
        "\n"
        "1. Entrenar los modelos recomendados con los hiperparametros sugeridos\n"
        "2. Validar con cross-validation sobre el train set\n"
        "3. Evaluar en test set con las metricas seleccionadas\n"
        "4. Iterar segun resultados\n"
        "\n"
        "### Ejemplo de entrenamiento:"
    ))

    # Build model training example
    models = state.get("modelos_recomendados", [])
    model_name = models[0].get("name", "Ridge") if models and isinstance(models[0], dict) else "Ridge"
    is_regression = (state.get("tarea_sugerida") or "regression") == "regression"

    if is_regression:
        cells.append(_cell_code(
            f"from sklearn.linear_model import Ridge\n"
            f"from sklearn.model_selection import GridSearchCV\n"
            f"from sklearn.metrics import mean_squared_error\n"
            f"\n"
            f"# Separar features y target\n"
            f"X_train = df_train.drop(columns=['{target}'], errors='ignore')\n"
            f"y_train = df_train['{target}'] if '{target}' in df_train.columns else None\n"
            f"\n"
            f"if y_train is not None:\n"
            f"    # Solo columnas numericas\n"
            f"    X_train_num = X_train.select_dtypes(include=[np.number]).fillna(0)\n"
            f"\n"
            f"    # Ejemplo: Ridge con GridSearchCV\n"
            f"    model = Ridge()\n"
            f"    params = {{'alpha': [0.01, 0.1, 1.0, 10.0]}}\n"
            f"    grid = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')\n"
            f"    grid.fit(X_train_num, y_train)\n"
            f"\n"
            f"    print(f'Mejor alpha: {{grid.best_params_}}')\n"
            f"    print(f'Mejor RMSE (CV): {{(-grid.best_score_)**0.5:.4f}}')\n"
            f"else:\n"
            f"    print('Target no encontrado en df_train')"
        ))
    else:
        cells.append(_cell_code(
            f"from sklearn.ensemble import RandomForestClassifier\n"
            f"from sklearn.model_selection import GridSearchCV\n"
            f"from sklearn.metrics import accuracy_score, classification_report\n"
            f"\n"
            f"X_train = df_train.drop(columns=['{target}'], errors='ignore')\n"
            f"y_train = df_train['{target}'] if '{target}' in df_train.columns else None\n"
            f"\n"
            f"if y_train is not None:\n"
            f"    X_train_num = X_train.select_dtypes(include=[np.number]).fillna(0)\n"
            f"    model = RandomForestClassifier(random_state=RANDOM_SEED)\n"
            f"    params = {{'n_estimators': [50, 100], 'max_depth': [5, 10, None]}}\n"
            f"    grid = GridSearchCV(model, params, cv=5, scoring='accuracy')\n"
            f"    grid.fit(X_train_num, y_train)\n"
            f"    print(f'Mejores params: {{grid.best_params_}}')\n"
            f"    print(f'Mejor accuracy (CV): {{grid.best_score_:.4f}}')"
        ))

    cells.append(_cell_markdown(
        "---\n"
        f"*Generado automaticamente por EDA Agents (Run: `{run_id}`)*"
    ))

    # ======================== BUILD NOTEBOOK ========================
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
                "mimetype": "text/x-python",
                "file_extension": ".py",
            },
        },
        "cells": cells,
    }

    nb_path = nb_dir / f"eda_reproducible_{run_id}.ipynb"
    nb_path.write_text(
        json.dumps(notebook, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return nb_path
