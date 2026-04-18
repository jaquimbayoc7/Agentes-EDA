"""Skill — Generador de notebook Jupyter reproducible.

Produce un archivo .ipynb con CÓDIGO EJECUTABLE paso a paso:
1. Carga de datos y perfilado
2. Train/Test split
3. Encoding paso a paso
4. EDA con visualizaciones
5. Tests estadísticos (normalidad, VIF, Breusch-Pagan)
6. Feature importance (MI, Permutation)
7. Conclusiones y modelado

Cada celda ejecuta código real — nada está precalculado.
Solo se usa el state como CONFIGURACIÓN (dataset, target, seed, encoding recipe).
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
    """Genera un notebook Jupyter reproducible con código ejecutable.

    El notebook computa TODO desde cero a partir del CSV original.
    Del state solo se usa configuración (dataset, target, seed, encoding recipe),
    NO resultados precalculados.

    Parameters
    ----------
    state : dict
        Estado final del pipeline (se usa solo como configuración).
    output_dir : str or Path
        Directorio base de outputs del run.

    Returns
    -------
    Path al archivo .ipynb generado.
    """
    output_dir = Path(output_dir)
    nb_dir = output_dir / "notebooksFinales"
    nb_dir.mkdir(parents=True, exist_ok=True)

    # ---- Configuración del pipeline (NO resultados) ----
    run_id = state.get("run_id", "unknown")
    question = state.get("research_question", "N/A")
    target = state.get("target", "N/A")
    data_type = state.get("data_type", "tabular")
    dataset_path = state.get("dataset_path", "")
    seed = state.get("random_seed", 42)
    tarea = state.get("tarea_sugerida") or "regression"
    model_family = state.get("model_family") or "linear"
    modelos = state.get("modelos_recomendados", [])
    metrica = state.get("metrica_principal") or "RMSE"
    hp_technique = state.get("hyperparams_technique") or "GridSearchCV"
    hip = state.get("hipotesis") or {}

    # Encoding recipe — QUÉ encodear, no los resultados
    encoding_log = state.get("encoding_log", {})
    enc_recipe = {}
    for col_name, info in encoding_log.items():
        if isinstance(info, dict):
            enc_recipe[col_name] = info.get("encoding", "label")

    # Ruta relativa para portabilidad
    def _rel(p: str) -> str:
        if not p:
            return ""
        try:
            return str(Path(p).relative_to(Path.cwd()))
        except ValueError:
            return p

    dataset_rel = _rel(dataset_path)
    is_regression = tarea == "regression"

    cells: list[dict] = []

    # ======================== TITULO ========================
    cells.append(_cell_markdown(
        f"# EDA Reproducible — Paso a Paso\n"
        f"\n"
        f"**Run ID:** `{run_id}`\n"
        f"\n"
        f"**Pregunta de investigación:** {question}\n"
        f"\n"
        f"**Target:** `{target}` | **Tipo:** `{data_type}` | **Tarea:** `{tarea}`\n"
        f"\n"
        f"---\n"
        f"\n"
        f"### Metodología del Análisis\n"
        f"\n"
        f"Este notebook implementa un **Análisis Exploratorio de Datos (EDA) completo y reproducible**,\n"
        f"diseñado para responder a la pregunta de investigación planteada. Cada celda ejecuta código real\n"
        f"— nada está precalculado.\n"
        f"\n"
        f"El flujo sigue el estándar **CRISP-DM**:\n"
        f"1. **Comprensión de datos** — Carga, perfil, calidad, distribuciones\n"
        f"2. **Preparación** — Split, encoding, balanceo (sin data leakage)\n"
        f"3. **EDA cuantitativo** — Correlaciones, outliers, normalidad, VIF\n"
        f"4. **Feature engineering** — Importancia, selección informada\n"
        f"5. **Modelado baseline** — Modelo inicial para validar hallazgos\n"
        f"\n"
        f"Cada test estadístico incluye **interpretación contextualizada** a la pregunta\n"
        f"de investigación y las hipótesis formuladas."
    ))

    # ======================== 1. SETUP ========================
    cells.append(_cell_markdown("## 1. Configuración e Importaciones"))
    cells.append(_cell_code(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "# Visualización\n"
        "import plotly.express as px\n"
        "import plotly.graph_objects as go\n"
        "from plotly.subplots import make_subplots\n"
        "\n"
        "# Estadística\n"
        "from scipy import stats as sp_stats\n"
        "from statsmodels.stats.outliers_influence import variance_inflation_factor\n"
        "from statsmodels.stats.diagnostic import het_breuschpagan\n"
        "import statsmodels.api as sm\n"
        "\n"
        "# ML\n"
        "from sklearn.model_selection import train_test_split, GridSearchCV\n"
        "from sklearn.feature_selection import mutual_info_regression, mutual_info_classif\n"
        "from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier\n"
        "from sklearn.inspection import permutation_importance\n"
        "\n"
        f"RANDOM_SEED = {seed}\n"
        f"TARGET = '{target}'\n"
        f"TASK = '{tarea}'\n"
        "np.random.seed(RANDOM_SEED)\n"
        "\n"
        "print('Importaciones listas.')\n"
        "print(f'Target: {TARGET} | Tarea: {TASK} | Seed: {RANDOM_SEED}')"
    ))

    # ======================== 2. CARGA Y PERFIL ========================
    cells.append(_cell_markdown(
        "## 2. Carga y Perfil del Dataset\n"
        "\n"
        "Cargamos el CSV original y generamos un perfil completo\n"
        "antes de cualquier transformación."
    ))
    cells.append(_cell_code(
        f"df = pd.read_csv(r'{dataset_rel}')\n"
        "print(f'Shape: {df.shape[0]} filas × {df.shape[1]} columnas')\n"
        "print(f'Columnas: {list(df.columns)}')\n"
        "df.head()"
    ))
    cells.append(_cell_code(
        "# Información general del dataset\n"
        "df.info()"
    ))
    cells.append(_cell_code(
        "# Estadísticas descriptivas\n"
        "df.describe(include='all')"
    ))
    cells.append(_cell_code(
        "# Análisis de calidad de datos\n"
        "quality = pd.DataFrame({\n"
        "    'dtype': df.dtypes,\n"
        "    'nunique': df.nunique(),\n"
        "    'nulos': df.isnull().sum(),\n"
        "    'pct_nulos': (df.isnull().mean() * 100).round(2)\n"
        "})\n"
        "quality = quality.sort_values('pct_nulos', ascending=False)\n"
        "print('=== Perfil de Calidad ===')\n"
        "print(quality.to_string())"
    ))

    # ======================== 3. HIPOTESIS ========================
    cells.append(_cell_markdown(
        "## 3. Hipótesis de Investigación\n"
        "\n"
        "Formuladas por el Research Lead a partir de la pregunta de investigación:\n"
        "\n"
        f"- **H1 (confirmatoria):** {hip.get('h1', 'N/A')}\n"
        f"- **H2 (exploratoria):** {hip.get('h2', 'N/A')}\n"
        f"- **H3 (alternativa):** {hip.get('h3', 'N/A')}"
    ))

    # ======================== 4. TRAIN/TEST SPLIT ========================
    cells.append(_cell_markdown(
        "## 4. Train / Test Split\n"
        "\n"
        "### ¿Por qué separar antes de transformar?\n"
        "\n"
        "El **data leakage** ocurre cuando información del conjunto de test se filtra\n"
        "al entrenamiento, produciendo métricas artificialmente optimistas. Separamos\n"
        "**antes** de encoding, normalización o balanceo para garantizar una evaluación\n"
        "honesta del modelo.\n"
        "\n"
        "**Principio:** Todo estadístico (medias, frecuencias, mappings) se calcula\n"
        "exclusivamente con datos de entrenamiento y se aplica tal cual al test."
    ))
    cells.append(_cell_code(
        "TEST_SIZE = 0.2\n"
        "\n"
        "if TARGET not in df.columns:\n"
        "    raise ValueError(f'Target \"{TARGET}\" no encontrado. Columnas: {list(df.columns)}')\n"
        "\n"
        "df_train, df_test = train_test_split(\n"
        "    df, test_size=TEST_SIZE, random_state=RANDOM_SEED\n"
        ")\n"
        "df_train = df_train.reset_index(drop=True)\n"
        "df_test = df_test.reset_index(drop=True)\n"
        "\n"
        "print(f'Train: {df_train.shape}')\n"
        "print(f'Test:  {df_test.shape}')\n"
        "print(f'Ratio: {len(df_test)/len(df):.1%} test')"
    ))

    # ======================== 5. ENCODING ========================
    if enc_recipe:
        recipe_str = json.dumps(enc_recipe, indent=4, ensure_ascii=False)
        cells.append(_cell_markdown(
            "## 5. Encoding de Variables Categóricas\n"
            "\n"
            "### Conceptos Técnicos\n"
            "\n"
            "Las variables categóricas deben convertirse a numéricas para que los modelos\n"
            "las procesen. La elección del encoding impacta directamente en el desempeño:\n"
            "\n"
            "- **Label Encoding:** Asigna enteros (0, 1, 2...). Adecuado para variables ordinales.\n"
            "- **One-Hot Encoding (OHE):** Crea columnas binarias. Preserva independencia entre categorías.\n"
            "- **Frequency Encoding:** Reemplaza por frecuencia relativa. Útil con alta cardinalidad.\n"
            "\n"
            f"**Importante para `{target}`:** El encoding se ajusta **solo** con datos de train\n"
            f"para prevenir data leakage. Categorías nuevas en test se marcan como -1."
        ))
        cells.append(_cell_code(
            f"# Receta de encoding (determinada por el análisis del pipeline)\n"
            f"ENCODING_RECIPE = {recipe_str}\n"
            f"\n"
            "print('=== Columnas a Encodear ===')\n"
            "for col, enc_type in ENCODING_RECIPE.items():\n"
            "    if col in df_train.columns:\n"
            "        print(f'  {col}: {enc_type} (unique={df_train[col].nunique()})')\n"
            "    else:\n"
            "        print(f'  {col}: SKIP (no existe en dataset)')"
        ))
        cells.append(_cell_code(
            "# Aplicar encoding paso a paso\n"
            "for col, enc_type in ENCODING_RECIPE.items():\n"
            "    if col not in df_train.columns:\n"
            "        continue\n"
            "\n"
            "    print(f'\\n--- {col} → {enc_type} ---')\n"
            "    print(f'  Antes: dtype={df_train[col].dtype}, unique={df_train[col].nunique()}')\n"
            "\n"
            "    if enc_type == 'ohe':\n"
            "        # One-Hot Encoding (ajustado en train, alineado en test)\n"
            "        dummies_train = pd.get_dummies(df_train[[col]], prefix=col, drop_first=True)\n"
            "        dummies_test = pd.get_dummies(df_test[[col]], prefix=col, drop_first=True)\n"
            "        dummies_test = dummies_test.reindex(columns=dummies_train.columns, fill_value=0)\n"
            "        df_train = pd.concat([df_train.drop(columns=[col]), dummies_train], axis=1)\n"
            "        df_test = pd.concat([df_test.drop(columns=[col]), dummies_test], axis=1)\n"
            "        print(f'  → OHE creó {list(dummies_train.columns)}')\n"
            "\n"
            "    elif 'label' in enc_type:\n"
            "        # Label Encoding (mapping ajustado en train)\n"
            "        unique_vals = df_train[col].dropna().unique()\n"
            "        mapping = {v: i for i, v in enumerate(sorted(unique_vals, key=str))}\n"
            "        df_train[col] = df_train[col].map(mapping).fillna(-1).astype(int)\n"
            "        df_test[col] = df_test[col].map(mapping).fillna(-1).astype(int)\n"
            "        print(f'  → Label: {len(mapping)} categorías')\n"
            "\n"
            "    elif 'freq' in enc_type:\n"
            "        # Frequency Encoding (frecuencias de train)\n"
            "        freq = df_train[col].value_counts(normalize=True).to_dict()\n"
            "        df_train[col] = df_train[col].map(freq).fillna(0).astype(float)\n"
            "        df_test[col] = df_test[col].map(freq).fillna(0).astype(float)\n"
            "        print(f'  → Frequency: valores [{min(freq.values()):.4f}, {max(freq.values()):.4f}]')\n"
            "\n"
            "    elif 'ordinal' in enc_type:\n"
            "        # Ordinal Encoding\n"
            "        unique_vals = sorted(df_train[col].dropna().unique(), key=str)\n"
            "        mapping = {v: i for i, v in enumerate(unique_vals)}\n"
            "        df_train[col] = df_train[col].map(mapping).fillna(-1).astype(int)\n"
            "        df_test[col] = df_test[col].map(mapping).fillna(-1).astype(int)\n"
            "        print(f'  → Ordinal: {mapping}')\n"
            "\n"
            "print(f'\\nTrain shape post-encoding: {df_train.shape}')\n"
            "print(f'Test shape post-encoding:  {df_test.shape}')"
        ))
        cells.append(_cell_code(
            "# Verificar que todo sea numérico\n"
            "non_numeric = df_train.select_dtypes(exclude=[np.number]).columns.tolist()\n"
            "if non_numeric:\n"
            "    print(f'⚠ Columnas no numéricas restantes: {non_numeric}')\n"
            "    for col in non_numeric:\n"
            "        mapping = {v: i for i, v in enumerate(df_train[col].dropna().unique())}\n"
            "        df_train[col] = df_train[col].map(mapping).fillna(-1).astype(int)\n"
            "        df_test[col] = df_test[col].map(mapping).fillna(-1).astype(int)\n"
            "    print('  → Label encoding automático aplicado.')\n"
            "else:\n"
            "    print('✓ Todas las columnas son numéricas.')"
        ))
    else:
        cells.append(_cell_markdown(
            "## 5. Verificación de Tipos\n"
            "\n"
            "No se detectaron columnas categóricas que requieran encoding."
        ))
        cells.append(_cell_code(
            "# Verificar tipos\n"
            "non_numeric = df_train.select_dtypes(exclude=[np.number]).columns.tolist()\n"
            "if non_numeric:\n"
            "    print(f'Columnas no numéricas: {non_numeric}')\n"
            "    for col in non_numeric:\n"
            "        mapping = {v: i for i, v in enumerate(df_train[col].dropna().unique())}\n"
            "        df_train[col] = df_train[col].map(mapping).fillna(-1).astype(int)\n"
            "        df_test[col] = df_test[col].map(mapping).fillna(-1).astype(int)\n"
            "    print('  → Label encoding aplicado automáticamente.')\n"
            "else:\n"
            "    print('✓ Todas las columnas son numéricas. No se requiere encoding.')"
        ))

    # ======================== 5b. SAMPLING VARIANTS (clasificación) ========================
    if not is_regression:
        cells.append(_cell_markdown(
            "## 5b. Estrategias de Muestreo (Clasificación)\n"
            "\n"
            "### ¿Por qué balancear las clases?\n"
            "\n"
            "En problemas de clasificación, el desbalance de clases sesga al modelo hacia\n"
            "la clase mayoritaria. Evaluamos 3 estrategias para encontrar el mejor equilibrio\n"
            "entre representatividad y rendimiento:\n"
            "\n"
            "| Estrategia | Técnica | Pros | Contras |\n"
            "|-----------|---------|------|----------|\n"
            "| **Oversample** | RandomOverSampler | Preserva toda la info original | Puede causar overfitting |\n"
            "| **Undersample** | RandomUnderSampler | Rápido, evita duplicados | Pierde datos mayoritarios |\n"
            "| **Hybrid** | SMOTEENN | Genera nuevos puntos + limpia ruido | Más costoso, posibles artefactos |\n"
            "\n"
            f"Se selecciona la variante con menor combinación de desbalance residual y pérdida\n"
            f"de datos, optimizando para la tarea de predicción de `{target}`."
        ))
        cells.append(_cell_code(
            "from imblearn.over_sampling import RandomOverSampler\n"
            "from imblearn.under_sampling import RandomUnderSampler\n"
            "from imblearn.combine import SMOTEENN\n"
            "\n"
            "X_bal = df_train.drop(columns=[TARGET]).select_dtypes(include=[np.number]).fillna(0)\n"
            "y_bal = df_train[TARGET]\n"
            "\n"
            "original_dist = y_bal.value_counts().to_dict()\n"
            "print(f'Distribución original: {original_dist}')\n"
            "majority = max(original_dist.values())\n"
            "minority = min(original_dist.values())\n"
            "ratio_original = majority / max(minority, 1)\n"
            "print(f'Ratio de desbalance original: {ratio_original:.2f}:1')\n"
            "\n"
            "# 1. Oversample\n"
            "ros = RandomOverSampler(random_state=RANDOM_SEED)\n"
            "X_over, y_over = ros.fit_resample(X_bal, y_bal)\n"
            "dist_over = pd.Series(y_over).value_counts().to_dict()\n"
            "ratio_over = max(dist_over.values()) / max(min(dist_over.values()), 1)\n"
            "print(f'\\nOversample: {len(X_over)} filas, ratio={ratio_over:.2f}:1')\n"
            "\n"
            "# 2. Undersample\n"
            "rus = RandomUnderSampler(random_state=RANDOM_SEED)\n"
            "X_under, y_under = rus.fit_resample(X_bal, y_bal)\n"
            "dist_under = pd.Series(y_under).value_counts().to_dict()\n"
            "ratio_under = max(dist_under.values()) / max(min(dist_under.values()), 1)\n"
            "print(f'Undersample: {len(X_under)} filas, ratio={ratio_under:.2f}:1')\n"
            "\n"
            "# 3. Hybrid (SMOTEENN)\n"
            "try:\n"
            "    se = SMOTEENN(random_state=RANDOM_SEED)\n"
            "    X_hyb, y_hyb = se.fit_resample(X_bal, y_bal)\n"
            "    dist_hyb = pd.Series(y_hyb).value_counts().to_dict()\n"
            "    ratio_hyb = max(dist_hyb.values()) / max(min(dist_hyb.values()), 1)\n"
            "    print(f'Hybrid:      {len(X_hyb)} filas, ratio={ratio_hyb:.2f}:1')\n"
            "except Exception as e:\n"
            "    print(f'Hybrid falló: {e}. Usando oversample como fallback.')\n"
            "    X_hyb, y_hyb = X_over.copy(), y_over.copy()\n"
            "    dist_hyb = dist_over.copy()\n"
            "    ratio_hyb = ratio_over"
        ))
        cells.append(_cell_markdown("### Comparación Visual de Variantes"))
        cells.append(_cell_code(
            "# Gráfico comparativo de distribuciones por variante\n"
            "import plotly.graph_objects as go\n"
            "\n"
            "all_classes = sorted(set(list(original_dist.keys()) + list(dist_over.keys())\n"
            "                         + list(dist_under.keys()) + list(dist_hyb.keys())), key=str)\n"
            "variants = ['Original', 'Oversample', 'Undersample', 'Hybrid']\n"
            "dists = [original_dist, dist_over, dist_under, dist_hyb]\n"
            "colors = ['#636EFA', '#00CC96', '#EF553B', '#AB63FA']\n"
            "\n"
            "fig = go.Figure()\n"
            "for var, dist, color in zip(variants, dists, colors):\n"
            "    fig.add_trace(go.Bar(\n"
            "        x=[str(c) for c in all_classes],\n"
            "        y=[dist.get(c, 0) for c in all_classes],\n"
            "        name=var, marker_color=color\n"
            "    ))\n"
            "fig.update_layout(\n"
            "    title='Distribución de clases por variante de muestreo',\n"
            "    barmode='group', xaxis_title='Clase', yaxis_title='Frecuencia',\n"
            "    template='plotly_white', height=500, width=900\n"
            ")\n"
            "fig.show()\n"
            "\n"
            "# Ratio de desbalance por variante\n"
            "ratios = [ratio_original, ratio_over, ratio_under, ratio_hyb]\n"
            "fig2 = go.Figure(go.Bar(\n"
            "    x=variants, y=ratios,\n"
            "    marker_color=['gray' if r != min(ratios) else '#00CC96' for r in ratios],\n"
            "    text=[f'{r:.2f}:1' for r in ratios], textposition='outside'\n"
            "))\n"
            "fig2.add_hline(y=1.0, line_dash='dash', line_color='green',\n"
            "               annotation_text='Balance perfecto')\n"
            "fig2.update_layout(\n"
            "    title='Ratio de desbalance por variante (menor = mejor)',\n"
            "    yaxis_title='Ratio (mayor/menor)', template='plotly_white',\n"
            "    height=450, width=700\n"
            ")\n"
            "fig2.show()"
        ))
        cells.append(_cell_markdown("### Selección de la Mejor Variante"))
        cells.append(_cell_code(
            "# Selección basada en cercanía a ratio 1.0 + penalización por pérdida de datos\n"
            "n_original = len(X_bal)\n"
            "scores = {}\n"
            "for name, n_rows, ratio in [\n"
            "    ('oversample', len(X_over), ratio_over),\n"
            "    ('undersample', len(X_under), ratio_under),\n"
            "    ('hybrid', len(X_hyb), ratio_hyb),\n"
            "]:\n"
            "    dist_penalty = abs(ratio - 1.0)\n"
            "    loss_penalty = max(0, 1.0 - n_rows / n_original) * 0.5\n"
            "    score = dist_penalty + loss_penalty\n"
            "    scores[name] = score\n"
            "    print(f'{name}: ratio={ratio:.2f}, filas={n_rows}, '\n"
            "          f'dist_penalty={dist_penalty:.3f}, loss_penalty={loss_penalty:.3f}, '\n"
            "          f'score={score:.3f}')\n"
            "\n"
            "best = min(scores, key=scores.get)\n"
            "print(f'\\n✓ Variante seleccionada: {best} (score={scores[best]:.3f})')\n"
            "print(f'  Justificación: menor combinación de desbalance y pérdida de datos.')"
        ))

    # ======================== 6. EDA VISUAL ========================
    cells.append(_cell_markdown(
        "## 6. Análisis Exploratorio — Visualizaciones\n"
        "\n"
        "Las visualizaciones se generan **exclusivamente desde datos de entrenamiento** para\n"
        "mantener la integridad del análisis. Cada gráfico está diseñado para revelar patrones\n"
        f"relevantes para predecir `{target}`."
    ))

    # 6.1 Correlaciones
    cells.append(_cell_markdown(
        "### 6.1 Matriz de Correlaciones (Spearman)\n"
        "\n"
        "**Spearman vs Pearson:** Usamos Spearman porque:\n"
        "- No asume linealidad (detecta relaciones monótonas no lineales)\n"
        "- Es robusta a outliers y distribuciones no normales\n"
        "- Mide la relación ordinal entre variables\n"
        "\n"
        "**Interpretación:**\n"
        "- |r| > 0.7 → correlación fuerte (posible redundancia de features)\n"
        "- |r| > 0.5 → correlación moderada (relación significativa)\n"
        f"- Las correlaciones con `{target}` indican predictores potenciales"
    ))
    cells.append(_cell_code(
        "# Correlación Spearman (robusta a no-linealidad)\n"
        "numeric_df = df_train.select_dtypes(include=[np.number])\n"
        "numeric_cols = numeric_df.columns.tolist()\n"
        "corr = numeric_df.corr(method='spearman')\n"
        "\n"
        "fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',\n"
        "                zmin=-1, zmax=1, title='Matriz de Correlaciones (Spearman)')\n"
        "fig.update_layout(width=800, height=700)\n"
        "fig.show()\n"
        "\n"
        "# Identificar pares con correlación fuerte\n"
        "print('\\n=== Pares con |r| > 0.7 ===')\n"
        "for col_a in corr.columns:\n"
        "    for col_b in corr.columns:\n"
        "        if col_a >= col_b:\n"
        "            continue\n"
        "        r = corr.loc[col_a, col_b]\n"
        "        if abs(r) > 0.7:\n"
        "            direction = 'positiva' if r > 0 else 'negativa'\n"
        "            print(f'  {col_a} ↔ {col_b}: r={r:.3f} ({direction})')"
    ))

    # 6.2 Distribuciones
    cells.append(_cell_markdown("### 6.2 Distribuciones"))
    cells.append(_cell_code(
        "# Histogramas de variables numéricas\n"
        "cols_to_plot = numeric_cols[:8]\n"
        "n = len(cols_to_plot)\n"
        "if n > 0:\n"
        "    rows = (n + 1) // 2\n"
        "    fig = make_subplots(rows=rows, cols=2, subplot_titles=cols_to_plot)\n"
        "    for i, col in enumerate(cols_to_plot):\n"
        "        r, c = i // 2 + 1, i % 2 + 1\n"
        "        fig.add_trace(go.Histogram(x=df_train[col].dropna(), name=col,\n"
        "                                   nbinsx=30, showlegend=False), row=r, col=c)\n"
        "    fig.update_layout(title='Distribuciones', height=300*rows, width=900)\n"
        "    fig.show()"
    ))

    # 6.3 Boxplots + outliers
    cells.append(_cell_markdown(
        "### 6.3 Boxplots y Detección de Outliers\n"
        "\n"
        "**Método IQR (Rango Intercuartil × 1.5):**\n"
        "- Q1 = percentil 25, Q3 = percentil 75, IQR = Q3 - Q1\n"
        "- Outlier inferior: x < Q1 - 1.5×IQR\n"
        "- Outlier superior: x > Q3 + 1.5×IQR\n"
        "\n"
        "**Impacto en modelos:**\n"
        "- Modelos lineales son sensibles a outliers extremos\n"
        "- Árboles de decisión son más robustos\n"
        "- >5% de outliers en una variable → considerar transformación o capping"
    ))
    cells.append(_cell_code(
        "# Boxplots interactivos\n"
        "cols_to_box = numeric_cols[:8]\n"
        "fig = go.Figure()\n"
        "for col in cols_to_box:\n"
        "    fig.add_trace(go.Box(y=df_train[col].dropna(), name=col))\n"
        "fig.update_layout(title='Boxplots', height=500, width=900, showlegend=False)\n"
        "fig.show()\n"
        "\n"
        "# Cuantificar outliers con IQR 1.5x\n"
        "print('\\n=== Outliers (IQR × 1.5) ===')\n"
        "for col in numeric_cols:\n"
        "    data = df_train[col].dropna()\n"
        "    Q1 = data.quantile(0.25)\n"
        "    Q3 = data.quantile(0.75)\n"
        "    IQR = Q3 - Q1\n"
        "    n_outliers = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)).sum()\n"
        "    pct = n_outliers / len(data) * 100\n"
        "    if n_outliers > 0:\n"
        "        flag = ' ⚠ CRÍTICO' if pct > 5 else ''\n"
        "        print(f'  {col}: {n_outliers} outliers ({pct:.1f}%){flag}')"
    ))

    # 6.4 Target
    if target and target != "N/A":
        cells.append(_cell_markdown(f"### 6.4 Distribución del Target (`{target}`)"))
        cells.append(_cell_code(
            "# Distribución del target\n"
            "target_data = df_train[TARGET]\n"
            "if pd.api.types.is_numeric_dtype(target_data) and target_data.nunique() > 10:\n"
            "    fig = px.histogram(df_train, x=TARGET, nbins=30,\n"
            "                       title=f'Distribución: {TARGET}',\n"
            "                       color_discrete_sequence=['coral'])\n"
            "else:\n"
            "    vc = target_data.value_counts().reset_index()\n"
            "    vc.columns = [TARGET, 'count']\n"
            "    fig = px.bar(vc, x=TARGET, y='count',\n"
            "                 title=f'Distribución de clases: {TARGET}',\n"
            "                 color_discrete_sequence=['coral'])\n"
            "fig.update_layout(height=500, width=800)\n"
            "fig.show()\n"
            "\n"
            "print(f'\\nEstadísticas de {TARGET}:')\n"
            "print(target_data.describe())"
        ))

    # 6.5 Scatter matrix
    cells.append(_cell_markdown("### 6.5 Scatter Matrix"))
    cells.append(_cell_code(
        "# Scatter Matrix (top-6 numéricas)\n"
        "pair_cols = numeric_cols[:6]\n"
        "if len(pair_cols) >= 2:\n"
        "    pair_df = df_train[pair_cols].dropna()\n"
        "    if len(pair_df) > 500:\n"
        "        pair_df = pair_df.sample(500, random_state=RANDOM_SEED)\n"
        "    fig = px.scatter_matrix(pair_df, dimensions=pair_cols,\n"
        "                            title='Scatter Matrix — Top 6')\n"
        "    fig.update_layout(height=800, width=900)\n"
        "    fig.update_traces(diagonal_visible=True, marker=dict(size=3, opacity=0.5))\n"
        "    fig.show()"
    ))

    # ======================== 7. TESTS ESTADÍSTICOS ========================
    cells.append(_cell_markdown(
        "## 7. Tests Estadísticos\n"
        "\n"
        "Ejecutamos tests clave para informar la selección de modelos.\n"
        "**Todo se computa en vivo** — el investigador puede verificar cada paso."
    ))

    # 7.1 Normalidad
    cells.append(_cell_markdown(
        "### 7.1 Test de Normalidad (Shapiro-Wilk / Anderson-Darling)\n"
        "\n"
        "**¿Por qué evaluar normalidad?**\n"
        "- Los tests paramétricos (t-test, ANOVA, Pearson) asumen distribución normal\n"
        "- La regresión lineal asume normalidad de residuos para inferencia válida\n"
        "- Si la mayoría de variables NO son normales → usar métodos no paramétricos\n"
        "\n"
        "**Criterios:**\n"
        "- n < 5000: Shapiro-Wilk (mayor potencia en muestras pequeñas)\n"
        "- n ≥ 5000: Anderson-Darling (funciona en muestras grandes)\n"
        "- α = 0.05: p < 0.05 → rechazamos normalidad"
    ))
    cells.append(_cell_code(
        "# Test de normalidad para cada variable numérica\n"
        "normality_results = {}\n"
        "for col in numeric_cols:\n"
        "    data = df_train[col].dropna()\n"
        "    if len(data) < 8:\n"
        "        continue\n"
        "    if len(data) < 5000:\n"
        "        stat, p = sp_stats.shapiro(data)\n"
        "        normality_results[col] = {\n"
        "            'test': 'shapiro', 'stat': stat, 'p': p, 'normal': p > 0.05\n"
        "        }\n"
        "    else:\n"
        "        result = sp_stats.anderson(data)\n"
        "        is_normal = result.statistic < result.critical_values[2]\n"
        "        normality_results[col] = {\n"
        "            'test': 'anderson', 'stat': result.statistic, 'normal': is_normal\n"
        "        }\n"
        "\n"
        "# Resultados\n"
        "print('=== Test de Normalidad ===')\n"
        "n_normal = 0\n"
        "for col, info in normality_results.items():\n"
        "    symbol = '✓' if info['normal'] else '✗'\n"
        "    n_normal += info['normal']\n"
        "    label = 'Normal' if info['normal'] else 'No normal'\n"
        "    p_str = f\", p={info['p']:.4f}\" if 'p' in info else ''\n"
        "    print(f'  {symbol} {col}: {label} ({info[\"test\"]}, stat={info[\"stat\"]:.4f}{p_str})')\n"
        "\n"
        "print(f'\\nNormales: {n_normal}/{len(normality_results)}')\n"
        "if n_normal < len(normality_results) / 2:\n"
        "    print('→ Mayoría no normal → Spearman preferido sobre Pearson.')\n"
        "else:\n"
        "    print('→ Mayoría normal → tests paramétricos son válidos.')"
    ))

    # 7.2 Q-Q Plots
    cells.append(_cell_markdown("### 7.2 Q-Q Plots"))
    cells.append(_cell_code(
        "# Q-Q Plots de las primeras 6 variables\n"
        "qq_cols = numeric_cols[:6]\n"
        "n_qq = len(qq_cols)\n"
        "if n_qq > 0:\n"
        "    rows_qq = (n_qq + 1) // 2\n"
        "    fig = make_subplots(rows=rows_qq, cols=2,\n"
        "                        subplot_titles=[f'Q-Q: {c}' for c in qq_cols])\n"
        "    for i, col in enumerate(qq_cols):\n"
        "        data = df_train[col].dropna().values\n"
        "        if len(data) < 8:\n"
        "            continue\n"
        "        n_pts = min(len(data), 200)\n"
        "        theoretical = sp_stats.norm.ppf(np.linspace(0.01, 0.99, n_pts))\n"
        "        sample = np.sort(np.random.choice(data, size=n_pts, replace=False))\n"
        "        sample_z = (sample - np.mean(data)) / (np.std(data) + 1e-12)\n"
        "        r_, c_ = i // 2 + 1, i % 2 + 1\n"
        "        fig.add_trace(go.Scatter(x=theoretical, y=sample_z, mode='markers',\n"
        "                                 marker=dict(size=4, opacity=0.6),\n"
        "                                 showlegend=False), row=r_, col=c_)\n"
        "        fig.add_trace(go.Scatter(x=[-3, 3], y=[-3, 3], mode='lines',\n"
        "                                 line=dict(color='red', dash='dash'),\n"
        "                                 showlegend=False), row=r_, col=c_)\n"
        "    fig.update_layout(title_text='Q-Q Plots — Normalidad',\n"
        "                      height=300*rows_qq, template='plotly_white')\n"
        "    fig.show()"
    ))

    # 7.3 VIF
    cells.append(_cell_markdown(
        "### 7.3 VIF — Multicolinealidad\n"
        "\n"
        "**¿Qué mide el VIF?** Cuánto se explica una variable X por las demás features.\n"
        f"Si dos o más features están altamente correlacionadas, los coeficientes del modelo\n"
        f"se vuelven inestables y dificultan interpretar qué variables realmente afectan a `{target}`.\n"
        "\n"
        "**Umbrales:**\n"
        "| VIF | Nivel | Acción recomendada |\n"
        "|-----|-------|--------------------|\n"
        "| < 5 | OK | Sin problemas |\n"
        "| 5-10 | Moderado | Monitorear |\n"
        "| > 10 | Severo | Eliminar variable o usar regularización (Ridge/Lasso) |"
    ))
    cells.append(_cell_code(
        "# Calcular VIF para cada feature numérica\n"
        "feature_cols = [c for c in numeric_cols if c != TARGET]\n"
        "vif_input = df_train[feature_cols].dropna()\n"
        "\n"
        "# Remover columnas con varianza cero\n"
        "zero_var = vif_input.columns[vif_input.std() == 0].tolist()\n"
        "if zero_var:\n"
        "    print(f'Removidas (varianza=0): {zero_var}')\n"
        "    vif_input = vif_input.drop(columns=zero_var)\n"
        "\n"
        "if len(vif_input.columns) >= 2:\n"
        "    vif_values = []\n"
        "    for i in range(len(vif_input.columns)):\n"
        "        try:\n"
        "            v = variance_inflation_factor(vif_input.values, i)\n"
        "        except Exception:\n"
        "            v = float('inf')\n"
        "        vif_values.append(v)\n"
        "\n"
        "    vif_df = pd.DataFrame({\n"
        "        'feature': vif_input.columns,\n"
        "        'VIF': vif_values\n"
        "    }).sort_values('VIF', ascending=False)\n"
        "\n"
        "    # Gráfico de barras color-código\n"
        "    vif_plot = vif_df.sort_values('VIF', ascending=True).copy()\n"
        "    vif_plot['VIF_cap'] = vif_plot['VIF'].clip(upper=100)\n"
        "    colors = ['#e74c3c' if v > 10 else '#f39c12' if v > 5 else '#27ae60'\n"
        "              for v in vif_plot['VIF']]\n"
        "    fig = go.Figure(go.Bar(x=vif_plot['VIF_cap'], y=vif_plot['feature'],\n"
        "                           orientation='h', marker_color=colors))\n"
        "    fig.add_vline(x=10, line_dash='dash', line_color='red',\n"
        "                  annotation_text='VIF=10')\n"
        "    fig.add_vline(x=5, line_dash='dot', line_color='orange',\n"
        "                  annotation_text='VIF=5')\n"
        "    fig.update_layout(title='VIF — Multicolinealidad',\n"
        "                      xaxis_title='VIF', template='plotly_white',\n"
        "                      height=max(400, len(vif_df)*30), width=800)\n"
        "    fig.show()\n"
        "\n"
        "    # Resumen\n"
        "    n_high = (vif_df['VIF'] > 10).sum()\n"
        "    n_mod = ((vif_df['VIF'] > 5) & (vif_df['VIF'] <= 10)).sum()\n"
        "    n_ok = len(vif_df) - n_high - n_mod\n"
        "    print(f'\\nAlta (>10): {n_high} | Moderada (5-10): {n_mod} | OK (<5): {n_ok}')\n"
        "    if n_high > 0:\n"
        "        print('→ Multicolinealidad alta → regularización recomendada (Ridge/Lasso).')\n"
        "else:\n"
        "    print('Menos de 2 features numéricas para VIF.')\n"
        "    vif_df = pd.DataFrame()"
    ))

    # 7.4 Breusch-Pagan (solo regresión)
    if is_regression and target and target != "N/A":
        cells.append(_cell_markdown(
            "### 7.4 Breusch-Pagan — Heteroscedasticidad\n"
            "\n"
            "**¿Qué es la heteroscedasticidad?** Cuando la varianza de los errores del modelo\n"
            "NO es constante a lo largo de los valores predichos. Esto invalida los errores\n"
            "estándar OLS y los intervalos de confianza.\n"
            "\n"
            "**Hipótesis del test:**\n"
            "- H₀: Varianza constante (homoscedasticidad) → OLS válido\n"
            "- H₁: Varianza no constante → necesita corrección (HC3, WLS, o regularización)\n"
            "\n"
            "**Si p < 0.05:** Los errores estándar OLS están sesgados.\n"
            f"Para el modelo de `{target}`, esto significa que las inferencias pueden ser poco fiables."
        ))
        cells.append(_cell_code(
            "# Test de Breusch-Pagan\n"
            "bp_features = [c for c in numeric_cols if c != TARGET][:10]\n"
            "bp_data = df_train[[TARGET] + bp_features].dropna()\n"
            "\n"
            "if len(bp_data) > 20 and len(bp_features) >= 1:\n"
            "    y_bp = bp_data[TARGET].values\n"
            "    X_bp = sm.add_constant(bp_data[bp_features].values)\n"
            "    ols_model = sm.OLS(y_bp, X_bp).fit()\n"
            "\n"
            "    bp_stat, bp_pval, f_stat, f_pval = het_breuschpagan(ols_model.resid, X_bp)\n"
            "    is_heteroscedastic = bp_pval < 0.05\n"
            "\n"
            "    print('=== Test de Breusch-Pagan ===')\n"
            "    print(f'  BP Statistic: {bp_stat:.4f}')\n"
            "    print(f'  BP p-valor:   {bp_pval:.4f}')\n"
            "    print(f'  F Statistic:  {f_stat:.4f}')\n"
            "    print(f'  F p-valor:    {f_pval:.4f}')\n"
            "    print()\n"
            "    if is_heteroscedastic:\n"
            "        print('⚠ HETEROSCEDASTICIDAD detectada (p < 0.05).')\n"
            "        print('  Errores estándar OLS sesgados → usar HC3 o regularización.')\n"
            "    else:\n"
            "        print('✓ Homoscedasticidad (p ≥ 0.05). OLS clásico es válido.')\n"
            "\n"
            "    # Gráfico de residuos vs ajustados\n"
            "    residuals = ols_model.resid\n"
            "    fitted = ols_model.fittedvalues\n"
            "    lbl = 'Heteroscedástico' if is_heteroscedastic else 'Homoscedástico'\n"
            "    fig = go.Figure()\n"
            "    fig.add_trace(go.Scatter(x=fitted, y=residuals, mode='markers',\n"
            "                             marker=dict(size=4, opacity=0.5, color='#3498db'),\n"
            "                             name='Residuos'))\n"
            "    fig.add_hline(y=0, line_dash='dash', line_color='red')\n"
            "    fig.update_layout(title=f'Residuos vs Ajustados ({lbl})',\n"
            "                      xaxis_title='Ajustados', yaxis_title='Residuos',\n"
            "                      template='plotly_white', height=500, width=800)\n"
            "    fig.show()\n"
            "else:\n"
            "    print('Datos insuficientes para Breusch-Pagan.')"
        ))

    # ======================== 8. FEATURE IMPORTANCE ========================
    mi_func = "mutual_info_regression" if is_regression else "mutual_info_classif"
    rf_class = "RandomForestRegressor" if is_regression else "RandomForestClassifier"
    rf_scoring = "neg_mean_squared_error" if is_regression else "accuracy"

    cells.append(_cell_markdown(
        "## 8. Importancia de Variables\n"
        "\n"
        "Usamos dos métodos complementarios para una evaluación robusta:\n"
        "\n"
        "| Método | Tipo | Fortaleza |\n"
        "|--------|------|----------|\n"
        "| **Mutual Information** | Filter (sin modelo) | Detecta relaciones no lineales, rápido |\n"
        "| **Permutation Importance** | Wrapper (con modelo) | Mide impacto real en predicción |\n"
        "\n"
        "**¿Por qué combinar ambos?** MI puede sobrevalorar variables con muchos valores únicos,\n"
        "mientras que Permutation depende del modelo elegido. Al promediar rankings, obtenemos\n"
        f"una visión más equilibrada de qué variables son más informativas para predecir `{target}`."
    ))

    cells.append(_cell_markdown("### 8.1 Mutual Information"))
    cells.append(_cell_code(
        "# Mutual Information\n"
        "mi_features = [c for c in numeric_cols if c != TARGET]\n"
        "mi_data = df_train[mi_features + [TARGET]].dropna()\n"
        "X_mi = mi_data[mi_features]\n"
        "y_mi = mi_data[TARGET]\n"
        "\n"
        f"mi_scores = {mi_func}(X_mi, y_mi, random_state=RANDOM_SEED)\n"
        "mi_df = pd.DataFrame({'feature': mi_features, 'MI': mi_scores})\n"
        "mi_df = mi_df.sort_values('MI', ascending=True)\n"
        "\n"
        "fig = go.Figure(go.Bar(x=mi_df['MI'], y=mi_df['feature'], orientation='h',\n"
        "                       marker_color='steelblue'))\n"
        "fig.update_layout(title='Mutual Information', xaxis_title='MI Score',\n"
        "                  template='plotly_white',\n"
        "                  height=max(300, len(mi_df)*25), width=800)\n"
        "fig.show()\n"
        "\n"
        "print('Top 5 MI:')\n"
        "for _, row in mi_df.sort_values('MI', ascending=False).head(5).iterrows():\n"
        "    print(f'  {row[\"feature\"]}: {row[\"MI\"]:.4f}')"
    ))

    cells.append(_cell_markdown("### 8.2 Permutation Importance"))
    cells.append(_cell_code(
        f"# Permutation Importance con RandomForest\n"
        "X_perm = mi_data[mi_features].fillna(0)\n"
        "y_perm = mi_data[TARGET]\n"
        "\n"
        f"rf_model = {rf_class}(\n"
        "    n_estimators=50, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1\n"
        ")\n"
        "rf_model.fit(X_perm, y_perm)\n"
        "\n"
        "perm_result = permutation_importance(\n"
        f"    rf_model, X_perm, y_perm, n_repeats=5, scoring='{rf_scoring}',\n"
        "    random_state=RANDOM_SEED\n"
        ")\n"
        "\n"
        "perm_df = pd.DataFrame({\n"
        "    'feature': mi_features,\n"
        "    'importance': perm_result.importances_mean,\n"
        "    'std': perm_result.importances_std\n"
        "}).sort_values('importance', ascending=True)\n"
        "\n"
        "fig = go.Figure(go.Bar(\n"
        "    x=perm_df['importance'], y=perm_df['feature'], orientation='h',\n"
        "    error_x=dict(type='data', array=perm_df['std']),\n"
        "    marker_color='darkorange'\n"
        "))\n"
        "fig.update_layout(title='Permutation Importance', xaxis_title='Importance',\n"
        "                  template='plotly_white',\n"
        "                  height=max(300, len(perm_df)*25), width=800)\n"
        "fig.show()\n"
        "\n"
        "print('Top 5 Permutation:')\n"
        "for _, row in perm_df.sort_values('importance', ascending=False).head(5).iterrows():\n"
        "    print(f'  {row[\"feature\"]}: {row[\"importance\"]:.4f} ± {row[\"std\"]:.4f}')"
    ))

    cells.append(_cell_markdown("### 8.3 Ranking Combinado"))
    cells.append(_cell_code(
        "# Combinar rankings: promedio de posición en MI y Permutation\n"
        "mi_ranked = mi_df.sort_values('MI', ascending=False).reset_index(drop=True)\n"
        "mi_ranked['rank_mi'] = range(1, len(mi_ranked) + 1)\n"
        "\n"
        "perm_ranked = perm_df.sort_values('importance', ascending=False).reset_index(drop=True)\n"
        "perm_ranked['rank_perm'] = range(1, len(perm_ranked) + 1)\n"
        "\n"
        "combined = mi_ranked[['feature', 'rank_mi']].merge(\n"
        "    perm_ranked[['feature', 'rank_perm']], on='feature'\n"
        ")\n"
        "combined['avg_rank'] = (combined['rank_mi'] + combined['rank_perm']) / 2\n"
        "combined = combined.sort_values('avg_rank')\n"
        "\n"
        "print('=== Ranking Combinado ===')\n"
        "for _, row in combined.iterrows():\n"
        "    print(f\"  {row['feature']}: avg_rank={row['avg_rank']:.1f} \"\n"
        "          f\"(MI={row['rank_mi']}, Perm={row['rank_perm']})\")\n"
        "\n"
        "top_features = combined['feature'].head(min(8, len(combined))).tolist()\n"
        "print(f'\\n→ Top features: {top_features}')"
    ))

    # ======================== 9. CONCLUSIONES Y MODELADO ========================
    model_names = ", ".join(
        m.get("name", "?") for m in modelos if isinstance(m, dict)
    )
    cells.append(_cell_markdown(
        f"## 9. Conclusiones y Entrenamiento del Modelo\n"
        f"\n"
        f"### Conexión con las Hipótesis\n"
        f"\n"
        f"Los hallazgos computados arriba permiten evaluar las hipótesis de investigación:\n"
        f"\n"
        f"- **H1:** {hip.get('h1', 'N/A')}\n"
        f"  - → Revisar correlaciones y feature importance para evidencia\n"
        f"- **H2:** {hip.get('h2', 'N/A')}\n"
        f"  - → Revisar distribuciones y patrones exploratorios\n"
        f"- **H3:** {hip.get('h3', 'N/A')}\n"
        f"  - → Considerar si los resultados sugieren un enfoque alternativo\n"
        f"\n"
        f"El modelo baseline a continuación sirve como **primera validación cuantitativa**\n"
        f"de estas hipótesis."
    ))

    cells.append(_cell_code(
        "# Resumen automático de hallazgos\n"
        "print('=' * 60)\n"
        "print('RESUMEN DE HALLAZGOS (computados en vivo)')\n"
        "print('=' * 60)\n"
        "\n"
        "n_norm = sum(1 for v in normality_results.values() if v['normal'])\n"
        "print(f'\\n📊 Normalidad: {n_norm}/{len(normality_results)} variables normales')\n"
        "\n"
        "if len(vif_df) > 0:\n"
        "    n_high_vif = (vif_df['VIF'] > 10).sum()\n"
        "    print(f'🔗 VIF: {n_high_vif} variables con multicolinealidad alta')\n"
        "\n"
        "if 'is_heteroscedastic' in dir():\n"
        "    bp_lbl = 'Heteroscedástico' if is_heteroscedastic else 'Homoscedástico'\n"
        "    print(f'📈 Breusch-Pagan: {bp_lbl} (p={bp_pval:.4f})')\n"
        "\n"
        "print(f'🏆 Top features: {top_features}')\n"
        "print('=' * 60)"
    ))

    cells.append(_cell_markdown(
        f"### Entrenamiento\n"
        f"\n"
        f"- **Familia:** `{model_family}`\n"
        f"- **Modelos recomendados:** `{model_names}`\n"
        f"- **Métrica:** `{metrica}`\n"
        f"- **Hiperparámetros:** `{hp_technique}`"
    ))

    if is_regression:
        cells.append(_cell_code(
            "from sklearn.linear_model import Ridge\n"
            "from sklearn.metrics import mean_squared_error, r2_score\n"
            "\n"
            "# Preparar datos\n"
            "X_train_model = df_train.drop(columns=[TARGET]).select_dtypes(include=[np.number]).fillna(0)\n"
            "y_train_model = df_train[TARGET]\n"
            "\n"
            "# Ridge con GridSearchCV\n"
            "ridge = Ridge()\n"
            "params = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}\n"
            "grid = GridSearchCV(ridge, params, cv=5, scoring='neg_mean_squared_error')\n"
            "grid.fit(X_train_model, y_train_model)\n"
            "\n"
            "print('=== Entrenamiento ===')\n"
            "print(f'Modelo: Ridge')\n"
            "print(f'Mejor alpha: {grid.best_params_}')\n"
            "print(f'RMSE (CV): {(-grid.best_score_)**0.5:.4f}')\n"
            "\n"
            "# Evaluar en test\n"
            "X_test_model = df_test.drop(columns=[TARGET]).select_dtypes(include=[np.number]).fillna(0)\n"
            "X_test_model = X_test_model.reindex(columns=X_train_model.columns, fill_value=0)\n"
            "y_test_model = df_test[TARGET]\n"
            "\n"
            "y_pred = grid.predict(X_test_model)\n"
            "rmse = mean_squared_error(y_test_model, y_pred, squared=False)\n"
            "r2 = r2_score(y_test_model, y_pred)\n"
            "print(f'\\n=== Evaluación en Test ===')\n"
            "print(f'RMSE: {rmse:.4f}')\n"
            "print(f'R²:   {r2:.4f}')"
        ))
    else:
        cells.append(_cell_code(
            "from sklearn.metrics import accuracy_score, classification_report\n"
            "\n"
            "X_train_model = df_train.drop(columns=[TARGET]).select_dtypes(include=[np.number]).fillna(0)\n"
            "y_train_model = df_train[TARGET]\n"
            "\n"
            "rf = RandomForestClassifier(random_state=RANDOM_SEED)\n"
            "params = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}\n"
            "grid = GridSearchCV(rf, params, cv=5, scoring='accuracy')\n"
            "grid.fit(X_train_model, y_train_model)\n"
            "\n"
            "print('=== Entrenamiento ===')\n"
            "print(f'Modelo: RandomForest')\n"
            "print(f'Mejores params: {grid.best_params_}')\n"
            "print(f'Accuracy (CV): {grid.best_score_:.4f}')\n"
            "\n"
            "X_test_model = df_test.drop(columns=[TARGET]).select_dtypes(include=[np.number]).fillna(0)\n"
            "X_test_model = X_test_model.reindex(columns=X_train_model.columns, fill_value=0)\n"
            "y_test_model = df_test[TARGET]\n"
            "\n"
            "y_pred = grid.predict(X_test_model)\n"
            "print(f'\\n=== Evaluación en Test ===')\n"
            "print(f'Accuracy: {accuracy_score(y_test_model, y_pred):.4f}')\n"
            "print(classification_report(y_test_model, y_pred))"
        ))

    # Footer
    cells.append(_cell_markdown(
        "---\n"
        f"*Generado por EDA Agents (Run: `{run_id}`)*\n"
        "\n"
        "**Todo el código es ejecutable.** Los resultados se computan en vivo\n"
        "desde el dataset original — el investigador puede verificar cada paso."
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
