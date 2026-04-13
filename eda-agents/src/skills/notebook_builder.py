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
        "# Correlaciones\n"
        "numeric_df = df_train.select_dtypes(include=[np.number])\n"
        "if len(numeric_df.columns) >= 2:\n"
        "    corr = numeric_df.corr()\n"
        "    fig, ax = plt.subplots(figsize=(10, 8))\n"
        "    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)\n"
        "    ax.set_title('Matriz de Correlaciones')\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "else:\n"
        "    print('Menos de 2 columnas numericas para correlacion.')"
    ))

    cells.append(_cell_code(
        "# Distribuciones de variables numericas\n"
        "numeric_cols = numeric_df.columns.tolist()\n"
        "n_cols = len(numeric_cols)\n"
        "if n_cols > 0:\n"
        "    fig, axes = plt.subplots(1, min(n_cols, 4), figsize=(5*min(n_cols, 4), 4))\n"
        "    if n_cols == 1:\n"
        "        axes = [axes]\n"
        "    for i, col in enumerate(numeric_cols[:4]):\n"
        "        axes[i].hist(df_train[col].dropna(), bins=30, edgecolor='black', alpha=0.7)\n"
        "        axes[i].set_title(f'Dist: {col}')\n"
        "    plt.tight_layout()\n"
        "    plt.show()"
    ))

    cells.append(_cell_code(
        "# Boxplots\n"
        "if n_cols > 0:\n"
        "    fig, axes = plt.subplots(1, min(n_cols, 4), figsize=(5*min(n_cols, 4), 4))\n"
        "    if n_cols == 1:\n"
        "        axes = [axes]\n"
        "    for i, col in enumerate(numeric_cols[:4]):\n"
        "        axes[i].boxplot(df_train[col].dropna(), patch_artist=True)\n"
        "        axes[i].set_title(f'Box: {col}')\n"
        "    plt.tight_layout()\n"
        "    plt.show()"
    ))

    if target and target != "N/A":
        cells.append(_cell_code(
            f"# Distribucion del target\n"
            f"if '{target}' in df_train.columns:\n"
            f"    fig, ax = plt.subplots(figsize=(8, 5))\n"
            f"    if pd.api.types.is_numeric_dtype(df_train['{target}']) and df_train['{target}'].nunique() > 10:\n"
            f"        ax.hist(df_train['{target}'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='coral')\n"
            f"        ax.set_title('Distribucion del target: {target}')\n"
            f"    else:\n"
            f"        df_train['{target}'].value_counts().plot(kind='bar', ax=ax, color='coral', edgecolor='black')\n"
            f"        ax.set_title('Distribucion de clases: {target}')\n"
            f"    plt.tight_layout()\n"
            f"    plt.show()"
        ))

    cells.append(_cell_code(
        "# Pairplot (top-6 numericas)\n"
        "pair_cols = numeric_cols[:6]\n"
        "if len(pair_cols) >= 2:\n"
        "    pair_df = df_train[pair_cols].dropna()\n"
        "    if len(pair_df) > 500:\n"
        "        pair_df = pair_df.sample(500, random_state=RANDOM_SEED)\n"
        "    g = sns.pairplot(pair_df, diag_kind='kde', plot_kws={'alpha': 0.5, 's': 15})\n"
        "    g.figure.suptitle('Pairplot Top-6', y=1.02)\n"
        "    plt.show()"
    ))

    # ======================== 7. HALLAZGOS ========================
    cells.append(_cell_markdown("## 7. Hallazgos del Pipeline"))
    cells.append(_cell_code(
        f"hallazgos = json.loads('''{hallazgos_json}''')\n"
        f"\n"
        f"# Correlaciones encontradas\n"
        f"if 'correlations' in hallazgos:\n"
        f"    print('=== Correlaciones ===')\n"
        f"    print(json.dumps(hallazgos['correlations'], indent=2))\n"
        f"\n"
        f"# Outliers\n"
        f"if 'outliers' in hallazgos:\n"
        f"    print('\\n=== Outliers ===')\n"
        f"    for col, info in hallazgos['outliers'].items():\n"
        f"        print(f'  {{col}}: {{info.get(\"n_outliers\", 0)}} outliers ({{info.get(\"pct\", 0):.1f}}%)')\n"
        f"\n"
        f"# Normalidad\n"
        f"if 'normality' in hallazgos:\n"
        f"    print('\\n=== Test de Normalidad ===')\n"
        f"    for col, info in hallazgos['normality'].items():\n"
        f"        normal = 'Si' if info.get('normal') else 'No'\n"
        f"        print(f'  {{col}}: {{normal}} (p={{info.get(\"p_value\", 0):.4f}})')"
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
            "from IPython.display import Image, display\n"
            "\n"
            "# Mostrar figuras generadas por el pipeline\n"
            "figuras = [\n" +
            "".join(f"    (r'{fp}', '{desc}'),\n" for fp, desc in fig_paths) +
            "]\n"
            "\n"
            "for path, desc in figuras:\n"
            "    try:\n"
            "        print(f'--- {desc} ---')\n"
            "        display(Image(filename=path))\n"
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
