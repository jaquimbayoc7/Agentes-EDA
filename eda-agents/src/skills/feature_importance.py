"""Skill — Feature Importance y Feature Selection.

Funciones puras para calcular importancia de variables:
- Mutual Information (MI) para regresión y clasificación
- Permutation Importance (wrapper sobre sklearn)
- Selección automática de top-K features

Usada por Agent 04 (Statistician) para enriquecer hallazgos EDA.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_mutual_information(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    task: str = "regression",
    random_state: int = 42,
) -> dict[str, float]:
    """Calcula Mutual Information entre cada feature y el target.

    Parameters
    ----------
    df : DataFrame con features y target.
    features : lista de columnas numéricas.
    target : nombre de la columna target.
    task : "regression" o "classification".
    random_state : semilla para reproducibilidad.

    Returns
    -------
    dict feature → MI score, ordenado descendente.
    """
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    valid_cols = [c for c in features if c in df.columns and c != target]
    if not valid_cols or target not in df.columns:
        return {}

    clean = df[valid_cols + [target]].dropna()
    if len(clean) < 10:
        return {}

    X = clean[valid_cols].values
    y = clean[target].values

    mi_func = mutual_info_regression if task == "regression" else mutual_info_classif
    mi_scores = mi_func(X, y, random_state=random_state)

    result = {col: round(float(score), 6) for col, score in zip(valid_cols, mi_scores)}
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


def compute_permutation_importance(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    task: str = "regression",
    random_state: int = 42,
    n_repeats: int = 5,
) -> dict[str, dict[str, float]]:
    """Calcula Permutation Importance con un modelo rápido (RF con pocos árboles).

    Parameters
    ----------
    df : DataFrame con features y target.
    features : lista de columnas numéricas.
    target : nombre de la columna target.
    task : "regression" o "classification".
    random_state : semilla para reproducibilidad.
    n_repeats : número de repeticiones del permutation test.

    Returns
    -------
    dict feature → {mean, std}, ordenado por mean descendente.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.inspection import permutation_importance

    valid_cols = [c for c in features if c in df.columns and c != target]
    if not valid_cols or target not in df.columns:
        return {}

    clean = df[valid_cols + [target]].dropna()
    if len(clean) < 20:
        return {}

    X = clean[valid_cols].values
    y = clean[target].values

    if task == "regression":
        model = RandomForestRegressor(
            n_estimators=50, max_depth=8, random_state=random_state, n_jobs=-1,
        )
        scoring = "neg_mean_squared_error"
    else:
        model = RandomForestClassifier(
            n_estimators=50, max_depth=8, random_state=random_state, n_jobs=-1,
        )
        scoring = "accuracy"

    model.fit(X, y)
    perm = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
    )

    result = {}
    for i, col in enumerate(valid_cols):
        result[col] = {
            "mean": round(float(perm.importances_mean[i]), 6),
            "std": round(float(perm.importances_std[i]), 6),
        }

    return dict(sorted(result.items(), key=lambda x: x[1]["mean"], reverse=True))


def select_top_features(
    mi_scores: dict[str, float],
    perm_scores: dict[str, dict[str, float]],
    top_k: int = 10,
) -> list[str]:
    """Selecciona las top-K features combinando MI y Permutation Importance.

    Usa ranking promedio de ambos métodos.

    Parameters
    ----------
    mi_scores : dict feature → MI score.
    perm_scores : dict feature → {mean, std}.
    top_k : número máximo de features a retornar.

    Returns
    -------
    Lista de nombres de features, ordenada por importancia combinada.
    """
    all_features = set(mi_scores.keys()) | set(perm_scores.keys())
    if not all_features:
        return []

    # Rank MI (1 = best)
    mi_ranked = list(mi_scores.keys())  # already sorted desc
    mi_rank = {f: i + 1 for i, f in enumerate(mi_ranked)}

    # Rank Permutation (1 = best)
    perm_ranked = list(perm_scores.keys())  # already sorted desc
    perm_rank = {f: i + 1 for i, f in enumerate(perm_ranked)}

    n = len(all_features)
    combined = {}
    for f in all_features:
        r_mi = mi_rank.get(f, n)
        r_perm = perm_rank.get(f, n)
        combined[f] = (r_mi + r_perm) / 2.0

    sorted_features = sorted(combined.items(), key=lambda x: x[1])
    return [f for f, _ in sorted_features[:top_k]]
