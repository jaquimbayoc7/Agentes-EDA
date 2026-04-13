"""Skill reutilizable — Tests estadísticos.

Funciones puras para análisis estadístico EDA:
- Correlaciones (Pearson, Spearman)
- IQR outliers
- Normalidad (Shapiro-Wilk / Anderson-Darling)
- VIF (multicolinealidad)
- Breusch-Pagan (heteroscedasticidad)
- Corrección de heteroscedasticidad (WLS / HC3)

Usada por Agent 04 (Statistician) y cualquier componente que necesite
tests estadísticos.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Correlaciones
# ---------------------------------------------------------------------------


def compute_correlations(
    df: pd.DataFrame,
    features: list[str],
    methods: list[str] | None = None,
) -> dict[str, Any]:
    """Calcula matrices de correlación para las columnas numéricas.

    Parameters
    ----------
    df : DataFrame con columnas numéricas.
    features : lista de columnas a incluir.
    methods : métodos de correlación ("pearson", "spearman"). Default: ["pearson"].

    Returns
    -------
    dict con clave por método, valor = dict-de-dict (serializable).
    """
    if methods is None:
        methods = ["pearson"]

    result: dict[str, Any] = {}
    valid_cols = [c for c in features if c in df.columns]
    if not valid_cols:
        return result

    for method in methods:
        corr = df[valid_cols].corr(method=method)
        result[method] = corr.to_dict()

    return result


# ---------------------------------------------------------------------------
# Outliers IQR
# ---------------------------------------------------------------------------


def detect_outliers_iqr(
    df: pd.DataFrame, features: list[str]
) -> dict[str, dict[str, Any]]:
    """Detecta outliers por método IQR (1.5 × IQR).

    Returns
    -------
    dict col → { n_outliers, pct, lower, upper }
    """
    summary: dict[str, dict[str, Any]] = {}
    for col in features:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        n_outliers = int(mask.sum())
        summary[col] = {
            "n_outliers": n_outliers,
            "pct": round(float(n_outliers / len(df) * 100), 2) if len(df) > 0 else 0.0,
            "lower_bound": float(lower),
            "upper_bound": float(upper),
        }
    return summary


# ---------------------------------------------------------------------------
# Normalidad
# ---------------------------------------------------------------------------


def test_normality(
    df: pd.DataFrame,
    features: list[str],
    max_cols: int = 20,
    shapiro_threshold: int = 5000,
) -> dict[str, dict[str, Any]]:
    """Tests de normalidad: Shapiro-Wilk (n < threshold) o Anderson-Darling.

    Parameters
    ----------
    max_cols : máximo de columnas a testear.
    shapiro_threshold : si n >= threshold, usar Anderson-Darling en lugar de Shapiro.

    Returns
    -------
    dict col → { test, statistic, p_value | critical_values }
    """
    from scipy import stats

    result: dict[str, dict[str, Any]] = {}
    n = len(df)

    for col in features[:max_cols]:
        if col not in df.columns:
            continue
        clean = df[col].dropna()
        if len(clean) < 8:
            continue

        if n < shapiro_threshold:
            stat, pval = stats.shapiro(clean)
            result[col] = {
                "test": "shapiro",
                "statistic": float(stat),
                "p_value": float(pval),
                "normal": bool(pval > 0.05),
            }
        else:
            ad = stats.anderson(clean)
            result[col] = {
                "test": "anderson",
                "statistic": float(ad.statistic),
                "critical_values": [float(v) for v in ad.critical_values],
                "significance_levels": [float(s) for s in ad.significance_level],
            }

    return result


# ---------------------------------------------------------------------------
# VIF
# ---------------------------------------------------------------------------


def compute_vif(
    df: pd.DataFrame, features: list[str], threshold: float = 10.0
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Calcula VIF para detectar multicolinealidad.

    Returns
    -------
    (flagged, all_vif) donde:
        flagged = lista de { column, vif } con VIF > threshold
        all_vif = dict col → valor VIF
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    valid_cols = [c for c in features if c in df.columns]
    if len(valid_cols) < 2:
        return [], {}

    vif_df = df[valid_cols].dropna()
    if len(vif_df) == 0:
        return [], {}

    all_vif: dict[str, float] = {}
    flagged: list[dict[str, Any]] = []

    for i, col in enumerate(valid_cols):
        vif_val = variance_inflation_factor(vif_df.values, i)
        all_vif[col] = float(vif_val)
        if vif_val > threshold:
            flagged.append({"column": col, "vif": float(vif_val)})

    return flagged, all_vif


# ---------------------------------------------------------------------------
# Breusch-Pagan
# ---------------------------------------------------------------------------


def breusch_pagan_test(
    df: pd.DataFrame, target: str, features: list[str], alpha: float = 0.05
) -> dict[str, Any]:
    """Ejecuta test de Breusch-Pagan para heteroscedasticidad.

    Solo aplica en tareas de regresión.

    Returns
    -------
    dict con bp_statistic, bp_pvalue, f_statistic, f_pvalue, heteroscedastic.
    """
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan

    valid_features = [c for c in features if c in df.columns and c != target]
    if not valid_features or target not in df.columns:
        return {"error": "insufficient_columns"}

    clean = df[[target] + valid_features].dropna()
    if len(clean) < len(valid_features) + 2:
        return {"error": "insufficient_rows"}

    y = clean[target].values
    X = sm.add_constant(clean[valid_features].values)

    ols_model = sm.OLS(y, X).fit()
    bp_stat, bp_pval, f_stat, f_pval = het_breuschpagan(ols_model.resid, X)

    return {
        "bp_statistic": float(bp_stat),
        "bp_pvalue": float(bp_pval),
        "f_statistic": float(f_stat),
        "f_pvalue": float(f_pval),
        "heteroscedastic": bool(bp_pval < alpha),
    }


def suggest_heteroscedasticity_correction(
    bp_result: dict[str, Any],
    vif_flagged: list[dict[str, Any]],
) -> str | None:
    """Sugiere método de corrección de heteroscedasticidad.

    - Si heteroscedastic + VIF alto → GLS
    - Si heteroscedastic sin VIF alto → WLS
    - Si nada converge → HC3 (robust SE)
    - Si no heteroscedastic → None
    """
    if not bp_result.get("heteroscedastic", False):
        return None

    if vif_flagged:
        return "GLS"
    return "WLS"
