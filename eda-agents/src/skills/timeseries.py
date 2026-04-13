"""Skill reutilizable — Análisis de series temporales.

Funciones puras para:
- Tests de estacionariedad (ADF + KPSS)
- Determinación de orden de diferenciación (d)
- Diagnóstico de residuos (Ljung-Box, Jarque-Bera)
- Selección de modelo ARIMA/SARIMA (stub para pmdarima)

Usada por Agent 05 (TS Analyst).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Estacionariedad
# ---------------------------------------------------------------------------


def test_stationarity(series: pd.Series, alpha: float = 0.05) -> dict[str, Any]:
    """Ejecuta ADF + KPSS y combina resultados.

    Interpretación conjunta:
        ADF stat < crit & KPSS stat < crit → estacionaria
        ADF no-reject & KPSS reject → no estacionaria
        Conflicto → trend-stationary (necesita más análisis)

    Returns
    -------
    dict con resultados de cada test y conclusión combinada.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    clean = series.dropna()
    if len(clean) < 10:
        return {"error": "insufficient_data", "n": len(clean)}

    result: dict[str, Any] = {}

    # ADF
    try:
        adf = adfuller(clean, autolag="AIC")
        result["adf_statistic"] = float(adf[0])
        result["adf_pvalue"] = float(adf[1])
        result["adf_lags"] = int(adf[2])
        result["adf_stationary"] = bool(adf[1] < alpha)
    except Exception as e:
        result["adf_error"] = str(e)

    # KPSS
    try:
        kpss_result = kpss(clean, regression="c", nlags="auto")
        result["kpss_statistic"] = float(kpss_result[0])
        result["kpss_pvalue"] = float(kpss_result[1])
        result["kpss_stationary"] = bool(kpss_result[1] > alpha)
    except Exception as e:
        result["kpss_error"] = str(e)

    # Conclusión combinada
    adf_stat = result.get("adf_stationary")
    kpss_stat = result.get("kpss_stationary")

    if adf_stat is True and kpss_stat is True:
        result["conclusion"] = "stationary"
    elif adf_stat is False and kpss_stat is False:
        result["conclusion"] = "non_stationary"
    elif adf_stat is True and kpss_stat is False:
        result["conclusion"] = "trend_stationary"
    elif adf_stat is False and kpss_stat is True:
        result["conclusion"] = "difference_stationary"
    else:
        result["conclusion"] = "inconclusive"

    return result


def determine_differencing_order(
    series: pd.Series, max_d: int = 2, alpha: float = 0.05
) -> int:
    """Determina d (orden de diferenciación) por ADF iterativo.

    Diferencia la serie hasta que ADF rechace H0 o d alcance max_d.
    """
    from statsmodels.tsa.stattools import adfuller

    current = series.dropna()
    for d in range(max_d + 1):
        if len(current) < 10:
            return d
        try:
            _, pval, *_ = adfuller(current, autolag="AIC")
            if pval < alpha:
                return d
        except Exception:
            return d
        current = current.diff().dropna()

    return max_d


# ---------------------------------------------------------------------------
# Diagnóstico de residuos
# ---------------------------------------------------------------------------


def diagnose_residuals(residuals: pd.Series | np.ndarray) -> dict[str, Any]:
    """Diagnóstico de residuos de modelo de series temporales.

    Tests:
    - Ljung-Box: autocorrelación residual (p > 0.05 → residuos blancos)
    - Jarque-Bera: normalidad de residuos
    - Mean / Std de residuos

    Returns
    -------
    dict con resultados de cada test.
    """
    from scipy import stats

    if isinstance(residuals, np.ndarray):
        residuals = pd.Series(residuals)

    clean = residuals.dropna()
    result: dict[str, Any] = {
        "n_residuals": len(clean),
        "mean": float(clean.mean()),
        "std": float(clean.std()),
    }

    if len(clean) < 10:
        result["error"] = "insufficient_residuals"
        return result

    # Ljung-Box
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lb = acorr_ljungbox(clean, lags=[10], return_df=True)
        result["ljung_box"] = {
            "statistic": float(lb["lb_stat"].iloc[0]),
            "p_value": float(lb["lb_pvalue"].iloc[0]),
            "white_noise": bool(lb["lb_pvalue"].iloc[0] > 0.05),
        }
    except Exception as e:
        result["ljung_box_error"] = str(e)

    # Jarque-Bera
    try:
        jb_stat, jb_pval = stats.jarque_bera(clean)
        result["jarque_bera"] = {
            "statistic": float(jb_stat),
            "p_value": float(jb_pval),
            "normal": bool(jb_pval > 0.05),
        }
    except Exception as e:
        result["jarque_bera_error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Selección de modelo (pmdarima auto_arima)
# ---------------------------------------------------------------------------


def select_ts_model(
    series: pd.Series,
    d: int = 1,
    seasonal: bool = False,
    m: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Selecciona modelo ARIMA/SARIMA usando pmdarima.auto_arima.

    Falls back to heuristic defaults if pmdarima is unavailable.

    Returns
    -------
    (modelo_info, params_pdq)
    """
    try:
        from pmdarima import auto_arima

        clean = series.dropna()
        if len(clean) < 20:
            raise ValueError("Series too short for auto_arima")

        result = auto_arima(
            clean,
            d=d,
            seasonal=seasonal,
            m=m if seasonal and m > 1 else 1,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_p=5,
            max_q=5,
            max_order=10,
            trace=False,
        )

        order = result.order  # (p, d, q)
        modelo: dict[str, Any] = {
            "type": "SARIMA" if seasonal and m > 1 else "ARIMA",
            "selected": True,
            "aic": float(result.aic()),
            "bic": float(result.bic()),
            "note": f"auto_arima selected order {order}",
        }
        params: dict[str, Any] = {
            "p": order[0],
            "d": order[1],
            "q": order[2],
            "seasonal": seasonal and m > 1,
        }
        if seasonal and m > 1:
            seasonal_order = result.seasonal_order  # (P, D, Q, m)
            params.update({
                "P": seasonal_order[0],
                "D": seasonal_order[1],
                "Q": seasonal_order[2],
                "m": seasonal_order[3],
            })
            modelo["seasonal_order"] = list(seasonal_order)

        return modelo, params

    except Exception:
        # Fallback heurístico
        if seasonal and m > 1:
            modelo = {
                "type": "SARIMA",
                "selected": True,
                "note": "Seasonal pattern detected (fallback defaults)",
            }
            params = {
                "p": 1, "d": d, "q": 1,
                "P": 1, "D": 1, "Q": 1, "m": m,
                "seasonal": True,
            }
        else:
            modelo = {
                "type": "ARIMA",
                "selected": True,
                "note": "Non-seasonal model (fallback defaults)",
            }
            params = {
                "p": 1, "d": d, "q": 1,
                "seasonal": False,
            }

        return modelo, params
