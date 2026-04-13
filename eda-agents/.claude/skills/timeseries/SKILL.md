# Skill: Análisis de series temporales

## Descripción
Herramientas para análisis y modelado de series de tiempo.

## Ubicación
`src/skills/timeseries.py`

## Funciones
- `test_stationarity(series)` → dict (ADF + KPSS)
- `select_arima_model(series, exog, seasonal_period)` → dict
  - Usa pmdarima.auto_arima con AIC/BIC
  - Retorna: modelo, (p,d,q), tipo, diagnóstico
- `diagnose_residuals(model, fitted)` → dict (Ljung-Box, Jarque-Bera)

## Lógica de selección
- Sin estacionalidad → ARIMA(p,d,q)
- Con estacionalidad (STL) → SARIMA(p,d,q)(P,D,Q,s)
- Con exógenas → SARIMAX
- Múltiples series → VAR
