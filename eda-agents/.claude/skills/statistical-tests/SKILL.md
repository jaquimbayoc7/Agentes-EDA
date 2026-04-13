# Skill: Tests estadísticos

## Descripción
Tests estadísticos para EDA: Breusch-Pagan, VIF, normalidad, corrección de heteroscedasticidad.

## Ubicación
`src/skills/statistical_tests.py`

## Funciones
- `run_breusch_pagan(df, target, features)` → dict
- `run_vif(df, features)` → dict
- `run_normality_tests(series)` → dict
- `correct_heteroscedasticity(df, target, features, method)` → dict
  - method: "WLS" | "GLS" | "HC3" | "boxcox"

## Umbrales
- VIF > `config.vif_threshold` (default 10)
- p-value BP < `config.bp_pvalue` (default 0.05)
