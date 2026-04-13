# EDA Agents — Sistema multi-agente de análisis exploratorio

## Propósito
Pipeline de 8 agentes especializados para EDA eficiente sobre datos tabulares
y series de tiempo. Produce un informe completo con recomendaciones de modelos,
hipótesis de investigación y referencias bibliográficas verificadas.

## Stack
Python 3.10 · LangGraph · Claude API (claude-sonnet-4-5)

## Los 8 agentes
1. Research Lead — ecuaciones PICO + 3 hipótesis (Claude literature search)
2. Data Steward — perfilado + train/test split (ydata-profiling)
3. Data Engineer — encoding + resample + feature engineering
4. Statistician — EDA tabular + Breusch-Pagan
5. TS Analyst — ARIMA/SARIMA/SARIMAX/VAR (condicional)
6. ML Strategist — modelos + hiperparáms + model_family
7. Visualization Designer — figuras PNG + Plotly
8. Technical Writer — informe 12 secciones + PDF

## Ejecución
```bash
python main.py --question "..." --dataset data/archivo.csv --data-type tabular
```

## Convenciones críticas
- fit() SOLO sobre train_path, NUNCA sobre dataset completo
- Todo agente retorna dict parcial, NUNCA muta el estado
- agent_status siempre se escribe: "ok" | "fallback" | "error"
- RANDOM_SEED desde config, no hardcodeado
- Outputs en outputs/{run_id}/ — nunca sobreescribir runs anteriores

## Estructura del proyecto
```
src/state.py          — EDAState TypedDict (corazón del sistema)
src/graph.py          — StateGraph con paralelismo LangGraph
src/agents/           — 8 agentes especializados
src/skills/           — Lógica reutilizable (encoding, stats, ts, reports)
src/utils/            — Config, logger, validadores
config/pipeline.yaml  — Umbrales centralizados
tests/                — Tests con pytest
outputs/{run_id}/     — Artefactos de cada ejecución
```

## Tests
```bash
pytest tests/ -v
```
