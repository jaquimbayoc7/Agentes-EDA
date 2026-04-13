# EDA Agents

Sistema multi-agente de **Analisis Exploratorio de Datos** (EDA) construido con LangGraph y Claude API. Ejecuta un pipeline de 8 agentes especializados que analiza datasets tabulares o de series de tiempo, y genera un informe completo con hipotesis, estadisticas, visualizaciones y recomendaciones de modelos.

## Arquitectura

```
START
  |
  +--[parallel]--> Research Lead ----+
  |                                  |
  +--[parallel]--> Data Steward -----+
                                     |
                               Data Engineer
                                     |
                    +--[parallel]--> Statistician --------+
                    |                                     |
                    +--[parallel]--> TS Analyst* ---------+
                                     (* solo si timeseries)
                                     |
                               ML Strategist
                                     |
                               Re-Encoder
                                     |
                           Visualization Designer
                                     |
                            Technical Writer
                                     |
                                    END
```

## Los 8 Agentes

| # | Agente | Responsabilidad |
|---|--------|----------------|
| 1 | **Research Lead** | Genera ecuaciones PICO, busca literatura con Claude, formula 3 hipotesis (confirmatoria, exploratoria, alternativa), infiere tipo de tarea |
| 2 | **Data Steward** | Perfila columnas, detecta tipos, calcula nulos/cardinalidad, hace train/test split, detecta desbalance y series temporales |
| 3 | **Data Engineer** | Aplica encoding (OHE/Label/Ordinal), resampling para desbalance, feature engineering, genera datasets provisionales |
| 4 | **Statistician** | EDA tabular: correlaciones, test de Breusch-Pagan, VIF, medianas, distribucion del target |
| 5 | **TS Analyst** | Analisis de series de tiempo: estacionariedad (ADF/KPSS), deteccion de cambios, ARIMA/SARIMA/SARIMAX/VAR (solo si flag_timeseries=True) |
| 6 | **ML Strategist** | Recomienda modelos, hiperparametros, tecnica de busqueda, metrica principal, define model_family (linear/tree) |
| 7 | **Viz Designer** | Genera figuras PNG: matriz de correlaciones, distribuciones, boxplots, pairplot, distribucion del target |
| 8 | **Technical Writer** | Produce el informe final `report.md` con 12 secciones estructuradas |

## Stack Tecnologico

- **Python 3.10+**
- **LangGraph** — Orquestacion del grafo de agentes con paralelismo
- **Claude API** (claude-sonnet-4-5) — LLM para todos los agentes
- **Pandas / NumPy / SciPy / Statsmodels / Scikit-learn** — Procesamiento y estadistica
- **Matplotlib / Seaborn / Plotly** — Visualizaciones
- **Pydantic v2** — Validacion de estado
- **structlog** — Logging estructurado

## Requisitos Previos

- Python 3.10 o superior
- Una API key de [Anthropic](https://console.anthropic.com/)

## Instalacion

```bash
# Clonar el repositorio
git clone https://github.com/jaquimbayoc7/Agentes-EDA.git
cd Agentes-EDA/eda-agents

# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Instalar dependencias
pip install -e ".[dev]"
```

## Configuracion

### 1. API Key

Crea un archivo `.env` en la carpeta `eda-agents/`:

```bash
cp .env.example .env
```

Edita `.env` y agrega tu API key real:

```
ANTHROPIC_API_KEY=sk-ant-api03-TU-CLAVE-AQUI
```

### 2. Parametros del Pipeline (opcional)

El archivo `config/pipeline.yaml` contiene todos los umbrales centralizados:

```yaml
random_seed: 42
model: claude-sonnet-4-5
max_tokens: 4096

imbalance_thresholds:
  oversample: 3
  hybrid: 10
  undersample: 30

vif_threshold: 10
bp_pvalue: 0.05

encoding:
  ohe_max_categories: 3

split:
  test_size: 0.2
  stratify: true
```

## Uso

### Colocar tus datos

Coloca tus archivos CSV en la carpeta `data/`:

```
eda-agents/
  data/
    mi_dataset.csv
    ventas_2024.csv
    ...
```

### Ejecutar el pipeline

```bash
python main.py \
  --question "Tu pregunta de investigacion" \
  --dataset data/mi_dataset.csv \
  --data-type tabular \
  --target nombre_columna_objetivo
```

### Parametros

| Parametro | Requerido | Valores | Descripcion |
|-----------|-----------|---------|-------------|
| `--question` | Si | texto libre | Pregunta de investigacion que guia el analisis |
| `--dataset` | Si | ruta CSV | Ruta al archivo de datos |
| `--data-type` | Si | `tabular` / `timeseries` | Tipo de datos a analizar |
| `--target` | Si | nombre columna | Columna objetivo del analisis |
| `--resume` | No | run_id | ID de una ejecucion previa para reanudar |

### Ejemplos

**Datos tabulares:**
```bash
python main.py \
  --question "Que factores predicen el precio de venta" \
  --dataset data/ventas.csv \
  --data-type tabular \
  --target precio_venta
```

**Series de tiempo:**
```bash
python main.py \
  --question "Cual es la tendencia y estacionalidad de las ventas mensuales" \
  --dataset data/ventas_mensuales.csv \
  --data-type timeseries \
  --target ventas
```

**Reanudar un pipeline interrumpido:**
```bash
python main.py \
  --question "..." \
  --dataset data/mi_dataset.csv \
  --data-type tabular \
  --target target \
  --resume abc12345
```

## Salida

Cada ejecucion genera una carpeta en `outputs/<run_id>/`:

```
outputs/<run_id>/
  report.md                    # Informe completo (secciones 1-12)
  decision.json                # Tarea, modelos, metrica, model_family
  state_final.json             # Estado completo del grafo
  run.log.jsonl                # Log estructurado de la ejecucion
  train.csv                    # Split de entrenamiento original
  test.csv                     # Split de prueba original
  dataset_train_provisional.csv   # Train con encoding provisional
  dataset_train_final.csv      # Train con encoding final
  dataset_test_procesado.csv   # Test procesado
  dataset_test_final.csv       # Test con encoding final
  figures/
    corr_matrix.png            # Matriz de correlaciones
    dist_*.png                 # Distribuciones de variables
    box_*.png                  # Boxplots
    pairplot.png               # Pairplot top-6 numericas
    target_dist.png            # Distribucion del target
  reportesFinales/
    reporte_eda.html           # Pagina HTML dinamica con todo el EDA
  notebooksFinales/
    eda_reproducible_<run_id>.ipynb  # Notebook Jupyter reproducible
```

### Reporte HTML Dinamico

Se genera automaticamente una pagina HTML auto-contenida en `reportesFinales/` con:
- Navegacion lateral por secciones
- Figuras embebidas (base64, no requiere archivos externos)
- Tablas ordenables
- Tema claro/oscuro
- KPIs visuales
- Modal para zoom de imagenes

### Notebook Reproducible

Se genera automaticamente un notebook Jupyter en `notebooksFinales/` con:
- Carga de datos y configuracion
- Perfil de datos (nulos, cardinalidad, tipos)
- Preprocesamiento y encoding aplicado
- Visualizaciones (correlaciones, distribuciones, boxplots, pairplot)
- Hallazgos estadisticos del pipeline
- Decision de modelos y ejemplo de entrenamiento

### Ejemplo de `decision.json`

```json
{
  "tarea": "regression",
  "modelos_recomendados": [
    {"name": "Ridge", "reason": "N bajo -> regularizacion"}
  ],
  "hyperparams_technique": "GridSearchCV",
  "model_family": "linear",
  "metrica_principal": "RMSE"
}
```

## Estructura del Proyecto

```
eda-agents/
  main.py                    # CLI - punto de entrada
  pyproject.toml             # Dependencias y metadata
  config/
    pipeline.yaml            # Umbrales centralizados
  data/                      # Coloca tus CSVs aqui
  src/
    state.py                 # EDAState TypedDict (estado compartido)
    graph.py                 # StateGraph con paralelismo LangGraph
    agents/
      agent_01_research_lead.py
      agent_02_data_steward.py
      agent_03_data_engineer.py
      agent_04_statistician.py
      agent_05_ts_analyst.py
      agent_06_ml_strategist.py
      agent_07_viz_designer.py
      agent_08_technical_writer.py
    skills/
      encoding.py            # OHE, Label, Ordinal, Frequency encoding
      report_builder.py      # Generacion del informe markdown
      html_report.py         # Generacion de pagina HTML dinamica
      notebook_builder.py    # Generacion de notebook Jupyter
      statistical_tests.py   # Breusch-Pagan, VIF, correlaciones
      timeseries.py          # ADF, KPSS, deteccion de cambios
    utils/
      config.py              # PipelineConfig (carga YAML + .env)
      llm.py                 # Cliente Claude (call_claude, call_claude_json)
      logger.py              # structlog configurado
      state_validator.py     # Validacion Pydantic del estado
  tests/
    test_agents.py           # Tests de los 8 agentes
    test_config.py           # Tests de configuracion
    test_graph.py            # Tests del grafo completo
    test_llm.py              # Tests del cliente Claude
    test_main.py             # Tests end-to-end del pipeline
    test_skills.py           # Tests de encoding, stats, timeseries
    test_state.py            # Tests del estado
    test_validator.py        # Tests de validacion
    fixtures/
      sample_100.csv         # Dataset de prueba (100 filas)
  outputs/                   # Generado automaticamente por cada run
```

## Tests

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Con cobertura
pytest tests/ --cov=src --cov-report=term-missing

# Solo un modulo
pytest tests/test_agents.py -v
```

**141 tests** cubren agentes, skills, grafo, estado, validacion, HTML report, notebook builder y pipeline end-to-end.

## Convenciones del Proyecto

- `fit()` solo sobre `train_path`, nunca sobre el dataset completo
- Cada agente retorna un dict parcial, nunca muta el estado directamente
- `agent_status` siempre se escribe: `"ok"` / `"fallback"` / `"error"`
- `RANDOM_SEED` se lee de config, nunca se hardcodea
- Outputs en `outputs/{run_id}/` — nunca se sobreescriben runs anteriores
- Toda la comunicacion con LLM pasa por `src/utils/llm.py`

## Licencia

MIT
