# Skill: Encoding categórico

## Descripción
Lógica de encoding reutilizable para variables categóricas según tipo semántico
y familia de modelo destino.

## Ubicación
`src/skills/encoding.py`

## Función principal
`encode_column(df_train, df_test, col, semantic_type, n_unique, model_family, target, ordinal_order, random_seed)`

## Reglas
- BINARIA: LabelEncoder con mapping explícito
- NOMINAL, n_unique <= 3: One-Hot (drop_first=True)
- NOMINAL, n_unique >= 4, tree: LabelEncoder
- NOMINAL, n_unique >= 4, linear: FrequencyEncoder
- ORDINAL: OrdinalEncoder con orden definido por LLM
- FECHA: extracción year/month/dow/quarter + sin/cos cíclico
- ALTA_CARD (>15): FrequencyEncoder o eliminación

## Regla crítica
fit() SOLO sobre df_train. transform sobre ambos.
