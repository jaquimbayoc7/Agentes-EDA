# Convenciones de Estado

## EDAState
- Es un TypedDict, NO una clase mutable
- Cada agente retorna un dict parcial — NUNCA muta el estado directamente
- LangGraph fusiona automáticamente los dicts parciales

## agent_status
- SIEMPRE se escribe en cada retorno de agente
- Valores permitidos: "ok" | "fallback" | "error"
- Acumulativo: `{**state.get("agent_status", {}), "agX": "ok"}`

## Campos acumulativos
- `refs`, `figures`, `error_log` usan `Annotated[list, operator.add]`
- Se extienden automáticamente al retornarlos como lista

## Validación
- Todo output de agente pasa por validador Pydantic antes de retornar
- Usar `validate_agX_output()` de `src/utils/state_validator.py`

## fit() / transform()
- fit() SOLO sobre train_path — esta regla NO tiene excepción
- transform() sobre train Y test
- NUNCA usar dataset completo para fit

## random_state
- Siempre desde `state["random_seed"]` o `config.random_seed`
- NUNCA hardcodear seeds
