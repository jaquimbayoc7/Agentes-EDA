# Reglas de Testing

## Estructura
- Un archivo de test por agente: `tests/agents/test_agent_0X.py`
- Fixtures compartidos en `tests/conftest.py`
- Datos de prueba en `tests/fixtures/`

## Tests obligatorios por agente
1. Test con dataset fixture de 100 filas
2. Test de fallback: simula fallo de API → verifica `agent_status: "error"`
3. Test de validación: verifica que output pasa Pydantic
4. Test de no-leakage (Data Engineer): verifica que fit() no usa test set

## Convenciones
- Usar `tmp_path` de pytest para archivos temporales
- Usar `monkeypatch` para mockear APIs externas
- `random_seed: 42` en todos los tests
- Aserciones descriptivas con mensajes claros

## Ejecución
```bash
pytest tests/ -v --tb=short
pytest tests/agents/test_agent_01.py -v  # un agente
pytest tests/ -k "fallback" -v           # solo fallbacks
```
