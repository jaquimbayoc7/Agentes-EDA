# Estilo de código

## Imports
- Orden: stdlib → third-party → local, con línea en blanco entre grupos
- Usar `from __future__ import annotations` en todos los módulos

## Type hints
- Todas las funciones deben tener type hints incluyendo retornos
- Usar `Optional[X]` para valores que pueden ser None

## Docstrings
- Formato Google-style
- Obligatorio en funciones de agente con Rol y Responsabilidad

## Logging
- Usar `structlog` para todos los logs — nunca `print()` en producción
- Bind `agent` y `run_id` en cada agente

## Nombres
- Figuras: nombres semánticos (`dist_{col}.png`, `corr_matrix.png`)
- Variables: snake_case
- Clases: PascalCase
- Constantes: UPPER_SNAKE_CASE
