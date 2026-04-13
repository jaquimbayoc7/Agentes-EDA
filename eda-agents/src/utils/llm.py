"""Utilidad compartida para llamadas a Claude API.

Provee funciones de alto nivel para:
- Llamadas síncronas a la API de Claude (Anthropic)
- Parsing seguro de respuestas JSON
- Fallback graceful si la API no está disponible
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Claude API
# ---------------------------------------------------------------------------


def call_claude(
    prompt: str,
    *,
    system: str = "",
    model: str = "claude-sonnet-4-5",
    max_tokens: int = 4096,
    api_key: str = "",
    temperature: float = 0.3,
) -> str:
    """Llama a Claude API y retorna el texto de respuesta.

    Si api_key está vacía, lanza ValueError.
    """
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not configured")

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "temperature": temperature,
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    return response.content[0].text


def call_claude_json(
    prompt: str,
    *,
    system: str = "",
    model: str = "claude-sonnet-4-5",
    max_tokens: int = 4096,
    api_key: str = "",
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Llama a Claude API y parsea la respuesta como JSON.

    Extrae JSON de bloques ```json ... ``` o del texto directo.
    Si el parsing falla, retorna {"raw": texto_respuesta}.
    """
    text = call_claude(
        prompt,
        system=system,
        model=model,
        max_tokens=max_tokens,
        api_key=api_key,
        temperature=temperature,
    )
    return parse_json_response(text)


def parse_json_response(text: str) -> dict[str, Any]:
    """Extrae y parsea JSON de una respuesta de LLM.

    Intenta en orden:
    1. Bloque ```json ... ```
    2. Primer { ... } encontrado
    3. Retorna {"raw": texto} si falla
    """
    # Intentar extraer bloque JSON
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Intentar todo el texto como JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Intentar encontrar el primer { ... }
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return {"raw": text}
