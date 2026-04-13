"""Utilidad — Sanitización recursiva de tipos numpy para serialización.

LangGraph usa msgpack para serializar el state, y msgpack no soporta
numpy.int64, numpy.float64, numpy.bool_, etc.  Este módulo convierte
recursivamente cualquier tipo numpy a su equivalente nativo de Python.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def sanitize_state(obj: Any) -> Any:
    """Convierte recursivamente tipos numpy a nativos de Python.

    Maneja: dict, list, tuple, numpy scalars, numpy arrays, numpy bool.
    """
    if isinstance(obj, dict):
        return {sanitize_state(k): sanitize_state(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_state(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_state(v) for v in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
