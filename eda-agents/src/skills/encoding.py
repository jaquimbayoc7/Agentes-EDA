"""Skill reutilizable — Encoding categórico.

Funciones puras que transforman columnas según flag semántico y model_family.
Usadas por Agent 03 (Data Engineer), re_encoder (graph.py) y cualquier
componente que necesite encodificar columnas.

Reglas de encoding:
    BINARIA       → LabelEncoder (siempre)
    NOMINAL (≤ N) → OneHotEncoder (drop_first)
    NOMINAL (> N) → LabelEncoder (tree) / FrequencyEncoder (linear)
    ALTA_CARD     → FrequencyEncoder (siempre)
    ORDINAL       → OrdinalEncoder con mapeo explícito

N = ohe_max_categories (default 3, configurado en pipeline.yaml).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def encode_column(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    col: str,
    flag: str,
    model_family: str = "tree",
    ohe_max_categories: int = 3,
    ordinal_order: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Codifica *una* columna en train y test según flag y model_family.

    Parameters
    ----------
    df_train, df_test : DataFrames con la columna ``col``.
    col : nombre de la columna a encodificar.
    flag : tipo semántico (BINARIA, NOMINAL, ALTA_CARD, ORDINAL).
    model_family : "tree" o "linear" — afecta estrategia para NOMINAL alta-card.
    ohe_max_categories : umbral para decidir OHE vs Label/Freq.
    ordinal_order : orden explícito para ORDINAL (e.g. ["bajo","medio","alto"]).

    Returns
    -------
    (df_train, df_test, log_entry) donde log_entry describe el encoding aplicado.
    """
    if flag == "BINARIA":
        return _label_encode(df_train, df_test, col, flag)

    if flag == "ORDINAL":
        return _ordinal_encode(df_train, df_test, col, flag, ordinal_order)

    if flag == "NOMINAL":
        n_unique = df_train[col].nunique()
        if n_unique <= ohe_max_categories:
            return _onehot_encode(df_train, df_test, col, flag)
        if model_family == "linear":
            return _frequency_encode(df_train, df_test, col, flag)
        return _label_encode(df_train, df_test, col, flag)

    if flag == "ALTA_CARD":
        return _frequency_encode(df_train, df_test, col, flag)

    # Flag desconocido → no hacer nada
    return df_train, df_test, {"col": col, "encoding": "none", "flag": flag}


def reencode_column(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    col: str,
    current_encoding: str,
    target_family: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Re-codifica una columna ya encodificada para un nuevo model_family.

    Solo actúa si: current_encoding == "label" y target_family == "linear"
    → convierte a FrequencyEncoder.

    Returns
    -------
    (df_train, df_test, new_encoding_name)
    """
    if current_encoding == "label" and target_family == "linear":
        freq = df_train[col].value_counts(normalize=True).to_dict()
        df_train = df_train.copy()
        df_test = df_test.copy()
        df_train[col] = df_train[col].map(freq).fillna(0)
        df_test[col] = df_test[col].map(freq).fillna(0)
        return df_train, df_test, "frequency"
    return df_train, df_test, current_encoding


def encode_all(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    encoding_flags: dict[str, str],
    target: str | None,
    model_family: str = "tree",
    ohe_max_categories: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Encodifica todas las columnas según flags, de una sola vez.

    Ignora flags TARGET, NUMERICA, FECHA, y vacíos.
    """
    skip = {"TARGET", "NUMERICA", "FECHA", ""}
    encoding_log: dict[str, Any] = {}

    for col in list(df_train.columns):
        flag = encoding_flags.get(col, "")
        if flag in skip:
            continue
        df_train, df_test, entry = encode_column(
            df_train, df_test, col, flag,
            model_family=model_family,
            ohe_max_categories=ohe_max_categories,
        )
        encoding_log[col] = entry

    return df_train, df_test, encoding_log


# ---------------------------------------------------------------------------
# Implementaciones internas
# ---------------------------------------------------------------------------


def _label_encode(
    df_train: pd.DataFrame, df_test: pd.DataFrame, col: str, flag: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """LabelEncoder: mapea categorías a enteros según orden de aparición en train."""
    mapping = {v: i for i, v in enumerate(df_train[col].dropna().unique())}
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train[col] = df_train[col].map(mapping)
    df_test[col] = df_test[col].map(mapping)
    return df_train, df_test, {
        "col": col, "encoding": "label", "flag": flag, "moment": 1,
        "n_classes": len(mapping),
    }


def _frequency_encode(
    df_train: pd.DataFrame, df_test: pd.DataFrame, col: str, flag: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """FrequencyEncoder: reemplaza categorías con su frecuencia relativa en train."""
    freq = df_train[col].value_counts(normalize=True).to_dict()
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train[col] = df_train[col].map(freq).fillna(0)
    df_test[col] = df_test[col].map(freq).fillna(0)
    return df_train, df_test, {
        "col": col, "encoding": "frequency", "flag": flag, "moment": 1,
        "n_classes": len(freq),
    }


def _onehot_encode(
    df_train: pd.DataFrame, df_test: pd.DataFrame, col: str, flag: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """OneHotEncoder con drop_first, alineando columnas test ← train."""
    df_train = df_train.copy()
    df_test = df_test.copy()

    train_dummies = pd.get_dummies(df_train[[col]], prefix=col, drop_first=True)
    test_dummies = pd.get_dummies(df_test[[col]], prefix=col, drop_first=True)

    # Alinear test con columnas de train
    for c in train_dummies.columns:
        if c not in test_dummies.columns:
            test_dummies[c] = 0
    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)

    df_train = pd.concat([df_train.drop(columns=[col]), train_dummies], axis=1)
    df_test = pd.concat([df_test.drop(columns=[col]), test_dummies], axis=1)

    return df_train, df_test, {
        "col": col, "encoding": "onehot", "flag": flag, "moment": 1,
        "new_cols": list(train_dummies.columns),
    }


def _ordinal_encode(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    col: str,
    flag: str,
    order: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """OrdinalEncoder con mapeo explícito. Si no hay orden, usa sorted unique."""
    if order is None:
        order = sorted(df_train[col].dropna().unique())
    mapping = {v: i for i, v in enumerate(order)}
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train[col] = df_train[col].map(mapping)
    df_test[col] = df_test[col].map(mapping)
    return df_train, df_test, {
        "col": col, "encoding": "ordinal", "flag": flag, "moment": 1,
        "order": order,
    }
