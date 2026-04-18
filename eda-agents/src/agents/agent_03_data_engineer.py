"""Agente 3 — Data Engineer.

Rol: Ingeniero de datos
Responsabilidad: Encoding, imputación, balanceo, feature engineering,
scaling. Opera EXCLUSIVAMENTE sobre train_path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from src.state import EDAState
from src.utils.config import PipelineConfig
from src.utils.llm import call_claude_json
from src.utils.state_validator import validate_ag3_output
from src.skills.encoding import encode_all

logger = structlog.get_logger()


def data_engineer(state: EDAState) -> dict[str, Any]:
    """Agente 3 — Data Engineer.

    Rol: Ingeniero de datos del equipo EDA.
    Responsabilidad:
        - Clasificación semántica de columnas con Claude API
        - Encoding provisional (Momento 1 — para EDA) con model_family="tree"
        - Imputación según reglas de porcentaje
        - Resample solo sobre train según umbrales de desbalanceo
        - Feature engineering con Claude API
        - Scaling: StandardScaler fit sobre train, transform ambos
    """
    run_id = state["run_id"]
    log = logger.bind(agent="ag3", run_id=run_id)
    config = PipelineConfig.from_state(state)

    try:
        log.info("starting")
        train_path = state["train_path"]
        test_path = state["test_path"]
        target = state.get("target")
        encoding_flags = state.get("encoding_flags", {})

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        log.info("data_loaded", train_rows=len(df_train), test_rows=len(df_test))

        encoding_log: dict[str, Any] = {}
        features_nuevas: list[str] = []
        balanceo_log: dict[str, Any] = {"applied": False}

        # --- Imputación ---
        df_train, df_test, impute_log = _impute(df_train, df_test, target)
        log.info("imputation_done", log=impute_log)

        # --- Encoding provisional (model_family="tree" por defecto en Momento 1) ---
        df_train, df_test, encoding_log = encode_all(
            df_train, df_test, encoding_flags, target,
            model_family="tree",
            ohe_max_categories=config.encoding.ohe_max_categories,
        )
        log.info("encoding_done", n_encoded=len(encoding_log))

        # --- Resample (solo train, clasificación: ejecutar las 3 variantes) ---
        desbalance_ratio = state.get("desbalance_ratio")
        tarea = state.get("tarea_sugerida") or state.get("task_override") or "classification"
        sampling_variants: dict[str, Any] = {}

        if desbalance_ratio is not None and target and target in df_train.columns and tarea == "classification":
            df_train, balanceo_log, sampling_variants = _resample_all_variants(
                df_train, target, desbalance_ratio, config, run_id
            )
            log.info("resampling_done", log=balanceo_log)
        elif desbalance_ratio is not None and target and target in df_train.columns:
            df_train, balanceo_log = _resample(
                df_train, target, desbalance_ratio, config
            )
            log.info("resampling_done", log=balanceo_log)

        # --- Feature engineering ---
        df_train, df_test, features_nuevas = _feature_engineering(
            df_train, df_test, state, config
        )
        log.info("feature_engineering_done", n_new=len(features_nuevas))

        # --- Scaling ---
        df_train, df_test = _scale(df_train, df_test, target, config)

        # --- Guardar datasets provisionales ---
        output_dir = Path("outputs") / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        train_prov_path = str(output_dir / "dataset_train_provisional.csv")
        test_proc_path = str(output_dir / "dataset_test_procesado.csv")
        df_train.to_csv(train_prov_path, index=False)
        df_test.to_csv(test_proc_path, index=False)

        # --- Verificación final ---
        _verify_no_objects_or_nans(df_train, df_test, log)

        output: dict[str, Any] = {
            "encoding_log": encoding_log,
            "features_nuevas": features_nuevas,
            "balanceo_log": balanceo_log,
            "sampling_variants": sampling_variants,
            "dataset_train_provisional": train_prov_path,
            "dataset_test_procesado": test_proc_path,
            "agent_status": {**state.get("agent_status", {}), "ag3": "ok"},
        }

        validate_ag3_output(output)
        log.info("completed")
        return output

    except Exception as e:
        log.error("failed", error=str(e))
        return {
            "encoding_log": {},
            "features_nuevas": [],
            "balanceo_log": {},
            "sampling_variants": {},
            "dataset_train_provisional": "",
            "dataset_test_procesado": "",
            "agent_status": {**state.get("agent_status", {}), "ag3": "error"},
            "error_log": [{"agent": "ag3", "error": str(e), "run_id": run_id}],
        }


def _impute(
    df_train: pd.DataFrame, df_test: pd.DataFrame, target: str | None
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Imputa valores nulos según reglas de porcentaje.

    - Numéricas < 5% nulos → mediana
    - Numéricas 5-20% → mediana (KNN en PASO 5)
    - Categóricas → moda
    - Columnas > 40% nulos → eliminar
    """
    impute_log: dict[str, Any] = {"dropped": [], "imputed": {}}

    for col in df_train.columns:
        if col == target:
            continue
        null_pct = df_train[col].isna().mean() * 100

        if null_pct == 0:
            continue
        elif null_pct > 40:
            impute_log["dropped"].append(col)
            df_train = df_train.drop(columns=[col])
            df_test = df_test.drop(columns=[col], errors="ignore")
        elif pd.api.types.is_numeric_dtype(df_train[col]):
            median_val = df_train[col].median()
            df_train[col] = df_train[col].fillna(median_val)
            df_test[col] = df_test[col].fillna(median_val)
            impute_log["imputed"][col] = {"method": "median", "value": float(median_val)}
        else:
            mode_val = df_train[col].mode().iloc[0] if not df_train[col].mode().empty else "unknown"
            df_train[col] = df_train[col].fillna(mode_val)
            df_test[col] = df_test[col].fillna(mode_val)
            impute_log["imputed"][col] = {"method": "mode", "value": str(mode_val)}

    return df_train, df_test, impute_log


def _resample(
    df_train: pd.DataFrame,
    target: str,
    ratio: float,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resample según umbrales de desbalanceo. SOLO sobre train.

    Multiclass-aware: opera sobre TODAS las clases.
    """
    thresholds = config.imbalance_thresholds
    balanceo_log: dict[str, Any] = {"ratio_before": ratio}
    seed = config.random_seed

    counts = df_train[target].value_counts()
    n_minority = int(counts.min())
    n_majority = int(counts.max())
    median_count = int(np.median(counts.values))

    if ratio < thresholds.oversample:
        balanceo_log["method"] = "none"
        balanceo_log["applied"] = False
    elif ratio < thresholds.hybrid:
        # Oversample: todas las clases suben al nivel de la mayoritaria
        parts = []
        for cls, cnt in counts.items():
            cls_df = df_train[df_train[target] == cls]
            if cnt < n_majority:
                extra = cls_df.sample(
                    n=n_majority - cnt, replace=True, random_state=seed,
                )
                parts.append(pd.concat([cls_df, extra], ignore_index=True))
            else:
                parts.append(cls_df)
        df_train = pd.concat(parts, ignore_index=True)
        balanceo_log["method"] = "oversample"
        balanceo_log["applied"] = True
    elif ratio < thresholds.undersample:
        # Hybrid: target = mediana; sube inferiores, baja superiores
        target_n = max(median_count, n_minority + 1)
        parts = []
        for cls, cnt in counts.items():
            cls_df = df_train[df_train[target] == cls]
            if cnt < target_n:
                extra = cls_df.sample(
                    n=target_n - cnt, replace=True, random_state=seed,
                )
                parts.append(pd.concat([cls_df, extra], ignore_index=True))
            elif cnt > target_n:
                parts.append(cls_df.sample(
                    n=target_n, replace=False, random_state=seed,
                ))
            else:
                parts.append(cls_df)
        df_train = pd.concat(parts, ignore_index=True)
        balanceo_log["method"] = "hybrid"
        balanceo_log["applied"] = True
    else:
        # Undersample: todas las clases bajan al nivel de la minoritaria
        parts = []
        for cls, cnt in counts.items():
            cls_df = df_train[df_train[target] == cls]
            if cnt > n_minority:
                parts.append(cls_df.sample(
                    n=n_minority, replace=False, random_state=seed,
                ))
            else:
                parts.append(cls_df)
        df_train = pd.concat(parts, ignore_index=True)
        balanceo_log["method"] = "undersample"
        balanceo_log["applied"] = True

    balanceo_log["ratio_after"] = float(
        df_train[target].value_counts().max() / df_train[target].value_counts().min()
    )
    return df_train, balanceo_log


def _resample_all_variants(
    df_train: pd.DataFrame,
    target: str,
    ratio: float,
    config: PipelineConfig,
    run_id: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Genera las 3 variantes de muestreo (oversample, undersample, hybrid),
    las guarda como CSVs y selecciona la mejor con justificación.

    Multiclass-aware: opera sobre TODAS las clases, no solo la
    mayoritaria/minoritaria.

    - Oversample: sube todas las clases al nivel de la más grande.
    - Undersample: baja todas las clases al nivel de la más pequeña.
    - Hybrid: lleva todas las clases al conteo mediano (sube las
      que están por debajo, baja las que están por encima).

    Solo se aplica en clasificación. Retorna el df_train con la variante
    elegida, el balanceo_log y el dict de sampling_variants.
    """
    seed = config.random_seed
    counts = df_train[target].value_counts()
    minority_class = counts.idxmin()
    majority_class = counts.idxmax()
    n_minority = int(counts.min())
    n_majority = int(counts.max())
    median_count = int(np.median(counts.values))

    output_dir = Path("outputs") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    variants: dict[str, Any] = {}

    # ---- 1. OVERSAMPLE — todas las clases suben al nivel de la mayoritaria ----
    try:
        parts = []
        for cls, cnt in counts.items():
            cls_df = df_train[df_train[target] == cls]
            if cnt < n_majority:
                extra = cls_df.sample(
                    n=n_majority - cnt, replace=True, random_state=seed,
                )
                parts.append(pd.concat([cls_df, extra], ignore_index=True))
            else:
                parts.append(cls_df)
        df_over = pd.concat(parts, ignore_index=True)
        over_path = str(output_dir / "train_oversample.csv")
        df_over.to_csv(over_path, index=False)
        over_counts = df_over[target].value_counts()
        variants["oversample"] = {
            "path": over_path,
            "n_rows": len(df_over),
            "ratio_after": float(over_counts.max() / over_counts.min()),
            "class_distribution": {str(k): int(v) for k, v in over_counts.items()},
            "description": (
                f"Sobremuestreo de TODAS las clases al nivel de la mayoritaria "
                f"({majority_class}: {n_majority})"
            ),
        }
    except Exception:
        variants["oversample"] = {"path": "", "error": "failed"}

    # ---- 2. UNDERSAMPLE — todas las clases bajan al nivel de la minoritaria ----
    try:
        parts = []
        for cls, cnt in counts.items():
            cls_df = df_train[df_train[target] == cls]
            if cnt > n_minority:
                parts.append(cls_df.sample(
                    n=n_minority, replace=False, random_state=seed,
                ))
            else:
                parts.append(cls_df)
        df_under = pd.concat(parts, ignore_index=True)
        under_path = str(output_dir / "train_undersample.csv")
        df_under.to_csv(under_path, index=False)
        under_counts = df_under[target].value_counts()
        variants["undersample"] = {
            "path": under_path,
            "n_rows": len(df_under),
            "ratio_after": float(under_counts.max() / under_counts.min()),
            "class_distribution": {str(k): int(v) for k, v in under_counts.items()},
            "description": (
                f"Submuestreo de TODAS las clases al nivel de la minoritaria "
                f"({minority_class}: {n_minority})"
            ),
        }
    except Exception:
        variants["undersample"] = {"path": "", "error": "failed"}

    # ---- 3. HYBRID — target = mediana; sube las inferiores, baja las superiores ----
    try:
        target_n = max(median_count, n_minority + 1)
        parts = []
        for cls, cnt in counts.items():
            cls_df = df_train[df_train[target] == cls]
            if cnt < target_n:
                extra = cls_df.sample(
                    n=target_n - cnt, replace=True, random_state=seed,
                )
                parts.append(pd.concat([cls_df, extra], ignore_index=True))
            elif cnt > target_n:
                parts.append(cls_df.sample(
                    n=target_n, replace=False, random_state=seed,
                ))
            else:
                parts.append(cls_df)
        df_hybrid = pd.concat(parts, ignore_index=True)
        hybrid_path = str(output_dir / "train_hybrid.csv")
        df_hybrid.to_csv(hybrid_path, index=False)
        hybrid_counts = df_hybrid[target].value_counts()
        variants["hybrid"] = {
            "path": hybrid_path,
            "n_rows": len(df_hybrid),
            "ratio_after": float(hybrid_counts.max() / hybrid_counts.min()),
            "class_distribution": {str(k): int(v) for k, v in hybrid_counts.items()},
            "description": (
                f"Híbrido: todas las clases al conteo mediano ({target_n}). "
                f"Submuestreo de clases grandes + sobremuestreo de clases pequeñas"
            ),
        }
    except Exception:
        variants["hybrid"] = {"path": "", "error": "failed"}

    # ---- Selección de la mejor variante ----
    best_method = "oversample"
    best_score = float("inf")

    for method, info in variants.items():
        if info.get("error"):
            continue
        r = info["ratio_after"]
        n = info["n_rows"]
        data_loss_penalty = max(0, (len(df_train) - n) / len(df_train)) * 0.5
        score = abs(r - 1.0) + data_loss_penalty
        if score < best_score:
            best_score = score
            best_method = method

    # Justificación de la selección
    def _fmt_dist(m: str) -> str:
        d = variants.get(m, {}).get("class_distribution", {})
        return ", ".join(f"{k}: {v}" for k, v in sorted(d.items()))

    reasons = {
        "oversample": (
            f"Sobremuestreo seleccionado: todas las clases elevadas a {n_majority} filas. "
            f"Ratio original {ratio:.2f} → {variants.get('oversample', {}).get('ratio_after', ratio):.2f}. "
            f"Distribución resultante: {_fmt_dist('oversample')}. "
            f"No pierde datos originales; ideal cuando el dataset es pequeño."
        ),
        "undersample": (
            f"Submuestreo seleccionado: todas las clases reducidas a {n_minority} filas. "
            f"Ratio original {ratio:.2f} → {variants.get('undersample', {}).get('ratio_after', ratio):.2f}. "
            f"Distribución resultante: {_fmt_dist('undersample')}. "
            f"Balance perfecto a costa de perder datos de clases grandes."
        ),
        "hybrid": (
            f"Método híbrido seleccionado: todas las clases llevadas al conteo mediano ({median_count}). "
            f"Ratio original {ratio:.2f} → {variants.get('hybrid', {}).get('ratio_after', ratio):.2f}. "
            f"Distribución resultante: {_fmt_dist('hybrid')}. "
            f"Balancea pérdida de información con representación equitativa."
        ),
    }

    # Marcar la variante elegida
    for method in variants:
        if not variants[method].get("error"):
            variants[method]["selected"] = (method == best_method)

    # Aplicar la variante elegida al df_train
    chosen_info = variants.get(best_method, {})
    chosen_path = chosen_info.get("path", "")
    if chosen_path and Path(chosen_path).exists():
        df_train = pd.read_csv(chosen_path)

    balanceo_log: dict[str, Any] = {
        "ratio_before": ratio,
        "method": best_method,
        "applied": True,
        "reason": reasons.get(best_method, ""),
        "ratio_after": chosen_info.get("ratio_after", ratio),
        "all_variants_computed": True,
        "variants_summary": {
            m: {
                "ratio_after": v.get("ratio_after", "N/A"),
                "n_rows": v.get("n_rows", "N/A"),
                "class_distribution": v.get("class_distribution", {}),
            }
            for m, v in variants.items()
            if not v.get("error")
        },
    }

    return df_train, balanceo_log, variants


def _feature_engineering(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    state: EDAState,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Feature engineering guiado por contexto.

    Usa Claude API para proponer features relevantes basadas en las
    referencias encontradas y el contexto del dominio.
    """
    features_nuevas: list[str] = []

    if not config.anthropic_api_key:
        return df_train, df_test, features_nuevas

    try:
        cols = df_train.columns.tolist()
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        refs = state.get("refs", [])
        refs_ctx = "\n".join(f"- {r.get('title', '')}" for r in refs[:5]) or "None"
        question = state.get("research_question", "")

        result = call_claude_json(
            prompt=(
                f"Research question: {question}\n"
                f"Available columns: {cols}\n"
                f"Numeric columns: {numeric_cols}\n"
                f"References:\n{refs_ctx}\n\n"
                "Propose up to 5 feature engineering operations using only basic "
                "arithmetic on existing numeric columns (ratios, products, logs). "
                "Each must be justifiable from domain knowledge.\n"
                "Return JSON: {\"features\": [{\"name\": \"new_col\", "
                "\"expr\": \"col_a / col_b\", \"reason\": \"...\"}]}"
            ),
            system="You are a feature engineering expert. Return only valid JSON.",
            model=config.model,
            max_tokens=1024,
            api_key=config.anthropic_api_key,
        )

        for feat in result.get("features", []):
            name = feat.get("name", "")
            expr = feat.get("expr", "")
            if not name or not expr:
                continue

            try:
                # Evaluate safely against dataframe columns only
                train_val = df_train.eval(expr)
                test_val = df_test.eval(expr)
                if train_val.isna().all() or not np.isfinite(train_val.dropna()).all():
                    continue
                df_train[name] = train_val
                df_test[name] = test_val
                features_nuevas.append(name)
            except Exception:
                continue

    except Exception:
        pass

    return df_train, df_test, features_nuevas


def _scale(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target: str | None,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """StandardScaler: fit sobre train, transform sobre ambos."""
    from sklearn.preprocessing import StandardScaler

    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    if target and target in numeric_cols:
        numeric_cols.remove(target)

    if not numeric_cols:
        return df_train, df_test

    scaler = StandardScaler()
    df_train[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
    df_test[numeric_cols] = scaler.transform(
        df_test[numeric_cols].reindex(columns=numeric_cols, fill_value=0)
    )
    return df_train, df_test


def _verify_no_objects_or_nans(
    df_train: pd.DataFrame, df_test: pd.DataFrame, log: Any
) -> None:
    """Verifica que no queden columnas object ni NaN."""
    obj_cols_train = df_train.select_dtypes(include=["object"]).columns.tolist()
    obj_cols_test = df_test.select_dtypes(include=["object"]).columns.tolist()

    if obj_cols_train:
        log.warning("object_cols_in_train", cols=obj_cols_train)
    if obj_cols_test:
        log.warning("object_cols_in_test", cols=obj_cols_test)

    nan_train = df_train.isna().sum().sum()
    nan_test = df_test.isna().sum().sum()
    if nan_train > 0:
        log.warning("nans_in_train", count=int(nan_train))
    if nan_test > 0:
        log.warning("nans_in_test", count=int(nan_test))
