import pandas as pd
import numpy as np
from typing import Dict, Any


def _find_col(df, name):
    for c in df.columns:
        if c.lower().strip().replace(" ", "_") == name.lower().strip().replace(" ", "_"):
            return c
    return None


def clean_for_ml(df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
    """
    Advanced data cleaning / preprocessing pipeline for ML.
    Returns cleaned df + stats about what was done.
    """
    result = {"samples_before": len(df), "outliers_removed": 0, "imputed_values": 0}
    d = df.copy()

    # 1. Drop fully empty columns
    d = d.dropna(axis=1, how="all")

    # 2. Drop rows where target is null
    if target_col and target_col in d.columns:
        before = len(d)
        d = d.dropna(subset=[target_col])
        result["target_nulls_dropped"] = before - len(d)

    # 3. Separate numeric and categorical
    num_cols = d.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = d.select_dtypes(include=["object", "category"]).columns.tolist()

    # 4. Impute numeric: median (robust to outliers)
    for c in num_cols:
        if c == target_col:
            continue
        null_count = d[c].isna().sum()
        if null_count > 0:
            d[c] = d[c].fillna(d[c].median())
            result["imputed_values"] += null_count

    # 5. Impute categorical: mode
    for c in cat_cols:
        null_count = d[c].isna().sum()
        if null_count > 0 and not d[c].mode().empty:
            d[c] = d[c].fillna(d[c].mode()[0])
            result["imputed_values"] += null_count

    # 6. Remove outliers using IQR (only for numeric features, not target)
    for c in num_cols:
        if c == target_col:
            continue
        q1 = d[c].quantile(0.01)
        q3 = d[c].quantile(0.99)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (d[c] >= lower) & (d[c] <= upper)
        removed = (~mask).sum()
        if removed > 0:
            d = d[mask]
            result["outliers_removed"] += removed

    # 7. Clip extreme values (99th percentile cap as fallback)
    for c in num_cols:
        if c == target_col:
            continue
        upper = d[c].quantile(0.99)
        lower = d[c].quantile(0.01)
        d[c] = d[c].clip(lower, upper)

    # 8. Standardize numeric features (z-score)
    for c in num_cols:
        if c == target_col:
            continue
        mean = d[c].mean()
        std = d[c].std()
        if std > 0:
            d[c] = (d[c] - mean) / std

    result["samples_after"] = len(d)
    return {"df": d, "stats": result}
