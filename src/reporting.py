from __future__ import annotations

from typing import Dict, Any, List, Optional
import pandas as pd


def select_best_model(df_metrics: pd.DataFrame, primary: str = "auc") -> str:
    if primary not in df_metrics.columns:
        raise ValueError(f"Kolom '{primary}' tidak ada pada metrics dataframe.")
    return str(df_metrics.sort_values(primary, ascending=False).iloc[0]["model"])


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def merge_baseline_and_tuned(
    baseline: pd.DataFrame,
    tuned: Optional[pd.DataFrame],
) -> pd.DataFrame:
    baseline = baseline.copy()
    baseline["stage"] = "baseline"

    if tuned is None or tuned.empty:
        return baseline.sort_values(["stage", "auc"], ascending=[True, False]).reset_index(drop=True)

    tuned = tuned.copy()
    tuned["stage"] = "tuned"

    out = pd.concat([baseline, tuned], ignore_index=True)
    # urutkan: tuned di atas bila AUC lebih besar, tapi tetap informatif
    out = out.sort_values(["stage", "auc"], ascending=[True, False]).reset_index(drop=True)
    return out