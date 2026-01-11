from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional


EvalScheme = Literal["split_80_20", "split_70_30", "cv_10fold"]


@dataclass
class AppConfig:
    # Upload & Konfigurasi
    target_col: str = "exam_score"
    threshold: float = 75.0

    # Evaluasi
    eval_scheme: EvalScheme = "split_80_20"
    random_state: int = 42
    cv_splits: int = 10

    # Model & tuning
    selected_model: Optional[str] = None
    tuning_method: Literal["none", "grid", "random"] = "grid"
    n_iter: int = 40  # untuk randomized
    scoring: str = "roc_auc"  # fokus ke AUC

    # Feature selection (opsional)
    use_feature_selection: bool = False
    top_k_features: int = 20