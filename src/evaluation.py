from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)


def predict_scores(estimator, X_test, y_test) -> Dict[str, float]:
    y_pred = estimator.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    y_score = None
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X_test)[:, 1]
    elif hasattr(estimator, "decision_function"):
        y_score = estimator.decision_function(X_test)

    auc = roc_auc_score(y_test, y_score) if y_score is not None else float("nan")
    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "auc": float(auc)}


def get_confusion(estimator, X_test, y_test) -> np.ndarray:
    y_pred = estimator.predict(X_test)
    return confusion_matrix(y_test, y_pred)


def get_roc(estimator, X_test, y_test) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    y_score = None
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X_test)[:, 1]
    elif hasattr(estimator, "decision_function"):
        y_score = estimator.decision_function(X_test)
    if y_score is None:
        return None
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    return fpr, tpr, float(auc)