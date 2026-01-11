from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_validate

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# XGBoost optional
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def build_models(random_state: int = 42) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "RandomForest": RandomForestClassifier(n_estimators=500, random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        "AdaBoost": AdaBoostClassifier(random_state=random_state),
        "SVM_Linear": SVC(kernel="linear", probability=True, random_state=random_state),
        "SVM_RBF": SVC(kernel="rbf", probability=True, random_state=random_state),
        "SVM_Poly": SVC(kernel="poly", probability=True, random_state=random_state),
        "SVM_Sigmoid": SVC(kernel="sigmoid", probability=True, random_state=random_state),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            eval_metric="logloss",
        )
    return models


def build_pipeline(preprocess, model) -> Pipeline:
    return Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model),
    ])


def get_param_space(model_name: str) -> Dict[str, Any]:
    if model_name == "LogisticRegression":
        return {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs"],
        }

    if model_name.startswith("SVM_"):
        if model_name == "SVM_Linear":
            return {
                "model__C": np.logspace(-2, 2, 10),
                "model__class_weight": [None, "balanced"],
            }
        if model_name == "SVM_Poly":
            return {
                "model__C": np.logspace(-2, 2, 10),
                "model__gamma": np.logspace(-5, -1, 10),
                "model__degree": [2, 3, 4],
                "model__coef0": [0.0, 0.5, 1.0],
                "model__class_weight": [None, "balanced"],
            }
        if model_name == "SVM_Sigmoid":
            return {
                "model__C": np.logspace(-2, 2, 10),
                "model__gamma": np.logspace(-5, -1, 10),
                "model__coef0": [0.0, 0.5, 1.0],
                "model__class_weight": [None, "balanced"],
            }
        return {
            "model__C": np.logspace(-2, 2, 12),
            "model__gamma": np.logspace(-5, -1, 12),
            "model__class_weight": [None, "balanced"],
        }

    if model_name == "KNN":
        return {
            "model__n_neighbors": list(range(3, 32, 2)),
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        }

    if model_name == "RandomForest":
        return {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        }

    if model_name == "GradientBoosting":
        return {
            "model__n_estimators": [100, 200, 400],
            "model__learning_rate": [0.01, 0.03, 0.1, 0.2],
            "model__max_depth": [2, 3, 4],
        }

    if model_name == "AdaBoost":
        return {
            "model__n_estimators": [50, 100, 200, 400],
            "model__learning_rate": [0.01, 0.03, 0.1, 0.3, 1.0],
        }

    if model_name == "XGBoost":
        return {
            "model__n_estimators": [100, 200, 300, 500],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [3, 4, 6, 8],
            "model__subsample": [0.7, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.9, 1.0],
            "model__reg_lambda": [0.5, 1.0, 2.0],
            "model__reg_alpha": [0.0, 0.5, 1.0],
        }

    return {}


def tune_model(
    model_name: str,
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str,
    n_iter: int,
    cv_splits: int,
    random_state: int,
    scoring: str = "roc_auc",
) -> Tuple[Pipeline, Dict[str, Any], float]:
    param_space = get_param_space(model_name)
    if not param_space:
        raise ValueError(f"Parameter space belum didefinisikan untuk model: {model_name}")

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    if method == "grid":
        search = GridSearchCV(
            pipe,
            param_grid=param_space,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )
    elif method == "random":
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_space,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            verbose=0,
        )
    else:
        raise ValueError("method harus 'grid' atau 'random'")

    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def cv_metrics(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cv_splits: int, random_state: int) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    res = cross_validate(
        pipe, X, y,
        cv=cv,
        scoring={"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "auc": "roc_auc"},
        n_jobs=-1,
        return_train_score=False,
    )
    return {
        "cv_accuracy": float(np.mean(res["test_accuracy"])),
        "cv_precision": float(np.mean(res["test_precision"])),
        "cv_recall": float(np.mean(res["test_recall"])),
        "cv_f1": float(np.mean(res["test_f1"])),
        "cv_auc": float(np.mean(res["test_auc"])),
    }