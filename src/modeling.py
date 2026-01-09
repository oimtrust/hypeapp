from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

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


@dataclass
class DatasetSpec:
    target_col: str = "exam_score"
    threshold: float = 75.0
    positive_label_name: str = "High"
    negative_label_name: str = "Low"


def make_target(df: pd.DataFrame, spec: DatasetSpec) -> Tuple[pd.Series, pd.DataFrame]:
    if spec.target_col not in df.columns:
        raise ValueError(f"Kolom target '{spec.target_col}' tidak ditemukan. Kolom tersedia: {df.columns.tolist()}")
    y = (df[spec.target_col] >= spec.threshold).astype(int)
    X = df.drop(columns=[spec.target_col])
    return y, X


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocess


def get_scores(estimator, X_test, y_test) -> Dict[str, float]:
    y_pred = estimator.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # AUC
    y_score = None
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X_test)[:, 1]
    elif hasattr(estimator, "decision_function"):
        y_score = estimator.decision_function(X_test)

    auc = roc_auc_score(y_test, y_score) if y_score is not None else float("nan")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


def get_roc_data(estimator, X_test, y_test) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    y_score = None
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X_test)[:, 1]
    elif hasattr(estimator, "decision_function"):
        y_score = estimator.decision_function(X_test)
    if y_score is None:
        return None
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    return fpr, tpr, auc


def get_confusion(estimator, X_test, y_test) -> np.ndarray:
    y_pred = estimator.predict(X_test)
    return confusion_matrix(y_test, y_pred)


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


def evaluate_split_strategy(
    X: pd.DataFrame,
    y: pd.Series,
    base_pipe: Pipeline,
    test_size: float,
    random_state: int = 42,
) -> Dict[str, float]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    base_pipe.fit(X_train, y_train)
    return get_scores(base_pipe, X_test, y_test)


def evaluate_cv_strategy(
    X: pd.DataFrame,
    y: pd.Series,
    base_pipe: Pipeline,
    n_splits: int = 10,
    random_state: int = 42,
) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    res = cross_validate(
        base_pipe,
        X, y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "auc": "roc_auc",
        },
        n_jobs=-1,
        return_train_score=False,
    )
    return {
        "accuracy": float(np.mean(res["test_accuracy"])),
        "precision": float(np.mean(res["test_precision"])),
        "recall": float(np.mean(res["test_recall"])),
        "f1": float(np.mean(res["test_f1"])),
        "auc": float(np.mean(res["test_auc"])),
    }


def compare_models_on_split(
    X: pd.DataFrame,
    y: pd.Series,
    preprocess: ColumnTransformer,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    models = build_models(random_state=random_state)

    rows: List[Dict[str, Any]] = []
    fitted: Dict[str, Pipeline] = {}

    for name, mdl in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", mdl)])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        s = get_scores(pipe, X_test, y_test)
        rows.append({"model": name, **s})

    df_scores = pd.DataFrame(rows).sort_values(by="auc", ascending=False).reset_index(drop=True)
    return df_scores, fitted, (X_train, X_test, y_train, y_test)


def get_param_space(model_name: str) -> Dict[str, Any]:
    # Param names follow Pipeline step "model__"
    if model_name == "LogisticRegression":
        return {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs"],
        }

    if model_name.startswith("SVM_"):
        # gamma only meaningful for rbf/poly/sigmoid but harmless to include; sklearn will ignore for linear? (it won't ignore)
        # So tailor by kernel:
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
        # default RBF
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
    base_pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "random",
    n_iter: int = 40,
    cv_splits: int = 10,
    random_state: int = 42,
) -> Tuple[Any, Dict[str, Any], float]:
    param_space = get_param_space(model_name)
    if not param_space:
        raise ValueError(f"Parameter space belum didefinisikan untuk model: {model_name}")

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    if method.lower() == "grid":
        search = GridSearchCV(
            base_pipe,
            param_grid=param_space,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=0
        )
    else:
        # Randomized
        search = RandomizedSearchCV(
            base_pipe,
            param_distributions=param_space,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )

    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)