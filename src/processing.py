from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from .config import AppConfig


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocess: ColumnTransformer
    num_cols: List[str]
    cat_cols: List[str]


def make_target(df: pd.DataFrame, cfg: AppConfig) -> Tuple[pd.Series, pd.DataFrame]:
    if cfg.target_col not in df.columns:
        raise ValueError(
            f"Kolom target '{cfg.target_col}' tidak ditemukan. Kolom tersedia: {df.columns.tolist()}"
        )
    y = (df[cfg.target_col] >= cfg.threshold).astype(int)
    X = df.drop(columns=[cfg.target_col])
    return y, X


def build_preprocess(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
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
    return preprocess, num_cols, cat_cols


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: AppConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, float]:
    # Split stratified sesuai gambar (Split Stratified)
    if cfg.eval_scheme == "split_70_30":
        test_size = 0.30
    else:
        # default split_80_20 dan cv_10fold tetap butuh test-set untuk visualisasi output
        test_size = 0.20

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=cfg.random_state,
    )
    return X_train, X_test, y_train, y_test, test_size


def prepare_data(df: pd.DataFrame, cfg: AppConfig) -> PreparedData:
    y, X = make_target(df, cfg)
    preprocess, num_cols, cat_cols = build_preprocess(X)
    X_train, X_test, y_train, y_test, _ = split_data(X, y, cfg)

    return PreparedData(
        X=X, y=y,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        preprocess=preprocess,
        num_cols=num_cols,
        cat_cols=cat_cols,
    )