from __future__ import annotations

import io
import time
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.pipeline import Pipeline

from src.modeling import (
    DatasetSpec,
    make_target,
    build_preprocess,
    evaluate_split_strategy,
    evaluate_cv_strategy,
    compare_models_on_split,
    tune_model,
    get_roc_data,
    get_confusion,
)
from src.viz import plot_roc_curve, plot_confusion_matrix

st.set_page_config(page_title="hypeapp - Student Performance Modeling", layout="wide")

st.title("hypeapp — Prediksi Prestasi Akademik & Optimasi Hyperparameter")
st.caption("Upload CSV → bandingkan model → tuning hyperparameter → pilih model terbaik (dinamis).")

# -------- Sidebar: upload & settings --------
st.sidebar.header("1) Upload Data")
uploaded = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

st.sidebar.header("2) Konfigurasi Target")
target_col = st.sidebar.text_input("Nama kolom target (nilai skor)", value="exam_score")
threshold = st.sidebar.number_input("Threshold (>= dianggap High/1)", value=75.0, step=1.0)

st.sidebar.header("3) Skema Evaluasi")
split_choice = st.sidebar.selectbox(
    "Pilih skema evaluasi",
    ["Stratified Split 80:20", "Stratified Split 70:30", "Stratified 10-Fold CV (mean)"]
)
random_state = st.sidebar.number_input("Random State", value=42, step=1)

st.sidebar.header("4) Tuning (opsional)")
do_tune = st.sidebar.checkbox("Aktifkan Hyperparameter Tuning", value=True)
tune_method = st.sidebar.selectbox("Metode Tuning", ["Randomized Search", "Grid Search"])
n_iter = st.sidebar.slider("n_iter (Randomized)", min_value=10, max_value=150, value=40, step=5)
cv_splits = st.sidebar.selectbox("Jumlah fold CV", [5, 10], index=1)

# ---------- Helpers ----------
def df_to_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ---------- Main ----------
if not uploaded:
    st.info("Silakan upload file CSV terlebih dahulu via sidebar.")
    st.stop()

# Read uploaded CSV
df = pd.read_csv(uploaded)
st.subheader("Preview Dataset")
st.write(f"Shape: **{df.shape[0]} baris** × **{df.shape[1]} kolom**")
st.dataframe(df.head(20), use_container_width=True)

# Build target
spec = DatasetSpec(target_col=target_col, threshold=float(threshold))
try:
    y, X = make_target(df, spec)
except Exception as e:
    st.error(str(e))
    st.stop()

colA, colB = st.columns(2)
with colA:
    st.subheader("Distribusi Target")
    vc = y.value_counts().rename("count").to_frame()
    vc["proportion"] = (y.value_counts(normalize=True)).round(4)
    st.dataframe(vc, use_container_width=True)
with colB:
    st.subheader("Ringkasan Fitur")
    st.write(f"Jumlah fitur: **{X.shape[1]}**")
    st.write(f"Jumlah numerik: **{X.select_dtypes(include=['number']).shape[1]}**")
    st.write(f"Jumlah kategorik: **{X.select_dtypes(exclude=['number']).shape[1]}**")

preprocess = build_preprocess(X)

# ---------- (A) Compare splitting technique for a baseline model (optional) ----------
st.divider()
st.subheader("A) Evaluasi Skema Pembagian Data (contoh baseline)")
st.caption("Bagian ini menilai dampak skema evaluasi terhadap metrik (menggunakan satu pipeline baseline).")

# Use a lightweight baseline pipeline: Logistic Regression
from sklearn.linear_model import LogisticRegression
baseline_pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=2000))
])

with st.expander("Lihat hasil evaluasi skema yang dipilih"):
    if split_choice == "Stratified Split 80:20":
        scores = evaluate_split_strategy(X, y, baseline_pipe, test_size=0.2, random_state=int(random_state))
        st.write("Skema: **80:20**")
    elif split_choice == "Stratified Split 70:30":
        scores = evaluate_split_strategy(X, y, baseline_pipe, test_size=0.3, random_state=int(random_state))
        st.write("Skema: **70:30**")
    else:
        scores = evaluate_cv_strategy(X, y, baseline_pipe, n_splits=int(cv_splits), random_state=int(random_state))
        st.write(f"Skema: **Stratified {cv_splits}-Fold CV (mean)**")

    st.json({k: round(v, 6) for k, v in scores.items()})

# Determine best split for model comparison (for simplicity: use split 80:20 or 70:30; CV is mean-based)
best_test_size = 0.2 if split_choice == "Stratified Split 80:20" else 0.3 if split_choice == "Stratified Split 70:30" else 0.2

# ---------- (B) Algorithm comparison on best split ----------
st.divider()
st.subheader("B) Perbandingan Algoritma (Baseline)")

algo_df, fitted_pipes, splits = compare_models_on_split(
    X, y, preprocess, test_size=best_test_size, random_state=int(random_state)
)
X_train, X_test, y_train, y_test = splits

st.dataframe(algo_df, use_container_width=True)

st.download_button(
    "Download CSV: baseline model comparison",
    data=df_to_download_bytes(algo_df),
    file_name="baseline_model_comparison.csv",
    mime="text/csv"
)

best_baseline_model = algo_df.loc[0, "model"]
st.success(f"Best BASELINE model by AUC: **{best_baseline_model}**")

# ---------- (C) Inspect a chosen model ----------
st.divider()
st.subheader("C) Detail Model Terpilih (ROC & Confusion Matrix)")

selected_model = st.selectbox("Pilih model untuk ditampilkan detailnya", algo_df["model"].tolist(), index=0)
pipe = fitted_pipes[selected_model]

c1, c2 = st.columns(2)

with c1:
    st.markdown("**ROC Curve**")
    roc = get_roc_data(pipe, X_test, y_test)
    if roc is None:
        st.warning("Model ini tidak menyediakan skor probabilitas/decision untuk ROC.")
    else:
        fpr, tpr, auc = roc
        fig = plot_roc_curve(fpr, tpr, auc)
        st.pyplot(fig, use_container_width=True)

with c2:
    st.markdown("**Confusion Matrix**")
    cm = get_confusion(pipe, X_test, y_test)
    fig2 = plot_confusion_matrix(cm, labels=("0", "1"))
    st.pyplot(fig2, use_container_width=True)

# ---------- (D) Hyperparameter tuning ----------
st.divider()
st.subheader("D) Hyperparameter Tuning (Grid / Randomized + CV)")

if not do_tune:
    st.info("Tuning dimatikan. Aktifkan di sidebar jika ingin tuning.")
    st.stop()

tune_target_model = st.selectbox(
    "Pilih model yang akan di-tuning",
    algo_df["model"].tolist(),
    index=0
)

method = "grid" if tune_method == "Grid Search" else "random"

st.write(
    f"Tuning model: **{tune_target_model}** dengan **{tune_method}** dan **Stratified {cv_splits}-Fold CV** "
    f"(scoring utama: **ROC-AUC**)."
)

base_pipe = fitted_pipes[tune_target_model]

run_btn = st.button("Jalankan Tuning")

if run_btn:
    with st.spinner("Sedang tuning..."):
        t0 = time.time()
        try:
            best_estimator, best_params, best_cv_auc = tune_model(
                model_name=tune_target_model,
                base_pipe=base_pipe,
                X_train=X_train,
                y_train=y_train,
                method=method,
                n_iter=int(n_iter),
                cv_splits=int(cv_splits),
                random_state=int(random_state),
            )
        except Exception as e:
            st.error(f"Gagal tuning: {e}")
            st.stop()

        elapsed = time.time() - t0

    # Evaluate tuned on test set
    from src.modeling import get_scores
    tuned_scores = get_scores(best_estimator, X_test, y_test)

    st.success(f"Selesai tuning dalam {elapsed:.2f} detik.")
    st.write("**Best CV AUC:**", round(best_cv_auc, 6))
    st.write("**Best Params:**")
    st.json(best_params)

    report = pd.DataFrame([{
        "best_search": f"{tune_target_model}_{tune_method.replace(' ', '_')}",
        **{k: float(v) for k, v in tuned_scores.items()},
        "best_cv_auc": float(best_cv_auc),
        "best_params": str(best_params),
        "time_sec": float(elapsed),
    }])

    st.subheader("Hasil Model Setelah Tuning (Test Set)")
    st.dataframe(report, use_container_width=True)

    st.download_button(
        "Download CSV: tuning report",
        data=df_to_download_bytes(report),
        file_name="tuning_report.csv",
        mime="text/csv"
    )

    st.subheader("ROC & Confusion Matrix (Tuned Model)")
    c3, c4 = st.columns(2)
    with c3:
        roc2 = get_roc_data(best_estimator, X_test, y_test)
        if roc2 is None:
            st.warning("Tidak ada skor untuk ROC.")
        else:
            fpr2, tpr2, auc2 = roc2
            st.pyplot(plot_roc_curve(fpr2, tpr2, auc2), use_container_width=True)
    with c4:
        cm2 = get_confusion(best_estimator, X_test, y_test)
        st.pyplot(plot_confusion_matrix(cm2, labels=("0", "1")), use_container_width=True)