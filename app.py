from __future__ import annotations

import time
import pandas as pd
import streamlit as st

from src.config import AppConfig
from src.processing import prepare_data
from src.modeling import build_models, build_pipeline, tune_model, cv_metrics
from src.evaluation import predict_scores, get_confusion, get_roc
from src.viz import plot_confusion_matrix, plot_roc_curve
from src.reporting import to_csv_bytes, select_best_model, merge_baseline_and_tuned

st.set_page_config(page_title="hypeapp - Student Performance Modeling", layout="wide")

st.title("hypeapp — Prediksi Prestasi Akademik & Optimasi Hyperparameter")
st.caption("Alur: Upload & Konfigurasi → Data Processing → Model & Tuning → Evaluation → Output (Download Report).")

# =========================================================
# 1) Upload Data & Konfigurasi
# =========================================================
st.sidebar.header("Upload Data & Konfigurasi")

uploaded = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

cfg = AppConfig(
    target_col=st.sidebar.text_input("Nama kolom target (nilai skor)", value="exam_score"),
    threshold=float(st.sidebar.number_input("Threshold (>= dianggap High/1)", value=75.0, step=1.0)),
    random_state=int(st.sidebar.number_input("Random State", value=42, step=1)),
    cv_splits=int(st.sidebar.selectbox("Fold CV (untuk tuning & CV metrics)", [5, 10], index=1)),
)

cfg.eval_scheme = st.sidebar.selectbox(
    "Skema evaluasi (split stratified / CV)",
    [
        ("split_80_20", "Stratified Split 80:20"),
        ("split_70_30", "Stratified Split 70:30"),
        ("cv_10fold", "Stratified 10-Fold CV (mean)"),
    ],
    format_func=lambda x: x[1],
)[0]

st.sidebar.header("Model & Tuning")
cfg.tuning_method = st.sidebar.selectbox(
    "Metode tuning",
    [("none", "Tanpa tuning"), ("grid", "Grid Search"), ("random", "Randomized Search")],
    format_func=lambda x: x[1],
)[0]
cfg.n_iter = int(st.sidebar.slider("n_iter (Randomized)", 10, 150, 40, 5))

# opsional: feature selection (placeholder)
cfg.use_feature_selection = st.sidebar.checkbox("Aktifkan Feature Selection (opsional)", value=False)
cfg.top_k_features = int(st.sidebar.slider("Top-k fitur (jika aktif)", 5, 50, 20, 5))

if not uploaded:
    st.info("Silakan upload file CSV terlebih dahulu melalui sidebar.")
    st.stop()

df = pd.read_csv(uploaded)

st.subheader("Preview Dataset")
st.write(f"Shape: **{df.shape[0]} baris** × **{df.shape[1]} kolom**")
st.dataframe(df.head(15), use_container_width=True)

# =========================================================
# 2) Data Processing
# =========================================================
st.divider()
st.subheader("Data Processing")
st.caption("Cleaning & Encoding → Normalisasi → Split (Stratified) → (opsional) Feature Selection.")

try:
    prepared = prepare_data(df, cfg)
except Exception as e:
    st.error(str(e))
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Jumlah fitur", prepared.X.shape[1])
with c2:
    st.metric("Fitur numerik", len(prepared.num_cols))
with c3:
    st.metric("Fitur kategorik", len(prepared.cat_cols))

st.write("Distribusi target (0=Low, 1=High):")
target_dist = prepared.y.value_counts().rename("count").to_frame()
target_dist["pct"] = (prepared.y.value_counts(normalize=True) * 100).round(2)
st.dataframe(target_dist, use_container_width=True)

# =========================================================
# 3) Model & Tuning
# =========================================================
st.divider()
st.subheader("Model & Tuning")
st.caption("Baseline model → (opsional) tuning hyperparameter (Grid/Randomized) dengan Stratified CV (scoring ROC-AUC).")

models = build_models(random_state=cfg.random_state)
model_names = list(models.keys())
cfg.selected_model = st.selectbox("Pilih model", model_names, index=model_names.index("SVM_RBF") if "SVM_RBF" in model_names else 0)

# Baseline fit
baseline_pipe = build_pipeline(prepared.preprocess, models[cfg.selected_model])
baseline_pipe.fit(prepared.X_train, prepared.y_train)

baseline_scores = predict_scores(baseline_pipe, prepared.X_test, prepared.y_test)
baseline_cv = cv_metrics(baseline_pipe, prepared.X, prepared.y, cv_splits=cfg.cv_splits, random_state=cfg.random_state) if cfg.eval_scheme == "cv_10fold" else {}

baseline_row = {
    "model": cfg.selected_model,
    "stage": "baseline",
    **baseline_scores,
    **({"best_cv_auc": baseline_cv.get("cv_auc")} if baseline_cv else {}),
    "best_params": None,
    "time_sec": None,
}
baseline_df = pd.DataFrame([baseline_row])

st.write("**Baseline metrics (test set):**")
st.dataframe(baseline_df[["model", "accuracy", "precision", "recall", "f1", "auc"]], use_container_width=True)

# Tuning
tuned_df = pd.DataFrame()
best_estimator = None

if cfg.tuning_method != "none":
    st.markdown("#### Tuning Hyperparameter")
    run_tune = st.button("Jalankan Tuning")

    if run_tune:
        with st.spinner("Sedang tuning..."):
            t0 = time.time()
            best_estimator, best_params, best_cv_auc = tune_model(
                model_name=cfg.selected_model,
                pipe=baseline_pipe,  # pipeline yang sama (preprocess + model)
                X_train=prepared.X_train,
                y_train=prepared.y_train,
                method=cfg.tuning_method,
                n_iter=cfg.n_iter,
                cv_splits=cfg.cv_splits,
                random_state=cfg.random_state,
                scoring="roc_auc",
            )
            elapsed = time.time() - t0

        tuned_scores = predict_scores(best_estimator, prepared.X_test, prepared.y_test)
        tuned_row = {
            "model": cfg.selected_model,
            "stage": f"tuned_{cfg.tuning_method}",
            **tuned_scores,
            "best_cv_auc": float(best_cv_auc),
            "best_params": str(best_params),
            "time_sec": float(elapsed),
        }
        tuned_df = pd.DataFrame([tuned_row])

        st.success(f"Selesai tuning dalam {elapsed:.2f} detik.")
        st.write("**Best CV AUC:**", round(best_cv_auc, 6))
        st.write("**Best Params:**")
        st.json(best_params)

        st.write("**Tuned metrics (test set):**")
        st.dataframe(tuned_df[["model", "accuracy", "precision", "recall", "f1", "auc", "best_cv_auc"]], use_container_width=True)

# =========================================================
# 4) Evaluation (Confusion Matrix + ROC)
# =========================================================
st.divider()
st.subheader("Evaluation")
st.caption("Confusion Matrix + ROC Curve digunakan untuk memperjelas pola kesalahan dan kualitas pemisahan kelas.")

use_estimator = best_estimator if best_estimator is not None else baseline_pipe
stage_label = tuned_df.iloc[0]["stage"] if not tuned_df.empty else "baseline"

cc1, cc2 = st.columns(2)

with cc1:
    cm = get_confusion(use_estimator, prepared.X_test, prepared.y_test)
    st.pyplot(plot_confusion_matrix(cm, labels=("0", "1"), title=f"Confusion Matrix ({cfg.selected_model} - {stage_label})"), use_container_width=True)

with cc2:
    roc = get_roc(use_estimator, prepared.X_test, prepared.y_test)
    if roc is None:
        st.warning("Model ini tidak menyediakan skor probabilitas/decision untuk ROC.")
    else:
        fpr, tpr, auc = roc
        st.pyplot(plot_roc_curve(fpr, tpr, auc, title=f"ROC Curve ({cfg.selected_model} - {stage_label})"), use_container_width=True)

# =========================================================
# 5) Output (Tabel evaluasi + Best model + Download)
# =========================================================
st.divider()
st.subheader("Output")
st.caption("Tabel evaluasi → visualisasi → ringkasan model terbaik → unduh report CSV.")

final_report = merge_baseline_and_tuned(baseline_df, tuned_df if not tuned_df.empty else None)
# untuk 1 model, best model = baris dengan auc terbesar
best_model_stage = select_best_model(final_report.rename(columns={"auc": "auc"}), primary="auc")

st.write("**Report gabungan (baseline + tuning jika ada):**")
st.dataframe(final_report, use_container_width=True)

st.success(f"Ringkasan: performa terbaik untuk model terpilih diperoleh pada **{best_model_stage}** (berdasarkan AUC).")

st.download_button(
    "Download CSV: report evaluasi",
    data=to_csv_bytes(final_report),
    file_name="hypeapp_report.csv",
    mime="text/csv",
)