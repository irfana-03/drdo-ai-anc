"""Model performance and dataset metrics views."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

DARK = dict(paper_bgcolor="#111827", plot_bgcolor="#0a0e17", font=dict(color="#94a3b8"))


def render_performance_metrics(metrics: Optional[Dict[str, Any]]) -> None:
    if metrics is None:
        st.warning("N/A — insufficient validated reference data")
        return

    cols = st.columns(4)
    cols[0].metric("TEST ACCURACY", f"{metrics['accuracy']:.1%}")
    cols[1].metric("MACRO F1", f"{metrics['macro_f1']:.3f}")
    cols[2].metric("MACRO PRECISION", f"{metrics['macro_precision']:.3f}")
    cols[3].metric("MACRO RECALL", f"{metrics['macro_recall']:.3f}")

    st.caption("Metrics reflect the current real-data test split.")

    rows = []
    for cls, vals in metrics.get("per_class", {}).items():
        support = 0
        cm = metrics.get("confusion_matrix", [])
        class_names = list(metrics.get("per_class", {}).keys())
        if cls in class_names:
            idx = class_names.index(cls)
            if idx < len(cm):
                support = sum(cm[idx])
        rows.append({
            "Class": cls,
            "Precision": f"{vals['precision']:.3f}",
            "Recall": f"{vals['recall']:.3f}",
            "F1": f"{vals['f1']:.3f}",
            "Support": support,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_confusion_matrix_image(path: Optional[Path]) -> None:
    if path is None:
        st.info("Confusion matrix image not available.")
        return
    st.image(str(path), use_container_width=True)


def render_dataset_cards(report: Optional[Dict[str, Any]]) -> None:
    if report is None:
        st.warning("Dataset report not available. Run scripts/dataset_report.py first.")
        return

    cols = st.columns(6)
    cols[0].metric("REAL RECORDINGS", report.get("total_recordings", "N/A"))
    cols[1].metric("DURATION (h)", report.get("total_duration_h", "N/A"))
    splits = report.get("splits", {})
    cols[2].metric("TRAIN", splits.get("train", "N/A"))
    cols[3].metric("VALIDATION", splits.get("validation", "N/A"))
    cols[4].metric("TEST", splits.get("test", "N/A"))
    cols[5].metric("CLASSES", len(report.get("classes", {})))


def render_dataset_table(report: Optional[Dict], availability: Dict[str, str]) -> None:
    rows = []
    for key, label in [("demand", "DEMAND"), ("sonyc_ust", "SONYC-UST"), ("chime3", "CHiME-3")]:
        ds = report.get("datasets", {}).get(key, {}) if report else {}
        rows.append({
            "Dataset": label,
            "Recordings": ds.get("recordings", "—"),
            "Duration (s)": ds.get("duration_s", "—"),
            "Classes": ", ".join(ds.get("classes", {}).keys()) if ds.get("classes") else "—",
            "Status": availability.get(key, "NOT AVAILABLE"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_distribution_charts(meta: Optional[pd.DataFrame]) -> None:
    if meta is None or meta.empty:
        st.info("Metadata not available.")
        return

    c1, c2 = st.columns(2)
    with c1:
        class_counts = meta["mapped_context"].value_counts().reset_index()
        class_counts.columns = ["Class", "Count"]
        fig = px.bar(class_counts, x="Class", y="Count", title="Class Distribution")
        fig.update_layout(**DARK)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        ds_counts = meta["dataset"].value_counts().reset_index()
        ds_counts.columns = ["Dataset", "Count"]
        fig = px.pie(ds_counts, names="Dataset", values="Count", title="Dataset Distribution")
        fig.update_layout(**DARK)
        st.plotly_chart(fig, use_container_width=True)

    split_counts = meta["split"].value_counts().reset_index()
    split_counts.columns = ["Split", "Count"]
    fig = px.bar(split_counts, x="Split", y="Count", title="Train / Validation / Test Split")
    fig.update_layout(**DARK)
    st.plotly_chart(fig, use_container_width=True)
