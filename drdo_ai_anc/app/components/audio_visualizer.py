"""Plotly audio visualizations."""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

DARK_LAYOUT = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#0a0e17",
    font=dict(color="#94a3b8", size=11, family="JetBrains Mono"),
    margin=dict(l=40, r=20, t=30, b=40),
    xaxis=dict(gridcolor="#1e2d45", zerolinecolor="#1e2d45"),
    yaxis=dict(gridcolor="#1e2d45", zerolinecolor="#1e2d45"),
)


def waveform_figure(audio: np.ndarray, sr: int, title: str, color: str = "#22d3ee") -> go.Figure:
    if audio is None or len(audio) == 0:
        fig = go.Figure()
        fig.update_layout(title=title, **DARK_LAYOUT)
        fig.add_annotation(text="NO LIVE DATA", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color="#64748b"))
        return fig

    t = np.arange(len(audio)) / sr * 1000
    fig = go.Figure(go.Scatter(x=t, y=audio, mode="lines", line=dict(color=color, width=1)))
    fig.update_layout(title=title, xaxis_title="ms", yaxis_title="amp", height=220, **DARK_LAYOUT)
    return fig


def spectrogram_figure(S: np.ndarray, sr: int, title: str = "LIVE SPECTRAL ANALYSIS") -> go.Figure:
    if S is None or S.size == 0 or S.shape[0] < 2:
        fig = go.Figure()
        fig.update_layout(title=title, **DARK_LAYOUT)
        fig.add_annotation(text="NO LIVE DATA", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color="#64748b"))
        return fig

    freqs = np.linspace(0, sr / 2, S.shape[0])
    times = np.arange(S.shape[1]) * 256 / sr
    fig = go.Figure(
        data=go.Heatmap(z=S, x=times, y=freqs, colorscale="Viridis", colorbar=dict(title="dB"))
    )
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Hz", height=280, **DARK_LAYOUT)
    return fig


def context_probabilities_bar(probs: dict, class_names: list) -> go.Figure:
    if not probs:
        fig = go.Figure()
        fig.update_layout(title="CURRENT ACOUSTIC CONTEXT", **DARK_LAYOUT)
        return fig

    labels, values = [], []
    for c in class_names:
        labels.append(c)
        values.append(probs.get(c, 0.0) * 100)

    colors = ["#22d3ee" if v == max(values) and v > 0 else "#1e3a5f" for v in values]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors))
    fig.update_layout(
        title="CURRENT ACOUSTIC CONTEXT",
        xaxis_title="%",
        height=220,
        **DARK_LAYOUT,
    )
    return fig


def context_timeline(timestamps: list, contexts: list, class_names: list) -> go.Figure:
    if not timestamps or not contexts:
        fig = go.Figure()
        fig.update_layout(title="NOISE CONTEXT OVER TIME", **DARK_LAYOUT)
        fig.add_annotation(text="NO DATA", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    mapping = {c: i for i, c in enumerate(class_names)}
    y = [mapping.get(c, 0) for c in contexts]
    fig = go.Figure(go.Scatter(x=timestamps, y=y, mode="lines+markers", line=dict(color="#22d3ee")))
    fig.update_layout(
        title="NOISE CONTEXT OVER TIME",
        yaxis=dict(tickvals=list(range(len(class_names))), ticktext=class_names),
        height=300,
        **DARK_LAYOUT,
    )
    return fig


def impulse_timeline(timestamps: list, impulsive: list) -> go.Figure:
    if not timestamps:
        fig = go.Figure()
        fig.update_layout(title="IMPULSIVE EVENT TIMELINE", **DARK_LAYOUT)
        return fig

    fig = go.Figure(go.Scatter(
        x=timestamps, y=[1 if i else 0 for i in impulsive],
        mode="markers", marker=dict(color="#f87171", size=8),
    ))
    fig.update_layout(title="IMPULSIVE EVENT TIMELINE", yaxis=dict(tickvals=[0, 1], ticktext=["NO", "YES"]), height=200, **DARK_LAYOUT)
    return fig
