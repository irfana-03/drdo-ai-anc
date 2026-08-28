"""Status metric cards."""

from __future__ import annotations

import streamlit as st


def _card(label: str, value: str, css_class: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {css_class}">{value}</div>
    </div>
    """


def render_status_cards(state, running: bool) -> None:
    if not running and state.noise_context == "WAITING FOR AUDIO":
        cols = st.columns(5)
        labels = ["NOISE CONTEXT", "CONFIDENCE", "IMPULSIVE EVENT", "SPEECH PROBABILITY", "PROCESSING LATENCY"]
        for i, lbl in enumerate(labels):
            cols[i].markdown(_card(lbl, "WAITING FOR AUDIO", "metric-value-dim"), unsafe_allow_html=True)
        return

    ctx = state.noise_context if running or state.noise_context != "WAITING FOR AUDIO" else "NO LIVE DATA"
    conf = f"{state.confidence * 100:.1f}%" if state.confidence > 0 else "N/A"
    imp = "YES" if state.is_impulsive else "NO"
    imp_cls = "metric-value-warn" if state.is_impulsive else "metric-value-ok"
    speech = f"{state.speech_probability * 100:.1f}%" if state.speech_probability > 0 else "N/A"
    lat = f"{state.latency_ms:.1f} ms" if state.latency_ms > 0 else "N/A"

    cols = st.columns(5)
    cols[0].markdown(_card("NOISE CONTEXT", ctx), unsafe_allow_html=True)
    cols[1].markdown(_card("CONFIDENCE", conf), unsafe_allow_html=True)
    cols[2].markdown(_card("IMPULSIVE EVENT", imp, imp_cls), unsafe_allow_html=True)
    cols[3].markdown(_card("SPEECH PROBABILITY", speech), unsafe_allow_html=True)
    cols[4].markdown(_card("PROCESSING LATENCY", lat), unsafe_allow_html=True)
