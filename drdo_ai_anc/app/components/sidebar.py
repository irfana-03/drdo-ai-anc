"""Sidebar navigation component."""

from __future__ import annotations

import streamlit as st

PAGES = [
    ("live", "LIVE MONITOR"),
    ("acoustic", "ACOUSTIC INTELLIGENCE"),
    ("pipeline", "AI PIPELINE"),
    ("performance", "MODEL PERFORMANCE"),
    ("data", "DATA & EVALUATION"),
    ("logs", "SESSION LOGS"),
    ("settings", "SYSTEM SETTINGS"),
]


def render_sidebar(model_status: str, audio_status: str, inference_status: str) -> str:
    with st.sidebar:
        st.markdown('<div class="brand-title">DRDO AI ANC</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="brand-sub">Context-Aware Adaptive Noise Cancellation</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="status-online">● SYSTEM ONLINE</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.markdown(f"<small>MODEL: {model_status}</small>", unsafe_allow_html=True)
        c2.markdown(f"<small>AUDIO: {audio_status}</small>", unsafe_allow_html=True)
        st.markdown(f"<small>INFERENCE: {inference_status}</small>", unsafe_allow_html=True)
        st.markdown("---")

        if "page" not in st.session_state:
            st.session_state.page = "live"

        for key, label in PAGES:
            active = "nav-active" if st.session_state.page == key else ""
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key

        st.markdown("---")
        if st.button("LAUNCH SIH DEMO MODE", use_container_width=True):
            st.session_state.sih_demo = not st.session_state.get("sih_demo", False)

        return st.session_state.page
