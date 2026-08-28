"""Pipeline visualization component."""

from __future__ import annotations

import streamlit as st


def render_pipeline_stages(stages: list) -> None:
    html_parts = []
    for i, stage in enumerate(stages):
        status = stage.get("status", "READY")
        if status in ("PROCESSING", "READY"):
            css = "stage-processing" if status == "PROCESSING" else "stage-ready"
        elif status == "WARNING" or status == "NOT FOUND":
            css = "stage-warning"
        else:
            css = "stage-ready"
        html_parts.append(
            f'<div class="pipeline-stage {css}">{stage["name"]}<br>● {status}</div>'
        )
        if i < len(stages) - 1:
            html_parts.append('<span class="pipeline-arrow">↓</span>')

    st.markdown(
        f'<div style="text-align:center;padding:0.5rem 0;">{"".join(html_parts)}</div>',
        unsafe_allow_html=True,
    )


def render_architecture_diagram() -> None:
    st.markdown(
        """
        <div class="panel">
        <div class="panel-title">AI Processing Architecture</div>
        <pre style="color:#94a3b8;font-family:'JetBrains Mono',monospace;font-size:0.75rem;line-height:1.6;text-align:center;">
                 LIVE AUDIO
                     │
                     ▼
             ┌───────────────┐
             │ AUDIO STREAM  │
             └───────┬───────┘
                     ▼
             ┌───────────────┐
             │   FEATURES    │
             │ STFT / MEL    │
             └───────┬───────┘
                     ▼
        ┌────────────────────────┐
        │ ACOUSTIC INTELLIGENCE  │
        │ CONTEXT CLASSIFIER     │
        └───────────┬────────────┘
                    ▼
       ┌────────────┼────────────┐
       ▼            ▼            ▼
 STATIONARY      DYNAMIC      IMPULSIVE
       └────────────┼────────────┘
                    ▼
          ADAPTIVE CONTROLLER
                    ▼
             DEEPFILTERNET
                    ▼
             RESIDUAL LMS
                    ▼
            SPEECH PRIORITY
                    ▼
             ENHANCED OUTPUT
        </pre>
        </div>
        """,
        unsafe_allow_html=True,
    )
