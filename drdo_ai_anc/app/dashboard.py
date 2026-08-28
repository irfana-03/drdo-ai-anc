"""DRDO AI-ANC — SIH Demonstration Dashboard."""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st

# Project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.audio_visualizer import (
    context_probabilities_bar,
    context_timeline,
    impulse_timeline,
    spectrogram_figure,
    waveform_figure,
)
from app.components.metrics_view import (
    render_confusion_matrix_image,
    render_dataset_cards,
    render_dataset_table,
    render_distribution_charts,
    render_performance_metrics,
)
from app.components.pipeline_view import render_architecture_diagram, render_pipeline_stages
from app.components.sidebar import render_sidebar
from app.components.status_cards import render_status_cards
from app.components.styles import CUSTOM_CSS
from app.utils.data_loader import (
    check_dataset_availability,
    find_sample_wav,
    get_confusion_matrix_path,
    load_audio_metadata,
    load_dataset_report,
    load_metrics,
    load_session_logs,
)
from app.utils.model_loader import (
    build_pipeline,
    get_audio_devices,
    get_enhancement_status,
    get_model_paths,
    get_system_info,
    load_classifier,
    load_config,
)
from app.utils.realtime_manager import RealtimeManager

st.set_page_config(
    page_title="DRDO AI ANC",
    page_icon="🎙",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def _cached_classifier():
    return load_classifier()


@st.cache_resource
def _cached_pipeline(_clf_tuple):
    clf, _ = _clf_tuple
    if clf is None:
        return None
    return build_pipeline(clf)


def _init_session():
    defaults = {
        "page": "live",
        "system_running": False,
        "sih_demo": False,
        "offline_result": None,
        "offline_sr": 48000,
        "engine_choice": "auto",
        "block_size": 4800,
        "sample_rate": 48000,
        "input_device": None,
        "output_device": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _get_manager() -> RealtimeManager | None:
    clf, status = _cached_classifier()
    if clf is None:
        return None
    pipeline = _cached_pipeline((clf, status))
    if "rt_manager" not in st.session_state or st.session_state.rt_manager is None:
        st.session_state.rt_manager = RealtimeManager(pipeline)
    return st.session_state.rt_manager


def _current_state():
    mgr = _get_manager()
    if mgr is None:
        from src.realtime.pipeline import PipelineState
        return PipelineState()
    if st.session_state.system_running:
        return mgr.get_state()
    if st.session_state.offline_result:
        return st.session_state.offline_result.get("state", mgr.get_state())
    return mgr.get_state()


def _status_strings():
    clf, model_st = _cached_classifier()
    audio_st = "RUNNING" if st.session_state.system_running else (
        "READY" if clf else "NOT CONNECTED"
    )
    inf_st = "RUNNING" if st.session_state.system_running or st.session_state.offline_result else "IDLE"
    return model_st, audio_st, inf_st


# ------------------------------------------------------------------ Pages ----

def page_live_monitor():
    st.markdown("## LIVE ACOUSTIC ENVIRONMENT MONITOR")
    st.caption("Real-time context-aware noise suppression")

    state = _current_state()
    running = st.session_state.system_running or st.session_state.offline_result is not None
    render_status_cards(state, running)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            waveform_figure(state.input_buffer, 48000, "INPUT SIGNAL", "#64748b"),
            use_container_width=True,
            key="wf_in",
        )
    with c2:
        st.plotly_chart(
            waveform_figure(state.output_buffer, 48000, "ENHANCED SIGNAL", "#22d3ee"),
            use_container_width=True,
            key="wf_out",
        )

    st.plotly_chart(
        spectrogram_figure(state.spectrogram, 48000),
        use_container_width=True,
        key="spec",
    )

    clf, _ = _cached_classifier()
    class_names = clf.class_names if clf else ["STATIONARY", "DYNAMIC", "IMPULSIVE", "SPEECH", "OTHER"]

    c3, c4 = st.columns([1, 1])
    with c3:
        st.plotly_chart(
            context_probabilities_bar(state.probabilities, class_names),
            use_container_width=True,
            key="ctx_bar",
        )
    with c4:
        st.markdown('<div class="panel"><div class="panel-title">ADAPTIVE PROCESSING STRATEGY</div>', unsafe_allow_html=True)
        if state.strategy_description:
            st.markdown(f"**{state.strategy_name}**")
            st.markdown(state.strategy_description)
            st.markdown(f"Engine: `{state.engine or 'N/A'}`")
        else:
            st.markdown("NO LIVE DATA")
        st.markdown("</div>", unsafe_allow_html=True)

    mgr = _get_manager()
    enh = get_enhancement_status()
    engine_status = {
        "audio": "PROCESSING" if st.session_state.system_running else "READY",
        "running": st.session_state.system_running,
        "deepfilternet": enh.get("deepfilternet", "NOT FOUND"),
    }
    if mgr and clf:
        stages = mgr.pipeline.controller.pipeline_stages(
            state.noise_context if state.noise_context != "WAITING FOR AUDIO" else "",
            engine_status,
        )
        st.markdown('<div class="panel"><div class="panel-title">PROCESSING PIPELINE</div>', unsafe_allow_html=True)
        render_pipeline_stages(stages)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">LIVE CONTROL PANEL</div>', unsafe_allow_html=True)
    devices = get_audio_devices()
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        in_opts = {f"{d['index']}: {d['name']}": d["index"] for d in devices["inputs"]}
        in_sel = st.selectbox("INPUT DEVICE", ["Default"] + list(in_opts.keys()))
        st.session_state.input_device = in_opts.get(in_sel) if in_sel != "Default" else None
    with cc2:
        out_opts = {f"{d['index']}: {d['name']}": d["index"] for d in devices["outputs"]}
        out_sel = st.selectbox("OUTPUT DEVICE", ["Default"] + list(out_opts.keys()))
        st.session_state.output_device = out_opts.get(out_sel) if out_sel != "Default" else None
    with cc3:
        st.session_state.engine_choice = st.selectbox("ENGINE", ["auto", "Spectral+LMS"])

    cc4, cc5 = st.columns(2)
    with cc4:
        st.session_state.block_size = st.number_input("BLOCK SIZE", 480, 9600, st.session_state.block_size, 480)
    with cc5:
        st.session_state.sample_rate = st.selectbox("SAMPLE RATE", [16000, 48000], index=1)

    if RealtimeManager.feedback_warning(st.session_state.input_device, st.session_state.output_device):
        st.markdown(
            '<div class="warn-banner">INPUT AND OUTPUT DEVICE MAY CAUSE ACOUSTIC FEEDBACK</div>',
            unsafe_allow_html=True,
        )

    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        if st.button("▶ START SYSTEM"):
            if mgr is None:
                st.error("MODEL NOT AVAILABLE")
            else:
                try:
                    mgr.engine_choice = st.session_state.engine_choice
                    mgr.start(
                        input_device=st.session_state.input_device,
                        block_size=int(st.session_state.block_size),
                        sample_rate=int(st.session_state.sample_rate),
                    )
                    st.session_state.system_running = True
                    st.session_state.offline_result = None
                except Exception as exc:
                    st.error(str(exc))
    with bc2:
        if st.button("■ STOP SYSTEM"):
            if mgr:
                mgr.stop()
            st.session_state.system_running = False
    with bc3:
        if st.button("⟳ RESET"):
            if mgr:
                mgr.reset()
            st.session_state.system_running = False
            st.session_state.offline_result = None

    st.markdown("---")
    st.markdown("**OFFLINE DEMO** — Process a real WAV file through the same pipeline")
    sample = find_sample_wav()
    if sample:
        st.caption(f"Sample available: {sample.name}")

    uploaded = st.file_uploader("Upload WAV file", type=["wav", "flac", "ogg"])
    if st.button("PROCESS OFFLINE FILE"):
        if mgr is None:
            st.error("MODEL NOT AVAILABLE")
        else:
            try:
                if uploaded:
                    audio, sr = sf.read(io.BytesIO(uploaded.read()), dtype="float32")
                elif sample:
                    audio, sr = sf.read(str(sample), dtype="float32")
                else:
                    st.warning("No file uploaded and no sample WAV found in data/raw/")
                    st.stop()
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                mgr.engine_choice = st.session_state.engine_choice
                result = mgr.process_offline(audio, sr)
                st.session_state.offline_result = result
                st.session_state.offline_sr = sr
                st.session_state.system_running = False
                st.success(f"Processed in {result['total_time_s']:.2f}s — RTF: {result['rtf']:.3f}")
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.offline_result:
        res = st.session_state.offline_result
        sr = st.session_state.offline_sr
        buf_in = io.BytesIO()
        buf_out = io.BytesIO()
        sf.write(buf_in, res["input"], sr, format="WAV")
        sf.write(buf_out, res["enhanced"], sr, format="WAV")
        a1, a2 = st.columns(2)
        a1.audio(buf_in.getvalue(), format="audio/wav")
        a2.audio(buf_out.getvalue(), format="audio/wav")


def page_acoustic():
    st.markdown("## ACOUSTIC INTELLIGENCE")
    state = _current_state()
    clf, _ = _cached_classifier()
    class_names = clf.class_names if clf else []

    c1, c2 = st.columns(2)
    c1.metric("Current Classification", state.noise_context if state.probabilities else "NO DATA")
    c2.metric("Confidence", f"{state.confidence * 100:.1f}%" if state.confidence else "N/A")

    if state.features:
        fc = st.columns(4)
        fc[0].metric("RMS", state.features.get("rms", "N/A"))
        fc[1].metric("Spectral Centroid", state.features.get("spectral_centroid", "N/A"))
        fc[2].metric("Spectral Flux", state.features.get("spectral_flux", "N/A"))
        fc[3].metric("Zero Crossing Rate", state.features.get("zero_crossing_rate", "N/A"))
    else:
        st.info("NO LIVE DATA — start system or run offline demo")

    st.plotly_chart(
        context_timeline(state.history_timestamps, state.history_context, class_names),
        use_container_width=True,
    )
    st.plotly_chart(
        impulse_timeline(state.history_timestamps, state.history_impulsive),
        use_container_width=True,
    )


def page_pipeline():
    st.markdown("## AI PROCESSING ARCHITECTURE")
    render_architecture_diagram()


def page_performance():
    st.markdown("## MODEL PERFORMANCE")
    metrics = load_metrics()
    render_performance_metrics(metrics)
    st.markdown("### Confusion Matrix")
    render_confusion_matrix_image(get_confusion_matrix_path())


def page_data():
    st.markdown("## REAL DATASET INTELLIGENCE")
    report = load_dataset_report()
    availability = check_dataset_availability()
    render_dataset_cards(report)
    render_dataset_table(report, availability)
    st.markdown("### Dataset Distribution")
    render_distribution_charts(load_audio_metadata())


def page_logs():
    st.markdown("## SESSION LOGS")
    df = load_session_logs()
    if df.empty:
        st.info("NO SESSION DATA AVAILABLE")
        return
    col1, col2 = st.columns(2)
    with col1:
        ctx_filter = st.multiselect("Filter context", df["noise_context"].unique().tolist())
    with col2:
        mode_filter = st.multiselect("Filter mode", df["mode"].unique().tolist())

    filtered = df.copy()
    if ctx_filter:
        filtered = filtered[filtered["noise_context"].isin(ctx_filter)]
    if mode_filter:
        filtered = filtered[filtered["mode"].isin(mode_filter)]

    sort_col = st.selectbox("Sort by", filtered.columns.tolist(), index=0)
    filtered = filtered.sort_values(sort_col, ascending=False)
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", filtered.to_csv(index=False), "sessions.csv", "text/csv")


def page_settings():
    st.markdown("## SYSTEM SETTINGS")
    clf, model_st = _cached_classifier()
    enh = get_enhancement_status()
    sysinfo = get_system_info()
    cfg = load_config()
    paths = get_model_paths()

    st.markdown("### Model Status")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classifier", model_st)
    c2.metric("DeepFilterNet", enh.get("deepfilternet", "NOT FOUND"))
    c3.metric("RNNoise", enh.get("rnnoise", "NOT FOUND"))
    c4.metric("CUDA", sysinfo["cuda"])

    st.markdown("### System")
    st.write(f"CPU: {sysinfo['cpu']}")
    if sysinfo.get("ram_total_gb"):
        st.write(f"RAM: {sysinfo['ram_used_gb']} / {sysinfo['ram_total_gb']} GB ({sysinfo['ram_pct']}%)")
    st.write(f"Python: {sysinfo['python']}")

    st.markdown("### Audio Devices")
    devices = get_audio_devices()
    st.write("Inputs:", devices["inputs"] or "None detected")
    st.write("Outputs:", devices["outputs"] or "None detected")

    st.markdown("### Configuration")
    st.json({
        "sample_rate": cfg.get("audio", {}).get("target_sample_rate"),
        "block_size": cfg.get("realtime", {}).get("block_size"),
        "confidence_threshold": cfg.get("training", {}).get("confidence_threshold"),
        "model_path": str(paths["classifier"]),
    })


def page_sih_demo():
    state = _current_state()
    st.markdown(
        f"""
        <div class="sih-demo">
            <div class="sih-title">DRDO AI ANC</div>
            <div class="sih-sub">CONTEXT-AWARE ADAPTIVE NOISE CANCELLATION</div>
            <div class="status-online">● SYSTEM ONLINE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    ctx = state.noise_context if state.probabilities else "NO DATA"
    c1.metric("CURRENT ENVIRONMENT", ctx)
    c2.metric("AI CONFIDENCE", f"{state.confidence * 100:.0f}%" if state.confidence else "N/A")
    c3.metric("SPEECH PROBABILITY", f"{state.speech_probability * 100:.0f}%" if state.speech_probability else "N/A")
    c4.metric("IMPULSIVE EVENT", "YES" if state.is_impulsive else "NO")

    st.markdown(f"**ADAPTIVE PROCESSING:** {state.strategy_description or 'N/A'}")

    w1, w2 = st.columns(2)
    w1.plotly_chart(waveform_figure(state.input_buffer, 48000, "INPUT AUDIO"), use_container_width=True)
    w2.plotly_chart(waveform_figure(state.output_buffer, 48000, "ENHANCED SPEECH"), use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("LATENCY", f"{state.latency_ms:.1f} ms" if state.latency_ms else "N/A")
    m2.metric("REAL-TIME FACTOR", f"{state.rtf:.3f}" if state.rtf else "N/A")
    m3.metric("ENGINE", state.engine or "N/A")


# ------------------------------------------------------------------ Main ----

def main():
    _init_session()
    model_st, audio_st, inf_st = _status_strings()
    render_sidebar(model_st, audio_st, inf_st)

    if st.session_state.get("sih_demo"):
        page_sih_demo()
    else:
        pages = {
            "live": page_live_monitor,
            "acoustic": page_acoustic,
            "pipeline": page_pipeline,
            "performance": page_performance,
            "data": page_data,
            "logs": page_logs,
            "settings": page_settings,
        }
        pages.get(st.session_state.page, page_live_monitor)()

    if st.session_state.system_running:
        time.sleep(0.4)
        st.rerun()


if __name__ == "__main__":
    main()
