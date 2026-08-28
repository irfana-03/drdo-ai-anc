"""Dashboard styling — tactical AI communication monitor theme."""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-panel: #111827;
    --bg-card: #151d2e;
    --border: #1e2d45;
    --accent: #22d3ee;
    --accent-dim: #0891b2;
    --text: #e2e8f0;
    --text-dim: #94a3b8;
    --warn: #f87171;
    --ok: #34d399;
}

.stApp {
    background: var(--bg-primary);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header {visibility: hidden;}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 1400px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1320;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

.brand-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.08em;
    margin-bottom: 0.15rem;
}
.brand-sub {
    font-size: 0.65rem;
    color: var(--text-dim);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

.status-online {
    color: var(--ok);
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 1rem;
}

.nav-item {
    padding: 0.55rem 0.75rem;
    margin: 0.15rem 0;
    border: 1px solid transparent;
    border-radius: 4px;
    font-size: 0.8rem;
    letter-spacing: 0.06em;
    color: var(--text-dim);
    cursor: pointer;
}
.nav-item:hover {
    border-color: var(--border);
    color: var(--text);
    background: rgba(34, 211, 238, 0.05);
}
.nav-active {
    border-color: var(--accent-dim) !important;
    color: var(--accent) !important;
    background: rgba(34, 211, 238, 0.08) !important;
}

/* Cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.85rem 1rem;
    min-height: 90px;
    box-shadow: 0 0 20px rgba(34, 211, 238, 0.03);
}
.metric-label {
    font-size: 0.65rem;
    color: var(--text-dim);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--accent);
}
.metric-value-warn { color: var(--warn); }
.metric-value-ok { color: var(--ok); }
.metric-value-dim { color: var(--text-dim); font-size: 0.9rem; }

.panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}
.panel-title {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
}

.context-hero {
    text-align: center;
    padding: 1.5rem;
}
.context-main {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
    text-shadow: 0 0 30px rgba(34, 211, 238, 0.3);
}
.context-pct {
    font-size: 1.1rem;
    color: var(--text-dim);
    margin-top: 0.25rem;
}

.ctx-bar-row {
    display: flex;
    align-items: center;
    margin: 0.4rem 0;
    font-size: 0.8rem;
}
.ctx-bar-label {
    width: 100px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-dim);
}
.ctx-bar-active { color: var(--accent); font-weight: 600; }

.pipeline-stage {
    display: inline-block;
    text-align: center;
    padding: 0.5rem 0.6rem;
    margin: 0 0.15rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    font-size: 0.6rem;
    letter-spacing: 0.05em;
    min-width: 90px;
}
.pipeline-arrow {
    display: inline-block;
    color: var(--accent-dim);
    font-size: 1rem;
    vertical-align: middle;
    margin: 0 0.1rem;
}
.stage-ready { border-color: #1e3a2f; color: var(--ok); }
.stage-processing { border-color: var(--accent-dim); color: var(--accent); box-shadow: 0 0 8px rgba(34,211,238,0.2); }
.stage-warning { border-color: #7f1d1d; color: var(--warn); }

.sih-demo {
    background: linear-gradient(180deg, #0a0e17 0%, #111827 100%);
    border: 1px solid var(--accent-dim);
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
}
.sih-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    color: var(--accent);
    letter-spacing: 0.15em;
}
.sih-sub {
    font-size: 0.75rem;
    color: var(--text-dim);
    letter-spacing: 0.2em;
    margin: 0.5rem 0 1.5rem;
}

.warn-banner {
    background: rgba(248, 113, 113, 0.1);
    border: 1px solid var(--warn);
    border-radius: 4px;
    padding: 0.6rem 1rem;
    font-size: 0.8rem;
    color: var(--warn);
    margin: 0.5rem 0;
}

.stButton > button {
    background: var(--bg-card);
    color: var(--accent);
    border: 1px solid var(--accent-dim);
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
}
.stButton > button:hover {
    border-color: var(--accent);
    box-shadow: 0 0 12px rgba(34, 211, 238, 0.15);
}
</style>
"""
