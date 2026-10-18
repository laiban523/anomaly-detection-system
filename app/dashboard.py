import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from storage.db import AnomalyRecord, get_engine, get_session
from detection.metrics import full_report

# Page configuration
st.set_page_config(
    page_title="IoT Anomaly Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Lock sidebar open. hides the collapse button and auto-expands if somehow collapsed
components.html("""
<script>
(function() {
  var doc = window.parent.document;

  function lockSidebar() {
    // 1. Hide every button that could be the collapse/expand toggle
    doc.querySelectorAll('button').forEach(function(btn) {
      var label = (btn.getAttribute('aria-label') || '').toLowerCase();
      var testid = (btn.getAttribute('data-testid') || '').toLowerCase();
      if (
        label.includes('sidebar') ||
        testid.includes('collapse') ||
        testid.includes('headernopadding') ||
        testid.includes('header-no-padding')
      ) {
        btn.style.setProperty('display', 'none', 'important');
      }
    });

    // 2. Hide known container testids
    ['stSidebarCollapseButton','stSidebarCollapsedControl'].forEach(function(id) {
      doc.querySelectorAll('[data-testid="' + id + '"]').forEach(function(el) {
        el.style.setProperty('display', 'none', 'important');
      });
    });

    // 3. If sidebar collapsed anyway, click the expand button or force aria attr
    var sidebar = doc.querySelector('[data-testid="stSidebar"]');
    if (sidebar && sidebar.getAttribute('aria-expanded') === 'false') {
      var expandBtn = doc.querySelector(
        '[data-testid="stSidebarCollapsedControl"] button, button[aria-label*="xpand"]'
      );
      if (expandBtn) expandBtn.click();
    }
  }

  // Run immediately and on every DOM change
  lockSidebar();
  new MutationObserver(lockSidebar).observe(doc.body, {
    childList: true, subtree: true, attributes: true, attributeFilter: ['aria-expanded']
  });
})();
</script>
""", height=0)

# Hero Section with Sensor Theme 
st.markdown("""
<style>
/* Hero Container */
.iot-hero {
    position: relative;
    margin-bottom: 2rem;
    padding: 2rem 2rem 1.8rem 2rem;
    background: linear-gradient(135deg, #0D2137 0%, #1E3A5F 100%);
    border-radius: 24px;
    overflow: hidden;
    border: 1px solid rgba(59,130,246,0.3);
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    min-height: 200px;
}

/* Animated radar/sensor pulse */
.iot-radar {
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    animation: radarPulse 3s ease-in-out infinite;
    pointer-events: none;
}

@keyframes radarPulse {
    0% { transform: scale(0.8); opacity: 0.3; }
    50% { transform: scale(1.2); opacity: 0.1; }
    100% { transform: scale(0.8); opacity: 0.3; }
}

/* Sensor wave rings */
.iot-wave {
    position: absolute;
    bottom: -30px;
    left: 10%;
    display: flex;
    gap: 8px;
    pointer-events: none;
}

.wave-ring {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    border: 2px solid rgba(59,130,246,0.4);
    animation: waveExpand 2s ease-out infinite;
}

.wave-ring:nth-child(1) { animation-delay: 0s; }
.wave-ring:nth-child(2) { animation-delay: 0.3s; }
.wave-ring:nth-child(3) { animation-delay: 0.6s; }

@keyframes waveExpand {
    0% { transform: scale(0.5); opacity: 0.8; }
    100% { transform: scale(2); opacity: 0; }
}

/* Signal bars */
.iot-signal {
    position: absolute;
    top: 20px;
    right: 30px;
    display: flex;
    gap: 4px;
    align-items: flex-end;
    pointer-events: none;
}

.signal-bar {
    width: 4px;
    background: rgba(59,130,246,0.6);
    border-radius: 2px;
    animation: signalBounce 1s ease-in-out infinite;
}

.signal-bar:nth-child(1) { height: 12px; animation-delay: 0s; }
.signal-bar:nth-child(2) { height: 20px; animation-delay: 0.15s; }
.signal-bar:nth-child(3) { height: 28px; animation-delay: 0.3s; }
.signal-bar:nth-child(4) { height: 36px; animation-delay: 0.45s; }

@keyframes signalBounce {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 1; }
}

/* Data nodes */
.iot-nodes {
    position: absolute;
    top: 30%;
    left: 5%;
    display: flex;
    gap: 60px;
    pointer-events: none;
}

.node {
    position: relative;
    width: 8px;
    height: 8px;
    background: #3B82F6;
    border-radius: 50%;
    animation: nodePulse 2s ease-in-out infinite;
}

.node::before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    border: 1px solid rgba(59,130,246,0.5);
    animation: nodeRing 2s ease-in-out infinite;
}

.node::after {
    content: '';
    position: absolute;
    top: 8px;
    left: 8px;
    width: 60px;
    height: 1px;
    background: linear-gradient(90deg, rgba(59,130,246,0.6), transparent);
}

.node:nth-child(1) { animation-delay: 0s; }
.node:nth-child(2) { animation-delay: 0.5s; }
.node:nth-child(3) { animation-delay: 1s; }

@keyframes nodePulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.5); opacity: 0.5; }
}

@keyframes nodeRing {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.8); opacity: 0; }
}

/* Floating particles */
.iot-particles {
    position: absolute;
    bottom: 20%;
    right: 15%;
    pointer-events: none;
}

.particle {
    position: absolute;
    width: 3px;
    height: 3px;
    background: #22C55E;
    border-radius: 50%;
    animation: floatUp 4s linear infinite;
}

.particle:nth-child(1) { left: 0; animation-delay: 0s; animation-duration: 3s; }
.particle:nth-child(2) { left: 20px; animation-delay: 1s; animation-duration: 4s; }
.particle:nth-child(3) { left: 40px; animation-delay: 2s; animation-duration: 3.5s; }
.particle:nth-child(4) { left: 60px; animation-delay: 0.5s; animation-duration: 4.5s; }
.particle:nth-child(5) { left: 80px; animation-delay: 1.5s; animation-duration: 3.8s; }

@keyframes floatUp {
    0% { transform: translateY(0) scale(1); opacity: 1; }
    100% { transform: translateY(-100px) scale(0); opacity: 0; }
}

/* Sensor icon */
.sensor-icon {
    position: absolute;
    bottom: 15px;
    left: 20px;
    font-size: 32px;
    opacity: 0.3;
    filter: blur(2px);
    animation: sensorGlow 2s ease-in-out infinite;
    pointer-events: none;
}

@keyframes sensorGlow {
    0%, 100% { opacity: 0.2; filter: blur(1px); }
    50% { opacity: 0.5; filter: blur(3px); }
}

/* Hero Content - Make sure it's on top */
.hero-content {
    position: relative;
    z-index: 10;
}

.hero-title {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.02em !important;
}

.hero-subtitle {
    color: #94A3B8 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
}

.hero-badge {
    display: inline-block;
    background: rgba(59,130,246,0.2);
    border: 1px solid rgba(59,130,246,0.4);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.7rem;
    color: #60A5FA !important;
    margin-bottom: 1rem;
}

/* Status indicator */
.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-top: 1rem;
}

.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #22C55E;
    animation: blinkStatus 1.5s ease-in-out infinite;
    box-shadow: 0 0 8px #22C55E;
}

.status-text {
    font-size: 0.7rem;
    color: #94A3B8 !important;
    letter-spacing: 0.1em;
}

@keyframes blinkStatus {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
</style>

<div class="iot-hero">
    <div class="iot-radar"></div>
    <div class="iot-wave"><div class="wave-ring"></div><div class="wave-ring"></div><div class="wave-ring"></div></div>
    <div class="iot-signal"><div class="signal-bar"></div><div class="signal-bar"></div><div class="signal-bar"></div><div class="signal-bar"></div></div>
    <div class="iot-nodes"><div class="node"></div><div class="node"></div><div class="node"></div></div>
    <div class="iot-particles"><div class="particle"></div><div class="particle"></div><div class="particle"></div><div class="particle"></div><div class="particle"></div></div>
    <div class="sensor-icon">📡</div>
    <div class="hero-content">
        <div class="hero-badge"> LIVE TELEMETRY STREAM</div>
        <h1 class="hero-title">IoT Sensor Network<br>Anomaly Detection</h1>
        <div class="hero-subtitle">Real-time monitoring | Multi-variate analysis | Predictive alerts</div>
        <div class="status-indicator"><div class="status-dot"></div><div class="status-text">Sensors Active | Data Streaming</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# CSS 
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Ubuntu:wght@300;400;500;700;800&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root {
  --sidebar-bg-start: #0D2137;
  --sidebar-bg-end: #1E3A5F;
  --text-primary: #F1F5F9;
  --text-secondary: #CBD5E1;
  --text-muted: #64748B;
  --accent-blue: #3B82F6;
  --accent-green: #22C55E;
  --accent-red: #EF4444;
  --accent-orange: #F97316;
  --border-light: #DDE3EE;
  --bg-page: #EEF2F7;
  --card-bg: #ffffff;
  --tint: #F8FAFC;  /* ADD THIS LINE */
}

/* Main app */
.stApp { background-color: var(--bg-page); }

/* Sidebar — always visible, no collapse button */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--sidebar-bg-start) 0%, var(--sidebar-bg-end) 100%);
  border-right: none;
}

/* Hide the collapse/expand toggle — covers all known Streamlit testid variants */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stBaseButton-headerNoPadding"],
button[aria-label="Collapse sidebar"],
button[aria-label="Expand sidebar"],
button[kind="header"],
[data-testid="stSidebar"] > div > button:first-of-type,
[data-testid="stSidebarContent"] > div:first-child > button {
  display: none !important;
  visibility: hidden !important;
  pointer-events: none !important;
}

[data-testid="stSidebar"] *:not([data-testid="stIconMaterial"]):not([data-testid="stIconEmoji"]):not([class*="material-icons"]):not([class*="material-symbols"]) {
  color: var(--text-secondary) !important;
}

[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown h3 {
  color: var(--text-primary) !important;
  font-size: 0.95rem !important;
  font-weight: 500 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: rgba(255,255,255,0.07) !important;
  border-color: rgba(255,255,255,0.14) !important;
}

[data-testid="stSidebar"] .stRadio > div,
[data-testid="stSidebar"] .stCheckbox label {
  color: var(--text-secondary) !important;
}

.main .block-container {
  padding-left: 2rem;
  padding-right: 2rem;
  max-width: 100%;
}

/* Typography */
h1 {
  font-size: 1.85rem !important;
  font-weight: 800 !important;
  color: #0D2137 !important;
  letter-spacing: -0.02em !important;
}

h2, h3 {
  font-weight: 700 !important;
  color: #0D2137 !important;
}

.stCaption, [data-testid="stCaptionContainer"] p {
  color: var(--text-muted) !important;
  font-size: 0.82rem !important;
}

hr {
  border-color: var(--border-light) !important;
  margin: 1.5rem 0 !important;
}

/* KPI cards */
.kpi-card {
  border-radius: 16px;
  padding: 16px 20px 14px;
  border: 1px solid var(--border-light);
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  position: relative;
  overflow: hidden;
  min-height: 118px;
  background: linear-gradient(135deg, var(--tint, #F8FAFC) 0%, var(--card-bg) 62%);
}

.kpi-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 4px;
  background: var(--accent, var(--accent-blue));
  border-radius: 16px 16px 0 0;
}

.kpi-glow {
  position: absolute;
  top: -24px; right: -24px;
  width: 88px; height: 88px;
  border-radius: 50%;
  background: var(--accent, var(--accent-blue));
  opacity: 0.09;
}

.kpi-label {
  font-size: 0.68rem;
  font-weight: 700;
  color: #94A3B8;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 7px;
}

.kpi-icon-box {
  position: absolute;
  top: 14px; right: 14px;
  width: 32px; height: 32px;
  border-radius: 9px;
  background: var(--accent, var(--accent-blue));
  opacity: 0.14;
}

.kpi-icon-svg {
  position: absolute;
  top: 14px; right: 14px;
  width: 32px; height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--accent, var(--accent-blue));
}

.kpi-value {
  font-size: 1.85rem;
  font-weight: 800;
  color: #0D2137;
  letter-spacing: -0.04em;
  line-height: 1.05;
}

.kpi-sub {
  font-size: 0.7rem;
  color: #94A3B8;
  margin-top: 5px;
  font-weight: 500;
}

.kpi-bar-bg {
  height: 4px;
  background: rgba(0,0,0,0.06);
  border-radius: 99px;
  margin-top: 12px;
  overflow: hidden;
}

.kpi-bar-fill {
  height: 100%;
  border-radius: 99px;
  background: var(--accent, var(--accent-blue));
}

/* Status chips */
.chip-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 4px;
}

.chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 11px;
  border-radius: 99px;
  font-size: 0.73rem;
  font-weight: 600;
  letter-spacing: 0.03em;
}

.chip-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  animation: blink 2.2s ease-in-out infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  55% { opacity: 0.25; }
}

.sec-title {
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 10px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border-light);
}

.js-plotly-plot, .plotly {
  border-radius: 16px;
  border: 1px solid var(--border-light);
  overflow: hidden;
  background: var(--card-bg);
  box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--border-light);
  border-radius: 14px;
  overflow: hidden;
  background: var(--card-bg);
}

[data-testid="stAlert"] {
  border-radius: 10px;
}

/* Material Icons protection — exclude sidebar toggle spans */
[data-testid="stIconMaterial"],
[data-testid="stIconEmoji"],
.material-icons,
.material-icons-outlined,
.material-icons-round,
.material-icons-sharp,
.material-icons-two-tone,
.material-symbols-outlined,
.material-symbols-rounded,
.material-symbols-sharp,
[class*="material-icons"],
[class*="material-symbols"] {
  font-family: 'Material Symbols Rounded', 'Material Icons' !important;
  font-style: normal !important;
  font-weight: normal !important;
  line-height: 1 !important;
}

/* Sidebar collapse/expand button — force visibility + clickability */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {
  display: block !important;
  visibility: visible !important;
  opacity: 1 !important;
  position: relative !important;
  z-index: 99999 !important;
  pointer-events: auto !important;
}
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stBaseButton-headerNoPadding"] {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  pointer-events: auto !important;
  cursor: pointer !important;
  z-index: 99999 !important;
  background: rgba(255,255,255,0.15) !important;
  border-radius: 8px !important;
  border: 1px solid rgba(255,255,255,0.25) !important;
  color: #ffffff !important;
}
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="stBaseButton-headerNoPadding"]:hover {
  background: rgba(255,255,255,0.28) !important;
  cursor: pointer !important;
}
[data-testid="stSidebarCollapseButton"] button svg,
[data-testid="stSidebarCollapsedControl"] button svg,
[data-testid="stBaseButton-headerNoPadding"] svg {
  fill: #ffffff !important;
  stroke: #ffffff !important;
  display: block !important;
  visibility: visible !important;
  pointer-events: none !important;
}
[data-testid="stSidebarCollapseButton"] button span,
[data-testid="stSidebarCollapsedControl"] button span,
[data-testid="stBaseButton-headerNoPadding"] span {
  color: #ffffff !important;
  display: block !important;
  visibility: visible !important;
}
</style>
""", unsafe_allow_html=True)

# ── Colour palette 

C_SCORE_LINE  = "#3B82F6"
C_POSITIVE_DK = "#15803D"
C_IF          = "#3B82F6"
C_ZS          = "#22C55E"

C_SEV = {
    "high":   "#EF4444",
    "medium": "#F97316",
    "low":    "#EAB308",
    "normal": "#CBD5E1",
}

# Light tint per accent colour (background gradient start)
_TINT = {
    "#3B82F6": "#EFF6FF",
    "#EF4444": "#FEF2F2",
    "#10B981": "#ECFDF5",
    "#14B8A6": "#F0FDFA",
    "#F97316": "#FFF7ED",
    "#22C55E": "#F0FDF4",
    "#EAB308": "#FEFCE8",
}

_LAYOUT_BASE = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Ubuntu, sans-serif", color="#0F172A", size=12),
    xaxis=dict(
        gridcolor="#F1F5F9", linecolor="#E2E8F0",
        tickfont=dict(family="Ubuntu, sans-serif", color="#64748B", size=11),
        title_font=dict(family="Ubuntu, sans-serif", color="#64748B", size=12),
    ),
    yaxis=dict(
        gridcolor="#F1F5F9", linecolor="#E2E8F0",
        tickfont=dict(family="Ubuntu, sans-serif", color="#64748B", size=11),
        title_font=dict(family="Ubuntu, sans-serif", color="#64748B", size=12),
    ),
    legend=dict(
        orientation="h",        # horizontal layout
        yanchor="bottom",
        y=1.02,                 
        xanchor="left",
        x=0,
        font=dict(family="Ubuntu, sans-serif", size=11)
    ),
    margin=dict(l=48, r=24, t=80, b=44),
)


def _apply_base(fig: go.Figure, height: int = 380, title: str = "") -> go.Figure:
    layout = dict(**_LAYOUT_BASE, height=height)
    if title:
        layout["title"] = dict(
            text=title,
            font=dict(family="Ubuntu, sans-serif", size=14, color="#0D2137", weight="bold"),
            x=0, xanchor="left", pad=dict(l=0),
        )
    fig.update_layout(**layout)
    return fig


# DB helpers 

@st.cache_resource
def _engine():
    return get_engine()


@st.cache_data(ttl=60)
def fetch_channels() -> list[str]:
    session = get_session(_engine())
    try:
        rows = (
            session.query(AnomalyRecord.channel)
            .distinct().order_by(AnomalyRecord.channel).all()
        )
        return [r[0] for r in rows]
    finally:
        session.close()


@st.cache_data(ttl=300)
def fetch_records(channel: str, detector: str | None) -> pd.DataFrame:
    session = get_session(_engine())
    try:
        q = session.query(
            AnomalyRecord.timestamp,
            AnomalyRecord.detector,
            AnomalyRecord.anomaly_score,
            AnomalyRecord.is_anomaly,
            AnomalyRecord.severity,
            AnomalyRecord.ground_truth,
        ).filter(AnomalyRecord.channel == channel)
        if detector:
            q = q.filter(AnomalyRecord.detector == detector)
        q = q.order_by(AnomalyRecord.timestamp)
        df = pd.DataFrame(
            q.all(),
            columns=["timestamp", "detector", "anomaly_score",
                     "is_anomaly", "severity", "ground_truth"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    finally:
        session.close()

# Sidebar
channels = fetch_channels()
if not channels:
    st.warning("No data yet. Run `python main.py` (or `docker-compose up pipeline`) first.")
    st.stop()

with st.sidebar:
    st.markdown("### Controls")
    channel = st.selectbox("Channel", channels)
    detector_choice = st.radio(
        "Detector",
        options=["Isolation Forest", "Z-Score", "LSTM Autoencoder", "All"],
        index=3,
        help=(
            "**Isolation Forest** — multivariate, tree-based.\n\n"
            "**Z-Score** — univariate statistical baseline.\n\n"
            "**LSTM Autoencoder** — sequence-aware deep learning.\n\n"
            "**All** — side-by-side comparison of all three."
        ),
    )
    severity_filter = st.multiselect(
        "Severity", options=["low", "medium", "high"],
        default=["low", "medium", "high"],
    )
    show_ground_truth    = st.checkbox("Overlay ground truth", value=True)
    show_trend           = st.checkbox("Show trend line", value=True)
    anomalies_only_table = st.checkbox("Table: anomalies only", value=True)

# Data 

detector_map = {
    "Isolation Forest":  "isolation_forest",
    "Z-Score":           "zscore",
    "LSTM Autoencoder":  "lstm_autoencoder",
    "All":               None,
}
selected_detector = detector_map[detector_choice]

df_all = fetch_records(channel, selected_detector)
if df_all.empty:
    st.warning(f"No records for channel **{channel}**.")
    st.stop()

df_if   = df_all[df_all["detector"] == "isolation_forest"].copy()
df_zs   = df_all[df_all["detector"] == "zscore"].copy()
df_lstm = df_all[df_all["detector"] == "lstm_autoencoder"].copy()


# Lucide SVG icons

_I_DB = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"'
    ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<ellipse cx="12" cy="5" rx="9" ry="3"/>'
    '<path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/>'
    '<path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/>'
    '</svg>'
)
_I_ALERT = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"'
    ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>'
    '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'
    '</svg>'
)
_I_ACTIVITY = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"'
    ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'
    '</svg>'
)
_I_CROSSHAIR = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"'
    ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="10"/>'
    '<line x1="22" y1="12" x2="18" y2="12"/><line x1="6" y1="12" x2="2" y2="12"/>'
    '<line x1="12" y1="6" x2="12" y2="2"/><line x1="12" y1="22" x2="12" y2="18"/>'
    '</svg>'
)
_I_SEARCH = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"'
    ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>'
    '</svg>'
)
_I_GAUGE = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"'
    ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 2a10 10 0 0 1 10 10"/>'
    '<path d="M12 2a10 10 0 0 0-10 10"/>'
    '<path d="M12 12l3-3"/>'
    '<circle cx="12" cy="12" r="2"/>'
    '</svg>'
)


# KPI helpers

@st.cache_data(show_spinner=False)
def _compute_kpis(df: pd.DataFrame) -> dict:
    total     = len(df)
    anomalies = int(df["is_anomaly"].sum())
    rate      = anomalies / total if total else 0.0
    precision = recall = None
    if df["ground_truth"].notna().any():
        gt   = df["ground_truth"].fillna(False).astype(bool)
        pred = df["is_anomaly"].astype(bool)
        tp   = int(( pred &  gt).sum())
        fp   = int(( pred & ~gt).sum())
        fn   = int((~pred &  gt).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return dict(total=total, anomalies=anomalies, rate=rate,
                precision=precision, recall=recall)


def _card(label: str, value: str, sub: str, accent: str,
          icon: str, bar_pct: float | None = None) -> str:
    tint = _TINT.get(accent, "#F8FAFC")
    bar = ""
    if bar_pct is not None:
        pct = min(100.0, bar_pct * 100)
        bar = (
            '<div class="kpi-bar-bg">'
            '<div class="kpi-bar-fill" style="width:' + f"{pct:.1f}" + '%"></div>'
            '</div>'
        )
    return (
        '<div class="kpi-card" style="--accent:' + accent + ';--tint:' + tint + ';">'
        '<div class="kpi-glow"></div>'
        '<div class="kpi-icon-box"></div>'
        '<div class="kpi-icon-svg">' + icon + '</div>'
        '<div class="kpi-label">' + label + '</div>'
        '<div class="kpi-value">' + value + '</div>'
        '<div class="kpi-sub">' + sub + '</div>'
        + bar + '</div>'
    )

def _kpi_row(label: str, df: pd.DataFrame) -> None:
    kpi = _compute_kpis(df)
    rate = kpi["rate"]
    a_col = "#EF4444" if rate > 0.15 else ("#F97316" if rate > 0.05 else "#22C55E")
    tint = _TINT.get(a_col, "#F8FAFC")

    st.markdown(f'<div class="sec-title">{label}</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.markdown(_card("Total Records", f"{kpi['total']:,}", "timesteps stored", "#3B82F6", _I_DB), unsafe_allow_html=True)
    with c2:
        st.markdown(_card("Flagged", f"{kpi['anomalies']:,}", "anomaly events", "#EF4444", _I_ALERT), unsafe_allow_html=True)
    with c3:
        st.markdown(_card("Anomaly Rate", f"{rate*100:.2f}%", "of test window", a_col, _I_ACTIVITY, bar_pct=rate), unsafe_allow_html=True)
    with c4:
        pv = f"{kpi['precision']:.3f}" if kpi["precision"] is not None else "—"
        st.markdown(_card("Precision", pv, "TP / (TP+FP)", "#10B981", _I_CROSSHAIR), unsafe_allow_html=True)
    with c5:
        rv = f"{kpi['recall']:.3f}" if kpi["recall"] is not None else "—"
        st.markdown(_card("Recall", rv, "TP / (TP+FN)", "#14B8A6", _I_SEARCH), unsafe_allow_html=True)
    with c6:
        # SVG arc gauge
        pct = min(rate * 100, 30) / 30
        angle = pct * 180
        r, cx, cy = 38, 50, 54
        rad = math.radians(180 - angle)
        ex = cx + r * math.cos(rad)
        ey = cy - r * math.sin(rad)
        large = 1 if angle > 180 else 0
        arc_d = f"M {cx-r},{cy} A {r},{r} 0 {large},1 {ex:.2f},{ey:.2f}"
        st.markdown(
            f'<div class="kpi-card" style="--accent:{a_col};--tint:{tint};min-height:118px;">'
            '<div class="kpi-glow"></div>'
            '<div class="kpi-label">Rate Gauge</div>'
            f'<svg viewBox="0 0 100 60" width="100%" style="margin-top:6px;">'
            f'<path d="M {cx-r},{cy} A {r},{r} 0 1,1 {cx+r},{cy}" fill="none" stroke="#E2E8F0" stroke-width="7" stroke-linecap="round"/>'
            f'<path d="{arc_d}" fill="none" stroke="{a_col}" stroke-width="7" stroke-linecap="round"/>'
            f'<text x="{cx}" y="{cy+2}" text-anchor="middle" font-size="13" font-weight="700" fill="#0D2137" font-family="Ubuntu,sans-serif">{rate*100:.1f}%</text>'
            f'</svg></div>',
            unsafe_allow_html=True,
        )

# Status strip 

total_recs   = len(df_all)
total_anom   = int(df_all["is_anomaly"].sum())
overall_rate = total_anom / total_recs if total_recs else 0.0

if overall_rate > 0.15:
    h_label, h_bg, h_fg, h_dot = "CRITICAL", "#FEE2E2", "#991B1B", "#EF4444"
elif overall_rate > 0.05:
    h_label, h_bg, h_fg, h_dot = "WARNING",  "#FEF9C3", "#92400E", "#EAB308"
else:
    h_label, h_bg, h_fg, h_dot = "HEALTHY",  "#DCFCE7", "#166534", "#22C55E"

det_badges = ""
if detector_choice in ("Isolation Forest", "All"):
    det_badges += (
        '<span class="chip" style="background:#EFF6FF;color:#1D4ED8;">'
        '<span class="chip-dot" style="background:#3B82F6;"></span>Isolation Forest</span>'
    )
if detector_choice in ("Z-Score", "All"):
    det_badges += (
        '<span class="chip" style="background:#F0FDF4;color:#15803D;">'
        '<span class="chip-dot" style="background:#22C55E;"></span>Z-Score</span>'
    )
if detector_choice in ("LSTM Autoencoder", "All"):
    det_badges += (
        '<span class="chip" style="background:#FDF4FF;color:#7E22CE;">'
        '<span class="chip-dot" style="background:#A855F7;"></span>LSTM</span>'
    )

st.markdown(
    '<div class="chip-row">'
    '<span class="chip" style="background:' + h_bg + ';color:' + h_fg + ';">'
    '<span class="chip-dot" style="background:' + h_dot + ';"></span>' + h_label + '</span>'
    '<span class="chip" style="background:#F8FAFC;color:#475569;">'
    '<span class="chip-dot" style="background:#94A3B8;animation:none;"></span>'
    'CH: ' + channel + '</span>'
    + det_badges +
    '<span class="chip" style="background:#F8FAFC;color:#64748B;">'
    + f"{total_recs:,}" + ' records</span>'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()

# KPI section 

if detector_choice == "Isolation Forest":
    _kpi_row("Isolation Forest", df_if)
elif detector_choice == "Z-Score":
    _kpi_row("Z-Score", df_zs)
elif detector_choice == "LSTM Autoencoder":
    _kpi_row("LSTM Autoencoder", df_lstm)
else:  # All
    _kpi_row("Isolation Forest", df_if)
    st.markdown("<br>", unsafe_allow_html=True)
    _kpi_row("Z-Score", df_zs)
    st.markdown("<br>", unsafe_allow_html=True)
    _kpi_row("LSTM Autoencoder", df_lstm)

st.divider()


# Time-series helpers 

@st.cache_data(show_spinner=False)
def _gt_regions(df: pd.DataFrame) -> list:
    """Return list of (x0, x1) timestamp pairs for contiguous GT=True runs."""
    if not df["ground_truth"].notna().any():
        return []
    gt     = df["ground_truth"].fillna(False).astype(bool).to_numpy()
    ts     = df["timestamp"].reset_index(drop=True)
    padded = np.concatenate(([False], gt, [False])).astype(np.int8)
    diff   = np.diff(padded)
    starts = np.where(diff ==  1)[0]
    ends   = np.where(diff == -1)[0] - 1
    return [(ts.iloc[s], ts.iloc[e]) for s, e in zip(starts, ends)]


def _add_gt_vrects(fig: go.Figure, df: pd.DataFrame, show_gt: bool) -> None:
    if not show_gt:
        return
    for x0, x1 in _gt_regions(df):
        fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(13,33,55,0.06)",
                      line_width=0, layer="below")


def _trend_traces(df: pd.DataFrame) -> list:
    if df.empty or len(df) < 5:
        return []
    roll = df["anomaly_score"].rolling(20, min_periods=1).mean().values
    ts   = df["timestamp"].values
    diff = pd.Series(roll).diff().fillna(0).values

    traces, seg_x, seg_y = [], [ts[0]], [roll[0]]
    seg_imp = bool(diff[0] <= 0)

    for i in range(1, len(roll)):
        improving = bool(diff[i] <= 0)
        if improving != seg_imp:
            colour = C_POSITIVE_DK if seg_imp else "#0D2137"
            seg_x.append(ts[i]); seg_y.append(roll[i])
            traces.append(go.Scatter(
                x=seg_x, y=seg_y, mode="lines",
                line=dict(color=colour, width=2, dash="dot"),
                showlegend=(len(traces) < 2),
                name="Trend (improving)" if seg_imp else "Trend (worsening)",
                legendgroup="trend",
            ))
            seg_x, seg_y, seg_imp = [ts[i]], [roll[i]], improving
        else:
            seg_x.append(ts[i]); seg_y.append(roll[i])

    if seg_x:
        colour = C_POSITIVE_DK if seg_imp else "#0D2137"
        traces.append(go.Scatter(
            x=seg_x, y=seg_y, mode="lines",
            line=dict(color=colour, width=2, dash="dot"),
            showlegend=False, legendgroup="trend",
        ))
    return traces


@st.cache_data(show_spinner=False)
def _anomaly_traces(df: pd.DataFrame, prefix: str, sev_filter: tuple) -> list:
    size_map = {"high": 9, "medium": 7, "low": 5}
    traces = []
    for sev in ("high", "medium", "low"):
        if sev not in sev_filter:
            continue
        sub = df[df["is_anomaly"] & (df["severity"] == sev)]
        if sub.empty:
            continue
        traces.append(go.Scatter(
            x=sub["timestamp"], y=sub["anomaly_score"],
            mode="markers",
            marker=dict(color=C_SEV[sev], size=size_map[sev],
                        symbol="circle", line=dict(color="#ffffff", width=1)),
            name=prefix + " · " + sev,
        ))
    return traces


@st.cache_data(show_spinner=False)
def _build_ts(df: pd.DataFrame, title: str, line_colour: str,
              show_trend: bool, show_gt: bool, sev_filter: tuple) -> go.Figure:
    fig = go.Figure()

    if df.empty:
        _apply_base(fig, height=390, title=title)
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=13, color="#94A3B8"))
        return fig

    # Green safe-zone fill
    med = float(df["anomaly_score"].median())
    safe = df[df["anomaly_score"] <= med]
    if not safe.empty:
        fig.add_trace(go.Scatter(
            x=safe["timestamp"], y=safe["anomaly_score"],
            mode="none", fill="tozeroy",
            fillcolor="rgba(34,197,94,0.08)",
            showlegend=False, hoverinfo="skip",
        ))

    # Score line
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["anomaly_score"],
        mode="lines", line=dict(color=line_colour, width=1.6),
        name="Score", opacity=0.85,
    ))

    # Anomaly markers (severity_filter passed explicitly — no captured globals)
    for t in _anomaly_traces(df, title, sev_filter):
        fig.add_trace(t)

    # Trend (coloured segments)
    if show_trend:
        for t in _trend_traces(df):
            fig.add_trace(t)

    # Ground-truth shading — regions pre-computed and cached separately
    _add_gt_vrects(fig, df, show_gt)

    _apply_base(fig, height=390, title=title)
    fig.update_layout(
        xaxis_title="Timestamp", yaxis_title="Anomaly Score",
        margin=dict(l=48, r=24, t=100, b=44),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(family="Ubuntu, sans-serif", size=10),
        ),
    )
    return fig


# Time-series render 

st.subheader("Anomaly Score Time Series")
st.caption(
    "Green fill = healthy zone (below median).  "
    "Markers: red = high, orange = medium, amber = low.  "
    "Dotted trend: green = improving, navy = worsening."
)

_sev_tuple = tuple(severity_filter)   # hashable for cache keys

C_LSTM = "#A855F7"

if detector_choice == "All":
    st.plotly_chart(
        _build_ts(df_if, "Isolation Forest", C_IF,
                  show_trend, show_ground_truth, _sev_tuple),
        width='stretch',
    )
    st.plotly_chart(
        _build_ts(df_zs, "Z-Score", C_ZS,
                  show_trend, show_ground_truth, _sev_tuple),
        width='stretch',
    )
    st.plotly_chart(
        _build_ts(df_lstm, "LSTM Autoencoder", C_LSTM,
                  show_trend, show_ground_truth, _sev_tuple),
        width='stretch',
    )
elif detector_choice == "Isolation Forest":
    st.plotly_chart(
        _build_ts(df_if, "Isolation Forest", C_IF,
                  show_trend, show_ground_truth, _sev_tuple),
        width='stretch',
    )
elif detector_choice == "Z-Score":
    st.plotly_chart(
        _build_ts(df_zs, "Z-Score", C_ZS,
                  show_trend, show_ground_truth, _sev_tuple),
        width='stretch',
    )
else:  # LSTM Autoencoder
    st.plotly_chart(
        _build_ts(df_lstm, "LSTM Autoencoder", C_LSTM,
                  show_trend, show_ground_truth, _sev_tuple),
        width='stretch',
    )

st.divider()


# Detector agreement

if detector_choice == "All" and not df_if.empty and not df_zs.empty and not df_lstm.empty:
    st.subheader("Detector Agreement")
    st.caption(
        "Flagged by **all three** = highest-confidence anomaly.  "
        "Unique flags expose each method's blind spot."
    )
    merged = (
        df_if[["timestamp", "is_anomaly"]].rename(columns={"is_anomaly": "if_flag"})
        .merge(
            df_zs[["timestamp", "is_anomaly"]].rename(columns={"is_anomaly": "zs_flag"}),
            on="timestamp", how="inner",
        )
        .merge(
            df_lstm[["timestamp", "is_anomaly"]].rename(columns={"is_anomaly": "lstm_flag"}),
            on="timestamp", how="inner",
        )
    )
    all3     = int(( merged["if_flag"] &  merged["zs_flag"] &  merged["lstm_flag"]).sum())
    if_zs    = int(( merged["if_flag"] &  merged["zs_flag"] & ~merged["lstm_flag"]).sum())
    if_lstm  = int(( merged["if_flag"] & ~merged["zs_flag"] &  merged["lstm_flag"]).sum())
    zs_lstm  = int((~merged["if_flag"] &  merged["zs_flag"] &  merged["lstm_flag"]).sum())
    if_only  = int(( merged["if_flag"] & ~merged["zs_flag"] & ~merged["lstm_flag"]).sum())
    zs_only  = int((~merged["if_flag"] &  merged["zs_flag"] & ~merged["lstm_flag"]).sum())
    lstm_only= int((~merged["if_flag"] & ~merged["zs_flag"] &  merged["lstm_flag"]).sum())

    fig_ag = go.Figure(go.Bar(
        x=["All 3", "IF+ZS", "IF+LSTM", "ZS+LSTM", "IF only", "ZS only", "LSTM only"],
        y=[all3, if_zs, if_lstm, zs_lstm, if_only, zs_only, lstm_only],
        marker_color=["#0D2137", "#6366F1", C_IF, C_ZS, "#93C5FD", "#86EFAC", C_LSTM],
        marker_line_color="#ffffff", marker_line_width=1.5,
        text=[all3, if_zs, if_lstm, zs_lstm, if_only, zs_only, lstm_only],
        textposition="outside",
        textfont=dict(family="Ubuntu, sans-serif", size=12, color="#0F172A"),
    ))
    _apply_base(fig_ag, height=300)
    fig_ag.update_layout(yaxis_title="Timesteps")
    st.plotly_chart(fig_ag, width='stretch')
    st.divider()

# Benchmark comparison table (point-adjusted F1) 

st.subheader("Benchmark Results")
st.caption(
    "**Point-adjusted F1** is the standard evaluation metric for SMAP/MSL. "
    "If any predicted point falls inside a ground-truth anomaly window, the "
    "entire window counts as detected. Raw metrics are shown for comparison."
)

@st.cache_data(show_spinner=False)
def _benchmark_table(ch: str) -> pd.DataFrame:
    from detection.metrics import full_report
    rows = []
    all_df = fetch_records(ch, None)
    for det_key, det_label in [
        ("isolation_forest",  "Isolation Forest"),
        ("zscore",            "Z-Score"),
        ("lstm_autoencoder",  "LSTM Autoencoder"),
    ]:
        df_det = all_df[all_df["detector"] == det_key]
        if df_det.empty or not df_det["ground_truth"].notna().any():
            continue
        gt   = df_det["ground_truth"].fillna(False).astype(bool).to_numpy()
        pred = df_det["is_anomaly"].astype(bool).to_numpy()
        r    = full_report(pred, gt, name=det_label)
        rows.append({
            "Detector":      det_label,
            "PA-F1 ↑":       f"{r['pa_f1']:.3f}",
            "PA-Precision":  f"{r['pa_precision']:.3f}",
            "PA-Recall":     f"{r['pa_recall']:.3f}",
            "Raw F1":        f"{r['f1']:.3f}",
            "Raw Precision": f"{r['precision']:.3f}",
            "Raw Recall":    f"{r['recall']:.3f}",
        })
    return pd.DataFrame(rows)

bm_df = _benchmark_table(channel)
if not bm_df.empty:
    numeric_cols = ["PA-F1 ↑", "PA-Precision", "PA-Recall", "Raw F1", "Raw Precision", "Raw Recall"]
    bm_display = bm_df.copy()
    bm_display[numeric_cols] = bm_display[numeric_cols].apply(pd.to_numeric, errors="coerce")
    st.dataframe(
        bm_display.style
            .highlight_max(subset=numeric_cols, color="#D1FAE5")
            .format({c: "{:.3f}" for c in numeric_cols}),
        width='stretch',
        hide_index=True,
    )
else:
    st.info("Run the pipeline with ground-truth labels to populate the benchmark table.")
st.divider()


# Score distribution 

@st.cache_data(show_spinner=False)
def _build_distribution(df_if: pd.DataFrame, df_zs: pd.DataFrame,
                        df_lstm: pd.DataFrame, det_choice: str) -> go.Figure:
    fig = go.Figure()
    traces = []
    if det_choice in ("Isolation Forest", "All"):
        traces.append((df_if,   "Isolation Forest", C_IF))
    if det_choice in ("Z-Score", "All"):
        traces.append((df_zs,   "Z-Score",          C_ZS))
    if det_choice in ("LSTM Autoencoder", "All"):
        traces.append((df_lstm, "LSTM Autoencoder",  C_LSTM))
    for df_src, lbl, col in traces:
        if not df_src.empty:
            fig.add_trace(go.Histogram(
                x=df_src["anomaly_score"], nbinsx=60, name=lbl,
                marker_color=col, opacity=0.70,
            ))
    _apply_base(fig, height=280)
    fig.update_layout(barmode="overlay", xaxis_title="Anomaly Score",
                      yaxis_title="Count", legend=dict(orientation="h"))
    return fig


st.subheader("Score Distribution")
st.plotly_chart(
    _build_distribution(df_if, df_zs, df_lstm, detector_choice),
    width='stretch',
)
st.divider()


# Severity breakdown 

@st.cache_data(show_spinner=False)
def _build_severity(df_if: pd.DataFrame, df_zs: pd.DataFrame,
                    df_lstm: pd.DataFrame, det_choice: str) -> go.Figure:
    fig = go.Figure()
    def _bars(df: pd.DataFrame, x_label: str, show_legend: bool) -> None:
        counts = df[df["is_anomaly"]]["severity"].value_counts()
        for sev in ("high", "medium", "low"):
            fig.add_trace(go.Bar(
                name=sev.capitalize(), x=[x_label], y=[counts.get(sev, 0)],
                marker_color=C_SEV[sev], marker_line_color="#ffffff",
                marker_line_width=1, legendgroup=sev, showlegend=show_legend,
            ))
    if det_choice == "All":
        _bars(df_if,   "Isolation Forest", show_legend=True)
        _bars(df_zs,   "Z-Score",          show_legend=False)
        _bars(df_lstm, "LSTM Autoencoder", show_legend=False)
    elif det_choice == "Isolation Forest":
        _bars(df_if, "Isolation Forest", show_legend=True)
    elif det_choice == "Z-Score":
        _bars(df_zs, "Z-Score", show_legend=True)
    else:
        _bars(df_lstm, "LSTM Autoencoder", show_legend=True)
    _apply_base(fig, height=280)
    fig.update_layout(barmode="stack", yaxis_title="Anomaly count")
    return fig


st.subheader("Severity Breakdown")
st.plotly_chart(
    _build_severity(df_if, df_zs, df_lstm, detector_choice),
    width='stretch',
)
st.divider()


# ── Flagged records table ─────────────────────────────────────────────────────

st.subheader("Flagged Records")

table_df = df_all.copy()
if anomalies_only_table:
    table_df = table_df[table_df["is_anomaly"]]
if severity_filter:
    table_df = table_df[table_df["severity"].isin(severity_filter)]
table_df = table_df.sort_values("timestamp", ascending=False).reset_index(drop=True)

st.dataframe(
    table_df.rename(columns={
        "timestamp":     "Timestamp",
        "detector":      "Detector",
        "anomaly_score": "Score",
        "is_anomaly":    "Anomaly",
        "severity":      "Severity",
        "ground_truth":  "Ground Truth",
    }),
    width='stretch',
    height=350,
)
st.caption(f"Showing {len(table_df):,} rows.")

st.divider()


# ── Sensitivity Analysis ──────────────────────────────────────────────────────

st.subheader("Isolation Forest — Contamination Sensitivity")
st.caption(
    "PA-F1 across different `IF_CONTAMINATION` values on channel P-1. "
    "Lower contamination reduces false positives; the point-adjust protocol "
    "rewards segment-level detection, so precision improvements outweigh recall losses."
)

_SENSITIVITY = [
    {"Contamination": 0.01, "PA-F1": 0.942, "PA-Precision": 0.891, "PA-Recall": 1.000, "Raw F1": 0.019},
    {"Contamination": 0.03, "PA-F1": 0.853, "PA-Precision": 0.744, "PA-Recall": 1.000, "Raw F1": 0.046},
    {"Contamination": 0.05, "PA-F1": 0.781, "PA-Precision": 0.641, "PA-Recall": 1.000, "Raw F1": 0.064},
    {"Contamination": 0.07, "PA-F1": 0.728, "PA-Precision": 0.572, "PA-Recall": 1.000, "Raw F1": 0.085},
    {"Contamination": 0.10, "PA-F1": 0.646, "PA-Precision": 0.477, "PA-Recall": 1.000, "Raw F1": 0.093},
    {"Contamination": 0.15, "PA-F1": 0.556, "PA-Precision": 0.385, "PA-Recall": 1.000, "Raw F1": 0.117},
    {"Contamination": 0.20, "PA-F1": 0.485, "PA-Precision": 0.320, "PA-Recall": 1.000, "Raw F1": 0.127},
]
sens_df = pd.DataFrame(_SENSITIVITY)

sens_col1, sens_col2 = st.columns([1, 1])

with sens_col1:
    st.dataframe(
        sens_df.style.highlight_max(subset=["PA-F1"], color="#D1FAE5")
                     .highlight_min(subset=["PA-F1"], color="#FEE2E2"),
        width='stretch',
        hide_index=True,
    )

with sens_col2:
    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=sens_df["Contamination"], y=sens_df["PA-F1"],
        mode="lines+markers",
        name="PA-F1",
        line=dict(color="#3B82F6", width=2),
        marker=dict(size=8),
    ))
    fig_sens.add_trace(go.Scatter(
        x=sens_df["Contamination"], y=sens_df["Raw F1"],
        mode="lines+markers",
        name="Raw F1",
        line=dict(color="#F97316", width=2, dash="dot"),
        marker=dict(size=8),
    ))
    fig_sens.add_vline(
        x=0.01, line_dash="dash", line_color="#10B981",
        annotation_text="selected (0.01)", annotation_position="top right",
        annotation_font_color="#10B981",
    )
    _apply_base(fig_sens, height=300, title="PA-F1 vs Contamination")
    fig_sens.update_layout(
        xaxis_title="IF_CONTAMINATION",
        yaxis_title="F1 Score",
        xaxis=dict(tickformat=".2f"),
    )
    st.plotly_chart(fig_sens, width='stretch')
