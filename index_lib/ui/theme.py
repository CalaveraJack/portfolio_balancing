"""
Page chrome: Streamlit page config, the dark stylesheet, and the Plotly template.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"

APP_TITLE = "Strategy Forge"
APP_VERSION = "v0.3"
APP_SUBTITLE = (
    "Research terminal for strategy construction, funding-aware overlays, "
    "and forward simulation."
)

BG = "#0f1115"
PANEL = "#171a21"
TEXT = "#f2f2f2"
MUTED = "#a0a6b3"
BORDER = "#2b313d"
ACCENT = "#d9822b"

#: Categorical series colors, accent-first and distinguishable on the dark panel.
SERIES_COLORS = (
    "#d9822b",
    "#4c9be8",
    "#59c9a5",
    "#c77dff",
    "#e8c547",
    "#ef6f6c",
    "#7fd1b9",
    "#9aa5b1",
)

PLOTLY_TEMPLATE = "strategy_forge"


def register_plotly_template() -> None:
    """Register and activate the app's Plotly template."""
    template = go.layout.Template(pio.templates["plotly_dark"])

    template.layout.paper_bgcolor = PANEL
    template.layout.plot_bgcolor = PANEL
    template.layout.font = dict(
        family="Inter, Segoe UI, Roboto, Arial, sans-serif",
        color=TEXT,
        size=12,
    )
    template.layout.title = dict(font=dict(color=TEXT, size=15), x=0.0, xanchor="left")
    template.layout.colorway = list(SERIES_COLORS)
    template.layout.xaxis = dict(gridcolor=BORDER, zerolinecolor=BORDER)
    template.layout.yaxis = dict(gridcolor=BORDER, zerolinecolor=BORDER)
    template.layout.legend = dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED))
    template.layout.margin = dict(l=60, r=30, t=50, b=40)
    template.layout.hoverlabel = dict(bgcolor=PANEL, bordercolor=BORDER)

    pio.templates[PLOTLY_TEMPLATE] = template
    pio.templates.default = PLOTLY_TEMPLATE


def configure_page() -> None:
    """Set page config, load the stylesheet, and activate the Plotly template."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=str(ASSETS_DIR / "logo.svg")
        if (ASSETS_DIR / "logo.svg").exists()
        else "📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    stylesheet = ASSETS_DIR / "styles.css"
    if stylesheet.exists():
        st.markdown(
            f"<style>{stylesheet.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )

    register_plotly_template()


def render_header() -> None:
    """App title block."""
    st.markdown(
        f"""
        <div class="app-header">
          <div>
            <div class="app-title">{APP_TITLE}</div>
            <div class="app-subtitle">{APP_SUBTITLE}</div>
          </div>
          <div class="app-badge">{APP_VERSION}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    """Small uppercase accent heading used to open a panel."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    """Muted explanatory copy."""
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)
