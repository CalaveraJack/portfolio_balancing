"""
Page chrome: Streamlit page config, the dark stylesheet, and the Plotly template.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"

APP_TITLE = "Strategy Forge"
APP_VERSION = "v0.4"
APP_SUBTITLE = (
    "Research terminal for strategy construction, funding-aware overlays, "
    "and forward simulation."
)
APP_CREDIT = "UI/UX co-authored with Claude (Anthropic)."

# Warm black and cream, gold as the house colour. Keep in step with the CSS
# tokens in assets/styles.css and the base theme in .streamlit/config.toml.
BG = "#0c0a07"
PANEL = "#15120d"
TEXT = "#efe5cf"
MUTED = "#a39881"
BORDER = "#3a3020"
GRID = "#2a2418"
GOLD = "#e3a33b"

DISPLAY_FONT = "Rajdhani, Segoe UI, sans-serif"

#: Categorical series colours, warm-led, in fixed order (never cycled past the
#: seventh: the eighth is a neutral for "the rest"). Validated for the dark
#: panel: lightness band, chroma floor, contrast, and colour-blind separation
#: of neighbours (worst pair verdigris/terracotta at the 8.0 target, so charts
#: using both keep a legend).
SERIES_COLORS = (
    "#C68206",  # gold
    "#418AD1",  # steel blue
    "#C9593D",  # terracotta
    "#3CA17B",  # verdigris
    "#926ABE",  # violet
    "#9D9631",  # brass olive
    "#C1628A",  # rose
    "#8A8170",  # warm grey, for overflow
)

#: Polarity: gold for rising, blue for falling, warm grey at zero.
UP_COLOR = SERIES_COLORS[0]
DOWN_COLOR = SERIES_COLORS[1]
NEUTRAL = "#5c5242"

PLOTLY_TEMPLATE = "strategy_forge"


def register_plotly_template() -> None:
    """Register and activate the app's Plotly template."""
    template = go.layout.Template(pio.templates["plotly_dark"])

    template.layout.paper_bgcolor = PANEL
    template.layout.plot_bgcolor = PANEL
    template.layout.font = dict(
        family="Source Sans Pro, Segoe UI, Roboto, Arial, sans-serif",
        color=TEXT,
        size=12,
    )
    template.layout.title = dict(
        font=dict(family=DISPLAY_FONT, color=TEXT, size=18),
        x=0.0,
        xanchor="left",
    )
    template.layout.colorway = list(SERIES_COLORS[:7])
    template.layout.xaxis = dict(gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER)
    template.layout.yaxis = dict(gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER)
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
            <div class="app-credit">{APP_CREDIT}</div>
          </div>
          <div class="app-badge">{APP_VERSION}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def screen(name: str) -> None:
    """
    Mark which main screen this is, so the stylesheet gives it its accent.

    The marker is hidden; the tab panel holding it picks the accent up.
    """
    st.markdown(
        f'<div class="screen-mark screen-{name}"></div>', unsafe_allow_html=True
    )


def section(title: str, help: Optional[str] = None) -> None:
    """
    Small uppercase accent heading used to open a panel.

    ``help`` adds an info mark whose tooltip explains the panel, for the
    things worth knowing that do not belong on screen all the time.
    """
    mark = ""
    if help:
        text = html.escape(help, quote=True)
        mark = f' <span class="section-help" title="{text}">ⓘ</span>'
    st.markdown(
        f'<div class="section-header">{title}{mark}</div>', unsafe_allow_html=True
    )


def note(text: str) -> None:
    """
    Muted explanatory copy.

    Streamlit does not parse Markdown inside a raw HTML block, so emphasis and
    inline code are converted here; otherwise the markers show up verbatim.
    """
    html = re.sub(r"\*\*(.+?)\*\*", lambda m: f"<strong>{m.group(1)}</strong>", text)
    html = re.sub(r"`(.+?)`", lambda m: f"<code>{m.group(1)}</code>", html)
    st.markdown(f'<div class="note">{html}</div>', unsafe_allow_html=True)
