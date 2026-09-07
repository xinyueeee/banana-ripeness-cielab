"""
theme.py -- purely cosmetic "colourful banana" theming for the Streamlit UI.

No image-processing or model logic lives here: CSS injection + small HTML snippets
only. Safe to delete without affecting the pipeline, the classifier, or any result.
"""
from __future__ import annotations

import base64
import functools
from pathlib import Path

from config import CLASS_COLOUR

CLASS_EMOJI = {"unripe": "🟢", "ripe": "🟡", "overripe": "🟠", "rotten": "🟤"}

# Decorative mascot (kept out of the public deployment -- see DEPLOY notes in README).
_MASCOT = Path(__file__).resolve().parent / "assets" / "mascot" / "minions_group.png"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@600;700;800&family=Nunito:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 6% 8%, rgba(255,199,44,0.14) 0, transparent 42%),
        radial-gradient(circle at 96% 12%, rgba(124,179,66,0.12) 0, transparent 40%),
        radial-gradient(circle at 50% 100%, rgba(255,160,0,0.08) 0, transparent 55%),
        #FFFDF5;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFF3C4 0%, #FFE9A8 100%);
    border-right: 3px solid #F2A900;
}

/* tighten Streamlit's default block spacing so content fits without scrolling */
[data-testid="block-container"] { padding-top: 1.1rem; padding-bottom: 1.2rem; }
[data-testid="stVerticalBlock"] { gap: 0.85rem; }
small, [data-testid="stCaptionContainer"] {
    line-height: 1.7 !important;
    display: block;
    margin: 4px 0 !important;
}
[data-testid="stAlert"] { margin: 10px 0 !important; padding: 14px 16px !important; }
[data-testid="stAlert"] p { line-height: 1.6 !important; }

/* ---- hero banner ----------------------------------------------------- */
.banana-hero {
    background: linear-gradient(120deg, #FFE066 0%, #FFC72C 55%, #F2A900 100%);
    border-radius: 16px;
    padding: 14px 22px;
    margin-bottom: 6px;
    box-shadow: 0 4px 12px rgba(178, 122, 0, 0.22);
    display: flex; align-items: center; justify-content: space-between; gap: 14px;
}
.banana-hero h1 {
    font-family: 'Baloo 2', sans-serif; font-weight: 800; font-size: 24px;
    color: #4A3220; margin: 0; letter-spacing: 0.2px; line-height: 1.2;
}
.banana-hero p { color: #5D3E22; font-size: 13px; margin: 3px 0 0 0; font-weight: 600; }
.banana-hero .hero-mascot-wrap { flex-shrink: 0; text-align: center; }
.banana-hero .hero-mascot { height: 90px; width: auto;
    filter: drop-shadow(0 3px 6px rgba(0,0,0,0.2)); }
.banana-hero .hero-mascot-cap { margin: 2px 0 0 0; font-size: 11px; font-style: italic;
    color: #6B4A26; font-weight: 600; }

/* ---- section headers --------------------------------------------------- */
h2, h3, .stSubheader, [data-testid="stMarkdownContainer"] h3 {
    font-family: 'Baloo 2', sans-serif; color: #4A3220 !important;
}

/* ---- buttons ------------------------------------------------------------ */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-family: 'Baloo 2', sans-serif;
    border: none !important;
    padding: 0.55rem 1.1rem !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(120deg, #FFD93D, #F2A900) !important;
    color: #4A3220 !important;
    box-shadow: 0 4px 10px rgba(178,122,0,0.3);
}
.stButton > button[kind="secondary"] {
    background: #FFF3C4 !important;
    color: #4A3220 !important;
    border: 1.5px solid #F2A900 !important;
}
.stButton > button:hover { transform: translateY(-1px); }

/* ---- file uploader -------------------------------------------------- */
[data-testid="stFileUploaderDropzone"] {
    background: #FFF9E6 !important;
    border: 2px dashed #F2A900 !important;
    border-radius: 14px !important;
}

/* uploaded / result photo -- capped so it never pushes content off-screen */
[data-testid="stImage"] img {
    max-height: 260px !important;
    width: auto !important;
    object-fit: contain;
    border-radius: 12px;
    display: block;
    margin: 0 auto;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
}

/* ---- tabs ------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px; background: #FFF3C4; padding: 8px; border-radius: 12px;
    margin-bottom: 18px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; font-weight: 700; color: #7A5230;
    font-family: 'Baloo 2', sans-serif;
    padding: 10px 20px !important;
    height: auto !important;
}
.stTabs [data-baseweb="tab"] p { font-size: 15px !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(120deg, #FFD93D, #F2A900) !important;
    color: #4A3220 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 8px; }

/* ---- containers / cards ------------------------------------------------ */
[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-radius: 16px !important;
}
div[data-testid="stExpander"] {
    border: 1.5px solid #F2A900; border-radius: 12px; overflow: hidden;
}

/* ---- tables / dataframes ------------------------------------------------ */
[data-testid="stTable"] table thead tr th, [data-testid="stDataFrame"] div[role="columnheader"] {
    background-color: #FFE066 !important; color: #4A3220 !important; font-weight: 800 !important;
}

/* ---- divider ------------------------------------------------------------ */
hr { border: none; height: 3px; border-radius: 3px; margin: 0.6rem 0;
     background: linear-gradient(90deg, #7CB342, #FFD93D, #E8B923, #6D4C41); }

/* ---- class legend chips -------------------------------------------------- */
.chip-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 2px 0; }
.chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 11px; border-radius: 999px; font-weight: 700; font-size: 12.5px;
    color: #fff; box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}

/* ---- result badge -- compact, evenly spaced, not oversized ------------- */
.result-badge { text-align: center; margin: 4px 0; }
.result-badge .badge-inner {
    display: inline-flex; align-items: center; gap: 10px;
    padding: 10px 26px; border-radius: 12px;
    font-family: 'Baloo 2', sans-serif; font-size: 22px; font-weight: 700;
    letter-spacing: 1px; line-height: 1; color: #fff;
}
.result-badge .badge-inner .dot { font-size: 16px; line-height: 1; }

.footer-note { text-align: center; color: #8A6A46; font-size: 12.5px; margin-top: 6px; }
</style>
"""


@functools.lru_cache(maxsize=2)
def _b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def inject_css() -> str:
    return _CSS


def hero_html(title: str, subtitle: str) -> str:
    mascot = ""
    if _MASCOT.is_file():
        mascot = (f'<div class="hero-mascot-wrap">'
                  f'<img class="hero-mascot" src="data:image/png;base64,{_b64(str(_MASCOT))}">'
                  f'<p class="hero-mascot-cap">Even minions agree &ndash; bananas deserve a '
                  f'proper analysis.</p></div>')
    return (f'<div class="banana-hero">'
            f'<div><h1>🍌 {title}</h1><p>{subtitle}</p></div>{mascot}</div>')


def class_legend_html() -> str:
    chips = "".join(
        f'<span class="chip" style="background:{CLASS_COLOUR[c]}">'
        f'{CLASS_EMOJI[c]} {c.upper()}</span>'
        for c in ["unripe", "ripe", "overripe", "rotten"]
    )
    return f'<div class="chip-row">{chips}</div>'


def footer_html() -> str:
    return ('<div class="footer-note">🍌 CIELAB colour-space prototype &middot; '
            'StandardScaler + SVC &middot; built for BMDS2133 Image Processing 🍌</div>')
