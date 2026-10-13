"""
Ma'man — the look of the app (dark navy, AI mint/teal, glass panels, Sadu-AI pattern), taken from the team's design.
Only presentation lives here: colours, fonts, icons, logo. Every number on screen still comes from the risk engine.
"""
import base64
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def _data_uri(name, mime):
    return f"data:{mime};base64," + base64.b64encode((ASSETS / name).read_bytes()).decode()


LOGO = _data_uri("logo.png", "image/png")
SADU = _data_uri("sadu-pattern.svg", "image/svg+xml")

# Lucide icons (ISC licence), inner SVG only
ICONS = {
 "shield-check": "<path d=\"M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z\" /> <path d=\"m9 12 2 2 4-4\" />",
 "shield-alert": "<path d=\"M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z\" /> <path d=\"M12 8v4\" /> <path d=\"M12 16h.01\" />",
 "file-search": "<path d=\"M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z\" /> <path d=\"M14 2v5a1 1 0 0 0 1 1h5\" /> <circle cx=\"11.5\" cy=\"14.5\" r=\"2.5\" /> <path d=\"M13.3 16.3 15 18\" />",
 "lock-keyhole": "<circle cx=\"12\" cy=\"16\" r=\"1\" /> <rect x=\"3\" y=\"10\" width=\"18\" height=\"12\" rx=\"2\" /> <path d=\"M7 10V7a5 5 0 0 1 10 0v3\" />",
 "cloud-upload": "<path d=\"M12 13v8\" /> <path d=\"M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242\" /> <path d=\"m8 17 4-4 4 4\" />",
 "lock": "<rect width=\"18\" height=\"11\" x=\"3\" y=\"11\" rx=\"2\" ry=\"2\" /> <path d=\"M7 11V7a5 5 0 0 1 10 0v4\" />",
 "sparkles": "<path d=\"M11.017 2.814a1 1 0 0 1 1.966 0l1.051 5.558a2 2 0 0 0 1.594 1.594l5.558 1.051a1 1 0 0 1 0 1.966l-5.558 1.051a2 2 0 0 0-1.594 1.594l-1.051 5.558a1 1 0 0 1-1.966 0l-1.051-5.558a2 2 0 0 0-1.594-1.594l-5.558-1.051a1 1 0 0 1 0-1.966l5.558-1.051a2 2 0 0 0 1.594-1.594z\" /> <path d=\"M20 2v4\" /> <path d=\"M22 4h-4\" /> <circle cx=\"4\" cy=\"20\" r=\"2\" />",
 "check-circle": "<circle cx=\"12\" cy=\"12\" r=\"10\" /> <path d=\"m16 9-5.5 5.5L8 12\" />",
 "table": "<path d=\"M12 3v18\" /> <rect width=\"18\" height=\"18\" x=\"3\" y=\"3\" rx=\"2\" /> <path d=\"M3 9h18\" /> <path d=\"M3 15h18\" />",
 "arrow-left": "<path d=\"m12 19-7-7 7-7\" /> <path d=\"M19 12H5\" />",
 "arrow-right": "<path d=\"M5 12h14\" /> <path d=\"m12 5 7 7-7 7\" />",
 "alert-triangle": "<path d=\"m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3\" /> <path d=\"M12 9v4\" /> <path d=\"M12 17h.01\" />",
 "flame": "<path d=\"M12 3q1 4 4 6.5t3 5.5a1 1 0 0 1-14 0 5 5 0 0 1 1-3 1 1 0 0 0 5 0c0-2-1.5-3-1.5-5q0-2 2.5-4\" />",
 "columns": "<rect width=\"18\" height=\"18\" x=\"3\" y=\"3\" rx=\"2\" /> <path d=\"M9 3v18\" /> <path d=\"M15 3v18\" />",
 "database": "<ellipse cx=\"12\" cy=\"5\" rx=\"9\" ry=\"3\" /> <path d=\"M3 5V19A9 3 0 0 0 21 19V5\" /> <path d=\"M3 12A9 3 0 0 0 21 12\" />",
 "fingerprint": "<path d=\"M12 10a2 2 0 0 0-2 2c0 1.02-.1 2.51-.26 4\" /> <path d=\"M14 13.12c0 2.38 0 6.38-1 8.88\" /> <path d=\"M17.29 21.02c.12-.6.43-2.3.5-3.02\" /> <path d=\"M2 12a10 10 0 0 1 18-6\" /> <path d=\"M2 16h.01\" /> <path d=\"M21.8 16c.2-2 .131-5.354 0-6\" /> <path d=\"M5 19.5C5.5 18 6 15 6 12a6 6 0 0 1 .34-2\" /> <path d=\"M8.65 22c.21-.66.45-1.32.57-2\" /> <path d=\"M9 6.8a6 6 0 0 1 9 5.2v2\" />",
 "users": "<path d=\"M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2\" /> <path d=\"M16 3.128a4 4 0 0 1 0 7.744\" /> <path d=\"M22 21v-2a4 4 0 0 0-3-3.87\" /> <circle cx=\"9\" cy=\"7\" r=\"4\" />",
 "lightbulb": "<path d=\"M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5\" /> <path d=\"M9 18h6\" /> <path d=\"M10 22h4\" />",
 "badge-check": "<path d=\"M3.85 8.62a4 4 0 0 1 4.78-4.77 4 4 0 0 1 6.74 0 4 4 0 0 1 4.78 4.78 4 4 0 0 1 0 6.74 4 4 0 0 1-4.77 4.78 4 4 0 0 1-6.75 0 4 4 0 0 1-4.78-4.77 4 4 0 0 1 0-6.76Z\" /> <path d=\"m16 9-5.5 5.5L8 12\" />",
 "shield-x": "<path d=\"M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z\" /> <path d=\"m14.5 9.5-5 5\" /> <path d=\"m9.5 9.5 5 5\" />",
 "check-check": "<path d=\"M18 6 7 17l-5-5\" /> <path d=\"m22 10-7.5 7.5L13 16\" />",
 "trash": "<path d=\"M10 11v6\" /> <path d=\"M14 11v6\" /> <path d=\"M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6\" /> <path d=\"M3 6h18\" /> <path d=\"M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2\" />",
 "layers": "<path d=\"M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z\" /> <path d=\"M2 12a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 12\" /> <path d=\"M2 17a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 17\" />",
 "git-compare": "<circle cx=\"18\" cy=\"18\" r=\"3\" /> <circle cx=\"6\" cy=\"6\" r=\"3\" /> <path d=\"M13 6h3a2 2 0 0 1 2 2v7\" /> <path d=\"M11 18H8a2 2 0 0 1-2-2V9\" />",
 "download": "<path d=\"M12 15V3\" /> <path d=\"M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4\" /> <path d=\"m7 10 5 5 5-5\" />",
 "file-text": "<path d=\"M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z\" /> <path d=\"M14 2v5a1 1 0 0 0 1 1h5\" /> <path d=\"M10 9H8\" /> <path d=\"M16 13H8\" /> <path d=\"M16 17H8\" />",
 "sliders": "<path d=\"M10 5H3\" /> <path d=\"M12 19H3\" /> <path d=\"M14 3v4\" /> <path d=\"M16 17v4\" /> <path d=\"M21 12h-9\" /> <path d=\"M21 19h-5\" /> <path d=\"M21 5h-7\" /> <path d=\"M8 10v4\" /> <path d=\"M8 12H3\" />",
 "list-filter": "<path d=\"M2 5h20\" /> <path d=\"M6 12h12\" /> <path d=\"M9 19h6\" />",
 "binary": "<rect x=\"14\" y=\"14\" width=\"4\" height=\"6\" rx=\"2\" /> <rect x=\"6\" y=\"4\" width=\"4\" height=\"6\" rx=\"2\" /> <path d=\"M6 20h4\" /> <path d=\"M14 10h4\" /> <path d=\"M6 14h2v6\" /> <path d=\"M14 4h2v6\" />",
 "file-sliders": "<path d=\"M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z\" /> <path d=\"M14 2v5a1 1 0 0 0 1 1h5\" /> <path d=\"M8 12h8\" /> <path d=\"M10 11v2\" /> <path d=\"M8 17h8\" /> <path d=\"M14 16v2\" />",
 "git-merge": "<circle cx=\"18\" cy=\"18\" r=\"3\" /> <circle cx=\"6\" cy=\"6\" r=\"3\" /> <path d=\"M6 21V9a9 9 0 0 0 9 9\" />",
 "info": "<circle cx=\"12\" cy=\"12\" r=\"10\" /> <path d=\"M12 16v-4\" /> <path d=\"M12 8h.01\" />",
 "circle": "<circle cx=\"12\" cy=\"12\" r=\"10\" />",
 "check": "<path d=\"M21.801 10A10 10 0 1 1 17 3.335\" /> <path d=\"m9 11 3 3L22 4\" />",
 "eye-off": "<path d=\"M10.733 5.076a10.744 10.744 0 0 1 11.205 6.575 1 1 0 0 1 0 .696 10.747 10.747 0 0 1-1.444 2.49\" /> <path d=\"M14.084 14.158a3 3 0 0 1-4.242-4.242\" /> <path d=\"M17.479 17.499a10.75 10.75 0 0 1-15.417-5.151 1 1 0 0 1 0-.696 10.75 10.75 0 0 1 4.446-5.143\" /> <path d=\"m2 2 20 20\" />"
}


def icon(name, cls="", size=18):
    return (f'<svg class="mm-ic {cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            f'{ICONS[name]}</svg>')


LEVEL = {  # colour, soft background, border, gauge gradient — green low, amber medium, red high and critical
    "critical": ("#F87171", "rgba(239,68,68,.18)", "rgba(239,68,68,.45)", ("#EF4444", "#DC2626")),
    "high": ("#F87171", "rgba(239,68,68,.14)", "rgba(239,68,68,.38)", ("#F97316", "#EF4444")),
    "medium": ("#FBBF24", "rgba(245,158,11,.16)", "rgba(245,158,11,.40)", ("#FDE047", "#F59E0B")),
    "low": ("#34D399", "rgba(16,185,129,.16)", "rgba(16,185,129,.40)", ("#00D2B8", "#00F5A0")),
    "none": ("#94A3B8", "rgba(148,163,184,.14)", "rgba(148,163,184,.35)", ("#64748B", "#94A3B8")),  # not scored
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=IBM+Plex+Sans+Arabic:wght@400;500;600&display=swap');
:root {
  --bg: #071526; --bg2: #0A1C33; --card: rgba(12,32,55,.75); --card2: #091C33; --deep: #050E1B;
  --line: rgba(0,210,184,.18); --line2: rgba(0,210,184,.30); --glow: rgba(0,245,160,.45);
  --mint: #00F5A0; --teal: #00D2B8; --cyan: #00A3CE; --blue: #1D4ED8;
  --ink: #F8FAFC; --text: #CBD5E1; --muted: #94A3B8; --faint: #64748B;
  --red: #F87171; --amber: #FBBF24; --violet: #C4B5FD;
}
html, body, .stApp, .stMarkdown, p, label, input, textarea, button, [data-testid="stWidgetLabel"] {
  font-family: 'Plus Jakarta Sans', 'IBM Plex Sans Arabic', -apple-system, BlinkMacSystemFont, sans-serif; }
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }
.stApp { background-color: var(--bg); color: var(--ink);
  background-image: radial-gradient(circle at 50% 0%, rgba(0,210,184,.12) 0%, transparent 60%),
    linear-gradient(to right, rgba(255,255,255,.02) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(255,255,255,.02) 1px, transparent 1px);
  background-size: 100% 100%, 40px 40px, 40px 40px; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
/* The page uses the full window width (the header and step bar span it); each screen keeps its own readable width. */
.block-container { max-width: none; padding-top: 3.2rem; padding-bottom: 0;
  padding-left: clamp(1rem, 3vw, 3rem) !important; padding-right: clamp(1rem, 3vw, 3rem) !important; }
.element-container:has(iframe[height="0"]) { display: none; }
.mm-ic { display: inline-block; vertical-align: -3px; flex: none; }
.mono { font-family: 'JetBrains Mono', ui-monospace, monospace; }

/* Sadu-AI pattern bars + footer */
.mm-sadu-top { position: fixed; top: 0; left: 0; right: 0; height: 32px; z-index: 999990; opacity: .85;
  background: var(--bg) url("__SADU__") repeat-x; background-size: auto 32px; border-bottom: 1px solid rgba(0,210,184,.2); }
.mm-foot { width: 100vw; margin: 3rem 0 0 calc(50% - 50vw); }
.mm-foot .sadu { height: 32px; opacity: .85; background: url("__SADU__") repeat-x; background-size: auto 32px; }
.mm-foot .bar { display: flex; flex-wrap: wrap; justify-content: space-between; gap: .6rem; padding: 1rem 1.2rem;
  border-top: 1px solid rgba(0,210,184,.2); background: rgba(12,32,55,.75); font-size: .82rem; color: var(--muted); }
.mm-foot .bar b { color: var(--teal); }

/* Header */
div:has(> .st-key-topbar) { position: sticky; top: 38px; z-index: 999980; }
.st-key-topbar { background: rgba(9,26,46,.97); backdrop-filter: blur(16px); border: 1px solid var(--line);
  border-radius: 16px; padding: .55rem .9rem; margin-bottom: 2rem; align-items: center !important; }
.mm-logo img { height: 46px; width: auto; display: block; }
.mm-badge2026 { display: flex; align-items: center; gap: .45rem; justify-content: flex-end; }
.mm-badge2026 span { background: linear-gradient(90deg, rgba(30,58,138,.4), rgba(19,78,74,.3)); border: 1px solid var(--line2);
  border-radius: 7px; padding: .25rem .55rem; font-size: .72rem; color: #99F6E4; line-height: 1.2; }
.mm-badge2026 b { color: var(--mint); }
/* Step bar: fills the space between logo and badge; the five steps share it equally, in one row, never scrolling. */
.st-key-topbar > div:has(> .st-key-navbar) { flex: 1 1 0 !important; min-width: 0 !important; }
.st-key-navbar { width: 100%; min-width: 0; background: rgba(5,14,27,.8); border: 1px solid var(--line); border-radius: 99px;
  padding: .25rem .35rem; gap: .15rem !important; justify-content: space-between; flex-wrap: nowrap !important; }
.st-key-navbar > div { flex: 1 1 0 !important; min-width: 0 !important; width: auto !important; }
[class*="st-key-nav_"] button { width: 100%; }
[class*="st-key-nav_"] button { background: transparent !important; border: 1px solid transparent !important; border-radius: 99px !important;
  min-height: 2.1rem !important; padding: .2rem .7rem !important; color: var(--muted) !important; box-shadow: none !important; }
[class*="st-key-nav_"] button p { font-size: .76rem !important; font-weight: 500 !important; white-space: nowrap; }
[class*="st-key-nav_"] button:hover:not(:disabled) { color: var(--ink) !important; }
[class*="st-key-nav_"] button:disabled { opacity: .45; }
__NAV_ACTIVE__

/* Glass panels */
.mm-glass { background: var(--card); backdrop-filter: blur(16px); border: 1px solid var(--line); border-radius: 16px;
  box-shadow: 0 10px 30px -10px rgba(0,0,0,.6), inset 0 1px 0 rgba(255,255,255,.08); }
[class*="st-key-panel"] { background: var(--card); backdrop-filter: blur(16px); border: 1px solid var(--line); border-radius: 18px;
  box-shadow: 0 10px 30px -10px rgba(0,0,0,.6), inset 0 1px 0 rgba(255,255,255,.08); padding: 1.2rem 1.3rem; }
.mm-narrow { max-width: 780px; margin: 0 auto; }
[class*="st-key-narrow"] { max-width: 1080px; margin: 0 auto; width: 100%; }
[class*="st-key-wide"] { max-width: 1240px; margin: 0 auto; width: 100%; }

/* Screen 1: ingest */
.mm-pillchip { display: inline-flex; align-items: center; gap: .45rem; padding: .3rem .8rem; border-radius: 99px;
  background: rgba(0,210,184,.1); border: 1px solid var(--line2); color: #5EEAD4; font-size: .78rem; font-weight: 500; }
.mm-hero { text-align: center; max-width: 760px; margin: 0 auto 2rem; }
.mm-hero h1 { font-size: 3rem; line-height: 1.12; font-weight: 800; letter-spacing: -.025em; color: #fff; margin: 1rem 0 .9rem; padding: 0; }
.grad { background: linear-gradient(135deg, #00F5A0 0%, #00D2B8 45%, #38BDF8 100%); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; background-clip: text; }
.mm-hero p { color: var(--muted); font-size: 1.02rem; line-height: 1.6; max-width: 40rem; margin: 0 auto; }
.st-key-ingest { max-width: 680px; margin: 0 auto; width: 100%; }
[data-testid="stFileUploaderDropzone"] { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: .7rem;
  min-height: 270px; padding: 2.4rem 1rem; border: 2px dashed rgba(0,210,184,.3); border-radius: 18px; background: var(--card);
  transition: border-color .15s ease, background-color .15s ease; cursor: pointer; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: #2DD4BF; background: rgba(12,34,59,.8); }
body.mm-dragging [data-testid="stFileUploaderDropzone"] { border-color: var(--mint); border-style: solid; }
[data-testid="stFileUploaderDropzone"]::before { content: ""; order: 0; width: 64px; height: 64px; border-radius: 16px;
  border: 1px solid rgba(0,210,184,.4); background: linear-gradient(45deg, rgba(20,184,166,.2), rgba(59,130,246,.2)) center / 100% no-repeat,
  url("__UPLOAD__") center / 32px no-repeat; background-blend-mode: normal; }
[data-testid="stFileUploaderDropzoneInstructions"] { order: 1; text-align: center; display: flex !important; flex-direction: column;
  align-items: center; }
[data-testid="stFileUploaderDropzoneInstructions"] span { display: none; }
[data-testid="stFileUploaderDropzoneInstructions"]::before { content: "Drag & drop your CSV here"; font-size: 1.12rem; font-weight: 700; color: #fff; }
[data-testid="stFileUploaderDropzoneInstructions"]::after { content: "Arabic or English column names, one row per person"; display: block;
  color: var(--muted); font-size: .78rem; margin-top: .3rem; }
[data-testid="stFileUploaderDropzone"] > span { order: 2; }
[data-testid="stFileUploaderDropzone"]::after { content: "Your file stays on this computer."; order: 3; color: var(--muted); font-size: .72rem; }
[data-testid="stFileUploaderDropzone"] button { background: linear-gradient(135deg, #00F5A0 0%, #00D2B8 50%, #0284C7 100%) !important;
  border: 0 !important; color: #061524 !important; border-radius: 12px !important; padding: .55rem 1.5rem !important;
  box-shadow: 0 4px 18px -2px rgba(0,245,160,.35); }
[data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"],
[data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"], [data-testid="stFileUploaderDropzone"] button > div > span { display: none; }
[data-testid="stFileUploaderDropzone"] button > div::after { content: "Browse"; font-weight: 700; color: #061524; }
.mm-pillars { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 2rem; }
.mm-pillar { display: flex; align-items: center; gap: .85rem; padding: 1rem; }
.mm-pillar .ico { width: 40px; height: 40px; border-radius: 10px; display: grid; place-items: center; flex: none; border: 1px solid; }
.mm-pillar h4 { margin: 0; padding: 0; font-size: .78rem; font-weight: 700; color: #fff; white-space: nowrap; }
.mm-pillar p { margin: .1rem 0 0; font-size: .72rem; color: var(--muted); line-height: 1.4; }
.ico-teal { background: rgba(20,184,166,.1); border-color: rgba(20,184,166,.3) !important; color: #5EEAD4; }
.ico-cyan { background: rgba(6,182,212,.1); border-color: rgba(6,182,212,.3) !important; color: #67E8F9; }
.ico-blue { background: rgba(59,130,246,.1); border-color: rgba(59,130,246,.3) !important; color: #93C5FD; }
.ico-red { background: rgba(244,63,94,.1); border-color: rgba(244,63,94,.3) !important; color: #FDA4AF; }
.ico-green { background: rgba(16,185,129,.1); border-color: rgba(16,185,129,.3) !important; color: #6EE7B7; }
.ico-amber { background: rgba(245,158,11,.15); border-color: rgba(245,158,11,.3) !important; color: #FCD34D; }

/* Screen 2: preview */
.mm-banner { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 1rem; padding: 1rem 1.1rem; }
.mm-banner .l { display: flex; align-items: center; gap: .8rem; min-width: 0; }
.mm-banner .dot { width: 40px; height: 40px; border-radius: 50%; display: grid; place-items: center; flex: none; }
.mm-eyebrow { font-size: .74rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }
.mm-chip { font-family: 'JetBrains Mono', monospace; font-size: .72rem; padding: .1rem .5rem; border-radius: 6px;
  background: rgba(0,210,184,.1); color: #5EEAD4; border: 1px solid rgba(0,210,184,.2); }
.mm-fname { font-family: 'JetBrains Mono', monospace; font-size: .9rem; font-weight: 700; color: #fff; margin-top: .15rem; word-break: break-all; }
.mm-meta { color: var(--muted); font-size: .78rem; display: flex; gap: .5rem; flex-wrap: wrap; }
.mm-tablecard { overflow: hidden; padding: 0; margin-top: 1.4rem; }
.mm-tablecard .h { display: flex; justify-content: space-between; align-items: center; gap: 1rem; padding: 1rem 1.2rem;
  border-bottom: 1px solid var(--line); background: #081A30; }
.mm-tablecard .h b { color: #fff; font-size: .9rem; display: flex; align-items: center; gap: .45rem; }
.mm-tablecard .h small { display: block; color: var(--muted); font-size: .74rem; margin-top: .1rem; }
.mm-tscroll { overflow-x: auto; max-height: 20rem; }
table.mm-t { width: 100%; border-collapse: collapse; font-size: .78rem; }
table.mm-t th { background: rgba(10,28,51,.95); color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: .7rem;
  letter-spacing: .06em; text-align: left; padding: .7rem 1rem; white-space: nowrap; }
table.mm-t td { padding: .7rem 1rem; border-top: 1px solid rgba(255,255,255,.05); color: var(--text);
  font-family: 'JetBrains Mono', 'IBM Plex Sans Arabic', monospace; white-space: nowrap; }
table.mm-t tr:hover td { background: rgba(0,210,184,.04); }

/* Buttons */
div[data-testid="stButton"] button[kind="primary"], div[data-testid="stDownloadButton"] button[kind="primary"] {
  background: linear-gradient(135deg, #00F5A0 0%, #00D2B8 50%, #0284C7 100%); border: 0; color: #061524; border-radius: 13px;
  min-height: 3rem; box-shadow: 0 4px 18px -2px rgba(0,245,160,.35); transition: transform .25s ease, box-shadow .25s ease, filter .25s ease; }
div[data-testid="stButton"] button[kind="primary"] p, div[data-testid="stDownloadButton"] button[kind="primary"] p { font-weight: 700; color: #061524; font-size: .95rem; }
div[data-testid="stButton"] button[kind="primary"]:hover, div[data-testid="stDownloadButton"] button[kind="primary"]:hover {
  transform: translateY(-2px); box-shadow: 0 8px 24px -2px rgba(0,245,160,.55); filter: brightness(1.08); color: #061524; }
div[data-testid="stButton"] button[kind="secondary"], div[data-testid="stDownloadButton"] button[kind="secondary"] {
  background: rgba(16,42,73,.7); border: 1px solid rgba(0,210,184,.3); color: #E2E8F0; border-radius: 12px; min-height: 2.7rem; }
div[data-testid="stButton"] button[kind="secondary"]:hover, div[data-testid="stDownloadButton"] button[kind="secondary"]:hover {
  background: rgba(22,58,102,.85); border-color: var(--mint); color: #fff; }
div[data-testid="stButton"] button:focus-visible, div[data-testid="stDownloadButton"] button:focus-visible { outline: 3px solid rgba(0,245,160,.4); outline-offset: 2px; }
div[data-testid="stExpander"] details { border-radius: 14px; border: 1px solid var(--line); background: var(--card); }
div[data-testid="stExpander"] summary p { font-weight: 600; color: #E2E8F0; }

/* Screen 3: scan */
.mm-scan { text-align: center; max-width: 560px; margin: 0 auto; padding: 1rem 0; }
.mm-radar { position: relative; width: 176px; height: 176px; margin: 0 auto 1.8rem; display: grid; place-items: center; }
.mm-radar .ring { position: absolute; inset: 0; border-radius: 50%; border: 2px solid rgba(0,245,160,.25);
  background: radial-gradient(circle, rgba(0,210,184,.05) 0%, rgba(7,21,38,.8) 70%); }
.mm-radar .sweep { position: absolute; inset: 0; border-radius: 50%; background: conic-gradient(from 0deg, rgba(0,245,160,.5) 0deg, rgba(0,210,184,0) 60deg);
  animation: mmspin 2.4s linear infinite; }
.mm-radar .core { position: relative; width: 80px; height: 80px; border-radius: 16px; background: #0C223B; border: 1px solid rgba(45,212,191,.5);
  display: grid; place-items: center; color: #5EEAD4; animation: mmpulse 2.5s ease-in-out infinite; }
@keyframes mmspin { to { transform: rotate(360deg); } }
@keyframes mmpulse { 0%,100% { opacity: .55; transform: scale(.98); } 50% { opacity: 1; transform: scale(1.03); } }
.mm-scan h2 { font-size: 1.5rem; font-weight: 800; color: #fff; margin: 0 0 .5rem; padding: 0; }
.mm-scan p.sub { color: var(--muted); font-size: .8rem; margin: 0 auto 1.4rem; max-width: 28rem; }
.mm-stages { background: #07192F; border: 1px solid var(--line); border-radius: 12px; padding: 1rem; text-align: left; }
.mm-stages .row { display: flex; justify-content: space-between; font-size: .76rem; margin-bottom: .6rem; gap: .5rem; }
.mm-stages .row span { display: flex; align-items: center; gap: .35rem; }
.st-on { color: #5EEAD4; font-weight: 600; } .st-off { color: var(--faint); }
.mm-track { height: 8px; border-radius: 99px; background: #1E293B; overflow: hidden; }
.mm-track span { display: block; height: 100%; background: linear-gradient(90deg, #2DD4BF, #22D3EE); transition: width .4s ease; }
.mm-term { margin-top: 1.4rem; text-align: left; background: #040C16; border: 1px solid var(--line); border-radius: 10px; padding: .75rem;
  font-family: 'JetBrains Mono', monospace; font-size: .7rem; color: rgba(45,212,191,.8); height: 8.5rem; overflow: hidden;
  display: flex; flex-direction: column; justify-content: flex-end; gap: .2rem; }
.mm-spin { animation: mmspin 1s linear infinite; }

/* Screen 4: scorecard */
.mm-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: 1.4rem; }
.mm-head h2 { font-size: 1.55rem; font-weight: 800; color: #fff; margin: 0; padding: 0; display: flex; align-items: center; gap: .55rem; }
.mm-head p { color: var(--muted); font-size: .78rem; margin: .2rem 0 0; }
.mm-score { display: grid; grid-template-columns: 1fr 2fr; gap: 1.5rem; align-items: center; padding: 1.8rem; }
.mm-gauge { display: flex; flex-direction: column; align-items: center; }
.mm-gauge .g { position: relative; width: 148px; height: 148px; }
.mm-gauge .g svg { width: 100%; height: 100%; transform: rotate(-90deg); }
.mm-gauge .g div { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.mm-gauge .g strong { font-family: 'JetBrains Mono', monospace; font-size: 2.3rem; font-weight: 800; color: #fff; line-height: 1; }
.mm-gauge .g small { font-size: .74rem; color: var(--muted); font-weight: 600; margin-top: .2rem; }
.mm-sev { margin-top: .6rem; display: inline-flex; align-items: center; gap: .3rem; padding: .15rem .75rem; border-radius: 99px; font-size: .76rem; font-weight: 700; }
.mm-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: .75rem; }
.mm-metric { background: var(--card2); border: 1px solid var(--line); border-radius: 12px; padding: 1rem; }
.mm-metric .t { display: flex; justify-content: space-between; align-items: center; color: var(--muted); font-size: .76rem; font-weight: 500; margin-bottom: .3rem; }
.mm-metric .v { font-family: 'JetBrains Mono', monospace; font-size: 1.55rem; font-weight: 700; color: #fff; line-height: 1.2; }
.mm-metric .d { font-size: .68rem; color: var(--muted); margin-top: .2rem; }
.mm-meaning { display: flex; gap: 1rem; align-items: flex-start; padding: 1.25rem; margin-top: 1.4rem; border-color: rgba(245,158,11,.3) !important; background: rgba(10,31,54,.9); }
.mm-meaning .ico { width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center; flex: none; margin-top: .1rem; }
.mm-meaning h4 { font-size: .9rem; font-weight: 700; color: #fff; margin: 0 0 .3rem; padding: 0; }
.mm-meaning p { font-size: .8rem; color: var(--text); line-height: 1.65; margin: 0; }
.mm-meaning p.fine { color: var(--muted); font-size: .74rem; margin-top: .4rem; }
.mm-review { background: rgba(245,158,11,.1); border: 1px solid rgba(245,158,11,.3); color: #FDE68A; border-radius: 12px; padding: .8rem 1rem;
  font-size: .82rem; line-height: 1.55; margin: 1.2rem 0 0; }
.mm-review ul { margin: .35rem 0 .2rem 1.1rem; padding: 0; }
.mm-h2 { font-size: .95rem; font-weight: 700; color: #fff; margin: 1.6rem 0 .7rem; display: flex; align-items: center; gap: .45rem; }
.mm-kind { display: inline-block; font-size: .7rem; font-weight: 700; border-radius: 5px; padding: .1rem .45rem; }
.mm-kind.k-direct { background: rgba(239,68,68,.2); color: #FDA4AF; } .mm-kind.k-combo { background: rgba(245,158,11,.2); color: #FCD34D; }
.mm-kind.k-private { background: rgba(139,92,246,.2); color: #C4B5FD; }
.mm-tag { display: inline-block; font-family: 'JetBrains Mono', 'IBM Plex Sans Arabic', monospace; background: rgba(255,255,255,.06);
  color: #fff; border: 1px solid rgba(255,255,255,.08); border-radius: 6px; padding: 0 .4rem; font-size: .92em; line-height: 1.65;
  direction: rtl; unicode-bidi: isolate; }
.mm-small { color: var(--muted); font-size: .76rem; line-height: 1.5; }
/* One list of protection actions: Result (what to do) and Safer copy (what changed) */
.mm-recs { padding: .3rem 1.1rem; }
.mm-rec { display: grid; grid-template-columns: 8.6rem 1fr; gap: .9rem; align-items: start; padding: .8rem 0; }
.mm-rec + .mm-rec { border-top: 1px solid rgba(255,255,255,.06); }
.mm-rec .mm-kind { text-align: center; padding: .2rem .45rem; margin-top: .1rem; }
.mm-rec .cols { display: flex; flex-wrap: wrap; gap: .35rem .7rem; }
.mm-ci { font-size: .8rem; color: var(--text); }
.mm-ci em { font-style: normal; color: var(--muted); font-size: .76rem; margin-left: .35rem; }
.mm-rec > div > b { font-size: .82rem; color: #fff; font-weight: 600; }
.mm-rec p { margin: .3rem 0 0; font-size: .76rem; color: var(--muted); line-height: 1.5; }
.mm-rec p + .cols { margin-top: .7rem; }
@media (max-width: 640px) { .mm-rec { grid-template-columns: 1fr; gap: .4rem; } .mm-rec .mm-kind { justify-self: start; } }

/* Screen 5: mitigation */
.mm-done { display: flex; align-items: center; gap: .8rem; padding: 1rem 1.1rem; border-color: rgba(16,185,129,.4) !important; background: rgba(2,44,34,.2); }
.mm-done .dot { width: 40px; height: 40px; border-radius: 50%; background: rgba(16,185,129,.2); color: #6EE7B7; display: grid; place-items: center; flex: none; }
.mm-done h2 { font-size: 1rem; font-weight: 700; color: #fff; margin: .1rem 0 0; padding: 0; }
.mm-done p { color: var(--muted); font-size: .76rem; margin: .15rem 0 0; }
.mm-ba { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; padding: 1.5rem; margin-top: 1.4rem; }
.mm-side { border-radius: 12px; padding: 1.25rem; display: flex; justify-content: space-between; align-items: center; gap: .8rem; border: 1px solid; }
.mm-side .l { font-size: .74rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.mm-side strong { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 800; line-height: 1.2; }
.mm-side strong small { font-size: .74rem; color: var(--muted); font-weight: 500; }
.mm-side .tagl { display: inline-block; margin-top: .45rem; padding: .1rem .5rem; border-radius: 5px; font-size: .7rem; font-weight: 700; }
.mm-side .ring { width: 48px; height: 48px; border-radius: 50%; display: grid; place-items: center; flex: none; }
.mm-list { list-style: none; padding: 0; margin: 0; }
.mm-list li { display: flex; gap: .65rem; padding: .55rem 0; border-bottom: 1px solid rgba(255,255,255,.05); font-size: .8rem; color: var(--text); line-height: 1.55; }
.mm-list li:last-child { border-bottom: 0; }
.mm-list li i { flex: none; width: 20px; height: 20px; border-radius: 50%; background: rgba(16,185,129,.2); color: #6EE7B7; font-style: normal;
  font-weight: 800; display: grid; place-items: center; font-size: .68rem; margin-top: 1px; }
.mm-note { color: var(--muted); font-size: .78rem; line-height: 1.55; margin: .8rem 0 0; }
.mm-caveat { background: rgba(245,158,11,.1); border: 1px solid rgba(245,158,11,.3); color: #FDE68A; border-radius: 12px; padding: .75rem .95rem;
  font-size: .8rem; line-height: 1.55; margin: 1rem 0 0; }
.mm-diff td.raw { color: #FDA4AF; } .mm-diff td.safe { color: #6EE7B7; }
.mm-diff td small { display: block; color: var(--faint); font-size: .66rem; }
.mm-defense { font-size: .66rem; padding: .1rem .4rem; border-radius: 4px; background: rgba(0,210,184,.12); color: #5EEAD4;
  border: 1px solid rgba(0,210,184,.3); white-space: nowrap; font-family: 'JetBrains Mono', monospace; }
/* Screen 6: tabs */
.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 10px; padding: .45rem .9rem; color: var(--muted); }
.stTabs [aria-selected="true"] { background: rgba(0,210,184,.2); color: #5EEAD4; border: 1px solid rgba(0,210,184,.4); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
@media (max-width: 860px) {
  .mm-hero h1 { font-size: 2.1rem; } .mm-pillars { grid-template-columns: 1fr; }
  .mm-score, .mm-ba { grid-template-columns: 1fr; } }
/* Narrower screens: the step bar moves to its own full-width row under the logo and badge. */
@media (max-width: 1100px) {
  .st-key-topbar { flex-wrap: wrap !important; row-gap: .5rem !important; }
  .st-key-topbar > div:has(> .st-key-navbar), .st-key-topbar > .st-key-navbar { order: 3; flex: 1 1 100% !important; }
}
/* Phones: the five steps wrap onto two rows instead of scrolling sideways. */
@media (max-width: 640px) {
  .st-key-navbar { flex-wrap: wrap !important; border-radius: 16px; }
  .st-key-navbar > div { flex: 1 1 30% !important; }
}
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
"""


UPLOAD_SVG = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%235EEAD4' "
              "stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M12 13v8'/>"
              "<path d='M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242'/><path d='m8 17 4-4 4 4'/></svg>")


def css(active_step):
    nav = (f".st-key-nav_{active_step} button {{ background: rgba(0,210,184,.12) !important; border-color: rgba(0,210,184,.4) !important; "
           f"color: #5EEAD4 !important; }} .st-key-nav_{active_step} button p {{ font-weight: 700 !important; }}")
    return CSS.replace("__SADU__", SADU).replace("__UPLOAD__", UPLOAD_SVG).replace("__NAV_ACTIVE__", nav)
