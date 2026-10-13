"""
Ma'man — how easily can the people in a file be recognised, and a safer copy of it.

Run:  streamlit run app.py      (or double-click run_app.command)
Everything runs locally: the CSV, the model and the risk engine stay on this machine.

Screens: 1 Upload → 2 Preview → 3 Analyzing (Qwen labels the columns) → 4 Result → 5 Safer copy.
Every number on the page comes from scripts/risk_engine.py (group sizes over the real values);
the safer copy is measured again after it is written. The look lives in scripts/ui_style.py.
"""

import gc
import hashlib
import html
import io
import json
import math
import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import dataset_profile as DP  # noqa: E402
import risk_engine as R  # noqa: E402
import ui_style as U  # noqa: E402
from linkage_attack import simulate  # noqa: E402
from maman_common import ROLE_KEYS, describe_column  # noqa: E402

ROLE_LABELS = {
    "direct_identifier": "Names a person",
    "quasi_identifier": "Reveals when combined",
    "sensitive_attribute": "Private",
    "record_identifier": "Row or system number",
}
AUDIENCE = {
    "public": "Anyone could see it",
    "partner": "A trusted organisation under an agreement",
    "internal": "Staff inside your organisation",
}
STEPS = ["Upload", "Preview", "Analyzing", "Result", "Safer copy"]
LEVEL_COLOR = {k: v[0] for k, v in U.LEVEL.items()}

st.set_page_config(page_title="Ma'man — privacy risk check", page_icon="🛡️", layout="wide")
st.session_state.setdefault("step", 1)
st.session_state.setdefault("context", "public")
st.session_state.setdefault("row_kind", "auto")
st.markdown(U.css(st.session_state["step"]), unsafe_allow_html=True)
st.markdown('<div class="mm-sadu-top"></div>', unsafe_allow_html=True)

# Each new screen opens at the top (Streamlit keeps the scroll position between reruns)
# (always drawn in the same place, so it never shifts the page; it only runs again when the step changes)
components.html(f"""<script>/* step {st.session_state['step']} */
const m = window.parent.document.querySelector('[data-testid="stMain"]') || window.parent.document.querySelector('section.main');
if (m) m.scrollTo({{top: 0}}); window.parent.scrollTo({{top: 0}});</script>""", height=0)

# Highlights the upload zone while a file is dragged over the page (Streamlit has no drag state of its own)
components.html("""<script>
const d = window.parent.document;
if (!d.mmDrag) { d.mmDrag = true; let n = 0;
  d.addEventListener('dragenter', () => { n++; d.body.classList.add('mm-dragging'); });
  d.addEventListener('dragleave', () => { n = Math.max(0, n - 1); if (!n) d.body.classList.remove('mm-dragging'); });
  d.addEventListener('drop', () => { n = 0; d.body.classList.remove('mm-dragging'); });
}
</script>""", height=0)


# ═════════════════════════════════════════════
# Models: one in memory at a time (fine-tuned for columns, base for the optional summary)
# ═════════════════════════════════════════════
class ModelWorker:
    """All Qwen work runs on ONE fixed thread, one job at a time.
    - MLX ties its GPU stream to the thread that created it; Streamlit runs every click on a new
      thread, which caused "There is no Stream(cpu, 8) in current thread".
    - A single queue also means two clicks or two tabs can never load two models at once
      (that, plus swapping models without freeing them, caused the Metal out-of-memory error)."""

    def __init__(self):
        import concurrent.futures as cf
        self.pool = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen")
        self.kind, self.obj = None, None
        self.pool.submit(self._setup).result()

    def _setup(self):
        try:
            import mlx.core as mx
            mx.set_cache_limit(256 * 2**20)  # do not hoard freed GPU buffers between calls
        except Exception:
            pass

    def _free(self):
        self.kind, self.obj = None, None
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass

    def _get(self, kind):
        if self.kind != kind:
            self._free()  # the old model is gone before the new one loads
            if kind == "classifier":
                from maman_classifier import MamanClassifier
                merged = ROOT / "adapters_rag"  # RAG + fine-tuning: best on the unseen test files (0.856 vs 0.723)
                self.obj = (MamanClassifier("rag_finetuned", adapter_path=merged) if merged.exists()
                            else MamanClassifier("finetuned", adapter_path=ROOT / "adapters"))
            else:
                from explain_report import Explainer
                self.obj = Explainer()
            self.kind = kind
        return self.obj

    def _call(self, kind, method, args):
        try:
            return getattr(self._get(kind), method)(*args)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
        # raised outside the except block: the old traceback (which holds the model) is dropped
        self._free()  # so the next try starts clean
        raise RuntimeError(msg)

    def run(self, kind, method, *args):
        return self.pool.submit(self._call, kind, method, args).result()

    def release(self):
        self.pool.submit(self._free).result()


@st.cache_resource
def models():
    return ModelWorker()


def classify_column(desc):
    """One retry after freeing memory; if it still fails the caller shows a friendly message."""
    if os.environ.get("MAMAN_FAKE_MODEL"):  # automated tests only (tests/test_app.py): no Qwen
        sys.path.insert(0, str(ROOT / "tests"))
        from fake_model import classify
        return classify(desc)
    try:
        return models().run("classifier", "classify", desc)
    except RuntimeError:
        return models().run("classifier", "classify", desc)


# ═════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════
def md(text):
    st.markdown(text, unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x))


def tag(c):
    """Column name as a small tag, so Arabic names keep their order inside English sentences."""
    return f'<span class="mm-tag">{esc(c)}</span>'


def chips(text):
    """Plain text whose “quoted” column names or values become tags, so Arabic keeps its order in English text."""
    parts = re.split(r"“([^”]*)”", str(text))
    return "".join(tag(x) if i % 2 else esc(x) for i, x in enumerate(parts))


def tags(cols):
    return " ".join(tag(c) for c in cols)


def num(x):
    """A 0–100 value for display, never rounded up to 100 or down to 0 when it is not exactly that
    (99.5 stays 99.5, 0.3 stays 0.3)."""
    if x <= 0:
        return "0"
    if x >= 100:
        return "100"
    if x > 99:
        return f"{math.floor(x * 10) / 10:.1f}"
    if x >= 10:
        return f"{x:.0f}"
    if x >= 0.1:
        return f"{x:.1f}"
    return "&lt;0.1"


index_text = num



def pill(level, label=None):
    c, soft, border, _ = U.LEVEL[level]
    icon = "flame" if level in R.RISK_LEVELS and level != "low" else ("shield-check" if level == "low" else "info")
    return (f'<span class="mm-sev" style="background:{soft};color:{c}">{U.icon(icon, size=14)}'
            f'{label or R.RISK_LEVELS[level]["en"] + " risk"}</span>')


def gauge(x, level, label=None, small="/ 100"):
    """The score ring. x=None: not scored (neutral, no number). label: a pill text other than '<level> risk'."""
    c1, c2 = U.LEVEL[level][3]
    off = 251.2 * (1 - (max(min(x, 100), 1.5) if x else 0) / 100)
    return ('<div class="mm-gauge"><div class="g"><svg viewBox="0 0 100 100">'
            f'<defs><linearGradient id="gg{level}" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{c1}"/>'
            f'<stop offset="1" stop-color="{c2}"/></linearGradient></defs>'
            '<circle cx="50" cy="50" r="40" stroke="#1E293B" stroke-width="8" fill="none"/>'
            f'<circle cx="50" cy="50" r="40" stroke="url(#gg{level})" stroke-width="8" stroke-dasharray="251.2" '
            f'stroke-dashoffset="{off:.1f}" stroke-linecap="round" fill="none"/></svg>'
            f'<div><strong>{"—" if x is None else index_text(x)}</strong><small>{small}</small></div></div>'
            f'{pill(level, label)}</div>')


def html_table(frame, n=5):
    head = "".join(f"<th>{esc(c)}</th>" for c in frame.columns)
    body = "".join("<tr>" + "".join(f"<td>{'' if pd.isna(v) else esc(v)}</td>" for v in row) + "</tr>"
                   for row in frame.head(n).itertuples(index=False))
    return f'<div class="mm-tscroll"><table class="mm-t"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def go(step):
    st.session_state["step"] = step
    st.rerun()


def back_button(label, step, key):
    if st.button(f"← {label}", key=key):
        go(step)


def read_csv(raw):
    """Keeps leading zeros (phones, IDs): a column is numeric only if no value starts with 0."""
    df = pd.read_csv(io.BytesIO(raw), dtype=str, skipinitialspace=True)
    for c in df.columns:
        v = df[c].dropna().astype(str).str.strip()
        if len(v) and v.str.fullmatch(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?").all():
            df[c] = pd.to_numeric(df[c])
    return df


@st.cache_data(show_spinner=False, max_entries=8)
def load_table(raw):
    """Finds where the table starts (title and note lines are skipped), then decides what one row is."""
    df, info = DP.read_table(raw)
    return df, info, DP.dataset_type(df, info["notes_top"] + info["notes_bottom"])


def row_kind(prof):
    """'aggregate' only when detected with no sign of personal data, or when the user says so."""
    choice = st.session_state.get("row_kind") or "auto"
    if choice == "stat":
        return "aggregate"
    if choice == "person":
        return "records"
    return "aggregate" if prof["kind"] == "aggregate" else "records"


def skipped_text(info):
    n_top, n_bottom = len(info["notes_top"]), len(info["notes_bottom"])
    if not n_top and not n_bottom:
        return ""
    parts = ([f"{n_top} line{'s' if n_top > 1 else ''} above the table"] if n_top else []) + \
            ([f"{n_bottom} note line{'s' if n_bottom > 1 else ''} under it"] if n_bottom else [])
    return f"Skipped {' and '.join(parts)} (title, description or notes)."


def level_text(lv):
    t = lv["en"]
    return t[0].lower() + t[1:] if t[:1].isascii() else t


def changed_columns(p):
    """Indirect identifiers the safer copy makes less exact (not removed)."""
    return {c: lv for c, lv in p["levels"].items()
            if c not in p["hidden_columns"] and lv["en"] not in (R.RAW["en"], R.HIDE["en"])}


def extra_line(name, e):
    """One extra column made safer: exact values with rare cells blanked read as 'kept', not 'less exact'."""
    if e.get("level_index") == 0 and e.get("blanked"):
        return f"Kept {name} as it is, except {e['blanked']:,} rare values blanked because they would stand out"
    return f"Made {name} less exact: {level_text(e['level'])}"


def changes(p, wrap=lambda c: c, sep=", "):
    """What the safer copy changes, in everyday words (one line per technique)."""
    out = []
    ex = p.get("extras", {})
    if p["dropped_identifiers"]:
        out.append("Removed columns that name people or number the rows: " + sep.join(map(wrap, p["dropped_identifiers"])))
    for c, lv in p.get("masked", {}).items():
        out.append(f"Masked {wrap(c)}: {level_text(lv)} (kept only because it no longer points to anyone)")
    for c, lv in changed_columns(p).items():
        out.append(f"Made {wrap(c)} less exact: {level_text(lv)}")
    for c, e in ex.items():
        if e["result"] == "generalized":
            out.append(extra_line(wrap(c), e))
    if p["hidden_columns"]:
        out.append("Removed columns that still pointed to people: " + sep.join(map(wrap, p["hidden_columns"])))
    # identifiers that could not be masked safely are already listed with the removed identifiers above
    gone = [c for c, e in ex.items() if e["result"] == "removed" and e["kind"] not in ("free_text", "masked")]
    if gone:
        out.append("Removed columns that would still single people out, even made less exact: " + sep.join(map(wrap, gone)))
    text = [c for c, e in ex.items() if e["kind"] == "free_text"]
    if text:
        out.append("Removed free text, which can hold names or details nobody checked: " + sep.join(map(wrap, text)))
    if p["suppressed_records"]:
        out.append(f"Left out {p['suppressed_records']:,} rows ({p['suppressed_pct']:.1f}%) whose details still stood out")
    return out


def pure_private(report, unchanged_only=False):
    r = report["roles"]
    out = [c for c in r["sensitive"] if c not in r["qi"] and c not in r["direct"] and c not in r["record"]]
    if unchanged_only:
        out = [c for c in out if report["protection"]["column_status"].get(c) == "kept"]
    return out


def utility_line(p):
    u = p["utility_summary"]
    c = u["columns"]
    bits = [f"{c['kept']} kept as they are", f"{c['generalized']} made less exact"]
    bits += [f"{c['masked']} masked"] if c["masked"] else []
    bits += [f"{c['removed']} removed"]
    return (f"Columns: {', '.join(bits)} (of {u['columns_total']}). Rows kept: {u['rows_kept']:,} of {u['rows_total']:,}. "
            f"Values unchanged: {u['cells_unchanged_pct']:g}%.")


LIMITS = ("The index assumes someone knows the main revealing details of a person and is sure the person is in "
          "the file; every other column is checked as one extra detail at a time. Someone who knows several extra "
          "details at once, or holds other data, may do better. It is an index, not the chance that an attack "
          "succeeds, and a lower number does not make the copy anonymous or guaranteed safe.")


SHARE_WORDS = {"public": "public release", "partner": "sharing with a trusted partner", "internal": "internal use"}


def name_list(cols, n=3):
    """'a, b and c' (with 'and N more' after n names), each name tagged."""
    n = len(cols) if len(cols) == n + 1 else n  # "a, b, c and d" reads better than "a, b, c and 1 more"
    shown = [tag(c) for c in cols[:n]]
    if len(cols) > n:
        return ", ".join(shown) + f" and {len(cols) - n} more"
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]


def risk_summary(report, context):
    """Privacy risk summary of THIS file: the strongest verified drivers, each said once, only from report values."""
    r, b = report["roles"], report["before"]
    d = report.get("before_details", b)  # the revealing details alone (identifiers are said separately)
    level = R.RISK_LEVELS[b["risk_level"]]["en"].lower()
    head = f"Your file is at {level} risk" + (" overall, but" if b["risk_level"] == "low" else ":")
    record = [c for c in r["record"] if c not in r["direct"]]
    ids = list(r["direct"]) + record
    combos = report["risky_combinations"]
    combo = [c for c in (combos[0]["columns"] if combos else r["qi"][:3]) if c not in ids]
    out = []
    if combos and combo:
        on = f"{name_list(combo)} alone" if len(combo) == 1 else f"just {name_list(combo)} together"
        out.append(f"{head} {num(combos[0]['unique_pct'])}% of its rows are one of a kind on {on}. "
                   f"That does not confirm who they are, but anyone who already knows "
                   f"{'this detail' if len(combo) == 1 else 'these details'} about someone could find their row.")
    elif combo and d["below_k_pct"] > 0:
        out.append(f"Your file is at {level} risk: no row is one of a kind on its revealing details, but {num(d['below_k_pct'])}% of rows share "
                   f"{name_list(combo)} with fewer than {report['thresholds']['k']} rows, the minimum for "
                   f"{SHARE_WORDS.get(context, 'sharing')}.")
    if ids:
        lead = "It also has" if out else f"Your file is at {level} risk because it has"
        out.append(f"{lead} identifiers that point to a person or a record directly: {name_list(ids, 4)}.")
    if not out:
        out.append(f"Your file is at {level} risk: no row stands out on the details checked.")
    private = pure_private(report)
    same = sorted(((lv["homogeneous_groups_records_pct"], s) for s, lv in d["l_diversity"].items()
                   if s in private and lv["homogeneous_groups_records_pct"] >= 1), reverse=True)
    if same:
        out.append(f"For {num(same[0][0])}% of rows, everyone with the same details also has the same {tag(same[0][1])}, "
                   f"so it can be learned even without finding the exact row.")
    elif private and (combos or ids):
        out.append(f"A row that is found would also reveal private information: {name_list(private)}.")
    return " ".join(out)


# The protection plan, grouped by action. The Result page shows it as what to do, the Safer copy page as what was done:
# both read the same plan from the risk engine, so they always agree with the downloaded file.
ACTIONS = {  # key: (to do, done, why, colour)
    "ids": ("Remove", "Removed", "They point to a person or number the rows.", "direct"),
    "mask": ("Mask", "Masked", "Only the part that no longer points to anyone is kept.", "combo"),
    "generalize": ("Make less exact", "Made less exact", "So these details are shared by more people.", "combo"),
    "hide": ("Remove", "Removed", "Even made less exact, they would still single people out.", "direct"),
    "text": ("Remove", "Removed", "Free text can hold names or details nobody checked.", "direct"),
    "other": ("Change", "Changed", "Part of the protection plan.", "combo"),
    "rows": ("Leave out rows", "Left out rows", "Their details still stood out after the other changes.", "combo"),
    "private": ("Share only if needed", "Kept unchanged", "Private information: harder to link to a person, but not hidden.",
                "private"),
}


def protection_plan(report):
    """[(action key, [(column, how) ...], note)] from the plan the risk engine made (report["protection"])."""
    p = report["protection"]
    ex = p.get("extras", {})
    groups = []

    def add(key, items, note=""):
        if items or note:
            groups.append((key, items, note))
    add("ids", [(c, "") for c in p["dropped_identifiers"]])
    add("mask", [(c, level_text(lv)) for c, lv in p.get("masked", {}).items()])
    gen = [(c, level_text(lv)) for c, lv in changed_columns(p).items()]
    for c, e in ex.items():
        if e["result"] == "generalized":
            gen.append((c, f"{e['blanked']:,} rare values blanked"
                        if e.get("level_index") == 0 and e.get("blanked") else level_text(e["level"])))
    add("generalize", gen)
    add("hide", [(c, "") for c in p["hidden_columns"]] +
        [(c, "") for c, e in ex.items() if e["result"] == "removed" and e["kind"] not in ("free_text", "masked")])
    add("text", [(c, "") for c, e in ex.items() if e["result"] == "removed" and e["kind"] == "free_text"])
    seen = {c for _, items, _ in groups for c, _ in items}
    add("other", [(c, s) for c, s in p["column_status"].items() if s != "kept" and c not in seen])
    if p["suppressed_records"]:
        add("rows", [], f"{p['suppressed_records']:,} of {p['utility_summary']['rows_total']:,} rows "
                        f"({p['suppressed_pct']:.1f}%)")
    add("private", [(c, "") for c in pure_private(report, unchanged_only=True)])
    return groups


def plan_list(report, done, private_note=""):
    """One list: an action, the columns it applies to (with how), and why in a few words.
    Every removal is one row (with a line per reason), so each action appears once."""
    rows = {}  # action label -> html parts, in first-seen order
    for key, items, note in protection_plan(report):
        todo, past, why, kind = ACTIONS[key]
        cols = "".join(f'<span class="mm-ci">{tag(c)}{f"<em>{esc(how)}</em>" if how else ""}</span>' for c, how in items)
        part = (f'<div class="cols">{cols}</div>' if cols else "") + (f"<b>{esc(note)}</b>" if note else "") + \
            f'<p>{why}{private_note if key == "private" else ""}</p>'
        label = past if done else todo
        rows.setdefault((label, kind), []).append(part)
    html_rows = [f'<div class="mm-rec"><span class="mm-kind k-{kind}">{label}</span><div>{"".join(parts)}</div></div>'
                 for (label, kind), parts in rows.items()]
    return '<div class="mm-glass mm-recs">' + "".join(html_rows) + "</div>" if html_rows else ""


def column_editor():
    """Qwen's column labels, which the user can correct; the result is measured again with the changes."""
    with st.expander("Check what each column is"):
        st.caption("Ma'man filled this in; fix a tick if it is wrong. The result is measured again with your changes.")
        ver = st.session_state.get("roles_ver", 0)
        new = st.data_editor(
            st.session_state["roles_df"], hide_index=True, width="stretch", disabled=["Column"],
            column_config={
                "Names a person": st.column_config.CheckboxColumn(
                    help="Points to one person on its own: name, national ID, phone, email, IBAN"),
                "Reveals when combined": st.column_config.CheckboxColumn(
                    help="Harmless alone, but together can point to someone: age, gender, district, job, dates"),
                "Private": st.column_config.CheckboxColumn(help="Would hurt the person if it got out: health, income"),
                "Row or system number": st.column_config.CheckboxColumn(help="A row number or a file number"),
            }, key=f"editor_{file_key}_{ver}")
        if not new.equals(st.session_state["roles_df"]):
            st.session_state["roles_df"] = new
            st.session_state["roles_ver"] = ver + 1
            st.rerun()


def visible_metrics(protected_csv, report, context):
    """Measure the safer copy again from the exact CSV that will be downloaded."""
    copy = read_csv(protected_csv)
    p = report["protection"]
    qi = [c for c in report["roles"]["qi"] if c in copy.columns and c not in p["hidden_columns"]]
    for c in qi:
        copy[c] = R._as_text(copy[c])
    sens = [c for c in report["roles"]["sensitive"] if c in copy.columns]
    ctx = R.CONTEXTS[context]
    m = R.risk_metrics(copy, qi, R.l_columns(copy, qi, sens), ctx)
    extra = [c for c, e in p.get("extras", {}).items() if e["kind"] != "free_text" and c in copy.columns]
    residual = R.residual_risk(copy, qi, extra, ctx)
    return {**m, **R.risk_level(m, ctx, 0), "residual": residual}, qi


def final_report(report, after):
    """The report as it applies to the saved safer copy: 'after' is measured again on that exact file."""
    return {**report, "after": after} if after else report


def retriever_note():
    """Which retriever found the example columns for Qwen — shown, never assumed (a fallback is visible here)."""
    used = st.session_state.get("retriever_used") or []
    if not used:
        return "Example columns: none needed (every column was settled by the rules)."
    names = {"e5": "multilingual E5 (intfloat/multilingual-e5-small)", "char": "character n-grams (fallback or rollback)"}
    return "Example columns for Qwen were found with: " + ", ".join(names.get(u, u) for u in used) + "."


def report_text(report, after, atk, df_rows):
    from explain_report import template
    b, p = report["before"], report["protection"]
    kind_note = {"mixed": "What one row is: checked as one person (signs of totals too; please review).",
                 "uncertain": "What one row is: checked as one person (could not confirm a summary table; please review)."}
    lines = ["Ma'man privacy report", "=" * 21, "", f"Sharing assumed: {report['context_en']}",
             *([kind_note[prof["kind"]]] if prof["kind"] in kind_note and (st.session_state.get("row_kind") or "auto") == "auto" else []),
             *([skipped_text(info)] if skipped_text(info) else []),
             f"Re-identification risk index (0-100), original file: {num(b['avg_reidentification_risk_pct'])} "
             f"({b['risk_level_en']} risk)"]
    if after:
        lines += [f"Re-identification risk index (0-100), safer copy (measured on the saved file): "
                  f"{num(after['avg_reidentification_risk_pct'])} ({after['risk_level_en']} risk)",
                  f"Rows kept: {after['records']:,} of {df_rows:,}"]
    if after and after.get("residual", {}).get("worst_column"):
        rs = after["residual"]
        lines.append(f"If someone also knew one more detail (worst: {rs['worst_column']}), the safer copy's index "
                     f"would be {num(rs['worst_index'])}.")
    if after:
        lines.append(utility_line(p))
    lines += ["", "Main revealing details assumed known: " + (", ".join(report["roles"]["qi"]) or "none"),
              "Limits: " + LIMITS, ""]
    r = report["roles"]
    if r["direct"]:
        lines.append("Columns that name people: " + ", ".join(r["direct"]))
    record = [c for c in r["record"] if c not in r["direct"]]
    if record:
        lines.append("Row or record numbers: " + ", ".join(record))
    if report.get("identifiers_demoted"):
        lines.append("Protected as combined details (each value is shared by many people): "
                     + ", ".join(report["identifiers_demoted"]))
    if report.get("names_found"):
        lines.append("Treated as names (found by their header and values): " + ", ".join(report["names_found"]))
    lines.append("Riskiest combinations in the original file:")
    lines += [f"- {num(c['unique_pct'])}% of people are one of a kind on: " + " + ".join(c["columns"])
              for c in report["risky_combinations"]]
    lines += ["", "Changes made to the safer copy:"] + [f"- {x}" for x in changes(p)]
    if pure_private(report, unchanged_only=True):
        lines.append("Private columns kept unchanged: " + ", ".join(pure_private(report, unchanged_only=True)))
    words = {"kept": "kept as is", "generalized": "less exact", "masked": "masked", "removed": "removed"}
    lines += ["", "Column by column:"] + [f"- {c}: {words[v]}" for c, v in p["column_status"].items()]
    if atk and atk["known_columns"]:
        def res(x):
            t = f"{x['reidentified']} matched exactly one row"
            if atk["sensitive"]:
                t += f"; their {atk['sensitive']} could be read with certainty for {x['sensitive_learned']}"
            return t
        lines += ["", f"Matching test on {atk['targets']} people (someone knowing {', '.join(atk['known_columns'])}):",
                  f"- names removed only: {res(atk['names_removed'])}",
                  f"- safer copy: {res(atk['protected'])}",
                  "- No exact match does not mean nothing is revealed: a private value can still be read when",
                  "  everyone in a person's group shares it."]
    lines += ["", retriever_note()]
    lines += ["", "Summary (written from the changes made and the measurements above):", template(final_report(report, after), "en"),
              "", "Ma'man is a decision-support tool. It does not replace legal review."]
    return "\n".join(lines)


def reset(*keys):
    for k in keys:
        st.session_state.pop(k, None)


def review_note(prof):
    """When the file may be a summary table but that is not certain: say so, and say it was protected anyway."""
    if prof["kind"] in ("mixed", "uncertain") and (st.session_state.get("row_kind") or "auto") == "auto":
        signs = prof["aggregate_signs"][:2] + prof["person_signs"][:2]
        md('<div class="mm-review"><b>Please check what one row is.</b> '
           + ("Ma'man found signs of both personal rows and totals:" if prof["kind"] == "mixed" else
              "Parts of this file look like a summary table, but Ma'man could not confirm it:")
           + "<ul>" + "".join(f"<li>{chips(x)}</li>" for x in signs) + "</ul>"
           "To be safe it checked every row as a person.</div>")


def statistics_text(df, info, prof, context, small):
    k = R.CONTEXTS[context]["k"]
    lines = ["Ma'man privacy report", "=" * 21, "", f"Sharing assumed: {R.CONTEXTS[context]['label']['en']}",
             "What one row is: a total or a statistic (summary table), not a person.",
             ("Person-level exposure: 0 / 100 (no row describes a person; this is not a re-identification index and "
              "does not cover other confidentiality questions)." if not small else
              f"Person-level exposure: not scored (some totals are counts below {k})."), ""]
    if skipped_text(info):
        lines += [skipped_text(info)] + [f"  - {t}" for t in info["notes_top"] + info["notes_bottom"]] + [""]
    lines += ["Why:"] + [f"- {r}" for r in prof["aggregate_signs"]]
    lines += ["- No names, phone numbers, emails, ID numbers or IBANs were found in the values.", ""]
    if small:
        lines += [f"Totals below {k} to check before sharing (a count of 1 or 2 can point to the people behind it):"]
        lines += [f"- {x['where'] or 'row ' + str(x['row'] + 1)} | {x['column']}: {x['value']}" for x in small]
        lines += [f"Recommendation: combine small groups, or show them as 'fewer than {k}'."]
    else:
        lines += [f"No count below {k} was found."]
    lines += ["", "No changes were made: removing or blurring columns of totals would only lose information.",
              "", "Ma'man is a decision-support tool. It does not replace legal review."]
    return "\n".join(lines)


def classify_all(df, name, box):
    """Screen 3: Qwen labels each column (one at a time, on the model thread); then code adds proxy columns.
    The radar, the stages and the log show the real progress, column by column, drawn into `box`."""
    log = [f"Read {len(df):,} rows and {len(df.columns)} columns"]

    def draw(i, stage):
        pct = 6 + 88 * i / max(1, len(df.columns)) if stage < 3 else 100
        s = lambda k, t: (f'<span class="st-on">{U.icon("check" if k < stage else "circle", size=15)}{t}</span>' if k <= stage
                          else f'<span class="st-off">{U.icon("circle", size=15)}{t}</span>')
        box.markdown(
            '<div class="mm-scan"><div class="mm-radar"><div class="ring"></div><div class="sweep"></div>'
            f'<div class="core">{U.icon("file-search", size=40)}</div></div>'
            '<h2>Analyzing your file…</h2>'
            f'<p class="sub">Column {min(i + 1, len(df.columns))} of {len(df.columns)}</p>'
            f'<div class="mm-stages"><div class="row">{s(1, "Reading the table")}{s(2, "Labelling columns")}'
            f'{s(3, "Measuring risk")}</div><div class="mm-track"><span style="width:{pct:.0f}%"></span></div></div>'
            '<div class="mm-term">' + "".join(f"<div>{esc(x)}</div>" for x in log[-5:]) + "</div></div>",
            unsafe_allow_html=True)

    rows, used = [], set()
    draw(0, 1)
    try:
        for i, col in enumerate(df.columns):
            draw(i, 2)
            res = classify_column(describe_column(name, col, df[col]))
            if res.get("retriever"):
                used.add(res["retriever"])
            labels = [ROLE_LABELS[k].lower() for k in ROLE_KEYS if res.get(k)]
            log.append(f"{col} → {', '.join(labels) or 'nothing personal'}")
            rows.append({"Column": col, **{ROLE_LABELS[k]: bool(res.get(k)) for k in ROLE_KEYS}})
    except RuntimeError:
        box.empty()
        models().release()
        st.error("Your computer ran low on memory while reading the columns. "
                 "Close other heavy apps or browser tabs, then press Analyze again.")
        st.session_state["step"] = 2
        st.stop()
    log.append("Measuring the risk…")
    draw(len(df.columns), 3)
    roles_df = pd.DataFrame(rows)
    # A column that gives away a private one (e.g. the clinic follows the diagnosis) is private too
    found = R.proxy_columns(df, {
        "direct": roles_df.loc[roles_df["Names a person"], "Column"].tolist(),
        "record": roles_df.loc[roles_df["Row or system number"], "Column"].tolist(),
        "sensitive": roles_df.loc[roles_df["Private"], "Column"].tolist()})
    roles_df.loc[roles_df["Column"].isin(list(found)), "Private"] = True
    st.session_state["roles_df"], st.session_state["proxies"] = roles_df, found
    st.session_state["roles_model"] = roles_df.copy()
    st.session_state["retriever_used"] = sorted(used)


def show_statistics(df, info, prof, context, name):
    """A summary table: nothing to anonymize. Say why, check small totals, and leave the file as it is."""
    k = R.CONTEXTS[context]["k"]
    small = DP.small_counts(df, prof, k)
    user_chose = (st.session_state.get("row_kind") or "auto") == "stat"
    md(f'<div class="mm-head"><div><h2>{U.icon("table", size=24)}Your result</h2>'
       f'<p>What is in this file, before anything is shared</p></div></div>')
    # 0/100 only when the file is read as totals with no sign of personal data AND no small count could point to
    # a few people; otherwise the score is left out (neutral), never shown as a misleading 0.
    score = (gauge(0, "low", "No person-level records", "/ 100") if not small else
             gauge(None, "none", "Not scored", "small counts to check"))
    note = ("<b>0 / 100 is the person-level exposure Ma'man found:</b> no row describes a person, so no row can be "
            "traced back to someone. It is not a re-identification index (that needs one row per person), and it does "
            "not cover other confidentiality or security questions about these figures." if not small else
            f"<b>Not scored:</b> some totals are counts below {k}, which could point to a few people (listed below).")
    md(f'<div class="mm-glass mm-score" style="border-color:{U.LEVEL["low" if not small else "none"][2]}">{score}'
       '<div><div class="mm-eyebrow" style="color:#5EEAD4">Person-level exposure · what is in this file</div>'
       '<div style="font-size:2rem;font-weight:800;color:#fff;margin:.3rem 0 .6rem">Summary statistics</div>'
       '<p style="color:var(--text);font-size:.85rem;line-height:1.6;margin:0">Each row is a total or a measure, not a person. '
       'There is nothing to anonymize, so Ma\'man left your file as it is: removing or blurring these columns would only '
       f'lose information.</p><p class="mm-small" style="margin-top:.5rem">{note}</p>'
       f'<div class="mm-metrics" style="margin-top:1rem"><div class="mm-metric"><div class="t">Small totals to check'
       f'{U.icon("alert-triangle", size=16)}</div><div class="v">{len(small)}</div><div class="d">'
       f'{"Counts below " + str(k) + " that could point to a few people." if small else "No count below " + str(k) + " was found."}'
       '</div></div></div></div></div>')
    md(f'<div class="mm-h2">{U.icon("info", size=17)}Why Ma\'man thinks so</div>')
    why = ([chips(r) for r in prof["aggregate_signs"][:4]] if not user_chose else
           ["This file was set to be read as <b>totals or statistics</b>."] + [chips(r) for r in prof["aggregate_signs"][:3]])
    why.append("No names, phone numbers, emails, ID numbers or IBANs were found in the values.")
    md('<div class="mm-glass" style="padding:.4rem 1.1rem"><ul class="mm-list">'
       + "".join(f"<li><i>✓</i><span>{w}</span></li>" for w in why) + "</ul></div>")
    if user_chose and prof["person_signs"]:
        md('<div class="mm-review"><b>Check this:</b> Ma\'man also found signs of personal rows:<ul>'
           + "".join(f"<li>{chips(r)}</li>" for r in prof["person_signs"][:3]) + "</ul></div>")
    if small:
        items = "".join(f"<li>{tag(x['where']) if x['where'] else 'Row ' + str(x['row'] + 1)} · {tag(x['column'])}: <b>{x['value']}</b></li>"
                        for x in small[:6])
        more = f"<li>and {len(small) - 6} more (in the report)</li>" if len(small) > 6 else ""
        md(f'<div class="mm-caveat"><b>{len(small)} total{"s are" if len(small) > 1 else " is"} below {k}.</b> '
           f'If these count people, a count of 1 or 2 can point to them. Before sharing, combine small groups '
           f'or show them as “fewer than {k}”.<ul>{items}{more}</ul></div>')
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        back_button("Back to preview", 2, "back_stats")
    with c2:
        st.download_button("Download the report (.txt)", statistics_text(df, info, prof, context, small).encode("utf-8-sig"),
                           file_name=f"{name}_privacy_report.txt", mime="text/plain", type="primary", width="stretch")


# ═════════════════════════════════════════════
# The file (kept across screens)
# ═════════════════════════════════════════════
def load_file(raw, name):
    file_key = hashlib.md5(raw).hexdigest()
    if st.session_state.get("file_key") != file_key:
        reset("analyzed", "roles_df", "roles_model", "proxies", "extras", "extras_checked", "result", "safer", "summary",
              "retriever_used", "keep_cols", "strict_on")
        st.session_state["row_kind"] = "auto"
        st.session_state.update(file_key=file_key, raw=raw, name=name, step=2)


if "raw" not in st.session_state and os.environ.get("MAMAN_TEST_FILE"):  # automated tests only (tests/test_app.py)
    load_file(Path(os.environ["MAMAN_TEST_FILE"]).read_bytes(), Path(os.environ["MAMAN_TEST_FILE"]).stem)

if st.session_state["step"] not in range(1, len(STEPS) + 1):  # a session from the old six-step version
    st.session_state["step"] = 4
step = st.session_state["step"]
has_file = "raw" in st.session_state
if has_file:
    raw, name, file_key = st.session_state["raw"], st.session_state["name"], st.session_state["file_key"]
    df, info, prof = load_table(raw)
analyzed = has_file and st.session_state.get("analyzed")
aggregate = has_file and row_kind(prof) == "aggregate"
reachable = {1: True, 2: has_file, 3: step == 3, 4: bool(analyzed), 5: bool(st.session_state.get("safer")) and not aggregate}

# ═════════════════════════════════════════════
# Header: logo · steps
# ═════════════════════════════════════════════
# The logo takes only the room it needs; the step bar stretches over the rest, each step an equal share.
with st.container(key="topbar", horizontal=True, vertical_alignment="center"):
    st.markdown(f'<div class="mm-logo"><img src="{U.LOGO}" alt="Ma\'man مأمن"></div>', unsafe_allow_html=True,
                width="content")
    with st.container(key="navbar", horizontal=True, width="stretch"):
        for i, label in enumerate(STEPS, 1):
            if st.button(f"{i}  {label}", key=f"nav_{i}", width="stretch",
                         disabled=not reachable[i] and i != step) and i != step:
                go(i)


# Every screen is drawn into this one slot: a new screen replaces the old one in place, so nothing from the
# previous screen (such as the file uploader) can stay on the page next to it.
screen = st.empty()


def footer():
    md('<div class="mm-foot"><div class="sadu"></div><div class="bar"><span><b>مأمن • MA\'MAN</b> &nbsp;·&nbsp; '
       'Privacy risk check before you share</span><span>Understand your data. Reduce privacy risks. Share more safely.</span></div></div>')


# ═════════════════════════════════════════════
# 1. Upload
# ═════════════════════════════════════════════
if step == 1 or not has_file:
    with screen.container():
        md('<div class="mm-hero"><span class="mm-pillchip">' + U.icon("shield-check", size=15) +
           'Check your data before sharing</span>'
           '<h1>Know your privacy risk<br><span class="grad">before you share.</span></h1>'
           '<p>Even without names, combinations of personal details can make individuals stand out. '
           'Ma\'man helps you assess the risk and share data more safely.</p></div>')
        with st.container(key="ingest"):
            up = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed",
                                  key=f"uploader_{st.session_state.get('uploader_ver', 0)}")
            if up is not None:
                load_file(up.getvalue(), Path(up.name).stem)
                st.session_state["uploader_ver"] = st.session_state.get("uploader_ver", 0) + 1  # drop the used widget
                st.rerun()
            md('<div class="mm-pillars">'
               f'<div class="mm-glass mm-pillar"><div class="ico ico-teal">{U.icon("shield-alert", size=20)}</div><div>'
               '<h4>Spot the Risk</h4><p>See where your data may expose individuals.</p></div></div>'
               f'<div class="mm-glass mm-pillar"><div class="ico ico-cyan">{U.icon("file-search", size=20)}</div><div>'
               '<h4>Understand Your Data</h4><p>AI identifies personal and sensitive columns.</p></div></div>'
               f'<div class="mm-glass mm-pillar"><div class="ico ico-blue">{U.icon("lock-keyhole", size=20)}</div><div>'
               '<h4>Share More Safely</h4><p>Create a safer version of your dataset.</p></div></div></div>')
        footer()
    st.stop()

# ═════════════════════════════════════════════
# 2. Preview + settings → Analyze
# ═════════════════════════════════════════════
if step == 2:
    with screen.container():
        with st.container(key="narrow_preview"):
            skipped = skipped_text(info)
            md('<div class="mm-glass mm-banner"><div class="l"><div class="dot ico-teal">' + U.icon("check-circle", size=20) + '</div>'
               f'<div><div style="display:flex;gap:.5rem;align-items:center;flex-wrap:wrap"><span class="mm-eyebrow" style="color:#2DD4BF">'
               f'File ready</span><span class="mm-chip">{len(df):,} rows</span></div>'
               f'<div class="mm-fname">{esc(name)}.csv</div></div></div>'
               f'<div class="mm-meta"><span>{len(raw) / 1024:,.1f} KB</span><span>•</span><span>{len(df.columns)} columns</span>'
               f'<span>•</span><span>{len(df):,} rows</span></div></div>'
               + (f'<div class="mm-note">{esc(skipped)}</div>' if skipped else ''))
            md('<div class="mm-glass mm-tablecard"><div class="h"><div><b>' + U.icon("table", size=16) + 'Preview the file</b>'
               '<small>The first 5 rows, before anything is checked</small></div><span class="mm-small mono">CSV</span></div>'
               + html_table(df) + '</div>')
            context = st.session_state["context"]
            md(f'<div class="mm-small" style="margin:1rem 0">Assumes: <b>{esc(AUDIENCE[context].lower())}</b>.</div>')
            c1, c2 = st.columns([1, 1.4])
            with c1:
                if st.button("← Replace file", key="replace"):
                    reset("raw", "name", "file_key", "analyzed", "roles_df", "roles_model", "result", "safer", "summary")
                    go(1)
            with c2:
                if st.button("Analyze", type="primary", width="stretch"):
                    st.session_state["analyzed"] = True
                    reset("safer")
                    go(3 if row_kind(prof) == "records" and "roles_df" not in st.session_state else 4)
        footer()
    st.stop()

# ═════════════════════════════════════════════
# 3. Analyzing (Qwen, column by column)
# ═════════════════════════════════════════════
if step == 3:
    # The progress screen is ONE element drawn straight into the screen slot. It replaces the whole previous screen
    # at once; drawn inside a container instead, the old screen's leftover elements (the preview)
    # would stay on the page under it until the labelling run ends.
    if row_kind(prof) == "records" and "roles_df" not in st.session_state:
        classify_all(df, name, screen)
    go(4)

context = st.session_state["context"]
if aggregate:
    if step != 4:
        go(4)
    with screen.container():
        with st.container(key="wide_stats"):
            show_statistics(df, info, prof, context, name)
        footer()
    st.stop()
if "roles_df" not in st.session_state:  # switched to "One person" after a summary-table result
    go(3)

# ═════════════════════════════════════════════
# What each column is (Qwen's labels, which the user can correct on the Result screen), and the analysis
# ═════════════════════════════════════════════
edited = st.session_state["roles_df"]
roles = {"direct": [], "qi": [], "sensitive": [], "record": []}
for _, r in edited.iterrows():
    if r["Names a person"]:
        roles["direct"].append(r["Column"])
    if r["Row or system number"]:
        roles["record"].append(r["Column"])
    if r["Reveals when combined"] and not r["Names a person"]:
        roles["qi"].append(r["Column"])
    if r["Private"]:
        roles["sensitive"].append(r["Column"])
keep = [c for c in st.session_state.get("keep_cols", []) if c in df.columns]
strict = bool(st.session_state.get("strict_on"))

if not roles["qi"] and not roles["direct"] and not roles["record"]:
    if step != 4:
        go(4)
    with screen.container():
        with st.container(key="narrow_none"):
            md(f'<div class="mm-head"><div><h2>{U.icon("table", size=24)}Your result</h2>'
               f'<p>Measured on your file, before anything is changed</p></div></div>')
            md(f'<div class="mm-glass mm-score" style="border-color:{U.LEVEL["none"][2]}">'
               f'{gauge(None, "none", "Not scored", "no score")}<div>'
               '<div class="mm-eyebrow" style="color:#CBD5E1">Re-identification risk index</div>'
               '<p style="color:var(--text);font-size:.9rem;line-height:1.6;margin:.4rem 0 0">Not measured: no column was '
               'found that could point to a person, so there is nothing to measure it on. This is not a 0. If the rows do '
               'describe people, tick what each column is below and the score is measured.</p></div></div>')
            review_note(prof)
            st.info("Ma'man did not find columns that could point to people. If this looks wrong, check what each "
                    "column is below.")
            column_editor()
            back_button("Back to preview", 2, "back_none_found")
        footer()
    st.stop()
else:
    run_key = json.dumps([file_key, roles, context, sorted(keep), strict], ensure_ascii=False)
    if st.session_state.get("result", (None,))[0] != run_key:
        with st.spinner("Measuring the risk…"):
            report, protected = R.analyze(df, roles, context, None, keep, strict=strict)
        st.session_state["result"] = (run_key, report, protected)
        reset("safer")
    _, report, protected = st.session_state["result"]
    b, p = report["before"], report["protection"]

safer = st.session_state.get("safer")
safer = safer if safer and report is not None and safer[0] == run_key else None
if step == 5 and not safer:
    go(4)

# ═════════════════════════════════════════════
# 4. Result: how risky · why · what to do
# ═════════════════════════════════════════════
if step == 4:
    with screen.container():
        with st.container(key="narrow_result"):
            lv, x = b["risk_level"], b["avg_reidentification_risk_pct"]
            c, soft, border, _ = U.LEVEL[lv]
            md(f'<div class="mm-head"><div><h2>{U.icon("alert-triangle", size=24)}Your result</h2>'
               f'<p>Measured on your file, before anything is changed</p></div></div>')
            status = p["column_status"]
            removed = [c_ for c_, v in status.items() if v == "removed"]
            less_exact = [c_ for c_, v in status.items() if v in ("generalized", "masked")]
            n_protect = len(removed) + len(less_exact)
            parts = [f"{len(less_exact)} to make less exact"] if less_exact else []
            parts += [f"{len(removed)} to remove"] if removed else []
            md(f'<div class="mm-glass mm-score" style="border-color:{border}">{gauge(x, lv)}<div class="mm-metrics">'
               f'<div class="mm-metric"><div class="t">Columns that need protection{U.icon("columns", "", 16)}</div>'
               f'<div class="v">{n_protect} of {len(df.columns)}</div><div class="d" style="color:{c}">'
               f'{" and ".join(parts) or "No changes needed"}</div></div>'
               f'<div class="mm-metric"><div class="t">Rows checked{U.icon("database", "", 16)}</div>'
               f'<div class="v">{len(df):,}</div><div class="d">One row per person</div></div>'
               f'<div class="mm-metric"><div class="t">One of a kind{U.icon("fingerprint", "", 16)}</div>'
               f'<div class="v" style="color:#FBBF24">{num(b["unique_pct"])}%</div><div class="d">Rows whose revealing details '
               f'nobody else has</div></div>'
               f'<div class="mm-metric"><div class="t">Sharing assumed{U.icon("users", "", 16)}</div>'
               f'<div class="v" style="font-size:1.05rem;color:#C4B5FD">{esc(R.CONTEXTS[context]["label"]["en"])}</div>'
               f'<div class="d">{esc(AUDIENCE[context])}</div></div></div></div>')
            md(f'<div class="mm-glass mm-meaning"><div class="ico ico-amber">{U.icon("shield-alert", size=20)}</div><div>'
               f'<h4>Privacy risk summary</h4><p>{risk_summary(report, context)}</p></div></div>')
            review_note(prof)
            plan = plan_list(report, done=False)
            if plan:
                md(f'<div class="mm-h2">{U.icon("list-filter", size=17)}What to do before sharing</div>{plan}')
            st.write("")
            c1, c2 = st.columns([1, 1.4])
            with c1:
                back_button("Back to preview", 2, "back_result")
            with c2:
                if st.button("Create a safer copy", type="primary", width="stretch"):
                    if protected.shape[1] == 0:  # every column had to go: say so instead of handing over an empty file
                        st.session_state["safer"] = (run_key, None, None, None)
                    else:
                        csv_bytes = protected.to_csv(index=False).encode("utf-8-sig")
                        after, qi_after = visible_metrics(csv_bytes, report, context)
                        with st.spinner("Testing the safer copy…"):
                            atk = simulate(df, report["roles"], report, protected)
                        st.session_state["safer"] = (run_key, csv_bytes, after, atk)
                    go(5)
            column_editor()
        footer()
    st.stop()

# ═════════════════════════════════════════════
# 5. Safer copy: before/after · download · what changed
# ═════════════════════════════════════════════
with screen.container():
    with st.container(key="narrow_safer"):
        _, csv_bytes, after, atk = safer
        review_note(prof)
        if csv_bytes is None:
            md('<div class="mm-glass mm-done" style="border-color:rgba(245,158,11,.4) !important"><div class="dot ico-amber">'
               + U.icon("alert-triangle", size=20) + '</div><div><span class="mm-eyebrow" style="color:#FBBF24">Your safer copy</span>'
               f'<h2>Nothing could be kept safely for this audience</h2><p>With only {len(df):,} rows, every column that is '
               f'left would point to people, so the copy would be empty. Share totals instead of rows.</p></div></div>')
            st.write("")
            back_button("Back to result", 4, "back_none")
        else:
            md('<div class="mm-glass mm-done"><div class="dot">' + U.icon("badge-check", size=22) + '</div><div>'
               '<span class="mm-eyebrow" style="color:#34D399">Your safer copy</span>'
               '<h2>A new file; your original file is not changed</h2></div></div>')

            def side(title, m, before):
                c, soft, border, _ = U.LEVEL[m["risk_level"]]
                return (f'<div class="mm-side" style="background:{"#0C1B2E" if before else "#092636"};border-color:{border}">'
                        f'<div><div class="l" style="color:{c if not before else "var(--muted)"}">{title}</div>'
                        f'<strong style="color:{c}">{index_text(m["avg_reidentification_risk_pct"])}<small> / 100</small></strong><br>'
                        f'<span class="tagl" style="background:{soft};color:{c}">{R.RISK_LEVELS[m["risk_level"]]["en"]} risk</span></div>'
                        f'<div class="ring" style="background:{soft};color:{c}">{U.icon("shield-x" if before else "shield-check", size=24)}</div></div>')

            md('<div class="mm-glass mm-ba">' + side("Your file", b, True) + side("Safer copy", after, False) + '</div>')
            checked = "Measured again on the saved file."
            if atk and atk["known_columns"]:
                checked += (f' Matching test on {atk["targets"]} people, assuming someone knows their '
                            f'{name_list(atk["known_columns"])}: <b>{atk["names_removed"]["reidentified"]}</b> matched exactly one '
                            f'row with names removed only, <b>{atk["protected"]["reidentified"]}</b> in the safer copy.')
            md(f'<div class="mm-small" style="margin:.6rem 0 1rem">{checked}</div>')
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("Download safer copy (CSV)", csv_bytes, file_name=f"{name}_safer_copy.csv",
                                   mime="text/csv", type="primary", width="stretch")
            with c2:
                st.download_button("Download the full report (.txt)",
                                   report_text(report, after, atk, len(df)).encode("utf-8-sig"),
                                   file_name=f"{name}_privacy_report.txt", mime="text/plain", width="stretch")
            learned = atk["protected"]["sensitive_learned"] if atk and atk.get("sensitive") else 0
            private_note = (f' In the matching test, the {tag(atk["sensitive"])} of <b>{learned}</b> of {atk["targets"]} people '
                            f'could still be read: everyone in their group has the same value.' if learned else '')
            md(f'<div class="mm-h2">{U.icon("check-check", size=17)}What changed in your file?</div>'
               + plan_list(report, done=True, private_note=private_note))
            rs = after.get("residual", {})
            limit = ("A lower index makes people harder to single out; it does not make the copy anonymous or guaranteed safe.")
            if rs.get("worst_column"):
                limit += (f' If someone also knew {tag(rs["worst_column"])}, the index of the copy would be '
                          f'<b>{num(rs["worst_index"])}</b>.')
            md(f'<div class="mm-small" style="margin:.8rem 0 1rem">{limit}</div>')
            with st.expander("Preview the safer copy (original → safer, side by side)"):
                first = protected.index[:5]
                rows_ = []
                for c_ in df.columns:
                    st_ = p["column_status"].get(c_, "kept")
                    if st_ == "kept":
                        continue
                    o = df.loc[first[0], c_] if len(first) else ""
                    s = protected.loc[first[0], c_] if c_ in protected.columns and len(first) else "(removed)"
                    word = {"removed": "Removed", "masked": "Masked", "generalized": "Less exact"}[st_]
                    rows_.append(f'<tr><td><small>{esc(c_)}</small>{"" if pd.isna(o) else esc(o)}</td><td>→</td>'
                                 f'<td class="safe"><small>Safer copy</small>{"" if pd.isna(s) else esc(s)}</td>'
                                 f'<td><span class="mm-defense">{word}</span></td></tr>')
                md('<div class="mm-tscroll"><table class="mm-t mm-diff"><thead><tr><th>Your file (first row)</th><th></th>'
                   '<th>Safer copy</th><th>Change</th></tr></thead><tbody>' + "".join(rows_) + '</tbody></table></div>')
                st.dataframe(protected.head(10), width="stretch", hide_index=True)
            back_button("Back to result", 4, "back_safer")
    footer()
