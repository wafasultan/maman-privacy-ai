"""
Ma'man — automated checks (no Qwen needed; a stand-in uses the approved labels).
Run:  .venv-mlx/bin/python tests/test_app.py        (or double-click tests/run_tests.command)

1. Reading + "what is one row" on every file in data/ and tests/files/ (expected type for each).
2. The whole app flow (Upload → Analyze → result → safer copy) for each file, with no errors.
3. The downloaded safer copy (the CSV bytes) gives the same index the app shows, holds exactly the columns
   the report lists, and keeps no identifier unmasked. Privacy of every copy: tests/test_protection.py.
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dataset_profile import dataset_type, read_table  # noqa: E402
import risk_engine as R  # noqa: E402
import ui_style as U  # noqa: E402

EXPECTED = {
    "sdaia_ehsan_donations.csv": "aggregate", "edge_wide_crosstab.csv": "aggregate",
    "edge_long_stats_gender.csv": "aggregate", "edge_pivot_export.csv": "aggregate",
    "edge_stats_with_email.csv": "mixed", "edge_personal_total_row.csv": "mixed",
    "edge_personal_vague_headers.csv": "uncertain", "sdaia_budget_2022.csv": "aggregate",
}
FILES = sorted((ROOT / "data" / "synthetic").glob("*.csv")) + sorted((ROOT / "data" / "holdout").glob("*.csv")) + \
    [ROOT / "data" / "adult_preprocessed.csv"] + sorted((ROOT / "tests" / "files").glob("*.csv"))
fails = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        fails.append(msg)


print("1. What one row is")
for f in FILES:
    df, info = read_table(f.read_bytes())
    kind = dataset_type(df, info["notes_top"] + info["notes_bottom"])["kind"]
    check(kind == EXPECTED.get(f.name, "records"), f"{f.name}: {kind} (expected {EXPECTED.get(f.name, 'records')})")

print("\n0. Retriever (the app's default and its fallback)")
from maman_classifier import default_retriever  # noqa: E402
from maman_common import Retriever, load_labeled  # noqa: E402
_kb = load_labeled({"train", "valid"})
check(default_retriever() == "e5", f"app default retriever = {default_retriever()} (expected e5)")
try:
    _r = Retriever(_kb, backend="e5", strict=True)
    check(_r.backend == "e5", "multilingual E5 loads in the app environment")
except Exception as _e:
    check(False, f"multilingual E5 loads in the app environment — {type(_e).__name__}: {_e}")
_bad = Retriever(_kb, backend="e5", e5_model="not-a-real/model-name")
check(_bad.backend == "char" and bool(_bad.fallback_reason), "a broken E5 falls back to char n-grams and records why")

from streamlit.testing.v1 import AppTest  # noqa: E402

os.environ["MAMAN_FAKE_MODEL"] = "1"


def text_of(at):
    return "\n".join(m.value for m in at.markdown)


def run_app(script, f, row_kind=None, safer=True):
    os.environ["MAMAN_TEST_FILE"] = str(f)
    at = AppTest.from_file(str(script), default_timeout=300)
    at.run()
    if row_kind:
        at.session_state["row_kind"] = row_kind
    [b for b in at.button if b.label == "Analyze"][0].click().run()
    for _ in range(3):  # the app reruns itself once after adding columns people may know
        at.run()
    if safer and any(b.label == "Create a safer copy" for b in at.button):
        [b for b in at.button if b.label == "Create a safer copy"][0].click().run()
    return at


def numbers(at):
    """Risk numbers and counts shown in the result and safer-copy cards."""
    t = text_of(at)
    t = t[t.find("Your result"):] if "Your result" in t else t
    return re.findall(r"<strong[^>]*>([\d.]+)|(\d[\d,]*) of ([\d,]+) rows kept|<div class=\"v\">(\d+)<small> of (\d+)", t)


print("\n2. App flow (new app)")
results = {}
for f in FILES:
    at = run_app(ROOT / "app.py", f)
    errs = [e.value for e in at.exception]
    t = text_of(at)
    kind = EXPECTED.get(f.name, "records")
    if kind == "aggregate":
        ok = not errs and "Summary statistics" in t and "Create a safer copy" not in [b.label for b in at.button]
        check(ok, f"{f.name}: summary-table result, nothing anonymized" + (f" — {errs[0][:200]}" if errs else ""))
        zero = "No person-level records" in t and "person-level exposure" in t and "Not scored" not in t
        check(not errs and (zero or ("Not scored" in t and "small counts to check" in t)),
              f"{f.name}: {'0 / 100 person-level exposure (green)' if zero else 'not scored: small counts to check'}")
    else:
        ok = not errs and ("Your safer copy" in t or "did not find columns" in t.lower() or
                           any("did not find columns" in i.value for i in at.info))
        if kind in ("mixed", "uncertain"):
            ok = ok and "Please check what one row is" in t
        check(ok, f"{f.name}: personal-rows result + safer copy" + (" + review note" if kind != "records" else "")
              + (f" — {errs[0][:200]}" if errs else ""))
    if not errs and "result" in at.session_state:
        if "safer" in at.session_state and at.session_state["safer"][1] is None:  # "no copy" page: read the result page
            at = run_app(ROOT / "app.py", f, safer=False)
            t = text_of(at)
        rb = at.session_state["result"][1]["before"]
        lv = R.level_for_index(rb["avg_reidentification_risk_pct"])
        check(rb["risk_level"] == lv and f'{R.RISK_LEVELS[lv]["en"]} risk' in t and U.LEVEL[lv][0] in t,
              f"{f.name}: score {rb['avg_reidentification_risk_pct']:.1f}, label and colour agree ({lv})")
    elif not errs and kind != "aggregate":
        check("Not scored" in t, f"{f.name}: no column points to people → not scored (neutral), not a 0")
    results[f.name] = numbers(at)
    labels = [b.label for b in at.button]
    if not errs and "Your safer copy" in t:
        check(len(at.get("file_uploader")) == 0 and not any("Details" in x or "sample" in x.lower() for x in labels)
              and t.count("What changed in your file?") == (1 if "your original file is not changed" in t else 0) and len([x for x in labels if x[:1].isdigit()]) == 5,
              f"{f.name}: five steps, no Details or samples, one 'What changed' section, no uploader after upload")

print("\n   Override: Ehsan file, user says 'One person'")
at = run_app(ROOT / "app.py", ROOT / "tests/files/sdaia_ehsan_donations.csv", row_kind="person")
check(not at.exception and "Summary statistics" not in text_of(at), "runs the personal-rows check instead")
print("   Override: hospital file, user says 'A total or statistic'")
at = run_app(ROOT / "app.py", ROOT / "data/synthetic/sa_hospital.csv", row_kind="stat")
check(not at.exception and "Summary statistics" in text_of(at) and "also found signs of personal rows" in text_of(at),
      "shows the summary view with a warning about personal signs")

print("   Result page: one summary and one list of what to do")
at = run_app(ROOT / "app.py", ROOT / "tests/files/privacy_test_25_columns.csv", safer=False)
t = text_of(at)
check(not at.exception and t.count("Privacy risk summary") == 1 and t.count("What to do before sharing") == 1
      and "What does this score mean" not in t and "Why it is risky" not in t, "privacy_test_25_columns.csv: result sections")

print("\n3. The downloaded safer copy matches what the app shows (counted here from the CSV bytes)")
import csv as _csv  # noqa: E402
import io as _io  # noqa: E402
from collections import Counter as _Counter  # noqa: E402
for f in FILES:
    if EXPECTED.get(f.name, "records") == "aggregate":
        continue
    at = run_app(ROOT / "app.py", f)
    safer = at.session_state["safer"] if "safer" in at.session_state else None
    if not safer or safer[1] is None:
        check(not at.exception, f"{f.name}: no copy offered (nothing could be kept safely)")
        continue
    _, csv_bytes, after, _atk = safer
    _, report, _prot = at.session_state["result"]
    rows = list(_csv.DictReader(_io.StringIO(csv_bytes.decode("utf-8-sig"))))
    cols = list(rows[0].keys()) if rows else []
    core = [c for c in report["roles"]["qi"] if c in cols]
    mine = 100 * len(_Counter(tuple((r[c] or "?") for c in core) for r in rows)) / len(rows) if rows else 0
    shown = after["avg_reidentification_risk_pct"]
    status = report["protection"]["column_status"]
    mismatch = [c for c, st in status.items() if (st == "removed") == (c in cols)]
    leaked = [c for c in report["roles"]["direct"] if c in cols and c not in report["protection"]["masked"]]
    check(abs(mine - shown) < 0.01 and not mismatch and not leaked and len(rows) == after["records"],
          f"{f.name}: index {shown:.2f} = {mine:.2f} from the CSV · {len(cols)} columns as reported · {len(rows)} rows"
          + (f" — status mismatch {mismatch}" if mismatch else "") + (f" — identifiers kept {leaked}" if leaked else ""))

print(f"\n{'ALL PASSED' if not fails else str(len(fails)) + ' FAILED'}")
sys.exit(1 if fails else 0)
