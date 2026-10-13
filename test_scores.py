"""
Ma'man — the score, its level and colour, and the safer-copy fixes of 13 Oct (no Qwen needed).
Run:  .venv-mlx/bin/python tests/test_scores.py      (also run by tests/run_tests.command)

1. Levels: low ≤ 10 < medium ≤ 33.3 < high ≤ 66.7 < critical, read from the index alone; one file per level,
   the index counted again here without engine code.
2. Direct identifiers count among the details someone could know (a file with names scores high).
3. Files without person-level records (SDAIA budget 2022): read as a summary table, no small counts → 0 / 100.
4. Safer copy fixes: no revealing detail at all (only an ID) no longer crashes; a private column with one value
   for everyone (or empty everywhere) no longer leaves out every row; the original table is never changed.
5. review_sample.csv with the labels the real model gave (13 Oct run): 30 rows kept (was 0).
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pandas as pd  # noqa: E402

import dataset_profile as DP  # noqa: E402
import risk_engine as R  # noqa: E402

fails = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        fails.append(msg)


def roles(direct=(), qi=(), sensitive=(), record=()):
    return {"direct": list(direct), "qi": list(qi), "sensitive": list(sensitive), "record": list(record)}


def index_by_hand(df, cols):
    keys = [tuple("?" if pd.isna(v) else str(v).strip() for v in row) for row in df[cols].itertuples(index=False)]
    return 100 * len(Counter(keys)) / len(keys)


print("1. One file per level (40 rows; groups of 20, 4, 2 and 1 on area + sex)")
for size, want in [(20, "low"), (4, "medium"), (2, "high"), (1, "critical")]:
    area = [f"A{i // max(size, 2)}" for i in range(40)]
    sex = ["M", "F"] * 20 if size == 1 else ["M"] * 40  # groups of 2 on area, split by sex → everyone unique
    df = pd.DataFrame({"area": area, "sex": sex, "visits": [i % 3 for i in range(40)]})
    rep, prot = R.analyze(df, roles(qi=["area", "sex"]), "public")
    x = rep["before"]["avg_reidentification_risk_pct"]
    check(abs(x - index_by_hand(df, ["area", "sex"])) < 1e-9 and rep["before"]["risk_level"] == want == R.level_for_index(x),
          f"groups of {size}: index {x:.1f} → {rep['before']['risk_level']} (expected {want})")
    a = rep["after"]["avg_reidentification_risk_pct"]
    check(a <= 10 + 1e-9 and rep["after"]["risk_level"] == "low", f"groups of {size}: safer copy index {a:.2f} → low")
check(R.level_for_index(10) == "low" and R.level_for_index(10.0001) == "medium" and R.level_for_index(100 / 3) == "medium"
      and R.level_for_index(34) == "high" and R.level_for_index(200 / 3) == "high" and R.level_for_index(67) == "critical",
      "band edges: 10 low · 33.3 medium · 66.7 high · above critical")

print("\n2. Direct identifiers raise the score through the index itself")
df = pd.DataFrame({"name": [f"Person {i}" for i in range(40)], "area": [f"A{i // 20}" for i in range(40)]})
rep, prot = R.analyze(df, roles(direct=["name"], qi=["area"]), "public")
check(rep["before"]["avg_reidentification_risk_pct"] == 100 and rep["before"]["risk_level"] == "critical"
      and abs(rep["before_details"]["avg_reidentification_risk_pct"] - 5) < 1e-9 and "name" not in prot.columns,
      f"names + area in groups of 20: index {rep['before']['avg_reidentification_risk_pct']} (area alone "
      f"{rep['before_details']['avg_reidentification_risk_pct']}) → critical; the safer copy has no names")

print("\n3. SDAIA budget 2022 (a published budget table)")
sd = ROOT / "tests" / "files" / "sdaia_budget_2022.csv"
df, info = DP.read_table(sd.read_bytes())
prof = DP.dataset_type(df, info["notes_top"] + info["notes_bottom"])
small = DP.small_counts(df, prof, R.CONTEXTS["public"]["k"])
check(prof["kind"] == "aggregate" and not prof["person_signs"], f"read as a summary table ({prof['kind']}), "
      f"no sign of personal rows ('اسم البند' is the name of a budget item, not of a person)")
check(len(df) == 5 and not small, f"{len(df)} rows of totals, no count below 10 → person-level exposure 0 / 100")
check(DP.dataset_type(pd.DataFrame({"اسم الموظف": [f"x{i}" for i in range(30)], "المبلغ": range(30)}))["person_signs"] != [],
      "a column named 'اسم الموظف' (employee name) is still a sign of personal rows")

print("\n4. Safer copy fixes")
df = pd.DataFrame({"member_no": [f"M-{1000 + i}" for i in range(30)], "plan": ["a", "b", "c"] * 10})
before = df  # the same object the engine receives
try:
    rep, prot = R.analyze(df, roles(record=["member_no"]), "public")
    check("member_no" not in prot.columns and len(prot) == 30, "only a record number, no revealing detail: no crash, "
          f"number removed, {len(prot)} rows kept")
except Exception as e:  # noqa: BLE001
    check(False, f"only a record number: {type(e).__name__}: {e}")
df = pd.DataFrame({"area": [f"A{i // 10}" for i in range(40)], "note": [None] * 40, "flag": ["x"] * 40,
                   "diag": (["HIV"] + [None] * 9) * 4})
rep, prot = R.analyze(df, roles(qi=["area"], sensitive=["note", "flag", "diag"]), "public")
lc = R.l_columns(df, ["area"], ["note", "flag", "diag"])
check(lc == ["diag"] and len(prot) == 40, f"private columns with one value for everyone are not l-checked ({lc}); "
      f"'HIV or empty' still is; {len(prot)} of 40 rows kept")
check(before.equals(pd.DataFrame({"member_no": [f"M-{1000 + i}" for i in range(30)], "plan": ["a", "b", "c"] * 10})),
      "the original table is not changed by the analysis")

print("\n5. review_sample.csv with the real model's labels (13 Oct)")
df, _ = DP.read_table((ROOT / "tests" / "files" / "review_sample.csv").read_bytes())
orig = df.copy()
real = roles(direct=["id", "plan_id", "source_document_ids"], qi=["user_register"], sensitive=["split", "notes"],
             record=["source_line_index", "id", "plan_id", "source_chunk_ids", "source_document_ids",
                     "source_manifest_hash", "generation_fingerprint"])
rep, prot = R.analyze(df, real, "public")
check(len(prot) == 30 and rep["protection"]["suppressed_records"] == 0,
      f"{len(prot)} of 30 rows kept (the old engine left out all 30: 'notes' is empty everywhere)")
check(rep["before"]["risk_level"] == R.level_for_index(rep["before"]["avg_reidentification_risk_pct"])
      and rep["after"]["avg_reidentification_risk_pct"] <= 10, f"index {rep['before']['avg_reidentification_risk_pct']} "
      f"({rep['before']['risk_level']}) → {rep['after']['avg_reidentification_risk_pct']:.2f} ({rep['after']['risk_level']})")
check(df.equals(orig), "the uploaded table is unchanged after the analysis")

print(f"\n{'ALL PASSED' if not fails else str(len(fails)) + ' FAILED'}")
sys.exit(1 if fails else 0)
