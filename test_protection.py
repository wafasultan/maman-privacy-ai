"""
Ma'man — privacy checks on the SAFER COPY AS EXPORTED (the CSV text), counted here without the engine's own code.
Run:  .venv-mlx/bin/python tests/test_protection.py      (also run by tests/run_tests.command)

A. Each technique on its own: date periods, age ranges, numeric ranges, masking, removal of names / free text.
B. Every personal file × every audience (public, partner, internal), with labels from the approved annotations:
   1. no original identifier value appears anywhere in the copy (masked values are coarse, not the originals)
   2. every group of people sharing the visible main details has at least k people, and private values vary (l)
   3. every other kept column passes "main details + this one" (at most the 5% budget below k)
   4. free text is gone; rows left out stay within 5%
   5. what the report says (index, column by column) is what the file holds
C. Utility: columns kept vs the previous version (reported, and never with a privacy check failing).
"""
import csv
import io
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import pandas as pd  # noqa: E402

import dataset_profile as DP  # noqa: E402
import risk_engine as R  # noqa: E402
from fake_model import classify  # noqa: E402
from maman_common import describe_column  # noqa: E402

fails = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        fails.append(msg)


def roles_for(df):
    roles = {"direct": [], "qi": [], "sensitive": [], "record": []}
    for c in df.columns:
        r = classify(describe_column("x", c, df[c]))
        for k, key in [("direct_identifier", "direct"), ("quasi_identifier", "qi"),
                       ("sensitive_attribute", "sensitive"), ("record_identifier", "record")]:
            if r.get(k):
                roles[key].append(c)
    roles["qi"] = [c for c in roles["qi"] if c not in roles["direct"]]
    return roles


def exported_rows(protected):
    text = protected.to_csv(index=False)
    return list(csv.DictReader(io.StringIO(text)))


def groups(rows, cols):
    return Counter(tuple((r[c] or "").strip() or "?" for c in cols) for r in rows)


# ── A. techniques ─────────────────────────────────────────────
print("A. Techniques")
dates = pd.Series([f"{1950 + i % 45}-{1 + i % 12:02d}-{1 + i % 28:02d}" for i in range(300)])
h = R.build_hierarchy(pd.DataFrame({"d": dates}), "d")
names = [n["en"] for n in h["names"]]
check({"Year only", "5-year periods", "10-year periods"} <= set(names), f"dates: {names}")
check(h["values"][names.index("10-year periods")].iloc[0] == "1950-1959", "date → '1950-1959'")
ages = pd.Series(list(range(18, 78)) * 2)
h = R.build_hierarchy(pd.DataFrame({"age": ages}), "age")
check(any("10-year ranges" == n["en"] for n in h["names"]) and "30-39" in set(h["values"][[n["en"] for n in h["names"]].index("10-year ranges")]),
      "ages → 10-year ranges such as '30-39'")
sal = pd.Series([4000 + 250 * (i % 120) for i in range(600)])
h = R.build_hierarchy(pd.DataFrame({"salary": sal}), "salary")
check(len(h["names"]) >= 4 and all(re.fullmatch(r"\d+-\d+", v) for v in h["values"][1].unique()),
      f"salary → ranges ({h['names'][1]['en']}: {sorted(h['values'][1].unique())[:3]}…)")
m = R.mask_identifier(pd.Series(["dana@example.com", "a.very.long.name@Gmail.com", None] + [f"u{i}@example.com" for i in range(30)]
                                + ["not-an-email"]))
check(m is not None and list(m[0][:2]) == ["*****@example.com", "*****@gmail.com"] and pd.isna(m[0][2])
      and m[0].iloc[-1] == "*****", "email → *****@domain (fixed 5 stars), missing stays missing, malformed fully hidden")
m = R.mask_identifier(pd.Series(["+966551234567", "0509876543"]))
check(m is not None and list(m[0]) == ["055xxxxxxx", "050xxxxxxx"], "Saudi mobile → operator prefix only")
m = R.mask_identifier(pd.Series(["SA0380000000608010167519"]))
check(m is not None and list(m[0]) == ["SA bank 80"], "IBAN → bank code only")
check(R.mask_identifier(pd.Series(["1234567890", "2234567890"])) is None, "national ID → no masked form (removed)")
check(R.looks_like_person_name(pd.Series([f"name {i} family" for i in range(40)]), "اسم_العميل"), "a name column is found by header + values")
check(not R.looks_like_person_name(pd.Series(["Bank A", "Bank B"] * 20), "bank_name"), "'bank_name' with 2 values is not a person's name")
check(R.is_free_text(pd.Series([f"Patient said the pain started {i} days ago after work" for i in range(30)])), "free text is detected")
codes = pd.Series([f"SA-SYN-{i:018d}" for i in range(60)])
check(not R.is_free_text(codes) and R.looks_unique_text(codes), "a long code with no spaces (IBAN-like) is a row key, not free text")
check(not R.looks_unique_text(pd.Series(["A1", "B2", "X9"] * 30)), "a short repeated code is not a row key")

# a column labelled as naming people but with a few shared values (a bank) is protected in combination, not dropped blindly
few = pd.DataFrame({"bank": [f"Bank {'ABCD'[i % 4]}" for i in range(200)], "age": [20 + i % 40 for i in range(200)],
                    "pid": [f"P{i:05d}" for i in range(200)]})
rep, _ = R.analyze(few, {"direct": ["bank", "pid"], "record": [], "qi": ["age"], "sensitive": []}, "public")
check(rep["identifiers_demoted"] == ["bank"] and "bank" in rep["roles"]["qi"] and "pid" in rep["roles"]["direct"],
      "a 4-value 'identifier' is treated as a combined detail; a unique ID stays an identifier")

# combined identifiers: each column alone is common, the combination is unique
n = 400
combo = pd.DataFrame({"g": ["M", "F"] * (n // 2), "city": [f"c{i % 5}" for i in range(n)],
                      "dept": [f"d{(i // 2) % 7}" for i in range(n)], "age": [20 + (i * 7) % 45 for i in range(n)],
                      "diag": [f"x{i % 3}" for i in range(n)]})
rep, prot = R.analyze(combo, {"direct": [], "record": [], "qi": ["g", "city", "dept", "age"], "sensitive": ["diag"]}, "public")
rows = exported_rows(prot)
vis = [c for c in ["g", "city", "dept", "age"] if rows and c in rows[0]]
check(rep["before"]["unique_pct"] > 50 and (not vis or min(groups(rows, vis).values()) >= 10),
      f"combined details: {rep['before']['unique_pct']:.0f}% unique before → every group ≥ 10 after (visible: {vis})")

# ── B. personal files × audiences ────────────────────────────
print("\nB. Safer copies, checked on the exported CSV")
files = sorted((ROOT / "data" / "synthetic").glob("*.csv")) + [ROOT / "data" / "holdout" / "sa_gym_members.csv",
                                                               ROOT / "data" / "adult_preprocessed.csv"]
files += [f for f in sorted((ROOT / "tests" / "files").glob("*.csv"))
          if f.name in ("privacy_test_25_columns.csv", "edge_small_personal.csv", "edge_personal_with_title.csv")]
utility_rows = []
for f in files:
    df, _ = DP.read_table(f.read_bytes())
    roles = roles_for(df)
    for ctx_name in ("public", "partner", "internal"):
        ctx = R.CONTEXTS[ctx_name]
        rep, prot = R.analyze(df, roles, ctx_name)
        p, rows = rep["protection"], exported_rows(prot)
        tag = f"{f.name} [{ctx_name}]"
        cols = list(rows[0].keys()) if rows else list(prot.columns)
        problems = []
        # 1. identifiers
        for c in rep["roles"]["direct"] + rep["roles"]["record"]:
            originals = set(df[c].dropna().astype(str).str.strip())
            if c in cols:
                if c not in p["masked"]:
                    problems.append(f"{c} kept although it is an identifier")
                leaked = {r[c] for r in rows} & originals
                if leaked:
                    problems.append(f"{c}: {len(leaked)} original values in the copy")
            flat = {v for r in rows for k, v in r.items() if k != c}
            longs = {o for o in originals if len(o) >= 6}  # short numbers (1, 25, 300) legitimately repeat elsewhere
            if len(longs) > 20 and len(flat & longs) > 0.01 * len(longs):
                problems.append(f"values of {c} appear in other columns")
        # 2. k and l on the visible main details
        core = [c for c in rep["roles"]["qi"] if c in cols]
        if rows and core:
            g = groups(rows, core)
            small = sum(v for v in g.values() if v < ctx["k"])
            if small:
                problems.append(f"{small} rows in groups smaller than k={ctx['k']}")
            # l-diversity policy: private columns that are categories in the original file (≤ 20 values);
            # numeric private values (salary) are protected against linkage by ranges instead (a stated limitation)
            policy = R.l_columns(df, rep["roles"]["qi"], rep["roles"]["sensitive"])
            for s in [c for c in policy if c in cols]:
                vals = defaultdict(set)
                for r in rows:
                    vals[tuple((r[c] or "").strip() or "?" for c in core)].add(r[s])
                if ctx["l"] > 1 and any(len(v) < ctx["l"] for v in vals.values()):
                    problems.append(f"'{s}' is the same for everyone in some group (l={ctx['l']})")
        # 3. one extra detail: every value still shown sits in a group of ≥ k people (blanked cells are the rest)
        for c, e in p["extras"].items():
            if e["result"] in ("kept", "generalized", "masked"):
                g = groups(rows, core + [c])
                below = sum(v for key, v in g.items() if v < ctx["k"] and key[-1] != "?")
                if below:
                    problems.append(f"'{c}' + main details: {below} shown values in groups below k")
                blank = sum(1 for r in rows if not (r[c] or "").strip()) - int(df.loc[prot.index, c].isna().sum())
                if rows and blank > ctx["max_suppression"] * len(rows) + 1:
                    problems.append(f"'{c}': {blank} cells blanked (> 5%)")
            elif c in cols:
                problems.append(f"'{c}' reported removed but present")
        # 4. free text + rows budget
        for c, e in p["extras"].items():
            if e["kind"] == "free_text" and c in cols:
                problems.append(f"free text '{c}' kept")
        if len(rows) < (1 - ctx["max_suppression"]) * len(df) - 1e-9 and len(rows) > 0:
            problems.append(f"{len(df) - len(rows)} rows left out (> 5%)")
        # 5. report = file
        if rows and core:
            mine = 100 * len(groups(rows, core)) / len(rows)
            if abs(mine - rep["after"]["avg_reidentification_risk_pct"]) > 0.01:
                problems.append(f"index reported {rep['after']['avg_reidentification_risk_pct']} vs file {mine:.4f}")
        for c, st in p["column_status"].items():
            if (st == "removed") == (c in cols):
                problems.append(f"'{c}' status {st} but {'present' if c in cols else 'absent'}")
            if st == "kept" and c in cols and rows:
                orig = ["" if pd.isna(o) else str(o).strip() for o in df.loc[prot.index, c]]
                if [str(r[c]).strip() for r in rows] != orig:
                    problems.append(f"'{c}' reported unchanged but differs")
        # 7. coverage: every column in the copy is a main detail, a private category (l-diversity),
        #    or was checked as "main details + this one" — nothing is kept unchecked
        lcols = set(R.l_columns(df, rep["roles"]["qi"], rep["roles"]["sensitive"]))
        for c in cols:
            if c not in rep["roles"]["qi"] and c not in lcols and c not in p["extras"] and df[c].nunique() > 1:
                problems.append(f"'{c}' kept without any check")
        # 6. near-unique text columns (IDs the model may miss): never shown, in any form
        for c in df.columns:
            s = df[c].dropna().astype(str)
            if len(s) >= 50 and s.nunique() / len(s) > 0.9 and not pd.api.types.is_numeric_dtype(df[c]) and c in cols \
                    and not R.is_free_text(df[c]):
                if {r[c] for r in rows} & set(s.str.strip()):
                    problems.append(f"near-unique '{c}' shows original values")
        check(not problems, f"{tag}: {len(cols)} of {len(df.columns)} columns, {len(rows)} rows"
              + ("" if not problems else " — " + "; ".join(problems[:3])))
        if ctx_name == "public":
            utility_rows.append((f.name, len(df.columns), len(cols), p["utility_summary"]["cells_unchanged_pct"],
                                 rep["before"]["avg_reidentification_risk_pct"], rep["after"]["avg_reidentification_risk_pct"],
                                 rep["residual"]["after"]["worst_index"]))

print("\nC. Utility and risk (public), on the exported copy")
print(f"  {'file':<30}{'columns kept':>14}{'values kept':>13}{'index before→after':>22}{'one extra detail':>18}")
for name, total, kept, cells, b, a, resid in utility_rows:
    print(f"  {name:<30}{f'{kept} of {total}':>14}{f'{cells:g}%':>13}{f'{b:.1f} → {a:.2f}':>22}{(f'{resid:.1f}' if resid is not None else '–'):>18}")

print(f"\n{'ALL PASSED' if not fails else str(len(fails)) + ' FAILED'}")
sys.exit(1 if fails else 0)
