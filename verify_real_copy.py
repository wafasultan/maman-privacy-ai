"""
Ma'man — the safer copy with the REAL model (Qwen + the app's retriever), checked on the exported CSV bytes.
Run:  .venv-mlx/bin/python tests/verify_real_copy.py      (about 5 minutes; writes outputs/real_copy_check.txt)

Columns are labelled exactly as the app does it (one at a time, then the proxy check), the copy is exported
exactly as the download button does (UTF-8 with BOM), and everything below is counted again from those bytes:
  - the index shown = 100 × groups / rows on the visible main details
  - no original value of a removed or masked identifier is in the file
  - every value still shown in another column sits in a group of ≥ k people (main details + that column)
  - every column in the file is a main detail, a private category (l-diversity) or was checked as one extra detail
  - the column-by-column status the report gives is what the file holds
"""
import csv
import io
import resource
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import dataset_profile as DP  # noqa: E402
import risk_engine as R  # noqa: E402
from maman_classifier import MamanClassifier  # noqa: E402
from maman_common import describe_column  # noqa: E402

FILES = ["tests/files/privacy_test_25_columns.csv", "data/synthetic/sa_hr.csv", "data/holdout/sa_gym_members.csv"]
lines, fails = [], []


def say(m):
    print(m, flush=True)
    lines.append(m)


def check(ok, m):
    say(("  ok   " if ok else "  FAIL ") + m)
    if not ok:
        fails.append(m)


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)


clf = MamanClassifier("rag_finetuned", adapter_path=ROOT / "adapters_rag")
say(f"retriever in use: {clf.retriever.backend}")
for f in FILES:
    df, _ = DP.read_table((ROOT / f).read_bytes())
    t = time.time()
    flags = {c: clf.classify(describe_column(Path(f).stem, c, df[c])) for c in df.columns}
    secs = (time.time() - t) / len(df.columns)
    roles = {"direct": [], "qi": [], "sensitive": [], "record": []}
    for c, r in flags.items():
        if r.get("direct_identifier"):
            roles["direct"].append(c)
        if r.get("record_identifier"):
            roles["record"].append(c)
        if r.get("quasi_identifier") and not r.get("direct_identifier"):
            roles["qi"].append(c)
        if r.get("sensitive_attribute"):
            roles["sensitive"].append(c)
    for c in R.proxy_columns(df, roles):
        if c not in roles["sensitive"]:
            roles["sensitive"].append(c)
    say(f"\n{Path(f).name}: {len(df.columns)} columns labelled in {secs:.1f} s each · revealing together: {roles['qi']}")
    for ctx_name in ("public", "partner", "internal"):
        ctx = R.CONTEXTS[ctx_name]
        rep, prot = R.analyze(df, roles, ctx_name)
        p = rep["protection"]
        rows = list(csv.DictReader(io.StringIO(prot.to_csv(index=False).encode("utf-8-sig").decode("utf-8-sig"))))
        cols = list(rows[0].keys()) if rows else []
        core = [c for c in rep["roles"]["qi"] if c in cols]
        problems = []
        if rows and core:
            g = Counter(tuple((r[c] or "").strip() or "?" for c in core) for r in rows)
            mine = 100 * len(g) / len(rows)
            if abs(mine - rep["after"]["avg_reidentification_risk_pct"]) > 0.01:
                problems.append(f"index {rep['after']['avg_reidentification_risk_pct']} vs file {mine:.4f}")
            if any(v < ctx["k"] for v in g.values()):
                problems.append("main-detail group below k")
        for c in rep["roles"]["direct"] + rep["roles"]["record"]:
            if c in cols:
                orig = set(df[c].dropna().astype(str).str.strip())
                if {r[c] for r in rows} & orig:
                    problems.append(f"original values of {c} in the file")
        lcols = set(R.l_columns(df, rep["roles"]["qi"], rep["roles"]["sensitive"]))
        for c in cols:
            if c in core or c in lcols or df[c].nunique() <= 1:
                continue
            if c not in p["extras"]:
                problems.append(f"{c} kept without any check")
                continue
            g = Counter(tuple((r[x] or "").strip() or "?" for x in core + [c]) for r in rows)
            if sum(v for key, v in g.items() if v < ctx["k"] and key[-1] != "?"):
                problems.append(f"{c}: shown values in groups below k")
        for c, s in p["column_status"].items():
            if (s == "removed") == (c in cols):
                problems.append(f"{c} status {s} but {'present' if c in cols else 'absent'}")
        check(not problems, f"[{ctx_name}] {len(cols)} of {len(df.columns)} columns, {len(rows)} rows, index "
              f"{rep['before']['avg_reidentification_risk_pct']:.1f} → {rep['after']['avg_reidentification_risk_pct']:.2f}"
              + ("" if not problems else " — " + "; ".join(problems[:3])))
say(f"\npeak process memory {rss_mb():.0f} MB")
say("ALL PASSED" if not fails else f"{len(fails)} FAILED")
(ROOT / "outputs" / "real_copy_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
sys.exit(1 if fails else 0)
