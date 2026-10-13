"""
اختبار شرح Qwen بالعربي والإنجليزي على 3 ملفات، والنتيجة في outputs/explain_test.md
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import risk_engine as R  # noqa: E402
from explain_report import Explainer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("data/adult_preprocessed.csv", "adult", "public", ["age", "education"]),
    ("data/synthetic/sa_hospital.csv", "sa_hospital", "partner", []),
    ("data/synthetic/sa_bank_loans.csv", "sa_bank_loans", "public", []),
]


def main():
    ex = Explainer()
    results, md = [], ["# Qwen explanation test\n"]
    for path, ds, ctx, keep in CASES:
        report, _ = R.analyze(pd.read_csv(ROOT / path), R.load_roles(path, roles_dataset=ds), ctx, None, keep)
        for lang in ("ar", "en"):
            start = time.time()
            res = ex.explain(report, lang, force_qwen=True)  # نختبر Qwen باللغتين حتى لو العربي يستخدم القالب
            res.update(case=f"{ds} / {ctx}", seconds=round(time.time() - start, 1))
            results.append(res)
            print(f"[{ds} {lang}] {res['source']} ({res['seconds']}s)", flush=True)
            md.append(f"## {ds} / {ctx} / {lang} → **{res['source']}** ({res['seconds']}s)\n")
            for i, att in enumerate(res["attempts"], 1):
                verdict = "✅ accepted" if not att["problems"] else "❌ " + "; ".join(att["problems"])
                md.append(f"**Qwen attempt {i}:** {verdict}\n\n> {att['text']}\n")
            if res["source"] == "template":
                md.append(f"**Template used:**\n\n> {res['text']}\n")
    out = ROOT / "outputs"
    (out / "explain_test.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "explain_test.md").write_text("\n".join(md), encoding="utf-8")
    ok = {lang: sum(r["source"] == "qwen" for r in results if r["lang"] == lang) for lang in ("ar", "en")}
    print(f"\nQwen accepted: Arabic {ok['ar']}/3, English {ok['en']}/3  ->  outputs/explain_test.md")


if __name__ == "__main__":
    main()
