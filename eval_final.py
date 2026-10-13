"""
Ma'man — evaluation of the FINAL app configuration on the 102 held-out test columns, one retriever per process
(so memory is measured cleanly):

  .venv-mlx/bin/python scripts/eval_final.py --retriever char
  .venv-mlx/bin/python scripts/eval_final.py --retriever e5      (= the app default, built exactly like the app)
  .venv-mlx/bin/python scripts/eval_final.py --compare           → outputs/final_eval.md

Same classifier class and settings the app uses (rag_finetuned, adapters_rag, k=3, rules + guards, greedy).
Leakage checks from eval_retriever.py run first. E5 is strict here: if it cannot load, the run fails
instead of quietly measuring char n-grams.
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_retriever import leakage_checks, metrics  # noqa: E402
from maman_common import ROLE_KEYS, ROOT, Retriever, load_labeled  # noqa: E402

OUT = ROOT / "outputs"


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)


def run(backend):
    import mlx.core as mx
    from maman_classifier import MamanClassifier, default_retriever

    test, kb = load_labeled({"test"}), load_labeled({"train", "valid"})
    leak = leakage_checks(test, kb)
    rss_start = rss_mb()

    t = time.time()
    r = Retriever(kb, backend=backend, strict=True)
    retriever_init = time.time() - t
    rss_after_retriever = rss_mb()
    del r

    t = time.time()
    kwargs = {} if backend == default_retriever() else {"retriever_backend": backend}  # the default = built like the app
    clf = MamanClassifier("rag_finetuned", adapter_path=ROOT / "adapters_rag", retriever_strict=True, **kwargs)
    load_s = time.time() - t
    assert clf.retriever.backend == backend, f"expected {backend}, got {clf.retriever.backend}"

    t = time.time()
    for c in test:
        clf.retriever.search(c, 3, exclude_dataset=True)
    search_ms = 1000 * (time.time() - t) / len(test)

    gold, preds, t = [c["labels"] for c in test], [], time.time()
    for i, c in enumerate(test, 1):
        preds.append(clf.classify(c))
        print(f"\r[final eval, retriever={backend}] {i}/{len(test)}", end="", flush=True)
    print()
    total = time.time() - t

    m = metrics(gold, preds, test)
    result = {"retriever": backend, "app_default": default_retriever(), "built_like_app": not kwargs,
              "leakage": leak, **m,
              "time": {"classifier_load_s": round(load_s, 1), "retriever_init_s": round(retriever_init, 2),
                       "retrieval_ms_per_column": round(search_ms, 2),
                       "classify_s_per_column": round(total / len(test), 2), "classify_total_s": round(total, 1)},
              "memory": {"rss_before_mb": round(rss_start), "retriever_rss_added_mb": round(rss_after_retriever - rss_start),
                         "peak_rss_mb": round(rss_mb()), "mlx_peak_mb": round(mx.get_peak_memory() / 2**20)},
              "per_column": [{"column": f"{c['dataset']}.{c['column']}", "correct": all(c["labels"][k] == p[k] for k in ROLE_KEYS),
                              "pred": [k for k in ROLE_KEYS if p[k]], "gold": [k for k in ROLE_KEYS if c["labels"][k]],
                              "retriever": p.get("retriever"), "examples": p.get("examples_used")} for c, p in zip(test, preds)]}
    (OUT / f"final_eval_{backend}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in result.items() if k not in ("per_column", "leakage")})


def compare():
    r = {b: json.loads((OUT / f"final_eval_{b}.json").read_text(encoding="utf-8")) for b in ("char", "e5")}
    c, e = r["char"], r["e5"]
    rows = [("Macro-F1 (4 roles)", "macro_f1", 3), ("Exact match", "exact_match", 3),
            ("Sensitive recall", "sensitive_attribute_recall", 3), ("Sensitive F1", "sensitive_attribute_f1", 3),
            ("QI F1", "quasi_identifier_f1", 3), ("Exact match, Arabic", "exact_Arabic", 3),
            ("Exact match, English", "exact_English", 3), ("Exact match, Arabizi", "exact_Arabizi", 3)]
    lines = ["# Final configuration: char n-grams vs multilingual E5 (measured in the app environment)",
             f"Same classifier as the app (rag_finetuned, adapters_rag, k=3), 102 held-out columns. "
             f"App default: **{e['app_default']}** (the e5 run was built exactly like the app: {e['built_like_app']}).", "",
             "| Metric | char | E5 | change |", "|---|---|---|---|"]
    lines += [f"| {lab} | {c[k]:.{d}f} | {e[k]:.{d}f} | {e[k] - c[k]:+.{d}f} |" for lab, k, d in rows]
    lines += ["", "| Cost | char | E5 |", "|---|---|---|"]
    for lab, sect, k in [("Classifier load (s)", "time", "classifier_load_s"), ("Retriever start-up (s)", "time", "retriever_init_s"),
                         ("Retrieval per column (ms)", "time", "retrieval_ms_per_column"),
                         ("Classification per column (s)", "time", "classify_s_per_column"),
                         ("Memory added by the retriever (MB, RSS)", "memory", "retriever_rss_added_mb"),
                         ("Peak process memory (MB, RSS)", "memory", "peak_rss_mb"), ("Peak MLX memory (MB)", "memory", "mlx_peak_mb")]:
        lines.append(f"| {lab} | {c[sect][k]} | {e[sect][k]} |")
    fixed = [x["column"] for x, y in zip(c["per_column"], e["per_column"]) if not x["correct"] and y["correct"]]
    broken = [x["column"] for x, y in zip(c["per_column"], e["per_column"]) if x["correct"] and not y["correct"]]
    from math import comb
    n = len(fixed) + len(broken)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(len(fixed), len(broken)) + 1)) / 2 ** n) if n else 1.0
    lines += ["", f"Columns fixed by E5: {len(fixed)} · broken by E5: {len(broken)} · McNemar exact p = {p:.2f}",
              f"Leakage checks: {e['leakage']['result']} (pool {e['leakage']['pool_columns']} columns from "
              f"{len(e['leakage']['pool_datasets'])} datasets; test {e['leakage']['test_columns']} columns from "
              f"{len(e['leakage']['test_datasets'])} other datasets)."]
    (OUT / "final_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--retriever", choices=["char", "e5"])
    a.add_argument("--compare", action="store_true")
    args = a.parse_args()
    compare() if args.compare else run(args.retriever)
