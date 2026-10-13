"""
Ma'man — retriever experiment, step 2: char n-grams vs multilingual E5, everything else fixed.

Same fine-tuned model (adapters_rag), same 102 test columns, same prompt, same k (3), same rules layer and guards,
greedy decoding. Only the 3 retrieved examples change. The model is loaded once; only the retriever is swapped.

Also measures retrieval on its own (no model): how often the retrieved neighbours carry the test column's roles,
split by Arabic / English / Arabizi column names.

Leakage checks (the script stops if any fails):
  - no test dataset or test column in the RAG pool (train + valid)
  - no test dataset mentioned in training_data_rag/train.jsonl or valid.jsonl (what the adapter was trained on)

Writes outputs/retriever_eval.md and outputs/retriever_eval.json. Does NOT touch outputs/eval_*.json.
Run: .venv-mlx/bin/python scripts/eval_retriever.py   (after scripts/e5_embed.py has written outputs/e5_cache.npz)
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_multi import score  # noqa: E402
from maman_common import ALL_KEYS, ROLE_KEYS, ROOT, Retriever, load_labeled, retrieval_text  # noqa: E402

ARABIC = re.compile(r"[؀-ۿ]")
SHOWCASE = ["maradh_mozmen", "raatib", "jins", "omr", "madina", "esm"]


def language(c):
    if c["dataset"] == "sa_arabizi":
        return "Arabizi"
    return "Arabic" if ARABIC.search(c["column"]) else "English"


def leakage_checks(test, kb):
    test_ds = {c["dataset"] for c in test}
    assert not test_ds & {c["dataset"] for c in kb}, "test dataset in the RAG pool"
    assert not {(c["dataset"], c["column"]) for c in test} & {(c["dataset"], c["column"]) for c in kb}
    for split in ("train", "valid"):
        text = (ROOT / "training_data_rag" / f"{split}.jsonl").read_text(encoding="utf-8")
        seen = set(re.findall(r'\\"dataset\\": \\"([^"\\]+)', text)) | set(re.findall(r'"dataset": "([^"]+)"', text))
        assert not seen & test_ds, f"test dataset in training_data_rag/{split}.jsonl: {seen & test_ds}"
    return {"test_datasets": sorted(test_ds), "pool_datasets": sorted({c["dataset"] for c in kb}),
            "pool_columns": len(kb), "test_columns": len(test), "result": "passed"}


def retrieval_quality(retriever, test, k=3):
    """Per test column: share of the k neighbours that have each role the column really has,
    and whether the top neighbour has exactly the same roles."""
    rows = []
    for c in test:
        nb = retriever.search(c, k, exclude_dataset=True)
        gold = {r for r in ROLE_KEYS if c["labels"][r]}
        rows.append({"column": f"{c['dataset']}.{c['column']}", "lang": language(c), "gold": sorted(gold),
                     "neighbours": [f"{n['dataset']}.{n['column']}" for n in nb],
                     "role_hits": {r: sum(n["labels"][r] for n in nb) / k for r in gold},
                     "top1_same_roles": {r for r in ROLE_KEYS if nb[0]["labels"][r]} == gold})

    def summarise(sub):
        out = {"n": len(sub), "top1_same_roles": round(sum(r["top1_same_roles"] for r in sub) / max(len(sub), 1), 3)}
        for role in ROLE_KEYS:
            vals = [r["role_hits"][role] for r in sub if role in r["role_hits"]]
            out[f"{role}_neighbour_hit"] = round(sum(vals) / len(vals), 3) if vals else None
        return out
    groups = {"all": rows, **{g: [r for r in rows if r["lang"] == g] for g in ("Arabic", "English", "Arabizi")}}
    return {g: summarise(sub) for g, sub in groups.items()}, rows


def metrics(gold, preds, test):
    s = score(gold, preds)
    out = {"macro_f1": s["macro_f1_roles"], "exact_match": s["exact_match_roles"], "errors": s["errors"]}
    for role in ROLE_KEYS:
        out[f"{role}_f1"] = s["per_key"][role]["f1"]
        out[f"{role}_recall"] = s["per_key"][role]["recall"]
        out[f"{role}_precision"] = s["per_key"][role]["precision"]
    for g in ("Arabic", "English", "Arabizi"):
        idx = [i for i, c in enumerate(test) if language(c) == g]
        out[f"exact_{g}"] = sum(all(gold[i][k] == preds[i][k] for k in ROLE_KEYS) for i in idx) / len(idx)
        out[f"n_{g}"] = len(idx)
    return out


def main():
    test = load_labeled({"test"})
    kb = load_labeled({"train", "valid"})
    leak = leakage_checks(test, kb)
    print("leakage checks passed:", leak)
    gold = [c["labels"] for c in test]
    retrievers = {"char": Retriever(kb, backend="char"), "e5": Retriever(kb, backend="e5_cache")}

    report = {"leakage": leak, "retrieval": {}, "model": {}, "showcase": {}, "per_column": {}}
    for name, r in retrievers.items():
        report["retrieval"][name], rows = retrieval_quality(r, test)
        report["per_column"][name] = {x["column"]: {"neighbours": x["neighbours"]} for x in rows}
        for col in SHOWCASE:
            c = next((t for t in test if t["column"] == col), None)
            if c:
                report["showcase"].setdefault(col, {"text": retrieval_text(c)[:80],
                                                    "gold": [k for k in ROLE_KEYS if c["labels"][k]]})
                report["showcase"][col][name] = [
                    f"{n['dataset']}.{n['column']} ({', '.join(k for k in ROLE_KEYS if n['labels'][k]) or 'none'})"
                    for n in r.search(c, 3, exclude_dataset=True)]

    from maman_classifier import MamanClassifier
    clf = MamanClassifier("rag_finetuned", adapter_path=ROOT / "adapters_rag", retriever_backend="char")
    for name, r in retrievers.items():
        clf.retriever = r  # same loaded model, only the retriever changes
        start, preds = time.time(), []
        for i, c in enumerate(test, 1):
            preds.append(clf.classify(c))
            print(f"\r[rag_finetuned + {name}] {i}/{len(test)}", end="", flush=True)
        print()
        report["model"][name] = {**metrics(gold, preds, test), "seconds": round(time.time() - start, 1)}
        for c, p in zip(test, preds):
            key = f"{c['dataset']}.{c['column']}"
            report["per_column"][name][key].update(
                pred=[k for k in ROLE_KEYS if p[k]], gold=[k for k in ROLE_KEYS if c["labels"][k]],
                correct=all(c["labels"][k] == p[k] for k in ROLE_KEYS), source=p.get("source"))
        print(name, {k: round(v, 3) for k, v in report["model"][name].items() if isinstance(v, float)})

    ch, e5 = report["per_column"]["char"], report["per_column"]["e5"]
    report["flips"] = {"fixed_by_e5": [k for k in ch if not ch[k]["correct"] and e5[k]["correct"]],
                       "broken_by_e5": [k for k in ch if ch[k]["correct"] and not e5[k]["correct"]]}

    (ROOT / "outputs" / "retriever_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                          encoding="utf-8")
    m = report["model"]
    lines = ["# Retriever experiment: char n-grams vs multilingual E5",
             "Same adapters_rag, same 102 test columns (9 unseen datasets), k=3, greedy decoding. Only the retriever changes.",
             "", "| Metric | char n-grams | E5 | change |", "|---|---|---|---|"]
    rows = [("Macro-F1 (4 roles)", "macro_f1"), ("Exact match", "exact_match"),
            ("Sensitive F1", "sensitive_attribute_f1"), ("Sensitive recall", "sensitive_attribute_recall"),
            ("Sensitive precision", "sensitive_attribute_precision"), ("QI F1", "quasi_identifier_f1"),
            ("Direct identifier F1", "direct_identifier_f1"), ("Record key F1", "record_identifier_f1"),
            ("Exact match, Arabic", "exact_Arabic"), ("Exact match, English", "exact_English"),
            ("Exact match, Arabizi", "exact_Arabizi")]
    for label, key in rows:
        lines.append(f"| {label} | {m['char'][key]:.3f} | {m['e5'][key]:.3f} | {m['e5'][key] - m['char'][key]:+.3f} |")
    lines += ["", f"Columns fixed by E5: {len(report['flips']['fixed_by_e5'])} · broken by E5: "
              f"{len(report['flips']['broken_by_e5'])}", "",
              "## Retrieval alone (no model): share of the 3 neighbours that carry the column's real role",
              "| Group | n | retriever | top-1 same roles | direct | QI | sensitive | record |", "|---|---|---|---|---|---|---|---|"]
    for g in ("all", "Arabic", "English", "Arabizi"):
        for name in ("char", "e5"):
            q = report["retrieval"][name][g]
            fmt = lambda v: "–" if v is None else f"{v:.2f}"  # noqa: E731
            lines.append(f"| {g} | {q['n']} | {name} | {q['top1_same_roles']:.2f} | {fmt(q['direct_identifier_neighbour_hit'])} | "
                         f"{fmt(q['quasi_identifier_neighbour_hit'])} | {fmt(q['sensitive_attribute_neighbour_hit'])} | "
                         f"{fmt(q['record_identifier_neighbour_hit'])} |")
    lines += ["", "## Examples (top 3 neighbours)"]
    for col, v in report["showcase"].items():
        lines += [f"- **{col}** (gold: {', '.join(v['gold']) or 'none'})",
                  f"  - char: {'; '.join(v['char'])}", f"  - e5: {'; '.join(v['e5'])}"]
    lines += ["", f"Leakage checks: {leak['result']} (pool: {leak['pool_columns']} columns from "
              f"{len(leak['pool_datasets'])} datasets; test: {leak['test_columns']} columns from {len(leak['test_datasets'])} other datasets)."]
    (ROOT / "outputs" / "retriever_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
