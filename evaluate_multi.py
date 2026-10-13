"""
مَأْمَن — تقييم تصنيف الأعمدة على داتا سِتات test (ما شافها النموذج ولا قاعدة الـ RAG)

الأوضاع:
  knn       : تصويت أقرب 5 أعمدة بدون أي نموذج لغوي (خط أساس للاسترجاع)
  zero_shot : Qwen بالبرومبت فقط
  rag       : Qwen + أقرب 5 أعمدة كأمثلة
  finetuned : Qwen + LoRA adapter

  python scripts/evaluate_multi.py --modes knn zero_shot rag
  python scripts/evaluate_multi.py --modes finetuned --adapter-path adapters
  أضيفوا --limit 20 لتجربة سريعة

المخرجات: outputs/eval_<mode>.json (كل عمود) و outputs/eval_summary.md (جدول المقارنة)
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maman_common import ALL_KEYS, ROLE_KEYS, ROOT, Retriever, apply_rules, load_labeled  # noqa: E402

LABEL_AR = {"direct_identifier": "معرّف مباشر", "quasi_identifier": "معرّف غير مباشر",
            "sensitive_attribute": "حساس", "record_identifier": "مفتاح سجل", "needs_review": "يحتاج مراجعة"}


def knn_predict(retriever, features, k=5):
    hit = apply_rules(features)
    if hit:
        return {**hit, "status": "ok"}
    neighbours = retriever.search(features, k)
    pred = {key: sum(n["labels"][key] for n in neighbours) * 2 > len(neighbours) for key in ALL_KEYS}
    return {**pred, "reason": "", "source": "knn", "status": "ok",
            "examples_used": [f"{n['dataset']}.{n['column']}" for n in neighbours]}


def f1_stats(gold, pred, key):
    tp = sum(g[key] and p[key] for g, p in zip(gold, pred))
    fp = sum(not g[key] and p[key] for g, p in zip(gold, pred))
    fn = sum(g[key] and not p[key] for g, p in zip(gold, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}


def score(gold, pred):
    per_key = {k: f1_stats(gold, pred, k) for k in ALL_KEYS}
    return {
        "per_key": per_key,
        "macro_f1_roles": sum(per_key[k]["f1"] for k in ROLE_KEYS) / len(ROLE_KEYS),
        "exact_match_roles": sum(all(g[k] == p[k] for k in ROLE_KEYS) for g, p in zip(gold, pred)) / len(gold),
        "errors": sum(p.get("status") == "error" for p in pred),
        "rules_hits": sum(str(p.get("source", "")).startswith("rule:") for p in pred),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modes", nargs="+", default=["knn", "zero_shot", "rag"],
                   choices=["knn", "zero_shot", "rag", "finetuned", "rag_finetuned"])
    p.add_argument("--adapter-path", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--retriever", default="auto", choices=["auto", "e5", "char"])
    p.add_argument("--use-unreviewed-proposals", action="store_true")
    a = p.parse_args()

    test = load_labeled({"test"}, a.use_unreviewed_proposals)[: a.limit]
    kb = load_labeled({"train", "valid"}, a.use_unreviewed_proposals)
    assert not {c["dataset"] for c in test} & {c["dataset"] for c in kb}, "test dataset leaked into knowledge base"
    gold = [c["labels"] for c in test]
    print(f"test: {len(test)} columns | knowledge base: {len(kb)} columns")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    summary = {}
    for mode in a.modes:
        start = time.time()
        if mode == "knn":
            retriever = Retriever(kb, backend=a.retriever)
            preds = [knn_predict(retriever, c) for c in test]
            label = f"knn ({retriever.backend})"
        else:
            from maman_classifier import MamanClassifier
            clf = MamanClassifier(mode, adapter_path=a.adapter_path, retriever_backend=a.retriever,
                                  allow_unreviewed=a.use_unreviewed_proposals)
            preds = []
            for i, c in enumerate(test, 1):
                preds.append(clf.classify(c))
                print(f"\r[{mode}] {i}/{len(test)}", end="", flush=True)
            print()
            label = f"rag ({clf.retriever.backend})" if mode == "rag" else mode
            del clf

        result = score(gold, preds)
        result["seconds"] = round(time.time() - start, 1)
        result["label"] = label
        result["n_test"] = len(test)
        summary[label] = result
        rows = [{"dataset": c["dataset"], "column": c["column"], "gold": c["labels"],
                 "pred": {k: pr[k] for k in ALL_KEYS}, "correct_roles": all(c["labels"][k] == pr[k] for k in ROLE_KEYS),
                 **{k: pr.get(k) for k in ("reason", "source", "status", "examples_used", "error")}}
                for c, pr in zip(test, preds)]
        (out_dir / f"eval_{mode}.json").write_text(
            json.dumps({"metrics": result, "columns": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{label}: macro-F1 {result['macro_f1_roles']:.3f} | exact {result['exact_match_roles']:.3f} "
              f"| errors {result['errors']} | {result['seconds']}s")

    # الجدول يجمع كل الطرق اللي انقيّمت على نفس مجموعة الاختبار (مو بس اللي في هذا التشغيل)
    for mode in ("knn", "zero_shot", "rag", "finetuned", "rag_finetuned"):
        path = out_dir / f"eval_{mode}.json"
        if path.exists():
            m = json.loads(path.read_text(encoding="utf-8"))["metrics"]
            if m.get("n_test") == len(test) and m.get("label") not in summary:
                summary[m["label"]] = m
    order = ["knn", "zero_shot", "rag (", "finetuned", "rag_finetuned"]
    summary = dict(sorted(summary.items(),
                          key=lambda kv: next((i for i, m in reversed(list(enumerate(order))) if kv[0].startswith(m)), 9)))

    lines = ["| الطريقة | Macro-F1 | تطابق كامل | " + " | ".join(LABEL_AR[k] + " F1" for k in ALL_KEYS) + " | أخطاء صيغة |",
             "|---" * (4 + len(ALL_KEYS)) + "|"]
    for label, r in summary.items():
        lines.append(f"| {label} | {r['macro_f1_roles']:.3f} | {r['exact_match_roles']:.3f} | " +
                     " | ".join(f"{r['per_key'][k]['f1']:.3f}" for k in ALL_KEYS) + f" | {r['errors']} |")
    note = (f"\nTest: {len(test)} columns from {len({c['dataset'] for c in test})} unseen datasets. "
            f"Macro-F1 and exact match are over the 4 roles (needs_review reported separately).\n")
    (out_dir / "eval_summary.md").write_text("\n".join(lines) + note, encoding="utf-8")
    print("\n" + "\n".join(lines) + note)


if __name__ == "__main__":
    main()
