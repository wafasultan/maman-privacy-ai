"""
مَأْمَن — مصنّف الأعمدة الموحّد (Qwen2.5-3B عبر MLX)

الأوضاع:
  zero_shot : البرومبت فقط بدون أمثلة
  rag       : البرومبت + أقرب 5 أعمدة مصنّفة من train/valid (RAG)
  finetuned : النموذج مع LoRA adapter المدرَّب (بدون أمثلة، نفس صيغة التدريب)
  rag_finetuned : الدمج — adapter مدرَّب على برومبتات فيها أمثلة مسترجعة (adapters_rag) + أقرب 3 أمثلة وقت الاستخدام

في كل الأوضاع طبقة القواعد تشتغل أول (إيميل، جوال، هوية، آيبان).

الاستخدام:
  python scripts/maman_classifier.py data/synthetic/sa_hospital.csv --mode rag
  python scripts/maman_classifier.py my.csv --mode finetuned --adapter-path adapters
"""

import argparse
import os
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maman_common import (ALL_KEYS, MODEL, ROLE_KEYS, ROOT, Retriever, apply_rules,  # noqa: E402
                          build_messages, describe_column, id_guard, load_labeled, parse_prediction)

MAX_TOKENS = 160
RAG_FT_K = 3  # نفس عدد الأمثلة في training_data_rag (prepare_training_multi.py --rag 3)
FORMAT_REMINDER = ("\nFORMAT REMINDER: your previous answer failed validation ({error}). "
                   "Return one JSON object with the five boolean flags and an Arabic reason.")


def default_retriever():
    """The app's retriever: multilingual E5 (adopted 11 Oct). Roll back without code changes:
    MAMAN_RETRIEVER=char (run_app_char.command does this)."""
    value = os.environ.get("MAMAN_RETRIEVER", "e5").strip().lower()
    if value not in ("e5", "char"):
        raise ValueError(f"MAMAN_RETRIEVER must be e5 or char, not {value!r}")
    return value


class MamanClassifier:
    def __init__(self, mode="rag", model=MODEL, adapter_path=None, k=5, use_rules=True,
                 retriever_backend=None, allow_unreviewed=False, retriever_strict=False):
        assert mode in ("zero_shot", "rag", "finetuned", "rag_finetuned"), mode
        from mlx_lm import load

        tuned = mode in ("finetuned", "rag_finetuned")
        if tuned and adapter_path is None:
            adapter_path = ROOT / ("adapters_rag" if mode == "rag_finetuned" else "adapters")
        if tuned and not Path(adapter_path).exists():
            raise SystemExit(f"Adapter not found: {adapter_path}. Train first (scripts/train_multi.sh).")
        if mode == "rag_finetuned":
            k = RAG_FT_K
        self.mode, self.k, self.use_rules = mode, k, use_rules
        self.model, self.tokenizer = load(model, adapter_path=str(adapter_path) if tuned else None)
        self.retriever = None
        if mode in ("rag", "rag_finetuned"):
            kb = load_labeled({"train", "valid"}, allow_unreviewed)
            self.retriever = Retriever(kb, backend=retriever_backend or default_retriever(), strict=retriever_strict)
            print(f"[rag] knowledge base: {len(kb)} columns, retriever: {self.retriever.backend}")

    def _generate(self, messages):
        from mlx_lm import generate

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False)

    def classify(self, features):
        if self.use_rules:
            hit = apply_rules(features)
            if hit:
                return {**hit, "status": "ok"}

        # rag_finetuned اتدرّب على أمثلة من داتا سِتات ثانية، فنستبعد نفس الداتا سِت وقت الاستخدام بعد
        # (يفرق بس لملفات العيّنة اللي موجودة في قاعدة المعرفة؛ ملفات المستخدم والاختبار ما لها أمثلة هناك أصلاً)
        examples = (self.retriever.search(features, self.k, exclude_dataset=self.mode == "rag_finetuned")
                    if self.retriever else [])
        messages = build_messages(features, examples)
        error = None
        for _ in range(2):
            raw = self._generate(messages)
            try:
                prediction = id_guard(features, parse_prediction(raw))
                return {**prediction, "source": self.mode, "status": "ok",
                        "retriever": self.retriever.backend if self.retriever else None,
                        "examples_used": [f"{e['dataset']}.{e['column']}" for e in examples]}
            except ValueError as exc:
                error = str(exc)
                messages = messages[:1] + [{"role": "user",
                                            "content": messages[1]["content"] + FORMAT_REMINDER.format(error=error)}]
        return {**{k: False for k in ALL_KEYS}, "reason": "", "source": self.mode, "status": "error",
                "error": error, "raw_output": raw}

    def classify_dataframe(self, df, dataset_name):
        return [{"column": col, **self.classify(describe_column(dataset_name, col, df[col]))} for col in df.columns]


def main():
    import pandas as pd

    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--mode", default="rag", choices=["zero_shot", "rag", "finetuned", "rag_finetuned"])
    p.add_argument("--adapter-path", default=None)
    p.add_argument("--use-unreviewed-proposals", action="store_true",
                   help="RAG: allow PROPOSED (not yet human-reviewed) labels in the knowledge base")
    a = p.parse_args()

    path = Path(a.csv)
    df = pd.read_csv(path, skipinitialspace=True)
    clf = MamanClassifier(a.mode, adapter_path=a.adapter_path, allow_unreviewed=a.use_unreviewed_proposals)
    results = clf.classify_dataframe(df, path.stem)

    for r in results:
        roles = [k for k in ROLE_KEYS if r[k]] or ["none"]
        flag = "  ⚠ needs review" if r["needs_review"] else ""
        print(f"{r['column']:<28} {', '.join(roles):<45} [{r['source']}]{flag}\n    {r['reason']}")

    out = ROOT / "outputs" / f"classified_{path.stem}_{a.mode}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
