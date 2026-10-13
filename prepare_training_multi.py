"""
مَأْمَن — تجهيز بيانات الـ Fine-tuning (MLX-LM chat JSONL) من الأعمدة المصنّفة

  train.jsonl / valid.jsonl : من داتا سِتات train و valid
  test.jsonl                : من داتا سِتات test (للتقييم فقط، mlx_lm.lora ما يتدرب عليه)

نفس البرومبت اللي يستخدمه maman_classifier.py، عشان التدريب والاستخدام يتطابقون.
الافتراضي: الأعمدة المعتمَدة (APPROVED) فقط.

  python scripts/prepare_training_multi.py
  python scripts/prepare_training_multi.py --use-unreviewed-proposals   # تجربة صريحة بتصنيفات غير مراجَعة
  python scripts/prepare_training_multi.py --rag 3   # دمج RAG + Fine-tuning → training_data_rag/

مع --rag: كل مثال تدريب ينكتب وقدامه أقرب K أعمدة مصنّفة من داتا سِتات ثانية (نفس صيغة وضع rag)،
فالنموذج يتعلّم يستفيد من الأمثلة المسترجعة. نستبعد نفس الداتا سِت عشان التدريب يشابه الاختبار
(وقت الاختبار الداتا سِت جديدة وما لها أمثلة في قاعدة المعرفة).
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maman_common import ROOT, Retriever, answer_json, build_messages, load_labeled  # noqa: E402


def to_example(col, examples=()):
    messages = build_messages(col, examples)
    messages.append({"role": "assistant", "content": answer_json(col["labels"], col["reason"])})
    return {"messages": messages}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--use-unreviewed-proposals", action="store_true")
    p.add_argument("--rag", type=int, default=0, metavar="K", help="add K retrieved examples to every prompt")
    a = p.parse_args()
    if a.use_unreviewed_proposals:
        print("WARNING: using PROPOSED labels that a human has not reviewed yet.")

    out = ROOT / ("training_data_rag" if a.rag else "training_data")
    out.mkdir(exist_ok=True)
    retriever = None
    if a.rag:
        retriever = Retriever(load_labeled({"train", "valid"}, a.use_unreviewed_proposals), backend="char")
    for split in ("train", "valid", "test"):
        cols = load_labeled({split}, a.use_unreviewed_proposals)
        examples = [to_example(c, retriever.search(c, a.rag, exclude_dataset=True) if retriever else ())
                    for c in cols]
        random.Random(42).shuffle(examples)
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for e in examples:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        datasets = sorted({c["dataset"] for c in cols})
        print(f"{split:<5} {len(examples):>4} columns from {len(datasets)} datasets: {', '.join(datasets)}")


if __name__ == "__main__":
    main()
