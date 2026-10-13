#!/bin/bash
# مَأْمَن — دمج RAG + Fine-tuning: نفس إعدادات train_multi.sh، بس كل مثال فيه أقرب 3 أعمدة مسترجعة.
# البرومبت أطول (~1,100–1,350 توكن بدل ~700) فرفعنا max-seq-length، والـ adapter ينحفظ لحاله في adapters_rag.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -s training_data_rag/train.jsonl ]; then echo 'Run scripts/prepare_training_multi.py --rag 3 first'; exit 1; fi

python -m mlx_lm lora \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --train --data training_data_rag \
  --batch-size 1 --num-layers 8 \
  --iters 330 --learning-rate 1e-4 \
  --max-seq-length 2048 --grad-checkpoint --mask-prompt \
  --steps-per-eval 50 --val-batches 24 \
  --adapter-path adapters_rag
