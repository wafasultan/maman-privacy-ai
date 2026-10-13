#!/bin/bash
# مَأْمَن — QLoRA على Qwen2.5-3B-Instruct-4bit (MLX) بالأعمدة المصنّفة متعددة الداتا سِتات
# ~110 عمود تدريب × 3 دورات ≈ 330 خطوة. على M2 بـ 8GB المتوقع دقائق مو ساعات.
# لو طلع out-of-memory: نزّل --num-layers إلى 4 أو --max-seq-length إلى 768.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -s training_data/train.jsonl ]; then echo 'Run scripts/prepare_training_multi.py first'; exit 1; fi

python -m mlx_lm lora \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --train --data training_data \
  --batch-size 1 --num-layers 8 \
  --iters 330 --learning-rate 1e-4 \
  --max-seq-length 1024 --grad-checkpoint --mask-prompt \
  --steps-per-eval 50 --val-batches 24 \
  --adapter-path adapters
