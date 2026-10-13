#!/bin/bash
# مَأْمَن — الاختبارات الآلية (بدون Qwen، دقيقة تقريباً). النتيجة في outputs/test_log.txt
cd "$(dirname "$0")/.." || exit 1
source .venv-mlx/bin/activate || { echo "venv not found: run run_app.command once first"; read -r; exit 1; }
{ python -u tests/test_protection.py; echo; python -u tests/test_scores.py; echo; python -u tests/test_app.py; } 2>&1 | tee outputs/test_log.txt
echo; echo "Done. You can close this window."; read -r
