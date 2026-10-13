#!/bin/bash
# مَأْمَن — الفحص النهائي قبل التسليم: الاختبارات + توافق E5 مع Qwen + النسخة الأكثر أماناً بـ Qwen الحقيقي
# + تقييم 102 عمود (بالـ E5) + المقارنة. يمنع الجهاز من النوم أثناء التشغيل. النتيجة: outputs/final_checks_log.txt
cd "$(dirname "$0")/.." || exit 1
source .venv-mlx/bin/activate || { echo "venv not found"; read -r; exit 1; }
caffeinate -dimsu -w $$ &
LOG=outputs/final_checks_log.txt
{
  echo "== started $(date)"
  echo "== 1. tests/test_protection.py"; python -u tests/test_protection.py; echo "exit=$?"
  echo "== 2. tests/test_app.py"; python -u tests/test_app.py; echo "exit=$?"
  echo "== 3. scripts/check_e5.py"; python -u scripts/check_e5.py; echo "exit=$?"
  echo "== 4. tests/verify_real_copy.py (real Qwen + E5)"; python -u tests/verify_real_copy.py; echo "exit=$?"
  if [ "$1" != "--no-eval" ]; then
    echo "== 5. eval e5 (102 columns)"; python -u scripts/eval_final.py --retriever e5; echo "exit=$?"
    python -u scripts/eval_final.py --compare
  fi
  echo "== FINAL_CHECKS_DONE $(date)"
} 2>&1 | tee "$LOG"
echo; echo "Done. You can close this window."; read -r
