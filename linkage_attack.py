"""
مَأْمَن — محاكاة هجوم ربط (linkage attack)

السيناريو: مهاجم يعرف عن N شخص موجودين في الملف معلومات عامة (أعمدة الـ QI: العمر، الجنس، الحي، التواريخ...)،
مثل اللي ينعرف من السوشال ميديا أو من جار أو زميل. يحاول يطابقها مع:
  1) الملف بعد حذف الأسماء والمعرّفات فقط (اللي يسويه أغلب الناس)
  2) نسخة مَأْمَن المحمية
ونقيس: كم شخص حدّد هويته بالضبط (تطابق واحد وهو الشخص الصحيح)،
وكم شخص عرف قيمته الحساسة (مثل التشخيص) بيقين، حتى لو ما حدد سجله بالضبط.

  python scripts/linkage_attack.py data/synthetic/sa_hospital.csv --context partner
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import risk_engine as R  # noqa: E402


def _generalizers(df, qi, protection):
    """لكل عمود QI ظاهر في النسخة المحمية: دالة تحوّل القيمة الأصلية لنفس التعميم اللي طُبّق."""
    out = {}
    for c in qi:
        if c in protection["hidden_columns"]:
            continue
        h = R.build_hierarchy(df, c)
        level = next(i for i, n in enumerate(h["names"]) if n == protection["levels"][c])
        out[c] = h["values"][level]  # نفس طول df ونفس ترتيبه
    return out


def _attack(release, target_keys, target_truth, target_idx, sensitive):
    """release: DataFrame فيه أعمدة المطابقة + العمود الحساس. target_keys: قيم المهاجم لكل هدف (نفس الأعمدة)."""
    cols = list(target_keys.columns)
    if cols:
        groups = release.groupby(cols, sort=False, dropna=False)
        index_of = groups.indices
    reidentified = learned = 0
    sizes = []
    for i, key in enumerate(target_keys.itertuples(index=False, name=None)):
        if cols:
            k = key[0] if len(cols) == 1 else key
            rows = index_of.get(k, np.array([], dtype=int))
        else:
            rows = np.arange(len(release))
        sizes.append(len(rows))
        if len(rows) == 1 and release.index[rows[0]] == target_idx[i]:
            reidentified += 1
        if sensitive and len(rows) > 0:
            values = set(release[sensitive].iloc[rows])
            if values == {target_truth[i]}:
                learned += 1
    n = len(target_keys)
    return {"reidentified": reidentified, "reidentified_pct": round(100 * reidentified / n, 1),
            "sensitive_learned": learned, "sensitive_learned_pct": round(100 * learned / n, 1),
            "median_matches": float(np.median(sizes)) if sizes else 0.0}


def simulate(df, roles, report, protected, n_targets=500, seed=0):
    qi = [c for c in roles["qi"] if c in df.columns]
    drop = [c for c in dict.fromkeys(roles["direct"] + roles["record"]) if c in df.columns]
    lcols = R.l_columns(df, qi, [c for c in roles["sensitive"] if c in df.columns])
    sensitive = lcols[0] if lcols else None
    rng = np.random.default_rng(seed)
    targets = np.sort(rng.choice(len(df), size=min(n_targets, len(df)), replace=False))
    target_idx = df.index[targets]

    text = {c: R._as_text(df[c]) for c in qi + ([sensitive] if sensitive else [])}
    truth = text[sensitive].iloc[targets].tolist() if sensitive else [None] * len(targets)

    # 1) الأسماء محذوفة فقط: المهاجم يطابق القيم الأصلية كما هي
    names_removed = pd.DataFrame({c: text[c] for c in qi}, index=df.index)
    if sensitive:
        names_removed[sensitive] = text[sensitive]
    keys = names_removed[qi].iloc[targets].reset_index(drop=True)
    before = _attack(names_removed, keys, truth, target_idx, sensitive)

    # 2) نسخة مَأْمَن: المهاجم يعمّم معلوماته بنفس الطريقة، والأعمدة المخفية ما يقدر يستخدمها
    gen = _generalizers(df, qi, report["protection"])
    visible = [c for c in qi if c in gen]
    release = pd.DataFrame({c: protected[c].astype(str) for c in visible}, index=protected.index)
    if sensitive:
        release[sensitive] = R._as_text(protected[sensitive])
    keys = pd.DataFrame({c: gen[c].iloc[targets].astype(str).to_numpy() for c in visible},
                        index=range(len(targets)))  # no visible column left: every target matches the whole file
    after = _attack(release, keys, truth, target_idx, sensitive)

    return {"targets": int(len(targets)), "known_columns": qi, "sensitive": sensitive,
            "names_removed": before, "protected": after}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--context", default="public", choices=list(R.CONTEXTS))
    ap.add_argument("--roles-dataset")
    ap.add_argument("--keep", default="")
    ap.add_argument("--targets", type=int, default=500)
    a = ap.parse_args()
    df = pd.read_csv(a.csv, skipinitialspace=True)
    roles = R.load_roles(a.csv, None, a.roles_dataset)
    keep = [c.strip() for c in a.keep.split(",") if c.strip()]
    report, protected = R.analyze(df, roles, a.context, None, keep)
    res = simulate(df, roles, report, protected, a.targets)
    b, p = res["names_removed"], res["protected"]
    print(f"Attacker knows {', '.join(res['known_columns'])} about {res['targets']} people in the file.")
    print(f"  Names removed only : identified {b['reidentified']} ({b['reidentified_pct']}%), "
          f"learned '{res['sensitive']}' of {b['sensitive_learned']} ({b['sensitive_learned_pct']}%), "
          f"median matches {b['median_matches']:g}")
    print(f"  Ma'man protected   : identified {p['reidentified']} ({p['reidentified_pct']}%), "
          f"learned '{res['sensitive']}' of {p['sensitive_learned']} ({p['sensitive_learned_pct']}%), "
          f"median matches {p['median_matches']:g}")


if __name__ == "__main__":
    main()
