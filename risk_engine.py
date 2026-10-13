"""
مَأْمَن — محرك حساب الخطر والحماية (pandas + numpy فقط)

المدخل: ملف CSV + دور كل عمود (من مصنّف Qwen أو من ملف التصنيفات) + سياق المشاركة
المخرج:
  1) تحليل الخطر:   k-anonymity، نسبة السجلات الفريدة، l-diversity، أخطر تركيبات الأعمدة
  2) الحماية:        حذف المعرّفات + تعميم تدريجي + حذف السجلات اللي بقيت مكشوفة
  3) قبل/بعد:        الخطر وفقدان المعلومات ودقة نموذج تنبؤ بسيط
  4) تقرير JSON:     هو اللي يستلمه Qwen ويشرحه بالعربي أو الإنجليزي (Qwen ما يحسب أي رقم)
كل النصوص (الإجراءات، أسماء مستويات التعميم، مستوى الخطر) بلغتين: *_ar و *_en

الاستخدام:
  python scripts/risk_engine.py data/adult_preprocessed.csv --context public --roles-dataset adult
  python scripts/risk_engine.py data/synthetic/sa_hospital.csv --context partner
  python scripts/risk_engine.py my.csv --roles outputs/classified_my_finetuned.json
"""

import argparse
import csv
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MISSING = "?"
STAR = "*"
ARABIC = re.compile("[\u0600-\u06FF]")  # بايثون فقط، ما نمرره لـ pandas (pyarrow ما يدعم \u)


def N(ar, en):
    return {"ar": ar, "en": en}


DATE_LIKE = re.compile(r"\d{4}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")
RAW, HIDE = N("القيمة الأصلية", "Original value"), N("إخفاء العمود", "Hide column")

# سياق المشاركة يحدد العتبة
CONTEXTS = {
    "public": {"k": 10, "l": 2, "max_suppression": 0.05, "label": N("نشر عام", "Public release")},
    "partner": {"k": 5, "l": 2, "max_suppression": 0.05, "label": N("جهة شريكة موثوقة", "Trusted partner")},
    "internal": {"k": 3, "l": 1, "max_suppression": 0.05, "label": N("استخدام داخلي", "Internal use")},
}
RISK_LEVELS = {"critical": N("حرج", "Critical"), "high": N("مرتفع", "High"),
               "medium": N("متوسط", "Medium"), "low": N("منخفض", "Low")}
# The level is read from the index alone, so the number, the label and the colour can never disagree.
# index = 100 × average(1/k): 10 = on average 1 in 10 (k = 10, the public-release threshold);
# 33.3 = 1 in 3 (k = 3, the lowest threshold Ma'man accepts, internal use); 66.7 = most rows have at most
# one look-alike. Low (green) ≤ 10 < Medium (amber) ≤ 33.3 < High (red) ≤ 66.7 < Critical (red).
# A low level is not a promise of anonymity: it is an average over the details checked.
LEVEL_BANDS = [(10.0, "low"), (100 / 3, "medium"), (200 / 3, "high"), (100.0, "critical")]


def _other(t):
    """قيمة «أخرى» بلغة البيانات نفسها."""
    return "أخرى" if any(ARABIC.search(x) for x in t.head(200)) else "Other"

# هرمية تعميم بمعنى (مستوى 1، مستوى 2 ...). تنطبق تلقائياً لو 80% من قيم العمود موجودة في المستوى الأول.
# مأخوذة من الهرميات المستخدمة لـ Adult في أدبيات الخصوصية، ومعها نسخ عربية للداتا السعودية.
def _groups(spec):
    return {v: g for g, vals in spec.items() for v in vals}


SEMANTIC = [
    [_groups({"North America": ["United-States", "Canada", "Outlying-US(Guam-USVI-etc)"],
              "Latin America": ["Mexico", "Puerto-Rico", "Cuba", "Jamaica", "Dominican-Republic", "Haiti",
                                "Guatemala", "El-Salvador", "Honduras", "Nicaragua", "Trinadad&Tobago",
                                "Columbia", "Ecuador", "Peru"],
              "Europe": ["England", "Germany", "Italy", "Poland", "Portugal", "France", "Ireland", "Greece",
                         "Scotland", "Yugoslavia", "Hungary", "Holand-Netherlands"],
              "Asia": ["India", "Philippines", "China", "Vietnam", "Japan", "Taiwan", "Iran", "Cambodia",
                       "Thailand", "Laos", "Hong", "South"]}),
     _groups({"United-States": ["North America"], "Other": ["Latin America", "Europe", "Asia"]})],
    [_groups({"Primary": ["Preschool", "1st-4th", "5th-6th", "7th-8th"],
              "Secondary (no diploma)": ["9th", "10th", "11th", "12th"], "High school": ["HS-grad"],
              "Some college / Associate": ["Some-college", "Assoc-acdm", "Assoc-voc"],
              "Bachelors": ["Bachelors"], "Graduate": ["Masters", "Prof-school", "Doctorate"]}),
     _groups({"Below college": ["Primary", "Secondary (no diploma)", "High school"],
              "College+": ["Some college / Associate", "Bachelors", "Graduate"]})],
    [_groups({"Married": ["Married-civ-spouse", "Married-AF-spouse", "Married-spouse-absent"],
              "Never married": ["Never-married"], "Previously married": ["Divorced", "Separated", "Widowed"]}),
     _groups({"Married": ["Married"], "Not married": ["Never married", "Previously married"]})],
    [_groups({"Private": ["Private"], "Self-employed": ["Self-emp-inc", "Self-emp-not-inc"],
              "Government": ["Federal-gov", "Local-gov", "State-gov"], "Not working": ["Without-pay", "Never-worked"]})],
    [_groups({"White-collar": ["Adm-clerical", "Exec-managerial", "Prof-specialty", "Sales", "Tech-support"],
              "Blue-collar": ["Craft-repair", "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct",
                              "Transport-moving"],
              "Service": ["Other-service", "Priv-house-serv", "Protective-serv", "Armed-Forces"]})],
    [_groups({"Spouse": ["Husband", "Wife"], "Relative": ["Own-child", "Other-relative"],
              "Not in family": ["Not-in-family", "Unmarried"]})],
    [_groups({"White": ["White"], "Non-white": ["Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]})],
    # عربي
    [_groups({"سعودي": ["سعودي"],
              "خليجي": ["إماراتي", "كويتي", "بحريني", "قطري", "عماني"],
              "عربي": ["مصري", "يمني", "سوداني", "أردني", "سوري", "لبناني", "فلسطيني", "مغربي", "تونسي", "جزائري",
                       "عراقي", "موريتاني", "ليبي", "صومالي"],
              "آسيوي": ["هندي", "باكستاني", "بنغلاديشي", "فلبيني", "إندونيسي", "نيبالي", "سريلانكي", "أفغاني",
                        "صيني", "كوري", "ياباني", "ماليزي", "تايلندي", "فيتنامي", "إيراني", "تركي"],
              "أفريقي": ["إثيوبي", "نيجيري", "كيني", "إريتري", "تشادي", "أوغندي", "جنوب أفريقي"],
              "أوروبي وأمريكي": ["أمريكي", "بريطاني", "كندي", "أسترالي", "فرنسي", "ألماني", "إيطالي", "إسباني",
                                 "برازيلي", "روسي"]}),
     _groups({"سعودي": ["سعودي"], "غير سعودي": ["خليجي", "عربي", "آسيوي", "أفريقي", "أوروبي وأمريكي"]})],
    [_groups({"الوسطى": ["الرياض", "بريدة"], "الغربية": ["جدة", "مكة المكرمة", "المدينة المنورة"],
              "الشرقية": ["الدمام", "الخبر"], "الجنوبية": ["أبها"], "الشمالية": ["تبوك", "حائل"]})],
    [_groups({"شمال الرياض": ["النرجس", "الملقا", "الياسمين", "حطين", "العارض", "قرطبة"],
              "وسط الرياض": ["العليا", "الروضة", "الربوة"], "غرب الرياض": ["السويدي"]})],
    [_groups({"متزوج": ["متزوج"], "غير متزوج": ["أعزب"], "سبق له الزواج": ["مطلق", "أرمل"]}),
     _groups({"متزوج": ["متزوج"], "غير متزوج": ["غير متزوج", "سبق له الزواج"]})],
    [_groups({"التقنية": ["مهندس برمجيات", "مهندس شبكات"], "المالية": ["محاسب", "محلل مالي"],
              "الموارد البشرية": ["أخصائي موارد بشرية", "أخصائي توظيف"], "المبيعات": ["مندوب مبيعات", "مدير مبيعات"],
              "العمليات": ["مدير مشروع", "منسق عمليات"]})],
    [_groups({"الحاسب": ["علوم الحاسب", "نظم المعلومات"], "الهندسة": ["هندسة كهربائية", "هندسة مدنية"],
              "إدارة الأعمال": ["محاسبة", "تسويق"], "العلوم": ["كيمياء", "فيزياء"], "الطب": ["طب بشري"]})],
    [_groups({"صحي": ["طبيب", "ممرض"], "هندسي وتقني": ["مهندس", "مبرمج", "فني"], "تعليم": ["معلم"],
              "إداري ومالي": ["محاسب", "موظف مبيعات", "مدير مشروع", "أخصائي موارد بشرية"]})],
]


# ═════════════════════════════════════════════
# الأدوار
# ═════════════════════════════════════════════
def roles_from_flags(rows):
    roles = {"direct": [], "qi": [], "sensitive": [], "record": []}
    for r in rows:
        flag = lambda k: str(r.get(k)).strip().lower() in ("1", "true")  # noqa: E731
        if flag("direct_identifier"):
            roles["direct"].append(r["column"])
        if flag("record_identifier"):
            roles["record"].append(r["column"])
        if flag("quasi_identifier") and not flag("direct_identifier"):
            roles["qi"].append(r["column"])
        if flag("sensitive_attribute"):
            roles["sensitive"].append(r["column"])
    return roles


def load_roles(csv_path, roles_json=None, roles_dataset=None):
    """من مخرجات maman_classifier.py، أو من ملف التصنيفات المعتمد حسب اسم الداتا سِت."""
    if roles_json:
        return roles_from_flags(json.loads(Path(roles_json).read_text(encoding="utf-8")))
    ann = ROOT / "data" / "column_annotations_multi.csv"
    names = [roles_dataset] if roles_dataset else [Path(csv_path).stem, Path(csv_path).stem.split("_")[0]]
    with ann.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for name in names:
        match = [r for r in rows if r["dataset"] == name]
        if match:
            return roles_from_flags(match)
    raise SystemExit(f"No roles for {csv_path}. Pass --roles <classifier output json> or --roles-dataset <name>.")


# ═════════════════════════════════════════════
# هرم التعميم لكل عمود QI: المستوى 0 = القيمة الأصلية ... الأخير = "*"
# ═════════════════════════════════════════════
def _nice(x):
    if x <= 0:
        return 1
    p = 10 ** math.floor(math.log10(x))
    return next(m * p for m in (1, 2, 5, 10) if m * p >= x)


def _as_text(s):
    return s.astype("string").fillna(MISSING).astype(str)


def _numeric_levels(s, name):
    v = pd.to_numeric(s, errors="coerce")
    lo, hi = float(v.min()), float(v.max())
    is_int = bool((v.dropna() % 1 == 0).all())
    if 0 <= lo and hi <= 120 and is_int and re.search(r"age|عمر|omr", name, re.I):
        widths, unit = [5, 10, 20, 40], N(" سنوات", "-year")
    else:
        span = hi - lo if hi > lo else 1
        widths = sorted({_nice(span / 20), _nice(span / 8), _nice(span / 3), _nice(span / 2)})
        unit = N("", "")

    def binner(w):
        def f(_s):
            start = np.floor(v / w) * w
            if is_int:
                lab = start.map(lambda a: f"{int(a)}-{int(a + w - 1)}" if pd.notna(a) else MISSING)
            else:
                lab = start.map(lambda a: f"{a:g}-{a + w:g}" if pd.notna(a) else MISSING)
            return lab.astype(str)
        return f

    levels = [lambda _s: _as_text(s)] + [binner(w) for w in widths] + [lambda _s: pd.Series(STAR, index=s.index)]
    names = [RAW] + [N(f"فئات كل {w:g}{unit['ar']}", f"{w:g}{unit['en']} ranges") for w in widths] + [HIDE]
    return levels, names


def _date_levels(s):
    d = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=False)

    def fmt(pattern):
        return lambda _s: d.dt.strftime(pattern).fillna(MISSING).astype(str)

    def years(width):
        def f(_s):
            y = (d.dt.year // width) * width
            return y.map(lambda a: f"{int(a)}-{int(a) + width - 1}" if pd.notna(a) else MISSING).astype(str)
        return f

    levels = [lambda _s: _as_text(s), fmt("%Y-%m"), fmt("%Y"), years(5), years(10),
              lambda _s: pd.Series(STAR, index=s.index)]
    return levels, [RAW, N("الشهر والسنة فقط", "Month and year only"), N("السنة فقط", "Year only"),
                    N("فترات كل 5 سنوات", "5-year periods"), N("فترات كل 10 سنوات", "10-year periods"), HIDE]


def _categorical_levels(s):
    t = _as_text(s)
    other = _other(t)
    freq = t.value_counts()
    known = t[t != MISSING]
    for maps in SEMANTIC:
        if len(known) and known.isin(maps[0].keys()).mean() > 0.8:
            levels, names, cur = [lambda _s: t], [RAW], t
            for i, m in enumerate(maps):
                cur = cur.map(lambda x, m=m: m.get(x, x if x == MISSING else other))
                levels.append(lambda _s, cur=cur: cur)
                groups = sorted(set(m.values()))[:6]
                names.append(N(f"تجميع في فئات أعم ({'، '.join(groups)})", f"Grouped into {', '.join(groups)}"))
            return levels + [lambda _s: pd.Series(STAR, index=s.index)], names + [HIDE]
    if len(freq) <= 2:
        return [lambda _s: t, lambda _s: pd.Series(STAR, index=s.index)], [RAW, HIDE]

    def keep(values, label):
        keepset = set(values)
        return (lambda _s: t.where(t.isin(keepset), other)), label

    rare = freq[freq >= max(2, 0.01 * len(t))].index
    candidates = [keep(rare, N("دمج القيم النادرة (أقل من 1%) في «أخرى»", "Rare values (<1%) merged into 'Other'")),
                  keep(freq.index[:5], N("الإبقاء على أكثر 5 قيم فقط", "Only the 5 most common values kept")),
                  keep(freq.index[:2], N("الإبقاء على أكثر قيمتين فقط", "Only the 2 most common values kept"))]
    levels, names, seen = [lambda _s: t], [RAW], {len(freq)}
    for fn, label in candidates:
        n_distinct = fn(None).nunique()
        if n_distinct not in seen and n_distinct > 1:
            levels.append(fn)
            names.append(label)
            seen.add(n_distinct)
    return levels + [lambda _s: pd.Series(STAR, index=s.index)], names + [HIDE]


def _looks_like_date(s):
    sample = [str(x) for x in s.dropna().head(200)]
    if not sample or sum(bool(DATE_LIKE.search(x)) for x in sample) / len(sample) <= 0.8:
        return False
    return pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean() > 0.8


def build_hierarchy(df, col):
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 12:
        fns, names = _numeric_levels(s, col)
    elif not pd.api.types.is_numeric_dtype(s) and _looks_like_date(s):
        fns, names = _date_levels(s)
    else:
        fns, names = _categorical_levels(s)
    values, kept = [], []
    for fn, name in zip(fns, names):
        v = fn(s)
        d = v.nunique()
        if values and d == values[-1].nunique() and name is not HIDE:
            continue  # مستوى ما يغيّر شي (مثل فئات كل 1)
        values.append(v)
        kept.append(name)
    return {"values": values, "names": kept, "distinct": [v.nunique() for v in values],
            "entropy": [_entropy(v) for v in values]}


# ═════════════════════════════════════════════
# مقاييس الخطر
# ═════════════════════════════════════════════
def class_ids(frame, cols):
    if not cols:
        return np.zeros(len(frame), dtype=np.int64)
    return frame[cols].groupby(cols, sort=False, dropna=False).ngroup().to_numpy()


def k_per_record(gid):
    return np.bincount(gid)[gid]


def sensitive_codes(values):
    codes, uniques = pd.factorize(_as_text(values))
    return codes, len(uniques)


def l_per_record(gid, codes, n_codes):
    pairs = np.unique(gid.astype(np.int64) * n_codes + codes)
    return np.bincount(pairs // n_codes, minlength=gid.max() + 1)[gid]


L_MAX_DISTINCT = 20  # private columns with at most this many values: l-diversity; more: checked for linkage


def l_columns(df, qi, sensitive, max_distinct=L_MAX_DISTINCT):
    """l-diversity ينطبق على الأعمدة الحساسة اللي مو QI (لأن QI جزء من تعريف المجموعة نفسها)،
    واللي قيمها فئات (≤ 20 قيمة). الأرقام المستمرة مثل الأرباح أغلبها صفر فشرط l عليها ما له معنى."""
    # A column with one value for everyone (or empty everywhere) tells nothing more about one person than about the
    # whole file, and no grouping can make it diverse: checking it would leave out every row. (Empty counts as a value:
    # "HIV" or empty is two values.)
    return [s for s in sensitive if s not in qi and 2 <= df[s].nunique(dropna=False) <= max_distinct]


def risk_metrics(frame, qi, sensitive, ctx, codes=None):
    """sensitive هنا = الأعمدة اللي ينطبق عليها l-diversity فقط. codes: ترميز مسبق للسرعة."""
    n = len(frame)
    if n == 0:
        return {"records": 0, "min_k": None, "unique_pct": 0.0, "below_k_pct": 0.0, "violating_pct": 0.0,
                "avg_reidentification_risk_pct": 0.0, "l_diversity": {}}
    gid = class_ids(frame, qi)
    k = k_per_record(gid)
    out = {
        "records": int(n),
        "equivalence_classes": int(gid.max() + 1),
        "min_k": int(k.min()),
        "unique_records": int((k == 1).sum()),
        "unique_pct": round(100 * float((k == 1).mean()), 2),
        "below_k_records": int((k < ctx["k"]).sum()),
        "below_k_pct": round(100 * float((k < ctx["k"]).mean()), 2),
        "avg_reidentification_risk_pct": round(100 * float((1 / k).mean()), 4),
        "max_reidentification_risk_pct": round(100 / int(k.min()), 2),
        "l_diversity": {},
    }
    violating = k < ctx["k"]
    for s in sensitive:
        c, nc = codes[s] if codes else sensitive_codes(frame[s])
        lval = l_per_record(gid, c, nc)
        homogeneous = (lval == 1) & (k >= 2)  # مجموعة كاملة تشترك بنفس القيمة الحساسة
        out["l_diversity"][s] = {
            "min_l": int(lval.min()),
            "below_l_pct": round(100 * float((lval < ctx["l"]).mean()), 2),
            "homogeneous_groups_records_pct": round(100 * float(homogeneous.mean()), 2),
        }
        violating |= lval < ctx["l"]
    out["violating_records"] = int(violating.sum())
    out["violating_pct"] = round(100 * float(violating.mean()), 2)
    return out


def level_for_index(x):
    x = float(x or 0)
    return next(key for top, key in LEVEL_BANDS if x <= top + 1e-9)


def risk_level(m, ctx=None, n_direct=0):
    """Level from the index only (see LEVEL_BANDS). Direct identifiers raise the level through the index itself:
    analyze() counts them among the details someone could know, so a file with names or ID numbers scores high."""
    key = level_for_index(m.get("avg_reidentification_risk_pct", 0))
    return {"risk_level": key, "risk_level_ar": RISK_LEVELS[key]["ar"], "risk_level_en": RISK_LEVELS[key]["en"]}


def looks_unique_text(s, min_unique=0.9):
    """Text where nearly every row has its own value (IDs in any format, account codes): works like a row key,
    whatever it looks like. Free text and dates are handled elsewhere; exact numbers are made into ranges."""
    if pd.api.types.is_numeric_dtype(s) or _looks_like_date(s) or is_free_text(s):
        return False
    v = s.dropna().astype(str).str.strip()
    v = v[v != ""]
    return len(v) >= 20 and v.nunique() / len(v) >= min_unique


PERSON_NAME_HEADER = re.compile(r"(?i)(^|[_\s\-])(name|full.?name|first.?name|last.?name|surname|esm|ism)($|[_\s\-])|اسم")


def looks_like_person_name(s, header):
    """A column whose header says it is a name and whose values are many different words: a person's name,
    even if the model did not label it (removed, never generalized or masked)."""
    if not PERSON_NAME_HEADER.search(str(header)) or pd.api.types.is_numeric_dtype(s):
        return False
    v = s.dropna().astype(str).str.strip()
    v = v[v != ""]
    return len(v) >= 10 and v.nunique() >= max(10, 0.05 * len(v)) and v.str.contains(r"[^\W\d_]").mean() > 0.9


RECORD_KEY = re.compile(r"^[A-Za-z]{1,6}[-_]?\d{3,}[A-Za-z]{0,2}$")


def looks_like_record_key(s, min_unique=0.95):
    """رقم سجل/عضوية: شبه فريد لكل صف، وشكله كود (حروف + أرقام مثل GYM-774102) أو أرقام بنفس الطول أو تسلسل 1..n.
    هذا عمود ما يتعمّم أبداً (تعميم «أكثر 5 قيم» يحوّله لـ Other ويخلي العمود بلا معنى)، مكانه الحذف."""
    v = s.dropna().astype(str).str.strip()
    v = v[v != ""]
    if len(v) < 20 or v.nunique() / len(v) < min_unique:
        return False
    if v.str.fullmatch(RECORD_KEY).mean() >= 0.9:
        return True
    if v.str.fullmatch(r"\d+").all():
        lengths = v.str.len()
        if lengths.min() == lengths.max() and lengths.min() >= 5:
            return True
        nums = v.astype("int64")
        return nums.max() - nums.min() + 1 <= 1.05 * len(v)  # تسلسل مثل 1..n
    return False


def extra_identifiers(df, roles, protected, ctx_name, max_distinct=50):
    """أعمدة ما لها أي دور، لكن لو أحد يعرفها عن الشخص (مثل نوع الاشتراك أو عدد الزيارات) تخلي النسخة المحمية
    تكسر شرط المشاركة. يرجّع {العمود: مؤشر النسخة المحمية لو كان معروف} للأعمدة اللي تكسر الشرط."""
    ctx = CONTEXTS[ctx_name]
    taken = {c for v in roles.values() for c in v}
    qi = [c for c in roles["qi"] if c in protected.columns]
    out = {}
    for c in df.columns:
        if c in taken or c not in protected.columns:
            continue
        n = df[c].nunique(dropna=True)
        if not 2 <= n <= max_distinct:
            continue
        frame = protected[qi + [c]].copy()
        for q in qi + [c]:
            frame[q] = _as_text(frame[q])
        m = risk_metrics(frame, qi + [c], [], ctx)
        if m["min_k"] is not None and m["min_k"] < ctx["k"] and m["below_k_pct"] > 100 * ctx["max_suppression"]:
            out[c] = m["avg_reidentification_risk_pct"]
    return out


def proxy_columns(df, roles, max_distinct=50, agree=0.99):
    """أعمدة غير مصنّفة حساسة لكنها تفضح عمود حساس: كل قيمة حساسة تقابلها قيمة وحدة هنا
    (مثل العيادة: كل تشخيص له عيادة وحدة، فـ«النفسية» تكشف «اكتئاب»). يرجّع {العمود: العمود الحساس}."""
    taken = set(roles["direct"]) | set(roles["record"]) | set(roles["sensitive"])
    out = {}
    for s in roles["sensitive"]:
        if s not in df.columns:
            continue
        sv = _as_text(df[s])
        ns = sv.nunique()
        if not 2 <= ns <= min(max_distinct, len(df) // 2):
            continue
        for c in df.columns:
            if c in taken or c in out:
                continue
            cv = _as_text(df[c])
            if not 2 <= cv.nunique() <= ns:
                continue
            pair = pd.DataFrame({"s": sv.to_numpy(), "c": cv.to_numpy()})
            modal = pair.groupby(["s", "c"]).size().groupby(level=0).max().sum()
            if modal / len(pair) >= agree:
                out[c] = s
    return out


def risky_combinations(frame, qi, top=5, max_size=3):
    rows = []
    for r in range(1, min(max_size, len(qi)) + 1):
        for combo in itertools.combinations(qi, r):
            k = k_per_record(class_ids(frame, list(combo)))
            rows.append({"columns": list(combo), "unique_pct": round(100 * float((k == 1).mean()), 2)})
    # الأخطر = أعلى نسبة فرادة، ولو تساوت نفضّل الأقل أعمدة (أسهل على المهاجم)
    rows.sort(key=lambda x: (-x["unique_pct"], len(x["columns"])))
    seen, out = set(), []
    for row in rows:
        if row["unique_pct"] == 0:
            break
        key = frozenset(row["columns"])
        if any(prev <= key for prev in seen):  # تجاهل تركيبة تحتوي تركيبة أخطر منها
            continue
        seen.add(key)
        out.append(row)
        if len(out) == top:
            break
    return out


# ═════════════════════════════════════════════
# الحماية: تعميم تدريجي (greedy) ثم حذف السجلات المتبقية
# ═════════════════════════════════════════════
def _apply(levels, hier, base):
    frame = base.copy()
    for c, lv in levels.items():
        frame[c] = hier[c]["values"][lv]
    return frame


def _entropy(v):
    p = v.value_counts(normalize=True).to_numpy()
    return float(-(p * np.log2(p)).sum())


def _info_loss(levels, hier):
    """الأعمدة اللي اختارها المستخدم كمهمة لتحليله (hier[c]["weight"] > 1) تُحسب معلوماتها بوزن أعلى.
فقدان المعلومات = نسبة المعلومات (بالـ bits، إنتروبي) اللي راحت من أعمدة QI مجتمعة.
    كذا عمود العمر بفئات 10 سنوات يحسب له معلومات أكثر من عمود الجنس، فالخوارزمية تحافظ على الأعمدة المفيدة."""
    if not levels:
        return 0.0
    total = sum(hier[c]["weight"] * hier[c]["entropy"][0] for c in levels)
    kept = sum(hier[c]["weight"] * hier[c]["entropy"][lv] for c, lv in levels.items())
    return 1.0 - kept / total if total else 0.0



# ═════════════════════════════════════════════
# Columns outside the core: masking, free text, one-extra-detail check
# ═════════════════════════════════════════════
EMAIL_MASK = "*****"  # fixed length: never reveals how long the original name before @ was
EMAIL = re.compile(r"^[\w.+\-]+@([\w\-]+\.[\w.\-]+)$")
SA_PHONE = re.compile(r"^(?:\+966|00966|0)?(5\d)\d{7}$")
SA_IBAN = re.compile(r"^SA\d{2}(\d{2})\d{18}$")


def mask_identifier(s):
    """A direct identifier reduced to a coarse, non-unique part — or None when nothing safe remains.
    email → *****@domain · Saudi mobile → operator prefix (05x) · Saudi IBAN → bank code. Names, national IDs and
    anything else: None (removed). A masked column is kept only if it also passes the one-extra-detail check."""
    v = s.dropna().astype(str).str.replace(r"[\s\-]", "", regex=True)
    if len(v) == 0:
        return None
    for pattern, label, fmt in ((EMAIL, N("الجزء قبل @ صار *****، والنطاق باقٍ", "part before @ replaced with *****, domain kept"),
                                 lambda m: EMAIL_MASK + "@" + m.group(1).lower()),
                                (SA_PHONE, N("بادئة المشغّل فقط (05x)", "operator prefix only (05x)"),
                                 lambda m: "0" + m.group(1) + "xxxxxxx"),
                                (SA_IBAN, N("رمز البنك فقط", "bank code only"), lambda m: "SA bank " + m.group(1))):
        if v.str.fullmatch(pattern).mean() >= 0.9:
            def f(x, pattern=pattern, fmt=fmt):
                if pd.isna(x):
                    return x
                m = pattern.fullmatch(re.sub(r"[\s\-]", "", str(x)))
                # a malformed value is replaced whole, never partly shown
                return fmt(m) if m else (EMAIL_MASK if pattern is EMAIL else STAR)
            return s.map(f), label
    return None


def is_free_text(s):
    """Sentences people typed (notes, comments): can hold names or details nobody checked."""
    if pd.api.types.is_numeric_dtype(s) or _looks_like_date(s):
        return False
    v = s.dropna().astype(str).str.strip()
    words = v.str.contains(r"\s").mean() if len(v) else 0  # sentences have spaces; long codes (IBAN, IDs) do not
    return len(v) > 0 and v.str.len().mean() >= 25 and v.nunique() / len(v) >= 0.3 and words >= 0.5


def extra_candidates(df, roles, core, base_cols):
    """Columns kept in the copy that are not in the core but could still help single someone out:
    unlabelled columns (Qwen may have missed them) and precise numbers or dates (exact salary, exact amounts,
    exact dates) even when they are private. Low-cardinality private columns (a diagnosis) are not here:
    they are what gets protected (l-diversity), and they stay as they are."""
    labelled = set(roles["direct"]) | set(roles["record"]) | set(roles["qi"]) | set(roles["sensitive"])
    out = {}
    for c in base_cols:
        if c in core:
            continue
        s = df[c]
        n = s.nunique(dropna=True)
        if n < 2:
            continue
        if is_free_text(s):
            out[c] = "free_text"
        elif (pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=False) > L_MAX_DISTINCT) or _looks_like_date(s):
            out[c] = "precise"
        elif c in roles["sensitive"] and s.nunique(dropna=False) > L_MAX_DISTINCT:  # same count as l_columns
            # a private column with too many values for the l-diversity check is checked for linkage instead,
            # so every kept column is covered by one of the two checks
            out[c] = "precise"
        elif c not in labelled:
            out[c] = "unlabelled"
    return out


def one_extra_ok(frame, core_visible, col, values, ctx):
    """Would someone who knows the core details AND this one extra detail single people out?
    Rows whose (core + this value) group is smaller than k get this value blanked (cell suppression); the level
    is accepted only if that blanks at most the suppression budget. Returns (ok, metrics, rows to blank)."""
    f = frame[core_visible].copy()
    f[col] = values
    gid = class_ids(f, core_visible + [col])
    small = k_per_record(gid) < ctx["k"]
    ok = small.mean() <= ctx["max_suppression"]
    m = risk_metrics(f, core_visible + [col], [], ctx)
    m["blanked"] = int(small.sum())
    return ok, m, small


def protect(df, roles, ctx, keep=(), strict=False):
    """strict=True: every unlabelled column is assumed known together with the core (the old, strictest model)."""
    if strict:
        extra_qi = [c for c in extra_candidates(df, roles, roles["qi"], df.columns)
                    if c not in roles["sensitive"] and not is_free_text(df[c]) and c not in roles["direct"] + roles["record"]]
        roles = {**roles, "qi": list(dict.fromkeys(list(roles["qi"]) + extra_qi))}
    drop = [c for c in dict.fromkeys(roles["direct"] + roles["record"]) if c in df.columns]
    masked = {}
    for c in roles["direct"]:
        if c in df.columns and c not in roles["record"]:
            m = mask_identifier(df[c])
            if m is not None:
                masked[c] = m
    qi = [c for c in roles["qi"] if c in df.columns and c not in drop]
    sensitive = [c for c in roles["sensitive"] if c in df.columns and c not in drop]
    base = df.drop(columns=drop)
    lcols = l_columns(df, qi, sensitive)
    codes = {s: sensitive_codes(df[s]) for s in lcols}
    hier = {c: {**build_hierarchy(df, c), "weight": 4.0 if c in keep else 1.0} for c in qi}
    budget = ctx["max_suppression"] * len(df)

    def violations(lv):
        frame = pd.DataFrame({c: hier[c]["values"][v] for c, v in lv.items()}, index=df.index)
        return risk_metrics(frame, qi, lcols, ctx, codes)["violating_records"]

    # Top-Down Specialization: نبدأ والكل مخفي (آمن دائماً: مجموعة وحدة فيها كل السجلات)
    # ثم نرجّع التفاصيل خطوة خطوة: كل مرة نختار الخطوة اللي ترجّع أكثر معلومات وتظل ضمن الحد المسموح.
    levels = {c: len(hier[c]["values"]) - 1 for c in qi}
    current = violations(levels)
    steps = []
    # Step 1 — keep as many columns as possible: bring back hidden columns at their coarsest level, the one that
    # adds the fewest violations first (a column kept coarse is worth more than one column kept very exact).
    while True:
        best = None
        for c in qi:
            top = len(hier[c]["values"]) - 1
            if levels[c] != top or top == 0:
                continue
            trial = {**levels, c: top - 1}
            v = violations(trial)
            if v <= budget and (best is None or (v, -hier[c]["weight"]) < best[0]):
                best = ((v, -hier[c]["weight"]), c, v)
        if best is None:
            break
        _, c, current = best
        levels[c] -= 1
        steps.append({"column": c, "level": levels[c], "action": hier[c]["names"][levels[c]],
                      "violating_after": int(current), "phase": "keep"})
    # Step 2 — then add detail where it still fits (Top-Down Specialization, as before)
    while True:
        best = None
        for c in qi:
            if levels[c] == 0:
                continue
            trial = {**levels, c: levels[c] - 1}
            v = violations(trial)
            if v > budget:
                continue
            gain = _info_loss(levels, hier) - _info_loss(trial, hier)
            key = (gain / (1 + v - current) if v > current else gain * 2, -v)
            if best is None or key > best[0]:
                best = (key, c, v)
        if best is None:
            break
        _, c, current = best
        levels[c] -= 1
        steps.append({"column": c, "level": levels[c], "action": hier[c]["names"][levels[c]],
                      "violating_after": int(current)})

    # تحسين محلي: نجرّب نرجّع تفاصيل عمود مقابل تعميم عمود ثاني، ونقبل التبديل لو زادت المعلومات وظلّت الحماية
    for _ in range(30):
        loss_now, best = _info_loss(levels, hier), None
        for up, down in itertools.permutations(qi, 2):
            if levels[up] == 0 or levels[down] + 1 >= len(hier[down]["values"]):
                continue
            for d_up in (1, 2):
                if levels[up] - d_up < 0:
                    continue
                trial = {**levels, up: levels[up] - d_up, down: levels[down] + 1}
                gain = loss_now - _info_loss(trial, hier)
                if gain <= 1e-9 or (best and gain <= best[0]):
                    continue
                v = violations(trial)
                if v <= budget:
                    best = (gain, trial, v, up, down)
        if best is None:
            break
        _, levels, current, up, down = best
        steps.append({"column": up, "level": levels[up], "action": hier[up]["names"][levels[up]],
                      "swapped_with": down, "violating_after": int(current)})

    cands_all = extra_candidates(df, roles, qi, [c for c in base.columns if c not in qi])
    for c in masked:
        cands_all[c] = "masked"
    ext_hier = {}
    base_entropy = {c: _entropy(_as_text(df[c])) for c in masked}  # a masked column keeps a share of the original

    def level_lower(t):
        return t[0].lower() + t[1:] if t[:1].isascii() else t

    def finish(levels):
        """Apply the core levels, leave out the rows that still break k or l, then check every column outside
        the core as 'core + this one detail' on the rows kept (finest level that passes, else removed)."""
        protected = _apply(levels, hier, base)
        gid = class_ids(protected, qi)
        k = k_per_record(gid)
        bad = k < ctx["k"]
        for s in lcols:
            bad |= l_per_record(gid, *codes[s]) < ctx["l"]
        suppressed = int(bad.sum())
        protected = protected.loc[~bad]
        hidden = [c for c in qi if hier[c]["values"][levels[c]].eq(STAR).all()]  # all "*": no use, removed
        protected = protected.drop(columns=hidden)
        core_visible = [c for c in qi if c not in hidden]
        kept_rows = protected.index
        view = protected[core_visible].astype(str)
        extras = {}
        for c, kind in cands_all.items():
            if kind == "free_text":
                if c in protected.columns:
                    protected = protected.drop(columns=[c])
                extras[c] = {"kind": kind, "result": "removed", "level": HIDE}
                continue
            if c not in ext_hier:  # levels are built once on all rows, then read on the rows kept
                source = masked[c][0] if kind == "masked" else df[c]
                ext_hier[c] = build_hierarchy(pd.DataFrame({c: source}), c)
            h = ext_hier[c]
            chosen = None
            for i, vals in enumerate(h["values"]):
                if h["names"][i] is HIDE or (kind == "masked" and i > 0):  # masked: as masked, or not at all
                    break
                vals = vals.loc[kept_rows]
                ok, m, small = one_extra_ok(view, core_visible, c, vals.astype(str).to_numpy(), ctx)
                if ok:
                    vals = vals.astype(object).copy()
                    vals[small] = None  # blanked: these rows would stand out with this value
                    chosen = (i, vals, m)
                    break
            if chosen is None:
                if c in protected.columns:
                    protected = protected.drop(columns=[c])
                extras[c] = {"kind": kind, "result": "removed", "level": HIDE}
                continue
            i, vals, m = chosen
            protected[c] = vals.to_numpy()
            name = h["names"][i]
            if kind == "masked":
                name = masked[c][1] if i == 0 else N(f"{masked[c][1]['ar']}، {h['names'][i]['ar']}",
                                                     f"{masked[c][1]['en']}, {level_lower(h['names'][i]['en'])}")
            result = "masked" if kind == "masked" else ("kept" if i == 0 else "generalized")
            if m["blanked"]:
                # some cells were emptied: the column is no longer "as it is", and the report says so
                if result == "kept":
                    result, name = "generalized", N("القيم كما هي", "Exact values")
                name = N(f"{name['ar']}، مع تفريغ {m['blanked']} خلية نادرة",
                         f"{name['en']}, with {m['blanked']:,} rare cells blanked")
            extras[c] = {"kind": kind, "result": result, "level": name, "level_index": i,
                         "index_if_also_known": m["avg_reidentification_risk_pct"], "blanked": m["blanked"]}
        # Information kept, column by column (entropy share: 1 = exact); columns the user marked count 4×.
        def kept_share(h, lv, c):
            full = base_entropy.get(c, h["entropy"][0])
            return h["entropy"][lv] / full if full > 0 else 1.0
        # a column kept is worth 0.5, plus up to 0.5 for the detail it keeps; a masked identifier only 0.1
        def worth(c, h, lv, kind=None):
            w = 4.0 if c in keep else 1.0
            return w * (0.1 if kind == "masked" else 0.5 + 0.5 * kept_share(h, lv, c))
        score = sum(worth(c, hier[c], levels[c]) for c in core_visible)
        score += sum(worth(c, ext_hier[c], e["level_index"], e["kind"]) for c, e in extras.items() if e["result"] != "removed")
        return protected, extras, suppressed, hidden, score

    # Balance: a core column made one step coarser can leave room for several other columns. Accept a step only
    # when it keeps MORE columns overall (privacy is unchanged: coarser core = bigger groups; every extra is
    # re-checked). Ties keep the more exact core.
    protected, extras, suppressed, hidden, n_kept = finish(levels)
    for _ in range(8):
        best = None
        for c in qi:
            for lv in range(levels[c] + 1, len(hier[c]["values"])):  # one or more steps coarser (up to hidden)
                trial = {**levels, c: lv}
                res = finish(trial)
                if res[4] > n_kept and (best is None or res[4] > best[1][4]):
                    best = (trial, res, c)
                    break  # the least coarsening of this column that helps
        if best is None:
            break
        levels, (protected, extras, suppressed, hidden, n_kept), c = best
        steps.append({"column": c, "level": levels[c], "action": hier[c]["names"][levels[c]],
                      "phase": "balance", "columns_kept": n_kept})
    order = [c for c in df.columns if c in protected.columns]
    protected = protected[order].copy()
    for c in protected.columns:  # a missing value stays empty in the file (levels use "?" only for grouping)
        missing = df.loc[protected.index, c].isna().to_numpy()
        if missing.any() and protected[c].dtype == object or str(protected[c].dtype).startswith(("str", "string")):
            protected.loc[missing, c] = None

    actions = []
    drop_now = [c for c in drop if not (c in extras and extras[c]["result"] == "masked")]
    if drop_now:
        drop = drop_now
        actions.append({"type": "drop_identifiers", "columns": drop,
                        "text_ar": "حذف المعرّفات المباشرة ومفاتيح السجلات: " + "، ".join(drop),
                        "text_en": "Removed direct identifiers and record keys: " + ", ".join(drop)})
    for c in qi:
        if levels[c] > 0 and c not in hidden:
            name = hier[c]["names"][levels[c]]
            actions.append({"type": "generalize", "column": c, "level": name,
                            "text_ar": f"تعميم «{c}»: {name['ar']}", "text_en": f"Generalized '{c}': {name['en']}"})
    for c, e in extras.items():
        if e["result"] == "masked":
            actions.append({"type": "mask", "column": c, "level": e["level"],
                            "text_ar": f"إخفاء جزئي لـ«{c}»: {e['level']['ar']}", "text_en": f"Masked '{c}': {e['level']['en']}"})
        elif e["result"] == "generalized":
            actions.append({"type": "generalize_extra", "column": c, "level": e["level"],
                            "text_ar": f"تعميم «{c}»: {e['level']['ar']}", "text_en": f"Generalized '{c}': {e['level']['en']}"})
        elif e["result"] == "removed":
            why = (N("نص حر قد يحوي أسماء أو تفاصيل", "free text can contain names or details") if e["kind"] == "free_text"
                   else N("حتى بعد التعميم يميّز الأشخاص", "still singles people out, even made less exact"))
            actions.append({"type": "remove_extra", "column": c, "reason": why,
                            "text_ar": f"حذف «{c}»: {why['ar']}", "text_en": f"Removed '{c}': {why['en']}"})
    if hidden:
        actions.append({"type": "hide_columns", "columns": hidden,
                        "text_ar": "إخفاء أعمدة كاملة لأن تعميمها ما كفى: " + "، ".join(hidden),
                        "text_en": "Hid columns that could not be generalized enough: " + ", ".join(hidden)})
    if suppressed:
        pct = 100 * suppressed / len(df)
        actions.append({"type": "suppress_records", "records": suppressed,
                        "text_ar": f"حذف {suppressed} سجل ({pct:.1f}%) بقيت قابلة للتمييز",
                        "text_en": f"Deleted {suppressed} records ({pct:.1f}%) from the file because they were still identifiable"})
    dropped_ids = [c for c in drop if not (c in extras and extras[c]["result"] == "masked")]
    status = {}
    for c in df.columns:
        if c in dropped_ids:
            status[c] = "removed"
        elif c in extras:
            status[c] = extras[c]["result"]
        elif c in qi:
            status[c] = "removed" if c in hidden else ("kept" if levels[c] == 0 else "generalized")
        else:
            status[c] = "kept" if c in protected.columns else "removed"
    counts = {k: sum(v == k for v in status.values()) for k in ("kept", "generalized", "masked", "removed")}
    same = 0
    for c in protected.columns:
        same += int((df.loc[protected.index, c].astype(str).to_numpy() == protected[c].astype(str).to_numpy()).sum())
    utility_summary = {"columns": counts, "columns_total": len(df.columns), "rows_kept": int(len(protected)),
                       "rows_total": int(len(df)),
                       "cells_unchanged_pct": round(100 * same / max(1, df.shape[0] * df.shape[1]), 1)}
    return protected, {
        "dropped_identifiers": dropped_ids, "masked": {c: e["level"] for c, e in extras.items() if e["result"] == "masked"},
        "extras": extras, "column_status": status, "utility_summary": utility_summary,
        "levels": {c: hier[c]["names"][lv] for c, lv in levels.items()},
        "hidden_columns": hidden, "suppressed_records": suppressed,
        "suppressed_pct": round(100 * suppressed / len(df), 2),
        "info_loss_pct": round(100 * _info_loss(levels, hier), 1), "steps": steps, "actions": actions,
    }, qi, lcols


# ═════════════════════════════════════════════
# قياس الفائدة: انحدار لوجستي (softmax) بـ numpy يتنبأ بعمود هدف، على نفس السجلات قبل وبعد
# ═════════════════════════════════════════════
def _discretize(s):
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 12:
        return pd.qcut(s.rank(method="first"), 10, labels=False).astype(str)
    return _as_text(s)


def logistic_accuracy(X, y, train_mask, iters=300, lr=0.5, l2=1e-4):
    onehot = pd.get_dummies(pd.DataFrame({c: _discretize(X[c]) for c in X.columns}), dtype=np.float32)
    A = np.hstack([onehot.to_numpy(), np.ones((len(onehot), 1), np.float32)])
    labels, yc = np.unique(_as_text(y).to_numpy(), return_inverse=True)
    tr = train_mask.to_numpy()
    Atr, ytr = A[tr], yc[tr]
    Y = np.eye(len(labels), dtype=np.float32)[ytr]
    W = np.zeros((A.shape[1], len(labels)), np.float32)
    for _ in range(iters):
        Z = Atr @ W
        P = np.exp(Z - Z.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        W -= lr * (Atr.T @ (P - Y) / len(Atr) + l2 * W)
    pred = (A[~tr] @ W).argmax(axis=1)
    truth = yc[~tr]
    baseline = (truth == np.bincount(ytr).argmax()).mean()
    return float((pred == truth).mean()), float(baseline)


def utility(original, protected, qi, target):
    if not target or target not in protected.columns or not qi or protected.empty:
        return None
    features = [c for c in qi if c != target]
    if not features:  # the only revealing column is the one we would predict: nothing to measure
        return None
    idx = protected.index
    protected = protected.copy()
    for c in features:
        if c not in protected.columns:  # عمود مخفي = ما فيه معلومة
            protected[c] = STAR
    rng = np.random.default_rng(0)
    train_mask = pd.Series(rng.random(len(idx)) < 0.7, index=idx)
    acc_before, baseline = logistic_accuracy(original.loc[idx, features], original.loc[idx, target], train_mask)
    acc_after, _ = logistic_accuracy(protected[features], protected[target], train_mask)
    # لو النموذج ما يتفوّق على خط الأساس قبل الحماية، فما فيه علاقة نقيسها (مثل الداتا الاصطناعية العشوائية)
    meaningful = (acc_before - baseline) >= 0.02
    return {"target": target, "model": "Logistic regression", "features": features, "meaningful": bool(meaningful),
            "accuracy_before_pct": round(100 * acc_before, 1), "accuracy_after_pct": round(100 * acc_after, 1),
            "accuracy_drop_pts": round(100 * (acc_before - acc_after), 1),
            "majority_baseline_pct": round(100 * baseline, 1)}


def pick_target(df, sensitive):
    for s in sensitive:
        if 2 <= df[s].nunique() <= 10:
            return s
    return None


# ═════════════════════════════════════════════
# التحليل الكامل
# ═════════════════════════════════════════════
def residual_risk(frame, core, cols, ctx):
    """The index if someone knew the core details plus ONE of these columns (worst one reported).
    Same formula, same rows; it shows what the core-only index leaves out."""
    out = {}
    base = frame[core].astype(str) if core else pd.DataFrame(index=frame.index)
    for c in cols:
        if c not in frame.columns:
            continue
        f = base.copy()
        f[c] = _as_text(frame[c])
        out[c] = risk_metrics(f, core + [c], [], ctx)["avg_reidentification_risk_pct"]
    worst = max(out, key=out.get) if out else None
    return {"per_column": out, "worst_column": worst, "worst_index": out.get(worst) if worst else None}


def analyze(df, roles, context="public", target=None, keep=(), strict=False):
    ctx = CONTEXTS[context]
    taken = {c for v in roles.values() for c in v}
    moved = [c for c in df.columns if (looks_like_record_key(df[c]) or looks_unique_text(df[c])) and
             (c in roles["qi"] or c not in taken)]  # رقم سجل انصنّف QI أو ما انصنّف أبداً
    if moved:  # رقم سجل انصنّف خطأ كـ QI: يُحذف بدل ما يتعمّم
        roles = {**roles, "qi": [c for c in roles["qi"] if c not in moved],
                 "record": list(dict.fromkeys(list(roles["record"]) + moved))}
    names = [c for c in df.columns if c not in roles["direct"] and looks_like_person_name(df[c], c)]
    if names:  # a person's name the model missed: removed like any direct identifier
        roles = {**roles, "direct": list(roles["direct"]) + names,
                 "qi": [c for c in roles["qi"] if c not in names],
                 "sensitive": [c for c in roles["sensitive"] if c not in names]}
    # A column labelled "names a person" that has only a few values, each shared by many people (a bank, a
    # branch), cannot point to one person on its own: it is protected as a detail that reveals people in
    # combination instead (generalized or removed with the others), never kept unchecked.
    demoted = [c for c in roles["direct"] if c in df.columns and c not in roles["record"] and c not in names
               and df[c].nunique(dropna=True) <= L_MAX_DISTINCT
               and df[c].value_counts().min() >= max(x["k"] for x in CONTEXTS.values())]
    if demoted:
        roles = {**roles, "direct": [c for c in roles["direct"] if c not in demoted],
                 "qi": list(dict.fromkeys(list(roles["qi"]) + demoted))}
    qi = [c for c in roles["qi"] if c in df.columns]
    sensitive = [c for c in roles["sensitive"] if c in df.columns]
    direct = [c for c in roles["direct"] if c in df.columns]
    original = df.copy()
    for c in qi:
        original[c] = _as_text(df[c])

    # The file as it is: someone who knows a person's name or ID number can look it up, so direct identifiers count
    # among the details someone could know (same formula; without them the index would understate a file with names).
    known = qi + [c for c in direct if c not in qi]
    for c in direct:
        original[c] = _as_text(df[c])
    before = risk_metrics(original, known, l_columns(df, qi, sensitive), ctx)
    before_details = risk_metrics(original, qi, l_columns(df, qi, sensitive), ctx)  # revealing details only
    combos = risky_combinations(original, qi)
    protected, protection, qi_used, sens_used = protect(df, roles, ctx, keep, strict=strict)
    after = risk_metrics(protected, [c for c in qi_used if c in protected.columns], sens_used, ctx)
    core_after = [c for c in qi_used if c in protected.columns]
    extra_cols = [c for c, e in protection["extras"].items() if e["kind"] != "free_text"]
    residual = {"before": residual_risk(original, qi, [c for c in extra_cols if c in df.columns], ctx),
                "after": residual_risk(protected, core_after, [c for c in extra_cols if c in protected.columns], ctx)}
    target = target or pick_target(df, sens_used or sensitive)

    report = {
        "context": context, "context_ar": ctx["label"]["ar"], "context_en": ctx["label"]["en"],
        "priority_columns": list(keep),
        "record_keys_moved": moved, "names_found": names, "identifiers_demoted": demoted,
        "thresholds": {"k": ctx["k"], "l": ctx["l"], "max_suppression_pct": 100 * ctx["max_suppression"]},
        "roles": {"direct": direct, "qi": qi, "sensitive": sensitive,
                  "record": [c for c in roles["record"] if c in df.columns]},
        "before": {**before, **risk_level(before, ctx, len(direct)), "known_columns": known},
        "before_details": before_details,
        "risky_combinations": combos,
        "protection": protection,
        "after": {**after, **risk_level(after, ctx, 0)},
        "residual": residual, "strict": strict,
        "utility": utility(df, protected, qi_used, target),
    }
    return report, protected


def summary(r, lang="ar"):
    b, a, p, u = r["before"], r["after"], r["protection"], r["utility"]
    ar = lang == "ar"
    sep = "، " if ar else ", "
    t = r["thresholds"]
    lines = [(f"السياق: {r['context_ar']} (k ≥ {t['k']}، l ≥ {t['l']})" if ar else
              f"Context: {r['context_en']} (k >= {t['k']}, l >= {t['l']})"),
             (f"قبل: الخطر {b['risk_level_ar']} | سجلات فريدة {b['unique_pct']}% | تحت العتبة {b['below_k_pct']}% "
              f"| مؤشر خطر إعادة التعرّف {b['avg_reidentification_risk_pct']} من 100" if ar else
              f"Before: risk {b['risk_level_en']} | unique records {b['unique_pct']}% | below threshold "
              f"{b['below_k_pct']}% | re-identification risk index {b['avg_reidentification_risk_pct']} out of 100 (not a probability)")]
    if r["roles"]["direct"]:
        lines.append(("  ⚠ فيه معرّفات مباشرة: " if ar else "  ! Direct identifiers present: ") + sep.join(r["roles"]["direct"]))
    for s, lv in b["l_diversity"].items():
        if lv["homogeneous_groups_records_pct"]:
            lines.append(f"  ⚠ {lv['homogeneous_groups_records_pct']}% من السجلات في مجموعات كلها نفس قيمة «{s}»" if ar else
                         f"  ! {lv['homogeneous_groups_records_pct']}% of records sit in groups that all share the same '{s}'")
    for c in r["risky_combinations"][:3]:
        lines.append(f"  أخطر تركيبة: {' + '.join(c['columns'])} → تكشف {c['unique_pct']}% من السجلات" if ar else
                     f"  Riskiest combination: {' + '.join(c['columns'])} -> singles out {c['unique_pct']}% of records")
    lines.append("الحماية:" if ar else "Protection:")
    lines += [f"  • {x['text_' + lang]}" for x in p["actions"]]
    lines.append(f"بعد: الخطر {a['risk_level_ar']} | أقل k = {a.get('min_k')} | سجلات فريدة {a.get('unique_pct')}% "
                 f"| مؤشر خطر إعادة التعرّف {a.get('avg_reidentification_risk_pct')} من 100" if ar else
                 f"After: risk {a['risk_level_en']} | min k = {a.get('min_k')} | unique records {a.get('unique_pct')}% "
                 f"| re-identification risk index {a.get('avg_reidentification_risk_pct')} out of 100 (not a probability)")
    if u and not u["meaningful"]:
        lines.append(f"الفائدة: ما نقدر نقيسها هنا: الأعمدة ما تتنبأ بـ«{u['target']}» حتى قبل الحماية "
                     f"| فقدان المعلومات {p['info_loss_pct']}%" if ar else
                     f"Utility: not measurable here (the columns do not predict '{u['target']}' even before "
                     f"protection) | information loss {p['info_loss_pct']}%")
    elif u:
        lines.append(f"الفائدة: دقة توقّع «{u['target']}» {u['accuracy_before_pct']}% ← {u['accuracy_after_pct']}% "
                     f"(خط الأساس {u['majority_baseline_pct']}%) | فقدان المعلومات {p['info_loss_pct']}%" if ar else
                     f"Utility: '{u['target']}' prediction accuracy {u['accuracy_before_pct']}% -> "
                     f"{u['accuracy_after_pct']}% (baseline {u['majority_baseline_pct']}%) "
                     f"| information loss {p['info_loss_pct']}%")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--context", default="public", choices=list(CONTEXTS))
    ap.add_argument("--roles", help="classifier output json (outputs/classified_*.json)")
    ap.add_argument("--roles-dataset", help="dataset name in data/column_annotations_multi.csv")
    ap.add_argument("--target", help="column for the utility test (default: a low-cardinality sensitive column)")
    ap.add_argument("--keep", default="", help="comma-separated columns the user needs most (kept in more detail)")
    ap.add_argument("--lang", default="en", choices=["en", "ar"])  # التطبيق إنجليزي؛ العربي متاح
    a = ap.parse_args()

    path = Path(a.csv)
    df = pd.read_csv(path, skipinitialspace=True)
    roles = load_roles(path, a.roles, a.roles_dataset)
    keep = [c.strip() for c in a.keep.split(",") if c.strip()]
    report, protected = analyze(df, roles, a.context, a.target, keep)

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    stem = f"{path.stem}_{a.context}" + ("_keep" if keep else "")
    (out / f"risk_report_{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    protected.to_csv(out / f"protected_{stem}.csv", index=False, encoding="utf-8-sig")
    print(summary(report, a.lang))
    print(f"\nSaved: outputs/risk_report_{stem}.json, outputs/protected_{stem}.csv")


if __name__ == "__main__":
    main()
