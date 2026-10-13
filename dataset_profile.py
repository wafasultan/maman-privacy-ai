"""
Ma'man — before any model runs: read the table correctly and decide what one row is.

read_table(raw)    finds where the table really starts: skips title / description lines above the header
                   and note lines under it (common in open-data portals), detects the delimiter and encoding,
                   and keeps leading zeros. A plain CSV is read exactly as before.
dataset_type(df)   'records'   one row per person (or per account, visit, …) → the full privacy check runs
                   'aggregate' each row is a total or a statistic          → nothing to anonymize; small totals checked
                   'mixed'     signs of both                               → protected as records + flagged for review
                   'uncertain' some signs of a statistics table, not enough → protected as records + flagged for review
                   Only 'aggregate' skips anonymization, and only when no sign of personal data was found.
small_counts(...)  totals below the sharing rule's group size: a count of 1 or 2 can still point to people.

All of this is code (no model), so it is fast, explainable and the same every time.
Usage:  python scripts/dataset_profile.py file.csv [more.csv …]
"""

import csv
import io
import math
import re
from collections import Counter

import pandas as pd

from maman_common import apply_rules, describe_column

NUMERIC = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?")  # same test as before: no leading zeros → numeric
NUMBER_IN_TEXT = re.compile(r"-?\d[\d,٬]*(?:\.\d+)?")
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
YEAR = re.compile(r"(19|20)\d\d")
PERIOD = re.compile(
    r"(?i)^(q[1-4]|[1-4]q|h[12]|(quarter|term|semester|month|week|wave|round)\s*\d+|(19|20)\d\d([-/_ ]?(q[1-4]|h[12]|\d{1,2}))?|"
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|"
    r"september|october|november|december)$|^(الربع|النصف|يناير|فبراير|مارس|أبريل|ابريل|مايو|يونيو|يوليو|"
    r"أغسطس|اغسطس|سبتمبر|أكتوبر|اكتوبر|نوفمبر|ديسمبر|محرم|صفر|ربيع|جمادى|رجب|شعبان|رمضان|شوال|ذو|الفصل|الشهر|الأسبوع)")
# words that describe a measure or a total (English as words, Arabic as stems: ال‍ can be attached)
AGG_EN = re.compile(r"(?i)\b(totals?|sum|counts?|number of|no\. of|average|avg|mean|median|rates?|ratio|"
                    r"percent(age)?|share|index|indicators?|statistics?|aggregate|per capita|amount of)\b|%")
AGG_AR = re.compile(r"إجمالي|اجمالي|مجموع|عدد|متوسط|معدل|نسبة|مؤشر|إحصا|احصا|قيمة")
COUNT_WORDS = re.compile(r"(?i)\b(counts?|number|no\.|cases|people|persons|individuals|patients|students|"
                         r"employees|beneficiaries|members)\b|#|عدد|حالات|أشخاص|اشخاص|أفراد|افراد|مستفيد")
INDICATOR_NAME = re.compile(r"(?i)\b(indicator|measure|metric|variable|item|kpi|description)\b|مؤشر|البند|بيان|وصف")
TOTAL_LABEL = re.compile(r"(?i)^\s*(grand\s+total|sub-?total|total|all|overall|sum|الإجمالي|الاجمالي|إجمالي|اجمالي|"
                         r"المجموع|مجموع|الكل|الإجمالي العام|المجموع الكلي)\s*:?\s*$")
# column names that describe one person (not measures: names with AGG words are skipped)
PERSON_NAME = re.compile(
    r"(?i)\b(name|first.?name|last.?name|full.?name|surname|patient|employee|customer|client|student|member|"
    r"applicant|passport|national.?id|iqama|ssn|phone|mobile|e-?mail|address|dob|birth.?date|date.?of.?birth|"
    r"salary|diagnosis|username|user.?id)\b|الاسم|اسم|المريض|مريض|الموظف|موظف|العميل|عميل|الطالب|طالب|"
    r"المتقدم|الهوية|هوية|الإقامة|إقامة|اقامة|جواز|الجوال|جوال|هاتف|البريد|بريد|العنوان|عنوان|الميلاد|ميلاد|"
    r"الراتب|راتب|التشخيص|تشخيص")
# "name" of a thing, not of a person: "اسم البند" (item name), "product name", "اسم الجهة" ...
THING_NAME = re.compile(r"(?i)\b(item|product|company|organi[sz]ation|entity|agency|department|program(me)?|project|"
                        r"indicator|measure|category|city|region|country|branch|budget)\b|البند|بند|"
                        r"المنتج|منتج|الشركة|شركة|الجهة|جهة|الإدارة|القسم|البرنامج|برنامج|المشروع|مشروع|المؤشر|مؤشر|"
                        r"الفئة|المدينة|المنطقة|الدولة|الفرع|الميزانية|الباب")
NOTE_LINE = re.compile(r"(?i)^\s*(source|sources|note|notes|footnote|prepared by|\*|†|المصدر|المصادر|ملاحظة|ملاحظات|"
                       r"إعداد|اعداد)")
RULE_WHAT = {"email": "email addresses", "sa_phone": "phone numbers", "sa_iban": "bank account numbers (IBAN)",
             "national_id": "national ID or iqama numbers"}


# ─────────────────────────────────────────────
# Reading the table
# ─────────────────────────────────────────────
def _decode(raw):
    for enc in ("utf-8-sig", "cp1256"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1"), "latin-1"


def _rows(text, delim):
    rows = []
    for r in csv.reader(io.StringIO(text), delimiter=delim):
        r = [c.strip() for c in r]
        while r and r[-1] == "":
            r.pop()
        rows.append(r)
    return rows


def _filled(r):
    return sum(1 for c in r if c)


def _pick_delimiter(text):
    lines = [ln for ln in text.splitlines()[:200] if ln.strip()]
    best, best_score = ",", 0
    for d in [",", ";", "\t", "|"]:
        spans = [len(r) for r in _rows("\n".join(lines), d) if r]
        if not spans:
            continue
        mode, freq = Counter(spans).most_common(1)[0]
        score = mode if mode > 1 and freq / len(spans) >= 0.6 else 0
        if score > best_score:  # ',' wins ties (checked first)
            best, best_score = d, score
    return best


def _labelish(c):
    c = c.strip()
    if not c:
        return False
    if YEAR.fullmatch(c):
        return True
    return not re.fullmatch(r"[-+]?[\d.,٬\s%]+", c.translate(ARABIC_DIGITS))


def _find_header(rows):
    """First row that is as wide as the table, mostly labels, and followed by rows of the same width."""
    nonblank = [i for i, r in enumerate(rows) if _filled(r)]
    if not nonblank:
        return 0, 0
    width = Counter(len(rows[i]) for i in nonblank).most_common(1)[0][0]
    for i in nonblank[:50]:
        r = rows[i]
        if len(r) < max(2, math.ceil(0.8 * width)) or _filled(r) < max(2, math.ceil(0.6 * width)):
            continue
        cells = [c for c in r if c]
        if sum(map(_labelish, cells)) / len(cells) < 0.5:
            continue
        after = [rows[j] for j in nonblank if j > i][:5]
        if not after or sum(len(a) >= 0.6 * width for a in after) / len(after) >= 0.6:
            return i, width
    return nonblank[0], width  # nothing better: the first line is the header (the old behaviour)


def _unique_names(names):
    out, seen = [], Counter()
    for j, n in enumerate(names):
        n = n or f"column_{j + 1}"
        out.append(n if not seen[n] else f"{n}.{seen[n]}")
        seen[n] += 1
    return out


def _to_numeric(df):
    """Keeps leading zeros (phones, IDs): a column is numeric only if no value starts with 0."""
    for c in df.columns:
        v = df[c].dropna().astype(str).str.strip()
        if len(v) and v.str.fullmatch(NUMERIC).all():
            df[c] = pd.to_numeric(df[c])
    return df


def read_table(raw):
    """Returns (df, info). info says what was skipped so the page can tell the user."""
    text, enc = _decode(raw)
    delim = _pick_delimiter(text)
    rows = _rows(text, delim)
    header, width = _find_header(rows)
    nonblank = [i for i, r in enumerate(rows) if _filled(r)]
    top = [i for i in nonblank if i < header]
    body = [i for i in nonblank if i > header]
    bottom = []
    while body and width >= 3 and _filled(rows[body[-1]]) <= 1 and NOTE_LINE.match(" ".join(rows[body[-1]])):
        bottom.insert(0, body.pop())  # "Source: …" / "Note: …" lines under the table
    info = {"encoding": enc, "delimiter": delim, "header_line": header + 1,
            "notes_top": [" ".join(c for c in rows[i] if c) for i in top],
            "notes_bottom": [" ".join(c for c in rows[i] if c) for i in bottom]}

    if not top and not bottom and delim == "," and enc == "utf-8-sig":
        df = pd.read_csv(io.BytesIO(raw), dtype=str, skipinitialspace=True)  # a plain CSV: read exactly as before
        return _to_numeric(df), info

    names = list(rows[header])
    ncols = max([len(names)] + [len(rows[i]) for i in body])
    names = _unique_names(names + [""] * (ncols - len(names)))
    data = [[(c if c else None) for c in rows[i]] + [None] * (ncols - len(rows[i])) for i in body]
    df = pd.DataFrame(data, columns=names, dtype="str")
    df = df.dropna(axis=1, how="all") if any(n.startswith("column_") for n in names) else df
    return _to_numeric(df), info


# ─────────────────────────────────────────────
# What is one row?
# ─────────────────────────────────────────────
NUMBER_WITH_UNIT = re.compile(
    r"(?:(?P<pre>[^\d\s][^\d]*?)\s+)?(?P<num>[-+]?\d[\d,٬]*(?:\.\d+)?)(?P<post>\s*%|\s*[$€£﷼]|\s+[^\d\s][^\d]*)?")


def parse_number(v):
    """'183 جمعية' → 183, '1,770,403,356 ريال' → 1770403356, '12.5%' → 12.5.
    Codes and dates ('C5145427', 'MRN-970472', '2026-05-29') are not numbers: None."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = NUMBER_WITH_UNIT.fullmatch(str(v).translate(ARABIC_DIGITS).strip())
    if not m:
        return None
    try:
        return float(m.group("num").replace(",", "").replace("٬", ""))
    except ValueError:
        return None


def _has_unit(v):
    m = NUMBER_WITH_UNIT.fullmatch(str(v).translate(ARABIC_DIGITS).strip())
    return bool(m and (m.group("pre") or (m.group("post") or "").strip(" %")))


def _text(s):
    return s.dropna().astype(str).str.strip().loc[lambda x: x != ""]


def _numberish(s, share=0.9):
    v = _text(s)
    return len(v) > 0 and v.map(lambda x: parse_number(x) is not None).mean() >= share


def _year_like(s):
    v = _text(s)
    return len(v) > 0 and v.str.fullmatch(YEAR).all()


def _has_agg(t):
    return bool(AGG_EN.search(t) or AGG_AR.search(t))


def dataset_type(df, notes=(), max_dim_values=50):
    """Decides what one row is from the table's shape and words. Never returns 'aggregate' when anything
    looks like personal data; when the signs disagree or are weak it says so ('mixed' / 'uncertain')."""
    n = len(df)
    cols = list(df.columns)
    agg, person = [], []          # (code, reason) pairs shown to the user
    facts = {"rows": n}

    # ── signs of personal records ──
    for c in cols:
        f = describe_column("file", c, df[c], n_samples=50)
        hit = apply_rules(f)
        if hit:
            rule = hit.get("source", "").split(":")[-1]
            person.append(("id_values", f"“{c}” holds {RULE_WHAT.get(rule, 'personal identifiers')}."))
    from risk_engine import looks_like_record_key
    for c in cols:
        if looks_like_record_key(df[c]):
            person.append(("row_key", f"“{c}” gives every row its own number, as in a list of people or accounts."))
    for c in cols:
        if PERSON_NAME.search(str(c)) and not _has_agg(str(c)) and not THING_NAME.search(str(c)):
            person.append(("person_column", f"The column name “{c}” describes a person."))

    # ── signs of a statistics table ──
    numberish = {c: _numberish(df[c]) for c in cols}
    years = {c for c in cols if _year_like(df[c])}
    dims = [c for c in cols if (not numberish[c] or c in years) and 1 <= df[c].nunique(dropna=True) <= max_dim_values]
    measures = [c for c in cols if numberish[c] and c not in years]
    others = [c for c in cols if c not in dims and c not in measures]
    facts.update(dims=dims, measures=measures)

    indicator = None
    for c in dims:
        vals = _text(df[c]).drop_duplicates()
        if len(vals) < 2:
            continue
        share = vals.map(_has_agg).mean()
        words = vals.str.split().str.len().mean()
        if (share >= 0.5 and words >= 2) or (INDICATOR_NAME.search(str(c)) and share >= 0.3):
            indicator = c
            example = vals[vals.map(_has_agg)].iloc[0]
            agg.append(("indicator", f"“{c}” lists measures (like “{example}”), not people."))
            break
    facts["indicator"] = indicator

    grid = False
    if dims and measures and not others and n >= 4:
        key = df[dims].astype(str).apply(lambda s: s.str.strip())
        if not key.duplicated().any():
            cells = math.prod(max(1, df[d].nunique(dropna=True)) for d in dims)
            grid = n / cells >= 0.5
    measure_names_agg = any(_has_agg(str(c)) for c in measures)
    if grid:
        varying = [d for d in dims if df[d].nunique(dropna=True) > 1] or dims[:1]
        reason = (f"Each row is a different combination of {' × '.join(f'“{d}”' for d in varying)}, like the cells "
                  f"of a summary table." if len(varying) > 1 else
                  f"Each row is a different “{varying[0]}” followed only by numbers, like the rows of a summary table.")
        if len(dims) >= 2 or indicator or measure_names_agg:
            agg.append(("grid", reason))
        else:
            agg.append(("grid_weak", reason))

    period_cols = [c for c in measures if PERIOD.search(str(c).strip())]
    if len(period_cols) >= 2:
        agg.append(("period_columns", f"Column names are periods (like “{period_cols[0]}”, “{period_cols[1]}”), "
                                      "so each value is a total for that period."))

    for c in [c for c in cols if not numberish[c]]:
        if c == indicator:  # measure names like "Total donations" are not total rows
            continue
        lab = _text(df[c])
        hit = lab[lab.map(lambda x: bool(TOTAL_LABEL.match(x)))]
        if len(hit) and len(hit) < len(lab):
            agg.append(("total_row", f"Rows labelled “{hit.iloc[0]}” in “{c}” add up other rows."))
            break

    for c in measures:
        v = _text(df[c])
        units = v.map(_has_unit)
        if len(v) and units.mean() >= 0.5:
            agg.append(("units", f"Values in “{c}” carry units (like “{v[units].iloc[0]}”), as in published statistics."))
            break

    note_text = " ".join(notes)
    if note_text and _has_agg(note_text):
        agg.append(("notes", "The description above the table talks about totals or amounts."))

    codes = {a for a, _ in agg}
    strong = bool(codes & {"indicator", "grid", "period_columns"})
    weak = bool(codes & {"grid_weak", "total_row", "units"})
    if strong and not person:
        kind = "aggregate"
    elif strong or (weak and person and "total_row" in codes):
        kind = "mixed"
    elif weak and not person:
        kind = "uncertain"
    else:
        kind = "records"
    return {"kind": kind, "aggregate_signs": [r for _, r in agg], "person_signs": [r for _, r in person],
            "signals": sorted(codes | {p for p, _ in person}), **facts}


def small_counts(df, profile, k, limit=None):
    """Totals that are counts and smaller than k (but above 0): a count of 1 or 2 can point to people.
    A value counts as a count when its row label or column name says so, or when the whole column is whole numbers."""
    out = []
    ind = profile.get("indicator")
    label_cols = [d for d in profile.get("dims", [])]
    for c in profile.get("measures", []):
        vals = df[c].map(parse_number)
        whole = vals.dropna().map(lambda x: x >= 0 and float(x).is_integer()).all() if vals.notna().any() else False
        col_is_count = bool(COUNT_WORDS.search(str(c)))
        for i, x in vals.items():
            if x is None or (isinstance(x, float) and math.isnan(x)):
                continue
            row_is_count = bool(ind and COUNT_WORDS.search(str(df.at[i, ind])))
            is_count = row_is_count or col_is_count or (whole and not ind)
            if is_count and float(x).is_integer() and 0 < x < k:
                where = " · ".join(str(df.at[i, d]) for d in label_cols if pd.notna(df.at[i, d]))
                out.append({"row": int(i), "column": c, "value": int(x), "where": where})
    return out[:limit] if limit else out


def main():
    import sys
    from pathlib import Path
    for p in sys.argv[1:]:
        df, info = read_table(Path(p).read_bytes())
        prof = dataset_type(df, info["notes_top"] + info["notes_bottom"])
        print(f"{Path(p).name}: {prof['kind']}  rows={len(df)} cols={len(df.columns)} header_line={info['header_line']} "
              f"skipped_top={len(info['notes_top'])} skipped_bottom={len(info['notes_bottom'])}")
        for s in prof["aggregate_signs"]:
            print("   + stats:", s)
        for s in prof["person_signs"]:
            print("   + person:", s)


if __name__ == "__main__":
    main()
