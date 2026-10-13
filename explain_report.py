"""
مَأْمَن — Qwen يشرح تقرير الخطر للمستخدم (عربي أو إنجليزي) — بدون ما يحسب ولا رقم

1) نحوّل تقرير risk_engine إلى «حقائق» مختصرة (FACTS) بأرقام مقرّبة.
2) Qwen2.5-3B الأصلي (بدون LoRA) يكتب شرح قصير من الحقائق فقط.
3) الكود يفحص الشرح:
     - كل رقم في النص لازم يكون موجود في الحقائق (أو تقريبه)
     - اللغة صحيحة (مافيه حروف صينية، والعربي فعلاً عربي)
   لو فشل: محاولة ثانية مع توضيح الخطأ لـ Qwen، ولو فشل مرة ثانية: قالب جاهز يكتبه الكود (مضمون 100%).

  python scripts/explain_report.py outputs/risk_report_adult_preprocessed_public_keep.json --lang ar
  python scripts/explain_report.py <report.json> --lang both
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maman_common import MODEL  # noqa: E402

MAX_TOKENS = 380
DIGITS = str.maketrans("\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
                       "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9\u066b",
                       "01234567890123456789.")
NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
CJK = re.compile("[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
OTHER_SCRIPTS = re.compile("[\u0370-\u03ff\u0400-\u04ff\u0590-\u05ff\u0900-\u097f]")  # Greek, Cyrillic, Hebrew, Hindi
ARABIC = re.compile("[\u0600-\u06ff]")
ARABIC_WORD = re.compile("[\u0600-\u06ff]+")
LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z+\-]*")
OVERCLAIM = re.compile(r"\b(impossible|completely safe|fully safe|100% safe|risk[- ]free|no risk|cannot be identified|"
                       r"can't be identified|fully anonymous|completely anonymous|guarantee\w*)\b", re.IGNORECASE)
TECH_TERMS = {"k", "l", "k-anonymity", "l-diversity", "csv", "pdpl"}


def _r(x):
    return f"{x:.1f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)


# ═════════════════════════════════════════════
# 1) الحقائق
# ═════════════════════════════════════════════
def facts(r):
    b, a, p, u, t = r["before"], r["after"], r["protection"], r["utility"], r["thresholds"]
    lines = [
        f"SHARING CONTEXT: {r['context_en']}. Rule: every person must share their combination with at least "
        f"{t['k']} people (k={t['k']})" + (f", and each group must contain at least {t['l']} different values of "
                                           f"each sensitive column (l={t['l']})." if t["l"] > 1 else "."),
        "BEFORE PROTECTION:",
        f"- risk level: {b['risk_level_en']}",
        f"- records: {b['records']}",
        f"- unique records (nobody else in the file shares their details): {_r(b['unique_pct'])}%",
        f"- records that break the rule: {_r(b['violating_pct'])}%",
        f"- re-identification risk index (0-100, an index, not a probability): {_r(b['avg_reidentification_risk_pct'])}",
        "- direct identifiers found: " + (", ".join(r["roles"]["direct"]) or "none"),
    ]
    for s, lv in r.get("before_details", b)["l_diversity"].items():
        if lv["homogeneous_groups_records_pct"]:
            lines.append(f"- {_r(lv['homogeneous_groups_records_pct'])}% of records are in groups where everyone "
                         f"has the same '{s}', so '{s}' is revealed without knowing who the person is")
    for c in r["risky_combinations"][:2]:
        lines.append(f"- risky combination: {' + '.join(c['columns'])} alone single out "
                     f"{_r(c['unique_pct'])}% of records")
    lines.append("PROTECTION APPLIED:")
    lines += [f"- {x['text_en']}" for x in p["actions"]]
    lines += [
        "AFTER PROTECTION:",
        f"- risk level: {a['risk_level_en']}",
        f"- records kept: {a['records']}",
        f"- unique records: {_r(a['unique_pct'])}%",
        f"- smallest group: {a['min_k']} people",
        f"- re-identification risk index (0-100, an index, not a probability): {_r(a['avg_reidentification_risk_pct'])}",
    ]
    if u and u["meaningful"]:
        lines.append(f"USEFULNESS: a model predicting '{u['target']}' was {_r(u['accuracy_before_pct'])}% accurate "
                     f"before and {_r(u['accuracy_after_pct'])}% after (a drop of {_r(u['accuracy_drop_pts'])} points)")
    return "\n".join(lines)


def allowed_numbers(fact_text):
    nums = set()
    for m in NUMBER.findall(fact_text):
        x = float(m.replace(",", ""))
        nums |= {x, round(x), round(x, 1)}
    return nums


def column_names(r):
    roles = r["roles"]
    return sorted({c for v in roles.values() for c in v} | set(r["protection"]["levels"]), key=len, reverse=True)


# ═════════════════════════════════════════════
# 2) الفحص
# ═════════════════════════════════════════════
PROBABILITY = re.compile(r"\b(chance|chances|probabilit\w*|likelihood|odds)\b|احتمال", re.IGNORECASE)
REMOVE_WORDS = re.compile(r"remov|delet|drop|\bhid|hidden|took out|excluded|left out", re.IGNORECASE)
GENERAL_WORDS = re.compile(r"generali|grouped|group(ed)? into|less exact|year|month|range|most common", re.IGNORECASE)


def executed_actions(p):
    """التغييرات اللي انطبقت فعلاً على الملف المحفوظ."""
    gen = {x["column"] for x in p["actions"] if x["type"] == "generalize"}
    return {"removed": set(p["dropped_identifiers"]) | set(p["hidden_columns"]), "generalized": gen}


def action_conflicts(text, actions):
    """كل جملة فرعية تذكر عمود مع فعل حذف/إخفاء لازم يكون العمود فعلاً محذوف، ومع فعل تعميم لازم يكون معمّم."""
    known = sorted(actions["removed"] | actions["generalized"], key=len, reverse=True)
    out = []
    for clause in re.split(r"[.;\n]|,\s(?:and|then)\s|\band\b(?=\s(?:we\s)?(?:remov|hid|generali|group|delet))", text):
        cols = [c for c in known if c in clause]
        if not cols:
            continue
        says_remove, says_general = bool(REMOVE_WORDS.search(clause)), bool(GENERAL_WORDS.search(clause))
        if says_remove == says_general:  # الجملة ما تحدد فعل واحد واضح
            continue
        should = actions["removed"] if says_remove else actions["generalized"]
        out += [f"'{c}' was not {'removed' if says_remove else 'generalized'}" for c in cols if c not in should]
    return out


def check(text, lang, fact_text, cols, actions=None):
    """يرجّع قائمة المشاكل (فاضية = مقبول). الكلمات الموجودة في الحقائق نفسها (أسماء أعمدة، قيم من البيانات
    مثل «سعودي» أو College+) مسموحة بأي لغة، والباقي لازم يكون بلغة الشرح."""
    problems = []
    clean = text.translate(DIGITS)
    for c in cols:  # أسماء الأعمدة ممكن فيها أرقام (G3) أو لغة ثانية
        clean = clean.replace(c, " ")
    allowed = allowed_numbers(fact_text)
    bad = [m for m in NUMBER.findall(clean)
           if not any(abs(float(m.replace(",", "")) - a) <= 0.051 for a in allowed)]
    if bad:
        problems.append("numbers not in FACTS: " + ", ".join(sorted(set(bad))))
    if CJK.search(clean) or OTHER_SCRIPTS.search(clean):
        problems.append("contains characters from another language")
    fact_rest = fact_text
    for c in cols:  # كلمة عربية مسموحة لو هي قيمة من البيانات، مو جزء مقصوص من اسم عمود (مثل «تاريخ الزيارة»)
        fact_rest = fact_rest.replace(c, " ")
    fact_words = {w.lower() for w in LATIN_WORD.findall(fact_text)} | set(ARABIC_WORD.findall(fact_rest))
    if lang == "ar":
        foreign = {w for w in LATIN_WORD.findall(clean) if w.lower() not in fact_words | TECH_TERMS}
        if foreign:
            problems.append("contains non-Arabic words: " + ", ".join(sorted(foreign)[:5]))
        if len(ARABIC.findall(clean)) < 40:
            problems.append("not written in Arabic")
    if lang == "en":
        foreign = {w for w in ARABIC_WORD.findall(clean) if w not in fact_words}
        if foreign:
            problems.append("contains Arabic words: " + " ".join(sorted(foreign)[:5]))
    if "USEFULNESS" not in fact_text and re.search(r"accura|predict|دقة|توقع", clean, re.IGNORECASE):
        problems.append("mentions prediction accuracy, which is not in FACTS")
    if PROBABILITY.search(text):
        problems.append("describes the risk index as a chance or probability")
    if actions:
        wrong = action_conflicts(text, actions)
        if wrong:
            problems.append("claims that contradict the changes made: " + "; ".join(wrong))
    overclaim = OVERCLAIM.findall(clean)
    if overclaim:
        problems.append("overclaims safety: " + ", ".join(sorted({o.lower() for o in overclaim})))
    if len(clean.strip()) < 80:
        problems.append("too short")
    return problems


# ═════════════════════════════════════════════
# 3) القالب الاحتياطي (يكتبه الكود، صحيح دائماً)
# ═════════════════════════════════════════════
def _join(items, lang):
    items = list(items)
    if lang == "ar":
        return "، ".join(items[:-1]) + " و" + items[-1] if len(items) > 1 else "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else "".join(items)


def _protection_sentence(p, lang):
    """جملة وحدة مختصرة عن الحماية، بأسماء الأعمدة فقط (بدون أرقام جديدة غير اللي في الحقائق)."""
    gen = [x["column"] for x in p["actions"] if x["type"] == "generalize"]
    sup = next((x for x in p["actions"] if x["type"] == "suppress_records"), None)
    ar = lang == "ar"
    parts = []
    if p["dropped_identifiers"]:
        parts.append("حُذفت المعرّفات المباشرة وأرقام السجلات" if ar else "direct identifiers and record numbers were removed")
    if gen:
        parts.append(("عُمّمت " if ar else "") + _join(gen, lang) + ("" if ar else " were generalized"))
    if p["hidden_columns"]:
        parts.append(("أُخفيت " if ar else "") + _join(p["hidden_columns"], lang) + ("" if ar else " were hidden"))
    if sup:
        parts.append(f"وحُذف {sup['records']} سجل بقيت قابلة للتمييز" if ar else
                     f"{sup['records']} records that were still identifiable were deleted")
    if not parts:
        return "الملف كان يحقق الشرط بدون تعديل." if ar else "The file already met the rule, so nothing was changed."
    return ("للحماية: " if ar else "To protect it, ") + _join(parts, lang).replace(" وو", " و") + "."


def template(r, lang):
    b, a, p, u, t = r["before"], r["after"], r["protection"], r["utility"], r["thresholds"]
    combo = r["risky_combinations"][0] if r["risky_combinations"] else None
    if lang == "ar":
        out = [f"قبل الحماية كان مستوى الخطر {b['risk_level_ar']}: {_r(b['unique_pct'])}% من السجلات فريدة، "
               f"ومؤشر خطر إعادة التعرّف {_r(b['avg_reidentification_risk_pct'])} من 100."]
        if r["roles"]["direct"]:
            out.append("الملف فيه معرّفات مباشرة مثل " + "، ".join(r["roles"]["direct"][:3]) + ".")
        if combo:
            out.append(f"حتى بدون الأسماء، «{' + '.join(combo['columns'])}» لحالها تميّز "
                       f"{_r(combo['unique_pct'])}% من الأشخاص.")
        out.append(_protection_sentence(p, "ar"))
        out.append(f"بعد الحماية صار الخطر {a['risk_level_ar']}: كل شخص يشبهه {a['min_k']} أشخاص على الأقل، "
                   f"ومؤشر خطر إعادة التعرّف {_r(a['avg_reidentification_risk_pct'])} من 100.")
        if u and u["meaningful"]:
            out.append(f"والبيانات ما زالت مفيدة: دقة توقّع «{u['target']}» كانت {_r(u['accuracy_before_pct'])}% "
                       f"وصارت {_r(u['accuracy_after_pct'])}%.")
        return " ".join(out)
    out = [f"Before protection the risk was {b['risk_level_en'].lower()}: {_r(b['unique_pct'])}% of records were "
           f"unique, and the re-identification risk index was {_r(b['avg_reidentification_risk_pct'])} out of 100."]
    if r["roles"]["direct"]:
        out.append("The file contained direct identifiers such as " + ", ".join(r["roles"]["direct"][:3]) + ".")
    if combo:
        out.append(f"Even without names, {' + '.join(combo['columns'])} alone single out "
                   f"{_r(combo['unique_pct'])}% of people.")
    out.append(_protection_sentence(p, "en"))
    out.append(f"After protection the risk is {a['risk_level_en'].lower()}: the smallest group has {a['min_k']} people, "
               f"and the re-identification risk index is {_r(a['avg_reidentification_risk_pct'])} out of 100.")
    if u and u["meaningful"]:
        out.append(f"The data is still useful: predicting '{u['target']}' went from {_r(u['accuracy_before_pct'])}% "
                   f"to {_r(u['accuracy_after_pct'])}% accuracy.")
    return " ".join(out)


# ═════════════════════════════════════════════
# 4) Qwen
# ═════════════════════════════════════════════
SYSTEM = """You explain a privacy risk report to a non-technical person who wants to share a data file.
Write exactly 3 sentences, plain words, no headings, no lists, no markdown:
1. Why the file was risky: name the riskiest combination of columns and the percentage of records it singles out (and mention direct identifiers if any were found).
2. What was done: summarize the protection steps. Say "hidden" for hidden columns and "grouped" or "generalized" only for generalized ones.
3. The result: the smallest group size and the re-identification risk index after protection (an index out of 100, never call it a chance or a probability){usefulness}.
Rules:
- Use ONLY the facts given. Copy every number exactly as written in FACTS. Never calculate, estimate or invent a number.
- Never claim the data is completely safe, risk-free, or that identification is impossible: the risk is reduced, not zero.
- Keep column names exactly as written, including underscores. Do not translate them or add translations."""
HEADINGS = re.compile(r"(?i)(?:^|(?<=[.\n]))\s*(?:\d\.\s*)?(?:\*\*)?(?:why the file was risky|what was done|the result)(?:\*\*)?\s*:\s*(?:\*\*)?\s*")


def strip_headings(text):
    """The three parts Qwen is asked for, without their labels (anywhere in the text), as plain sentences."""
    text = HEADINGS.sub(lambda m: " " if m.start() else "", text).strip()
    text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return re.sub(r"[ \t]{2,}", " ", text)

LANG_INSTRUCTION = {
    "ar": "Write the explanation in Arabic (simple Modern Standard Arabic). Use Western digits (0-9).",
    "en": "Write the explanation in English.",
}


class Explainer:
    """القرار بعد اختبار 2026-10-10 (outputs/explain_test.md): عربية Qwen2.5-3B خلطت لغات (fichier، риск)
    وقلبت معاني (أقل من 10 بدل 10 على الأقل)، فالشرح العربي من القالب. الإنجليزي من Qwen مع الفحص."""

    QWEN_LANGS = ("en",)

    def __init__(self, model=MODEL):
        from mlx_lm import load

        self.model, self.tokenizer = load(model)

    def _generate(self, messages):
        from mlx_lm import generate

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False).strip()

    def explain(self, report, lang="en", attempts=2, force_qwen=False):
        if lang not in self.QWEN_LANGS and not force_qwen:
            return {"lang": lang, "text": template(report, lang), "source": "template", "attempts": []}
        fact_text = facts(report)
        cols = column_names(report)
        user = f"FACTS:\n{fact_text}\n\n{LANG_INSTRUCTION[lang]}"
        system = SYSTEM.format(usefulness=", then the prediction accuracy before and after" if "USEFULNESS" in fact_text
                               else ". Do not mention prediction accuracy")
        log = []
        for _ in range(attempts):
            text = strip_headings(self._generate([{"role": "system", "content": system}, {"role": "user", "content": user}]))
            text = " ".join(text.split())
            problems = check(text, lang, fact_text, cols, executed_actions(report["protection"]))
            log.append({"text": text, "problems": problems})
            if not problems:
                return {"lang": lang, "text": text, "source": "qwen", "attempts": log}
            user += (f"\n\nYour previous answer was rejected: {'; '.join(problems)}. "
                     f"Write it again following the rules exactly.")
        return {"lang": lang, "text": template(report, lang), "source": "template", "attempts": log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--lang", default="en", choices=["en", "ar", "both"])
    a = ap.parse_args()
    report = json.loads(Path(a.report).read_text(encoding="utf-8"))
    ex = Explainer()
    for lang in (["ar", "en"] if a.lang == "both" else [a.lang]):
        res = ex.explain(report, lang)
        print(f"\n[{lang}] source={res['source']}\n{res['text']}")
        for i, att in enumerate(res["attempts"], 1):
            if att["problems"]:
                print(f"  attempt {i} rejected: {'; '.join(att['problems'])}")


if __name__ == "__main__":
    main()
