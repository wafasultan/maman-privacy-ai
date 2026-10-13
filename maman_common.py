"""
مَأْمَن — أدوات مشتركة: وصف العمود، طبقة القواعد، البرومبت، قراءة المخرجات، الاسترجاع (RAG).

كل السكربتات (البناء، التدريب، التصنيف، التقييم) تستخدم نفس البرومبت ونفس صيغة الأعمدة،
عشان النموذج المدرَّب يشوف وقت الاستخدام نفس اللي شافه وقت التدريب بالضبط.
"""

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ANNOTATIONS = DATA / "column_annotations_multi.csv"
FEATURES = DATA / "column_features_multi.jsonl"

MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
ROLE_KEYS = ["direct_identifier", "quasi_identifier", "sensitive_attribute", "record_identifier"]
ALL_KEYS = ROLE_KEYS + ["needs_review"]
N_SAMPLES = 5


# ─────────────────────────────────────────────
# وصف العمود
# ─────────────────────────────────────────────
def describe_column(dataset, name, series, n_samples=N_SAMPLES):
    import pandas as pd

    non_null = series.dropna()
    non_null = non_null[non_null.astype(str).str.strip() != ""]
    distinct = non_null.drop_duplicates()
    samples = distinct.sample(min(n_samples, len(distinct)), random_state=0) if len(distinct) else distinct
    rows = len(series)
    return {
        "dataset": str(dataset),
        "column": str(name),
        "dtype": "numeric" if pd.api.types.is_numeric_dtype(series) else "text",
        "rows": rows,
        "missing_count": int(rows - len(non_null)),
        "distinct_count": int(len(distinct)),
        "unique_value_ratio": round(len(distinct) / max(rows, 1), 6),
        "samples": [_clean_value(v) for v in samples.tolist()],
    }


def _clean_value(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def prompt_features(f):
    """اللي يشوفه النموذج عن العمود (بدون الأرقام اللي ما تفيد في التصنيف)."""
    return {
        "dataset": f["dataset"],
        "column": f["column"],
        "dtype": f["dtype"],
        "unique_value_ratio": round(float(f["unique_value_ratio"]), 4),
        "samples": [str(s)[:60] for s in f["samples"][:N_SAMPLES]],
    }


# ─────────────────────────────────────────────
# طبقة القواعد: تمسك الواضح قبل النموذج
# ─────────────────────────────────────────────
RULES = [
    ("email", re.compile(r"^[\w.+\-]+@[\w\-]+\.[\w.\-]+$"), "عنوان بريد إلكتروني"),
    ("sa_phone", re.compile(r"^(\+966|00966|0)?5\d{8}$"), "رقم جوال سعودي"),
    ("sa_iban", re.compile(r"^SA\d{22}$"), "رقم آيبان سعودي"),
    ("national_id", re.compile(r"^[12]\d{9}$"), "رقم هوية وطنية أو إقامة"),
]
# أعمدة الوقت ممكن تكون Unix timestamps (10 أرقام تبدأ بـ1) فتشبه رقم الهوية
TIME_HINT = re.compile(r"(time|date|ts|created|updated|epoch|وقت|تاريخ)", re.IGNORECASE)


def apply_rules(f, threshold=0.8):
    values = [re.sub(r"[\s\-]", "", str(v)) for v in f["samples"] if str(v).strip() not in ("", "nan")]
    values = [re.sub(r"\.0$", "", v) for v in values]
    if not values:
        return None
    for rule_name, pattern, what in RULES:
        ratio = sum(bool(pattern.match(v)) for v in values) / len(values)
        if ratio < threshold:
            continue
        if rule_name == "national_id" and (float(f["unique_value_ratio"]) < 0.5 or TIME_HINT.search(f["column"])):
            continue
        result = {k: False for k in ALL_KEYS}
        result.update(direct_identifier=True, reason=f"القيم تطابق نمط {what}", source=f"rule:{rule_name}")
        return result
    return None


ID_CODE = re.compile(r"^[A-Za-z]{1,6}[-_]?\d{3,}[A-Za-z]{0,2}$")


def id_guard(f, pred):
    """لو النموذج قال QI لعمود شبه فريد شكله كود (GYM-774102، C3004483)، فهو رقم سجل مو معلومة تميّز بالتجميع.
    نصحّحه لـ record_identifier (ونحافظ على direct لو النموذج قاله)."""
    if not pred.get("quasi_identifier") or pred.get("record_identifier") or float(f["unique_value_ratio"]) < 0.95:
        return pred
    values = [re.sub(r"\s", "", str(v)) for v in f["samples"] if str(v).strip() not in ("", "nan")]
    codes = sum(bool(ID_CODE.match(v)) or (v.isdigit() and len(v) >= 5) for v in values)
    if values and codes / len(values) >= 0.8:
        return {**pred, "quasi_identifier": False, "record_identifier": True,
                "reason": (pred.get("reason") or "") + " (صُحّح: رقم سجل فريد لكل صف)", "guard": "record_key"}
    return pred


# ─────────────────────────────────────────────
# البرومبت (نفسه في التدريب والتصنيف)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are MA'MAN, a privacy schema classifier. Classify ONE column of a dataset about people.

THREAT MODEL: an attacker may know a person's demographic, location, education, employment, or household facts and combine them to match records. Private financial and health details are not assumed externally available.

ROLES (independent; a column may have several, or none):
1. direct_identifier: identifies a specific person on its own: names, national ID or iqama, passport, phone, email, IBAN or account numbers, medical record numbers, customer, employee or student numbers, IP addresses, device IDs, exact addresses.
2. quasi_identifier: does not identify alone but narrows down candidate records when combined with externally known facts: age, birth date, sex, nationality, race, city, district, zip code, occupation, education, marital status, household composition, event dates, device or browser type. Few categories do NOT rule out this role.
3. sensitive_attribute: disclosure would harm the person: health (diagnoses, lab results, BMI, smoking, alcohol), finances (income, salary, balances, loans, credit scores, capital gains, personal bills and order amounts; a public product or service price is NOT sensitive), education records (grades, academic warnings), performance ratings. This project also treats race, sex, marital status, household relationship and household composition (number of children, dependents, siblings, family size) as sensitive, in addition to quasi-identifier.
4. record_identifier: a key for a row or a record in a system (row number, order ID, claim ID, session ID, customer or employee number). Uniqueness alone does NOT make a record identifier: statistical weights, measurements and amounts can be nearly unique.
needs_review: true only if the column name and sample values are not enough to know what the column means (cryptic abbreviations, codes without a codebook).

Return exactly one JSON object, no Markdown, no other text:
{"direct_identifier": false, "quasi_identifier": true, "sensitive_attribute": false, "record_identifier": false, "needs_review": false, "reason": "سبب قصير بالعربي"}
All five flags are JSON booleans. "reason" is one short Arabic sentence."""


def answer_json(labels, reason):
    out = {k: bool(labels[k]) for k in ALL_KEYS}
    out["reason"] = reason
    return json.dumps(out, ensure_ascii=False)


def build_messages(features, examples=()):
    user = ""
    if examples:
        user += "SIMILAR LABELED COLUMNS (guidance, not the answer):\n"
        for e in examples:
            user += (json.dumps(prompt_features(e), ensure_ascii=False) + "\n→ " +
                     answer_json(e["labels"], e["reason"]) + "\n")
        user += "\n"
    user += "COLUMN TO CLASSIFY:\n" + json.dumps(prompt_features(features), ensure_ascii=False)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_prediction(text):
    """أول كائن JSON صالح في النص (نفس منطق classify_columns.py مع record_identifier)."""
    decoder = json.JSONDecoder()
    problem = "No JSON object found."
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        bad = [k for k in ALL_KEYS if type(obj.get(k)) is not bool]
        if bad:
            problem = "Missing or non-boolean fields: " + ", ".join(bad)
            continue
        reason = obj.get("reason")
        return {**{k: obj[k] for k in ALL_KEYS}, "reason": reason if isinstance(reason, str) else ""}
    raise ValueError(problem)


# ─────────────────────────────────────────────
# قراءة البيانات المصنّفة
# ─────────────────────────────────────────────
def load_features():
    with FEATURES.open(encoding="utf-8") as f:
        return {(r["dataset"], r["column"]): r for r in map(json.loads, f)}


def load_labeled(splits, allow_unreviewed=False):
    """يرجّع الأعمدة المصنّفة في الـ splits المطلوبة. الافتراضي: المعتمَدة (APPROVED) فقط."""
    features = load_features()
    out, skipped = [], 0
    with ANNOTATIONS.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] not in splits:
                continue
            status = r["review_status"].strip().upper()
            if status != "APPROVED" and not (allow_unreviewed and status == "PROPOSED"):
                skipped += 1
                continue
            key = (r["dataset"], r["column"])
            if key not in features:
                raise ValueError(f"Missing features for {key}")
            labels = {k: r[k].strip() == "1" for k in ALL_KEYS}
            out.append({**features[key], "labels": labels, "reason": r["reason"], "split": r["split"]})
    if not out:
        raise SystemExit(
            f"No usable labeled columns in {sorted(splits)} ({skipped} not APPROVED). "
            "Review data/column_annotations_multi.csv and set review_status=APPROVED, "
            "or pass --use-unreviewed-proposals for an explicitly unreviewed experiment.")
    return out


# ─────────────────────────────────────────────
# الاسترجاع (RAG)
# ─────────────────────────────────────────────
def _split_name(name):
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return re.sub(r"[_\-\.]+", " ", name).lower().strip()


def retrieval_text(f):
    name = _split_name(f["column"])
    samples = ", ".join(str(s)[:40] for s in f["samples"][:N_SAMPLES])
    return f"{name} | {name} | {f['dtype']} | {samples}"


class CharNgramEmbedder:
    """بديل خفيف بدون تحميل نماذج: TF-IDF على حروف متتالية (2–4). يشتغل بالعربي والإنجليزي."""

    def __init__(self, ns=(2, 3, 4)):
        self.ns = ns

    def _grams(self, text):
        t = f" {text.lower()} "
        return [t[i:i + n] for n in self.ns for i in range(len(t) - n + 1)]

    def fit(self, texts):
        df = Counter(g for t in texts for g in set(self._grams(t)))
        self.vocab = {g: i for i, g in enumerate(df)}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + df[g])) + 1 for g in self.vocab], dtype=np.float32)
        return self

    def encode(self, texts):
        m = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, t in enumerate(texts):
            for g, c in Counter(self._grams(t)).items():
                j = self.vocab.get(g)
                if j is not None:
                    m[row, j] = c
        m = np.log1p(m) * self.idf
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


E5_MODEL = "intfloat/multilingual-e5-small"


class E5Embedder:
    """multilingual-e5-small, run with transformers + torch on the CPU (MLX keeps the GPU for Qwen).
    Same recipe as sentence-transformers for this model: "query: " prefix, mean pooling over real tokens,
    L2-normalised — checked against the vectors used in the retriever experiment (scripts/check_e5.py)."""

    def __init__(self, name=E5_MODEL):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval()

    def encode(self, texts, batch=32):
        torch, out = self.torch, []
        with torch.inference_mode():
            for i in range(0, len(texts), batch):
                enc = self.tokenizer([f"query: {t}" for t in texts[i:i + batch]], padding=True, truncation=True,
                                     max_length=512, return_tensors="pt")
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out.append(torch.nn.functional.normalize(vec, dim=1).numpy())
        return np.concatenate(out).astype(np.float32)


_E5_LOADED = {}


def get_e5(name=E5_MODEL):
    """One E5 per process: the app swaps Qwen models (classifier ↔ summary) and rebuilds the classifier each time;
    reloading E5 with it grew process memory by ~0.8 GB per swap (check_e5.py, round 2). Loaded once, it stays."""
    if name not in _E5_LOADED:
        _E5_LOADED[name] = E5Embedder(name)
    return _E5_LOADED[name]


class Retriever:
    """backend:
      'e5'       multilingual-e5-small (the app default since 11 Oct). If it cannot load, it falls back to char
                 n-grams and SAYS so (printed + self.fallback_reason) — unless strict=True, which raises instead.
      'char'     TF-IDF over character 2–4-grams (the earlier default; kept for rollback).
      'e5_cache' E5 vectors computed beforehand (scripts/e5_embed.py) — retriever experiment only."""

    def __init__(self, kb, backend="e5", e5_model=E5_MODEL, cache=None, strict=False):
        self.kb = kb
        texts = [retrieval_text(x) for x in kb]
        self.backend, self.fallback_reason = None, None
        if backend == "e5_cache":
            data = np.load(cache or ROOT / "outputs" / "e5_cache.npz", allow_pickle=False)
            table = dict(zip(data["texts"].tolist(), data["vectors"]))

            def lookup(ts):
                missing = [t for t in ts if t not in table]
                if missing:
                    raise KeyError(f"E5 cache has no vector for {missing[0][:60]!r}: run scripts/e5_embed.py again")
                return np.stack([table[t] for t in ts]).astype(np.float32)
            self._encode = lookup
            self.vectors = self._encode(texts)
            self.backend = "e5"
        elif backend in ("auto", "e5"):
            try:
                embedder = get_e5(e5_model)
                self._encode = embedder.encode
                self.vectors = self._encode(texts)
                self.backend = "e5"
            except Exception as error:
                if strict:
                    raise
                self.fallback_reason = f"{type(error).__name__}: {str(error)[:160]}"
                print(f"[retriever] WARNING: E5 could not load ({self.fallback_reason}); "
                      f"FALLING BACK to char n-grams", flush=True)
        elif backend != "char":
            raise ValueError(f"unknown retriever backend: {backend}")
        if self.backend is None:
            embedder = CharNgramEmbedder().fit(texts)
            self._encode = embedder.encode
            self.vectors = self._encode(texts)
            self.backend = "char"
        print(f"[retriever] backend in use: {self.backend}"
              + (f" ({e5_model})" if self.backend == "e5" else "") + f" · {len(kb)} labelled columns", flush=True)

    def search(self, features, k=5, exclude_dataset=False):
        """exclude_dataset=True: لا يرجّع أعمدة من نفس الداتا سِت (للتدريب، عشان يشابه وقت الاختبار)."""
        q = self._encode([retrieval_text(features)])[0]
        scores = self.vectors @ q
        same = [i for i, x in enumerate(self.kb) if x["dataset"] == features["dataset"] and
                (exclude_dataset or x["column"] == features["column"])]
        scores[same] = -np.inf  # لا يسترجع نفسه أبداً
        return [self.kb[i] for i in np.argsort(-scores)[:k]]
