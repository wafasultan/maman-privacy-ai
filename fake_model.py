"""Stand-in for Qwen in automated tests: the approved labels when the column is known, else the rules layer."""
import csv
from functools import lru_cache

from maman_common import ALL_KEYS, ANNOTATIONS, apply_rules


@lru_cache(maxsize=1)
def _gold():
    out = {}
    with open(ANNOTATIONS, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["column"], {k: r[k] == "1" for k in ALL_KEYS})
    return out


def classify(desc):
    return dict(_gold().get(desc["column"]) or apply_rules(desc) or {k: False for k in ALL_KEYS})
