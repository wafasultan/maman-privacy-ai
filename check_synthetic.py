"""
مَأْمَن — فحص منطقية الداتا السعودية الاصطناعية (data/synthetic). كل سطر لازم يطلع 0 مشاكل.

  python scripts/check_synthetic.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_column_dataset as B  # noqa: E402

D = Path(__file__).resolve().parents[1] / "data" / "synthetic"
NAMES = {**B.MALE, **B.FEMALE}


def load(name, **kw):
    return pd.read_csv(D / f"{name}.csv", **kw)


def first(n):
    return n.split()[0]


def checks():
    h, hr, u = load("sa_hospital", dtype={"رقم_الهوية": str}), load("sa_hr"), load("sa_university")
    b, a = load("sa_bank_loans", dtype={"رقم_الهوية_الوطنية": str}), load("sa_arabizi", dtype={"hawiya": str})
    tr, dl = load("sa_travel"), load("sa_delivery")
    male = set(B.MALE)
    yield "Gender matches first name (hospital, employees, students)", sum(
        int(((d[n].map(first).isin(male)) != (d["الجنس"] == "ذكر")).sum())
        for d, n in [(h, "اسم_المريض"), (hr, "اسم_الموظف"), (u, "اسم_الطالب")])
    yield "Gender matches first name (arabizi)", int(((a["esm"].map(first).isin(male)) != (a["jins"] == "m")).sum())
    yield "First name differs from father's name", sum(
        int((d[n].str.split().str[0] == d[n].str.split().str[1]).sum())
        for d, n in [(h, "اسم_المريض"), (hr, "اسم_الموظف"), (u, "اسم_الطالب"), (b, "اسم_العميل"), (tr, "اسم_المسافر"),
                     (dl, "اسم_المستلم"), (a, "esm")])
    yield "Email built from the employee's own name", sum(
        not (e.split(".")[0] == NAMES[first(n)] and e.split("@")[0].split(".")[1].rstrip("0123456789") == B.FAMILY[n.split()[-1]])
        for n, e in zip(hr["اسم_الموظف"], hr["البريد_الإلكتروني"]))
    yield "Everyone is Saudi", sum(int((d["الجنسية"] != "سعودي").sum()) for d in (h, hr, tr))
    yield "National IDs start with 1 (Saudi citizens)", sum(
        int((~d[c].str.startswith("1")).sum()) for d, c in [(h, "رقم_الهوية"), (b, "رقم_الهوية_الوطنية"), (a, "hawiya")])
    yield "Clinic matches diagnosis", int(sum(B.CLINIC_OF[d] != c for d, c in zip(h["التشخيص"], h["العيادة"])))
    yield "HbA1c >= 6.5 only for diabetics", int(((h["السكر_التراكمي"] >= 6.5) != (h["التشخيص"] == "سكري النوع الثاني")).sum())
    yield "Job title matches department", int(sum(B.HR_JOBS[j][0] != d for j, d in zip(hr["المسمى_الوظيفي"], hr["الإدارة"])))
    age_hired = (pd.to_datetime(hr["تاريخ_التعيين"]) - pd.to_datetime(hr["تاريخ_الميلاد"])).dt.days / 365
    yield "Hired at age 22 or older", int((age_hired < 22).sum())
    yield "Major belongs to its college", int(sum(B.MAJORS[m] != c for m, c in zip(u["التخصص"], u["الكلية"])))
    admit_age = (u["سنة_القبول"] - 1442 + 2020) - pd.to_datetime(u["تاريخ_الميلاد"]).dt.year
    yield "Admitted to university at 18-19", int((~admit_age.between(18, 19)).sum())
    yield "Academic warning only when GPA < 2", int(((u["الحالة_الأكاديمية"] != "منتظم") != (u["المعدل_التراكمي"] < 2)).sum())
    yield "Retired borrowers are 58+", int(((b["جهة_العمل"] == "متقاعد") & (b["العمر"] < 58)).sum())
    yield "Employer fits the profession", int(sum(e not in B.JOBS[j][1] + ["متقاعد"] for j, e in zip(b["المهنة"], b["جهة_العمل"])))
    yield "Loan at most ~50x monthly income", int((b["مبلغ_القرض"] > 51 * b["الدخل_الشهري"]).sum())
    yield "Removed districts (النسيم، الشفا) absent", sum(int(d["الحي"].isin(["النسيم", "الشفا"]).sum()) for d in (h, dl))


def main():
    bad = 0
    for name, problems in checks():
        bad += problems
        print(f"{'OK ' if problems == 0 else 'XX '} {problems:>4}  {name}")
    print("\nAll checks passed." if bad == 0 else f"\n{bad} problems found.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
