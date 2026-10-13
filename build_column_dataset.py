"""
مَأْمَن — بناء أعمدة مصنّفة (multi-label) للتدريب والـ RAG والتقييم

المصادر (20 داتا سِت):
  - Adult: من data/adult_preprocessed.csv نفسه (إحصائيات حقيقية)
  - داتا سِتات عامة معروفة (Titanic, Bank Marketing, Telco, ...) — أوصاف أعمدتها مكتوبة يدوياً
  - داتا سعودية اصطناعية بأعمدة عربية — تتولّد هنا وتنحسب إحصائياتها فعلياً
  - أسماء معرّبة ومختصرة (jawal, hawiya, dx_code ...) لاختبار المتانة

التقسيم حسب الداتا سِت (مو حسب العمود) عشان ما يصير تسريب:
  train / valid → للتدريب وقاعدة الـ RAG ،  test → للتقييم فقط ولا يدخل أي منهما

التصنيفات تطلع review_status=PROPOSED. راجعوها وغيّروها لـ APPROVED قبل التدريب.

التشغيل:  python scripts/build_column_dataset.py
المخرجات: data/column_annotations_multi.csv  (للمراجعة والتعديل)
          data/column_features_multi.jsonl   (وصف كل عمود + عينة قيم)
          data/synthetic/*.csv               (الداتا السعودية الاصطناعية)
"""

import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maman_common import ALL_KEYS, ANNOTATIONS, DATA, FEATURES, describe_column  # noqa: E402

# التصنيف الأساسي لكل عمود، وبعدها تنضاف الأدوار الإضافية (انظر to_labels)
D, Q, S, O = "direct", "qi", "sensitive", "none"
rng = np.random.default_rng(42)
N = 2000


def C(name, dtype, unique_ratio, samples, label, reason):
    return {"name": name, "dtype": dtype, "unique_ratio": unique_ratio,
            "samples": [str(s) for s in samples], "label": label, "reason": reason}


# ═════════════════════════════════════════════
# 1) داتا سِتات عامة (أوصاف يدوية بقيم حقيقية الشكل)
# ═════════════════════════════════════════════
PUBLIC = {
    "titanic": [
        C("PassengerId", "numeric", 1.0, [1, 2, 3, 4, 5], O, "رقم تسلسلي داخلي للصف لا يرتبط بشخص خارج الملف"),
        C("Survived", "numeric", 0.002, [0, 1, 1, 0], S, "النجاة أو الوفاة معلومة حياتية حساسة"),
        C("Pclass", "numeric", 0.003, [3, 1, 3, 2], Q, "درجة السفر صفة يمكن معرفتها من مصادر أخرى"),
        C("Name", "text", 1.0, ["Braund, Mr. Owen Harris", "Heikkinen, Miss. Laina", "Allen, Mr. William Henry"], D, "الاسم الكامل يحدد الشخص مباشرة"),
        C("Sex", "text", 0.002, ["male", "female"], Q, "الجنس معرّف غير مباشر"),
        C("Age", "numeric", 0.099, [22.0, 38.0, 26.0, 35.0], Q, "العمر معرّف غير مباشر"),
        C("SibSp", "numeric", 0.008, [1, 0, 3, 1], Q, "عدد الإخوة والأزواج المرافقين يكشف تركيبة العائلة"),
        C("Parch", "numeric", 0.008, [0, 2, 1, 0], Q, "عدد الوالدين والأبناء المرافقين يكشف تركيبة العائلة"),
        C("Ticket", "text", 0.764, ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803"], Q, "رقم التذكرة قد يكون مشتركاً بين أفراد العائلة فيضيّق الدائرة دون أن يحدد الشخص وحده"),
        C("Fare", "numeric", 0.278, [7.25, 71.2833, 7.925, 53.1], O, "سعر التذكرة لا يكشف الشخص ولا يضره"),
        C("Cabin", "text", 0.165, ["C85", "C123", "E46", "B28"], Q, "رقم المقصورة يضيّق دائرة الركاب كثيراً"),
        C("Embarked", "text", 0.003, ["S", "C", "Q"], Q, "ميناء الصعود يمكن معرفته من سجلات أخرى"),
    ],
    "heart_disease": [
        C("age", "numeric", 0.135, [63, 37, 41, 56], Q, "العمر معرّف غير مباشر"),
        C("sex", "numeric", 0.007, [1, 0, 1, 1], Q, "الجنس مرمّز رقمياً وهو معرّف غير مباشر"),
        C("cp", "numeric", 0.013, [3, 2, 1, 0], S, "نوع ألم الصدر معلومة صحية"),
        C("trestbps", "numeric", 0.162, [145, 130, 130, 120], S, "ضغط الدم أثناء الراحة قياس صحي"),
        C("chol", "numeric", 0.502, [233, 250, 204, 236], S, "مستوى الكوليسترول نتيجة تحليل طبي"),
        C("fbs", "numeric", 0.007, [1, 0, 0, 0], S, "سكر الصائم مؤشر صحي"),
        C("thalach", "numeric", 0.30, [150, 187, 172, 178], S, "أعلى نبض قلب قياس طبي"),
        C("exang", "numeric", 0.007, [0, 0, 0, 1], S, "الذبحة مع الجهد معلومة صحية"),
        C("oldpeak", "numeric", 0.13, [2.3, 3.5, 1.4, 0.8], S, "قراءة تخطيط القلب معلومة طبية"),
        C("target", "numeric", 0.007, [1, 1, 0, 1], S, "وجود مرض القلب تشخيص صحي"),
    ],
    "bank_marketing": [
        C("age", "numeric", 0.002, [58, 44, 33, 47], Q, "العمر معرّف غير مباشر"),
        C("job", "text", 0.0003, ["management", "technician", "entrepreneur", "blue-collar"], Q, "المهنة معرّف غير مباشر"),
        C("marital", "text", 0.0001, ["married", "single", "divorced"], Q, "الحالة الاجتماعية معرّف غير مباشر"),
        C("education", "text", 0.0001, ["tertiary", "secondary", "unknown", "primary"], Q, "المستوى التعليمي معرّف غير مباشر"),
        C("default", "text", 0.0001, ["no", "no", "yes"], S, "التعثر الائتماني معلومة مالية حساسة"),
        C("balance", "numeric", 0.16, [2143, 29, 2, 1506], S, "رصيد الحساب البنكي معلومة مالية خاصة"),
        C("housing", "text", 0.0001, ["yes", "no"], S, "وجود قرض سكني معلومة مالية"),
        C("loan", "text", 0.0001, ["no", "yes"], S, "وجود قرض شخصي معلومة مالية"),
        C("contact", "text", 0.0001, ["unknown", "cellular", "telephone"], O, "قناة التواصل في الحملة بيانات تشغيلية"),
        C("day", "numeric", 0.0007, [5, 5, 5, 6], O, "يوم آخر اتصال بالحملة بيانات تشغيلية"),
        C("month", "text", 0.0003, ["may", "jun", "jul"], O, "شهر الاتصال بيانات تشغيلية"),
        C("duration", "numeric", 0.035, [261, 151, 76, 92], O, "مدة المكالمة بالثواني بيانات تشغيلية"),
        C("campaign", "numeric", 0.001, [1, 1, 2, 3], O, "عدد الاتصالات في الحملة عدّاد"),
        C("pdays", "numeric", 0.012, [-1, -1, 151, 98], O, "الأيام منذ آخر حملة عدّاد"),
        C("poutcome", "text", 0.0001, ["unknown", "failure", "success"], O, "نتيجة الحملة السابقة بيانات تشغيلية"),
        C("y", "text", 0.0001, ["no", "yes"], O, "استجابة العميل للحملة هدف تسويقي"),
    ],
    "medical_insurance": [
        C("age", "numeric", 0.035, [19, 18, 28, 33], Q, "العمر معرّف غير مباشر"),
        C("sex", "text", 0.0015, ["female", "male"], Q, "الجنس معرّف غير مباشر"),
        C("bmi", "numeric", 0.41, [27.9, 33.77, 33.0, 22.705], S, "مؤشر كتلة الجسم معلومة صحية"),
        C("children", "numeric", 0.0045, [0, 1, 3, 0], Q, "عدد الأبناء يكشف تركيبة الأسرة"),
        C("smoker", "text", 0.0015, ["yes", "no"], S, "التدخين معلومة صحية حساسة"),
        C("region", "text", 0.003, ["southwest", "southeast", "northwest"], Q, "المنطقة الجغرافية معرّف غير مباشر"),
        C("charges", "numeric", 1.0, [16884.924, 1725.5523, 4449.462], S, "تكاليف العلاج معلومة صحية ومالية"),
    ],
    "hr_attrition": [
        C("EmployeeNumber", "numeric", 1.0, [1, 2, 4, 5, 7], D, "الرقم الوظيفي يحدد الموظف داخل الشركة"),
        C("Age", "numeric", 0.03, [41, 49, 37, 33], Q, "العمر معرّف غير مباشر"),
        C("Gender", "text", 0.001, ["Female", "Male"], Q, "الجنس معرّف غير مباشر"),
        C("Department", "text", 0.002, ["Sales", "Research & Development", "Human Resources"], Q, "القسم معروف لزملاء الموظف"),
        C("JobRole", "text", 0.006, ["Sales Executive", "Research Scientist", "Laboratory Technician"], Q, "المسمى الوظيفي معروف عن الشخص"),
        C("MaritalStatus", "text", 0.002, ["Single", "Married", "Divorced"], Q, "الحالة الاجتماعية معرّف غير مباشر"),
        C("MonthlyIncome", "numeric", 0.91, [5993, 5130, 2090, 2909], S, "الراتب الشهري معلومة مالية خاصة"),
        C("PerformanceRating", "numeric", 0.001, [3, 4, 3, 3], S, "تقييم الأداء معلومة يضر كشفها بالموظف"),
        C("YearsAtCompany", "numeric", 0.025, [6, 10, 0, 8], Q, "سنوات الخدمة يعرفها الزملاء وتضيّق الدائرة"),
        C("EmployeeCount", "numeric", 0.0007, [1, 1, 1, 1], O, "قيمة ثابتة لا تحمل معلومة"),
        C("StandardHours", "numeric", 0.0007, [80, 80, 80], O, "قيمة ثابتة لا تحمل معلومة"),
        C("Attrition", "text", 0.001, ["Yes", "No", "No"], O, "متغير الهدف للنموذج ولا يميّز الشخص"),
    ],
    "customer_personality": [
        C("ID", "numeric", 1.0, [5524, 2174, 4141, 6182], O, "رقم العميل يحدده في أنظمة الشركة"),
        C("Year_Birth", "numeric", 0.026, [1957, 1954, 1965, 1984], Q, "سنة الميلاد معرّف غير مباشر"),
        C("Education", "text", 0.002, ["Graduation", "PhD", "Master", "Basic"], Q, "المستوى التعليمي معرّف غير مباشر"),
        C("Marital_Status", "text", 0.004, ["Single", "Together", "Married", "Divorced"], Q, "الحالة الاجتماعية معرّف غير مباشر"),
        C("Income", "numeric", 0.88, [58138.0, 46344.0, 71613.0, 26646.0], S, "الدخل معلومة مالية خاصة"),
        C("Kidhome", "numeric", 0.0013, [0, 1, 0, 1], Q, "عدد الأطفال في البيت يكشف تركيبة الأسرة"),
        C("Teenhome", "numeric", 0.0013, [0, 1, 0, 0], Q, "عدد المراهقين في البيت يكشف تركيبة الأسرة"),
        C("Dt_Customer", "text", 0.30, ["04-09-2012", "08-03-2014", "21-08-2013"], Q, "تاريخ التسجيل حدث مرتبط بالشخص"),
        C("Recency", "numeric", 0.045, [58, 38, 26, 94], O, "أيام منذ آخر شراء مقياس تسويقي"),
        C("MntWines", "numeric", 0.35, [635, 11, 426, 173], S, "مشتريات الخمور تكشف استهلاك الكحول وهي معلومة حساسة"),
        C("NumWebPurchases", "numeric", 0.007, [8, 1, 8, 2], O, "عدد المشتريات عدّاد"),
        C("Complain", "numeric", 0.001, [0, 0, 0, 1], O, "مؤشر شكوى تشغيلي"),
        C("Z_CostContact", "numeric", 0.0004, [3, 3, 3], O, "قيمة ثابتة لا تحمل معلومة"),
    ],

    # ── للاختبار ──
    "telco_churn": [
        C("customerID", "text", 1.0, ["7590-VHVEG", "5575-GNVDE", "3668-QPYBK"], D, "رقم حساب العميل يحدده في أنظمة الشركة"),
        C("gender", "text", 0.0003, ["Female", "Male"], Q, "الجنس معرّف غير مباشر"),
        C("SeniorCitizen", "numeric", 0.0003, [0, 0, 1, 0], Q, "مؤشر كبار السن يكشف الفئة العمرية"),
        C("Partner", "text", 0.0003, ["Yes", "No"], Q, "وجود شريك يكشف الحالة الاجتماعية"),
        C("Dependents", "text", 0.0003, ["No", "Yes"], Q, "وجود معالين يكشف تركيبة الأسرة"),
        C("tenure", "numeric", 0.01, [1, 34, 2, 45], O, "مدة الاشتراك بالأشهر مقياس تشغيلي"),
        C("PhoneService", "text", 0.0003, ["No", "Yes"], O, "نوع الخدمة بيانات منتج"),
        C("InternetService", "text", 0.0004, ["DSL", "Fiber optic", "No"], O, "نوع الخدمة بيانات منتج"),
        C("Contract", "text", 0.0004, ["Month-to-month", "One year", "Two year"], O, "نوع العقد بيانات منتج"),
        C("PaymentMethod", "text", 0.0006, ["Electronic check", "Mailed check", "Credit card (automatic)"], O, "طريقة الدفع بيانات تشغيلية"),
        C("MonthlyCharges", "numeric", 0.22, [29.85, 56.95, 53.85, 42.3], S, "الفاتورة الشهرية تكشف مصروفات الشخص المالية"),
        C("TotalCharges", "text", 0.93, ["29.85", "1889.5", "108.15"], S, "إجمالي الفواتير يكشف مصروفات الشخص المالية"),
        C("Churn", "text", 0.0003, ["No", "Yes"], O, "متغير الهدف لإلغاء الاشتراك"),
    ],
    "german_credit": [
        C("checking_status", "text", 0.004, ["<0", "0<=X<200", "no checking"], S, "حالة الحساب الجاري معلومة مالية"),
        C("duration", "numeric", 0.033, [6, 48, 12, 42], O, "مدة القرض بالأشهر شرط عقد"),
        C("credit_history", "text", 0.005, ["critical/other existing credit", "existing paid", "delayed previously"], S, "السجل الائتماني معلومة مالية حساسة"),
        C("credit_amount", "numeric", 0.92, [1169, 5951, 2096, 7882], S, "مبلغ القرض معلومة مالية خاصة"),
        C("age", "numeric", 0.053, [67, 22, 49, 45], Q, "العمر معرّف غير مباشر"),
        C("personal_status", "text", 0.004, ["male single", "female div/dep/mar", "male mar/wid"], Q, "الجنس والحالة الاجتماعية معاً معرّف غير مباشر"),
        C("job", "text", 0.004, ["skilled", "unskilled resident", "high qualif/self emp/mgmt"], Q, "فئة الوظيفة معرّف غير مباشر"),
        C("housing", "text", 0.003, ["own", "for free", "rent"], Q, "نوع السكن يمكن معرفته عن الشخص"),
        C("foreign_worker", "text", 0.002, ["yes", "no"], Q, "كونه عاملاً أجنبياً يضيّق دائرة المتشابهين"),
        C("num_dependents", "numeric", 0.002, [1, 2, 1], Q, "عدد المعالين يكشف تركيبة الأسرة"),
        C("class", "text", 0.002, ["good", "bad"], S, "تصنيف المخاطر الائتمانية للشخص معلومة حساسة"),
    ],
    "student_performance": [
        C("school", "text", 0.003, ["GP", "MS"], Q, "المدرسة معروفة عن الطالب وتضيّق الدائرة"),
        C("sex", "text", 0.003, ["F", "M"], Q, "الجنس معرّف غير مباشر"),
        C("age", "numeric", 0.02, [18, 17, 15, 16], Q, "العمر معرّف غير مباشر"),
        C("address", "text", 0.003, ["U", "R"], Q, "نوع منطقة السكن معرّف غير مباشر"),
        C("famsize", "text", 0.003, ["GT3", "LE3"], Q, "حجم الأسرة معرّف غير مباشر"),
        C("Medu", "numeric", 0.012, [4, 1, 1, 4], Q, "تعليم الأم صفة عائلية تساعد في التمييز"),
        C("Fjob", "text", 0.012, ["teacher", "other", "services"], Q, "مهنة الأب صفة عائلية تساعد في التمييز"),
        C("Dalc", "numeric", 0.012, [1, 1, 2, 1], S, "استهلاك الكحول معلومة حساسة"),
        C("health", "numeric", 0.012, [3, 3, 3, 5], S, "الحالة الصحية معلومة حساسة"),
        C("absences", "numeric", 0.08, [6, 4, 10, 2], S, "غياب الطالب سجل تعليمي يضر كشفه"),
        C("G3", "numeric", 0.045, [6, 6, 10, 15], S, "الدرجة النهائية سجل تعليمي خاص"),
        C("studytime", "numeric", 0.01, [2, 2, 2, 3], O, "وقت المذاكرة الأسبوعي مقياس عام"),
    ],
    "pima_diabetes": [
        C("Pregnancies", "numeric", 0.022, [6, 1, 8, 1], S, "عدد مرات الحمل معلومة صحية"),
        C("Glucose", "numeric", 0.18, [148, 85, 183, 89], S, "مستوى الجلوكوز نتيجة تحليل طبي"),
        C("BloodPressure", "numeric", 0.06, [72, 66, 64, 66], S, "ضغط الدم قياس صحي"),
        C("SkinThickness", "numeric", 0.067, [35, 29, 0, 23], S, "سماكة الجلد قياس طبي"),
        C("Insulin", "numeric", 0.24, [0, 0, 0, 94], S, "مستوى الإنسولين نتيجة تحليل طبي"),
        C("BMI", "numeric", 0.32, [33.6, 26.6, 23.3, 28.1], S, "مؤشر كتلة الجسم معلومة صحية"),
        C("DiabetesPedigreeFunction", "numeric", 0.67, [0.627, 0.351, 0.672, 0.167], S, "مؤشر الاستعداد الوراثي للسكري معلومة صحية"),
        C("Age", "numeric", 0.068, [50, 31, 32, 21], Q, "العمر معرّف غير مباشر"),
        C("Outcome", "numeric", 0.003, [1, 0, 1, 0], S, "الإصابة بالسكري تشخيص صحي"),
    ],
    "web_logs": [
        C("user_email", "text", 0.35, ["sara.k@example.com", "a.alharbi@example.net", "mk2291@example.org"], D, "البريد الإلكتروني يحدد الشخص مباشرة"),
        C("ip_address", "text", 0.41, ["203.0.113.24", "198.51.100.7", "203.0.113.91"], D, "عنوان IP يمكن ربطه بمشترك محدد"),
        C("device_id", "text", 0.38, ["a3f9c2e1-77b4-4d1e-9a0e-5c1b2d3e4f60", "0b7e1d22-3c4a-4f5b-8e9d-1a2b3c4d5e6f"], D, "معرّف الجهاز يتتبّع شخصاً محدداً"),
        C("session_id", "text", 0.92, ["s_8f2a91", "s_8f2a92", "s_77c013"], O, "معرّف جلسة مؤقت داخلي"),
        C("user_agent", "text", 0.08, ["Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X)", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"], Q, "نوع الجهاز والمتصفح يساعد في بصمة المستخدم"),
        C("event_time", "text", 0.97, ["2026-03-14T09:12:44Z", "2026-03-14T09:13:02Z"], Q, "توقيت نشاط الشخص يمكن ربطه بأحداث معروفة"),
        C("page_url", "text", 0.05, ["/home", "/checkout", "/product/1123"], O, "الصفحة المزارة بيانات تصفح عامة"),
        C("country", "text", 0.001, ["SA", "AE", "EG"], Q, "الدولة معرّف غير مباشر"),
        C("latency_ms", "numeric", 0.21, [120, 87, 341, 95], O, "زمن الاستجابة قياس تقني"),
        C("row_hash", "text", 1.0, ["9f86d081884c7d65", "60303ae22b998861", "fd61a03af4f77d87"], O, "بصمة تقنية للصف وليست صفة لشخص رغم أنها فريدة"),
    ],
    "health_claims": [
        C("patient_ssn", "text", 0.97, ["512-48-3391", "604-12-8873", "433-90-1250"], D, "رقم الضمان الاجتماعي يحدد الشخص مباشرة"),
        C("pt_name", "text", 0.95, ["Maria Lopez", "James O'Neil", "Wei Chen"], D, "اسم المريض يحدده مباشرة"),
        C("dob", "text", 0.62, ["1984-07-02", "1951-11-23", "1990-01-15"], Q, "تاريخ الميلاد معرّف غير مباشر قوي"),
        C("zip5", "text", 0.08, ["02139", "94110", "60614"], Q, "الرمز البريدي يضيّق الموقع الجغرافي"),
        C("gender", "text", 0.001, ["F", "M"], Q, "الجنس معرّف غير مباشر"),
        C("dx_code", "text", 0.06, ["E11.9", "I10", "J45.909"], S, "رمز التشخيص ICD-10 معلومة صحية"),
        C("cpt_code", "text", 0.04, ["99213", "80053", "93000"], S, "رمز الإجراء الطبي معلومة صحية"),
        C("claim_amount", "numeric", 0.7, [184.5, 72.0, 1290.75], S, "مبلغ المطالبة معلومة صحية ومالية"),
        C("claim_id", "text", 1.0, ["CLM0000193", "CLM0000194", "CLM0000207"], O, "معرّف المطالبة داخلي ولا يحدد الشخص وحده"),
        C("admit_date", "text", 0.35, ["2025-02-11", "2025-02-14", "2025-03-01"], Q, "تاريخ الدخول للمستشفى حدث يمكن ربطه بالشخص"),
    ],
}


# ═════════════════════════════════════════════
# 2) داتا سعودية اصطناعية (تتولّد فعلياً)
# ═════════════════════════════════════════════
# كل الأشخاص سعوديون (قرار الفريق)، والأسماء والجنس والإيميل متطابقة، والوظيفة/الكلية/التواريخ منطقية
MALE = {"محمد": "mohammed", "عبدالله": "abdullah", "فهد": "fahad", "خالد": "khalid", "سعود": "saud",
        "عبدالرحمن": "abdulrahman", "فيصل": "faisal", "تركي": "turki", "ناصر": "nasser", "سلمان": "salman",
        "يوسف": "yousef", "عمر": "omar"}
FEMALE = {"نورة": "noura", "سارة": "sarah", "ريم": "reem", "هيا": "haya", "لمى": "lama", "منيرة": "munira",
          "العنود": "alanoud", "جواهر": "jawaher", "بشرى": "bushra", "أمل": "amal", "دانة": "dana", "شهد": "shahad"}
FAMILY = {"العتيبي": "alotaibi", "القحطاني": "alqahtani", "الشمري": "alshammari", "الدوسري": "aldosari",
          "الغامدي": "alghamdi", "الزهراني": "alzahrani", "الحربي": "alharbi", "المطيري": "almutairi",
          "السبيعي": "alsubaie", "العنزي": "alanazi", "الشهري": "alshehri", "البقمي": "albuqami"}
DISTRICTS = ["النرجس", "الملقا", "العليا", "الياسمين", "حطين", "الروضة", "السويدي", "العارض", "قرطبة", "الربوة"]
CITIES = ["الرياض", "جدة", "الدمام", "مكة المكرمة", "المدينة المنورة", "أبها", "تبوك", "بريدة", "الخبر", "حائل"]
REGIONS = ["منطقة الرياض", "منطقة مكة المكرمة", "المنطقة الشرقية", "منطقة عسير", "منطقة القصيم"]
CLINIC_OF = {"سكري النوع الثاني": "الغدد الصماء", "قصور الغدة الدرقية": "الغدد الصماء", "ارتفاع ضغط الدم": "القلب",
             "ربو": "الصدرية", "التهاب المفاصل": "الروماتيزم", "اكتئاب": "النفسية", "فقر الدم": "الباطنة"}
DIAGNOSES = list(CLINIC_OF)
# وظائف القروض: الدخل الشهري (بالآلاف) وجهات العمل الممكنة
JOBS = {"مهندس": ((14, 30), ["أرامكو", "سابك", "قطاع خاص"]), "فني": ((7, 13), ["أرامكو", "سابك", "قطاع خاص"]),
        "طبيب": ((25, 45), ["وزارة الصحة", "قطاع خاص"]), "ممرض": ((9, 16), ["وزارة الصحة", "قطاع خاص"]),
        "معلم": ((9, 17), ["وزارة التعليم", "قطاع خاص"]), "محاسب": ((9, 18), ["قطاع خاص", "أرامكو", "سابك"]),
        "مبرمج": ((12, 26), ["قطاع خاص", "أرامكو", "سابك"]), "مدير مشروع": ((18, 35), ["قطاع خاص", "أرامكو", "سابك"]),
        "موظف مبيعات": ((6, 12), ["قطاع خاص"]), "أخصائي موارد بشرية": ((9, 16), ["قطاع خاص", "أرامكو", "سابك"])}
# وظائف الموارد البشرية: الإدارة ونطاق الراتب الأساسي (بالآلاف)
HR_JOBS = {"مهندس برمجيات": ("التقنية", (14, 28)), "مهندس شبكات": ("التقنية", (12, 24)),
           "محاسب": ("المالية", (9, 18)), "محلل مالي": ("المالية", (12, 24)),
           "أخصائي موارد بشرية": ("الموارد البشرية", (9, 16)), "أخصائي توظيف": ("الموارد البشرية", (8, 14)),
           "مندوب مبيعات": ("المبيعات", (7, 13)), "مدير مبيعات": ("المبيعات", (20, 38)),
           "مدير مشروع": ("العمليات", (18, 35)), "منسق عمليات": ("العمليات", (8, 14))}
# التخصص ← الكلية
MAJORS = {"علوم الحاسب": "الحاسب", "نظم المعلومات": "الحاسب", "هندسة كهربائية": "الهندسة", "هندسة مدنية": "الهندسة",
          "محاسبة": "إدارة الأعمال", "تسويق": "إدارة الأعمال", "كيمياء": "العلوم", "فيزياء": "العلوم",
          "طب بشري": "الطب"}
BANKS = ["الراجحي", "الأهلي", "الرياض", "البلاد", "الإنماء"]


def pick(items, n=N, p=None):
    return rng.choice(items, size=n, p=p)


def digits(n_digits, n=N, prefix=""):
    return [prefix + "".join(rng.choice(list("0123456789"), n_digits)) for _ in range(n)]


def people(n=N):
    """أشخاص متّسقون: الجنس يطابق الاسم الأول، والاسم الثاني اسم الأب (ذكر)."""
    female = rng.random(n) < 0.5
    first = np.where(female, pick(list(FEMALE), n), pick(list(MALE), n))
    father = pick(list(MALE), n)
    same = father == first
    while same.any():  # الاسم الأول غير اسم الأب
        father[same] = pick(list(MALE), int(same.sum()))
        same = father == first
    family = pick(list(FAMILY), n)
    return {
        "female": female, "first": first, "family": family,
        "name": [f"{a} {b} {c}" for a, b, c in zip(first, father, family)],
        "sex": np.where(female, "أنثى", "ذكر"),
        "first_lat": [FEMALE.get(f) or MALE[f] for f in first], "family_lat": [FAMILY[f] for f in family],
    }


def emails(p, domain):
    """الإيميل من الاسم نفسه، ولو تكرر نضيف رقم."""
    seen, out = {}, []
    for f, l in zip(p["first_lat"], p["family_lat"]):
        base = f"{f}.{l}"
        seen[base] = seen.get(base, 0) + 1
        out.append(f"{base}{seen[base] if seen[base] > 1 else ''}@{domain}")
    return out


def rand_dates(start, end, n=N):
    """تاريخ عشوائي بين تاريخين (مصفوفات أو قيم)."""
    start, end = np.broadcast_to(np.asarray(start, "datetime64[D]"), n), np.broadcast_to(np.asarray(end, "datetime64[D]"), n)
    span = (end - start).astype(int)
    return start + (rng.random(n) * (span + 1)).astype(int).astype("timedelta64[D]")


def fmt(d):
    return np.datetime_as_string(d, unit="D")


def national_ids(n=N):
    return ["1" + "".join(rng.choice(list("0123456789"), 9)) for _ in range(n)]  # هوية وطنية تبدأ بـ 1 (سعوديون)


def build_synthetic():
    """كل داتا سِت: {اسم_العمود: (القيم, التصنيف, السبب)}"""
    saudi = np.full(N, "سعودي")

    # ── مستشفى
    hp = people()
    diagnoses = pick(DIAGNOSES)
    diabetic = diagnoses == "سكري النوع الثاني"
    hba1c = np.where(diabetic, rng.normal(8.4, 1.1, N).clip(6.5, 12.5), rng.normal(5.3, 0.35, N).clip(4.3, 6.4)).round(1)

    # ── موارد بشرية: التعيين بعد سن 22 (ومن 2005 وطالع)
    hr = people()
    birth = rand_dates("1965-01-01", "2002-12-31")
    earliest = np.maximum(birth + np.timedelta64(22 * 365, "D"), np.datetime64("2005-01-01"))
    hire = rand_dates(earliest, "2025-12-31")
    hr_job = pick(list(HR_JOBS))
    years = (np.datetime64("2026-01-01") - hire).astype(int) / 365
    lo, hi = np.array([HR_JOBS[j][1] for j in hr_job]).T
    salary = (np.minimum(lo + (hi - lo) * (0.25 + 0.75 * rng.random(N)) + 0.3 * years, hi) * 2).round() / 2 * 1000

    # ── سفر
    tr = people()

    # ── قروض: الوظيفة ← الدخل وجهة العمل؛ المتقاعد عمره 58+؛ القرض حسب الدخل؛ السداد حسب سمة
    bk = people()
    age_b = rng.integers(21, 70, N)
    job_b = pick(list(JOBS))
    retired = (age_b >= 58) & (rng.random(N) < 0.6)
    employer = np.array([pick(JOBS[j][1], 1)[0] for j in job_b])
    employer = np.where(retired, "متقاعد", employer)
    lo, hi = np.array([JOBS[j][0] for j in job_b]).T
    income = (lo + (hi - lo) * rng.random(N)) * np.where(retired, 0.55, 1.0)
    income = (income * 2).round() / 2 * 1000
    loan = np.maximum(20, (income / 1000 * rng.uniform(5, 50, N)).round()) * 1000
    marital = np.where(age_b < 25, pick(["أعزب", "متزوج"], p=[0.8, 0.2]),
                       np.where(age_b < 55, pick(["متزوج", "أعزب", "مطلق"], p=[0.75, 0.15, 0.10]),
                                pick(["متزوج", "مطلق", "أرمل"], p=[0.75, 0.10, 0.15])))
    simah = rng.normal(690, 85, N).clip(300, 900).round().astype(int)
    u = rng.random(N)
    repay = np.where(simah < 560, np.where(u < 0.45, "متعثر", np.where(u < 0.8, "متأخر", "منتظم")),
                     np.where(simah < 680, np.where(u < 0.08, "متعثر", np.where(u < 0.3, "متأخر", "منتظم")),
                              np.where(u < 0.08, "متأخر", "منتظم")))

    # ── جامعة: التخصص ← الكلية؛ الميلاد قبل القبول بـ 18 سنة تقريباً؛ الإنذار إذا المعدل < 2
    un = people()
    major = pick(list(MAJORS))
    admit_g = rng.integers(2020, 2025, N)
    birth_u = rand_dates([f"{y - 19}-09-01" for y in admit_g], [f"{y - 18}-08-31" for y in admit_g])
    gpa = rng.normal(3.6, 0.7, N).clip(1.0, 5.0).round(2)
    status = np.where(gpa < 2.0, pick(["إنذار أول", "إنذار ثاني"]), "منتظم")

    # ── معرّب
    ar = people()

    return {
        "sa_hospital": {
            "رقم_السجل": (np.arange(1, N + 1), O, "رقم تسلسلي للصف داخل الملف"),
            "رقم_الملف_الطبي": ([f"MRN-{x}" for x in rng.choice(range(100000, 999999), N, replace=False)], D, "رقم الملف الطبي فريد لكل مريض"),
            "اسم_المريض": (hp["name"], D, "اسم المريض يحدده مباشرة"),
            "رقم_الهوية": (national_ids(), D, "رقم الهوية الوطنية يحدد الشخص مباشرة"),
            "جوال_المريض": (digits(8, prefix="05"), D, "رقم الجوال يحدد الشخص مباشرة"),
            "العمر": (rng.integers(18, 85, N), Q, "العمر معرّف غير مباشر"),
            "الجنس": (hp["sex"], Q, "الجنس معرّف غير مباشر"),
            "الجنسية": (saudi, Q, "الجنسية معرّف غير مباشر"),
            "الحي": (pick(DISTRICTS), Q, "الحي السكني يضيّق دائرة الأشخاص"),
            "تاريخ_الزيارة": (fmt(rand_dates("2025-01-01", "2025-12-31")), Q, "تاريخ الزيارة حدث يمكن ربطه بالشخص"),
            "التشخيص": (diagnoses, S, "التشخيص معلومة صحية حساسة"),
            "السكر_التراكمي": (hba1c, S, "نتيجة تحليل طبي"),
            "العيادة": (np.array([CLINIC_OF[d] for d in diagnoses]), S, "نوع العيادة يكشف طبيعة الحالة الصحية"),
            "مدة_الانتظار_بالدقائق": (rng.integers(5, 180, N), O, "مقياس تشغيلي للمستشفى"),
        },
        "sa_hr": {
            "الرقم_الوظيفي": ([f"E{x}" for x in rng.choice(range(10000, 99999), N, replace=False)], D, "الرقم الوظيفي يحدد الموظف داخل الجهة"),
            "اسم_الموظف": (hr["name"], D, "اسم الموظف يحدده مباشرة"),
            "البريد_الإلكتروني": (emails(hr, "company.sa"), D, "البريد الإلكتروني يحدد الشخص مباشرة"),
            "الآيبان": (digits(22, prefix="SA"), D, "رقم الآيبان مرتبط بحساب شخص محدد"),
            "تاريخ_الميلاد": (fmt(birth), Q, "تاريخ الميلاد معرّف غير مباشر قوي"),
            "الجنس": (hr["sex"], Q, "الجنس معرّف غير مباشر"),
            "الجنسية": (saudi, Q, "الجنسية معرّف غير مباشر"),
            "المسمى_الوظيفي": (hr_job, Q, "المسمى الوظيفي معروف عن الشخص"),
            "الإدارة": (np.array([HR_JOBS[j][0] for j in hr_job]), Q, "الإدارة معروفة لزملاء الموظف"),
            "تاريخ_التعيين": (fmt(hire), Q, "تاريخ التعيين حدث يعرفه الزملاء"),
            "الراتب_الأساسي": (salary.astype(int), S, "الراتب معلومة مالية خاصة"),
            "بدل_السكن": (((salary * 0.25) / 100).round().astype(int) * 100, S, "البدلات جزء من الدخل الخاص"),
            "تقييم_الأداء": (pick(["ممتاز", "جيد جداً", "جيد", "يحتاج تحسين"], p=[0.2, 0.4, 0.3, 0.1]), S, "تقييم الأداء يضر كشفه بالموظف"),
            "رصيد_الإجازات": (rng.integers(0, 45, N), O, "رصيد الإجازات عدّاد إداري"),
        },
        "sa_delivery": {
            "رقم_الطلب": ([f"ORD-{x}" for x in rng.choice(range(1000000, 9999999), N, replace=False)], O, "رقم الطلب داخلي ولا يحدد الشخص وحده"),
            "اسم_المستلم": (people()["name"], D, "اسم المستلم يحدده مباشرة"),
            "جوال_المستلم": (digits(8, prefix="+9665"), D, "رقم الجوال يحدد الشخص مباشرة"),
            "العنوان_الوطني": ([f"{''.join(rng.choice(list('ABCDEFGHJKLMNPRSTUVWXYZ'), 4))}{rng.integers(1000, 9999)}" for _ in range(N)], D, "العنوان الوطني المختصر يحدد مسكن الشخص"),
            "الحي": (pick(DISTRICTS), Q, "الحي السكني يضيّق دائرة الأشخاص"),
            "المدينة": (np.full(N, "الرياض"), Q, "المدينة معرّف غير مباشر"),  # الأحياء كلها في الرياض
            "وقت_الطلب": ([f"{d} {rng.integers(8, 24):02d}:{rng.integers(0, 60):02d}" for d in fmt(rand_dates("2026-01-01", "2026-07-19"))], Q, "توقيت نشاط الشخص يمكن ربطه بأحداث معروفة"),
            "قيمة_الطلب": (np.round(rng.gamma(2, 60, N) + 15, 2), S, "قيمة الطلب الشخصي تكشف مصروفات الشخص المالية"),
            "طريقة_الدفع": (pick(["مدى", "Apple Pay", "بطاقة ائتمانية", "عند الاستلام"]), O, "طريقة الدفع بيانات تشغيلية"),
            "تقييم_المندوب": (rng.integers(1, 6, N), O, "تقييم الخدمة بيانات تشغيلية"),
        },
        "sa_travel": {
            "رقم_الجواز": ([f"{rng.choice(list('ABKLNP'))}{''.join(rng.choice(list('0123456789'), 8))}" for _ in range(N)], D, "رقم الجواز يحدد الشخص مباشرة"),
            "اسم_المسافر": (tr["name"], D, "اسم المسافر يحدده مباشرة"),
            "الجنسية": (saudi, Q, "الجنسية معرّف غير مباشر"),
            "تاريخ_الميلاد": (fmt(rand_dates("1950-01-01", "2005-12-31")), Q, "تاريخ الميلاد معرّف غير مباشر قوي"),
            "رقم_الرحلة": (pick(["SV1020", "SV1021", "XY214", "F3112", "SV303"]), Q, "رقم الرحلة مع التاريخ يحدد من كان على متنها"),
            "تاريخ_الوصول": (fmt(rand_dates("2026-01-01", "2026-04-30")), Q, "تاريخ الوصول حدث يمكن ربطه بالشخص"),
            "عدد_المرافقين": (rng.integers(0, 6, N), Q, "عدد المرافقين يكشف تركيبة المجموعة"),
            "رقم_المقعد": ([f"{rng.integers(1, 45)}{rng.choice(list('ABCDEF'))}" for _ in range(N)], O, "رقم المقعد لا يكشف الشخص وحده"),
            "فئة_الغرفة": (pick(["مفردة", "مزدوجة", "جناح"], p=[0.45, 0.45, 0.10]), O, "نوع الحجز بيانات منتج"),
            "سعر_الليلة": (rng.integers(250, 1800, N), O, "سعر الحجز بيانات تشغيلية"),
            "احتياجات_صحية": (pick(["لا يوجد", "كرسي متحرك", "حساسية فول سوداني", "سكري - وجبة خاصة"], p=[0.85, 0.04, 0.04, 0.07]), S, "الاحتياجات الصحية معلومة حساسة"),
        },

        # ── للاختبار ──
        "sa_bank_loans": {
            "رقم_العميل": ([f"C{x}" for x in rng.choice(range(1000000, 9999999), N, replace=False)], D, "رقم العميل يحدده في أنظمة البنك"),
            "اسم_العميل": (bk["name"], D, "اسم العميل يحدده مباشرة"),
            "رقم_الجوال": (digits(8, prefix="05"), D, "رقم الجوال يحدد الشخص مباشرة"),
            "رقم_الهوية_الوطنية": (national_ids(), D, "رقم الهوية يحدد الشخص مباشرة"),
            "العمر": (age_b, Q, "العمر معرّف غير مباشر"),
            "المدينة": (pick(CITIES), Q, "المدينة معرّف غير مباشر"),
            "المهنة": (job_b, Q, "المهنة معرّف غير مباشر"),
            "الحالة_الاجتماعية": (marital, Q, "الحالة الاجتماعية معرّف غير مباشر"),
            "جهة_العمل": (employer, Q, "جهة العمل معروفة عن الشخص"),
            "تاريخ_الطلب": (fmt(rand_dates("2025-06-01", "2026-03-27")), Q, "تاريخ الطلب حدث مرتبط بالشخص"),
            "مبلغ_القرض": (loan.astype(int), S, "مبلغ القرض معلومة مالية خاصة"),
            "الدخل_الشهري": (income.astype(int), S, "الدخل معلومة مالية خاصة"),
            "درجة_سمة": (simah, S, "الدرجة الائتمانية معلومة مالية حساسة"),
            "حالة_السداد": (repay, S, "التعثر في السداد معلومة حساسة"),
            "البنك": (pick(BANKS), O, "اسم البنك بيانات تشغيلية"),
            "رمز_المنتج": (pick(["PL-01", "PL-02", "MTG-10", "AUTO-3"]), O, "رمز المنتج بيانات تشغيلية"),
        },
        "sa_university": {
            "الرقم_الجامعي": ([f"4{y - 1979}{x}" for y, x in zip(admit_g, rng.choice(range(10000, 99999), N, replace=False))], D, "الرقم الجامعي يحدد الطالب"),
            "اسم_الطالب": (un["name"], D, "اسم الطالب يحدده مباشرة"),
            "الجنس": (un["sex"], Q, "الجنس معرّف غير مباشر"),
            "تاريخ_الميلاد": (fmt(birth_u), Q, "تاريخ الميلاد معرّف غير مباشر قوي"),
            "الكلية": (np.array([MAJORS[m] for m in major]), Q, "الكلية معروفة عن الطالب"),
            "التخصص": (major, Q, "التخصص معروف عن الطالب"),
            "سنة_القبول": (admit_g - 2020 + 1442, Q, "سنة القبول تضيّق دائرة الدفعة"),
            "المنطقة": (pick(REGIONS), Q, "منطقة السكن معرّف غير مباشر"),
            "المعدل_التراكمي": (gpa, S, "المعدل سجل تعليمي خاص"),
            "الحالة_الأكاديمية": (status, S, "الإنذار الأكاديمي يضر كشفه بالطالب"),
            "الساعات_المسجلة": (rng.integers(12, 20, N), O, "عدد الساعات عدّاد أكاديمي"),
        },
        "sa_arabizi": {
            "id": (np.arange(1, N + 1), O, "رقم تسلسلي للصف"),
            "esm": (ar["name"], D, "الاسم يحدد الشخص مباشرة"),
            "jawal": (digits(8, prefix="05"), D, "رقم الجوال يحدد الشخص مباشرة"),
            "hawiya": (national_ids(), D, "رقم الهوية يحدد الشخص مباشرة"),
            "omr": (rng.integers(18, 80, N), Q, "العمر معرّف غير مباشر"),
            "jins": (np.where(ar["female"], "f", "m"), Q, "الجنس معرّف غير مباشر"),
            "madina": (pick(["Riyadh", "Jeddah", "Dammam", "Abha", "Tabuk"]), Q, "المدينة معرّف غير مباشر"),
            "raatib": (rng.integers(4, 50, N) * 1000, S, "الراتب معلومة مالية خاصة"),
            "maradh_mozmen": (pick(["la", "sukkari", "daght", "rabw"], p=[0.7, 0.12, 0.12, 0.06]), S, "المرض المزمن معلومة صحية"),
            "created_at": ([f"{d}T{rng.integers(0, 24):02d}:{rng.integers(0, 60):02d}:00" for d in fmt(rand_dates("2026-05-01", "2026-05-30"))], Q, "وقت تسجيل الشخص في النظام يمكن ربطه بنشاط معروف عنه"),
        },
    }


# ═════════════════════════════════════════════
# 3) التقسيم حسب الداتا سِت
# ═════════════════════════════════════════════
SPLITS = {
    "train": ["adult", "titanic", "heart_disease", "bank_marketing", "medical_insurance",
              "hr_attrition", "sa_hospital", "sa_hr", "sa_delivery"],
    "valid": ["customer_personality", "sa_travel"],
    "test": ["telco_churn", "german_credit", "student_performance", "pima_diabetes", "web_logs",
             "health_claims", "sa_bank_loans", "sa_university", "sa_arabizi"],
}

# عدد صفوف الداتا سِتات العامة (لحساب distinct_count من unique_ratio)
ROWS = {"titanic": 891, "heart_disease": 303, "bank_marketing": 45211, "medical_insurance": 1338,
        "hr_attrition": 1470, "customer_personality": 2240, "telco_churn": 7043, "german_credit": 1000,
        "student_performance": 395, "pima_diabetes": 768, "web_logs": 50000, "health_claims": 20000}

# Adult (بأسماء أعمدة data/adult_preprocessed.csv)
ADULT = {
    "age": (Q, "العمر مع أعمدة أخرى يميّز الشخص"),
    "workclass": (Q, "نوع جهة العمل صفة معروفة عن الشخص"),
    "fnlwgt": (O, "وزن إحصائي للمسح وليس صفة للشخص رغم أنه شبه فريد"),
    "education": (Q, "المستوى التعليمي يساعد في التمييز"),
    "education_num": (Q, "ترميز رقمي للمستوى التعليمي"),
    "marital_status": (Q, "الحالة الاجتماعية صفة معروفة عن الشخص"),
    "occupation": (Q, "المهنة صفة معروفة عن الشخص"),
    "relationship": (Q, "دور الشخص في الأسرة يضيّق دائرة المتشابهين"),
    "race": (Q, "العرق صفة ظاهرة تساعد في التمييز"),
    "sex": (Q, "الجنس من المعرّفات غير المباشرة الكلاسيكية"),
    "capital_gain": (S, "أرباح رأس المال معلومة مالية خاصة"),
    "capital_loss": (S, "خسائر رأس المال معلومة مالية خاصة"),
    "hours_per_week": (O, "ساعات العمل لا تُعرف عادة عن الشخص من الخارج"),
    "native_country": (Q, "بلد الأصل يقلّل عدد المتشابهين"),
    "income": (S, "الدخل معلومة مالية يضر كشفها"),
}

# سياسة المشروع (من classify_columns.py): العرق والجنس والحالة الاجتماعية والعلاقة الأسرية = QI + حساسة
POLICY_SENSITIVE_QI = {
    "race", "sex", "Sex", "gender", "Gender", "marital_status", "marital", "MaritalStatus", "Marital_Status",
    "relationship", "Partner", "personal_status", "الجنس", "jins", "الحالة_الاجتماعية",
    # قرار الفريق: تركيبة الأسرة = QI + حساس
    "SibSp", "Parch", "children", "Kidhome", "Teenhome", "Dependents", "num_dependents", "famsize",
}
# مفاتيح سجلات (record_identifier). اللي تحدد الشخص في أنظمة الجهة: معرّف مباشر أيضاً
RECORD_ONLY = {"PassengerId", "رقم_السجل", "id", "claim_id", "رقم_الطلب", "session_id", "row_hash"}
RECORD_AND_DIRECT = {"ID", "EmployeeNumber", "customerID", "رقم_العميل", "الرقم_الوظيفي", "الرقم_الجامعي", "رقم_الملف_الطبي"}
# أسماء مختصرة ما يُفهم معناها من الاسم والقيم بدون دليل بيانات
NEEDS_REVIEW = {"fnlwgt", "cp", "trestbps", "fbs", "thalach", "exang", "oldpeak", "Medu", "Dalc", "pdays"}


def to_labels(column, base, reason):
    labels = {k: False for k in ALL_KEYS}
    labels["direct_identifier"] = base == D
    labels["quasi_identifier"] = base == Q
    labels["sensitive_attribute"] = base == S
    if column in POLICY_SENSITIVE_QI:
        labels["quasi_identifier"] = labels["sensitive_attribute"] = True
        reason += "، ويُعامل كحساس حسب سياسة المشروع"
    if column in RECORD_ONLY:
        labels["record_identifier"] = True
    if column in RECORD_AND_DIRECT:
        labels["record_identifier"] = labels["direct_identifier"] = True
    labels["needs_review"] = column in NEEDS_REVIEW
    return labels, reason


def main():
    data_only = "--data-only" in sys.argv  # يعيد توليد الداتا ووصف الأعمدة فقط، ويحافظ على التصنيفات المعتمدة
    if ANNOTATIONS.exists() and "--force" not in sys.argv and not data_only:
        with ANNOTATIONS.open(encoding="utf-8-sig", newline="") as fh:
            if any(r["review_status"].strip().upper() == "APPROVED" for r in csv.DictReader(fh)):
                raise SystemExit(f"{ANNOTATIONS.name} has APPROVED rows; rerunning would overwrite the review. "
                                 "Pass --force to regenerate anyway.")
    split_of = {ds: split for split, names in SPLITS.items() for ds in names}
    rows = []  # (features, base, reason)

    adult = pd.read_csv(DATA / "adult_preprocessed.csv")
    assert set(adult.columns) == set(ADULT), set(adult.columns) ^ set(ADULT)
    for col, (base, reason) in ADULT.items():
        rows.append((describe_column("adult", col, adult[col]), base, reason))

    for ds, cols in PUBLIC.items():
        for c in cols:
            n = ROWS[ds]
            features = {"dataset": ds, "column": c["name"], "dtype": c["dtype"], "rows": n, "missing_count": 0,
                        "distinct_count": max(1, round(c["unique_ratio"] * n)),
                        "unique_value_ratio": c["unique_ratio"], "samples": c["samples"]}
            rows.append((features, c["label"], c["reason"]))

    (DATA / "synthetic").mkdir(exist_ok=True)
    for ds, spec in build_synthetic().items():
        df = pd.DataFrame({col: vals for col, (vals, _, _) in spec.items()})
        df.to_csv(DATA / "synthetic" / f"{ds}.csv", index=False, encoding="utf-8-sig")
        for col, (_, base, reason) in spec.items():
            rows.append((describe_column(ds, col, df[col]), base, reason))

    datasets = {f["dataset"] for f, _, _ in rows}
    assert datasets == set(split_of), datasets ^ set(split_of)
    assert len({(f["dataset"], f["column"]) for f, _, _ in rows}) == len(rows), "duplicate column"

    with FEATURES.open("w", encoding="utf-8") as fh:
        for f, _, _ in rows:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")

    if data_only:
        with ANNOTATIONS.open(encoding="utf-8-sig", newline="") as fh:
            annotated = {(r["dataset"], r["column"]) for r in csv.DictReader(fh)}
        built = {(f["dataset"], f["column"]) for f, _, _ in rows}
        assert built == annotated, f"columns changed: {built ^ annotated}"
        print(f"Regenerated data and features for {len(rows)} columns; annotations untouched.")
        return

    header = ["split", "dataset", "column", *ALL_KEYS, "reason", "review_status", "reviewer_notes"]
    table = []
    for f, base, reason in rows:
        labels, reason = to_labels(f["column"], base, reason)
        table.append({"split": split_of[f["dataset"]], "dataset": f["dataset"], "column": f["column"],
                      **{k: int(v) for k, v in labels.items()}, "reason": reason,
                      "review_status": "PROPOSED", "reviewer_notes": ""})
    with ANNOTATIONS.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(table)

    t = pd.DataFrame(table)
    t["none"] = (t[["direct_identifier", "quasi_identifier", "sensitive_attribute", "record_identifier"]].sum(axis=1) == 0).astype(int)
    print(f"{len(t)} columns from {len(datasets)} datasets")
    print(t.groupby("split")[[*ALL_KEYS, "none"]].sum().assign(columns=t.groupby("split").size()).to_string())


if __name__ == "__main__":
    main()
