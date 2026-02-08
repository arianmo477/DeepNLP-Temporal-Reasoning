import re
import string
from collections import Counter

# ==================================================
# LANGUAGE NORMALIZATION
# ==================================================

EN_TO_FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
PERSIAN_TO_ENGLISH_TBL = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

def normalize_persian_digits(text: str) -> str:
    return text.translate(EN_TO_FA_DIGITS) if text else text


TAG_REGEX = re.compile(r"<[^>]+>")
ANSWER_REGEX = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
ANSWER_LINE_REGEX = re.compile(
    r"^\s*(?:final\s*answer|answer|antwort)\s*[:\-]\s*(.+?)\s*$",
    re.IGNORECASE,
)

UNKNOWN_TRIGGERS = [
    "unknown", "not specified", "not mentioned",
    "نامشخص", "معلوم نیست", "ذکر نشده",
    "sconosciuto", "non specificato",
    "unbekannt", "nicht angegeben",
]

TRUE_TRIGGERS = ["true", "yes", "vero", "ja", "درست"]
FALSE_TRIGGERS = ["false", "no", "falso", "nein", "نادرست"]


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = text.translate(PERSIAN_TO_ENGLISH_TBL)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def normalize_boolean(text: str) -> str:
    t = normalize_text(text)
    if any(x in t for x in TRUE_TRIGGERS):
        return "true"
    if any(x in t for x in FALSE_TRIGGERS):
        return "false"
    return ""

def normalize_boolean_translation(text: str, lang: str) -> str:
    if not text:
        return text

    t = text.strip().lower()
    t = t.strip(" .,!?:;")

    # --- ITALIAN ---
    if lang == "it":
        TRUE_FORMS = {"vero", "veri", "vera", "vere"}
        FALSE_FORMS = {"falso", "falsi", "falsa", "false"}

        if t in TRUE_FORMS:
            return "Vero"
        if t in FALSE_FORMS:
            return "Falso"

    # --- ENGLISH ---
    if lang == "en":
        if t in {"true", "yes"}:
            return "True"
        if t in {"false", "no"}:
            return "False"

    # --- PERSIAN ---
    if lang == "fa":
        if t in {"درست"}:
            return "درست"
        if t in {"نادرست"}:
            return "نادرست"

    # --- GERMAN ---
    if lang == "de":
        if t in {"wahr"}:
            return "Wahr"
        if t in {"falsch"}:
            return "Falsch"

    return text.strip()



def normalize_for_em(text: str) -> str:
    text = TAG_REGEX.sub("", str(text or "")).strip()
    low = normalize_text(text)
    if any(t in low for t in UNKNOWN_TRIGGERS):
        return "unknown"
    b = normalize_boolean(text)
    return b if b else low


def normalize_german(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\b(der|die|das|den|dem|des|ein|eine)\b", "", text)
    return text.strip()


def italian_stemmer(text: str) -> str:
    return " ".join(w[:-1] if len(w) > 3 and w[-1] in "aeiou" else w for w in text.split())


def normalize_italian(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\b(il|lo|la|i|gli|le|un|una)\b", "", text.lower())
    return italian_stemmer(text).strip()


def trim_answer_text(a: str) -> str:
    a = TAG_REGEX.sub("", str(a or "")).strip()
    lines = [ln for ln in a.splitlines() if ln.strip()]
    return lines[0].strip(" '\"") if lines else ""


def extract_answer_from_generation(full_text: str) -> str:
    if not full_text:
        return ""
    m = ANSWER_REGEX.search(full_text)
    if m:
        return trim_answer_text(m.group(1))
    for ln in reversed(full_text.splitlines()):
        m2 = ANSWER_LINE_REGEX.match(ln)
        if m2:
            return trim_answer_text(m2.group(1))
    return trim_answer_text(full_text)


def calculate_metrics(pred: str, gold: str):
    pred_b = normalize_boolean(pred)
    gold_b = normalize_boolean(gold)
    if gold_b:
        return int(pred_b == gold_b), int(pred_b == gold_b), float(pred_b == gold_b)

    p = normalize_for_em(pred)
    g = normalize_for_em(gold)

    em = int(
        p == g or
        normalize_german(p) == normalize_german(g) or
        normalize_italian(p) == normalize_italian(g)
    )

    soft = int(p in g or g in p)

    pt = p.split()
    gt = g.split()
    common = Counter(pt) & Counter(gt)
    overlap = sum(common.values())
    f1 = 0.0 if overlap == 0 else 2 * overlap / (len(pt) + len(gt))

    return em, soft, f1
