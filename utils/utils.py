# utils/utils.py


import os
import json
import gc
import re
import string
from collections import Counter
import torch
import random
from collections import defaultdict

# ==================================================
# GPU / MEMORY
# ==================================================

def verify_gpu():
    print("===== GPU CHECK =====")
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Training will fail unless on CPU mode.")
    else:
        print("GPU:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print("VRAM (GB):", round(props.total_memory / (1024**3), 2))
    print("=====================")


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================================================
# IO
# ==================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def _strip_invalid_unicode(obj):
    """
    Recursively removes invalid unicode surrogates from strings.
    """
    if isinstance(obj, str):
        return obj.encode("utf-8", "ignore").decode("utf-8")
    elif isinstance(obj, list):
        return [_strip_invalid_unicode(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _strip_invalid_unicode(v) for k, v in obj.items()}
    else:
        return obj


def save_json(path, data):
    clean_data = _strip_invalid_unicode(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=2, ensure_ascii=False)



def load_prompt_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_txt_as_string(path: str, fallback: str = "") -> str:
    if not os.path.exists(path):
        print(f"Warning: Prompt file not found at {path}. Using fallback.")
        return fallback
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return content if content else fallback
    except Exception as e:
        print(f" Error reading prompt file: {e}. Using fallback.")
        return fallback


# ==================================================
# UNICODE / TEMPORAL / ENTITY
# ==================================================

def repair_mangled_unicode(text: str) -> str:
    if not text or not isinstance(text, str):
        return text
    try:
        return text.encode("latin1").decode("utf8")
    except Exception:
        return text


_U_ESCAPE_RE = re.compile(r"u([0-9a-fA-F]{4})")

def repair_mangled_unicode(s: str) -> str:
    if not s:
        return ""
    if _U_ESCAPE_RE.search(s) is None:
        return s
    try:
        s2 = _U_ESCAPE_RE.sub(r"\\u\1", s)
        return s2.encode("utf-8").decode("unicode_escape")
    except Exception:
        return s


_RANGE_RE = re.compile(r"\b(\d{4})\s*-\s*(\d{4})\b")
_STARTS_RE = re.compile(r"\bstarts at\s+(\d{4})\b", re.IGNORECASE)
_ENDS_RE = re.compile(r"\bends at\s+(\d{4})\b", re.IGNORECASE)

def normalize_temporal(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\s+", " ", text).strip()
    text = _RANGE_RE.sub(r"From \1 to \2", text)
    text = _STARTS_RE.sub(r"started in \1", text)
    text = _ENDS_RE.sub(r"ended in \1", text)
    return text


PAREN_ENTITY_RE = re.compile(r"\(([^()]+)\)")
def make_ent_token(i: int) -> str:
    return f"⟪ENT{i:06d}⟫"



def mask_parenthesized_entities(text: str):
    if not text:
        return text, {}
    mapping = {}
    seen = {}
    out = text
    idx = 0
    for m in PAREN_ENTITY_RE.finditer(text):
        full = "(" + m.group(1) + ")"
        if full not in seen:
            tok = make_ent_token(idx)
            seen[full] = tok
            mapping[tok] = full
            idx += 1
    for full, tok in sorted(seen.items(), key=lambda x: len(x[0]), reverse=True):
        out = out.replace(full, tok)
    return out, mapping


def unmask_entities(text: str, mapping: dict) -> str:
    """
    Robustly restores masked parenthesized entities.
    Handles cases where MT slightly corrupts the placeholder
    (spaces, quotes, brackets, etc.).
    """
    if not text or not mapping:
        return text

    for tok, original in mapping.items():
        # Extract numeric ID from token ⟪ENT000001⟫
        m = re.search(r"ENT(\d{6})", tok)
        if not m:
            continue
        ent_id = m.group(1)

        # Robust pattern: allow spaces, quotes, brackets around ENT + id
        pattern = re.compile(
            r"[\s\"'«»⟪⟫\[\]\(\)]*ENT\s*"
            + ent_id +
            r"[\s\"'«»⟪⟫\[\]\(\)]*",
            re.IGNORECASE
        )

        text = pattern.sub(original, text)

    return text

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



def balance_by_dataset_name(data, max_samples, seed=42):
    random.seed(seed)

    buckets = defaultdict(list)
    for x in data:
        buckets[x["dataset_name"]].append(x)

    names = list(buckets.keys())
    n = len(names)
    base = max_samples // n

    selected = []
    leftovers = []

    for name in names:
        if len(buckets[name]) <= base:
            selected.extend(buckets[name])
            leftovers += buckets[name][:]
        else:
            sel = random.sample(buckets[name], base)
            selected.extend(sel)
            leftovers += [x for x in buckets[name] if x not in sel]

    remaining = max_samples - len(selected)
    if remaining > 0 and leftovers:
        selected.extend(random.sample(leftovers, min(remaining, len(leftovers))))

    random.shuffle(selected)
    return selected

