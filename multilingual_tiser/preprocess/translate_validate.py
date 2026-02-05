#!/usr/bin/env python3
import os
import sys
import json
import argparse
import re
from typing import Dict, Any, List, Tuple

# =============================
# ADD PROJECT ROOT TO PYTHONPATH
# =============================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.utils import (
    normalize_text,
    calculate_metrics,
)

# =============================
# Sentence-Transformers (Option A)
# =============================
from sentence_transformers import SentenceTransformer, util

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

# =============================
# Regex & heuristics
# =============================
PARENS_RE = re.compile(r"\(([^()]+)\)")

COUNTRY_TRIGGERS_IT = {"paese", "nazione", "stato"}
COUNTRY_TRIGGERS_EN = {"country", "nation", "state"}
COUNTRY_TRIGGERS_DE = {"land", "nation", "staat"}

ROLE_TRIGGERS_IT = {"posizione", "ruolo", "presidente", "governatore", "ministro", "capo"}
ROLE_TRIGGERS_EN = {"position", "role", "president", "governor", "minister", "head"}
ROLE_TRIGGERS_DE = {"position", "rolle", "präsident", "gouverneur", "minister", "chef"}

# =============================
# Helpers
# =============================
def _norm(s: str) -> str:
    return normalize_text(s or "")

def extract_entities_from_context(ctx: str) -> List[str]:
    if not ctx:
        return []
    return [_norm(m) for m in PARENS_RE.findall(ctx)]

def answer_supported_by_context(ans: str, ctx: str) -> bool:
    if not ans or not ctx:
        return False

    ans_n = _norm(ans)
    ctx_n = _norm(ctx)

    if ans_n in ctx_n:
        return True

    ents = extract_entities_from_context(ctx)
    if ans_n in ents:
        return True

    parts = re.split(r"\s*(?:,|;|\||/|\be\b|\band\b)\s*", ans_n)
    parts = [p for p in parts if p]

    if len(parts) > 1:
        return all(p in ctx_n or p in ents for p in parts)

    return False

def context_missing_answer_type(q_tgt: str, q_en: str, ctx_tgt: str, lang: str) -> bool:
    q_tgt = _norm(q_tgt)
    q_en = _norm(q_en)
    ctx = _norm(ctx_tgt)

    if lang == "it":
        country_triggers = COUNTRY_TRIGGERS_IT
        role_triggers = ROLE_TRIGGERS_IT
    elif lang == "de":
        country_triggers = COUNTRY_TRIGGERS_DE
        role_triggers = ROLE_TRIGGERS_DE
    else:
        country_triggers = set()
        role_triggers = set()

    is_country = (
        any(t in q_tgt.split() for t in country_triggers) or
        any(t in q_en.split() for t in COUNTRY_TRIGGERS_EN)
    )

    looks_admin = any(x in ctx for x in [
        "capitale", "capital", "hauptstadt",
        "district", "bezirk",
        "region", "region",
        "governorate", "provinz"
    ])

    if is_country and looks_admin:
        return True

    is_role = (
        any(t in q_tgt.split() for t in role_triggers) or
        any(t in q_en.split() for t in ROLE_TRIGGERS_EN)
    )

    return is_role


# =============================
# Semantic translation score (Option A)
# =============================
def translation_score_semantic(en_ans: str, it_ans: str) -> float:
    if not en_ans or not it_ans:
        return 0.0

    emb = _MODEL.encode([en_ans, it_ans], convert_to_tensor=True, normalize_embeddings=True)
    sim = util.cos_sim(emb[0], emb[1]).item()

    return max(0.0, min(1.0, float(sim)))

# =============================
# Core validation logic
# =============================
def validate_pair(
    en_obj: Dict[str, Any],
    tgt_obj: Dict[str, Any],
    lang: str
) -> Tuple[str, str, float]:

    en_ans = en_obj.get("answer", "")
    tgt_ans = tgt_obj.get("answer", "")

    en_q = en_obj.get("question", "")
    tgt_q = tgt_obj.get("question", "")
    tgt_ctx = tgt_obj.get("temporal_context", "")

    score = translation_score_semantic(en_ans, tgt_ans)

    # --- Hard rule: very low semantic similarity
    if score < 0.40:
        return "FAIL", "low_translation_score", score

    # --- Context support
    if answer_supported_by_context(tgt_ans, tgt_ctx):
        return "PASS", "validated", score

    if answer_supported_by_context(en_ans, tgt_ctx):
        if score >= 0.85:
            return "PASS", "validated_by_en_in_tgt_context", score
        return "AMBIGUOUS", "en_answer_supported_ctx_but_tgt_diff", score

    if context_missing_answer_type(tgt_q, en_q, tgt_ctx, lang):
        if score >= 0.85:
            return "PASS", "validated_translation_only_context_missing", score
        return "AMBIGUOUS", "context_missing_low_translation_confidence", score

    return "FAIL", "target_answer_not_supported", score


# =============================
# IO helpers
# =============================
def load_json_flexible(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]

def save_json_array(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# =============================
# Main
# =============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--en", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--lang", choices=["it", "de"], required=True)
    parser.add_argument("--category", required=True)
    
    args = parser.parse_args()

    lang = args.lang
    en_data = load_json_flexible(args.en)
    path_data = load_json_flexible(args.path)
    en_map = {x["question_id"]: x for x in en_data if "question_id" in x}

    stats = {"PASS": 0, "FAIL": 0, "AMBIGUOUS": 0}
    out = []

    score_sum = 0.0
    score_cnt = 0

    for it_item in path_data:
        qid = it_item.get("question_id")
        en_item = en_map.get(qid)

        if not en_item:
            it_item["validation_status"] = "FAIL"
            it_item["validation_reason"] = "missing_english_pair"
            it_item["translation_score"] = 0.0
            out.append(it_item)
            stats["FAIL"] += 1
            continue

        status, reason, score = validate_pair(en_item, it_item, lang)


        it_item["validation_status"] = status
        it_item["validation_reason"] = reason
        it_item["translation_score"] = float(score)
        it_item["answer_en"] = en_item.get("answer", "")

        out.append(it_item)
        stats[status] += 1
        score_sum += score
        score_cnt += 1

    save_json_array(args.path, out)


    avg = score_sum / score_cnt if score_cnt else 0.0
    print(f"Done. Processed {len(out)} items.")
    print(f"Results: PASS={stats['PASS']} | FAIL={stats['FAIL']} | AMBIGUOUS={stats['AMBIGUOUS']}")
    print(f"Average translation_score: {avg:.4f}")

if __name__ == "__main__":
    main()
