#!/usr/bin/env python3



import argparse
import re
from collections import defaultdict
from utils.utils import (
    load_json,
    repair_mangled_unicode,
    save_json,
    load_txt_as_string,
    UNKNOWN_TRIGGERS,
)


# ==================================================
# REGEX
# ==================================================

TEMPORAL_REGEX = re.compile(
    r"(Temporal context:)(.*)",
    re.DOTALL | re.IGNORECASE,
)

QUESTION_REGEX = re.compile(
    r"(Question:)(.*?)(Temporal context:|$)",
    re.DOTALL | re.IGNORECASE,
)

ANSWER_BLOCK_RE = re.compile(
    r"(###\s*Answer:\s*)(.*)$",
    re.DOTALL | re.IGNORECASE,
)

REQUIRED_FIELDS = [
    "dataset_name",
    "question_id",
    "question",
    "prompt",
    "answer",
]

# ==================================================
# UNKNOWN DETECTION (ONLY FILTER)
# ==================================================



def is_unknown_answer(answer: str) -> bool:
    if not answer:
        return True
    return answer.strip().lower() in UNKNOWN_TRIGGERS

def remove_unknown_answers(data):
    clean, removed = [], []
    for sample in data:
        if is_unknown_answer(sample.get("answer", "")):
            removed.append(sample)
        else:
            clean.append(sample)
    return clean, removed



# ==================================================
# EXTRACTION HELPERS
# ==================================================
def extract_question_and_context(prompt: str):
    """
    Extract Question and Temporal context from the ORIGINAL prompt.
    """
    question = ""
    temporal_context = ""

    if not prompt:
        return question, temporal_context

    m_q = QUESTION_REGEX.search(prompt)
    if m_q:
        question = m_q.group(2).strip()

    m_tc = TEMPORAL_REGEX.search(prompt)
    if m_tc:
        temporal_context = m_tc.group(2).strip()

    return question, temporal_context

# ==================================================
# NORMALIZATION
# ==================================================
def normalize_train(sample, idx, canonical_prompt):
    raw_prompt = sample.get("prompt", "")

    q_from_prompt, temporal_context = extract_question_and_context(raw_prompt)

    return {
        "dataset_name": sample.get("dataset_name"),
        "question_id": sample.get("question_id", f"auto_{idx}"),
        "language": sample.get("language", "en"),
        "question": sample.get("question", q_from_prompt).strip(),
        "temporal_context": temporal_context,
        "prompt": canonical_prompt,
        "output": sample.get("output", "").strip(),
        "answer": sample.get("answer", "").strip(),
    }

def normalize_test(sample, idx, canonical_prompt):
    raw_prompt = sample.get("prompt", "")

    q_from_prompt, temporal_context = extract_question_and_context(raw_prompt)

    return {
        "dataset_name": sample.get("dataset_name"),
        "question_id": sample.get("question_id", f"auto_{idx}"),
        "language": sample.get("language", "en"),
        "question": sample.get("question", q_from_prompt).strip(),
        "temporal_context": temporal_context,
        "prompt": canonical_prompt,
        "answer": sample.get("answer", "").strip(),
    }

# ==================================================
# DEDUPLICATION (SAFE)
# ==================================================
def remove_duplicates(data):
    seen = set()
    deduped = []

    for sample in data:
        qid = sample.get("question_id")
        if qid and qid in seen:
            continue
        if qid:
            seen.add(qid)
        deduped.append(sample)

    return deduped, len(data) - len(deduped)


# ==================================================
# Void Quesion ID Removal
# ==================================================

def remove_void_question_ids(data):
    deduped = []
    for sample in data:
        qid = sample.get("question_id")
        if qid and qid.strip() != "":
            deduped.append(sample)

    return deduped, len(data) - len(deduped)


# ==================================================
# VALIDATION (REPORT ONLY)
# ==================================================
def validate_sample(sample, split):
    errors = []

    required = REQUIRED_FIELDS if split == "train" else [
        "dataset_name", "question_id", "question", "prompt", "answer"
    ]

    for field in required:
        if field not in sample:
            errors.append(f"missing:{field}")
        elif not isinstance(sample[field], str):
            errors.append(f"non_string:{field}")
        elif sample[field].strip() == "":
            errors.append(f"empty:{field}")

    if split == "train" and is_unknown_answer(sample.get("answer", "")):
        errors.append("unknown_answer")

    return errors


# ==================================================
# MAIN
# ==================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--save_invalid", default=None)
    parser.add_argument("--max_print", type=int, default=5)
    args = parser.parse_args()

    canonical_prompt = load_txt_as_string(args.prompt_file)

    # 1. Load
    raw_data = load_json(args.input)
    print(f"Raw samples: {len(raw_data)}")

    #1.5 FIX UNICODE ISSUES ONCE, GLOBALLY
    for s in raw_data:
        for k in ["question", "temporal_context", "output", "answer", "answer_en", "prompt"]:
            if k in s and isinstance(s[k], str):
                s[k] = repair_mangled_unicode(s[k])

    # 2. Normalize
    if args.split == "train":
        data = [
            normalize_train(s, i, canonical_prompt)
            for i, s in enumerate(raw_data)
        ]
    else:
        data = [
            normalize_test(s, i, canonical_prompt)
            for i, s in enumerate(raw_data)
        ]

    # 3. Deduplicate
    data, n_dups = remove_duplicates(data)
    print(f"Removed {n_dups} duplicates (Remaining: {len(data)})")

    # 4. Remove ONLY unknown answers
    data, removed_unknowns = remove_unknown_answers(data)
    print(f"Removed {len(removed_unknowns)} samples with unknown answers")

    # 5. REMOVE VOID QUESTION IDS
    data, n_void_qids = remove_void_question_ids(data)
    print(f"Removed {n_void_qids} samples with void question IDs (Remaining: {len(data)})")

    # 6. Save cleaned dataset
    save_json(args.output, data)
    print(f"Clean dataset saved to: {args.output}")

    # . Prompt consistency check
    prompts = {s["prompt"] for s in data}
    print(
        "Prompt templates:",
        "CONSISTENT" if len(prompts) == 1 else f"INCONSISTENT ({len(prompts)})"
    )

    # 8. Validation report (no removal)
    invalid_samples = []
    error_stats = defaultdict(int)

    for idx, sample in enumerate(data):
        errs = validate_sample(sample, args.split)
        if errs:
            invalid_samples.append({
                "index": idx,
                "errors": errs,
                "sample": sample,
            })
            for e in errs:
                error_stats[e] += 1

    print(f"Samples with structural warnings: {len(invalid_samples)}")

    if error_stats:
        print("\nStructural warning breakdown:")
        for k, v in sorted(error_stats.items(), key=lambda x: -x[1]):
            print(f"  - {k}: {v}")

    if invalid_samples and args.save_invalid:
        save_json(args.save_invalid, invalid_samples)
        print(f"Structural warnings saved to: {args.save_invalid}")

    print("\nDone. Prompt is now fully canonical and file-based.")

if __name__ == "__main__":
    main()
