#!/usr/bin/env python3
import argparse
import random
import torch
import os
from tqdm import tqdm

from transformers import pipeline, AutoTokenizer
from transformers.utils import logging as hf_logging

from utils.utils import (
    load_json, save_json, repair_mangled_unicode,
    balance_by_dataset_name, normalize_persian_digits, clean_memory,
    normalize_temporal, mask_parenthesized_entities, unmask_entities,
     load_prompt_by_lang, parse_cot, rebuild_cot,normalize_boolean_translation
)

# ==================================================
# HF LOGGING
# ==================================================
hf_logging.set_verbosity_error()

# ==================================================
# CONFIG
# ==================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"

LANG_MAP = {
    "it": "ita_Latn",
    "fa": "pes_Arab",
    "de": "deu_Latn",
}

# ==================================================
# TOKENIZER
# ==================================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    src_lang=SRC_LANG
)

# ==================================================
# CHUNKING (NO TRUNCATION)
# ==================================================
def chunk_by_tokens(text: str, max_tokens: int = 900):
    if not text or not text.strip():
        return []

    ids = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False
    )["input_ids"]

    chunks = []
    for i in range(0, len(ids), max_tokens):
        sub = ids[i:i + max_tokens]
        chunks.append(
            tokenizer.decode(
                sub,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
        )
    return chunks


def translate_long_text(text, translator, max_new_tokens=1024):
    chunks = chunk_by_tokens(text)
    if not chunks:
        return ""

    outs = []
    for ch in chunks:
        with torch.inference_mode():
            out = translator(
                ch,
                truncation=False,
                max_new_tokens=max_new_tokens
            )[0]["translation_text"]
        outs.append(out.strip())

    return " ".join(outs)


# ==================================================
# BATCH TRANSLATION (SHORT TEXTS)
# ==================================================
def translate_batch(texts, translator, batch_size, desc, max_new_tokens):
    results = [""] * len(texts)
    valid = [(i, t) for i, t in enumerate(texts) if t and t.strip()]

    for i in tqdm(range(0, len(valid), batch_size), desc=desc):
        idxs, batch = zip(*valid[i:i + batch_size])
        with torch.inference_mode():
            outs = translator(
                list(batch),
                truncation=False,
                max_new_tokens=max_new_tokens
            )
        for j, out in zip(idxs, outs):
            results[j] = out["translation_text"].strip()

    return results


# ==================================================
# MAIN
# ==================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--category", choices=["train", "test"], required=True)
    parser.add_argument("--language", choices=["it", "fa", "de"], default="it")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clean_memory()

    # LOAD DATA
    data = load_json(args.input)
    if args.max_samples > 0:
        data = balance_by_dataset_name(data, args.max_samples)

    # LOAD PROMPT

    prompt = load_prompt_by_lang(args.language)

    def run(lang):
        print(f"\nProcessing language: {lang} | Mode: {args.category}")

        translator = pipeline(
            "translation",
            model=MODEL_NAME,
            device=0 if device == "cuda" else -1,
            src_lang=SRC_LANG,
            tgt_lang=LANG_MAP[lang],
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )

        
        

        final_data = []

        # PRECOLLECT
        questions, answers = [], []

        for ex in data:
            q = repair_mangled_unicode(ex.get("question", ""))
            q_m, _ = mask_parenthesized_entities(q)
            questions.append(q_m)

            a = repair_mangled_unicode(ex.get("answer", ""))
            a_m, _ = mask_parenthesized_entities(a)
            answers.append(a_m)

        tr_questions = translate_batch(
            questions, translator, args.batch_size, "Questions", 256
        )
        tr_answers = translate_batch(
            answers, translator, args.batch_size, "Answers", 128
        )

        # PER SAMPLE
        for i, ex in enumerate(tqdm(data, desc="Samples")):
            item = dict(ex)
            item["language"] = lang

            # QUESTION
            q_raw = repair_mangled_unicode(ex.get("question", ""))
            q_m, q_map = mask_parenthesized_entities(q_raw)
            q_out = unmask_entities(tr_questions[i], q_map)
            if lang == "fa":
                q_out = normalize_persian_digits(q_out)
            item["question"] = q_out

            # CONTEXT (CHUNKED, NO TRUNCATION)
            c_raw = normalize_temporal(
                repair_mangled_unicode(ex.get("temporal_context", ex.get("context", "")))
            )
            c_m, c_map = mask_parenthesized_entities(c_raw)
            c_tr = translate_long_text(c_m, translator, 1024)
            c_out = unmask_entities(c_tr, c_map)
            if lang == "fa":
                c_out = normalize_persian_digits(c_out)
            item["temporal_context"] = c_out

            # PROMPT
            item["prompt"] = prompt

            # ANSWER
            item["answer_en"] = ex.get("answer", "")
            a_m, a_map = mask_parenthesized_entities(item["answer_en"])
            a_out = unmask_entities(tr_answers[i], a_map)
            if lang == "fa":
                a_out = normalize_persian_digits(a_out)
            a_out = normalize_boolean_translation(a_out, lang)
            item["answer"] = a_out

            # OUTPUT (TRAIN ONLY → STRUCTURED CoT TRANSLATION)
            if args.category == "train" and "output" in ex:
                parts = parse_cot(ex["output"])

                parts_tr = {}
                for k, v in parts.items():
                    if not v.strip():
                        parts_tr[k] = ""
                        continue

                    # Translate each section safely
                    parts_tr[k] = translate_long_text(v, translator, 1024).strip()

                # Force answer consistency with translated answer
                parts_tr["answer"] = a_out

                item["output"] = rebuild_cot(parts_tr)
            else:
                item.pop("output", None)

 
            final_data.append(item)

        out_path = f"{args.output_dir}/TISER_{args.category}_{lang}.json"
        save_json(out_path, final_data)
        print(f"Saved {args.category.upper()} data to {out_path}")

        del translator
        clean_memory()

    if args.language in ("it"):
        run("it")
    if args.language in ("fa"):
        run("fa")
    if args.language in ("de"):
        run("de")


if __name__ == "__main__":
    main()
