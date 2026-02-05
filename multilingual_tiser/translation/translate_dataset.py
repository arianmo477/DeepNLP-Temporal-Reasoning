#!/usr/bin/env python3
import json
import argparse
import random
import re
import torch
import gc
import os
import sys
from tqdm import tqdm
from transformers import pipeline
from utils.utils import (
    load_json, save_json, load_prompt_file, UNKNOWN_TRIGGERS, repair_mangled_unicode,
    balance_by_dataset_name, normalize_persian_digits, clean_memory, normalize_temporal,
    mask_parenthesized_entities, unmask_entities, load_txt_as_string, _U_ESCAPE_RE,PAREN_ENTITY_RE,make_ent_token
)

# ==================================================
# CONFIG & CONSTANTS
# ==================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
RANDOM_SEED = 42

# Default Fallback Prompt if file is missing
tp = (
    "You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection to answer questions. "
    "Follow these steps: "
    "Step 1. Reason through the problem step-by-step within <reasoning> tags. "
    "Step 2. Considering your previous reasoning, identify relevant temporal events in the given context to answer the given question within <timeline> tags. "
    "Assume that the relations in the context are one-directional. "
    "Step 3. Reflect on your reasoning and timeline to check for any errors or improvements within <reflection> tags. "
    "Step 4. Make any necessary adjustments based on your reflection. "
    "If there is further reasoning required, return to step 1 (reasoning through the problem step-by-step), otherwise move to the next section (Step 5). "
    "Step 5. Provide your final answer within <answer> tags."
)




# ==================================================
# COT PARSING
# ==================================================
def parse_cot_parts(text: str):
    parts = {"reasoning": "", "timeline": "", "reflection": "", "answer": ""}
    if not text: return parts

    r_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL | re.IGNORECASE)
    t_match = re.search(r"<timeline>(.*?)</timeline>", text, re.DOTALL | re.IGNORECASE)
    f_match = re.search(r"<reflection>(.*?)</reflection>", text, re.DOTALL | re.IGNORECASE)
    a_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    
    if r_match: 
        raw_reasoning = r_match.group(1).strip()
        # Clean nested tags
        raw_reasoning = re.sub(r"<timeline>.*?</timeline>", "", raw_reasoning, flags=re.DOTALL | re.IGNORECASE)
        raw_reasoning = re.sub(r"<reflection>.*?</reflection>", "", raw_reasoning, flags=re.DOTALL | re.IGNORECASE)
        parts["reasoning"] = raw_reasoning.strip()
    if t_match: parts["timeline"] = t_match.group(1).strip()
    if f_match: parts["reflection"] = f_match.group(1).strip()
    if a_match: parts["answer"] = a_match.group(1).strip()
    return parts

def reconstruct_cot(parts_tr: dict) -> str:
    return (
        f"<reasoning>\n{parts_tr['reasoning']}\n</reasoning>\n"
        f"<timeline>\n{parts_tr['timeline']}\n</timeline>\n"
        f"<reflection>\n{parts_tr['reflection']}\n</reflection>\n"
        f"<answer>\n{parts_tr['answer']}\n</answer>"
    )

def clean_prompt_artifacts(prompt_text: str, lang: str) -> str:
    if lang == "fa":
        clean_ending = "در غیر این صورت به بخش بعدی بروید (مرحله 5). پاسخ نهایی خود را در داخل برچسب های <answer> ارائه دهید."
        if "TAGTAG" in prompt_text or prompt_text.count("مرحله 5") > 2:
             match = re.search(r"(در غیر این صورت به بخش بعدی بروید).*$", prompt_text, re.DOTALL)
             if match:
                 prompt_text = prompt_text[:match.start()] + clean_ending
    return prompt_text

# ==================================================
# TRANSLATION SETUP
# ==================================================
def build_translator(lang, device):
    device_id = 0 if device == "cuda" else -1
    dtype = torch.float16 if device == "cuda" else torch.float32

    if lang == "it":
        tgt = "ita_Latn"
    elif lang == "fa":
        tgt = "pes_Arab"
    elif lang == "de":
        tgt = "deu_Latn"
    else:
        raise ValueError(f"Unsupported language: {lang}")

    return pipeline(
        "translation",
        model="facebook/nllb-200-distilled-600M",
        device=device_id,
        dtype=dtype,
        src_lang="eng_Latn",
        tgt_lang=tgt
    )

def translate_batch(texts, translator, batch_size, desc, max_len=512):
    valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
    valid_texts = [texts[i] for i in valid_indices]
    results_map = {i: "" for i in range(len(texts))}
    if valid_texts:
        for i in tqdm(range(0, len(valid_texts), batch_size), desc=desc):
            batch = valid_texts[i:i + batch_size]
            with torch.inference_mode():
                res = translator(batch, truncation=True, max_length=max_len)
            for j, r in enumerate(res):
                results_map[valid_indices[i + j]] = r["translation_text"].strip()
    return [results_map[i] for i in range(len(texts))]

# ==================================================
# MAIN
# ==================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--category", choices=["train", "test"], required=True)
    parser.add_argument("--language", choices=["it", "fa", "de", "both"], default="both")
    parser.add_argument("--batch_size", type=int, default=8) 
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    clean_memory()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. LOAD PROMPT
    # We load the prompt text from the file (or fallback).
    prompt_file_path = "data/prompts/prompt_en.txt"
    TISER_INSTRUCTION_EN = load_txt_as_string(prompt_file_path)
    
    data = load_json(args.input)
    if args.max_samples > 0: data = balance_by_dataset_name(data, args.max_samples)

    # Use the official Instruction text instead of the input file's prompt
    raw_prompt_template = TISER_INSTRUCTION_EN
    
    def run(lang):
        print(f"\n Processing language: {lang} | Mode: {args.category}")
        translator = build_translator(lang, device)

        # 1. Translate PROMPT (Instruction)
        prompt_masked = raw_prompt_template.replace("<reasoning>", "⟪TAG1⟫").replace("</reasoning>", "⟪TAG2⟫").replace("<timeline>", "⟪TAG3⟫").replace("</timeline>", "⟪TAG4⟫").replace("<reflection>", "⟪TAG5⟫").replace("</reflection>", "⟪TAG6⟫").replace("<answer>", "⟪TAG7⟫").replace("</answer>", "⟪TAG8⟫")
        prompt_tr_raw = translate_batch([prompt_masked], translator, 1, "Prompt Template", max_len=512)[0]
        
        tag_map = {"1": "<reasoning>", "2": "</reasoning>", "3": "<timeline>", "4": "</timeline>", "5": "<reflection>", "6": "</reflection>", "7": "<answer>", "8": "</answer>"}
        final_prompt = prompt_tr_raw
        for num, xml_tag in tag_map.items():
            pattern = re.compile(r"[\⟪«<\[\"\']*TAG\s*" + num + r"[\⟫»>\]\"\']*", re.IGNORECASE)
            final_prompt = pattern.sub(xml_tag, final_prompt)
        final_prompt = clean_prompt_artifacts(final_prompt, lang)
        
        # 2. Deconstruct Data
        questions_masked, contexts_masked, answers_masked = [], [], []
        reasoning_masked, timeline_masked, reflection_masked = [], [], []
        q_maps, c_maps, a_maps, out_maps = [], [], [], []

        for ex in data:
            q_raw = repair_mangled_unicode(ex.get("question", ""))
            c_raw = repair_mangled_unicode(ex.get("temporal_context", ex.get("context", "")))
            a_raw = repair_mangled_unicode(ex.get("answer", "")) # Get Answer
            
            c_norm = normalize_temporal(c_raw)
            q_m, q_map = mask_parenthesized_entities(q_raw)
            c_m, c_map = mask_parenthesized_entities(c_norm)
            a_m, a_map = mask_parenthesized_entities(a_raw) # Mask Answer
            
            questions_masked.append(q_m); contexts_masked.append(c_m); answers_masked.append(a_m)
            q_maps.append(q_map); c_maps.append(c_map); a_maps.append(a_map)

            # ONLY process reasoning output if this is TRAIN mode
            if args.category == "train":
                out_raw = repair_mangled_unicode(ex.get("output", ""))
                out_ent_m, out_map = mask_parenthesized_entities(out_raw)
                parts = parse_cot_parts(out_ent_m)
                reasoning_masked.append(parts["reasoning"])
                timeline_masked.append(parts["timeline"])
                reflection_masked.append(parts["reflection"])
                out_maps.append(out_map)
            else:
                # Placeholders for test
                reasoning_masked.append("")
                timeline_masked.append("")
                reflection_masked.append("")
                out_maps.append({})

        # 3. Translate Batches
        tr_q = translate_batch(questions_masked, translator, args.batch_size, "Questions", max_len=512)
        # Use high max_len for Contexts
        tr_c = translate_batch(contexts_masked, translator, args.batch_size, "Contexts", max_len=1024)
        # Translate Answers
        tr_a = translate_batch(answers_masked, translator, args.batch_size, "Answers", max_len=128)
        
        tr_reas, tr_time, tr_refl = [], [], []
        if args.category == "train":
            tr_reas = translate_batch(reasoning_masked, translator, args.batch_size, "CoT: Reasoning", max_len=1024)
            tr_time = translate_batch(timeline_masked, translator, args.batch_size, "CoT: Timeline", max_len=1024)
            tr_refl = translate_batch(reflection_masked, translator, args.batch_size, "CoT: Reflection", max_len=1024)
        
        # 4. Reconstruct & Save
        final_data = []
        for i, ex in enumerate(data):
            item = dict(ex)
            item["language"] = lang
            
            # --- INPUTS ---
            q_out = repair_mangled_unicode(unmask_entities(tr_q[i], q_maps[i]))
            c_out = repair_mangled_unicode(unmask_entities(tr_c[i], c_maps[i]))

            if lang == "fa":
                q_out = normalize_persian_digits(q_out)
                c_out = normalize_persian_digits(c_out)

            item["question"] = q_out
            item["temporal_context"] = c_out

            # Inject the clean, translated PROMPT
            item["prompt"] = final_prompt
            
            # --- ANSWER HANDLING ---
            # 1. Keep original English answer in `answer_en`
            original_answer_str = repair_mangled_unicode(ex.get("answer", ""))
            item["answer_en"] = original_answer_str
            
            # 2. Store translated answer in `answer`
            translated_answer_str = repair_mangled_unicode(unmask_entities(tr_a[i], a_maps[i]))
            if lang == "fa":
                translated_answer_str = normalize_persian_digits(translated_answer_str)

            item["answer"] = translated_answer_str

            # --- OUTPUTS ---
            if args.category == "train":
                parts_tr = {
                    "reasoning": tr_reas[i], 
                    "timeline": tr_time[i],
                    "reflection": tr_refl[i], 
                    # Use the TRANSLATED answer in the CoT block for consistency
                    "answer": translated_answer_str 
                }
                full_out_masked = reconstruct_cot(parts_tr)
                out_text = repair_mangled_unicode(unmask_entities(full_out_masked, out_maps[i]))
                if lang == "fa":
                    out_text = normalize_persian_digits(out_text)
                item["output"] = out_text

            else:
                # TEST MODE: Remove output field entirely
                if "output" in item:
                    del item["output"]

            final_data.append(item)

        out_path = f"{args.output_dir}/TISER_{args.category}_{lang}.json"
        save_json(out_path, final_data)
        print(f" Saved {args.category.upper()} data to {out_path}")
        del translator; clean_memory()

    if args.language in ("it", "both"): run("it")
    if args.language in ("fa", "both"): run("fa")
    if args.language in ("de", "both"): run("de")

if __name__ == "__main__":
    main()