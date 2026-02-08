import json
import os
import gc
import random
import torch
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



#def load_prompt_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()
    
def load_prompt_by_lang(lang: str) -> str:
    path = f"data/prompts/prompt_{lang}.txt"
    return load_txt_as_string(
        path,
        fallback=load_txt_as_string("data/prompts/prompt_en.txt")
    )



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