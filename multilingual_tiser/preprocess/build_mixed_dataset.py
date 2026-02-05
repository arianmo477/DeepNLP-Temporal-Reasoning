#!/usr/bin/env python3

import random
import argparse
from pathlib import Path
from utils.utils import load_json, save_json



# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--en", type=str, required=True)
    parser.add_argument("--it", type=str, required=True)
    parser.add_argument("--fa", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--samples_per_lang", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------
    en_data = load_json(args.en)
    it_data = load_json(args.it)
    fa_data = load_json(args.fa)

    print(f"Loaded EN: {len(en_data)}")
    print(f"Loaded IT: {len(it_data)}")
    print(f"Loaded FA: {len(fa_data)}")

    # --------------------------------------------------
    # Shuffle each language independently
    # --------------------------------------------------
    random.shuffle(fa_data)
    random.shuffle(it_data)
    random.shuffle(en_data)

    # --------------------------------------------------
    # Take up to samples_per_lang from EACH language
    # Priority: FA → IT → EN
    # --------------------------------------------------
    fa_selected = fa_data[:args.samples_per_lang]
    it_selected = it_data[:args.samples_per_lang]
    en_selected = en_data[:args.samples_per_lang]

    # Tag language
    for x in fa_selected:
        x["language"] = "fa"
    for x in it_selected:
        x["language"] = "it"
    for x in en_selected:
        x["language"] = "en"

    print(f"Selected FA: {len(fa_selected)}")
    print(f"Selected IT: {len(it_selected)}")
    print(f"Selected EN: {len(en_selected)}")

    # --------------------------------------------------
    # Combine & shuffle
    # --------------------------------------------------
    mixed = fa_selected + it_selected + en_selected
    random.shuffle(mixed)

    print(f"Final mixed dataset size: {len(mixed)}")

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(mixed, out_path)

    print(f"Saved mixed dataset to: {out_path}")
    print("Mixed dataset created successfully!")


if __name__ == "__main__":
    main()
