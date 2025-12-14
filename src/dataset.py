import json

from torch.utils.data import Dataset


class TISERDataset(Dataset):
    """
    Loads a JSONL file where each line contains:
    {
        "prompt": "...",
        "output": "..."
    }

    Builds a single LM-style training string using:
    <prompt>\n<answer>
    """

    def __init__(self, path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load JSONL
        with open(path, "r") as f:
            self.data = [json.loads(line) for line in f]

        print(f"[INFO] Loaded {len(self.data)} samples from {path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        answer = item["output"]

        full_text = f"{prompt}\n{answer}{self.tokenizer.eos_token}"
        prompt_text = f"{prompt}\n"

        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        labels = input_ids.clone()

        # Mask padding
        labels[attention_mask == 0] = -100

        # Compute prompt length
        prompt_enc = self.tokenizer(
            prompt_text,
            add_special_tokens=False
        )
        prompt_len = len(prompt_enc["input_ids"])

        # Mask prompt tokens safely
        seq_len = attention_mask.sum().item()
        mask_len = min(prompt_len, seq_len)
        labels[:mask_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }