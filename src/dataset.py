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

        # Build training text
        text = f"{prompt}\n{answer}"

        
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": enc["input_ids"].squeeze(0),
        }
