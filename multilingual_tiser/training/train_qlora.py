#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List
from utils.utils import verify_gpu

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

# -------------------------
# CUDA / ALLOCATOR SAFETY
# -------------------------
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True




@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: Any

    def __call__(self, features):
        labels = [f["labels"] for f in features]

        features_no_labels = []
        for f in features:
            f = dict(f)
            f.pop("labels")
            features_no_labels.append(f)

        batch = self.tokenizer.pad(
            features_no_labels,
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        padded_labels = []

        for lbl in labels:
            padded_labels.append(lbl + [-100] * (max_len - len(lbl)))

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def find_lora_targets(model) -> List[str]:
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj"]
    chosen = [m for m in preferred if m in names]
    if not chosen:
        chosen = ["q_proj", "v_proj"]
    print("LoRA target_modules:", chosen)
    return chosen


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    
    # Handle optional max_train_samples safely
    # Bash script might pass empty string if variable is unset, so we handle that logic in python if needed
    # but strictly speaking argparse expects an int or nothing. 
    parser.add_argument("--max_train_samples", type=int, default=None)

    #  Training hyperparameters
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)

    parser.add_argument("--gradient_checkpointing", type=int, default=1)

    # LoRA (lighter)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # --- ADDED MISSING ARGUMENTS HERE ---
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    parser.add_argument(
        "--only_passed",
        action="store_true",
        help="Train only on samples with validation_status == PASS"
    )



    args = parser.parse_args()

    verify_gpu()

    # -------------------------
    # Tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # -------------------------
    # Dataset
    # -------------------------
    dataset = load_dataset("json", data_files=args.train_file)["train"]

    if args.only_passed:
        if "validation_status" not in dataset.column_names:
            raise ValueError(
                "Dataset does not contain 'validation_status'. "
                "Did you run translate_validate.py first?"
            )

    before = len(dataset)
    dataset = dataset.filter(lambda x: x["validation_status"] == "PASS")
    after = len(dataset)

    print(f"Filtered dataset: {before} → {after} (PASS only)")


    if args.max_train_samples:
        # Select subset
        
        limit = min(args.max_train_samples, len(dataset))
        dataset = dataset.shuffle(seed=42).select(range(limit))
        print(f"Training samples limited to: {len(dataset)}")
    else:
        print(f"Training samples: {len(dataset)}")

    # -------------------------
    # QLoRA 4-bit (FP16 compute)
    # -------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    
    # Handle int flag for gradient checkpointing
    if args.gradient_checkpointing == 1:
        model.gradient_checkpointing_enable()
        
    model.enable_input_require_grads()

    # -------------------------
    # LoRA
    # -------------------------
    targets = find_lora_targets(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -------------------------
    # Preprocessing (loss masking)
    # -------------------------
    def preprocess(ex):
        prompt = ex["prompt"].strip() + "\n"
        output = ex["output"].strip()

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        output_ids = tokenizer(output + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

        # Ensure we don't exceed max length while keeping output
        max_out = max(32, args.max_length - len(prompt_ids))
        output_ids = output_ids[:max_out]

        input_ids = prompt_ids + output_ids
        # -100 masks loss for prompt tokens
        labels = [-100] * len(prompt_ids) + output_ids
        attention_mask = [1] * len(input_ids)
        
        # Hard truncate if total length exceeds max_length
        if len(input_ids) > args.max_length:
            input_ids = input_ids[:args.max_length]
            labels = labels[:args.max_length]
            attention_mask = attention_mask[:args.max_length]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    tokenized = dataset.map(
        preprocess,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # -------------------------
    # Training args (3070 Ti)
    # -------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,

        fp16=True,
        bf16=False,

        logging_steps=args.logging_steps, # Use argument
        save_steps=args.save_steps,       # Use argument
        save_total_limit=1,
        report_to="none",

        optim="paged_adamw_8bit",
        lr_scheduler_type="linear",
        
        dataloader_num_workers=args.dataloader_num_workers, # Use argument

        remove_unused_columns=False,
        # Convert int 1/0 to boolean
        gradient_checkpointing=(args.gradient_checkpointing == 1),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForCausalLMWithPadding(tokenizer),
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete")


if __name__ == "__main__":
    main()