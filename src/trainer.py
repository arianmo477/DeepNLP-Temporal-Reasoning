import argparse
import os
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)

from peft import LoraConfig, get_peft_model

from dataset import TISERDataset

def load_model_and_tokenizer(model_name, lora_r, lora_alpha, lora_dropout):
    """
    Loads a HuggingFace causal language model and wraps it with LoRA adapters.
    """
    print(f"[INFO] Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # ensure proper padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    

    tokenizer.padding_side = "right"

    # QLoRA setup
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Recommended for QLoRA
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # laod model in bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
        quantization_config=bnb_config, 
        trust_remote_code=True
    )

    # LoRA configuration
    print("[INFO] Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "down_proj", "up_proj"
        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    # model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()

    # Data/Input
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)

    # Output directory
    parser.add_argument("--output_dir", type=str, default="models/tiser_model")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=2048)

    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    return parser.parse_args()


def main():
    args = parse_args()

    use_cuda = torch.cuda.is_available()
    print(f"[INFO] Using CUDA: {use_cuda}, using fp16: {use_cuda}, bf16: False")

    # Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    # Load dataset
    print(f"[INFO] Loading dataset from {args.train_file}")
    train_dataset = TISERDataset(
        path=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    # Causal LM → no MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Configure training
    print("[INFO] Configuring Trainer...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=20,
        save_steps=500,
        save_total_limit=3,
        bf16=False,
        fp16=torch.cuda.is_available(), 
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        report_to="tensorboard",
        logging_dir=f"{args.output_dir}/logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("[INFO] Starting training...")
    trainer.train()

    print(f"[INFO] Saving final model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()