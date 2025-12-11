import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(model_dir):

    print(f"[info] Loading tokenizer from:{model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # if peft adaptors exists load them
    try:
        model = PeftModel.from_pretrained(base_model,model_dir)
        print("[INFO] load LoRA adapters")
    except Exception:
        print("No LoRA adaptore found")
        model = base_model
    
    model.eval()
    return model, tokenizer


def generate_answer(model,tokenizer, prompt, max_new_tokens=256):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(model.device)

    with torch.no_grad:
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the trained model directory")
    parser.add_argument("--prompt", type=str, required=False,
                        help="Single input prompt for quick inference")
    parser.add_argument("--file", type=str, required=False,
                        help="Optional: JSONL file to batch-infer")
    parser.add_argument("--max_new_tokens", type=int, default=256)

    return parser.parse_args()


def main():
    args = parse_args()

    model, tokenizer = load_model(args.model_dir)

    # Case 1 — direct prompt inference from CLI
    if args.prompt:
        print("\n===== INPUT PROMPT =====")
        print(args.prompt)

        output = generate_answer(model, tokenizer, args.prompt, args.max_new_tokens)

        print("\n===== MODEL OUTPUT =====")
        print(output)
        return

    # Case 2 — batch inference on JSONL
    if args.file:
        import json

        print(f"[INFO] Running batch inference on: {args.file}")
        out_file = args.file.replace(".jsonl", "_pred.jsonl")

        with open(args.file, "r") as f_in, open(out_file, "w") as f_out:
            for line in f_in:
                item = json.loads(line)
                prompt = item["prompt"]

                pred = generate_answer(
                    model, tokenizer, prompt, args.max_new_tokens
                )

                item["prediction"] = pred
                f_out.write(json.dumps(item) + "\n")

        print(f"[INFO] Saved predictions to {out_file}")
        return

    print("[ERROR] Please provide either --prompt or --file.")
    print("Example:")
    print("  python inference.py --model_dir models/tiser_qwen --prompt \"<your prompt here>\"")


if __name__ == "__main__":
    main()
