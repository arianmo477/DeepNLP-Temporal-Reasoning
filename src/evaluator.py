import argparse
import json
import re
from collections import Counter


def normalize_text(text: str):
    """Lowercase, remove punctuation, and strip spaces."""
    import string

    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def extract_answer(text: str):
    """
    Extracts <answer>...</answer> from the model output.
    If not found, return the entire text.
    """

    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # fallback → whole text
    return text.strip()


def compute_f1(prediction: str, ground_truth: str):
    """
    Standard EM/F1 evaluation used in QA benchmarks.
    """

    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def compute_em(prediction: str, ground_truth: str):
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_file", type=str, required=True,
                        help="JSONL file that contains predictions (test_pred.jsonl)")
    parser.add_argument("--output", type=str, default="eval_results.txt",
                        help="File to save evaluation metrics")

    return parser.parse_args()


def main():
    args = parse_args()

    total = 0
    em_sum = 0.0
    f1_sum = 0.0

    print(f"[INFO] Evaluating predictions: {args.pred_file}")

    with open(args.pred_file, "r") as f:
        for line in f:
            item = json.loads(line)

            gold = item.get("answer", "").strip()
            pred_raw = item.get("prediction", "")

            pred = extract_answer(pred_raw)

            em = compute_em(pred, gold)
            f1 = compute_f1(pred, gold)

            em_sum += em
            f1_sum += f1
            total += 1

    em_final = em_sum / total if total > 0 else 0.0
    f1_final = f1_sum / total if total > 0 else 0.0

    # print to console
    print("\n===== Evaluation Results =====")
    print(f"Total samples: {total}")
    print(f"Exact Match: {em_final:.4f}")
    print(f"F1 Score:     {f1_final:.4f}")

    # save to file
    with open(args.output, "w") as out:
        out.write(f"Total: {total}\n")
        out.write(f"EM: {em_final:.4f}\n")
        out.write(f"F1: {f1_final:.4f}\n")

    print(f"[INFO] Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
