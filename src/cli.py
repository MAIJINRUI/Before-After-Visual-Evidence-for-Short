import argparse
import json
from typing import Dict

from .baseline import heuristic_recovery_action
from .evaluator import evaluate, filter_by_split, load_jsonl, load_split


def build_predictions(samples, mode: str) -> Dict[str, str]:
    predictions = {}
    for sample in samples:
        if mode == "oracle":
            action = sample["gold_recovery_action"]
        elif mode == "heuristic":
            action = heuristic_recovery_action(sample)
        else:
            action = "LookAround()"
        predictions[sample["sample_id"]] = json.dumps({"recovery_action": action})
    return predictions


def main():
    parser = argparse.ArgumentParser(description="DASE3156 Week 2 evaluator")
    parser.add_argument("--data", default="data/samples.jsonl")
    parser.add_argument("--split", default="data/split.json")
    parser.add_argument("--subset", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--mode", default="heuristic", choices=["oracle", "heuristic", "lookaround"])
    parser.add_argument("--max-steps", type=int, default=1)
    args = parser.parse_args()

    samples = load_jsonl(args.data)
    split = load_split(args.split)
    if args.subset != "all":
        samples = filter_by_split(samples, split[args.subset])

    predictions = build_predictions(samples, mode=args.mode)
    report = evaluate(samples, predictions, max_steps=args.max_steps)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
