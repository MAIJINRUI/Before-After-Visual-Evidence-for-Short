#!/usr/bin/env python3
"""
Split samples for partner collaboration. Generates two HTML galleries so you and
your partner can each review half the samples independently, then merge results.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SAMPLES_PATH = DATA_DIR / "samples.jsonl"

# Import render_gallery from generate_starter_dataset
sys.path.insert(0, str(ROOT / "scripts"))
from generate_starter_dataset import render_gallery


def main():
    parser = argparse.ArgumentParser(description="Split samples for partner A/B review")
    parser.add_argument(
        "--split",
        choices=["half", "alternate", "custom"],
        default="half",
        help="half: first/second half; alternate: odd/even indices; custom: use --range-a",
    )
    parser.add_argument(
        "--range-a",
        default="",
        help="Custom range for Partner A, e.g. '0:250' (first 250). Partner B gets the rest.",
    )
    args = parser.parse_args()

    if not SAMPLES_PATH.exists():
        print(f"Error: {SAMPLES_PATH} not found. Run generate_starter_dataset.py first.")
        sys.exit(1)

    samples = [json.loads(line) for line in open(SAMPLES_PATH, encoding="utf-8")]
    n = len(samples)

    if args.split == "half":
        mid = n // 2
        samples_a, samples_b = samples[:mid], samples[mid:]
    elif args.split == "alternate":
        samples_a = [s for i, s in enumerate(samples) if i % 2 == 0]
        samples_b = [s for i, s in enumerate(samples) if i % 2 == 1]
    else:
        if args.range_a:
            parts = args.range_a.replace(":", "-").split("-")
            start, end = int(parts[0]), int(parts[1])
            samples_a = samples[start:end]
            ids_a = {s["sample_id"] for s in samples_a}
            samples_b = [s for s in samples if s["sample_id"] not in ids_a]
        else:
            mid = n // 2
            samples_a, samples_b = samples[:mid], samples[mid:]

    out_a = DATA_DIR / "review_gallery_partner_a.html"
    out_b = DATA_DIR / "review_gallery_partner_b.html"

    out_a.write_text(render_gallery(samples_a, partner_suffix="_A"), encoding="utf-8")
    out_b.write_text(render_gallery(samples_b, partner_suffix="_B"), encoding="utf-8")

    print(f"Partner A: {len(samples_a)} samples -> {out_a.name}")
    print(f"Partner B: {len(samples_b)} samples -> {out_b.name}")
    print("\nEach person opens their HTML, reviews, then clicks 'Export Decisions JSON'.")
    print("Merge with: python scripts/merge_review_decisions.py <file_a.json> <file_b.json>")


if __name__ == "__main__":
    main()
