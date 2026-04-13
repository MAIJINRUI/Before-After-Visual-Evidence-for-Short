#!/usr/bin/env python3
"""
Merge partner review decision JSON files into one. Keeps only 'keep' decisions.
Usage:
  python merge_review_decisions.py manual_review_decisions_partner_A.json manual_review_decisions_partner_B.json
  python merge_review_decisions.py *.json  # if both files in current dir
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SAMPLES_PATH = DATA_DIR / "samples.jsonl"
OUT_JSONL = DATA_DIR / "samples_kept.jsonl"
OUT_SPLIT = DATA_DIR / "split_kept.json"


def main():
    parser = argparse.ArgumentParser(description="Merge partner review decisions")
    parser.add_argument("files", nargs="+", help="JSON files from Export Decisions")
    parser.add_argument("-o", "--output-dir", default=str(DATA_DIR), help="Output directory")
    args = parser.parse_args()

    merged = {}
    for path in args.files:
        p = Path(path)
        if not p.exists():
            print(f"Warning: {p} not found, skipping")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if "decisions" in data:
            decisions = data["decisions"]
        elif isinstance(data, dict):
            decisions = data
        else:
            print(f"Warning: unexpected format in {p}, skipping")
            continue
        for sid, status in decisions.items():
            if status == "keep":
                merged[sid] = status
            # discard/unreviewed: don't add to merged

    kept_ids = set(merged.keys())
    print(f"Merged: {len(kept_ids)} Keep decisions from {len(args.files)} file(s)")

    if not SAMPLES_PATH.exists():
        print(f"Warning: {SAMPLES_PATH} not found. Only writing merged decisions.")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "manual_review_decisions_merged.json"
        out_path.write_text(
            json.dumps(
                {
                    "schema": "dase3156_manual_review_v1",
                    "merged_from": args.files,
                    "decisions": merged,
                    "keep_count": len(kept_ids),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {out_path}")
        return

    # Build samples_kept.jsonl and split_kept.json
    samples = [json.loads(line) for line in open(SAMPLES_PATH, encoding="utf-8")]
    kept_samples = [s for s in samples if s["sample_id"] in kept_ids]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "samples_kept.jsonl", "w", encoding="utf-8") as f:
        for s in kept_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Reuse split structure from split.json if exists
    split_path = DATA_DIR / "split.json"
    if split_path.exists():
        split_data = json.loads(split_path.read_text(encoding="utf-8"))
        kept_ids_set = set(kept_ids)
        split_kept = {}
        for k, v in split_data.items():
            if isinstance(v, list):
                split_kept[k] = [x for x in v if x in kept_ids_set]
            else:
                split_kept[k] = v
    else:
        split_kept = {"train": [s["sample_id"] for s in kept_samples[: int(len(kept_samples) * 0.8)]}
        split_kept["val"] = [s["sample_id"] for s in kept_samples[int(len(kept_samples) * 0.8) :]]

    with open(out_dir / "split_kept.json", "w", encoding="utf-8") as f:
        json.dump(split_kept, f, indent=2, ensure_ascii=False)

    # Per-type stats
    from collections import Counter
    by_type = Counter(s["failure_type"] for s in kept_samples)
    print("\nPer failure type (kept):")
    for ft in ["F1", "F2", "F3", "F4", "F5"]:
        print(f"  {ft}: {by_type.get(ft, 0)}")

    print(f"\nWrote {out_dir / 'samples_kept.jsonl'} ({len(kept_samples)} samples)")
    print(f"Wrote {out_dir / 'split_kept.json'}")


if __name__ == "__main__":
    main()
