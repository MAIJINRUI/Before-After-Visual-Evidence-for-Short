#!/usr/bin/env python3
"""
human_verify.py
===============
随机抽取样本，格式化展示，供人工验证逻辑一致性。

v9: 增加 before/after 图像差异提示。

Usage:
    python scripts/human_verify.py              # 每种类型抽 3 个
    python scripts/human_verify.py --per_type 5  # 每种类型抽 5 个
    python scripts/human_verify.py --type F3     # 只看 F3
    python scripts/human_verify.py --id S0042    # 看指定 sample
"""

import json
import random
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLES_PATH = ROOT / "data" / "samples.jsonl"


def load_samples():
    samples = []
    with open(SAMPLES_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _images_are_different(s):
    """Check whether before and after images differ (by file size)."""
    bp = s.get("before_image", "")
    ap = s.get("after_image", "")
    if not bp or not ap:
        return None  # unknown
    bf = ROOT / bp
    af = ROOT / ap
    if not bf.is_file() or not af.is_file():
        return None
    return bf.stat().st_size != af.stat().st_size


def display_sample(s, idx=None):
    """Pretty-print one sample for human review."""
    header = f"{'─' * 65}"
    if idx is not None:
        header = f"{'─' * 25} [{idx}] {'─' * 25}"

    print(header)
    print(f"  📌 Sample ID:     {s['sample_id']}")
    print(f"  🏷️  Failure Type:  {s['failure_type']}")
    print(f"  📝 Instruction:   {s['instruction']}")
    print()
    print(f"  ❌ Attempted:     {s['attempted_action']}")
    print(f"  ✅ Gold Recovery: {s['gold_recovery_action']}")
    print(f"  💬 Diagnosis:     {s.get('diagnosis', 'N/A')}")
    print()

    sb = s.get("state_before", {})
    print(f"  📦 State Before:")
    print(f"     holding:  {sb.get('holding', [])}")
    print(f"     near:     {sb.get('near', [])}")
    print(f"     visible:  {sb.get('visible', [])}")
    print(f"     open:     {sb.get('open', [])}")
    print(f"     in:       {sb.get('in', [])}")
    print()

    gs = s.get("goal_state", {})
    print(f"  🎯 Goal State:")
    print(f"     holding:  {gs.get('holding', [])}")
    print(f"     near:     {gs.get('near', [])}")
    print(f"     open:     {gs.get('open', [])}")
    print(f"     in:       {gs.get('in', [])}")
    print()

    # ── Image info with v9 visual diff indicator ──
    img_diff = _images_are_different(s)
    if img_diff is True:
        diff_tag = "📸 DIFFERENT"
    elif img_diff is False:
        diff_tag = "📸 SAME"
    else:
        diff_tag = "📸 N/A"

    print(f"  🖼️  Before Image: {s.get('before_image', 'N/A')}")
    print(f"  🖼️  After Image:  {s.get('after_image', 'N/A')}")
    print(f"  {diff_tag}")
    print(f"  📂 Source:        {s.get('source', 'N/A')}")
    print(f"  📍 Step Index:    {s.get('step_index', 'N/A')}")
    print(f"  🏠 Task Type:     {s.get('task_type', 'N/A')}")
    print()

    # ── Consistency quick-check hints ──
    ft = s["failure_type"]
    hints = []

    if ft == "F1":
        # Target should NOT be in visible
        att_args = s["attempted_action"].split("(")[1].rstrip(")").split(",")
        target = att_args[0].strip() if att_args else ""
        if target in sb.get("visible", []):
            hints.append(f"⚠️  F1: target '{target}' IS in visible — should NOT be")
        if s["gold_recovery_action"] != "LookAround()":
            hints.append(f"⚠️  F1: gold should be LookAround()")
        # v9: F1 should have same before/after
        if img_diff is True:
            hints.append("⚠️  F1: before/after images DIFFER (expected same — action failed)")

    elif ft == "F2":
        if not s["attempted_action"].startswith("Pick("):
            hints.append("⚠️  F2: attempted should be Pick(wrong_obj)")
        if not s["gold_recovery_action"].startswith("Pick("):
            hints.append("⚠️  F2: gold should be Pick(correct_obj)")
        # Check wrong != right
        att_obj = s["attempted_action"].split("(")[1].rstrip(")")
        gold_obj = s["gold_recovery_action"].split("(")[1].rstrip(")")
        if att_obj == gold_obj:
            hints.append(f"⚠️  F2: wrong==right: '{att_obj}'")
        if att_obj in sb.get("holding", []):
            hints.append(f"⚠️  F2: wrong_obj '{att_obj}' in holding")
        # v9: F2 should ideally have different before/after
        if img_diff is False:
            hints.append("ℹ️  F2: before/after images are SAME (fallback — ideally different)")
        elif img_diff is True:
            hints.append("✅ F2: before/after images differ (visual signal present)")

    elif ft == "F3":
        gold_parts = s["gold_recovery_action"].split("(")
        gold_name = gold_parts[0]
        gold_arg = gold_parts[1].rstrip(")") if len(gold_parts) > 1 else ""
        if gold_name == "Navigate":
            if gold_arg in sb.get("near", []):
                hints.append(f"⚠️  F3: Navigate({gold_arg}) but already in near")
        elif gold_name == "Pick":
            if gold_arg in sb.get("holding", []):
                hints.append(f"⚠️  F3: Pick({gold_arg}) but already holding")
        # v9: F3 should have same before/after
        if img_diff is True:
            hints.append("⚠️  F3: before/after images DIFFER (expected same — action failed)")

    elif ft == "F4":
        if not s["gold_recovery_action"].startswith("Retry("):
            hints.append("⚠️  F4: gold should be Retry(...)")
        # v9: F4 should have same before/after
        if img_diff is True:
            hints.append("⚠️  F4: before/after images DIFFER (expected same — no effect)")

    elif ft == "F5":
        if not s["attempted_action"].startswith("Navigate("):
            hints.append("⚠️  F5: attempted should be Navigate(wrong)")
        if not s["gold_recovery_action"].startswith("Navigate("):
            hints.append("⚠️  F5: gold should be Navigate(correct)")
        att_loc = s["attempted_action"].split("(")[1].rstrip(")")
        gold_loc = s["gold_recovery_action"].split("(")[1].rstrip(")")
        if att_loc == gold_loc:
            hints.append(f"⚠️  F5: wrong==right: '{att_loc}'")
        # v9: F5 should ideally have different before/after
        if img_diff is False:
            hints.append("ℹ️  F5: before/after images are SAME (fallback — ideally different)")
        elif img_diff is True:
            hints.append("✅ F5: before/after images differ (visual signal present)")

    if hints:
        print("  🔍 Auto-check hints:")
        for h in hints:
            print(f"     {h}")
    else:
        print("  🔍 Auto-check: ✅ No issues detected")

    print()


def main():
    parser = argparse.ArgumentParser(description="Human verification of samples")
    parser.add_argument("--per_type", type=int, default=3, help="Samples per failure type")
    parser.add_argument("--type", type=str, default=None, help="Only show this failure type (F1-F5)")
    parser.add_argument("--id", type=str, default=None, help="Show specific sample_id")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    samples = load_samples()
    random.seed(args.seed)

    # ── Specific sample ──
    if args.id:
        found = [s for s in samples if s["sample_id"] == args.id]
        if not found:
            print(f"❌ Sample '{args.id}' not found")
            return
        print(f"\n🔍 Showing sample {args.id}\n")
        display_sample(found[0])
        return

    # ── Filter by type ──
    types = [args.type] if args.type else ["F1", "F2", "F3", "F4", "F5"]

    total = 0
    for ft in types:
        ft_samples = [s for s in samples if s["failure_type"] == ft]
        if not ft_samples:
            print(f"\n⚠️  No samples for {ft}")
            continue

        selected = random.sample(ft_samples, min(args.per_type, len(ft_samples)))

        print(f"\n{'═' * 65}")
        print(f"  {ft}  —  {len(selected)} samples (of {len(ft_samples)} total)")
        print(f"{'═' * 65}")

        for i, s in enumerate(selected, 1):
            display_sample(s, idx=i)
            total += 1

    print(f"{'═' * 65}")
    print(f"  Total displayed: {total} samples")
    print(f"  ✏️  Check each sample for:")
    print(f"     1. Diagnosis makes sense given state_before")
    print(f"     2. Gold recovery logically fixes the failure")
    print(f"     3. State_before is consistent (no contradictions)")
    print(f"     4. Goal_state matches what gold recovery achieves")
    print(f"     5. Visual signal: F2/F5 ideally have different images")
    print(f"{'═' * 65}\n")


if __name__ == "__main__":
    main()