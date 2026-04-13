#!/usr/bin/env python3
"""
sanity_check.py  v14.2
=======================
Dataset validation — updated for v15 generation policy.

v14.2 changes over v14.1:
- EMPTY_STATE_THRESH raised for F1 (0.30→0.50) and F3 (0.20→0.40) to
  accommodate pre-goto-state samples with sparser state.
- Added comment about file-size fallback limitation in check_visual_diversity.

Usage:
    python scripts/sanity_check.py
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluator import load_jsonl, load_split
from src.parser import validate_action

SAMPLES_PATH = ROOT / "data" / "samples.jsonl"
SPLIT_PATH = ROOT / "data" / "split.json"
MANIFEST_PATH = ROOT / "data" / "image_manifest.json"
IMG_DIR = ROOT / "data" / "images"

# v14: F2 → same (before pick, both objects visible), F4 → same (no effect)
VISUAL_POLICY = {
    "F1": "same",
    "F2": "same",
    "F3": "same",
    "F4": "same",
    "F5": "diff",
}

EXPECTED_DIFF_RATIO = {
    "F1": (0.00, 0.10),
    "F2": (0.00, 0.10),
    "F3": (0.00, 0.10),
    "F4": (0.00, 0.10),
    "F5": (0.80, 1.00),
}

# v14.2: raised for F1/F3 — they use pre-goto state which is naturally sparser
EMPTY_STATE_THRESH = {
    "F1": 0.50,
    "F2": 0.10,
    "F3": 0.40,
    "F4": 0.30,
    "F5": 0.30,
}


def parse_action(action_str):
    m = re.match(r"^(\w+)\((.*)\)$", action_str.strip())
    if not m:
        return None, []
    name = m.group(1)
    args_str = m.group(2).strip()
    args = [a.strip() for a in args_str.split(",")] if args_str else []
    return name, args


def load_manifest(path):
    """Load manifest; return empty list if not found."""
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ======================================================================
# Report helper
# ======================================================================

class Report:
    def __init__(self):
        self._current = {}

    def _ensure(self, tag):
        if tag not in self._current:
            self._current[tag] = {"pass": [], "warn": [], "fail": []}

    def ok(self, tag, msg, sid=""):
        self._ensure(tag)
        self._current[tag]["pass"].append((msg, sid))

    def warn(self, tag, msg, sid=""):
        self._ensure(tag)
        self._current[tag]["warn"].append((msg, sid))

    def fail(self, tag, msg, sid=""):
        self._ensure(tag)
        self._current[tag]["fail"].append((msg, sid))

    def summary(self):
        p = sum(len(v["pass"]) for v in self._current.values())
        w = sum(len(v["warn"]) for v in self._current.values())
        f = sum(len(v["fail"]) for v in self._current.values())
        return p, w, f

    def print_report(self):
        p_total, w_total, f_total = self.summary()

        print("\n" + "=" * 65)
        print("  SANITY CHECK REPORT")
        print("=" * 65)

        for tag in sorted(self._current.keys()):
            data = self._current[tag]
            fails = data["fail"]
            warns = data["warn"]
            passes = data["pass"]

            if fails:
                icon = "❌"
            elif warns:
                icon = "⚠️ "
            else:
                icon = "✅"

            print(f"\n{icon} [{tag}]  pass={len(passes)}  warn={len(warns)}  fail={len(fails)}")

            for msg, sid in fails[:5]:
                label = f" [{sid}] " if sid else " "
                print(f"    FAIL{label}{msg}")
            if len(fails) > 5:
                print(f"    ... and {len(fails) - 5} more FAIL")

            for msg, sid in warns[:5]:
                label = f" [{sid}] " if sid else " "
                print(f"    WARN{label}{msg}")
            if len(warns) > 5:
                print(f"    ... and {len(warns) - 5} more WARN")

        print(f"\n{'=' * 65}")
        print(f"  TOTALS:  ✅ {p_total}  ⚠️  {w_total}  ❌ {f_total}")
        if f_total == 0 and w_total == 0:
            print("  🎉 ALL CHECKS PASSED!")
        elif f_total == 0:
            print("  👍 No failures — review warnings above.")
        else:
            print("  🔧 Please fix the failures above.")
        print("=" * 65 + "\n")

        return f_total


# ======================================================================
# Core checks (schema, gold validity, split, counts)
# ======================================================================

def check_original(samples, split, report):
    tag_schema = "C01-schema"
    tag_gold = "C02-gold-valid"
    tag_split = "C03-split"
    tag_counts = "C04-counts"

    required_fields = {
        "sample_id", "instruction", "attempted_action",
        "before_image", "after_image", "source_type",
        "failure_type", "gold_recovery_action",
        "candidate_vocab", "state_before", "goal_state",
    }

    schema_fail = False
    for s in samples:
        sid = s.get("sample_id", "?")
        missing = required_fields - set(s.keys())
        if missing:
            schema_fail = True
            report.fail(tag_schema, f"missing fields: {missing}", sid)
    if not schema_fail:
        report.ok(tag_schema, f"All {len(samples)} samples have required fields")

    gold_fail = False
    for s in samples:
        sid = s["sample_id"]
        ok, err = validate_action(s["gold_recovery_action"], s["candidate_vocab"])
        if not ok:
            gold_fail = True
            report.fail(tag_gold, f"'{s['gold_recovery_action']}': {err}", sid)
    if not gold_fail:
        report.ok(tag_gold, "All gold_recovery_actions valid")

    all_ids = {s["sample_id"] for s in samples}
    split_ids = set(split["train"] + split["val"] + split["test"])

    if all_ids != split_ids:
        diff1 = all_ids - split_ids
        diff2 = split_ids - all_ids
        if diff1:
            report.fail(tag_split, f"{len(diff1)} sample IDs missing from split")
        if diff2:
            report.fail(tag_split, f"{len(diff2)} split IDs not in samples")
    else:
        report.ok(tag_split, "split covers all sample IDs")

    overlap_tv = set(split["train"]) & set(split["val"])
    overlap_tt = set(split["train"]) & set(split["test"])
    overlap_vt = set(split["val"]) & set(split["test"])
    if overlap_tv or overlap_tt or overlap_vt:
        report.fail(tag_split, f"Split overlap: tv={len(overlap_tv)} tt={len(overlap_tt)} vt={len(overlap_vt)}")
    else:
        report.ok(tag_split, f"Split disjoint: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")

    by_ft = Counter(s["failure_type"] for s in samples)
    for ft in ["F1", "F2", "F3", "F4", "F5"]:
        n = by_ft.get(ft, 0)
        if n == 60:
            report.ok(tag_counts, f"{ft}: {n}")
        elif n >= 55:
            report.warn(tag_counts, f"{ft}: {n} (close to target 60)")
        else:
            report.fail(tag_counts, f"{ft}: {n} (expected ~60)")

    total = len(samples)
    if total == 300:
        report.ok(tag_counts, f"Total: {total}")
    elif total >= 250:
        report.warn(tag_counts, f"Total: {total} (expected 300)")
    else:
        report.fail(tag_counts, f"Total: {total} (too low)")


# ======================================================================
# Vocabulary coverage
# ======================================================================

def check_vocab_coverage(samples, report):
    tag = "C05-vocab-cover"

    failed = False
    for s in samples:
        sid = s["sample_id"]
        cv = s["candidate_vocab"]

        for field in ["attempted_action", "gold_recovery_action"]:
            name, args = parse_action(s[field])
            if name is None:
                continue

            if name == "Navigate" and args:
                if args[0] not in cv.get("targets", []) and args[0] not in cv.get("locations", []):
                    failed = True
                    report.fail(tag, f"{field}: Navigate({args[0]}) not in targets/locations", sid)

            elif name == "Pick" and args:
                if args[0] not in cv.get("objects", []) and args[0] not in cv.get("targets", []):
                    failed = True
                    report.fail(tag, f"{field}: Pick({args[0]}) not in objects/targets", sid)

            elif name == "Place" and len(args) >= 2:
                if args[0] not in cv.get("objects", []):
                    failed = True
                    report.fail(tag, f"{field}: Place obj={args[0]} not in objects", sid)
                if args[1] not in cv.get("locations", []):
                    failed = True
                    report.fail(tag, f"{field}: Place loc={args[1]} not in locations", sid)

            elif name == "Open" and args:
                if args[0] not in cv.get("containers", []):
                    failed = True
                    report.fail(tag, f"{field}: Open({args[0]}) not in containers", sid)

            elif name == "Close" and args:
                if args[0] not in cv.get("containers", []):
                    failed = True
                    report.fail(tag, f"{field}: Close({args[0]}) not in containers", sid)

            elif name == "Retry" and args:
                if args[0] not in cv.get("retryable_actions", []):
                    failed = True
                    report.fail(tag, f"{field}: Retry({args[0]}) not in retryable_actions", sid)

    if not failed:
        report.ok(tag, "All action args covered by candidate_vocab")


# ======================================================================
# Goal state consistency
# ======================================================================

def check_goal_state(samples, report):
    tag = "G01-goal"
    failed = False

    for s in samples:
        sid = s["sample_id"]
        gs = s["goal_state"]
        gold_name, gold_args = parse_action(s["gold_recovery_action"])

        if gold_name == "Pick" and gold_args:
            if gold_args[0] not in gs.get("holding", []):
                failed = True
                report.fail(tag, f"Pick({gold_args[0]}) but goal.holding={gs.get('holding')}", sid)

        elif gold_name == "Place" and len(gold_args) >= 2:
            expected = f"{gold_args[0]}:{gold_args[1]}"
            if expected not in gs.get("in", []):
                failed = True
                report.fail(tag, f"Place but goal.in missing '{expected}'", sid)

        elif gold_name == "Navigate" and gold_args:
            if gold_args[0] not in gs.get("near", []):
                failed = True
                report.fail(tag, f"Navigate({gold_args[0]}) but goal.near={gs.get('near')}", sid)

        elif gold_name == "Open" and gold_args:
            if gold_args[0] not in gs.get("open", []):
                failed = True
                report.fail(tag, f"Open({gold_args[0]}) but goal.open={gs.get('open')}", sid)

    if not failed:
        report.ok(tag, f"Goal states consistent ({len(samples)} samples)")


# ======================================================================
# Image existence
# ======================================================================

def check_images(samples, report, manifest=None):
    """v14.1: uses manifest same_frame flag when available, falls back to file-size."""
    tag = "I01-images"
    missing = 0
    failed = False

    # v14.1: build manifest lookup
    manifest_map = {}
    if manifest:
        manifest_map = {m["sample_id"]: m for m in manifest}

    for s in samples:
        sid = s["sample_id"]
        bp = s.get("before_image", "")
        ap = s.get("after_image", "")

        if not bp and not ap:
            missing += 1
            continue

        bf = ROOT / bp if bp else None
        af = ROOT / ap if ap else None

        if bf and not bf.is_file():
            failed = True
            report.fail(tag, f"before_image missing: {bp}", sid)
            missing += 1
        if af and not af.is_file():
            failed = True
            report.fail(tag, f"after_image missing: {ap}", sid)
            missing += 1

        if bf and af and bf.is_file() and af.is_file():
            policy = VISUAL_POLICY.get(s["failure_type"], "either")

            # v14.1: prefer manifest same_frame; fallback to file-size
            m = manifest_map.get(sid)
            if m and "same_frame" in m:
                is_same = m["same_frame"]
            else:
                is_same = bf.stat().st_size == af.stat().st_size

            if policy == "same" and not is_same:
                report.warn(tag, f"before/after differ for {s['failure_type']} (expected same)", sid)
            elif policy == "diff" and is_same:
                report.warn(tag, f"before/after same for {s['failure_type']} (expected diff)", sid)

    with_img = sum(1 for s in samples if s.get("before_image"))
    if not failed:
        report.ok(tag, f"{with_img}/{len(samples)} have valid image pairs")
    if missing > 0:
        report.warn(tag, f"{missing} samples without images")


# ======================================================================
# v14: Image-text alignment (uses manifest)
# ======================================================================

def check_image_text_alignment(samples, manifest, report):
    """Cross-check image timing / same_frame against failure-type semantics."""
    tag = "A01-img-text-align"

    if not manifest:
        report.warn(tag, "No manifest loaded; skipping image-text alignment checks")
        return

    manifest_map = {m["sample_id"]: m for m in manifest}
    failed = False

    for s in samples:
        sid = s["sample_id"]
        ft = s["failure_type"]
        m = manifest_map.get(sid)
        if not m:
            continue

        # F1: image_step_idx should be < step_idx (before navigation to target)
        if ft == "F1":
            if m["image_step_idx"] >= m["step_idx"]:
                failed = True
                report.fail(
                    tag,
                    f"F1 image_step_idx ({m['image_step_idx']}) >= step_idx ({m['step_idx']}); "
                    f"image may show target",
                    sid,
                )

        # F2: should have same before/after (no correct-pick result shown)
        if ft == "F2":
            if not m.get("same_frame", True):
                failed = True
                report.fail(tag, f"F2 has different before/after (may show correct pick result)", sid)

        # F4: should have same before/after (no visible effect)
        if ft == "F4":
            if not m.get("same_frame", True):
                failed = True
                report.fail(tag, f"F4 has different before/after (contradicts 'no visible effect')", sid)

        # F5: should have different before/after (wrong location shown)
        if ft == "F5":
            if m.get("same_frame", True):
                failed = True
                report.fail(tag, f"F5 has same before/after (should show wrong location)", sid)

    if not failed:
        report.ok(tag, f"Image-text alignment OK for {len(samples)} samples")


# ======================================================================
# Visual diversity
# ======================================================================

def check_visual_diversity(samples, manifest, report):
    """Check per-failure-type before/after visual diversity.

    Prefers manifest same_frame flag; falls back to file-size comparison.

    NOTE (v14.2): the file-size fallback is an approximation.  Two different
    JPEG images may coincidentally share the same file size, and identical
    source files copied to before/after will always match.  When manifest is
    present this limitation does not apply.
    """
    tag = "V01-visual-div"

    manifest_map = {}
    if manifest:
        manifest_map = {m["sample_id"]: m for m in manifest}

    diff_counts = defaultdict(int)
    same_counts = defaultdict(int)

    for s in samples:
        sid = s["sample_id"]
        ft = s["failure_type"]

        # Try manifest first
        m = manifest_map.get(sid)
        if m and "same_frame" in m:
            if m["same_frame"]:
                same_counts[ft] += 1
            else:
                diff_counts[ft] += 1
            continue

        # Fallback: file-size comparison (see NOTE above)
        bp = s.get("before_image", "")
        ap = s.get("after_image", "")
        if not bp or not ap:
            continue

        bf = ROOT / bp
        af = ROOT / ap
        if not bf.is_file() or not af.is_file():
            continue

        same = bf.stat().st_size == af.stat().st_size
        if same:
            same_counts[ft] += 1
        else:
            diff_counts[ft] += 1

    for ft in ["F1", "F2", "F3", "F4", "F5"]:
        total_ft = diff_counts[ft] + same_counts[ft]
        if total_ft == 0:
            continue

        ratio = diff_counts[ft] / total_ft
        lo, hi = EXPECTED_DIFF_RATIO[ft]

        if not (lo <= ratio <= hi):
            report.warn(tag, f"{ft}: diff ratio {ratio:.2f} outside expected [{lo:.2f}, {hi:.2f}]")
        else:
            report.ok(tag, f"{ft}: {diff_counts[ft]}/{total_ft} different ({ratio:.0%})")

    total_diff = sum(diff_counts.values())
    total_all = total_diff + sum(same_counts.values())
    if total_all > 0:
        report.ok(tag, f"Overall: {total_diff}/{total_all} pairs visually different")


# ======================================================================
# Empty-state ratio
# ======================================================================

def check_empty_state_ratio(samples, report):
    tag = "Q02-empty-state"

    by_ft = defaultdict(list)
    for s in samples:
        by_ft[s["failure_type"]].append(s)

    for ft, group in sorted(by_ft.items()):
        empty = 0
        for s in group:
            sb = s.get("state_before", {})
            if (not sb.get("visible")
                and not sb.get("holding")
                and not sb.get("near")
                and not sb.get("open")
                and not sb.get("in")):
                empty += 1
        ratio = empty / len(group) if group else 0.0
        th = EMPTY_STATE_THRESH.get(ft, 1.0)
        if ratio > th:
            report.warn(tag, f"{ft}: empty-state ratio {ratio:.0%} > threshold {th:.0%}")
        else:
            report.ok(tag, f"{ft}: empty-state ratio {ratio:.0%}")


# ======================================================================
# Action diversity
# ======================================================================

def check_action_diversity(samples, report):
    tag = "Q03-action-diversity"

    by_ft = defaultdict(list)
    for s in samples:
        by_ft[s["failure_type"]].append(s)

    for ft, group in sorted(by_ft.items()):
        attempted_types = Counter(parse_action(s["attempted_action"])[0] for s in group)
        gold_types = Counter(parse_action(s["gold_recovery_action"])[0] for s in group)

        if ft == "F3":
            if len(attempted_types) < 2:
                report.warn(tag, f"{ft}: attempted action types too narrow: {dict(attempted_types)}")
            else:
                report.ok(tag, f"{ft}: attempted diversity {dict(attempted_types)}")

            if len(gold_types) < 2:
                report.warn(tag, f"{ft}: gold recovery types too narrow: {dict(gold_types)}")
            else:
                report.ok(tag, f"{ft}: gold diversity {dict(gold_types)}")
        else:
            report.ok(tag, f"{ft}: attempted={dict(attempted_types)} gold={dict(gold_types)}")


# ======================================================================
# Semantic logic per failure type
# ======================================================================

def check_semantic_logic(samples, report):
    tags = {
        "F1": "L01-F1",
        "F2": "L02-F2",
        "F3": "L03-F3",
        "F4": "L04-F4",
        "F5": "L05-F5",
    }

    grouped = defaultdict(list)
    for s in samples:
        grouped[s["failure_type"]].append(s)

    # ---- F1 ----
    failed = False
    for s in grouped["F1"]:
        sid = s["sample_id"]
        _, args = parse_action(s["attempted_action"])
        target = args[0] if args else ""
        visible = s["state_before"].get("visible", [])
        near = s["state_before"].get("near", [])
        if target and target in visible:
            failed = True
            report.fail(tags["F1"], f"target '{target}' unexpectedly in visible", sid)
        # v14: also check near
        if target and target in near:
            failed = True
            report.fail(tags["F1"], f"target '{target}' unexpectedly in near", sid)
        if s["gold_recovery_action"] != "LookAround()":
            failed = True
            report.fail(tags["F1"], f"gold='{s['gold_recovery_action']}' expected LookAround()", sid)
    if not failed:
        report.ok(tags["F1"], f"F1: {len(grouped['F1'])} samples OK")

    # ---- F2 ----
    failed = False
    for s in grouped["F2"]:
        sid = s["sample_id"]
        att_name, att_args = parse_action(s["attempted_action"])
        gold_name, gold_args = parse_action(s["gold_recovery_action"])
        if att_name == "Pick" and gold_name == "Pick":
            wrong = att_args[0] if att_args else ""
            right = gold_args[0] if gold_args else ""
            if wrong == right:
                failed = True
                report.fail(tags["F2"], f"wrong==right='{wrong}'", sid)
    if not failed:
        report.ok(tags["F2"], f"F2: {len(grouped['F2'])} samples OK")

    # ---- F3 ----
    failed = False
    for s in grouped["F3"]:
        sid = s["sample_id"]
        gold_name, gold_args = parse_action(s["gold_recovery_action"])

        # F3B: Pick(obj) → obj must NOT already be in holding
        if gold_name == "Pick" and gold_args:
            holding = s["state_before"].get("holding", [])
            if gold_args[0] in holding:
                failed = True
                report.fail(tags["F3"], f"Pick({gold_args[0]}) but already holding", sid)

        # v14: F3A/F3C: Navigate(dest) → dest must NOT already be in near
        if gold_name == "Navigate" and gold_args:
            near = s["state_before"].get("near", [])
            if gold_args[0] in near:
                failed = True
                report.fail(tags["F3"], f"Navigate({gold_args[0]}) but already near", sid)

    if not failed:
        report.ok(tags["F3"], f"F3: {len(grouped['F3'])} samples OK")

    # ---- F4 ----
    failed = False
    expected_map = {
        "PickupObject": "Pick",
        "PutObject": "Place",
        "OpenObject": "Open",
        "CloseObject": "Close",
        "Pick": "Pick",
        "Place": "Place",
        "Open": "Open",
        "Close": "Close",
        "Navigate": "Navigate",
    }
    for s in grouped["F4"]:
        sid = s["sample_id"]
        att_name, _ = parse_action(s["attempted_action"])
        gold_name, gold_args = parse_action(s["gold_recovery_action"])
        if gold_name != "Retry":
            failed = True
            report.fail(tags["F4"], f"gold='{s['gold_recovery_action']}' not Retry", sid)
            continue
        retry_what = gold_args[0] if gold_args else ""
        expected = expected_map.get(att_name, "")
        if retry_what != expected:
            failed = True
            report.fail(tags["F4"], f"attempted={att_name} but Retry({retry_what})", sid)
    if not failed:
        report.ok(tags["F4"], f"F4: {len(grouped['F4'])} samples OK")

    # ---- F5 ----
    failed = False
    for s in grouped["F5"]:
        sid = s["sample_id"]
        _, att_args = parse_action(s["attempted_action"])
        _, gold_args = parse_action(s["gold_recovery_action"])
        wrong = att_args[0] if att_args else ""
        right = gold_args[0] if gold_args else ""
        if wrong == right:
            failed = True
            report.fail(tags["F5"], f"wrong==right='{wrong}'", sid)
    if not failed:
        report.ok(tags["F5"], f"F5: {len(grouped['F5'])} samples OK")


# ======================================================================
# Uniqueness
# ======================================================================

def check_uniqueness(samples, report):
    tag = "U01-unique"

    ids = [s["sample_id"] for s in samples]
    dup_ids = [sid for sid, cnt in Counter(ids).items() if cnt > 1]
    if dup_ids:
        report.fail(tag, f"Duplicate sample_ids: {dup_ids[:5]}")
    else:
        report.ok(tag, f"All {len(ids)} sample_ids unique")

    combos = Counter()
    for s in samples:
        key = (s.get("source", ""), s.get("step_index", ""),
               s["attempted_action"], s["gold_recovery_action"])
        combos[key] += 1
    dups = {k: v for k, v in combos.items() if v > 1}
    if dups:
        report.warn(tag, f"{len(dups)} duplicate (source, step, attempted, gold) combos")
    else:
        report.ok(tag, "No duplicate combos")


# ======================================================================
# Cross-type overlap
# ======================================================================

def check_cross_type_overlap(samples, report):
    tag = "X01-overlap"
    patterns = defaultdict(list)
    for s in samples:
        att_name, _ = parse_action(s["attempted_action"])
        gold_name, _ = parse_action(s["gold_recovery_action"])
        patterns[(att_name, gold_name)].append(s["failure_type"])

    found = False
    for (att, gold), fts in sorted(patterns.items()):
        ft_set = set(fts)
        if len(ft_set) > 1:
            counts = Counter(fts)
            report.warn(tag, f"({att}→{gold}) shared by {dict(counts)}")
            found = True

    if not found:
        report.ok(tag, "No cross-type pattern overlap")


# ======================================================================
# Distribution / legacy report (informational)
# ======================================================================

def print_distribution(samples):
    print("\n📊 DISTRIBUTION SUMMARY")
    print("-" * 45)

    by_ft = Counter(s["failure_type"] for s in samples)
    for ft in sorted(by_ft):
        bar = "█" * (by_ft[ft] // 2)
        print(f"  {ft}: {by_ft[ft]:3d}  {bar}")

    print("\n  Attempted action types per failure type:")
    for ft in sorted(by_ft):
        ft_samples = [s for s in samples if s["failure_type"] == ft]
        att_types = Counter(parse_action(s["attempted_action"])[0] for s in ft_samples)
        print(f"    {ft}: {dict(att_types)}")

    print("\n  Gold recovery action types per failure type:")
    for ft in sorted(by_ft):
        ft_samples = [s for s in samples if s["failure_type"] == ft]
        gold_types = Counter(parse_action(s["gold_recovery_action"])[0] for s in ft_samples)
        print(f"    {ft}: {dict(gold_types)}")

    print()
    by_src = Counter(s.get("source_type", "?") for s in samples)
    print(f"  Source types: {dict(by_src)}")

    task_types = Counter(s.get("task_type", "?") for s in samples)
    print(f"  Task types ({len(task_types)}): {dict(task_types.most_common(8))}")

    print("\n  📸 Visual diversity (before ≠ after):")
    for ft in sorted(by_ft):
        ft_samples = [s for s in samples if s["failure_type"] == ft]
        diff = 0
        same = 0
        for s in ft_samples:
            bp = s.get("before_image", "")
            ap = s.get("after_image", "")
            if not bp or not ap:
                continue
            bf = ROOT / bp
            af = ROOT / ap
            if bf.is_file() and af.is_file():
                if bf.stat().st_size != af.stat().st_size:
                    diff += 1
                else:
                    same += 1
        total = diff + same
        pct = f"{diff/total*100:.0f}%" if total else "N/A"
        print(f"    {ft}: {diff}/{total} different ({pct})")


def print_legacy_report(samples, split):
    failure_counts = Counter(s["failure_type"] for s in samples)
    source_counts = Counter(s["source_type"] for s in samples)

    invalid_gold = 0
    for s in samples:
        ok, _ = validate_action(s["gold_recovery_action"], s["candidate_vocab"])
        if not ok:
            invalid_gold += 1

    legacy = {
        "num_samples": len(samples),
        "invalid_gold_actions": invalid_gold,
        "failure_counts": dict(failure_counts),
        "source_counts": dict(source_counts),
        "split_sizes": {k: len(v) for k, v in split.items()},
        "split_covers_all_ids": {s["sample_id"] for s in samples}
                                 == set(split["train"] + split["val"] + split["test"]),
        "split_disjoint": (
            len(set(split["train"]) & set(split["val"])) == 0
            and len(set(split["train"]) & set(split["test"])) == 0
            and len(set(split["val"]) & set(split["test"])) == 0
        ),
    }
    print("\n📋 LEGACY REPORT (JSON):")
    print(json.dumps(legacy, indent=2, default=str))


# ======================================================================
# Main
# ======================================================================

def main():
    print("🔍 Loading dataset...")
    samples = load_jsonl(str(SAMPLES_PATH))
    split = load_split(str(SPLIT_PATH))
    manifest = load_manifest(MANIFEST_PATH)
    print(f"   Loaded {len(samples)} samples, {len(manifest)} manifest entries\n")

    report = Report()

    check_original(samples, split, report)
    check_vocab_coverage(samples, report)
    check_goal_state(samples, report)
    check_images(samples, report, manifest)
    check_image_text_alignment(samples, manifest, report)
    check_visual_diversity(samples, manifest, report)
    check_empty_state_ratio(samples, report)
    check_action_diversity(samples, report)
    check_semantic_logic(samples, report)
    check_cross_type_overlap(samples, report)
    check_uniqueness(samples, report)

    fail_count = report.print_report()
    print_distribution(samples)
    print_legacy_report(samples, split)

    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())