#!/usr/bin/env python3
"""
generate_dataset_from_alfred.py  v15
====================================
Generates higher-quality failure-injection samples from ALFRED full_2.1.0 trajectories.

v15 fixes over v14:
  1. quality_score: softer visible penalty (-2→-1); exempt F1/F3 from near
     penalty since they use pre-goto state where near is naturally empty.
  2. Quality threshold lowered from 2 to 1 to avoid over-filtering F1/F3.
  3. F1 injection restricted to GotoLocation and PickupObject for cleaner
     semantics (Place/Open as F1 targets had ambiguous "target" definition).
  4. F1 inject_failure removed strict "need at least some context" gate;
     quality_score handles it instead.
  5. build_state_before now tracks SliceObject and ToggleObject* in visible.
  6. alfred_to_closed returns None for unknown action types (was LookAround).
  7. get_frame_for_goto_step has fallback to next step's first frame.
  8. CANDIDATES_PER_TYPE is now per-type dict; F1/F3/F5 get larger pools.
  9. Added inject_attempts tracking and conversion-rate reporting.

Usage:
    python scripts/generate_dataset_from_alfred.py
"""

import json
import random
import re
import shutil
import sys
import pathlib
from collections import defaultdict

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ALFRED_ROOT = pathlib.Path(
    "/Volumes/T7/ALFRED Full Dataset (~109GB)/full_2.1.0"
)

OUT_DIR = PROJECT_ROOT / "data"
IMG_DIR = OUT_DIR / "images"
OUT_FILE = OUT_DIR / "samples.jsonl"
SPLIT_FILE = OUT_DIR / "split.json"
MANIFEST_FILE = OUT_DIR / "image_manifest.json"

FINAL_PER_TYPE = 60

# v15: per-type candidate pool sizes — F1/F3/F5 need larger pools
CANDIDATES_PER_TYPE = {
    "F1": 360,   # F1 has stricter pre-goto quality filter
    "F2": 240,
    "F3": 300,   # F3 needs subtype diversity
    "F4": 240,
    "F5": 300,   # F5 requires different frames, many skipped
}

SEED = 42
FAILURE_TYPES = ["F1", "F2", "F3", "F4", "F5"]

ALL_OBJECTS = [
    "alarmclock", "apple", "baseballbat", "basketball", "book", "boots",
    "bottle", "bowl", "box", "bread", "butterknife", "candle", "cd",
    "cellphone", "cloth", "creditcard", "cup", "egg", "fork",
    "handtowel", "kettle", "keychain", "knife", "ladle", "laptop",
    "lettuce", "mug", "newspaper", "pan", "pen", "pencil",
    "peppershaker", "pillow", "plate", "plunger", "pot", "potato",
    "remotecontrol", "saltshaker", "soapbar", "soapbottle", "spatula",
    "spoon", "spraybottle", "statue", "teddybear", "tissuebox",
    "toiletpaper", "tomato", "towel", "vase", "watch", "wateringcan",
]

ALL_RECEPTACLES = [
    "armchair", "bathtub", "bed", "cabinet", "cart", "coffeemachine",
    "coffeetable", "countertop", "desk", "desklamp", "diningtable",
    "drawer", "dresser", "floorlamp", "fridge", "garbagecan",
    "handtowelholder", "laundryhamper", "microwave", "ottoman", "safe",
    "shelf", "sidetable", "sink", "sinkbasin", "sofa", "stoveburner",
    "table", "toilet", "toiletpaperhanger", "towelholder",
]

OPENABLE = {"fridge", "microwave", "cabinet", "drawer", "safe", "box"}
PICKABLE_OBJECTS = set(ALL_OBJECTS)

random.seed(SEED)


# ======================================================================
# Trial discovery & loading
# ======================================================================

def find_trials(root):
    trials = []
    for split_name in ["train", "valid_seen", "valid_unseen"]:
        split_dir = root / split_name
        if not split_dir.is_dir():
            continue
        for task_dir in sorted(split_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            for trial_dir in sorted(task_dir.iterdir()):
                if (trial_dir / "traj_data.json").is_file():
                    trials.append((split_name, trial_dir))
    return trials


def load_traj(trial_dir):
    with open(trial_dir / "traj_data.json", encoding="utf-8") as f:
        return json.load(f)


def get_task_desc(traj):
    anns = traj.get("turk_annotations", {}).get("anns", [])
    return anns[0].get("task_desc", "") if anns else ""


def get_high_actions(traj):
    actions = []
    for step in traj.get("plan", {}).get("high_pddl", []):
        da = step.get("discrete_action", {})
        action_name = da.get("action", "")
        raw_args = da.get("args", [])

        if action_name in ("NoOp", "End", ""):
            continue

        args = []
        if isinstance(raw_args, list):
            for a in raw_args:
                if isinstance(a, str):
                    args.append(a.lower())

        actions.append({
            "action_name": action_name,
            "args": args,
            "high_idx": step.get("high_idx", len(actions)),
            "obj": args[0] if len(args) > 0 else "",
            "recep": args[1] if len(args) > 1 else "",
        })
    return actions


# ======================================================================
# Action-space mapping
# ======================================================================

def alfred_to_closed(action_name, args):
    """Map ALFRED high-level action to our closed action space.

    Returns None for actions that have no clean mapping (e.g. Toggle,
    or any unknown action type).
    """
    if action_name == "GotoLocation":
        return f"Navigate({args[0]})" if args else "LookAround()"
    if action_name == "PickupObject":
        return f"Pick({args[0]})" if args else "LookAround()"
    if action_name == "PutObject":
        return f"Place({args[0]},{args[1]})" if len(args) >= 2 else "LookAround()"
    if action_name == "OpenObject":
        return f"Open({args[0]})" if args else "LookAround()"
    if action_name == "CloseObject":
        return f"Close({args[0]})" if args else "LookAround()"
    if action_name == "SliceObject":
        return "Pick(knife)"
    # v14: ToggleObject has no mapping in our action space — return None
    if action_name in ("ToggleObjectOn", "ToggleObjectOff", "ToggleObject"):
        return None
    # v15: unknown action types also return None (was LookAround fallback)
    return None


# ======================================================================
# Image helpers
# ======================================================================

def resolve_image_file(trial_dir, image_name):
    if not image_name:
        return None
    stem = pathlib.Path(image_name).stem
    for subdir in ["raw_images", "high_res_images"]:
        d = trial_dir / subdir
        if not d.is_dir():
            continue
        for ext in [".jpg", ".png", ".jpeg"]:
            p = d / (stem + ext)
            if p.is_file():
                return p
    return None


def build_frame_index(traj):
    by_high = defaultdict(list)
    for img in traj.get("images", []):
        by_high[img.get("high_idx", -1)].append(img["image_name"])
    for h in by_high:
        by_high[h].sort()
    return by_high


def get_frame_before_step(by_high, step_idx):
    """Last frame of the previous step (= view just before step_idx starts)."""
    if step_idx - 1 in by_high and by_high[step_idx - 1]:
        return by_high[step_idx - 1][-1]
    if step_idx in by_high and by_high[step_idx]:
        return by_high[step_idx][0]
    if 0 in by_high and by_high[0]:
        return by_high[0][0]
    return None


def get_frame_after_step(by_high, step_idx):
    """Last frame of step_idx (= view right after it finishes)."""
    if step_idx in by_high and by_high[step_idx]:
        return by_high[step_idx][-1]
    if step_idx + 1 in by_high and by_high[step_idx + 1]:
        return by_high[step_idx + 1][0]
    return None


def get_frame_for_goto_step(by_high, goto_idx):
    """Last frame of a specific GotoLocation step (showing arrival view).

    v15: added fallback to next step's first frame.
    """
    if goto_idx in by_high and by_high[goto_idx]:
        return by_high[goto_idx][-1]
    # v15: fallback — next step's first frame (arrival is visible there)
    if goto_idx + 1 in by_high and by_high[goto_idx + 1]:
        return by_high[goto_idx + 1][0]
    return None


# ======================================================================
# Vocabulary helpers
# ======================================================================

def build_candidate_vocab(actions):
    objs, receps, targets = set(), set(), set()

    for a in actions:
        if a["obj"]:
            objs.add(a["obj"])
            targets.add(a["obj"])
        if a["recep"]:
            receps.add(a["recep"])
            targets.add(a["recep"])
        if a["action_name"] == "GotoLocation" and a["args"]:
            targets.add(a["args"][0])
            receps.add(a["args"][0])

    for o in random.sample(ALL_OBJECTS, min(5, len(ALL_OBJECTS))):
        objs.add(o)
    for r in random.sample(ALL_RECEPTACLES, min(3, len(ALL_RECEPTACLES))):
        receps.add(r)
        targets.add(r)
    targets |= objs

    containers = sorted(r for r in receps if r in OPENABLE)
    if not containers:
        containers = ["cabinet"]

    return {
        "objects": sorted(objs),
        "targets": sorted(targets),
        "locations": sorted(receps),
        "containers": containers,
        "retryable_actions": ["Pick", "Place", "Open", "Close", "Navigate"],
    }


def ensure_vocab_coverage(vocab, action_str):
    if action_str is None:
        return
    m = re.match(r"^(\w+)\((.*)\)$", action_str.strip())
    if not m:
        return
    name, args_str = m.group(1), m.group(2).strip()
    args = [a.strip() for a in args_str.split(",")] if args_str else []

    def _add(field, val):
        if val and val not in vocab[field]:
            vocab[field].append(val)
            vocab[field].sort()

    if name == "Navigate" and args:
        _add("targets", args[0])
        _add("locations", args[0])
    elif name == "Pick" and args:
        _add("objects", args[0])
        _add("targets", args[0])
    elif name == "Place" and len(args) >= 2:
        _add("objects", args[0])
        _add("locations", args[1])
        _add("targets", args[1])
    elif name == "Open" and args:
        _add("containers", args[0])
    elif name == "Close" and args:
        _add("containers", args[0])
    elif name == "Retry" and args:
        _add("retryable_actions", args[0])


# ======================================================================
# State helpers
# ======================================================================

def build_state_before(actions, step_idx):
    """Build symbolic state just before *step_idx*.

    `near`    – **reset** on each GotoLocation (agent can only be at one place).
    `visible` – accumulated (objects / locations ever acted upon).
                This is an approximation; true visibility needs image analysis.
    `holding` – tracks pick / place.
    `open`    – tracks open / close.
    `in`      – tracks successful Place(obj, recep).
    """
    state = {"visible": [], "holding": [], "near": [], "open": [], "in": []}
    for a in actions[:step_idx]:
        nm, obj, recep = a["action_name"], a["obj"], a["recep"]

        if nm == "GotoLocation":
            t = a["args"][0] if a["args"] else ""
            if t:
                # v14: agent can only be near one location at a time
                state["near"] = [t]
                if t not in state["visible"]:
                    state["visible"].append(t)

        elif nm == "PickupObject":
            if obj and obj not in state["holding"]:
                state["holding"].append(obj)
            if obj and obj not in state["visible"]:
                state["visible"].append(obj)

        elif nm == "PutObject":
            if obj and obj in state["holding"]:
                state["holding"].remove(obj)
            pair = f"{obj}:{recep}" if obj and recep else ""
            if pair and pair not in state["in"]:
                state["in"].append(pair)

        elif nm == "OpenObject":
            if obj and obj not in state["open"]:
                state["open"].append(obj)
            if obj and obj not in state["visible"]:
                state["visible"].append(obj)

        elif nm == "CloseObject":
            if obj and obj in state["open"]:
                state["open"].remove(obj)

        # v15: track SliceObject in visible (agent interacted with the object)
        elif nm == "SliceObject":
            if obj and obj not in state["visible"]:
                state["visible"].append(obj)

        # v15: track ToggleObject* in visible
        elif nm in ("ToggleObjectOn", "ToggleObjectOff"):
            if obj and obj not in state["visible"]:
                state["visible"].append(obj)

    return state


def deep_copy_state(state):
    return {k: list(v) for k, v in state.items()}


def get_objects_near_step(actions, step_idx):
    """Return pickable objects mentioned between the enclosing GotoLocations.

    These objects are likely to appear in the same image frame as the current step.
    """
    area_start = 0
    for i in range(step_idx - 1, -1, -1):
        if actions[i]["action_name"] == "GotoLocation":
            area_start = i
            break

    area_end = len(actions)
    for i in range(step_idx + 1, len(actions)):
        if actions[i]["action_name"] == "GotoLocation":
            area_end = i
            break

    objs = set()
    for i in range(area_start, min(area_end, len(actions))):
        a = actions[i]
        if a["obj"] and a["obj"] in PICKABLE_OBJECTS:
            objs.add(a["obj"])
    return objs


def build_goal_state(gold_action_str):
    goal = {"in": [], "holding": [], "near": [], "open": []}
    if gold_action_str is None:
        return goal
    m = re.match(r"^(\w+)\((.*)\)$", gold_action_str.strip())
    if not m:
        return goal
    name, args_str = m.group(1), m.group(2).strip()
    args = [a.strip() for a in args_str.split(",")] if args_str else []

    if name == "Pick" and args:
        goal["holding"] = [args[0]]
    elif name == "Place" and len(args) >= 2:
        goal["in"] = [f"{args[0]}:{args[1]}"]
    elif name == "Navigate" and args:
        goal["near"] = [args[0]]
    elif name == "Open" and args:
        goal["open"] = [args[0]]
    return goal


# ======================================================================
# Quality scoring
# ======================================================================

def quality_score(sample):
    """Score sample quality.  Higher is better.

    v15 changes:
    - Visible-empty penalty softened from -2 to -1.
    - F1 and F3 exempted from near-empty penalty (they use pre-goto state
      where near is naturally empty/reset).
    """
    score = 0
    sb = sample.get("state_before", {})
    ft = sample.get("failure_type", "")

    non_empty = sum(
        1 for k in ["visible", "holding", "near", "open", "in"]
        if sb.get(k)
    )
    score += non_empty

    if not sample.get("same_frame", True):
        score += 2

    # v15: softer penalty (was -2)
    if not sb.get("visible"):
        score -= 1

    # v15: F1/F3 use pre-goto state where near is naturally empty; don't penalise
    if not sb.get("near") and ft not in ("F1", "F3"):
        score -= 1

    attempted = sample.get("attempted_action", "")
    if attempted and attempted.startswith("Pick("):
        arg = attempted[len("Pick("):-1]
        if arg in ALL_RECEPTACLES and arg not in PICKABLE_OBJECTS:
            score -= 2

    return score


# ======================================================================
# Failure injection
# ======================================================================

def inject_failure(actions, step_idx, ft, traj_objects, state_before):
    step = actions[step_idx]
    action_name = step["action_name"]
    obj, recep, args = step["obj"], step["recep"], step["args"]
    gold_closed = alfred_to_closed(action_name, args)

    # v14/v15: skip steps whose action has no clean mapping
    if gold_closed is None:
        return None

    holding = state_before.get("holding", [])
    visible = state_before.get("visible", [])
    near = state_before.get("near", [])
    opened = state_before.get("open", [])

    # ==================================================================
    # F1  —  target not visible / wrong view  →  LookAround()
    # ==================================================================
    if ft == "F1":
        # v15: restrict to GotoLocation and PickupObject for clear semantics
        # (Place/Open have ambiguous "target" definition for F1)
        if action_name not in ("GotoLocation", "PickupObject"):
            return None

        target = obj or (args[0] if args else "")
        if not target:
            return None

        # v14: find the GotoLocation that brings the agent to the target area
        goto_idx = None
        for i in range(step_idx - 1, -1, -1):
            if actions[i]["action_name"] == "GotoLocation" and actions[i]["args"]:
                goto_idx = i
                break

        if goto_idx is None:
            # Without a preceding GotoLocation we cannot reliably create an
            # image where the target is out of view.
            return None

        # Use state BEFORE that GotoLocation so the target area is not reached
        pre_goto_state = build_state_before(actions, goto_idx)

        # Verify the target is genuinely absent from the earlier state
        if target in pre_goto_state.get("visible", []):
            return None
        if target in pre_goto_state.get("near", []):
            return None

        # v15: removed strict "need at least some context" gate here.
        # quality_score handles filtering of empty-state samples.

        return {
            "attempted_action": gold_closed,
            "gold_recovery_action": "LookAround()",
            "diagnosis": f"Target '{target}' not visible in current view",
            "_image_step_idx": goto_idx,          # v14: frame from BEFORE navigation
            "_state_override": pre_goto_state,     # v14: consistent state
            "_after_mode": "same",
            "_subtype": "not_visible",
        }

    # ==================================================================
    # F2  —  wrong object picked
    # ==================================================================
    if ft == "F2":
        if action_name != "PickupObject":
            return None
        if not obj:
            return None

        # v14: limit wrong-object candidates to the same area (likely in frame)
        nearby_objs = get_objects_near_step(actions, step_idx)
        visible_pickables = [
            o for o in nearby_objs
            if o != obj and o not in holding
        ]
        if not visible_pickables:
            return None

        wrong_obj = random.choice(visible_pickables)

        patched_state = deep_copy_state(state_before)
        if obj not in patched_state["visible"]:
            patched_state["visible"].append(obj)
        if wrong_obj not in patched_state["visible"]:
            patched_state["visible"].append(wrong_obj)

        if not patched_state["near"]:
            for i in range(step_idx - 1, -1, -1):
                if actions[i]["action_name"] == "GotoLocation" and actions[i]["args"]:
                    loc = actions[i]["args"][0]
                    patched_state["near"] = [loc]
                    break

        return {
            "attempted_action": f"Pick({wrong_obj})",
            "gold_recovery_action": gold_closed,
            "diagnosis": f"Wrong object: picked '{wrong_obj}' instead of '{obj}'",
            "_state_override": patched_state,
            "_after_mode": "same",              # v14: same frame — no pick result shown
            "_subtype": "wrong_pick",
        }

    # ==================================================================
    # F3  —  precondition not met
    # ==================================================================
    if ft == "F3":
        # F3A: trying to pick before navigating close enough
        if action_name == "PickupObject" and obj:
            goto_idx, goto_dest = None, None
            for i in range(step_idx - 1, -1, -1):
                if actions[i]["action_name"] == "GotoLocation" and actions[i]["args"]:
                    goto_idx = i
                    goto_dest = actions[i]["args"][0]
                    break

            if goto_idx is not None:
                pre_goto_state = build_state_before(actions, goto_idx)
                if (
                    goto_dest not in pre_goto_state.get("near", [])
                    and obj not in pre_goto_state.get("holding", [])
                ):
                    # v15: removed strict "need context" gate;
                    # quality_score handles empty-state filtering
                    return {
                        "attempted_action": gold_closed,
                        "gold_recovery_action": f"Navigate({goto_dest})",
                        "diagnosis": f"Precondition: navigate to '{goto_dest}' before picking up '{obj}'",
                        "_image_step_idx": goto_idx,
                        "_state_override": pre_goto_state,
                        "_after_mode": "same",
                        "_subtype": "pick_then_navigate",
                    }

        # F3B: trying to place without holding the object
        # NOTE: the recovery Pick(obj) assumes the object is reachable from
        # the current location.  In multi-step reality the agent might need
        # Navigate first, but for short-horizon recovery this is acceptable.
        if action_name == "PutObject" and obj:
            patched_state = deep_copy_state(state_before)

            if obj in patched_state.get("holding", []):
                patched_state["holding"].remove(obj)

            if obj not in patched_state.get("visible", []):
                patched_state["visible"].append(obj)

            if patched_state.get("visible") or patched_state.get("near") or patched_state.get("in"):
                return {
                    "attempted_action": gold_closed,
                    "gold_recovery_action": f"Pick({obj})",
                    "diagnosis": f"Precondition: pick up '{obj}' before placing",
                    "_state_override": patched_state,
                    "_after_mode": "same",
                    "_subtype": "place_then_pick",
                }

        # F3C: trying to open before navigating close enough
        if action_name == "OpenObject" and obj:
            patched_state = deep_copy_state(state_before)

            if obj in patched_state.get("near", []):
                patched_state["near"].remove(obj)

            if obj not in patched_state.get("visible", []):
                patched_state["visible"].append(obj)

            if obj not in patched_state.get("open", []):
                # v15: removed strict "need context" gate
                return {
                    "attempted_action": gold_closed,
                    "gold_recovery_action": f"Navigate({obj})",
                    "diagnosis": f"Precondition: navigate close to '{obj}' before opening it",
                    "_state_override": patched_state,
                    "_after_mode": "same",
                    "_subtype": "open_then_navigate",
                }

        return None

    # ==================================================================
    # F4  —  no visible effect → retry
    # ==================================================================
    if ft == "F4":
        name_map = {
            "PickupObject": "Pick",
            "PutObject": "Place",
            "OpenObject": "Open",
            "CloseObject": "Close",
        }
        if action_name not in name_map:
            return None

        if action_name == "PickupObject" and obj:
            has_recent_goto = any(
                actions[i]["action_name"] == "GotoLocation"
                for i in range(max(0, step_idx - 2), step_idx)
            )
            if not has_recent_goto:
                return None

        if action_name == "PutObject" and obj and obj not in holding:
            return None

        if action_name == "PutObject" and recep:
            if recep in OPENABLE and recep not in opened:
                return None

        if action_name == "OpenObject" and obj:
            if obj not in near and obj not in visible:
                return None
            if obj in opened:
                return None

        if action_name == "CloseObject" and obj and obj not in opened:
            return None

        # v14: always same frame — "no visible effect" means nothing changed
        return {
            "attempted_action": gold_closed,
            "gold_recovery_action": f"Retry({name_map[action_name]})",
            "diagnosis": f"'{gold_closed}' had no visible effect; retry needed",
            "_after_mode": "same",
            "_subtype": f"retry_{name_map[action_name].lower()}",
        }

    # ==================================================================
    # F5  —  wrong location
    # ==================================================================
    if ft == "F5":
        if action_name != "GotoLocation":
            return None
        if not args:
            return None
        if args[0] in near:
            return None

        # v14: find a specific other GotoLocation step whose destination
        #      becomes wrong_loc.  The after image will come from THAT step,
        #      so text, location name, and image all agree.
        other_gotos = [
            i for i, a in enumerate(actions)
            if a["action_name"] == "GotoLocation"
            and i != step_idx
            and a["args"]
            and a["args"][0] != args[0]   # different destination
        ]
        if not other_gotos:
            return None

        random.shuffle(other_gotos)
        chosen_goto_idx = other_gotos[0]
        wrong_loc = actions[chosen_goto_idx]["args"][0]

        return {
            "attempted_action": f"Navigate({wrong_loc})",
            "gold_recovery_action": gold_closed,
            "diagnosis": f"Navigated to '{wrong_loc}' instead of '{args[0]}'",
            "_after_mode": "different_goto",
            "_after_goto_idx": chosen_goto_idx,   # v14: specific step for image
            "_subtype": "wrong_location",
        }

    return None


# ======================================================================
# Candidate selection helpers
# ======================================================================

def sort_pool(pool):
    pool.sort(
        key=lambda x: (
            x["quality_score"],
            len(x["state_before"].get("visible", [])),
            len(x["state_before"].get("near", [])),
            0 if x["same_frame"] else 1,
        ),
        reverse=True,
    )
    return pool


def dedup_key(c):
    return (
        c.get("source", ""),
        c.get("step_index", -1),
        c.get("attempted_action", ""),
        c.get("gold_recovery_action", ""),
    )


def select_f3_diverse(pool, k):
    pool = sort_pool(pool)

    by_subtype = defaultdict(list)
    for c in pool:
        by_subtype[c.get("subtype", "unknown")].append(c)

    preferred_order = [
        "pick_then_navigate",
        "place_then_pick",
        "open_then_navigate",
        "unknown",
    ]

    quota = {
        "pick_then_navigate": 25,
        "place_then_pick": 20,
        "open_then_navigate": 15,
    }

    selected = []
    used = set()

    def try_add(c):
        key = dedup_key(c)
        if key in used:
            return False
        used.add(key)
        selected.append(c)
        return True

    # Phase 1: fill quotas
    for sub in preferred_order:
        bucket = sort_pool(by_subtype.get(sub, []))
        want = quota.get(sub, 0)
        for c in bucket:
            if len(selected) >= k or want <= 0:
                break
            if try_add(c):
                want -= 1

    # Phase 2: round-robin remaining
    if len(selected) < k:
        idx = {sub: 0 for sub in preferred_order}
        exhausted = False
        while len(selected) < k and not exhausted:
            exhausted = True
            for sub in preferred_order:
                bucket = by_subtype.get(sub, [])
                while idx[sub] < len(bucket):
                    c = bucket[idx[sub]]
                    idx[sub] += 1
                    exhausted = False
                    if try_add(c):
                        break
                if len(selected) >= k:
                    break

    # Phase 3: fallback
    if len(selected) < k:
        for c in pool:
            if len(selected) >= k:
                break
            try_add(c)

    return selected[:k]


def select_candidates(ft, pool, k):
    if ft == "F3":
        return select_f3_diverse(pool, k)
    # v14: all other types use simple quality-sort top-k
    return sort_pool(pool)[:k]


# ======================================================================
# Main
# ======================================================================

def main():
    OUT_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)

    if not ALFRED_ROOT.is_dir():
        print(f"❌ ALFRED_ROOT not found: {ALFRED_ROOT}")
        sys.exit(1)

    print(f"📂 Scanning {ALFRED_ROOT} ...")
    trials = find_trials(ALFRED_ROOT)
    print(f"   Found {len(trials)} trials")
    if not trials:
        print("❌ No trials found!")
        return

    has_images = any((td / "raw_images").is_dir() for _, td in trials[:50])
    print("✅ raw_images present" if has_images else "⚠️  raw_images not found in first 50 trials")

    random.shuffle(trials)

    from src.parser import validate_action

    candidates_by_type = defaultdict(list)
    skipped = defaultdict(int)
    inject_attempts = defaultdict(int)        # v15: track injection attempts

    for split_name, trial_dir in trials:
        if all(len(candidates_by_type[ft]) >= CANDIDATES_PER_TYPE[ft] for ft in FAILURE_TYPES):
            break

        try:
            traj = load_traj(trial_dir)
        except Exception:
            continue

        actions = get_high_actions(traj)
        if len(actions) < 3:
            continue

        task_desc = (
            get_task_desc(traj)
            or trial_dir.parent.name.replace("-", " ").replace("_", " ")
        )
        traj_objects = {a["obj"] for a in actions if a["obj"]}
        by_high = build_frame_index(traj)
        trial_rel = str(trial_dir.relative_to(ALFRED_ROOT))

        for ft in FAILURE_TYPES:
            if len(candidates_by_type[ft]) >= CANDIDATES_PER_TYPE[ft]:
                continue

            step_candidates = list(range(1, max(2, len(actions) - 1)))
            random.shuffle(step_candidates)

            for step_idx in step_candidates:
                if len(candidates_by_type[ft]) >= CANDIDATES_PER_TYPE[ft]:
                    break

                state_before = build_state_before(actions, step_idx)

                inject_attempts[ft] += 1                      # v15
                result = inject_failure(actions, step_idx, ft, traj_objects, state_before)
                if result is None:
                    skipped[f"{ft}_inject_none"] += 1         # v15: finer tracking
                    continue

                # ---- Extract private metadata from result ----
                image_step = result.pop("_image_step_idx", step_idx)
                state_override = result.pop("_state_override", None)
                after_mode = result.pop("_after_mode", "same")
                subtype = result.pop("_subtype", None)
                after_goto_idx = result.pop("_after_goto_idx", None)
                effective_state = state_override if state_override is not None else state_before

                # ---- Vocabulary ----
                vocab = build_candidate_vocab(actions)
                ensure_vocab_coverage(vocab, result["gold_recovery_action"])
                ensure_vocab_coverage(vocab, result["attempted_action"])

                ok, _ = validate_action(result["gold_recovery_action"], vocab)
                if not ok:
                    skipped[f"{ft}_invalid_gold"] += 1
                    continue

                goal_state = build_goal_state(result["gold_recovery_action"])

                # ---- Image resolution ----
                before_frame = get_frame_before_step(by_high, image_step)
                before_src = resolve_image_file(trial_dir, before_frame)

                after_frame = None
                after_src = None
                same_frame = True

                if after_mode == "different_goto":
                    # v14: use the SPECIFIC goto step whose destination == wrong_loc
                    if after_goto_idx is not None:
                        af = get_frame_for_goto_step(by_high, after_goto_idx)
                        af_src = resolve_image_file(trial_dir, af)
                        if af_src and af != before_frame:
                            after_frame = af
                            after_src = af_src
                            same_frame = False

                elif after_mode == "post_step":
                    af = get_frame_after_step(by_high, step_idx)
                    af_src = resolve_image_file(trial_dir, af)
                    if af_src and af != before_frame:
                        after_frame = af
                        after_src = af_src
                        same_frame = False

                # after_mode == "same": keep defaults (after = before)

                if after_src is None:
                    after_frame = before_frame
                    after_src = before_src
                    same_frame = True

                # ---- v14: consistency gate ----
                # F5 MUST have different before/after (agent is at wrong place)
                if ft == "F5" and same_frame:
                    skipped[f"{ft}_same_frame"] += 1
                    continue

                # ---- Build candidate record ----
                candidate = {
                    "split": split_name,
                    "trial_rel": trial_rel,
                    "instruction": task_desc,
                    "attempted_action": result["attempted_action"],
                    "gold_recovery_action": result["gold_recovery_action"],
                    "failure_type": ft,
                    "candidate_vocab": vocab,
                    "state_before": effective_state,
                    "goal_state": goal_state,
                    "task_type": trial_dir.parent.name.split("-")[0],
                    "diagnosis": result["diagnosis"],
                    "source": trial_rel,
                    "step_index": step_idx,
                    "source_type": "alfred_injected",
                    "before_src": str(before_src) if before_src else "",
                    "after_src": str(after_src) if after_src else "",
                    "before_frame": before_frame,
                    "after_frame": after_frame,
                    "image_step_idx": image_step,
                    "same_frame": same_frame,
                    "subtype": subtype,
                }
                candidate["quality_score"] = quality_score(candidate)

                # v15: threshold lowered from 2 to 1
                if candidate["quality_score"] < 1:
                    skipped[f"{ft}_low_quality"] += 1
                    continue

                candidates_by_type[ft].append(candidate)

    # ---- v15: Report injection conversion rates ----
    print("\n📊 Injection conversion rates:")
    for ft in FAILURE_TYPES:
        attempts = inject_attempts[ft]
        successes = len(candidates_by_type[ft])
        rate = successes / attempts * 100 if attempts > 0 else 0
        print(f"   {ft}: {successes}/{attempts} ({rate:.1f}%)")

    # ---- Report candidate pool stats ----
    print("\n🧪 Candidate subtype pool sizes:")
    for ft in FAILURE_TYPES:
        sub_counter = defaultdict(int)
        for c in candidates_by_type[ft]:
            sub_counter[c.get("subtype", "unknown")] += 1
        print(f"   {ft}: {dict(sorted(sub_counter.items()))}")

    # ---- Select top-K per type ----
    selected = []
    for ft in FAILURE_TYPES:
        pool = candidates_by_type[ft]
        chosen = select_candidates(ft, pool, FINAL_PER_TYPE)
        selected.extend(chosen)

    selected.sort(key=lambda x: (FAILURE_TYPES.index(x["failure_type"]), -x["quality_score"]))

    # ---- Write output ----
    samples = []
    manifest = []
    visual_stats = defaultdict(int)
    subtype_stats = defaultdict(int)

    # Clean old images
    if IMG_DIR.exists():
        for p in IMG_DIR.iterdir():
            if re.match(r"^\d{4}_(before|after)\.(jpg|jpeg|png)$", p.name, flags=re.IGNORECASE):
                try:
                    p.unlink()
                except Exception:
                    pass

    sid = 0
    counts = defaultdict(int)

    for c in selected:
        sample_id = f"S{sid:04d}"
        before_rel, after_rel = "", ""

        if c["before_src"]:
            ext_b = pathlib.Path(c["before_src"]).suffix
            b_dst = IMG_DIR / f"{sid:04d}_before{ext_b}"
            shutil.copy2(c["before_src"], b_dst)
            before_rel = f"data/images/{sid:04d}_before{ext_b}"

        if c["after_src"]:
            ext_a = pathlib.Path(c["after_src"]).suffix
            a_dst = IMG_DIR / f"{sid:04d}_after{ext_a}"
            shutil.copy2(c["after_src"], a_dst)
            after_rel = f"data/images/{sid:04d}_after{ext_a}"

        sample = {
            "sample_id": sample_id,
            "instruction": c["instruction"],
            "attempted_action": c["attempted_action"],
            "before_image": before_rel,
            "after_image": after_rel,
            "source_type": c["source_type"],
            "failure_type": c["failure_type"],
            "gold_recovery_action": c["gold_recovery_action"],
            "candidate_vocab": c["candidate_vocab"],
            "state_before": c["state_before"],
            "goal_state": c["goal_state"],
            "task_type": c["task_type"],
            "diagnosis": c["diagnosis"],
            "source": c["source"],
            "step_index": c["step_index"],
        }
        samples.append(sample)

        manifest.append({
            "sample_id": sample_id,
            "split": c["split"],
            "trial": c["trial_rel"],
            "before_image_name": c["before_frame"],
            "after_image_name": c["after_frame"],
            "step_idx": c["step_index"],
            "image_step_idx": c["image_step_idx"],
            "same_frame": c["same_frame"],
            "quality_score": c["quality_score"],
            "subtype": c.get("subtype"),
        })

        if c["same_frame"]:
            visual_stats[f"{c['failure_type']}_same"] += 1
        else:
            visual_stats[f"{c['failure_type']}_diff"] += 1

        if c.get("subtype"):
            subtype_stats[f"{c['failure_type']}::{c['subtype']}"] += 1

        counts[c["failure_type"]] += 1
        sid += 1

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    ids = [s["sample_id"] for s in samples]
    random.shuffle(ids)
    n = len(ids)
    n_train, n_val = int(n * 0.7), int(n * 0.15)
    split_data = {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }

    with open(SPLIT_FILE, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    img_count = sum(1 for s in samples if s.get("before_image"))
    diff_count = sum(v for k, v in visual_stats.items() if k.endswith("_diff"))
    same_count = sum(v for k, v in visual_stats.items() if k.endswith("_same"))

    print(f"\n{'=' * 55}")
    print(f"✅ Generated {len(samples)} samples")
    for ft in FAILURE_TYPES:
        print(f"   {ft}: {counts[ft]}  (from {len(candidates_by_type[ft])} candidates)")
    print(f"   With image pairs: {img_count}/{len(samples)}")

    print(f"\n📸 Visual diversity:")
    print(f"   Different before/after: {diff_count}")
    print(f"   Same before/after:      {same_count}")
    for k in sorted(visual_stats):
        print(f"     {k}: {visual_stats[k]}")

    print("\n🧩 Subtype summary:")
    for k in sorted(subtype_stats):
        print(f"   {k}: {subtype_stats[k]}")

    if skipped:
        print(f"\n⚠️  Skipped: {dict(skipped)}")

    print(f"\n📄 {OUT_FILE}")
    print(f"📄 {SPLIT_FILE}")
    print(f"📄 {MANIFEST_FILE}")
    print(f"📁 {IMG_DIR}/")
    print(
        f"\nSplit: train={len(split_data['train'])} | "
        f"val={len(split_data['val'])} | test={len(split_data['test'])}"
    )
    print("\n🔍 Next: python scripts/sanity_check.py")


if __name__ == "__main__":
    main()