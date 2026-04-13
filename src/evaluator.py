import json
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from .executor import run_semi_loop
from .parser import parse_prediction


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_split(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    f1_values = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        if tp == 0 and fp == 0 and fn == 0:
            f1_values.append(1.0)
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1_values.append(0.0)
        else:
            f1_values.append(2 * precision * recall / (precision + recall))
    return sum(f1_values) / len(f1_values) if f1_values else 0.0


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def bootstrap_mean_ci95(values: List[float], n_boot: int = 1000, seed: int = 42) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "low": None, "high": None, "n_boot": n_boot}
    rng = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    return {
        "mean": sum(values) / n,
        "low": _percentile(boots, 0.025),
        "high": _percentile(boots, 0.975),
        "n_boot": n_boot,
    }


def bootstrap_macro_f1_ci95(y_true: List[str], y_pred: List[str], n_boot: int = 1000, seed: int = 42) -> Dict[str, Optional[float]]:
    if not y_true or len(y_true) != len(y_pred):
        return {"mean": None, "low": None, "high": None, "n_boot": n_boot}
    rng = random.Random(seed)
    n = len(y_true)
    idxs = list(range(n))
    boots = []
    for _ in range(n_boot):
        sampled = [idxs[rng.randrange(n)] for _ in range(n)]
        yt = [y_true[i] for i in sampled]
        yp = [y_pred[i] for i in sampled]
        boots.append(macro_f1(yt, yp))
    return {
        "mean": macro_f1(y_true, y_pred),
        "low": _percentile(boots, 0.025),
        "high": _percentile(boots, 0.975),
        "n_boot": n_boot,
    }


def evaluate_with_details(samples: List[Dict], predictions: Dict[str, str], max_steps: int = 1) -> Tuple[Dict, Dict]:
    y_true = []
    y_pred = []
    valid_json_count = 0
    valid_action_count_repaired = 0
    valid_action_count_raw = 0
    recovered = 0
    task_completed = 0
    executable_last = 0
    steps_to_recover = []
    per_failure = defaultdict(lambda: {"total": 0, "correct": 0})
    repair_applied_count = 0

    action_correct_flags: List[float] = []
    raw_valid_action_flags: List[float] = []
    repaired_valid_action_flags: List[float] = []
    recovered_flags: List[float] = []
    completion_flags: List[float] = []
    executable_flags: List[float] = []

    for sample in samples:
        sample_id = sample["sample_id"]
        pred_text = predictions.get(sample_id, '{"recovery_action":"LookAround()"}')
        parse_raw = parse_prediction(pred_text, sample["candidate_vocab"], enable_repair=False)
        parse_repaired = parse_prediction(pred_text, sample["candidate_vocab"], enable_repair=True)

        if parse_raw.valid_json:
            valid_json_count += 1
        if parse_raw.valid_action and parse_raw.recovery_action:
            valid_action_count_raw += 1
            raw_valid_action_flags.append(1.0)
        else:
            raw_valid_action_flags.append(0.0)

        if parse_repaired.valid_action and parse_repaired.recovery_action:
            valid_action_count_repaired += 1
            repaired_valid_action_flags.append(1.0)
            pred_action = parse_repaired.recovery_action
        else:
            repaired_valid_action_flags.append(0.0)
            pred_action = "LookAround()"

        if (
            parse_raw.recovery_action
            and parse_repaired.recovery_action
            and parse_raw.recovery_action != parse_repaired.recovery_action
        ):
            repair_applied_count += 1

        gold_action = sample["gold_recovery_action"]
        y_true.append(gold_action)
        y_pred.append(pred_action)
        is_correct = float(pred_action == gold_action)
        action_correct_flags.append(is_correct)

        if pred_action == gold_action:
            per_failure[sample["failure_type"]]["correct"] += 1
        per_failure[sample["failure_type"]]["total"] += 1

        loop_result = run_semi_loop(sample, pred_action, max_steps=max_steps)
        if loop_result["recovered"]:
            recovered += 1
            steps_to_recover.append(loop_result["steps_to_recover"])
            recovered_flags.append(1.0)
        else:
            recovered_flags.append(0.0)
        if loop_result["task_completed"]:
            task_completed += 1
            completion_flags.append(1.0)
        else:
            completion_flags.append(0.0)
        if loop_result["last_executable"]:
            executable_last += 1
            executable_flags.append(1.0)
        else:
            executable_flags.append(0.0)

    n = len(samples)
    action_acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n if n else 0.0
    m_f1 = macro_f1(y_true, y_pred)
    format_compliance_raw = valid_action_count_raw / n if n else 0.0
    format_compliance_repaired = valid_action_count_repaired / n if n else 0.0
    json_valid_rate = valid_json_count / n if n else 0.0
    recovery_rate_k = recovered / n if n else 0.0
    completion_rate = task_completed / n if n else 0.0
    executable_rate = executable_last / n if n else 0.0
    avg_steps = sum(steps_to_recover) / len(steps_to_recover) if steps_to_recover else None

    breakdown = {}
    for ftype, stats in per_failure.items():
        total = stats["total"]
        breakdown[ftype] = {"count": total, "action_acc": stats["correct"] / total if total else 0.0}

    report = {
        "num_samples": n,
        "action_accuracy": action_acc,
        "macro_f1": m_f1,
        "format_compliance_rate": format_compliance_repaired,
        "format_compliance_rate_raw": format_compliance_raw,
        "format_compliance_rate_repaired": format_compliance_repaired,
        "json_valid_rate": json_valid_rate,
        "recovery_rate_at_k": recovery_rate_k,
        "semi_loop_task_completion_rate": completion_rate,
        "last_action_executable_rate": executable_rate,
        "repair_applied_count": repair_applied_count,
        "avg_steps_to_recover": avg_steps,
        "failure_breakdown": breakdown,
        "label_counts": dict(Counter(sample["failure_type"] for sample in samples)),
    }
    details = {
        "y_true": y_true,
        "y_pred": y_pred,
        "action_correct_flags": action_correct_flags,
        "raw_valid_action_flags": raw_valid_action_flags,
        "repaired_valid_action_flags": repaired_valid_action_flags,
        "recovered_flags": recovered_flags,
        "completion_flags": completion_flags,
        "executable_flags": executable_flags,
    }
    return report, details


def evaluate(samples: List[Dict], predictions: Dict[str, str], max_steps: int = 1) -> Dict:
    report, _ = evaluate_with_details(samples, predictions, max_steps=max_steps)
    return report


def filter_by_split(samples: List[Dict], split_ids: List[str]) -> List[Dict]:
    idset = set(split_ids)
    return [s for s in samples if s["sample_id"] in idset]
