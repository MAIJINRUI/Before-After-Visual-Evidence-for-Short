import json
from collections import defaultdict

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

samples = load_jsonl("data/samples.jsonl")
preds_raw = load_jsonl("outputs/predictions_afteronly_gpt54.jsonl")

predictions = {}
for p in preds_raw:
    sid = p["sample_id"]
    parsed = p.get("parsed", {})
    predictions[sid] = parsed.get("recovery_action", "LookAround()")

correct = 0
total = 0
per_failure = defaultdict(lambda: {"total": 0, "correct": 0})

for s in samples:
    sid = s["sample_id"]
    gold = s["gold_recovery_action"]
    pred = predictions.get(sid, "LookAround()")
    total += 1
    if pred == gold:
        correct += 1
        per_failure[s["failure_type"]]["correct"] += 1
    per_failure[s["failure_type"]]["total"] += 1

print("="*50)
print(f"Model: gpt-5.4 | Mode: after-only")
print(f"Samples: {total}")
print(f"Action Accuracy: {correct/total:.3f} ({correct}/{total})")
print("="*50)
print("\nFailure Breakdown:")
for ft, info in sorted(per_failure.items()):
    acc = info['correct']/info['total']
    print(f"  {ft}: acc={acc:.3f} ({info['correct']}/{info['total']})")

# 对比表
print("\n" + "="*50)
print("COMPARISON: full vs after-only")
print("="*50)
