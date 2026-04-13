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
modes = {
    "full": "outputs/predictions_full_gpt54.jsonl",
    "after-only": "outputs/predictions_afteronly_gpt54.jsonl",
    "text-only": "outputs/predictions_textonly_gpt54.jsonl",
}

for mode, path in modes.items():
    preds_raw = load_jsonl(path)
    predictions = {}
    for p in preds_raw:
        parsed = p.get("parsed", {})
        predictions[p["sample_id"]] = parsed.get("recovery_action", "")

    correct = 0
    total = 0
    per_f = defaultdict(lambda: {"t": 0, "c": 0})
    for s in samples:
        sid = s["sample_id"]
        gold = s["gold_recovery_action"]
        pred = predictions.get(sid, "")
        total += 1
        if pred == gold:
            correct += 1
            per_f[s["failure_type"]]["c"] += 1
        per_f[s["failure_type"]]["t"] += 1

    print(f"\n{'='*50}")
    print(f"Mode: {mode} | Acc: {correct/total:.3f} ({correct}/{total})")
    for ft in sorted(per_f):
        info = per_f[ft]
        print(f"  {ft}: {info['c']/info['t']:.3f} ({info['c']}/{info['t']})")

print(f"\n{'='*50}")
print("SUMMARY TABLE")
print(f"{'Mode':<12} {'Overall':>8} {'F1':>8} {'F2':>8} {'F3':>8} {'F4':>8} {'F5':>8}")
