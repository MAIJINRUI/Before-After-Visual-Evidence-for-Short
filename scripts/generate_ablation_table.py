import argparse
import csv
import json
from pathlib import Path

MODES = ["full", "after-only", "swap-test", "text-only"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ablation table (Markdown + CSV) from run_ablations summary")
    parser.add_argument("--summary", default="outputs/ablations/ablation_summary.json")
    parser.add_argument("--out-dir", default="outputs/ablations")
    return parser.parse_args()


def _pct(x):
    return f"{100.0 * float(x):.2f}%"


def main():
    args = parse_args()
    summary_path = Path(args.summary).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    reports = data.get("reports", {})
    per_mode_cost = data.get("cost_projection", {}).get("per_mode", {})

    rows = []
    for mode in MODES:
        rep = reports.get(mode, {})
        cost = per_mode_cost.get(mode, {})
        rows.append({
            "mode": mode,
            "num_samples": rep.get("num_samples", 0),
            "action_accuracy": float(rep.get("action_accuracy", 0.0)),
            "macro_f1": float(rep.get("macro_f1", 0.0)),
            "format_compliance_rate": float(rep.get("format_compliance_rate", 0.0)),
            "json_valid_rate": float(rep.get("json_valid_rate", 0.0)),
            "recovery_rate_at_k": float(rep.get("recovery_rate_at_k", 0.0)),
            "semi_loop_task_completion_rate": float(rep.get("semi_loop_task_completion_rate", 0.0)),
            "projected_mode_cost_usd": float(cost.get("projected_mode_cost_usd", 0.0)),
        })

    csv_path = out_dir / "ablation_table.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "num_samples",
                "action_accuracy",
                "macro_f1",
                "format_compliance_rate",
                "json_valid_rate",
                "recovery_rate_at_k",
                "semi_loop_task_completion_rate",
                "projected_mode_cost_usd",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Ablation Results",
        "",
        "| Mode | N | Action Acc | Macro-F1 | Format Compliance | JSON Valid | Recovery@K | Completion | Projected Cost (USD) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            "| {mode} | {n} | {acc} | {f1} | {fmt} | {jsonv} | {rec} | {comp} | ${cost:.4f} |".format(
                mode=r["mode"],
                n=r["num_samples"],
                acc=_pct(r["action_accuracy"]),
                f1=f"{r['macro_f1']:.4f}",
                fmt=_pct(r["format_compliance_rate"]),
                jsonv=_pct(r["json_valid_rate"]),
                rec=_pct(r["recovery_rate_at_k"]),
                comp=_pct(r["semi_loop_task_completion_rate"]),
                cost=r["projected_mode_cost_usd"],
            )
        )

    total_cost = float(data.get("cost_projection", {}).get("projected_total_cost_usd_all_modes", 0.0))
    md_lines.extend(["", f"Projected total cost across all modes: ${total_cost:.4f}"])

    md_path = out_dir / "ablation_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
