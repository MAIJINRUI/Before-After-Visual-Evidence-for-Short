import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import (  # noqa: E402
    bootstrap_macro_f1_ci95,
    bootstrap_mean_ci95,
    evaluate_with_details,
    load_jsonl,
)

MODES = ["full", "after-only", "swap-test", "text-only"]

# Conservative defaults; override via CLI for your provider/model billing.
DEFAULT_PRICING_PER_1M = {
    "gpt-4o-mini": (0.15, 0.60),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-mode ablations with optional dry-run safety check")
    parser.add_argument("--data", default="data/samples.jsonl")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--output-dir", default="outputs/ablations")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep-ms", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Number of bootstrap resamples for 95% CIs")
    parser.add_argument("--parallel-workers", type=int, default=1, help="Number of modes to run in parallel (1 = sequential)")
    parser.add_argument("--dry-run", action="store_true", help="Run 2 samples per mode + pipeline integrity checks + cost estimate")
    parser.add_argument("--estimate-samples", type=int, default=300, help="Samples per mode for projected total cost")
    parser.add_argument("--input-price-per-1m", type=float, default=None, help="Override input token price (USD per 1M)")
    parser.add_argument("--output-price-per-1m", type=float, default=None, help="Override output token price (USD per 1M)")
    return parser.parse_args()


def _run_predictor(args, mode: str, out_path: Path, limit: int) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.vlm_predictor",
        "--data",
        str(Path(args.data).resolve()),
        "--output",
        str(out_path),
        "--project-root",
        str(Path(args.project_root).resolve()),
        "--model",
        args.model,
        "--mode",
        mode,
        "--start",
        str(args.start),
        "--limit",
        str(limit),
        "--sleep-ms",
        str(args.sleep_ms),
        "--overwrite",
    ]
    if args.base_url:
        cmd.extend(["--base-url", args.base_url])

    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def _load_mode_rows(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _predictions_dict(rows: List[Dict]) -> Dict[str, str]:
    preds = {}
    for row in rows:
        sid = row.get("sample_id")
        if not sid:
            continue
        text = row.get("raw_response_text")
        if isinstance(text, str) and text.strip():
            preds[sid] = text
            continue

        parsed = row.get("parsed") or {}
        action = parsed.get("recovery_action") if isinstance(parsed, dict) else None
        if isinstance(action, str) and action:
            preds[sid] = json.dumps({"recovery_action": action}, ensure_ascii=True)
        else:
            preds[sid] = '{"recovery_action":"LookAround()"}'
    return preds


def _usage_totals(rows: List[Dict]) -> Tuple[int, int, int]:
    prompt_total = 0
    completion_total = 0
    n = 0
    for row in rows:
        usage = row.get("usage") if isinstance(row, dict) else None
        if not isinstance(usage, dict):
            continue
        prompt_total += int(usage.get("prompt_tokens", 0) or 0)
        completion_total += int(usage.get("completion_tokens", 0) or 0)
        n += 1
    return prompt_total, completion_total, n


def _pick_pricing(model: str, in_override: float, out_override: float) -> Tuple[float, float]:
    if in_override is not None and out_override is not None:
        return in_override, out_override
    if model in DEFAULT_PRICING_PER_1M:
        d_in, d_out = DEFAULT_PRICING_PER_1M[model]
        return (
            in_override if in_override is not None else d_in,
            out_override if out_override is not None else d_out,
        )
    return (in_override or 0.0, out_override or 0.0)


def _usd_cost(prompt_tokens: float, completion_tokens: float, input_per_1m: float, output_per_1m: float) -> float:
    return (prompt_tokens / 1_000_000.0) * input_per_1m + (completion_tokens / 1_000_000.0) * output_per_1m


def main():
    args = parse_args()
    data_path = Path(args.data).resolve()
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl(str(data_path))
    if args.start >= len(samples):
        raise ValueError(f"--start {args.start} out of range for dataset size {len(samples)}")

    limit = 2 if args.dry_run else args.limit
    if args.dry_run:
        print("[dry-run] Running 2 samples per mode for safety validation")

    mode_reports = {}
    mode_token_stats = {}

    if args.parallel_workers > 1:
        with ThreadPoolExecutor(max_workers=min(args.parallel_workers, len(MODES))) as ex:
            futures = []
            for mode in MODES:
                out_path = out_dir / f"predictions_{mode}.jsonl"
                futures.append(ex.submit(_run_predictor, args, mode, out_path, limit))
            for fut in futures:
                fut.result()
    else:
        for mode in MODES:
            out_path = out_dir / f"predictions_{mode}.jsonl"
            _run_predictor(args, mode, out_path, limit=limit)

    for mode in MODES:
        out_path = out_dir / f"predictions_{mode}.jsonl"
        rows = _load_mode_rows(out_path)
        if args.dry_run:
            rows = rows[:2]

        preds = _predictions_dict(rows)
        if args.dry_run:
            dry_sample_ids = {row.get("sample_id") for row in rows if row.get("sample_id")}
            eval_samples = [s for s in samples if s.get("sample_id") in dry_sample_ids]
        else:
            if limit > 0:
                eval_samples = samples[args.start : min(len(samples), args.start + limit)]
            else:
                eval_samples = samples[args.start :]

        report, details = evaluate_with_details(eval_samples, preds, max_steps=args.max_steps)
        report["ci95"] = {
            "action_accuracy": bootstrap_mean_ci95(details["action_correct_flags"], n_boot=args.bootstrap_samples),
            "macro_f1": bootstrap_macro_f1_ci95(details["y_true"], details["y_pred"], n_boot=args.bootstrap_samples),
            "format_compliance_rate_raw": bootstrap_mean_ci95(
                details["raw_valid_action_flags"], n_boot=args.bootstrap_samples
            ),
            "format_compliance_rate_repaired": bootstrap_mean_ci95(
                details["repaired_valid_action_flags"], n_boot=args.bootstrap_samples
            ),
            "recovery_rate_at_k": bootstrap_mean_ci95(details["recovered_flags"], n_boot=args.bootstrap_samples),
            "semi_loop_task_completion_rate": bootstrap_mean_ci95(
                details["completion_flags"], n_boot=args.bootstrap_samples
            ),
        }
        mode_reports[mode] = report

        p_total, c_total, n_usage = _usage_totals(rows)
        mode_token_stats[mode] = {
            "prompt_tokens_total": p_total,
            "completion_tokens_total": c_total,
            "num_with_usage": n_usage,
        }

    pricing_in, pricing_out = _pick_pricing(args.model, args.input_price_per_1m, args.output_price_per_1m)

    cost_projection = {}
    total_projected_cost = 0.0
    for mode, stat in mode_token_stats.items():
        n = stat["num_with_usage"]
        if n <= 0:
            avg_p = 0.0
            avg_c = 0.0
        else:
            avg_p = stat["prompt_tokens_total"] / n
            avg_c = stat["completion_tokens_total"] / n

        projected_mode_cost = _usd_cost(
            prompt_tokens=avg_p * args.estimate_samples,
            completion_tokens=avg_c * args.estimate_samples,
            input_per_1m=pricing_in,
            output_per_1m=pricing_out,
        )
        total_projected_cost += projected_mode_cost
        cost_projection[mode] = {
            "avg_prompt_tokens_per_sample": avg_p,
            "avg_completion_tokens_per_sample": avg_c,
            "projected_cost_usd_for_samples": args.estimate_samples,
            "projected_mode_cost_usd": projected_mode_cost,
        }

    summary = {
        "model": args.model,
        "dry_run": args.dry_run,
        "modes": MODES,
        "reports": mode_reports,
        "token_stats": mode_token_stats,
        "pricing_usd_per_1m_tokens": {
            "input": pricing_in,
            "output": pricing_out,
        },
        "cost_projection": {
            "samples_per_mode": args.estimate_samples,
            "per_mode": cost_projection,
            "projected_total_cost_usd_all_modes": total_projected_cost,
        },
    }

    summary_path = out_dir / ("dry_run_summary.json" if args.dry_run else "ablation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
