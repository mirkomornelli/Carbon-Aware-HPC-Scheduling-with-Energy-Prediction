from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from project_rebuild.plotting import save_scheduler_plots, save_weekly_scheduler_plots
from project_rebuild.scheduler import available_week_starts, prepare_week_jobs, run_scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run carbon-aware scheduling simulation.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "results" / "carbon_prediction" / "job_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "scheduler")
    parser.add_argument(
        "--capacity-nodes",
        type=int,
        default=15680,
        help="Total schedulable resource units. For PM100 notes: 980 nodes * 16 processors/node = 15680 processors.",
    )
    parser.add_argument("--slot-minutes", type=int, default=30)
    parser.add_argument("--max-delay-hours", type=int, default=24)
    parser.add_argument("--max-jobs", type=int, default=1200, help="Maximum jobs per simulated week.")
    parser.add_argument(
        "--num-weeks",
        type=int,
        default=0,
        help="Number of real weekly windows to simulate. Use 0 to run every available week.",
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "test"],
        default="all",
        help="Subset of prediction rows to schedule. 'all' gives the widest multi-week simulation.",
    )
    parser.add_argument("--min-jobs-per-week", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


POLICY_ORDER = ["fcfs", "random", "carbon_aware"]


def aggregate_metrics(schedules: pd.DataFrame, skipped: pd.DataFrame, weekly_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, part in schedules.groupby("policy"):
        skipped_count = 0 if skipped.empty else int((skipped["policy"] == policy).sum())
        slowdown = (part["wait_h"] + part["duration_h"]) / part["duration_h"].clip(lower=1 / 60)
        weekly_part = weekly_metrics[weekly_metrics["policy"].eq(policy)] if not weekly_metrics.empty else pd.DataFrame()
        max_processors_used = (
            float(weekly_part["max_processors_used"].max())
            if "max_processors_used" in weekly_part.columns and not weekly_part.empty
            else float("nan")
        )
        rows.append(
            {
                "policy": policy,
                "scheduled_jobs": float(len(part)),
                "skipped_jobs": float(skipped_count),
                "total_actual_energy_kwh": float(part["actual_energy_kwh"].sum()),
                "total_actual_emissions_kg": float(part["actual_emissions_kg"].sum()),
                "total_predicted_emissions_kg": float(part["predicted_emissions_kg"].sum()),
                "avg_wait_h": float(part["wait_h"].mean()),
                "median_wait_h": float(part["wait_h"].median()),
                "p95_wait_h": float(part["wait_h"].quantile(0.95)),
                "avg_slowdown": float(slowdown.mean()),
                "max_processors_used": max_processors_used,
            }
        )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics
    if "fcfs" in set(metrics["policy"]):
        base = metrics.loc[metrics["policy"].eq("fcfs")].iloc[0]
        metrics["emissions_delta_vs_fcfs_kg"] = metrics["total_actual_emissions_kg"] - base["total_actual_emissions_kg"]
        metrics["emissions_reduction_vs_fcfs_pct"] = (
            (base["total_actual_emissions_kg"] - metrics["total_actual_emissions_kg"])
            / base["total_actual_emissions_kg"]
            * 100.0
        )
        metrics["avg_wait_delta_vs_fcfs_h"] = metrics["avg_wait_h"] - base["avg_wait_h"]
    metrics["policy"] = pd.Categorical(metrics["policy"], categories=POLICY_ORDER, ordered=True)
    return metrics.sort_values("policy").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    predictions["submit_time"] = pd.to_datetime(predictions["submit_time"], errors="coerce", utc=True)
    if args.split != "all" and "split" in predictions.columns:
        predictions = predictions[predictions["split"].eq(args.split)].copy()

    week_starts = available_week_starts(predictions, min_jobs=args.min_jobs_per_week)
    if args.num_weeks > 0:
        week_starts = week_starts[: args.num_weeks]

    weekly_results = []
    schedules = []
    skipped = []
    carbon_curves = []

    for week_idx, week_start in enumerate(week_starts):
        jobs, selected_week_start = prepare_week_jobs(predictions, week_start=week_start, max_jobs=args.max_jobs)
        if jobs.empty:
            continue

        for policy in ["fcfs", "random", "carbon_aware"]:
            result = run_scheduler(
                jobs=jobs,
                week_start=selected_week_start,
                policy=policy,
                capacity_nodes=args.capacity_nodes,
                slot_minutes=args.slot_minutes,
                max_delay_hours=args.max_delay_hours,
                random_seed=args.seed + week_idx,
            )
            metrics = dict(result.metrics)
            metrics["week_index"] = week_idx
            metrics["week_start"] = selected_week_start.isoformat()
            metrics["input_jobs"] = float(len(jobs))
            weekly_results.append(metrics)

            schedule = result.schedule.copy()
            if not schedule.empty:
                schedule["week_index"] = week_idx
                schedule["week_start"] = selected_week_start.isoformat()
                schedules.append(schedule)

            if not result.skipped.empty:
                tmp = result.skipped.copy()
                tmp["policy"] = policy
                tmp["week_index"] = week_idx
                tmp["week_start"] = selected_week_start.isoformat()
                skipped.append(tmp)

            if policy == "fcfs":
                curve = result.carbon_curve.copy()
                curve["week_index"] = week_idx
                curve["week_start"] = selected_week_start.isoformat()
                carbon_curves.append(curve)

    all_schedules = pd.concat(schedules, ignore_index=True) if schedules else pd.DataFrame()
    all_skipped = pd.concat(skipped, ignore_index=True) if skipped else pd.DataFrame(columns=["job_id", "reason", "policy"])
    all_carbon_curves = pd.concat(carbon_curves, ignore_index=True) if carbon_curves else pd.DataFrame()
    weekly_metrics = pd.DataFrame(weekly_results)
    if not weekly_metrics.empty and "fcfs" in set(weekly_metrics["policy"]):
        fcfs_weekly = weekly_metrics[weekly_metrics["policy"].eq("fcfs")][
            ["week_index", "total_actual_emissions_kg", "avg_wait_h"]
        ].rename(
            columns={
                "total_actual_emissions_kg": "fcfs_emissions_kg",
                "avg_wait_h": "fcfs_avg_wait_h",
            }
        )
        weekly_metrics = weekly_metrics.merge(fcfs_weekly, on="week_index", how="left")
        weekly_metrics["emissions_delta_vs_fcfs_kg"] = (
            weekly_metrics["total_actual_emissions_kg"] - weekly_metrics["fcfs_emissions_kg"]
        )
        weekly_metrics["emissions_reduction_vs_fcfs_pct"] = (
            (weekly_metrics["fcfs_emissions_kg"] - weekly_metrics["total_actual_emissions_kg"])
            / weekly_metrics["fcfs_emissions_kg"]
            * 100.0
        )
        weekly_metrics["avg_wait_delta_vs_fcfs_h"] = weekly_metrics["avg_wait_h"] - weekly_metrics["fcfs_avg_wait_h"]

    metrics = aggregate_metrics(all_schedules, all_skipped, weekly_metrics)

    metrics.to_csv(args.output_dir / "scheduler_metrics.csv", index=False)
    weekly_metrics.to_csv(args.output_dir / "scheduler_weekly_metrics.csv", index=False)
    all_schedules.to_csv(args.output_dir / "schedules.csv", index=False)
    all_skipped.to_csv(args.output_dir / "skipped_jobs.csv", index=False)
    if not all_carbon_curves.empty:
        all_carbon_curves.to_csv(args.output_dir / "synthetic_carbon_curve.csv", index=False)
        first_curve = all_carbon_curves[all_carbon_curves["week_index"].eq(all_carbon_curves["week_index"].min())].copy()
        save_scheduler_plots(metrics, all_schedules, first_curve, args.output_dir)
        save_weekly_scheduler_plots(weekly_metrics, args.output_dir)

    print("Scheduling simulation completed")
    print(f"Weeks simulated: {len(week_starts)}")
    print(f"Rows used: {len(predictions)} ({args.split})")
    print(metrics.to_string(index=False))
    print(f"Results written to: {args.output_dir}")


if __name__ == "__main__":
    main()
