from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_energy_plots(predictions: pd.DataFrame, feature_importance: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_df = predictions
    if "split" in predictions.columns and predictions["split"].eq("test").any():
        plot_df = predictions[predictions["split"].eq("test")].copy()
    sample = plot_df.sample(min(4000, len(plot_df)), random_state=42)

    plt.figure(figsize=(7, 6))
    plt.scatter(sample["energy_kwh"], sample["pred_energy_kwh"], s=8, alpha=0.25)
    limit = max(sample["energy_kwh"].max(), sample["pred_energy_kwh"].max())
    plt.plot([0, limit], [0, limit], color="#245c3a", linewidth=2)
    plt.xscale("symlog")
    plt.yscale("symlog")
    plt.xlabel("Actual energy (kWh)")
    plt.ylabel("Predicted energy (kWh)")
    plt.title("HPC job energy prediction")
    plt.tight_layout()
    plt.savefig(output_dir / "energy_actual_vs_predicted.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.6))
    ordered = feature_importance.sort_values("importance", ascending=True)
    plt.barh(ordered["feature"], ordered["importance"], color="#2e6f86")
    plt.xlabel("Importance")
    plt.title("Most useful features for energy prediction")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.6))
    plt.hist(plot_df["abs_emissions_error_kg"], bins=60, color="#c8643b", alpha=0.85)
    plt.xlabel("Absolute emissions error (kg CO2)")
    plt.ylabel("Number of jobs")
    plt.title("Distribution of estimated emissions error")
    plt.tight_layout()
    plt.savefig(output_dir / "emissions_error_histogram.png", dpi=160)
    plt.close()


def save_scheduler_plots(metrics: pd.DataFrame, schedules: pd.DataFrame, carbon_curve: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.bar(metrics["policy"], metrics["total_actual_emissions_kg"], color=["#52665a", "#e4b84a", "#245c3a"])
    plt.ylabel("kg CO2")
    plt.title("Total emissions by policy")
    plt.tight_layout()
    plt.savefig(output_dir / "scheduler_emissions_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.bar(metrics["policy"], metrics["avg_wait_h"], color=["#52665a", "#e4b84a", "#245c3a"])
    plt.ylabel("Hours")
    plt.title("Average waiting time by policy")
    plt.tight_layout()
    plt.savefig(output_dir / "scheduler_wait_comparison.png", dpi=160)
    plt.close()

    if "emissions_reduction_vs_fcfs_pct" in metrics.columns:
        reduction = metrics.copy()
        reduction = reduction[reduction["policy"].ne("fcfs")]
        plt.figure(figsize=(8, 4.5))
        plt.bar(reduction["policy"], reduction["emissions_reduction_vs_fcfs_pct"], color=["#e4b84a", "#245c3a"])
        plt.axhline(0, color="#132018", linewidth=1)
        plt.ylabel("Emissions reduction vs FCFS (%)")
        plt.title("Emissions improvement over the baseline")
        plt.tight_layout()
        plt.savefig(output_dir / "scheduler_emissions_reduction_pct.png", dpi=160)
        plt.close()

    resource_col = "max_processors_used" if "max_processors_used" in metrics.columns else "max_nodes_used"
    plt.figure(figsize=(8, 4.5))
    plt.bar(metrics["policy"], metrics[resource_col], color=["#52665a", "#e4b84a", "#245c3a"])
    plt.ylabel("Requested processors")
    plt.title("Maximum resource usage during the simulation")
    plt.tight_layout()
    plt.savefig(output_dir / "scheduler_nodes_usage.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4.6))
    plt.plot(carbon_curve["timestamp"], carbon_curve["gco2_per_kwh"], color="#245c3a")
    plt.ylabel("gCO2/kWh")
    plt.title("Synthetic carbon intensity used by the scheduler")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(output_dir / "synthetic_carbon_curve_week.png", dpi=160)
    plt.close()

    if schedules.empty:
        return

    _save_per_job_savings_plot(schedules, output_dir)
    _save_cumulative_emissions_plot(schedules, output_dir)

    sample = schedules.sort_values("scheduled_start").groupby("policy").head(70)
    policies = list(sample["policy"].drop_duplicates())
    fig, axes = plt.subplots(len(policies), 1, figsize=(11, max(3.2, 2.6 * len(policies))), sharex=True)
    if len(policies) == 1:
        axes = [axes]

    for ax, policy in zip(axes, policies):
        part = sample[sample["policy"].eq(policy)].reset_index(drop=True)
        for idx, row in part.iterrows():
            ax.plot([row["scheduled_start"], row["scheduled_end"]], [idx, idx], linewidth=3.2, color="#2e6f86")
        ax.set_title(f"Compact Gantt chart - {policy}")
        ax.set_ylabel("job")

    axes[-1].set_xlabel("time")
    fig.autofmt_xdate(rotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "scheduler_compact_gantt.png", dpi=160)
    plt.close(fig)


def save_weekly_scheduler_plots(weekly_metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if weekly_metrics.empty:
        return

    pivot = weekly_metrics.pivot_table(
        index="week_start",
        columns="policy",
        values="total_actual_emissions_kg",
        aggfunc="first",
    )
    if pivot.empty:
        return

    plt.figure(figsize=(10, 4.8))
    for policy in pivot.columns:
        plt.plot(pd.to_datetime(pivot.index), pivot[policy], marker="o", label=policy)
    plt.ylabel("kg CO2")
    plt.title("Weekly emissions by policy")
    plt.legend()
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(output_dir / "scheduler_weekly_emissions.png", dpi=160)
    plt.close()

    energy = weekly_metrics[weekly_metrics["policy"].eq("fcfs")].copy()
    if not energy.empty:
        plt.figure(figsize=(10, 4.8))
        plt.bar(pd.to_datetime(energy["week_start"]).astype(str), energy["total_actual_energy_kwh"], color="#2f6f88")
        plt.ylabel("kWh")
        plt.title("Weekly energy consumption of simulated jobs")
        plt.xticks(rotation=35)
        plt.tight_layout()
        plt.savefig(output_dir / "scheduler_weekly_energy.png", dpi=160)
        plt.close()

        plt.figure(figsize=(10, 4.8))
        plt.bar(pd.to_datetime(energy["week_start"]).astype(str), energy["input_jobs"], color="#52665a")
        plt.ylabel("job")
        plt.title("Number of simulated jobs per week")
        plt.xticks(rotation=35)
        plt.tight_layout()
        plt.savefig(output_dir / "scheduler_weekly_jobs.png", dpi=160)
        plt.close()

    wait = weekly_metrics.pivot_table(
        index="week_start",
        columns="policy",
        values="avg_wait_h",
        aggfunc="first",
    )
    if not wait.empty:
        plt.figure(figsize=(10, 4.8))
        for policy in wait.columns:
            plt.plot(pd.to_datetime(wait.index), wait[policy], marker="o", label=policy)
        plt.ylabel("Hours")
        plt.title("Average waiting time per week")
        plt.legend()
        plt.xticks(rotation=35)
        plt.tight_layout()
        plt.savefig(output_dir / "scheduler_weekly_wait.png", dpi=160)
        plt.close()

    carbon = weekly_metrics.copy()
    carbon["avg_carbon_gco2_kwh"] = (
        carbon["total_actual_emissions_kg"] / carbon["total_actual_energy_kwh"].replace(0, pd.NA) * 1000.0
    )
    carbon_pivot = carbon.pivot_table(
        index="week_start",
        columns="policy",
        values="avg_carbon_gco2_kwh",
        aggfunc="first",
    )
    if not carbon_pivot.empty:
        plt.figure(figsize=(10, 4.8))
        for policy in carbon_pivot.columns:
            plt.plot(pd.to_datetime(carbon_pivot.index), carbon_pivot[policy], marker="o", label=policy)
        plt.ylabel("gCO2/kWh")
        plt.title("Effective average carbon intensity achieved by each policy")
        plt.legend()
        plt.xticks(rotation=35)
        plt.tight_layout()
        plt.savefig(output_dir / "scheduler_weekly_avg_carbon.png", dpi=160)
        plt.close()

    if "emissions_reduction_vs_fcfs_pct" in weekly_metrics.columns:
        reduction = weekly_metrics[weekly_metrics["policy"].eq("carbon_aware")].copy()
        reduction = reduction.sort_values("week_start")
        plt.figure(figsize=(10, 4.8))
        plt.bar(
            pd.to_datetime(reduction["week_start"]).astype(str),
            reduction["emissions_reduction_vs_fcfs_pct"],
            color="#245c3a",
        )
        plt.ylabel("Emissions reduction vs FCFS (%)")
        plt.title("Weekly carbon-aware reduction")
        plt.xticks(rotation=35)
        plt.tight_layout()
        plt.savefig(output_dir / "scheduler_weekly_reduction_pct.png", dpi=160)
        plt.close()


def _save_per_job_savings_plot(schedules: pd.DataFrame, output_dir: Path) -> None:
    pivot = schedules.pivot_table(
        index="job_id",
        columns="policy",
        values="actual_emissions_kg",
        aggfunc="first",
    )
    if not {"fcfs", "carbon_aware"}.issubset(pivot.columns):
        return

    savings = pivot["fcfs"] - pivot["carbon_aware"]
    plt.figure(figsize=(8, 4.5))
    plt.hist(savings, bins=55, color="#245c3a", alpha=0.86)
    plt.axvline(0, color="#132018", linewidth=1)
    plt.xlabel("kg CO2 saved per job")
    plt.ylabel("Number of jobs")
    plt.title("Per-job savings: FCFS minus carbon-aware")
    plt.tight_layout()
    plt.savefig(output_dir / "scheduler_per_job_savings_histogram.png", dpi=160)
    plt.close()


def _save_cumulative_emissions_plot(schedules: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(10, 4.8))
    for policy, part in schedules.sort_values("scheduled_end").groupby("policy"):
        cumulative = part["actual_emissions_kg"].cumsum()
        plt.plot(pd.to_datetime(part["scheduled_end"]), cumulative, label=policy)
    plt.ylabel("Cumulative kg CO2")
    plt.title("Cumulative emissions during the simulated weeks")
    plt.legend()
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(output_dir / "scheduler_cumulative_emissions.png", dpi=160)
    plt.close()
