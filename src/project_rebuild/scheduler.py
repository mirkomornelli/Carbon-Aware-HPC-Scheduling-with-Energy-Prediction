from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


Policy = Literal["fcfs", "carbon_aware", "random"]


@dataclass(frozen=True)
class SchedulerResult:
    schedule: pd.DataFrame
    skipped: pd.DataFrame
    metrics: dict[str, float | str]
    carbon_curve: pd.DataFrame


def synthetic_weekly_carbon_curve(
    week_start: pd.Timestamp,
    slot_minutes: int = 30,
    days: int = 7,
    seed: int = 42,
    slots: int | None = None,
) -> pd.DataFrame:
    if slots is None:
        slots = int(days * 24 * 60 / slot_minutes)
    timestamps = pd.date_range(week_start, periods=slots, freq=f"{slot_minutes}min")
    hour = timestamps.hour
    weekday = timestamps.weekday
    month = timestamps.month

    seasonal = np.select(
        [
            np.isin(month, [12, 1, 2]),
            np.isin(month, [3, 4, 5]),
            np.isin(month, [6, 7, 8]),
        ],
        [250, 200, 360],
        default=280,
    )
    daily = 28 * np.cos((hour - 13) / 24 * 2 * np.pi)
    weekend = np.where(weekday >= 5, -22, 0)
    covid_like = np.where(np.isin(month, [3, 4, 5]), -15, 0)
    noise = np.random.default_rng(seed).normal(0, 7, size=slots)

    curve = np.clip(seasonal + daily + weekend + covid_like + noise, 50, None)
    return pd.DataFrame({"timestamp": timestamps, "gco2_per_kwh": curve})


def normalize_job_times(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions.copy()
    for col in ["submit_time", "start_time", "end_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    df = df.dropna(subset=["submit_time", "start_time", "end_time"]).copy()
    df["run_time"] = pd.to_numeric(df["run_time"], errors="coerce")
    df = df[df["run_time"] > 0].copy()
    return df


def available_week_starts(predictions: pd.DataFrame, min_jobs: int = 1) -> list[pd.Timestamp]:
    df = normalize_job_times(predictions)
    if df.empty:
        return []
    week_start = df["submit_time"].dt.normalize() - pd.to_timedelta(df["submit_time"].dt.weekday, unit="D")
    counts = week_start.value_counts().sort_index()
    return [pd.Timestamp(idx) for idx, count in counts.items() if int(count) >= min_jobs]


def prepare_week_jobs(
    predictions: pd.DataFrame,
    week_start: pd.Timestamp | None = None,
    days: int = 7,
    max_jobs: int = 1200,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    df = normalize_job_times(predictions)

    if week_start is None:
        first = df["submit_time"].min()
        week_start = (first - pd.Timedelta(days=int(first.weekday()))).normalize()
    else:
        week_start = pd.to_datetime(week_start, utc=True).normalize()

    week_end = week_start + pd.Timedelta(days=days)
    week = df[(df["submit_time"] >= week_start) & (df["submit_time"] < week_end)].copy()

    week = week.sort_values("submit_time").head(max_jobs).reset_index(drop=True)
    return week, week_start


def _window_carbon_cost(carbon: np.ndarray, start_slot: int, duration_slots: int, energy_kwh: float) -> float:
    window = carbon[start_slot:start_slot + duration_slots]
    if duration_slots <= 0 or len(window) == 0:
        return float("inf")
    return float(energy_kwh * window.mean() / 1000.0)


def _has_capacity(used_nodes: np.ndarray, start_slot: int, duration_slots: int, req_nodes: int, capacity_nodes: int) -> bool:
    end_slot = start_slot + duration_slots
    if start_slot < 0 or end_slot > len(used_nodes):
        return False
    return bool(np.all(used_nodes[start_slot:end_slot] + req_nodes <= capacity_nodes))


def _find_start(
    used_nodes: np.ndarray,
    carbon: np.ndarray,
    release_slot: int,
    duration_slots: int,
    req_nodes: int,
    capacity_nodes: int,
    energy_kwh: float,
    policy: Policy,
    max_delay_slots: int,
    rng: np.random.Generator,
) -> int | None:
    horizon = len(used_nodes)
    latest = horizon - duration_slots
    if latest < release_slot:
        return None

    if policy == "fcfs":
        for slot in range(release_slot, latest + 1):
            if _has_capacity(used_nodes, slot, duration_slots, req_nodes, capacity_nodes):
                return slot
        return None

    if policy == "random":
        search_end = min(latest, release_slot + max_delay_slots)
        feasible = [
            slot
            for slot in range(release_slot, search_end + 1)
            if _has_capacity(used_nodes, slot, duration_slots, req_nodes, capacity_nodes)
        ]
        if feasible:
            return int(rng.choice(feasible))
        for slot in range(search_end + 1, latest + 1):
            if _has_capacity(used_nodes, slot, duration_slots, req_nodes, capacity_nodes):
                return slot
        return None

    search_end = min(latest, release_slot + max_delay_slots)
    best_slot = None
    best_cost = float("inf")
    for slot in range(release_slot, search_end + 1):
        if not _has_capacity(used_nodes, slot, duration_slots, req_nodes, capacity_nodes):
            continue
        cost = _window_carbon_cost(carbon, slot, duration_slots, energy_kwh)
        if cost < best_cost:
            best_cost = cost
            best_slot = slot

    if best_slot is not None:
        return best_slot

    for slot in range(search_end + 1, latest + 1):
        if _has_capacity(used_nodes, slot, duration_slots, req_nodes, capacity_nodes):
            return slot
    return None


def run_scheduler(
    jobs: pd.DataFrame,
    week_start: pd.Timestamp,
    policy: Policy,
    capacity_nodes: int = 15680,
    slot_minutes: int = 30,
    days: int = 7,
    max_delay_hours: int = 24,
    random_seed: int = 42,
) -> SchedulerResult:
    jobs = jobs.copy()
    week_start = pd.to_datetime(week_start, utc=True)
    max_delay_slots = int(max_delay_hours * 60 / slot_minutes)
    rng = np.random.default_rng(random_seed)

    # In the PM100 trace used here, num_nodes_req is interpreted as requested processors.
    jobs["req_nodes"] = pd.to_numeric(jobs["num_nodes_req"], errors="coerce").fillna(1).clip(lower=1).astype(int)
    jobs["duration_slots"] = np.ceil(jobs["run_time"] / (slot_minutes * 60)).astype(int).clip(lower=1)
    jobs["pred_energy_kwh"] = pd.to_numeric(jobs["pred_energy_kwh"], errors="coerce").fillna(0).clip(lower=0)
    jobs["energy_kwh"] = pd.to_numeric(jobs["energy_kwh"], errors="coerce").fillna(0).clip(lower=0)
    jobs["release_slot"] = np.floor(
        (pd.to_datetime(jobs["submit_time"], utc=True) - week_start) / pd.Timedelta(minutes=slot_minutes)
    ).astype(int).clip(lower=0)

    base_slots = int(days * 24 * 60 / slot_minutes)
    max_release_slot = int(jobs["release_slot"].max()) if not jobs.empty else 0
    max_duration_slots = int(jobs["duration_slots"].max()) if not jobs.empty else 1
    slots = max(base_slots, max_release_slot + max_delay_slots + max_duration_slots + 1)
    carbon_curve = synthetic_weekly_carbon_curve(
        week_start,
        slot_minutes=slot_minutes,
        days=days,
        seed=random_seed,
        slots=slots,
    )
    carbon = carbon_curve["gco2_per_kwh"].to_numpy()
    used_nodes = np.zeros(slots, dtype=int)

    if policy == "random":
        jobs = jobs.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    elif policy == "carbon_aware":
        jobs = jobs.sort_values(["submit_time", "pred_energy_kwh"], ascending=[True, False]).reset_index(drop=True)
    else:
        jobs = jobs.sort_values("submit_time").reset_index(drop=True)

    scheduled = []
    skipped = []

    for _, row in jobs.iterrows():
        job_id = int(row["job_id"])
        req_nodes = int(row["req_nodes"])
        duration_slots = int(row["duration_slots"])
        release = pd.to_datetime(row["submit_time"], utc=True)
        release_slot = int(row["release_slot"])

        if req_nodes > capacity_nodes:
            skipped.append({"job_id": job_id, "reason": "too_many_nodes"})
            continue

        start_slot = _find_start(
            used_nodes=used_nodes,
            carbon=carbon,
            release_slot=release_slot,
            duration_slots=duration_slots,
            req_nodes=req_nodes,
            capacity_nodes=capacity_nodes,
            energy_kwh=float(row["pred_energy_kwh"]),
            policy=policy,
            max_delay_slots=max_delay_slots,
            rng=rng,
        )
        if start_slot is None:
            skipped.append({"job_id": job_id, "reason": "horizon_overflow"})
            continue

        end_slot = start_slot + duration_slots
        used_nodes[start_slot:end_slot] += req_nodes
        start = week_start + pd.Timedelta(minutes=start_slot * slot_minutes)
        end = week_start + pd.Timedelta(minutes=end_slot * slot_minutes)
        wait_h = max(0.0, (start - release) / pd.Timedelta(hours=1))

        actual_emissions_kg = _window_carbon_cost(carbon, start_slot, duration_slots, float(row["energy_kwh"]))
        predicted_emissions_kg = _window_carbon_cost(carbon, start_slot, duration_slots, float(row["pred_energy_kwh"]))

        scheduled.append(
            {
                "job_id": job_id,
                "policy": policy,
                "submit_time": release,
                "scheduled_start": start,
                "scheduled_end": end,
                "req_nodes": req_nodes,
                "duration_h": duration_slots * slot_minutes / 60,
                "wait_h": wait_h,
                "actual_energy_kwh": float(row["energy_kwh"]),
                "pred_energy_kwh": float(row["pred_energy_kwh"]),
                "actual_emissions_kg": actual_emissions_kg,
                "predicted_emissions_kg": predicted_emissions_kg,
                "avg_carbon_gco2_kwh": float(carbon[start_slot:end_slot].mean()),
            }
        )

    schedule = pd.DataFrame(scheduled)
    skipped_df = pd.DataFrame(skipped)

    if schedule.empty:
        metrics = {
            "policy": policy,
            "scheduled_jobs": 0.0,
            "skipped_jobs": float(len(skipped_df)),
            "total_actual_energy_kwh": 0.0,
            "total_actual_emissions_kg": 0.0,
            "avg_wait_h": float("nan"),
            "median_wait_h": float("nan"),
            "p95_wait_h": float("nan"),
            "avg_slowdown": float("nan"),
            "max_nodes_used": float(used_nodes.max()),
            "max_processors_used": float(used_nodes.max()),
        }
    else:
        slowdown = (schedule["wait_h"] + schedule["duration_h"]) / schedule["duration_h"].clip(lower=1 / 60)
        metrics = {
            "policy": policy,
            "scheduled_jobs": float(len(schedule)),
            "skipped_jobs": float(len(skipped_df)),
            "total_actual_energy_kwh": float(schedule["actual_energy_kwh"].sum()),
            "total_actual_emissions_kg": float(schedule["actual_emissions_kg"].sum()),
            "total_predicted_emissions_kg": float(schedule["predicted_emissions_kg"].sum()),
            "avg_wait_h": float(schedule["wait_h"].mean()),
            "median_wait_h": float(schedule["wait_h"].median()),
            "p95_wait_h": float(schedule["wait_h"].quantile(0.95)),
            "avg_slowdown": float(slowdown.mean()),
            "max_nodes_used": float(used_nodes.max()),
            "max_processors_used": float(used_nodes.max()),
        }

    return SchedulerResult(
        schedule=schedule,
        skipped=skipped_df,
        metrics=metrics,
        carbon_curve=carbon_curve,
    )
