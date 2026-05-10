from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURE_COLUMNS = [
    "num_nodes_req",
    "num_cores_req",
    "num_gpus_req",
    "mem_req",
    "time_limit",
    "num_tasks",
    "cores_per_task",
    "hour",
    "weekday",
    "month",
]


@dataclass(frozen=True)
class PredictionArtifacts:
    model: RandomForestRegressor
    metrics: dict[str, float]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame


def parse_power_samples(value) -> np.ndarray:
    """Convert a power-consumption cell to a numeric numpy array."""
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value)
    elif isinstance(value, str):
        try:
            arr = np.asarray(ast.literal_eval(value))
        except (SyntaxError, ValueError):
            return np.array([], dtype=float)
    else:
        return np.array([], dtype=float)

    arr = pd.to_numeric(pd.Series(arr), errors="coerce").dropna().to_numpy(dtype=float)
    return arr


def mean_or_nan(values: Iterable[float]) -> float:
    arr = parse_power_samples(values)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def synthetic_carbon_intensity(timestamps: pd.Series, seed: int = 42) -> pd.Series:
    """Return synthetic carbon intensity in gCO2/kWh for each timestamp."""
    ts = pd.to_datetime(timestamps, errors="coerce")
    hour = ts.dt.hour.fillna(0)
    weekday = ts.dt.weekday.fillna(0)
    month = ts.dt.month.fillna(1)

    seasonal = np.select(
        [
            month.isin([12, 1, 2]),
            month.isin([3, 4, 5]),
            month.isin([6, 7, 8]),
        ],
        [250, 200, 360],
        default=280,
    )
    daily = 28 * np.cos((hour - 13) / 24 * 2 * np.pi)
    weekend = np.where(weekday >= 5, -22, 0)
    covid_like = np.where(month.isin([3, 4, 5]), -15, 0)

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 9, size=len(ts))
    return pd.Series(np.clip(seasonal + daily + weekend + covid_like + noise, 50, None), index=timestamps.index)


def load_and_prepare_jobs(input_path: Path, max_rows: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Load the HPC trace and compute energy and synthetic emissions targets."""
    df = pd.read_parquet(input_path).copy()
    df = df[df["job_state"].eq("COMPLETED")].copy()
    df["run_time"] = pd.to_numeric(df["run_time"], errors="coerce")
    df = df[df["run_time"] > 0].copy()
    df = df.dropna(subset=["start_time", "submit_time", "end_time"]).copy()
    df = df.sort_values("start_time").reset_index(drop=True)

    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows).copy()

    for col in ["node_power_consumption", "cpu_power_consumption", "mem_power_consumption"]:
        df[f"{col}_mean_w"] = df[col].apply(mean_or_nan)

    df["avg_power_w"] = (
        df["node_power_consumption_mean_w"]
        + df["cpu_power_consumption_mean_w"]
        + df["mem_power_consumption_mean_w"]
    )
    df = df.dropna(subset=["avg_power_w"]).copy()
    df = df[df["avg_power_w"] > 0].copy()

    df["energy_kwh"] = df["avg_power_w"] * df["run_time"] / (3600.0 * 1000.0)
    df["gco2_per_kwh_at_start"] = synthetic_carbon_intensity(df["start_time"], seed=seed)
    df["emissions_kg"] = df["energy_kwh"] * df["gco2_per_kwh_at_start"] / 1000.0

    df["hour"] = df["submit_time"].dt.hour
    df["weekday"] = df["submit_time"].dt.weekday
    df["month"] = df["submit_time"].dt.month
    df["num_tasks"] = pd.to_numeric(df.get("num_tasks", 0), errors="coerce").fillna(0)
    df["cores_per_task"] = pd.to_numeric(df.get("cores_per_task", 0), errors="coerce").fillna(0)

    numeric_cols = [
        "num_nodes_req",
        "num_cores_req",
        "num_gpus_req",
        "mem_req",
        "time_limit",
        "num_tasks",
        "cores_per_task",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df.reset_index(drop=True)


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("start_time").reset_index(drop=True)
    cut = int(len(ordered) * train_ratio)
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def train_energy_model(df: pd.DataFrame, random_state: int = 42) -> PredictionArtifacts:
    train_df, test_df = chronological_split(df)
    x_train = train_df[FEATURE_COLUMNS]
    x_test = test_df[FEATURE_COLUMNS]
    y_train_log = np.log1p(train_df["energy_kwh"])

    model = RandomForestRegressor(
        n_estimators=90,
        max_depth=18,
        min_samples_leaf=4,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(x_train, y_train_log)

    test_pred_energy = np.expm1(model.predict(x_test))
    test_pred_energy = np.clip(test_pred_energy, 0, None)
    actual_energy = test_df["energy_kwh"].to_numpy()

    test_pred_emissions = test_pred_energy * test_df["gco2_per_kwh_at_start"].to_numpy() / 1000.0
    actual_emissions = test_df["emissions_kg"].to_numpy()

    metrics = {
        "rows_total": float(len(df)),
        "rows_train": float(len(train_df)),
        "rows_test": float(len(test_df)),
        "energy_mae_kwh": float(mean_absolute_error(actual_energy, test_pred_energy)),
        "energy_rmse_kwh": float(mean_squared_error(actual_energy, test_pred_energy) ** 0.5),
        "energy_r2": float(r2_score(actual_energy, test_pred_energy)),
        "emissions_mae_kg": float(mean_absolute_error(actual_emissions, test_pred_emissions)),
        "emissions_rmse_kg": float(mean_squared_error(actual_emissions, test_pred_emissions) ** 0.5),
        "emissions_r2": float(r2_score(actual_emissions, test_pred_emissions)),
    }

    predictions = df[
        [
            "job_id",
            "submit_time",
            "start_time",
            "end_time",
            "run_time",
            "num_nodes_req",
            "num_cores_req",
            "num_gpus_req",
            "mem_req",
            "time_limit",
            "energy_kwh",
            "emissions_kg",
            "gco2_per_kwh_at_start",
        ]
    ].copy()
    train_ids = set(train_df["job_id"])
    predictions["split"] = np.where(predictions["job_id"].isin(train_ids), "train", "test")
    predictions["pred_energy_kwh"] = np.expm1(model.predict(df[FEATURE_COLUMNS]))
    predictions["pred_energy_kwh"] = predictions["pred_energy_kwh"].clip(lower=0)
    predictions["pred_emissions_kg"] = (
        predictions["pred_energy_kwh"] * predictions["gco2_per_kwh_at_start"] / 1000.0
    )
    predictions["abs_energy_error_kwh"] = np.abs(predictions["energy_kwh"] - predictions["pred_energy_kwh"])
    predictions["abs_emissions_error_kg"] = np.abs(predictions["emissions_kg"] - predictions["pred_emissions_kg"])

    feature_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return PredictionArtifacts(
        model=model,
        metrics=metrics,
        predictions=predictions.reset_index(drop=True),
        feature_importance=feature_importance.reset_index(drop=True),
    )
