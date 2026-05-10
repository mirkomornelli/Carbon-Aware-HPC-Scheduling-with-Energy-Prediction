from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    model: object
    metrics: dict[str, object]
    model_comparison: pd.DataFrame
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


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.60,
    validation_ratio: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("start_time").reset_index(drop=True)
    train_cut = int(len(ordered) * train_ratio)
    validation_cut = int(len(ordered) * (train_ratio + validation_ratio))
    return (
        ordered.iloc[:train_cut].copy(),
        ordered.iloc[train_cut:validation_cut].copy(),
        ordered.iloc[validation_cut:].copy(),
    )


def build_candidate_models(random_state: int = 42) -> dict[str, object]:
    return {
        "ridge_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=90,
            max_depth=18,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=90,
            max_depth=18,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=160,
            learning_rate=0.06,
            l2_regularization=0.01,
            random_state=random_state,
        ),
    }


def _predict_energy(model: object, features: pd.DataFrame) -> np.ndarray:
    pred = np.expm1(model.predict(features))
    return np.clip(pred, 0, None)


def _evaluate_predictions(
    actual_energy: np.ndarray,
    predicted_energy: np.ndarray,
    carbon_intensity: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    actual_emissions = actual_energy * carbon_intensity / 1000.0
    predicted_emissions = predicted_energy * carbon_intensity / 1000.0
    return {
        f"{prefix}_energy_mae_kwh": float(mean_absolute_error(actual_energy, predicted_energy)),
        f"{prefix}_energy_rmse_kwh": float(mean_squared_error(actual_energy, predicted_energy) ** 0.5),
        f"{prefix}_energy_r2": float(r2_score(actual_energy, predicted_energy)),
        f"{prefix}_emissions_mae_kg": float(mean_absolute_error(actual_emissions, predicted_emissions)),
        f"{prefix}_emissions_rmse_kg": float(mean_squared_error(actual_emissions, predicted_emissions) ** 0.5),
        f"{prefix}_emissions_r2": float(r2_score(actual_emissions, predicted_emissions)),
    }


def _feature_importance(
    model: object,
    x_reference: pd.DataFrame,
    y_reference_log: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    fitted_model = model
    if isinstance(model, Pipeline):
        fitted_model = model.named_steps["regressor"]

    if hasattr(fitted_model, "feature_importances_"):
        values = fitted_model.feature_importances_
    elif hasattr(fitted_model, "coef_"):
        values = np.abs(np.ravel(fitted_model.coef_))
    else:
        sample_size = min(3000, len(x_reference))
        x_sample = x_reference.sample(sample_size, random_state=random_state)
        y_sample = y_reference_log.loc[x_sample.index]
        values = permutation_importance(
            model,
            x_sample,
            y_sample,
            n_repeats=5,
            random_state=random_state,
            scoring="r2",
        ).importances_mean

    if len(values) != len(FEATURE_COLUMNS):
        values = np.zeros(len(FEATURE_COLUMNS))

    return (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_energy_model(df: pd.DataFrame, random_state: int = 42) -> PredictionArtifacts:
    train_df, validation_df, test_df = chronological_split(df)
    x_train = train_df[FEATURE_COLUMNS]
    y_train_log = np.log1p(train_df["energy_kwh"])

    comparison_rows = []
    candidate_models = build_candidate_models(random_state=random_state)
    for model_name, candidate in candidate_models.items():
        candidate.fit(x_train, y_train_log)
        validation_pred_energy = _predict_energy(candidate, validation_df[FEATURE_COLUMNS])
        validation_metrics = _evaluate_predictions(
            actual_energy=validation_df["energy_kwh"].to_numpy(),
            predicted_energy=validation_pred_energy,
            carbon_intensity=validation_df["gco2_per_kwh_at_start"].to_numpy(),
            prefix="validation",
        )
        comparison_rows.append({"model": model_name, **validation_metrics})

    model_comparison = pd.DataFrame(comparison_rows).sort_values(
        ["validation_energy_r2", "validation_energy_mae_kwh"],
        ascending=[False, True],
    )
    selected_model_name = str(model_comparison.iloc[0]["model"])
    model = candidate_models[selected_model_name]

    train_validation_df = pd.concat([train_df, validation_df], ignore_index=True)
    model.fit(train_validation_df[FEATURE_COLUMNS], np.log1p(train_validation_df["energy_kwh"]))

    test_pred_energy = _predict_energy(model, test_df[FEATURE_COLUMNS])
    test_metrics = _evaluate_predictions(
        actual_energy=test_df["energy_kwh"].to_numpy(),
        predicted_energy=test_pred_energy,
        carbon_intensity=test_df["gco2_per_kwh_at_start"].to_numpy(),
        prefix="test",
    )

    metrics = {
        "selected_model": selected_model_name,
        "selection_metric": "validation_energy_r2",
        "selected_validation_energy_r2": float(model_comparison.iloc[0]["validation_energy_r2"]),
        "selected_validation_energy_mae_kwh": float(model_comparison.iloc[0]["validation_energy_mae_kwh"]),
        "rows_total": float(len(df)),
        "rows_train": float(len(train_df)),
        "rows_validation": float(len(validation_df)),
        "rows_test": float(len(test_df)),
        **test_metrics,
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
    validation_ids = set(validation_df["job_id"])
    predictions["split"] = np.select(
        [
            predictions["job_id"].isin(train_ids),
            predictions["job_id"].isin(validation_ids),
        ],
        ["train", "validation"],
        default="test",
    )
    predictions["selected_model"] = selected_model_name
    predictions["pred_energy_kwh"] = _predict_energy(model, df[FEATURE_COLUMNS])
    predictions["pred_emissions_kg"] = (
        predictions["pred_energy_kwh"] * predictions["gco2_per_kwh_at_start"] / 1000.0
    )
    predictions["abs_energy_error_kwh"] = np.abs(predictions["energy_kwh"] - predictions["pred_energy_kwh"])
    predictions["abs_emissions_error_kg"] = np.abs(predictions["emissions_kg"] - predictions["pred_emissions_kg"])

    feature_importance = _feature_importance(
        model,
        train_validation_df[FEATURE_COLUMNS],
        np.log1p(train_validation_df["energy_kwh"]),
        random_state=random_state,
    )

    return PredictionArtifacts(
        model=model,
        metrics=metrics,
        model_comparison=model_comparison.reset_index(drop=True),
        predictions=predictions.reset_index(drop=True),
        feature_importance=feature_importance.reset_index(drop=True),
    )
