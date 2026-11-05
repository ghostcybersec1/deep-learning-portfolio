"""Deterministic, probabilistic, and event-based weather verification."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import Normalizer


def decode_quantiles(
    prediction: dict[str, torch.Tensor], normalizer: Normalizer
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert normalized network outputs to °C, mm, and probabilities."""

    temperature = prediction["temperature_quantiles"].detach().float().cpu().numpy()
    rain = prediction["rain_quantiles"].detach().float().cpu().numpy()
    wet_probability = (
        torch.sigmoid(prediction["wet_logits"]).detach().float().cpu().numpy()
    )
    temperature = temperature * normalizer.temperature_std + normalizer.temperature_mean
    rain = np.expm1(rain * normalizer.rain_log_std + normalizer.rain_log_mean)
    rain = np.clip(rain, 0.0, None)
    # Small-data training can cause crossing. Sorting preserves the quantile contract.
    temperature.sort(axis=2)
    rain.sort(axis=2)
    return temperature, rain, wet_probability


def _new_accumulator() -> dict[str, float]:
    return defaultdict(float)


def _update(
    accumulator: dict[str, float],
    prediction: np.ndarray,
    target: np.ndarray,
    predicted_wet: np.ndarray,
    target_wet: np.ndarray,
    wet_probability: np.ndarray,
) -> None:
    difference = prediction - target
    accumulator["count"] += target.size
    accumulator["absolute_error"] += float(np.abs(difference).sum())
    accumulator["squared_error"] += float(np.square(difference).sum())
    accumulator["error"] += float(difference.sum())
    accumulator["brier"] += float(np.square(wet_probability - target_wet).sum())
    accumulator["hits"] += float(np.logical_and(predicted_wet, target_wet).sum())
    accumulator["misses"] += float(np.logical_and(~predicted_wet, target_wet).sum())
    accumulator["false_alarms"] += float(np.logical_and(predicted_wet, ~target_wet).sum())


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    normalizer: Normalizer,
    horizons: Sequence[int],
    quantiles: Sequence[float],
    rain_threshold_mm: float,
    device: torch.device,
) -> pd.DataFrame:
    """Compare the neural model with persistence on held-out forecast runs."""

    model.eval()
    median_index = int(np.argmin(np.abs(np.asarray(quantiles) - 0.5)))
    low_index = 0
    high_index = len(quantiles) - 1
    accumulators = {
        (system, variable, horizon): _new_accumulator()
        for system in ("AI model", "Persistence")
        for variable in ("temperature", "rainfall")
        for horizon in horizons
    }
    coverage = defaultdict(float)

    for batch in tqdm(loader, desc="verification", leave=False):
        x = batch["x"].to(device, non_blocking=True)
        output = model(x)
        temperature_q, rain_q, wet_probability = decode_quantiles(output, normalizer)
        target_temperature = batch["temperature_c"].numpy()
        target_rain = batch["rain_mm"].numpy()
        persistence_temperature = batch["persistence_temperature_c"].numpy()
        persistence_rain = batch["persistence_rain_mm"].numpy()

        for index, horizon in enumerate(horizons):
            truth_t = target_temperature[:, index]
            truth_r = target_rain[:, index]
            target_wet = truth_r >= rain_threshold_mm

            model_t = temperature_q[:, index, median_index]
            model_r = rain_q[:, index, median_index]
            model_probability = wet_probability[:, index]
            model_wet = model_probability >= 0.5
            # The hurdle component makes a dry prediction exactly zero.
            model_r = np.where(model_wet, model_r, 0.0)

            _update(
                accumulators[("AI model", "temperature", horizon)],
                model_t,
                truth_t,
                model_wet,
                target_wet,
                model_probability,
            )
            _update(
                accumulators[("AI model", "rainfall", horizon)],
                model_r,
                truth_r,
                model_wet,
                target_wet,
                model_probability,
            )

            baseline_wet = persistence_rain >= rain_threshold_mm
            _update(
                accumulators[("Persistence", "temperature", horizon)],
                persistence_temperature,
                truth_t,
                baseline_wet,
                target_wet,
                baseline_wet.astype(np.float32),
            )
            _update(
                accumulators[("Persistence", "rainfall", horizon)],
                persistence_rain,
                truth_r,
                baseline_wet,
                target_wet,
                baseline_wet.astype(np.float32),
            )

            coverage[("temperature", horizon)] += float(
                np.logical_and(
                    truth_t >= temperature_q[:, index, low_index],
                    truth_t <= temperature_q[:, index, high_index],
                ).sum()
            )
            coverage[("rainfall", horizon)] += float(
                np.logical_and(
                    truth_r >= rain_q[:, index, low_index],
                    truth_r <= rain_q[:, index, high_index],
                ).sum()
            )

    rows = []
    for (system, variable, horizon), values in accumulators.items():
        count = max(values["count"], 1.0)
        hits = values["hits"]
        misses = values["misses"]
        false_alarms = values["false_alarms"]
        row = {
            "system": system,
            "variable": variable,
            "horizon_hours": horizon,
            "mae": values["absolute_error"] / count,
            "rmse": np.sqrt(values["squared_error"] / count),
            "bias": values["error"] / count,
            "brier_score": values["brier"] / count if variable == "rainfall" else np.nan,
            "pod": hits / max(hits + misses, 1.0) if variable == "rainfall" else np.nan,
            "far": (
                false_alarms / max(hits + false_alarms, 1.0)
                if variable == "rainfall"
                else np.nan
            ),
            "csi": (
                hits / max(hits + misses + false_alarms, 1.0)
                if variable == "rainfall"
                else np.nan
            ),
            "p10_p90_coverage": (
                coverage[(variable, horizon)] / count if system == "AI model" else np.nan
            ),
        }
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        ["variable", "horizon_hours", "system"], ignore_index=True
    )
    baseline_mae = result[result.system == "Persistence"].set_index(
        ["variable", "horizon_hours"]
    )["mae"]
    result["mae_skill_vs_persistence_pct"] = np.nan
    model_rows = result.system == "AI model"
    result.loc[model_rows, "mae_skill_vs_persistence_pct"] = [
        100.0 * (1.0 - row.mae / max(baseline_mae[(row.variable, row.horizon_hours)], 1e-8))
        for row in result[model_rows].itertuples()
    ]
    return result
