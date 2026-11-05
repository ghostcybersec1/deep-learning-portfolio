"""Clear notebook visualizations using only geographic latitude/longitude axes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import WRFRun, accumulation_to_increment
from .inference import ForecastResult

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None


def _map(
    axis,
    longitude: np.ndarray,
    latitude: np.ndarray,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
):
    artist = axis.pcolormesh(
        longitude, latitude, values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax
    )
    axis.set_title(title)
    axis.set_xlabel("Longitude (°E)")
    axis.set_ylabel("Latitude (°N)")
    axis.grid(alpha=0.2)
    plt.colorbar(artist, ax=axis, shrink=0.82, label=colorbar_label)
    return artist


def plot_run_overview(run: WRFRun, time_index: int = -1):
    """Four physically interpretable maps from a preprocessed hourly run."""

    index = time_index % len(run.times)
    timestamp = pd.Timestamp(run.times[index])
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    _map(
        axes[0, 0],
        run.longitude,
        run.latitude,
        run.temperature_c[index],
        "2 m temperature",
        "°C",
        "coolwarm",
    )
    _map(
        axes[0, 1],
        run.longitude,
        run.latitude,
        run.rain_mm[index],
        "Hourly precipitation (de-accumulated)",
        "mm",
        "Blues",
        vmin=0,
    )
    _map(
        axes[1, 0],
        run.longitude,
        run.latitude,
        run.features[index, 7] * 100,
        "Derived 2 m relative humidity",
        "%",
        "YlGnBu",
        vmin=0,
        vmax=100,
    )
    _map(
        axes[1, 1],
        run.longitude,
        run.latitude,
        run.features[index, 8],
        "10 m wind speed",
        "m s⁻¹",
        "viridis",
        vmin=0,
    )
    figure.suptitle(f"{run.path.name} — valid {timestamp} UTC", fontsize=15)
    return figure


def plot_rain_accumulation_check(
    path_like: str | Path,
    latitude: float = -1.2864,
    longitude: float = 36.8172,
):
    """Show why accumulated WRF rain must be differenced before training."""

    if xr is None:
        raise ImportError("xarray is required for this plot")
    with xr.open_dataset(path_like, decode_times=True) as dataset:
        point = dataset.sel(lat=latitude, lon=longitude, method="nearest")
        accumulated = np.asarray(point["tp"].values, dtype=np.float32)[:, None, None]
        increments = accumulation_to_increment(accumulated)[:, 0, 0]
        time = pd.DatetimeIndex(dataset.time.values)
        selected_lat = float(point.lat.values)
        selected_lon = float(point.lon.values)

    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    axes[0].plot(time, accumulated[:, 0, 0], color="#225ea8")
    axes[0].set_ylabel("Run-total rain (mm)")
    axes[0].grid(alpha=0.25)
    axes[1].bar(time, increments, width=0.03, color="#238b45")
    axes[1].set_ylabel("Interval rain (mm)")
    axes[1].set_xlabel("Valid time (UTC)")
    axes[1].grid(alpha=0.25)
    figure.suptitle(
        f"Precipitation preprocessing near Nairobi ({selected_lat:.2f}, {selected_lon:.2f})"
    )
    return figure


def plot_training_history(history: pd.DataFrame):
    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    axis.plot(history.epoch, history.train_loss, marker="o", label="Training")
    axis.plot(history.epoch, history.validation_loss, marker="o", label="Validation")
    best = history.loc[history.validation_loss.idxmin()]
    axis.scatter([best.epoch], [best.validation_loss], s=100, marker="*", label="Selected")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Multi-task loss")
    axis.set_title("Learning curve")
    axis.grid(alpha=0.25)
    axis.legend()
    return figure


def plot_metric_summary(metrics: pd.DataFrame):
    """MAE comparison; temperature and rain use separate panels/units."""

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for axis, variable, unit in zip(
        axes, ("temperature", "rainfall"), ("°C", "mm per interval")
    ):
        subset = metrics[metrics.variable == variable]
        pivot = subset.pivot(index="horizon_hours", columns="system", values="mae")
        pivot.plot(kind="bar", ax=axis, color=["#2b8cbe", "#bdbdbd"])
        axis.set_title(f"{variable.title()} MAE")
        axis.set_xlabel("Forecast horizon (hours)")
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(title="")
    return figure


def plot_forecast(result: ForecastResult, median_quantile: float = 0.5):
    """Temperature, precipitation, and rain probability at every lead time."""

    median_index = int(np.argmin(np.abs(result.quantiles - median_quantile)))
    rows = len(result.horizons)
    figure, axes = plt.subplots(
        rows, 3, figsize=(16, max(4 * rows, 5)), constrained_layout=True, squeeze=False
    )
    for row, horizon in enumerate(result.horizons):
        valid = pd.Timestamp(result.valid_times[row]).strftime("%Y-%m-%d %H:%M")
        _map(
            axes[row, 0],
            result.longitude,
            result.latitude,
            result.temperature_c[row, median_index],
            f"+{horizon} h temperature\n{valid} UTC",
            "°C (p50)",
            "coolwarm",
        )
        _map(
            axes[row, 1],
            result.longitude,
            result.latitude,
            result.rain_mm[row, median_index],
            f"+{horizon} h precipitation",
            "mm (p50)",
            "Blues",
            vmin=0,
        )
        _map(
            axes[row, 2],
            result.longitude,
            result.latitude,
            result.wet_probability[row] * 100,
            f"+{horizon} h rain probability",
            "%",
            "YlGnBu",
            vmin=0,
            vmax=100,
        )
    figure.suptitle("KMD AI multi-horizon probabilistic forecast", fontsize=16)
    return figure


def plot_daily_product(path_like: str | Path, day_index: int = 0):
    """Inspect the supplied daily WRF aggregate without treating it as truth."""

    if xr is None:
        raise ImportError("xarray is required for this plot")
    with xr.open_dataset(path_like, decode_times=True) as dataset:
        index = day_index % dataset.sizes["time"]
        temperature = np.asarray(dataset.t2_mean.isel(time=index).values) - 273.15
        rainfall = np.asarray(dataset.tp_sum.isel(time=index).values)
        latitude = np.asarray(dataset.lat.values)
        longitude = np.asarray(dataset.lon.values)
        valid = pd.Timestamp(dataset.time.values[index])

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    _map(
        axes[0], longitude, latitude, temperature, "Daily mean temperature", "°C", "coolwarm"
    )
    _map(
        axes[1], longitude, latitude, rainfall, "Daily total precipitation", "mm", "Blues", vmin=0
    )
    figure.suptitle(f"Supplied daily WRF product — {valid:%Y-%m-%d} UTC (not observations)")
    return figure
