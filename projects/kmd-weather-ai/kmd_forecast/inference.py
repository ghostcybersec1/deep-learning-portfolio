"""Overlap-tiled whole-domain inference and CF-style NetCDF export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .data import Normalizer, WRFRun, make_model_input
from .evaluation import decode_quantiles

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None


@dataclass
class ForecastResult:
    valid_times: np.ndarray
    horizons: np.ndarray
    quantiles: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    temperature_c: np.ndarray  # [horizon, quantile, lat, lon]
    rain_mm: np.ndarray  # [horizon, quantile, lat, lon]
    wet_probability: np.ndarray  # [horizon, lat, lon]
    model_initialization_time: pd.Timestamp
    source_wrf_reference_time: pd.Timestamp


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size > length:
        raise ValueError(f"Tile size {tile_size} exceeds axis length {length}")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("Overlap must be non-negative and smaller than tile size")
    step = tile_size - overlap
    starts = list(range(0, max(length - tile_size + 1, 1), step))
    final = length - tile_size
    if not starts or starts[-1] != final:
        starts.append(final)
    return starts


def _blend_weight(tile_size: int) -> np.ndarray:
    one_dimensional = np.hanning(tile_size).astype(np.float32)
    # Non-zero edges are required at the outer border of the complete domain.
    one_dimensional = np.clip(one_dimensional, 0.05, None)
    weight = np.outer(one_dimensional, one_dimensional)
    return weight / weight.max()


@torch.no_grad()
def predict_full_grid(
    model: torch.nn.Module,
    run: WRFRun,
    normalizer: Normalizer,
    history_hours: int,
    horizons: Sequence[int],
    quantiles: Sequence[float],
    tile_size: int,
    overlap: int,
    device: torch.device,
    anchor: int | None = None,
    tile_batch_size: int = 4,
) -> ForecastResult:
    """Predict the complete domain while controlling GPU/CPU memory."""

    model.eval()
    model.to(device)
    anchor = len(run.times) - 1 if anchor is None else int(anchor)
    if anchor < history_hours - 1:
        raise ValueError("Anchor does not have enough input history")
    _, height, width = run.shape
    y_starts = _tile_starts(height, tile_size, overlap)
    x_starts = _tile_starts(width, tile_size, overlap)
    positions = [(y0, x0) for y0 in y_starts for x0 in x_starts]
    weight = _blend_weight(tile_size)

    n_horizons = len(horizons)
    n_quantiles = len(quantiles)
    temperature_sum = np.zeros(
        (n_horizons, n_quantiles, height, width), dtype=np.float32
    )
    rain_sum = np.zeros_like(temperature_sum)
    wet_sum = np.zeros((n_horizons, height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    for first in range(0, len(positions), tile_batch_size):
        batch_positions = positions[first : first + tile_batch_size]
        inputs = []
        for y0, x0 in batch_positions:
            inputs.append(
                make_model_input(
                    run,
                    anchor,
                    history_hours,
                    normalizer,
                    slice(y0, y0 + tile_size),
                    slice(x0, x0 + tile_size),
                )
            )
        tensor = torch.from_numpy(np.stack(inputs)).to(device, non_blocking=True)
        output = model(tensor)
        temperature, rain, wet_probability = decode_quantiles(output, normalizer)

        for item, (y0, x0) in enumerate(batch_positions):
            ys = slice(y0, y0 + tile_size)
            xs = slice(x0, x0 + tile_size)
            temperature_sum[:, :, ys, xs] += temperature[item] * weight
            rain_sum[:, :, ys, xs] += rain[item] * weight
            wet_sum[:, ys, xs] += wet_probability[item] * weight
            weight_sum[ys, xs] += weight

    denominator = np.maximum(weight_sum, 1e-6)
    temperature = temperature_sum / denominator[None, None]
    rain = rain_sum / denominator[None, None]
    wet_probability = wet_sum / denominator[None]
    # Weighted averages of ordered quantiles remain ordered; clip physical fields.
    rain = np.clip(rain, 0.0, None)
    wet_probability = np.clip(wet_probability, 0.0, 1.0)

    initialization_time = pd.Timestamp(run.times[anchor])
    valid_times = np.asarray(
        [initialization_time + pd.Timedelta(hours=int(value)) for value in horizons],
        dtype="datetime64[ns]",
    )
    return ForecastResult(
        valid_times=valid_times,
        horizons=np.asarray(horizons, dtype=np.int16),
        quantiles=np.asarray(quantiles, dtype=np.float32),
        latitude=run.latitude,
        longitude=run.longitude,
        temperature_c=temperature,
        rain_mm=rain,
        wet_probability=wet_probability,
        model_initialization_time=initialization_time,
        source_wrf_reference_time=run.reference_time,
    )


def export_forecast_netcdf(
    result: ForecastResult,
    path_like: str | Path,
    model_name: str = "KMD Temporal-Attention U-Net",
) -> Path:
    """Write probabilistic forecasts to a compressed, self-describing NetCDF file."""

    if xr is None:
        raise ImportError("xarray is required for NetCDF export")
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("valid_time", "quantile", "lat", "lon"),
                result.temperature_c,
                {
                    "long_name": "AI forecast of 2 m air temperature",
                    "standard_name": "air_temperature",
                    "units": "degree_Celsius",
                },
            ),
            "precipitation_interval": (
                ("valid_time", "quantile", "lat", "lon"),
                result.rain_mm,
                {
                    "long_name": "AI forecast of interval precipitation",
                    "units": "mm",
                },
            ),
            "probability_of_precipitation": (
                ("valid_time", "lat", "lon"),
                result.wet_probability,
                {
                    "long_name": "Probability of precipitation at or above training threshold",
                    "units": "1",
                },
            ),
            "forecast_period": (
                ("valid_time",),
                result.horizons.astype("timedelta64[h]"),
                {"long_name": "forecast lead time"},
            ),
        },
        coords={
            "valid_time": result.valid_times,
            "quantile": result.quantiles,
            "lat": (
                "lat",
                result.latitude,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "lon": (
                "lon",
                result.longitude,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": "KMD probabilistic deep-learning weather forecast",
            "model": model_name,
            "model_initialization_time": result.model_initialization_time.isoformat(),
            "source_wrf_reference_time": result.source_wrf_reference_time.isoformat(),
            "disclaimer": (
                "Research forecast. Validate against independent observations before "
                "any operational or public-safety use."
            ),
        },
    )
    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for name in (
            "temperature_2m",
            "precipitation_interval",
            "probability_of_precipitation",
        )
    }
    dataset.to_netcdf(path, encoding=encoding)
    return path
