"""WRF NetCDF discovery, physically correct preprocessing, and PyTorch datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:  # Give notebook users a useful error instead of failing at package import.
    import xarray as xr
except ImportError:  # pragma: no cover - exercised only in incomplete environments
    xr = None


RAW_VARIABLES = ("t2", "d2m", "msl", "tcc", "tp", "u10", "v10")
DYNAMIC_FEATURE_NAMES = (
    "temperature_2m_c",
    "dewpoint_2m_c",
    "mean_sea_level_pressure_hpa",
    "cloud_fraction",
    "log1p_hourly_rain_mm",
    "eastward_wind_ms",
    "northward_wind_ms",
    "relative_humidity_fraction",
    "wind_speed_ms",
)
CONTEXT_FEATURE_NAMES = (
    "latitude_scaled",
    "longitude_scaled",
    "hour_sin",
    "hour_cos",
    "forecast_lead_scaled",
)
INPUT_FEATURE_NAMES = DYNAMIC_FEATURE_NAMES + CONTEXT_FEATURE_NAMES


def _require_xarray() -> None:
    if xr is None:
        raise ImportError(
            "xarray is required to read the NetCDF files. In Jupyter, run "
            "`%pip install -r requirements.txt`, restart the kernel, and try again."
        )


def _as_timestamp(value) -> pd.Timestamp:
    """Convert xarray/numpy scalar time representations to a UTC-naive Timestamp."""

    arr = np.asarray(value)
    scalar = arr.reshape(-1)[0]
    stamp = pd.Timestamp(scalar)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp


def dataset_inventory(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Read only metadata and return a compact inventory of NetCDF products."""

    _require_xarray()
    rows: list[dict] = []
    for path_like in paths:
        path = Path(path_like)
        with xr.open_dataset(path, decode_times=True) as ds:
            variables = set(ds.data_vars)
            if set(RAW_VARIABLES).issubset(variables):
                product = "hourly WRF run"
            elif ds.attrs.get("wrfout_tidy_product"):
                product = f"{ds.attrs['wrfout_tidy_product']} WRF aggregate"
            else:
                product = "other NetCDF"

            ref = ds.attrs.get("forecast_reference_time")
            if ref is None and "forecast_reference_time" in ds:
                ref = _as_timestamp(ds["forecast_reference_time"].values)
            else:
                ref = pd.Timestamp(ref) if ref is not None else pd.NaT

            times = pd.DatetimeIndex(ds["time"].values) if "time" in ds else None
            rows.append(
                {
                    "file": path.name,
                    "product": product,
                    "forecast_reference_time": ref,
                    "time_start": times.min() if times is not None else pd.NaT,
                    "time_end": times.max() if times is not None else pd.NaT,
                    "time_steps": int(ds.sizes.get("time", 0)),
                    "latitude_points": int(ds.sizes.get("lat", 0)),
                    "longitude_points": int(ds.sizes.get("lon", 0)),
                    "data_variables": len(ds.data_vars),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["product", "forecast_reference_time"], ignore_index=True
    )


def identify_raw_forecast_files(paths: Sequence[str | Path]) -> list[Path]:
    """Return hourly WRF files sorted by forecast initialization time."""

    _require_xarray()
    found: list[tuple[pd.Timestamp, Path]] = []
    for path_like in paths:
        path = Path(path_like)
        with xr.open_dataset(path, decode_times=True) as ds:
            if not set(RAW_VARIABLES).issubset(ds.data_vars):
                continue
            ref = ds.attrs.get("forecast_reference_time")
            if ref is None:
                ref = _as_timestamp(ds["forecast_reference_time"].values)
            found.append((pd.Timestamp(ref), path))
    return [path for _, path in sorted(found, key=lambda item: item[0])]


def accumulation_to_increment(accumulated_mm: np.ndarray) -> np.ndarray:
    """Convert run-total precipitation to non-negative interval precipitation.

    A negative difference usually indicates an accumulation reset. In that case,
    the new accumulated value is the interval amount. The first frame has no
    preceding interval inside the file and is set to zero.
    """

    accumulated = np.asarray(accumulated_mm, dtype=np.float32)
    if accumulated.ndim < 1 or accumulated.shape[0] < 1:
        raise ValueError("Accumulated precipitation must have a non-empty time axis")
    increments = np.zeros_like(accumulated, dtype=np.float32)
    if accumulated.shape[0] > 1:
        difference = accumulated[1:] - accumulated[:-1]
        increments[1:] = np.where(difference >= -1e-4, difference, accumulated[1:])
    np.maximum(increments, 0.0, out=increments)
    return increments


def relative_humidity_fraction(temperature_c: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
    """Estimate 2 m relative humidity with the Magnus formula."""

    a, b = 17.625, 243.04
    exponent = (a * dewpoint_c / (b + dewpoint_c)) - (
        a * temperature_c / (b + temperature_c)
    )
    return np.clip(np.exp(exponent), 0.0, 1.0).astype(np.float32)


@dataclass
class WRFRun:
    """One in-memory, regularly gridded hourly WRF forecast trajectory."""

    path: Path
    reference_time: pd.Timestamp
    times: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    lead_hours: np.ndarray
    features: np.ndarray  # [time, dynamic_feature, latitude, longitude]
    temperature_c: np.ndarray  # [time, latitude, longitude]
    rain_mm: np.ndarray  # interval rainfall [time, latitude, longitude]
    attrs: dict

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.temperature_c.shape


def _validate_run(run: WRFRun) -> None:
    expected = (len(run.times), len(run.latitude), len(run.longitude))
    if run.temperature_c.shape != expected or run.rain_mm.shape != expected:
        raise ValueError(f"Inconsistent WRF shapes in {run.path.name}: expected {expected}")
    if run.features.shape != (expected[0], len(DYNAMIC_FEATURE_NAMES), *expected[1:]):
        raise ValueError(f"Unexpected feature cube shape in {run.path.name}")
    if not np.all(np.diff(run.times).astype("timedelta64[s]") > np.timedelta64(0, "s")):
        raise ValueError(f"Time is not strictly increasing in {run.path.name}")
    if not np.all(np.isfinite(run.features)):
        raise ValueError(f"Non-finite values remain after preprocessing {run.path.name}")
    if np.nanmin(run.rain_mm) < 0:
        raise ValueError(f"Negative interval rainfall found in {run.path.name}")


def load_wrf_run(path_like: str | Path, spatial_stride: int = 1) -> WRFRun:
    """Load and engineer one hourly WRF run.

    ``spatial_stride=2`` is useful for a fast laptop demonstration. Operational
    experiments should retain the native grid with ``spatial_stride=1``.
    """

    _require_xarray()
    path = Path(path_like)
    if spatial_stride < 1:
        raise ValueError("spatial_stride must be at least 1")

    with xr.open_dataset(path, decode_times=True, mask_and_scale=True) as ds:
        missing = sorted(set(RAW_VARIABLES).difference(ds.data_vars))
        if missing:
            raise ValueError(f"{path.name} is missing raw variables: {missing}")
        spatial_index = {
            "lat": slice(None, None, spatial_stride),
            "lon": slice(None, None, spatial_stride),
        }

        def field(name: str) -> np.ndarray:
            return np.asarray(ds[name].isel(**spatial_index).values, dtype=np.float32)

        temperature_c = field("t2") - np.float32(273.15)
        dewpoint_c = field("d2m") - np.float32(273.15)
        pressure_hpa = field("msl") / np.float32(100.0)
        cloud_fraction = np.clip(field("tcc"), 0.0, 1.0)
        accumulated_rain = field("tp")
        rain_mm = accumulation_to_increment(accumulated_rain)
        u10 = field("u10")
        v10 = field("v10")
        humidity = relative_humidity_fraction(temperature_c, dewpoint_c)
        wind_speed = np.hypot(u10, v10).astype(np.float32)

        features = np.stack(
            (
                temperature_c,
                dewpoint_c,
                pressure_hpa,
                cloud_fraction,
                np.log1p(rain_mm),
                u10,
                v10,
                humidity,
                wind_speed,
            ),
            axis=1,
        ).astype(np.float32, copy=False)

        times = np.asarray(ds["time"].values, dtype="datetime64[ns]")
        latitude = np.asarray(ds["lat"].isel(lat=spatial_index["lat"]).values, dtype=np.float32)
        longitude = np.asarray(ds["lon"].isel(lon=spatial_index["lon"]).values, dtype=np.float32)
        if "forecast_reference_time" in ds:
            reference_time = _as_timestamp(ds["forecast_reference_time"].values)
        else:
            reference_time = pd.Timestamp(ds.attrs["forecast_reference_time"])
        lead_hours = (
            (times - reference_time.to_datetime64()) / np.timedelta64(1, "h")
        ).astype(np.float32)
        attrs = dict(ds.attrs)

    run = WRFRun(
        path=path,
        reference_time=reference_time,
        times=times,
        latitude=latitude,
        longitude=longitude,
        lead_hours=lead_hours,
        features=features,
        temperature_c=temperature_c.astype(np.float32, copy=False),
        rain_mm=rain_mm.astype(np.float32, copy=False),
        attrs=attrs,
    )
    _validate_run(run)
    return run


@dataclass
class Normalizer:
    """Training-only normalization and geographic reference values."""

    input_mean: np.ndarray
    input_std: np.ndarray
    temperature_mean: float
    temperature_std: float
    rain_log_mean: float
    rain_log_std: float
    latitude_mid: float
    latitude_scale: float
    longitude_mid: float
    longitude_scale: float
    lead_scale: float
    wet_positive_weight: float

    def to_dict(self) -> dict:
        result = asdict(self)
        result["input_mean"] = self.input_mean.tolist()
        result["input_std"] = self.input_std.tolist()
        return result

    @classmethod
    def from_dict(cls, values: dict) -> "Normalizer":
        converted = dict(values)
        converted["input_mean"] = np.asarray(converted["input_mean"], dtype=np.float32)
        converted["input_std"] = np.asarray(converted["input_std"], dtype=np.float32)
        return cls(**converted)


def _streaming_mean_std(arrays: Sequence[np.ndarray], channel_axis: int | None = None):
    """Compute moments in time chunks to avoid a full float64 copy of the archive."""

    chunk_size = 8
    if channel_axis is None:
        count = 0
        total = 0.0
        square = 0.0
        for array in arrays:
            for start in range(0, array.shape[0], chunk_size):
                block = array[start : start + chunk_size]
                count += block.size
                total += float(np.sum(block, dtype=np.float64))
                square += float(np.sum(np.square(block, dtype=np.float64), dtype=np.float64))
    else:
        channels = arrays[0].shape[channel_axis]
        count = np.zeros(channels, dtype=np.float64)
        total = np.zeros(channels, dtype=np.float64)
        square = np.zeros(channels, dtype=np.float64)
        for array in arrays:
            for start in range(0, array.shape[0], chunk_size):
                block = array[start : start + chunk_size]
                reduce_axes = tuple(i for i in range(block.ndim) if i != channel_axis)
                elements = np.prod([block.shape[i] for i in reduce_axes])
                count += elements
                total += np.sum(block, axis=reduce_axes, dtype=np.float64)
                square += np.sum(
                    np.square(block, dtype=np.float64),
                    axis=reduce_axes,
                    dtype=np.float64,
                )
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 1e-8)
    return np.asarray(mean, dtype=np.float32), np.asarray(np.sqrt(variance), dtype=np.float32)


def fit_normalizer(
    runs: Sequence[WRFRun], rain_threshold_mm: float = 0.1
) -> Normalizer:
    """Fit scaling values using training runs only."""

    if not runs:
        raise ValueError("At least one training run is required")
    input_mean, input_std = _streaming_mean_std(
        [run.features for run in runs], channel_axis=1
    )
    temperature_mean, temperature_std = _streaming_mean_std(
        [run.temperature_c for run in runs]
    )
    rain_logs = [np.log1p(run.rain_mm) for run in runs]
    rain_log_mean, rain_log_std = _streaming_mean_std(rain_logs)

    wet = sum(int(np.count_nonzero(run.rain_mm >= rain_threshold_mm)) for run in runs)
    total = sum(run.rain_mm.size for run in runs)
    dry = total - wet
    positive_weight = float(np.clip(dry / max(wet, 1), 1.0, 20.0))

    latitude = runs[0].latitude
    longitude = runs[0].longitude
    latitude_mid = float((latitude.min() + latitude.max()) / 2)
    longitude_mid = float((longitude.min() + longitude.max()) / 2)
    latitude_scale = float(max((latitude.max() - latitude.min()) / 2, 1e-6))
    longitude_scale = float(max((longitude.max() - longitude.min()) / 2, 1e-6))
    lead_scale = float(max(max(run.lead_hours.max() for run in runs), 1.0))

    return Normalizer(
        input_mean=input_mean,
        input_std=np.maximum(input_std, 1e-4),
        temperature_mean=float(temperature_mean),
        temperature_std=float(max(temperature_std, 1e-4)),
        rain_log_mean=float(rain_log_mean),
        rain_log_std=float(max(rain_log_std, 1e-4)),
        latitude_mid=latitude_mid,
        latitude_scale=latitude_scale,
        longitude_mid=longitude_mid,
        longitude_scale=longitude_scale,
        lead_scale=lead_scale,
        wet_positive_weight=positive_weight,
    )


def make_model_input(
    run: WRFRun,
    anchor: int,
    history_hours: int,
    normalizer: Normalizer,
    y_slice: slice,
    x_slice: slice,
) -> np.ndarray:
    """Build one normalized [feature, time, y, x] model input cube."""

    first = anchor - history_hours + 1
    if first < 0 or anchor >= len(run.times):
        raise IndexError("The requested anchor does not have the required history")
    time_index = np.arange(first, anchor + 1)
    dynamic = run.features[time_index, :, y_slice, x_slice]
    dynamic = (dynamic - normalizer.input_mean[None, :, None, None]) / (
        normalizer.input_std[None, :, None, None]
    )
    dynamic = np.transpose(dynamic, (1, 0, 2, 3))

    latitude = run.latitude[y_slice]
    longitude = run.longitude[x_slice]
    lat_grid, lon_grid = np.meshgrid(latitude, longitude, indexing="ij")
    lat_scaled = (lat_grid - normalizer.latitude_mid) / normalizer.latitude_scale
    lon_scaled = (lon_grid - normalizer.longitude_mid) / normalizer.longitude_scale
    location = np.stack((lat_scaled, lon_scaled), axis=0).astype(np.float32)
    location = np.repeat(location[:, None], history_hours, axis=1)

    timestamps = pd.DatetimeIndex(run.times[time_index])
    decimal_hour = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60.0
    phase = 2 * np.pi * decimal_hour / 24.0
    temporal = np.stack(
        (
            np.sin(phase),
            np.cos(phase),
            np.clip(run.lead_hours[time_index] / normalizer.lead_scale, 0.0, 2.0),
        ),
        axis=0,
    ).astype(np.float32)
    height, width = dynamic.shape[-2:]
    temporal = np.broadcast_to(
        temporal[:, :, None, None], (3, history_hours, height, width)
    )
    return np.ascontiguousarray(np.concatenate((dynamic, location, temporal), axis=0))


class SpatioTemporalPatchDataset(Dataset):
    """Leakage-aware time/space patches drawn from one or more WRF runs."""

    def __init__(
        self,
        runs: Sequence[WRFRun],
        normalizer: Normalizer,
        history_hours: int,
        horizons: Sequence[int],
        patch_size: int,
        patches_per_anchor: int,
        rain_threshold_mm: float,
        training: bool,
        seed: int = 42,
    ) -> None:
        if not runs:
            raise ValueError("At least one run is required")
        self.runs = list(runs)
        self.normalizer = normalizer
        self.history_hours = int(history_hours)
        self.horizons = tuple(int(value) for value in horizons)
        self.patch_size = int(patch_size)
        self.patches_per_anchor = int(patches_per_anchor)
        self.rain_threshold_mm = float(rain_threshold_mm)
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0
        self.samples: list[tuple[int, int]] = []

        for run_index, run in enumerate(self.runs):
            _, height, width = run.shape
            if self.patch_size > min(height, width):
                raise ValueError(
                    f"Patch {self.patch_size} exceeds grid {height}x{width} in {run.path.name}"
                )
            last_exclusive = len(run.times) - max(self.horizons)
            anchors = range(self.history_hours - 1, last_exclusive)
            self.samples.extend((run_index, anchor) for anchor in anchors)
        if not self.samples:
            raise ValueError("No valid forecast samples; reduce history/horizons")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples) * self.patches_per_anchor

    def _patch_origin(self, index: int, height: int, width: int) -> tuple[int, int]:
        max_y = height - self.patch_size
        max_x = width - self.patch_size
        if self.training:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
            y0 = int(rng.integers(0, max_y + 1))
            x0 = int(rng.integers(0, max_x + 1))
        else:
            # Deterministic co-prime walks cover the domain without random validation.
            y0 = (index * 37) % (max_y + 1) if max_y else 0
            x0 = (index * 53) % (max_x + 1) if max_x else 0
        return y0, x0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        sample_index = index // self.patches_per_anchor
        run_index, anchor = self.samples[sample_index]
        run = self.runs[run_index]
        _, height, width = run.shape
        y0, x0 = self._patch_origin(index, height, width)
        ys = slice(y0, y0 + self.patch_size)
        xs = slice(x0, x0 + self.patch_size)

        model_input = make_model_input(
            run, anchor, self.history_hours, self.normalizer, ys, xs
        )
        target_indices = np.asarray([anchor + horizon for horizon in self.horizons])
        temperature_c = run.temperature_c[target_indices, ys, xs]
        rain_mm = run.rain_mm[target_indices, ys, xs]
        temperature_z = (
            temperature_c - self.normalizer.temperature_mean
        ) / self.normalizer.temperature_std
        rain_log_z = (
            np.log1p(rain_mm) - self.normalizer.rain_log_mean
        ) / self.normalizer.rain_log_std

        return {
            "x": torch.from_numpy(model_input),
            "temperature_z": torch.from_numpy(temperature_z.astype(np.float32)),
            "rain_log_z": torch.from_numpy(rain_log_z.astype(np.float32)),
            "wet": torch.from_numpy((rain_mm >= self.rain_threshold_mm).astype(np.float32)),
            "temperature_c": torch.from_numpy(temperature_c.astype(np.float32)),
            "rain_mm": torch.from_numpy(rain_mm.astype(np.float32)),
            "persistence_temperature_c": torch.from_numpy(
                run.temperature_c[anchor, ys, xs].astype(np.float32)
            ),
            "persistence_rain_mm": torch.from_numpy(
                run.rain_mm[anchor, ys, xs].astype(np.float32)
            ),
            "run_index": run_index,
            "anchor_index": anchor,
            "y0": y0,
            "x0": x0,
        }


def run_summary(runs: Sequence[WRFRun]) -> pd.DataFrame:
    """Human-readable table after physical preprocessing."""

    rows = []
    for run in runs:
        rows.append(
            {
                "file": run.path.name,
                "initialized_utc": run.reference_time,
                "first_valid_utc": pd.Timestamp(run.times[0]),
                "last_valid_utc": pd.Timestamp(run.times[-1]),
                "shape": " × ".join(map(str, run.shape)),
                "temperature_min_c": float(run.temperature_c.min()),
                "temperature_max_c": float(run.temperature_c.max()),
                "hourly_rain_max_mm": float(run.rain_mm.max()),
                "rainy_pixel_fraction": float(np.mean(run.rain_mm >= 0.1)),
            }
        )
    return pd.DataFrame(rows)
