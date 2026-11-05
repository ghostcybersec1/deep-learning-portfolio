from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kmd_forecast.data import WRFRun


@pytest.fixture
def synthetic_run() -> WRFRun:
    """Small, deterministic WRF-like trajectory for tests."""

    rng = np.random.default_rng(7)
    steps, height, width = 12, 16, 18
    latitude = np.linspace(-2.0, 2.0, height, dtype=np.float32)
    longitude = np.linspace(34.0, 40.0, width, dtype=np.float32)
    times = np.datetime64("2026-01-01T00:00") + np.arange(steps).astype(
        "timedelta64[h]"
    )
    temperature = 20.0 + rng.normal(size=(steps, height, width)).astype(np.float32)
    rain = np.clip(
        rng.normal(0.1, 0.2, size=(steps, height, width)), 0.0, None
    ).astype(np.float32)
    zeros = np.zeros_like(temperature)
    features = np.stack(
        (
            temperature,
            temperature - 2.0,
            np.full_like(temperature, 1013.0),
            np.full_like(temperature, 0.5),
            np.log1p(rain),
            zeros,
            np.ones_like(temperature),
            np.full_like(temperature, 0.7),
            np.ones_like(temperature),
        ),
        axis=1,
    )
    return WRFRun(
        path=Path("synthetic.nc"),
        reference_time=pd.Timestamp("2026-01-01T00:00"),
        times=times,
        latitude=latitude,
        longitude=longitude,
        lead_hours=np.arange(steps, dtype=np.float32),
        features=features,
        temperature_c=temperature,
        rain_mm=rain,
        attrs={},
    )
