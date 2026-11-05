# KMD Weather AI

[← Back to the Deep Learning Portfolio](../../README.md)

An end-to-end, probabilistic deep-learning workflow for Weather Research and
Forecasting (WRF) data over Kenya. The project preprocesses hourly forecast
grids, trains a Temporal-Attention U-Net, evaluates it against persistence, and
exports full-domain forecasts as CF-style NetCDF.

This is an independent research prototype, not an official KMD forecasting
service or endorsed operational product.

The main entry point is [KMD_Weather_AI.ipynb](KMD_Weather_AI.ipynb). The exact
schema and limitations of the development dataset are recorded in
[DATA_AUDIT.md](DATA_AUDIT.md).

> **Research status:** the development files are model forecasts, not weather
> observations. With only two independent hourly WRF initializations, this
> repository demonstrates the complete engineering pipeline but does not
> establish operational forecast skill. Do not use its output for warnings or
> other public-safety decisions without multi-season validation against
> independent, quality-controlled observations.

## What it does

The model consumes six hourly weather grids and directly predicts +1 h, +3 h,
and +6 h fields:

```text
past 6 hourly weather grids
            │
      3-D temporal stem
            │
   learned temporal attention
            │
 U-Net encoder → spatial attention → decoder
            │
      ┌─────┼──────────┐
      │     │          │
temperature quantiles  rainfall quantiles  wet probability
      p10 / p50 / p90 at +1 h, +3 h, and +6 h
```

The pipeline includes:

- physically correct conversion of accumulated WRF precipitation to interval
  rainfall;
- relative humidity, wind speed, cyclical time, lead-time, and location
  features;
- train/validation separation by complete forecast initialization;
- probabilistic p10/p50/p90 predictions and rain-occurrence probabilities;
- persistence baselines, MAE, RMSE, bias, Brier score, POD, FAR, and CSI;
- overlap-tiled inference across the complete grid;
- compressed NetCDF export and reproducible PyTorch checkpoints; and
- deterministic seeds, early stopping, mixed precision, and gradient clipping.

## Data

Raw NetCDF files are deliberately **not included** in this repository. This
keeps large or restricted data out of Git history and avoids redistributing
files without an explicit data license.

Create a `data/` directory and place at least two independent hourly WRF files
there. Each hourly file must contain `time`, `lat`, and `lon` coordinates and
these variables:

| Variable | Meaning | Expected unit |
|---|---|---|
| `t2` | 2 m air temperature | K |
| `d2m` | 2 m dew point | K |
| `msl` | Mean sea-level pressure | Pa |
| `tcc` | Total cloud fraction | 0–1 |
| `tp` | Run-total accumulated precipitation | mm |
| `u10` | 10 m eastward wind | m s⁻¹ |
| `v10` | 10 m northward wind | m s⁻¹ |

The notebook also inventories daily or weekly NetCDF products when present,
but never treats those forecasts as observed targets. To use another location,
set `KMD_DATA_DIR` before starting Jupyter.

## Quick start

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p data
# Copy authorized NetCDF inputs into data/, then:
jupyter lab KMD_Weather_AI.ipynb
```

Run the notebook from top to bottom. It defaults to the laptop-friendly `demo`
profile. Select the `research` profile in the configuration cell for native-grid
training and a longer schedule.

Generated files are written to `artifacts/` and are ignored by Git:

- `kmd_temporal_attention_unet.pt` — weights, normalization, and configuration;
- `kmd_ai_forecast.nc` — gridded probabilistic forecasts; and
- `validation_metrics.csv` — model-versus-persistence verification.

You can override the output location with `KMD_ARTIFACT_DIR`.

## Install as a package and test

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The tests use synthetic arrays only; no private or external weather data is
required.

## Repository layout

```text
KMD_Weather_AI.ipynb   reproducible training and forecasting walkthrough
kmd_forecast/          reusable preprocessing, model, training, and inference
tests/                 synthetic unit and end-to-end tests
DATA_AUDIT.md          development-data schema and scientific caveats
```

## Operational data contract

An operational experiment should use multiple years of archived WRF cycles and
observations aligned by valid time and location. Targets must come from stations,
radar, satellite products such as CHIRPS/IMERG, or another independently
analysed source—not an aggregate of the same WRF forecast. Use rolling-origin
validation: train on earlier years, tune on a later season, and reserve the
newest season for final testing.

## Privacy and repository hygiene

Raw weather files, model artifacts, credentials, local environment files,
notebook outputs, and machine-specific paths are excluded from version control.
Before adding new material, verify that you have redistribution rights and that
notebook output does not expose names, email addresses, usernames, tokens, or
local paths.
