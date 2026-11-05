# Data audit

Audit date: 2026-11-04. The values below were read directly from the supplied NetCDF/HDF5 arrays.

> The audited raw files are not distributed in this repository. They are large
> development inputs whose redistribution terms were not established. The
> filenames and measurements below document reproducibility only.

## Product lineage

| File | Type | Initialization | Valid coverage | Grid |
|---|---|---|---|---|
| `wrf_20261028T03Z.nc` | Hourly WRF 4.2.2 forecast | 2026-10-28 03:00 UTC | 2026-10-28 03:00 to 2026-10-31 06:00 | 76 × 328 × 291 |
| `GR--20261103T0300--wrf.nc` | Hourly WRF 4.2.2 forecast | 2026-11-03 03:00 UTC | 2026-11-03 03:00 to 2026-11-06 06:00 | 76 × 328 × 291 |
| `wrf_daily_20261104T06Z.nc` | Daily WRF aggregates | 2026-11-04 06:00 UTC | Seven complete UTC days starting 2026-11-05 | 7 × 328 × 291 |
| `wrf_weekly_20261104T06Z.nc` | Seven-day WRF aggregate | 2026-11-04 06:00 UTC | 2026-11-05 00:00 to 2026-11-12 00:00 | 1 × 328 × 291 |

All four files declare CF-1.8 conventions. The hourly files are independent initialization cycles. The daily and weekly files are aggregates from a third WRF cycle; they are not observations and cannot be used as ground-truth verification.

## Geographic and temporal structure

- Latitude: approximately 5.554°S to 6.191°N, 328 rows.
- Longitude: approximately 32.672°E to 43.106°E, 291 columns.
- Nominal grid interval: about 0.036°, roughly 4 km near the equator.
- Hourly files: one-hour temporal interval and 76 valid times each.
- Every inspected variable value is finite; no missing grid cells were detected.
- The daily/weekly product history says its latitude rows were linearly resampled to a uniform axis. It must not be assumed pixel-identical to the raw grid without coordinate alignment.

## Hourly variables

| Variable | Meaning | File unit | Learning transformation |
|---|---|---|---|
| `t2` | 2 m air temperature | K | Convert to °C |
| `d2m` | 2 m dew point | K | Convert to °C; derive relative humidity |
| `msl` | Mean sea-level pressure | Pa | Convert to hPa |
| `tcc` | Total cloud fraction | 0–1 | Clip to physical range |
| `tp` | Precipitation accumulated since run start | mm | Difference in time, handle resets, clip at zero, then `log1p` |
| `u10` | 10 m eastward wind | m s⁻¹ | Standardize; combine with `v10` for speed |
| `v10` | 10 m northward wind | m s⁻¹ | Standardize; combine with `u10` for speed |

## Observed ranges inside these forecast arrays

| Initialization | Temperature range | Mean hourly rain | Maximum hourly grid-cell rain | Fraction ≥ 0.1 mm h⁻¹ | Final mean run-total rain |
|---|---:|---:|---:|---:|---:|
| 2026-10-28 03Z | −11.15 to 38.48 °C | 0.0667 mm | 90.24 mm | 3.24% | 5.07 mm |
| 2026-11-03 03Z | −11.59 to 41.42 °C | 0.0116 mm | 45.47 mm | 1.01% | 0.88 mm |

No precipitation accumulation resets were found in either hourly file. Reset handling remains in the pipeline because it is required for a robust multi-run archive. Rain is strongly zero-inflated, which motivates a separate wet/dry probability head plus quantile rainfall regression.

The very low number of independent initialization cycles is the primary limitation—not the number of grid pixels. Adjacent pixels and forecast hours are correlated and do not create independent weather cases. The notebook therefore splits by complete initialization cycle and describes its current results as proof-of-pipeline scores.
