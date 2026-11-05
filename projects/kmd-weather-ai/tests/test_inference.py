import numpy as np
import torch
import xarray as xr

from kmd_forecast.data import fit_normalizer
from kmd_forecast.inference import _tile_starts, export_forecast_netcdf, predict_full_grid


class ConstantModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, _, _, height, width = inputs.shape
        quantile_shape = (batch, 2, 3, height, width)
        return {
            "temperature_quantiles": inputs.new_zeros(quantile_shape),
            "rain_quantiles": inputs.new_zeros(quantile_shape),
            "wet_logits": inputs.new_zeros((batch, 2, height, width)),
        }


def test_tile_starts_cover_trailing_edge() -> None:
    assert _tile_starts(length=18, tile_size=8, overlap=2) == [0, 6, 10]


def test_tiled_prediction_and_netcdf_export(synthetic_run, tmp_path) -> None:
    normalizer = fit_normalizer([synthetic_run])
    result = predict_full_grid(
        ConstantModel(),
        synthetic_run,
        normalizer,
        history_hours=3,
        horizons=(1, 2),
        quantiles=(0.1, 0.5, 0.9),
        tile_size=8,
        overlap=2,
        device=torch.device("cpu"),
        anchor=8,
        tile_batch_size=3,
    )
    assert result.temperature_c.shape == (2, 3, 16, 18)
    assert result.rain_mm.shape == (2, 3, 16, 18)
    np.testing.assert_allclose(result.wet_probability, 0.5, atol=1e-6)

    output = export_forecast_netcdf(result, tmp_path / "forecast.nc")
    with xr.open_dataset(output) as dataset:
        assert dataset["temperature_2m"].shape == (2, 3, 16, 18)
        assert dataset.attrs["Conventions"] == "CF-1.8"
