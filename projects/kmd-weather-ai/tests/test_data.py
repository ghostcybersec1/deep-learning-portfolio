import numpy as np

from kmd_forecast.data import (
    INPUT_FEATURE_NAMES,
    SpatioTemporalPatchDataset,
    accumulation_to_increment,
    fit_normalizer,
    make_model_input,
    relative_humidity_fraction,
)


def test_accumulation_to_increment_handles_resets() -> None:
    accumulated = np.array([0.0, 1.0, 3.0, 0.5, 2.0], dtype=np.float32)
    expected = np.array([0.0, 1.0, 2.0, 0.5, 1.5], dtype=np.float32)
    np.testing.assert_allclose(accumulation_to_increment(accumulated), expected)


def test_relative_humidity_is_bounded() -> None:
    temperature = np.array([20.0, 30.0], dtype=np.float32)
    dewpoint = np.array([20.0, 10.0], dtype=np.float32)
    humidity = relative_humidity_fraction(temperature, dewpoint)
    assert humidity[0] == 1.0
    assert np.all((humidity >= 0.0) & (humidity <= 1.0))


def test_model_input_and_patch_dataset(synthetic_run) -> None:
    normalizer = fit_normalizer([synthetic_run])
    model_input = make_model_input(
        synthetic_run,
        anchor=5,
        history_hours=3,
        normalizer=normalizer,
        y_slice=slice(0, 8),
        x_slice=slice(0, 8),
    )
    assert model_input.shape == (len(INPUT_FEATURE_NAMES), 3, 8, 8)
    assert np.isfinite(model_input).all()

    dataset = SpatioTemporalPatchDataset(
        [synthetic_run],
        normalizer,
        history_hours=3,
        horizons=(1, 2),
        patch_size=8,
        patches_per_anchor=1,
        rain_threshold_mm=0.1,
        training=False,
    )
    sample = dataset[0]
    assert tuple(sample["x"].shape) == (len(INPUT_FEATURE_NAMES), 3, 8, 8)
    assert tuple(sample["temperature_c"].shape) == (2, 8, 8)
    assert tuple(sample["wet"].shape) == (2, 8, 8)
