import torch

from kmd_forecast.data import INPUT_FEATURE_NAMES
from kmd_forecast.model import MultiTaskQuantileLoss, TemporalAttentionUNet


def test_model_outputs_and_loss_are_well_formed() -> None:
    model = TemporalAttentionUNet(
        input_channels=len(INPUT_FEATURE_NAMES),
        horizons=(1, 3),
        quantiles=(0.1, 0.5, 0.9),
        base_channels=4,
    )
    inputs = torch.randn(2, len(INPUT_FEATURE_NAMES), 3, 16, 16)
    output = model(inputs)

    assert output["temperature_quantiles"].shape == (2, 2, 3, 16, 16)
    assert output["rain_quantiles"].shape == (2, 2, 3, 16, 16)
    assert output["wet_logits"].shape == (2, 2, 16, 16)

    batch = {
        "temperature_z": torch.randn(2, 2, 16, 16),
        "rain_log_z": torch.randn(2, 2, 16, 16),
        "wet": torch.randint(0, 2, (2, 2, 16, 16), dtype=torch.float32),
    }
    loss, components = MultiTaskQuantileLoss(
        quantiles=(0.1, 0.5, 0.9), wet_positive_weight=2.0
    )(output, batch)
    assert torch.isfinite(loss)
    assert set(components) == {"temperature", "rain", "wet", "crossing"}
