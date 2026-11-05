"""Configuration profiles shared by the notebook and training pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ForecastConfig:
    """All parameters needed to reproduce a training run."""

    name: str
    history_hours: int = 6
    horizons: tuple[int, ...] = (1, 3, 6)
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    spatial_stride: int = 2
    patch_size: int = 64
    tile_overlap: int = 16
    base_channels: int = 16
    dropout: float = 0.10
    batch_size: int = 8
    epochs: int = 4
    patience: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    train_patches_per_anchor: int = 2
    validation_patches_per_anchor: int = 1
    rain_threshold_mm: float = 0.1
    num_workers: int = 0
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


PROFILES: dict[str, ForecastConfig] = {
    "demo": ForecastConfig(name="demo"),
    "research": ForecastConfig(
        name="research",
        spatial_stride=1,
        patch_size=96,
        tile_overlap=24,
        base_channels=32,
        dropout=0.15,
        batch_size=6,
        epochs=30,
        patience=6,
        learning_rate=2e-4,
        train_patches_per_anchor=6,
        validation_patches_per_anchor=3,
    ),
}


def get_profile(name: str) -> ForecastConfig:
    """Return a named immutable configuration profile."""

    if name not in PROFILES:
        choices = ", ".join(sorted(PROFILES))
        raise KeyError(f"Unknown profile {name!r}. Choose one of: {choices}")
    return PROFILES[name]
