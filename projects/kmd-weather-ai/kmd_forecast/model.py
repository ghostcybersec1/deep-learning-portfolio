"""Temporal-attention U-Net and probabilistic multi-task weather loss."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.residual = ResidualBlock(out_channels, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(self.down(x))


class DecoderStage(nn.Module):
    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int, dropout: float
    ) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1)
        self.residual = ResidualBlock(out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.residual(self.project(torch.cat((x, skip), dim=1)))


class SpatialSelfAttention(nn.Module):
    """Global attention at the compressed U-Net bottleneck."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        heads = 4 if channels % 4 == 0 else 1
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normalized = self.norm(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.feed_forward(self.output_norm(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class TemporalAttentionUNet(nn.Module):
    """Direct multi-horizon probabilistic forecast model.

    Input shape is ``[batch, feature, history, latitude, longitude]``. Output
    temperature and rainfall tensors have shape
    ``[batch, horizon, quantile, latitude, longitude]``; wet probability logits
    have shape ``[batch, horizon, latitude, longitude]``.
    """

    def __init__(
        self,
        input_channels: int,
        horizons: Sequence[int] = (1, 3, 6),
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        base_channels: int = 24,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizons = tuple(int(value) for value in horizons)
        self.quantiles = tuple(float(value) for value in quantiles)
        self.temporal_stem = nn.Sequential(
            nn.Conv3d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(base_channels), base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(base_channels), base_channels),
            nn.SiLU(),
        )
        self.temporal_score = nn.Conv3d(base_channels, 1, kernel_size=1)

        self.enc1 = ResidualBlock(base_channels, dropout)
        self.enc2 = EncoderStage(base_channels, base_channels * 2, dropout)
        self.enc3 = EncoderStage(base_channels * 2, base_channels * 4, dropout)
        self.bottleneck_down = EncoderStage(base_channels * 4, base_channels * 8, dropout)
        self.bottleneck_attention = SpatialSelfAttention(base_channels * 8, dropout)

        self.dec3 = DecoderStage(base_channels * 8, base_channels * 4, base_channels * 4, dropout)
        self.dec2 = DecoderStage(base_channels * 4, base_channels * 2, base_channels * 2, dropout)
        self.dec1 = DecoderStage(base_channels * 2, base_channels, base_channels, dropout)
        output_channels = len(self.horizons) * len(self.quantiles)
        self.temperature_head = nn.Conv2d(base_channels, output_channels, kernel_size=1)
        self.rain_head = nn.Conv2d(base_channels, output_channels, kernel_size=1)
        self.wet_head = nn.Conv2d(base_channels, len(self.horizons), kernel_size=1)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,T,H,W], received {tuple(x.shape)}")
        temporal = self.temporal_stem(x)
        weights = torch.softmax(self.temporal_score(temporal), dim=2)
        fused = torch.sum(temporal * weights, dim=2)

        e1 = self.enc1(fused)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        bottleneck = self.bottleneck_attention(self.bottleneck_down(e3))
        decoded = self.dec1(self.dec2(self.dec3(bottleneck, e3), e2), e1)

        batch, _, height, width = decoded.shape
        shape = (batch, len(self.horizons), len(self.quantiles), height, width)
        return {
            "temperature_quantiles": self.temperature_head(decoded).view(shape),
            "rain_quantiles": self.rain_head(decoded).view(shape),
            "wet_logits": self.wet_head(decoded),
        }

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class MultiTaskQuantileLoss(nn.Module):
    """Temperature/rain quantile loss plus a zero-inflated wet/dry classifier."""

    def __init__(
        self,
        quantiles: Sequence[float],
        wet_positive_weight: float,
        temperature_weight: float = 1.0,
        rain_weight: float = 0.7,
        wet_weight: float = 0.3,
        crossing_weight: float = 0.05,
    ) -> None:
        super().__init__()
        quantile_tensor = torch.tensor(tuple(quantiles), dtype=torch.float32)
        self.register_buffer("quantiles", quantile_tensor.view(1, 1, -1, 1, 1))
        self.register_buffer(
            "positive_weight", torch.tensor(float(wet_positive_weight), dtype=torch.float32)
        )
        self.temperature_weight = temperature_weight
        self.rain_weight = rain_weight
        self.wet_weight = wet_weight
        self.crossing_weight = crossing_weight

    def _pinball(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = target.unsqueeze(2) - prediction
        return torch.maximum((self.quantiles - 1.0) * error, self.quantiles * error).mean()

    @staticmethod
    def _crossing_penalty(prediction: torch.Tensor) -> torch.Tensor:
        if prediction.shape[2] < 2:
            return prediction.new_zeros(())
        return F.relu(prediction[:, :, :-1] - prediction[:, :, 1:]).mean()

    def forward(
        self, prediction: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        temperature_loss = self._pinball(
            prediction["temperature_quantiles"], batch["temperature_z"]
        )
        rain_loss = self._pinball(prediction["rain_quantiles"], batch["rain_log_z"])
        wet_loss = F.binary_cross_entropy_with_logits(
            prediction["wet_logits"], batch["wet"], pos_weight=self.positive_weight
        )
        crossing_loss = self._crossing_penalty(
            prediction["temperature_quantiles"]
        ) + self._crossing_penalty(prediction["rain_quantiles"])
        total = (
            self.temperature_weight * temperature_loss
            + self.rain_weight * rain_loss
            + self.wet_weight * wet_loss
            + self.crossing_weight * crossing_loss
        )
        parts = {
            "temperature": float(temperature_loss.detach()),
            "rain": float(rain_loss.detach()),
            "wet": float(wet_loss.detach()),
            "crossing": float(crossing_loss.detach()),
        }
        return total, parts


def model_size_megabytes(model: nn.Module) -> float:
    """Approximate parameter storage using each tensor's native dtype."""

    bytes_used = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    return bytes_used / math.pow(1024, 2)
