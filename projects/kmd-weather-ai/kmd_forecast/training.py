"""Reproducible PyTorch training utilities for notebook use."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import INPUT_FEATURE_NAMES, Normalizer


def set_reproducible(seed: int = 42) -> None:
    """Set common pseudo-random generators without forcing slow deterministic kernels."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
    description: str = "epoch",
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "temperature": 0.0, "rain": 0.0, "wet": 0.0, "crossing": 0.0}
    seen = 0
    amp_enabled = device.type == "cuda"
    iterator = tqdm(loader, desc=description, leave=False)

    for batch in iterator:
        batch = _move_batch(batch, device)
        batch_size = int(batch["x"].shape[0])
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                prediction = model(batch["x"])
                loss, parts = criterion(prediction, batch)

            if training:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        totals["loss"] += float(loss.detach()) * batch_size
        for key, value in parts.items():
            totals[key] += value * batch_size
        seen += batch_size
        iterator.set_postfix(loss=f"{totals['loss'] / seen:.4f}")

    return {key: value / max(seen, 1) for key, value in totals.items()}


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
) -> pd.DataFrame:
    """Train with AdamW, one-cycle scheduling, clipping, AMP, and early stopping."""

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    epochs_without_improvement = 0
    rows: list[dict] = []

    for epoch in range(1, epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        train_stats = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            description=f"train {epoch:02d}/{epochs:02d}",
        )
        with torch.no_grad():
            validation_stats = _run_epoch(
                model,
                validation_loader,
                criterion,
                device,
                description=f"valid {epoch:02d}/{epochs:02d}",
            )
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value for key, value in train_stats.items()})
        row.update({f"validation_{key}": value for key, value in validation_stats.items()})
        rows.append(row)

        if validation_stats["loss"] < best_validation - 1e-5:
            best_validation = validation_stats["loss"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return pd.DataFrame(rows)


def save_checkpoint(
    path_like: str | Path,
    model: nn.Module,
    normalizer: Normalizer,
    config: dict,
    history: pd.DataFrame,
    extra: dict | None = None,
) -> Path:
    """Save everything required to reproduce inference, not just model weights."""

    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "normalizer": normalizer.to_dict(),
        "config": config,
        "input_features": list(INPUT_FEATURE_NAMES),
        "training_history": history.to_dict(orient="list"),
        "torch_version": torch.__version__,
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path_like: str | Path, device: torch.device | str = "cpu") -> dict:
    """Load a trusted, locally produced checkpoint."""

    return torch.load(Path(path_like), map_location=device, weights_only=False)
