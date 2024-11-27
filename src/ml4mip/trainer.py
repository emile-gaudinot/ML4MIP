import logging

import torch
from monai.inferers import sliding_window_inference
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml4mip.utils.logging import log_metrics
from ml4mip.utils.metrics import MetricsManager

logger = logging.getLogger(__name__)


# --- TRAINING FUNCTION ---
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    metrics: MetricsManager,
    device: torch.device,
) -> float:
    """Train the model for one epoch.

    Parameters:
        model: The PyTorch model to train.
        train_loader: The DataLoader for the training dataset.
        optimizer: The optimizer for training.
        loss_fn: The loss function.
        device: The device to run training on.

    Returns:
        float: The average training loss for the epoch.
    """
    model.to(device)
    model.train()
    epoch_loss = 0.0
    metrics.reset()
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")

    for batch in progress_bar:
        images, masks = batch
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        if outputs.shape != masks.shape:
            msg = f"Output shape: {outputs.shape} | Mask shape: {masks.shape}"
            logger.warning(msg)

        loss = loss_fn(outputs, masks)
        metrics(y_pred=outputs, y=masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"Batch Loss": loss.item()})

    return {
        "loss": epoch_loss / len(train_loader),
        **(metrics.aggregate()),
    }


# --- VALIDATION FUNCTION ---
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    metrics: MetricsManager,
    device: torch.device,
) -> tuple[float, float]:
    """Validate the model on the validation dataset.

    Parameters:
        model: The PyTorch model to validate.
        val_loader: The DataLoader for the validation dataset.
        loss_fn: The loss function for validation.
        dice_metric: Dice metric for validation.
        device: The device to run validation on.

    Returns:
        Average validation loss and Dice score.
    """
    model.to(device)
    model.eval()
    loss = 0.0
    metrics.reset()
    progress_bar = tqdm(val_loader, desc="Validation", unit="batch")

    with torch.no_grad():
        for batch in progress_bar:
            images, masks = batch
            images, masks = images.to(device), masks.to(device)
            outputs = sliding_window_inference(images, (96, 96, 96), 4, model)
            loss = loss_fn(outputs, masks)
            loss += loss.item()

            metrics(y_pred=outputs, y=masks)
            progress_bar.set_postfix({"Batch Loss": loss.item()})

    return {
        "loss": loss / len(val_loader),
        **(metrics.aggregate()),
    }


# --- FINE-TUNING FUNCTION ---
def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    metrics: MetricsManager,
    device: torch.device,
    num_epochs: int,
    val_loader: DataLoader | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> None:
    """Fine-tune the model for several epochs.

    Parameters:
        model: The PyTorch model to fine-tune.
        train_loader: DataLoader for training dataset.
        val_loader: DataLoader for validation dataset.
        optimizer: The optimizer for fine-tuning.
        loss_fn: Loss function.
        dice_metric: Dice metric.
        device: The device to use.
        num_epochs: Number of epochs for fine-tuning.
        scheduler: Learning rate scheduler (optional).
    """
    for epoch in range(num_epochs):
        msg = f"Epoch {epoch + 1}/{num_epochs}: training..."
        logger.info(msg)
        # Train for one epoch
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            metrics=metrics,
            loss_fn=loss_fn,
            device=device,
        )
        log_metrics("train", train_metrics, epoch, num_epochs, logger)
        if val_loader is not None:
            msg = f"Epoch {epoch + 1}/{num_epochs}: validation..."
            logger.info(msg)
            # Validate
            val_result = validate(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                metrics=metrics,
                device=device,
            )
            log_metrics("val", val_result, epoch, num_epochs, logger)

        # Scheduler step (if applicable)
        if scheduler:
            scheduler.step()
