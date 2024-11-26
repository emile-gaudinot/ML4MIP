import logging

import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow

logger = logging.getLogger(__name__)


# --- TRAINING FUNCTION ---
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    dice_metric: DiceMetric,
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
    dice_metric.reset()
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")

    for batch in progress_bar:
        images, masks = batch
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        if outputs.shape != masks.shape:
            msg = f"Output shape: {outputs.shape} | Mask shape: {masks.shape}"
            logger.warning(msg)

        loss = loss_fn(outputs, masks)
        dice_metric(y_pred=outputs, y=masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"Batch Loss": loss.item()})

    avg_dice = dice_metric.aggregate().item()
    dice_metric.reset()

    return {
        "loss": epoch_loss / len(train_loader),
        "avg_dice": avg_dice,
    }


# --- VALIDATION FUNCTION ---
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    dice_metric: DiceMetric,
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
    dice_metric.reset()
    progress_bar = tqdm(val_loader, desc="Validation", unit="batch")

    with torch.no_grad():
        for batch in progress_bar:
            images, masks = batch
            images, masks = images.to(device), masks.to(device)
            outputs = sliding_window_inference(images, (96, 96, 96), 4, model)
            loss = loss_fn(outputs, masks)
            loss += loss.item()

            dice_metric(y_pred=outputs, y=masks)
            progress_bar.set_postfix({"Batch Loss": loss.item()})

    avg_dice = dice_metric.aggregate().item()
    dice_metric.reset()

    return {
        "loss": loss / len(val_loader),
        "avg_dice": avg_dice,
    }


# --- FINE-TUNING FUNCTION ---
def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    dice_metric: DiceMetric,
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
        train_result = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            dice_metric=dice_metric,
            loss_fn=loss_fn,
            device=device,
        )
        mlflow.log_metric("train_loss", train_result["loss"], step=epoch + 1)
        mlflow.log_metric("train_dice", train_result["avg_dice"], step=epoch + 1)
        msg = (
            f"Epoch {epoch + 1}/{num_epochs}: train_loss={train_result['loss']:.4f}, "
            f"train_dice={train_result['avg_dice']:.4f}"
        )
        logger.info(msg)
        if val_loader is not None:
            msg = f"Epoch {epoch + 1}/{num_epochs}: validation..."
            logger.info(msg)
            # Validate
            val_result = validate(model, val_loader, loss_fn, dice_metric, device)
            mlflow.log_metric("val_loss", val_result["loss"], step=epoch + 1)
            mlflow.log_metric("val_dice", val_result["avg_dice"], step=epoch + 1)
            msg = (
                f"Epoch {epoch + 1}/{num_epochs}: val_loss={val_result['loss']:.4f}, "
                f"val_dice={val_result['avg_dice']:.4f}"
            )

        # Scheduler step (if applicable)
        if scheduler:
            scheduler.step()
