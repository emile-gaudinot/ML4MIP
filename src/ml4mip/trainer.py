import logging
from enum import Enum

import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Resize
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
    batch_idx: int,
    model_type: str | None = None,
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
    batch_metric = metrics.copy()
    batch_metric.reset()
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")

    for batch in progress_bar:
        images, masks = batch
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        # TODO: Create Model Wrapper for MedSam and move the following logic there
        # Forward pass
        if model_type == "medsam":
            # Reshaping images: treat each z-slice image independantly
            images = images.permute(0, 4, 1, 2, 3)
            sh = images.shape
            images = images.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4])
            # Same for masks
            masks = masks.permute(0, 4, 1, 2, 3)
            sh = masks.shape
            masks = masks.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4])
            # Create "image", "boxes", "point_coords", "mask_inputs" and
            # "original_size" attributes to 'x'
            images = [
                {
                    "image": img,
                    "boxes": torch.tensor([0, 0, 96, 96], device="cuda"),
                    # "point_coords": None,
                    "mask_inputs": mask,
                    "original_size": None,  # torch.Tensor([96, 96]),
                }
                for img, mask in zip(images, masks, strict=False)
            ]
            # del masks ?
            output = []
            bs = 2
            for i in range(len(images) // bs):
                single_output = model(images[i : i + bs])
                output.append(single_output)
            output = torch.tensor(output)

        else:
            outputs = model(images)

        if outputs.shape != masks.shape:
            msg = f"Output shape: {outputs.shape} | Mask shape: {masks.shape}"
            logger.warning(msg)

        loss = loss_fn(outputs, masks)
        metrics(y_pred=outputs, y=masks)
        batch_metric(y_pred=outputs, y=masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_metrics = {
            "loss": loss.item(),
            **(batch_metric.aggregate()),
        }
        log_metrics(
            "train_batch",
            batch_metrics,
            step=batch_idx,
            logger=logger,
        )
        batch_idx += 1
        batch_metric.reset()
        progress_bar.set_postfix({"Batch Loss": loss.item()})

    return {
        "loss": epoch_loss / len(train_loader),
        **(metrics.aggregate()),
    }


class InferenceMode(Enum):
    SLIDING_WINDOW = "sliding_window"
    RESCALE = "rescale"
    STD = "standard"


# --- VALIDATION FUNCTION ---
@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    metrics: MetricsManager,
    device: torch.device,
    inference_mode: InferenceMode = InferenceMode.SLIDING_WINDOW,
    sw_size: int = 96,
    sw_batch_size: int = 4,
    sw_overlap: float = 0.25,
    model_input_size: tuple[int, int, int] = (96, 96, 96),
) -> tuple[float, float]:
    """Validate the model on the validation dataset.

    Parameters:
        model: The PyTorch model to validate.
        val_loader: The DataLoader for the validation dataset.
        loss_fn: The loss function for validation.
        dice_metric: Dice metric for validation.
        device: The device to run validation on.
        inference_mode: Inference mode. Default is InferenceMode.SLIDING_WINDOW.
        sw_size: Sliding window size for inference. If not None, use sliding window inference.
        sw_batch_size: Sliding window batch size. Default is 4.
        model_input_size: Used for rescale inference. Doesn't require Channel Dimension. (H, W, D)

    Returns:
        Average validation loss and Dice score.
    """
    model.to(device)
    model.eval()
    val_loss = 0.0
    metrics.reset()
    progress_bar = tqdm(val_loader, desc="Validation", unit="batch")

    for batch in progress_bar:
        images, masks = batch
        images, masks = images.to(device), masks.to(device)
        match inference_mode:
            case InferenceMode.SLIDING_WINDOW:
                # sliding_window_inference divides the input image into smaller overlapping windows
                # of the specified size (sw_size). It processes each window independently, allowing
                # for efficient inference on large images that may not fit in memory.
                # The predictions from all windows are then stitched together to reconstruct
                # the full-size output, averaging overlapping regions to ensure smooth transitions
                # and reduce artifacts.
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=(sw_size, sw_size, sw_size),
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=sw_overlap,
                )
            case InferenceMode.RESCALE:
                # Rescale the input image to the model input size
                resize_transform = Resize(spatial_size=model_input_size)
                rescaled_images = resize_transform(images)

                # Run inference on the rescaled image
                rescaled_outputs = model(rescaled_images)

                # Rescale the output back to the original image size
                original_size = images.shape[2:]  # Assuming (B, C, H, W, D) format
                resize_back_transform = Resize(spatial_size=original_size)
                outputs = resize_back_transform(rescaled_outputs)
            case _:
                outputs = model(images)
        loss = loss_fn(outputs, masks)
        val_loss += loss.item()
        metrics(y_pred=outputs, y=masks)
        progress_bar.set_postfix({"Batch Loss": loss.item()})

    return {
        "loss": val_loss / len(val_loader),
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
    model_type: str | None = None,
    val_inference_mode: InferenceMode = InferenceMode.SLIDING_WINDOW,
    val_sw_size: int = 96,
    val_sw_batch_size: int = 4,
    val_sw_overlap: float = 0.25,
    val_model_input_size: tuple[int, int, int] = (96, 96, 96),
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
        val_inference_mode: Inference mode for validation. Default is InferenceMode.SLIDING_WINDOW.
        val_sw_size: Sliding window size for validation inference. Default is 96.
        val_sw_batch_size: Sliding window batch size. Default is 4.
        val_sw_overlap: Sliding window overlap. Default is 0.25.
        val_model_input_size: Input size for rescale inference. Default is (96, 96, 96).
    """
    global_batch_idx = 0
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
            batch_idx=global_batch_idx,
            model_type=model_type,
        )
        global_batch_idx += len(train_loader)

        log_metrics(
            "train",
            train_metrics,
            step=epoch,
            epochs=(
                epoch,
                num_epochs,
            ),
            logger=logger,
        )
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
                inference_mode=val_inference_mode,
                sw_size=val_sw_size,
                sw_batch_size=val_sw_batch_size,
                sw_overlap=val_sw_overlap,
                model_input_size=val_model_input_size,
            )
            log_metrics(
                "val",
                val_result,
                step=epoch,
                epochs=(
                    epoch,
                    num_epochs,
                ),
                logger=logger,
            )

        # Scheduler step (if applicable)
        if scheduler:
            scheduler.step()
