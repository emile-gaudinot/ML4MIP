import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from monai.inferers import sliding_window_inference
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import cProfile
import pstats

from ml4mip.utils.logging import log_metrics
from ml4mip.utils.metrics import MetricsManager
from torch.profiler import profile, record_function, ProfilerActivity
from ml4mip.utils.torch import save_checkpoint

logger = logging.getLogger(__name__)

activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]

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
                    "boxes": torch.tensor([[[0, 0, 95, 95]]], device="cuda"),
                    # "point_coords": None,
                    "mask_inputs": mask[None],
                    "original_size": (96, 96),
                }
                for img, mask in zip(images, masks, strict=False)
            ]
            # del masks ?
            outputs = []
            bs = 2
            for i in range(len(images) // bs):
                single_batch_output = model(images[i : i + bs])
                outputs += [single_output["masks"][0] for single_output in single_batch_output]
            outputs = torch.stack(outputs)
            # Add the batch_size dimension
            outputs_sh = outputs.shape
            outputs = outputs.reshape(
                outputs_sh[0] // 96, 96, outputs_sh[1], outputs_sh[2], outputs_sh[3]
            )
            masks_sh = masks.shape
            masks = masks.reshape(masks_sh[0] // 96, 96, masks_sh[1], masks_sh[2], masks_sh[3])
            # Permute the z index to the end
            outputs = outputs.permute(0, 2, 3, 4, 1).to(dtype=torch.float)
            masks = masks.permute(0, 2, 3, 4, 1)

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


@dataclass
class InferenceConfig:
    mode: InferenceMode = InferenceMode.SLIDING_WINDOW
    sw_size: tuple[int, int, int] = (96, 96, 96)
    sw_batch_size: int = 4
    sw_overlap: float = 0.25
    model_input_size: tuple[int, int, int] = (96, 96, 96)


_cs = ConfigStore.instance()
_cs.store(
    name="base_inference_config",
    node=InferenceConfig,
)


def inference(
    images: torch.Tensor,
    model: nn.Module,
    cfg: InferenceConfig,
):
    match cfg.mode:
        case InferenceMode.SLIDING_WINDOW:
            # sliding_window_inference divides the input image into smaller overlapping windows
            # of the specified size (sw_size). It processes each window independently, allowing
            # for efficient inference on large images that may not fit in memory.
            # The predictions from all windows are then stitched together to reconstruct
            # the full-size output, averaging overlapping regions to ensure smooth transitions
            # and reduce artifacts.
            outputs = sliding_window_inference(
                inputs=images,
                roi_size=cfg.sw_size,
                sw_batch_size=cfg.sw_batch_size,
                predictor=model,
                overlap=cfg.sw_overlap,
            )
        case InferenceMode.RESCALE:
            # Rescale the input image to the model input size
            rescaled_images = F.interpolate(
                images,
                size=cfg.model_input_size,
                mode="trilinear",
                align_corners=False,
            )
            # Run inference on the rescaled image
            rescaled_outputs = model(rescaled_images)
            # Rescale the output back to the original image size
            original_size = images.shape[2:]  # Assuming (B, C, H, W, D) format
            # TODO: Here maybe it is better to apply sigmoid and create binary mask.
            outputs = F.interpolate(
                rescaled_outputs,
                size=original_size,
                mode="trilinear",  # TODO: debatable if trinlinear is the best choice / maybe nearest
                align_corners=False,
            )
        case _:
            outputs = model(images)

    return outputs


# --- VALIDATION FUNCTION ---
@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    metrics: MetricsManager,
    device: torch.device,
    inference_cfg: InferenceConfig,
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
        outputs = inference(
            images=images,
            model=model,
            cfg=inference_cfg,
        )
        loss = loss_fn(outputs, masks)
        val_loss += loss.item()
        metrics(y_pred=outputs, y=masks)
        progress_bar.set_postfix({"Batch Loss": loss.item()})

    return {
        "loss": val_loss / len(val_loader),
        **(metrics.aggregate()),
    }


# --- TRAINING FUNCTION ---
def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    metrics: MetricsManager,
    device: torch.device,
    current_epoch: int,
    num_epochs: int,
    inference_cfg: InferenceConfig,
    checkpoint_dir: str | Path,
    val_loader: DataLoader | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    model_type: str | None = None,
    val_inference_mode: InferenceMode = InferenceMode.SLIDING_WINDOW,
    val_sw_size: int = 96,
    val_sw_batch_size: int = 4,
    val_sw_overlap: float = 0.25,
    val_model_input_size: tuple[int, int, int] = (96, 96, 96),
    torch_profiling: bool = False,
    cpython_profiling: bool = False,
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
        inference_cfg: Inference configuration.
        checkpoint_dir: Directory to save checkpoints.
        val_loader: DataLoader for validation dataset (optional).
        scheduler: Learning rate scheduler (optional).
    """
    global_batch_idx = 0
    for epoch in range(current_epoch, num_epochs):
        msg = f"Epoch {epoch + 1}/{num_epochs}: training..."
        logger.info(msg)
        print(device)

        if torch_profiling:
            with profile(activities=activities, record_shapes=True) as prof:
                with record_function("batch"):
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
            
            print(prof.key_averages().table())
            prof.export_chrome_trace(f"trace_epoch_{epoch}.json")
        elif cpython_profiling:
            with cProfile.Profile() as pr:
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
            stats = pstats.Stats(pr)
            stats.dump_stats(f"cpython_trace_epoch_{epoch}.prof")

        else:
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
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            checkpoint_dir=checkpoint_dir,
        )
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
                inference_cfg=inference_cfg,
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
