from logging import getLogger
from pathlib import Path

import torch

# from monai.transforms import LoadImaged, Spacingd
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, EnsureChannelFirstd, Resized, ScaleIntensityd, ToTensord
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml4mip.dataset import NiftiDataset

logger = getLogger(__name__)


# --- TRAINING FUNCTION ---
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
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
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")

    for batch in progress_bar:
        images, masks = batch
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"Batch Loss": loss.item()})
    return epoch_loss / len(train_loader)


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
    val_loss = 0.0
    dice_metric.reset()
    with torch.no_grad():
        for batch in val_loader:
            images, masks = batch["image"].to(device), batch["mask"].to(device)
            outputs = sliding_window_inference(images, (96, 96, 96), 4, model)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

            dice_metric(y_pred=outputs, y=masks)
    avg_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return {
        "val_loss": val_loss / len(val_loader),
        "avg_dice": avg_dice,
    }


# --- FINE-TUNING FUNCTION ---
def finetune(
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
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

        if val_loader is not None:
            msg = f"Epoch {epoch + 1}/{num_epochs}: validation..."
            logger.info(msg)
            # Validate
            val_loss, val_dice = validate(model, val_loader, loss_fn, dice_metric, device)

        # Scheduler step (if applicable)
        if scheduler:
            scheduler.step()

        # Print epoch results
        msg = f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}"
        if val_loader is not None:
            msg += f", Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}"
        logger.info(msg)


# --- RUN FINE-TUNING FUNCTION ---
def run_finetuning(
    data_dir: str,
    model_path: str,
    model_dir: str,
    batch_size: int = 1,
    lr: float = 1e-4,
    num_epochs: int = 50,
) -> None:
    """Prepare data, model, and training loop for fine-tuning.

    Parameters:
        data_dir: Directory containing training data.
        model_path: Path to jit model.
        model_dir: Directory to save logs and checkpoints.
        batch_size: Batch size for training.
        lr: Learning rate.
        num_epochs: Number of epochs.
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preparation

    transforms = Compose(
        [
            EnsureChannelFirstd(keys=["image", "mask"]),
            # Spacingd(keys=["image", "mask"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            ScaleIntensityd(keys=["image"]),
            Resized(
                keys=["image", "mask"],
                spatial_size=(96, 96, 96),
                mode=("trilinear", "nearest"),  # 'trilinear' for image, 'nearest' for mask
                align_corners=(True, None),  # 'align_corners' is only relevant for 'trilinear' mode
            ),
            ToTensord(keys=["image", "mask"]),
        ]
    )

    train_ds = NiftiDataset(data_dir, transforms)
    # val_ds = Dataset(data=val_data, transform=transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model and optimizer
    model = torch.jit.load(model_path, map_location=device)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    # Fine-tuning
    finetune(model, train_loader, optimizer, loss_fn, dice_metric, device, num_epochs)

    # Save the final model
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "fine_tuned_model.pt")
