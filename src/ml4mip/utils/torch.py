import logging
from pathlib import Path

import mlflow
import torch
from torch import nn

logger = logging.getLogger(__name__)


def detect_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_module_device(module: nn.Module) -> torch.device:
    """Returns the module's device.

    This function requires all submodules to be on the same device.

    Returns:
        The module's device.
    """
    return next(module.parameters()).device


def save_model(model, model_path: str | Path):
    # ensure that the directory exists
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # check if it has a .pth extension
    if model_path.suffix != ".pt":
        model_path = model_path.with_suffix(".pt")

    # check if the model file exists and create a new name
    while model_path.exists():
        parts = model_path.stem.split("_")
        if len(parts) > 1 and parts[-1].isdigit():
            parts[-1] = str(int(parts[-1]) + 1)
        else:
            parts.append("1")

        new_name = "_".join(parts)
        # create a new file name
        model_path = model_path.with_name(new_name + model_path.suffix)

    torch.save(model.state_dict(), model_path)
    msg = f"Model saved to {model_path}"
    logger.info(msg)

    # check if there is an mlflow context
    if mlflow.active_run():
        try:
            mlflow.log_artifact(str(model_path))
        except RuntimeError as e:
            msg = f"Failed to log model: {e}"
            logger.exception(msg)


def save_checkpoint(model, optimizer, epoch, checkpoint_dir: str | Path, scheduler=None):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    path = checkpoint_dir / f"checkpoint_{epoch}.pt"
    if path.exists():
        msg = f"Checkpoint {path} already exists. Overwriting..."
        logger.warning(msg)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
        },
        path,
    )


# Load checkpoint
def load_checkpoint(model, optimizer, checkpoint_dir: str | Path, scheduler=None) -> int:
    checkpoint_dir = Path(checkpoint_dir)
    # Find the latest checkpoint
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        msg = f"No checkpoints found in {checkpoint_dir}"
        logger.info(msg)
        return 0

    path = max(checkpoints, key=lambda fname: fname.stat().st_mtime)
    msg = f"Loading checkpoint from {path}"
    logger.info(msg)

    checkpoint = torch.load(path, map_location=detect_device())
    required_keys = ["model_state_dict", "optimizer_state_dict", "epoch"]
    for key in required_keys:
        if key not in checkpoint:
            raise ValueError(f"Checkpoint {path} is missing key: {key}")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"]  # Start training from the next epoch


def load_model(model_path: str | Path):
    return torch.load(model_path)


def compare_models(model1: nn.Module, model2: nn.Module) -> bool:
    """Checks if two PyTorch models have identical architectures based on state_dict keys."""
    return list(model1.state_dict().keys()) == list(model2.state_dict().keys())
