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


def load_model(model_path: str | Path):
    return torch.load(model_path)


def compare_models(model1: nn.Module, model2: nn.Module) -> bool:
    """Checks if two PyTorch models have identical architectures based on state_dict keys."""
    return list(model1.state_dict().keys()) == list(model2.state_dict().keys())
