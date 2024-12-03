import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from hydra.core.config_store import ConfigStore
from monai.losses import DiceCELoss
from omegaconf import MISSING
from torch import optim
from torch.utils.data import DataLoader

from ml4mip import trainer
from ml4mip.dataset import DatasetConfig, get_dataset
from ml4mip.models import ModelConfig, get_model
from ml4mip.utils.logging import log_hydra_config_to_mlflow, log_metrics
from ml4mip.utils.metrics import get_metrics
from ml4mip.utils.torch import save_model
from ml4mip.visualize import visualize_model

logger = logging.getLogger(__name__)


class Mode(Enum):
    TRAIN = "train"
    EVAL = "eval"


@dataclass
class Config:
    ml_flow_uri: str = str(Path.cwd() / "runs")
    model_dir: str = MISSING
    model_tag: str = MISSING
    mode: Mode = Mode.TRAIN
    batch_size: int = 1
    lr: float = 1e-4
    num_epochs: int = 10
    model: ModelConfig = MISSING
    dataset: DatasetConfig = MISSING
    visualize_model_batches: int = 1


_cs = ConfigStore.instance()
_cs.store(
    name="base_config",
    node=Config,
)


def run_training(cfg: Config) -> None:
    """Prepare data, model, and training loop for fine-tuning."""
    logger.info("Starting model training script")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = get_dataset(cfg.dataset)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=torch.cuda.is_available()
    )

    cfg.dataset.train = False
    val_ds = get_dataset(cfg.dataset)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=torch.cuda.is_available()
    )

    msg = f"Training on {len(train_ds)} samples"
    logger.info(msg)

    # Model and optimizer
    model = get_model(cfg.model)
    model = model.to(device)

    # TODO: learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    # TODO: parameterize loss function and metric
    loss_fn = DiceCELoss(sigmoid=True)
    metrics = get_metrics()

    # Initialize MLflow
    mlflow.set_tracking_uri(cfg.ml_flow_uri)  # Update path as needed
    mlflow.set_experiment("model_training")

    try:
        with mlflow.start_run(run_name="training_run"):
            # Log configuration parameters
            log_hydra_config_to_mlflow(cfg)
            # Train the model and log metrics
            trainer.train(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                metrics=metrics,
                device=device,
                num_epochs=cfg.num_epochs,
                val_loader=val_loader,
            )

            # Save and log the final model
            save_model(
                model,
                Path(cfg.model_dir) / cfg.model_tag,
            )
            visualize_model(
                val_loader,
                model,
                device,
                n=cfg.visualize_model_batches,
                sigmoid=True,
            )
    except KeyboardInterrupt:
        # Handle manual stopping
        logger.info("Manual stopping detected. Saving model state...")
        # Save and log the final model
        save_model(
            model,
            Path(cfg.model_dir) / cfg.model_tag,
        )


def run_evaluation(cfg: Config):
    logger.info("Starting evaluation script")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.dataset.train = False
    val_ds = get_dataset(cfg.dataset)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=torch.cuda.is_available()
    )

    msg = f"Evaluation on {len(val_ds)} samples"
    logger.info(msg)

    # Model
    model = get_model(cfg.model)
    model = model.to(device)

    # TODO: parameterize loss function and metric
    loss_fn = DiceCELoss(sigmoid=True)
    metrics = get_metrics()

    # Initialize MLflow
    mlflow.set_tracking_uri(cfg.ml_flow_uri)  # Update path as needed
    mlflow.set_experiment("model_evaluation")

    with mlflow.start_run(run_name="evaluation_run"):
        # Log configuration details as parameters
        log_hydra_config_to_mlflow(cfg)

        val_result = trainer.validate(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            metrics=metrics,
            device=device,
        )
        log_metrics(
            "val",
            val_result,
            step=0,
            logger=logger,
        )
        visualize_model(
            val_loader,
            model,
            device,
            n=cfg.visualize_model_batches,
            sigmoid=True,
        )
