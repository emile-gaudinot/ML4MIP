import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from hydra.core.config_store import ConfigStore
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from omegaconf import MISSING, OmegaConf
from torch import optim
from torch.utils.data import DataLoader

from ml4mip import trainer
from ml4mip.dataset import DatasetConfig, get_dataset
from ml4mip.models import ModelConfig, get_model
from ml4mip.utils.torch import save_model

logger = logging.getLogger(__name__)


class Mode(Enum):
    TRAIN = "train"
    EVAL = "eval"


@dataclass
class ConfigBase:
    ml_flow_uri: str = str(Path.cwd() / "runs")
    model_dir: str = MISSING
    model_tag: str = MISSING
    mode: Mode = MISSING


@dataclass
class TrainingConfig(ConfigBase):
    batch_size: int = 1
    lr: float = 1e-4
    num_epochs: int = 10
    model: ModelConfig = MISSING
    dataset: DatasetConfig = MISSING
    mode: Mode = Mode.TRAIN


_cs = ConfigStore.instance()
_cs.store(
    name="base_config",
    node=ConfigBase,
)
_cs.store(
    name="base_train",
    node=TrainingConfig,
)


def run_training(cfg: TrainingConfig) -> None:
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
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    # Initialize MLflow
    mlflow.set_tracking_uri(cfg.ml_flow_uri)  # Update path as needed
    mlflow.set_experiment("model_training")

    with mlflow.start_run(run_name="training_run"):
        # Log configuration parameters
        mlflow.log_param("batch_size", cfg.batch_size)
        mlflow.log_param("learning_rate", cfg.lr)
        mlflow.log_param("num_epochs", cfg.num_epochs)
        mlflow.log_dict(OmegaConf.to_yaml(cfg), "training_config.yaml")

        # Train the model and log metrics
        trainer.train(
            model,
            train_loader,
            optimizer,
            loss_fn,
            dice_metric,
            device,
            cfg.num_epochs,
            val_loader=val_loader,
        )

        # Save and log the final model
        model_dir = Path(cfg.model_dir)
        save_model(model, model_dir / cfg.model_tag)

        try:
            mlflow.log_artifact(model_dir / cfg.model_tag)
        except RuntimeError as e:
            msg = f"Failed to log model: {e}"
            logger.exception(msg)


@dataclass
class EvaluationConfig(ConfigBase):
    batch_size: int = 1
    model: ModelConfig = MISSING
    dataset: DatasetConfig = MISSING
    mode: Mode = Mode.EVAL


_cs.store(
    name="base_eval",
    node=EvaluationConfig,
)


def run_evaluation(cfg: EvaluationConfig):
    logger.info("Starting unetr evaluation script")

    # Initialize MLflow
    mlflow.set_tracking_uri(cfg.ml_flow_uri)  # Change to your desired path
    mlflow.set_experiment("model_evaluation")

    with mlflow.start_run(run_name="evaluation_run"):
        # Log configuration details as parameters
        mlflow.log_param("batch_size", cfg.batch_size)
        mlflow.log_dict(cfg.model, "model_config.yaml")
        mlflow.log_dict(cfg.dataset, "dataset_config.yaml")

        cfg.dataset.train = False
        val_ds = get_dataset(cfg.dataset)
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=torch.cuda.is_available()
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        msg = f"Evaluating on {len(val_ds)} samples"
        logger.info(msg)

        # Model and optimizer
        model = get_model(cfg.model)
        model = model.to(device)

        loss_fn = DiceCELoss(sigmoid=True)
        dice_metric = DiceMetric(include_background=True, reduction="mean")

        # Evaluation
        result = trainer.validate(model, val_loader, loss_fn, dice_metric, device)

        # Log metrics to MLflow
        mlflow.log_metric("validation_dice", result["avg_dice"])
        mlflow.log_metric("validation_loss", result["val_loss"])

        # Log model weights (optional)
        mlflow.pytorch.log_model(model, "model")

        msg = (
            f"Validation Dice: {result['avg_dice']:.4f} | Validation Loss: {result['val_loss']:.4f}"
        )
        logger.info(msg)
