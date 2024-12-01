from logging import Logger

import mlflow
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


def get_log_message(
    label: str,
    metrics: dict[str, float],
    epochs: tuple[int, int] | None = None,
) -> str:
    """Create a formatted log message for training or validation metrics."""
    # Format the epoch information

    # Format the metrics
    metrics_info = ", ".join(f"{label}_{name}={value:.4f}" for name, value in metrics.items())

    if epochs:
        return f"Epoch {epochs[0] + 1}/{epochs[1]}: {metrics_info}"

    return metrics_info


def mlflow_log_metrics(label: str, metrics: dict[str, float], step: int) -> None:
    """Log metrics to MLflow with a specific label."""
    for name, value in metrics.items():
        mlflow.log_metric(f"{label}_{name}", value, step=step)


def log_metrics(
    label: str,
    metrics: dict[str, float],
    step: int,
    epochs: tuple[int, int] | None = None,
    logger: Logger | None = None,
) -> None:
    """Log metrics to the console and MLflow."""
    mlflow_log_metrics(label, metrics, step)
    if logger is not None:
        msg = get_log_message(
            label,
            metrics,
            epochs=epochs,
        )
        logger.info(msg)


def log_hydra_config_to_mlflow(config: DictConfig, prefix: str = "") -> None:
    """Log every parameter from a Hydra DictConfig object to MLflow."""

    def flatten_structure(d, parent_key="", sep="."):
        """Flatten nested dictionaries and lists into a single-level dictionary."""
        items = []
        if isinstance(d, dict):
            # Handle dictionary case
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                items.extend(flatten_structure(v, new_key, sep=sep).items())
        elif isinstance(d, list):
            # Handle list case
            for i, v in enumerate(d):
                new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                items.extend(flatten_structure(v, new_key, sep=sep).items())
        else:
            # Base case: scalar value
            items.append((parent_key, d))
        return dict(items)

    # Convert DictConfig to a standard dictionary
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Flatten the structure
    flat_config = flatten_structure(config_dict)

    # Log each parameter to MLflow
    for key, value in flat_config.items():
        # Convert non-stringable objects to strings for logging
        if not isinstance(value, str | int | float | bool):
            value = str(value)
        mlflow.log_param(f"{prefix}{key}", value)
