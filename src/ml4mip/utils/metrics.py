import copy
from enum import Enum
from typing import Any

import torch
from monai.metrics import Metric


class Metric(Enum):
    DICE = "dice"
    DICE_CE = "dice_ce"
    BCE = "bce"


# TODO: define get_metrics and make it accessable from hydra


class MetricsManager:
    def __init__(
        self,
        metrics: dict[str, Metric],
        sigmoid: bool = True,
        binary: bool = True,
        binary_threshold: float = 0.5,
    ):
        """Initialize the MetricsManager with a dictionary of metric objects."""
        # check that 'loss' isn't in the metrics
        if "loss" in metrics:
            msg = (
                "The 'loss' metric is reserved for the loss function and"
                "should not be included in the metrics."
            )
            raise ValueError(msg)
        self.metrics = metrics
        self.results = {name: None for name in metrics}
        self.sigmoid = sigmoid
        self.binary = binary
        self.binary_threshold = binary_threshold

    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()

    def update(self, y_pred: torch.Tensor, y: torch.Tensor):
        """Update all metrics with the predictions and ground truth."""
        if self.sigmoid:
            y_pred = torch.sigmoid(y_pred)

        if self.binary:
            y_pred = (y_pred > self.binary_threshold).float()

        for metric in self.metrics.values():
            metric(y_pred=y_pred, y=y)

    def __call__(self, y_pred, y):
        """Update all metrics with the predictions and ground truth."""
        self.update(y_pred, y)

    def aggregate(self) -> dict[str, Any]:
        """Compute the final results for all metrics."""
        self.results = {name: metric.aggregate().item() for name, metric in self.metrics.items()}
        return self.results

    def copy(self):
        return copy.deepcopy(self)
