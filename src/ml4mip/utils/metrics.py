import copy
import statistics
from enum import Enum
from typing import Any

import numpy as np
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, Metric
from skimage.morphology import skeletonize


class MetricType(Enum):
    DICE = "dice"
    JACCARD_INDEX = "jaccard_index"
    HAUSDORFF_DISTANCE = "hausdorff_distance"
    CENTERLINE_LOSS = "centerline_loss"


class JaccardIndex(Metric):
    def __init__(self):
        super().__init__()
        self.values = []

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the Jaccard Index for a batch and store it."""
        # Initialize DiceLoss to compute Jaccard Index
        jaccard_loss = DiceLoss(jaccard=True, include_background=True, reduction="mean")

        # Compute the Jaccard Index
        jaccard_index = 1 - jaccard_loss(y_pred, y)

        # Append to internal state
        self.values.append(jaccard_index)
        return jaccard_index

    def aggregate(self) -> torch.Tensor:
        """Aggregate the stored Jaccard Index values."""
        if len(self.values) == 0:
            msg = "No values have been computed. Call the metric before aggregating."
            raise ValueError(msg)
        # Compute the mean of the stored values
        return torch.mean(torch.stack(self.values))

    def reset(self):
        """Reset the stored values."""
        self.values = []


class ClDiceMetric(Metric):
    def __init__(self):
        """Initialize the clDice metric."""
        super().__init__()
        self.cldice_scores = []

    @staticmethod
    def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert a PyTorch tensor to a NumPy array."""
        return tensor.detach().cpu().numpy().astype(bool)

    @staticmethod
    def _cl_score(v: np.ndarray, s: np.ndarray) -> float:
        """Compute the skeleton volume overlap."""
        return np.sum(v * s) / (np.sum(s) + 1e-8)

    @staticmethod
    def _clDice_single(pred: np.ndarray, target: np.ndarray) -> float:
        """Compute clDice for a single prediction-target pair."""
        tprec = ClDiceMetric._cl_score(pred, skeletonize(target))
        tsens = ClDiceMetric._cl_score(target, skeletonize(pred))
        return 2 * tprec * tsens / (tprec + tsens + 1e-8)

    def update(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the clDice metric for the given predictions and ground truths.

        Args:
            y_pred: Predicted binary tensor of shape (BS x 1 x S^3).
            y_true: Ground truth binary tensor of shape (BS x 1 x S^3).

        Returns:
            torch.Tensor: clDice metric for each batch element.
        """
        if y_pred.shape != y.shape:
            msg = f"Shape mismatch: y_pred {y_pred.shape}, y_true {y.shape}"
            raise ValueError(msg)

        if y_pred.shape[1] != 1:
            msg = f"Expected y_pred to have 1 channel, got {y_pred.shape[1]}"
            raise ValueError(msg)

        batch_size = y_pred.shape[0]

        for i in range(batch_size):
            pred_np = self._tensor_to_numpy(
                y_pred[i, 0]
            )  # Convert to NumPy, remove batch and channel dims
            true_np = self._tensor_to_numpy(y[i, 0])
            cldice_score = self._clDice_single(pred_np, true_np)
            self.cldice_scores.append(cldice_score)

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the clDice metric for the given predictions and ground truths."""
        self.update(y_pred, y)

    def reset(self):
        """Reset the clDice scores."""
        self.cldice_scores = []

    def aggregate(self) -> torch.Tensor:
        """Compute the mean clDice score."""
        return torch.tensor(statistics.mean(self.cldice_scores))


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

    def __repr__(self):
        return f"MetricsManager({list(self.metrics.keys())})"

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


def get_metrics(
    metric_types: tuple[MetricType] = (
        MetricType.DICE,
        MetricType.JACCARD_INDEX,
        MetricType.HAUSDORFF_DISTANCE,
        MetricType.CENTERLINE_LOSS,
    ),
) -> MetricsManager:
    """Get a dictionary of metrics based on the given metric types."""
    metric_types = set(metric_types)
    metrics = {}
    for metric_type in metric_types:
        match metric_type:
            case MetricType.DICE:
                metrics["dice"] = DiceMetric(include_background=True, reduction="mean")
            case MetricType.JACCARD_INDEX:
                metrics["jaccard_index"] = JaccardIndex()
            case MetricType.HAUSDORFF_DISTANCE:
                metrics["hausdorff_distance"] = HausdorffDistanceMetric(
                    include_background=True, reduction="mean"
                )
            case MetricType.CENTERLINE_LOSS:
                metrics["centerline_loss"] = ClDiceMetric()

    return MetricsManager(metrics)
