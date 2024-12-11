import pytest
import torch
from monai.metrics import DiceMetric

from ml4mip.utils.metrics import MetricsManager


def test_metric_manager_copy():
    metrics = MetricsManager(
        metrics={
            "dice": DiceMetric(
                include_background=True,
                reduction="mean",
            )
        },
        sigmoid=False,
    )

    metrics_copy = metrics.copy()

    # example data

    # Example data (non-trivial predictions and ground truth)
    y_pred = torch.tensor([[[[0.97, 0.02], [0.0, 1]]]])
    y = torch.tensor([[[[1, 0], [0, 1]]]])

    y_pred2 = torch.tensor([[[[0.7, 0.3], [0.4, 0.6]]]])
    y2 = torch.tensor([[[[1, 0], [0, 1]]]])

    metrics(y_pred, y)
    metrics(y_pred2, y2)
    metrics(y_pred, y)

    # apply the metrics on copy
    metrics_copy(y_pred, y)
    res_copy_1 = metrics_copy.aggregate()
    metrics_copy.reset()

    metrics_copy(y_pred2, y2)
    res_copy_2 = metrics_copy.aggregate()
    metrics_copy.reset()

    metrics_copy(y_pred, y)
    res_copy_3 = metrics_copy.aggregate()
    metrics_copy.reset()

    res_copy_avg = {"dice": (res_copy_1["dice"] + res_copy_2["dice"] + res_copy_3["dice"]) / 3}

    res = metrics.aggregate()
    for key in res:
        assert res[key] == pytest.approx(res_copy_avg[key], rel=1e-5)
