import logging

import ipywidgets as widgets
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from ipywidgets import interact
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def interactive_display_slices(
    img: np.ndarray, mask: np.ndarray, slice_indices: list[int] | None = None
):
    """Display slices of a 3D image with the corresponding mask overlaid.

    Parameters:
        img: 3D NumPy array representing the image (e.g., grayscale volume).
        mask: 3D NumPy array representing the mask (e.g., segmentation labels).
        slice_indices: Indices of slices to display. If None, 4 evenly spaced slices are chosen.
    """
    # Ensure the image and mask have the same shape
    assert img.shape == mask.shape, "Image and mask must have the same shape"

    # Determine the indices for slices
    if slice_indices is None:
        slice_indices = np.linspace(0, img.shape[2] - 1, num=4, dtype=int)

    def plot_slices(show_mask: bool):
        # Create a figure for displaying slices
        fig, axes = plt.subplots(1, len(slice_indices), figsize=(5 * len(slice_indices), 5))
        fig.suptitle("Image with Mask Overlays", fontsize=16)

        if len(slice_indices) == 1:
            axes = [axes]  # Ensure axes is iterable for a single slice

        for i, idx in enumerate(slice_indices):
            # Extract the image slice and corresponding mask slice
            img_slice = img[:, :, idx]
            mask_slice = mask[:, :, idx]

            # Display the image slice in grayscale
            axes[i].imshow(img_slice, cmap="gray", interpolation="nearest")

            # Overlay the mask if toggled on
            if show_mask:
                masked_mask = np.ma.masked_where(mask_slice == 0, mask_slice)
                axes[i].imshow(masked_mask, cmap="jet", alpha=0.5, interpolation="nearest")

            # Add title and remove axes
            axes[i].set_title(f"Slice {idx}")
            axes[i].axis("off")

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # Create an interactive toggle
    interact(plot_slices, show_mask=widgets.ToggleButton(value=True, description="Show Mask"))


def display_comparison_slices(
    img: np.ndarray,
    mask: np.ndarray,
    pred: np.ndarray,
    slice_indices: list[int] | None = None,
    mlflow_name: str | None = None,
):
    """Display slices of a 3D image with the corresponding reference mask and predicted mask overlaid.

    Optionally logs the figure to MLflow if `mlflow_name` is provided and an active MLflow run exists.

    Parameters:
        img: 3D NumPy array representing the image (e.g., grayscale volume).
        mask: 3D NumPy array representing the reference mask (binary 0/1).
        pred: 3D NumPy array representing the predicted mask (binary 0/1).
        slice_indices: Indices of slices to display. If None, 8 evenly spaced slices are chosen.
        mlflow_name: Optional; if provided, logs the figure to MLflow with this name.
    """
    # Ensure the image, mask, and pred have the same shape
    assert (
        img.shape == mask.shape == pred.shape
    ), "Image, mask, and prediction must have the same shape"

    # Determine the indices for slices
    if slice_indices is None:
        slice_indices = np.linspace(0, img.shape[2] - 1, num=8, dtype=int)

    # Create a figure for displaying slices
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))  # 4x4 grid for comparison (ref/pred pairs)
    fig.suptitle("Image with Reference and Prediction Mask Overlays", fontsize=16)

    for i, idx in enumerate(slice_indices):
        row = i // 2  # Determine the row in the 4x4 grid
        col_base = (i % 2) * 2  # Columns alternate between ref and pred (0 or 2)

        # Extract slices for image, reference mask, and prediction mask
        img_slice = img[:, :, idx]
        mask_slice = mask[:, :, idx]
        pred_slice = pred[:, :, idx]

        # Reference (ref) column
        axes[row, col_base].imshow(img_slice, cmap="gray", interpolation="nearest")
        masked_ref = np.ma.masked_where(mask_slice == 0, mask_slice)
        axes[row, col_base].imshow(masked_ref, cmap="jet", alpha=0.5, interpolation="nearest")
        axes[row, col_base].set_title(f"Ref - Slice {idx}")
        axes[row, col_base].axis("off")

        # Prediction (pred) column
        axes[row, col_base + 1].imshow(img_slice, cmap="gray", interpolation="nearest")
        masked_pred = np.ma.masked_where(pred_slice == 0, pred_slice)
        axes[row, col_base + 1].imshow(masked_pred, cmap="jet", alpha=0.5, interpolation="nearest")
        axes[row, col_base + 1].set_title(f"Pred - Slice {idx}")
        axes[row, col_base + 1].axis("off")

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if mlflow_name:
        # Check if there is an active MLflow run
        active_run = mlflow.active_run()
        if active_run:
            mlflow.log_figure(fig, mlflow_name)
            msg = f"Figure logged to MLflow with name: {mlflow_name}"
            logging.info(msg)
        else:
            logger.info("No active MLflow run. Please start an MLflow run to log the figure.")
        plt.close(fig)  # Close the figure to free up memory
    else:
        # Display the plot
        plt.show()


@torch.no_grad()
def visualize_model(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    n: int = 1,
    threshold: float = 0.5,
    sigmoid: bool = True,
) -> None:
    """Log predictions and visualizations from a data loader to MLflow.

    Parameters:
        data_loader: PyTorch DataLoader object for input data.
        model: PyTorch model for inference.
        device: Device to run the model on (e.g., "cuda" or "cpu").
        n: Number of batches to process. Defaults to 1.
        threshold: Threshold value for binarizing predictions. Defaults to 0.5.
        sigmoid: Whether to apply sigmoid activation to model output. Defaults to True.
    """
    model.eval()  # Set the model to evaluation mode

    # Ensure there is an active MLflow run
    if not mlflow.active_run():
        logger.info(
            "No active MLflow run. Please start an MLflow run before calling this function."
        )
        return

    for batch_idx, (img, mask) in enumerate(data_loader):
        if batch_idx >= n:
            break
        img = img.to(device)  # Move to the correct device
        mask = mask.to(device)
        pred = model(img)
        if sigmoid:
            pred = torch.sigmoid(pred)

        for i in range(img.size(0)):  # Iterate through the batch
            img_numpy = img[i].squeeze(0).cpu().numpy()
            mask_numpy = mask[i].squeeze(0).cpu().numpy()
            pred_numpy = pred[i].squeeze(0).cpu().numpy()
            binary_pred = (pred_numpy >= threshold).astype(int)
            # Log visualization to MLflow
            display_comparison_slices(
                img_numpy,
                mask_numpy,
                binary_pred,
                mlflow_name=f"batch_{batch_idx}_sample_{i}.png",
            )
