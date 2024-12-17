import logging
from collections.abc import Callable

import ipywidgets as widgets
import matplotlib.pyplot as plt
import mlflow
import networkx as nx
import numpy as np
import torch
from ipywidgets import interact
from skimage.transform import resize
from torch import nn
from torch.utils.data import DataLoader

from ml4mip import trainer

logger = logging.getLogger(__name__)


def split_into_ranges(total: int, parts: int) -> list[range]:
    """Split a total number into nearly equal ranges."""
    # Calculate the base size of each part and the remainder
    if parts == 0:
        return []
    base_size = total // parts
    remainder = total % parts
    remainder_begin = range(remainder // 2 + remainder % 2)
    remainder_end = range(parts - (remainder // 2), parts)

    ranges = []
    start = 0

    for i in range(parts):
        # In case an even distribution is not possible, increase the width for the edges
        end = start + base_size + (1 if i in remainder_begin or i in remainder_end else 0)
        ranges.append(range(start, end))
        start = end

    return ranges


def get_max_slices(mask: np.ndarray, num_slices: int, ignore_edges_factor=0.15) -> list[int]:
    """Get the slices with the highest segmentation values."""
    assert mask.ndim == 3, "Mask must be a 3D array"
    assert 1 >= ignore_edges_factor >= 0, "Ignore edges factor must be between 0 and 1"
    ignore_edges = int(mask.shape[2] * ignore_edges_factor)
    split_ranges = split_into_ranges(mask.shape[2] - 2 * ignore_edges, num_slices)
    slice_indices = []
    for s_range in split_ranges:
        adaped_range = range(s_range.start + ignore_edges, s_range.stop + ignore_edges)
        # Find the slice in the current range with the highest portion of positive mask pixels
        slice_scores = [
            np.sum(mask[:, :, idx] > 0) / (mask.shape[0] * mask.shape[1]) for idx in adaped_range
        ]
        max_index = adaped_range.start + np.argmax(slice_scores)
        slice_indices.append(max_index)
    return slice_indices


def interactive_display_slices(
    img: np.ndarray,
    mask: np.ndarray,
    slice_indices: list[int] | None = None,
    num_slices=4,
    dim=2,
):
    """Display slices of a 3D image with the corresponding mask overlaid.

    Parameters:
        img: 3D NumPy array representing the image (e.g., grayscale volume).
        mask: 3D NumPy array representing the mask (e.g., segmentation labels).
        slice_indices: Indices of slices to display. If None, slices with the highest segmentation values
                       in evenly spaced ranges are chosen.
        num_slices: Number of slices to display if slice_indices is not specified.
        dim: Dimension along which to take slices (0=x, 1=y, 2=z).
    """
    # Ensure the image and mask have the same shape
    assert img.ndim == 3, "Image must be a 3D array"
    assert img.shape == mask.shape, "Image and mask must have the same shape"
    assert dim in [0, 1, 2], "Dimension must be 0 (x), 1 (y), or 2 (z)"

    # Transpose the image and mask to make the selected dimension the last axis
    if dim != 2:
        img = np.moveaxis(img, dim, 2)
        mask = np.moveaxis(mask, dim, 2)

    slice_indices = slice_indices or get_max_slices(mask, num_slices)

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
    dim=2,
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
    assert img.ndim == 3, "Image must be a 3D array"
    assert (
        img.shape == mask.shape == pred.shape
    ), "Image, mask, and prediction must have the same shape"
    assert dim in [0, 1, 2], "Dimension must be 0 (x), 1 (y), or 2 (z)"

    if dim != 2:
        img = np.moveaxis(img, dim, 2)
        mask = np.moveaxis(mask, dim, 2)
        pred = np.moveaxis(pred, dim, 2)

    # Determine the indices for slices
    NUM_SLICES = 8
    slice_indices = slice_indices or get_max_slices(mask, NUM_SLICES)

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
            logger.info(msg)
        else:
            logger.info("No active MLflow run. Please start an MLflow run to log the figure.")
        plt.close(fig)  # Close the figure to free up memory
    else:
        # Display the plot
        plt.show()


def project_mask_2d(
    mask: np.ndarray,
    dim: int = 2,
    mode: str = "heatmap",
    title: str = "2D Projection of 3D Mask",
):
    """Projects a 3D binary mask along a specified dimension and creates a 2D output plot.

    Parameters:
        mask: 3D NumPy binary array (0 and 1 values).
        dim: Dimension along which to project (0=x, 1=y, 2=z).
        mode: Mode of projection, either "heatmap" or "binary".
              - "heatmap": Color shows multiple positive voxels.
              - "binary": 2D map has only 0 and 1s (presence or absence).
        title: Title for the plot.

    Returns:
        projected: 2D array of the projected mask.
    """
    assert mask.ndim == 3, "Input mask must be a 3D array."
    assert dim in [0, 1, 2], "Dimension must be 0 (x), 1 (y), or 2 (z)."
    assert mode in ["heatmap", "binary"], "Mode must be either 'heatmap' or 'binary'."

    # Sum along the specified dimension
    projected = np.sum(mask, axis=dim)

    if mode == "binary":
        # Convert to binary (0 and 1)
        projected = (projected > 0).astype(int)

    # Create the plot
    plt.figure(figsize=(8, 8))
    cmap = "hot" if mode == "heatmap" else "gray"
    plt.imshow(projected, cmap=cmap, interpolation="nearest")
    plt.colorbar(label="Projection Intensity" if mode == "heatmap" else "Binary Value")
    plt.title(title)
    plt.xlabel("Axis 1")
    plt.ylabel("Axis 2")
    plt.show()


def display_projection_comparison(
    mask: np.ndarray,
    pred: np.ndarray,
    dim: int = 2,
    mode: str = "heatmap",
    title: str = "Projection Comparison: Reference vs Prediction",
    mlflow_name: str | None = None,
):
    """Projects a 3D binary mask and prediction along a specified dimension and compares them.

    Parameters:
        mask: 3D NumPy binary array (0 and 1 values) representing the reference mask.
        pred: 3D NumPy binary array (0 and 1 values) representing the predicted mask.
        dim: Dimension along which to project (0=x, 1=y, 2=z).
        mode: Mode of projection, either "heatmap" or "binary".
              - "heatmap": Color shows multiple positive voxels.
              - "binary": 2D map has only 0 and 1s (presence or absence).
        title: Title for the overall comparison plot.

    Returns:
        projected_mask: 2D projection of the reference mask.
        projected_pred: 2D projection of the predicted mask.
    """
    assert mask.ndim == 3, "Input mask must be a 3D array."
    assert pred.ndim == 3, "Input prediction must be a 3D array."
    assert mask.shape == pred.shape, "Mask and prediction must have the same shape."
    assert dim in [0, 1, 2], "Dimension must be 0 (x), 1 (y), or 2 (z)."
    assert mode in ["heatmap", "binary"], "Mode must be either 'heatmap' or 'binary'."

    # Project the mask and prediction along the specified dimension
    projected_mask = np.sum(mask, axis=dim)
    projected_pred = np.sum(pred, axis=dim)

    if mode == "binary":
        # Convert to binary (0 and 1)
        projected_mask = (projected_mask > 0).astype(int)
        projected_pred = (projected_pred > 0).astype(int)

    cmap = "hot" if mode == "heatmap" else "gray"

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))  # 1 row, 2 columns
    fig.suptitle(title, fontsize=16)

    # Reference Mask
    axes[0].imshow(projected_mask, cmap=cmap, interpolation="nearest")
    axes[0].set_title("Reference", fontsize=14)
    axes[0].axis("off")

    # Prediction
    axes[1].imshow(projected_pred, cmap=cmap, interpolation="nearest")
    axes[1].set_title("Prediction", fontsize=14)
    axes[1].axis("off")

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if mlflow_name:
        # Check if there is an active MLflow run
        active_run = mlflow.active_run()
        if active_run:
            mlflow.log_figure(fig, mlflow_name)
            msg = f"Figure logged to MLflow with name: {mlflow_name}"
            logger.info(msg)
        else:
            logger.info("No active MLflow run. Please start an MLflow run to log the figure.")
        plt.close(fig)  # Close the figure to free up memory
    else:
        # Display the plot
        plt.show()


def plot_3d_view(
    ax,
    binary_volume: np.ndarray | None = None,
    skeleton: np.ndarray | None = None,
    graph: nx.Graph | None = None,
    voxel_color: str = "orange",
    skeleton_color: str = "red",
    node_color: str = "blue",
    edge_color: str = "green",
) -> None:
    """Helper function to render a single 3D view."""
    # Plot the 3D volume if provided
    if binary_volume is not None:
        ax.voxels(binary_volume, facecolors=voxel_color, alpha=0.2)

    # Plot the skeleton if provided
    if skeleton is not None:
        x, y, z = np.nonzero(skeleton)
        ax.scatter(x, y, z, c=skeleton_color, marker="o", s=2, label="Skeleton")

    # Plot the graph if provided
    if graph is not None:
        # Plot graph nodes
        has_node_label = False
        for _, data in graph.nodes(data=True):
            coord = data["coordinate"]
            ax.scatter(
                coord[0],
                coord[1],
                coord[2],
                c=node_color,
                s=20,
                label=("Nodes" if not has_node_label else None),
            )
            has_node_label = True

        # Plot graph edges
        has_edge_label = False
        for u, v in graph.edges():
            coord_u = graph.nodes[u]["coordinate"]
            coord_v = graph.nodes[v]["coordinate"]
            ax.plot(
                [coord_u[0], coord_v[0]],
                [coord_u[1], coord_v[1]],
                [coord_u[2], coord_v[2]],
                c=edge_color,
                linewidth=2,
                label="Edges" if not has_edge_label else None,
            )
            has_edge_label = True

    # Set labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best")


def downsample_volume(volume, limit):
    """Downsamples a 3D volume if it exceeds the voxel limit."""
    num_voxels = np.prod(volume.shape)
    if num_voxels > limit:
        msg = f"Volume exceeds the voxel limit ({limit}). Downsampling for visualization."
        logger.info(msg)
        downsample_factor = (limit / num_voxels) ** (1 / 3)
        target_shape = tuple(int(s * downsample_factor) for s in volume.shape)
        volume = resize(
            volume,
            output_shape=target_shape,
            preserve_range=True,
            anti_aliasing=False,
            mode="constant",
        )
    return volume


# TODO: not efficient for plotting, doesn't handle too many voxels
@DeprecationWarning
def plot_comparison(
    prediction_volume: np.ndarray | None = None,
    prediction_skeleton: np.ndarray | None = None,
    prediction_graph: nx.Graph | None = None,
    ground_truth_volume: np.ndarray | None = None,
    ground_truth_skeleton: np.ndarray | None = None,
    ground_truth_graph: nx.Graph | None = None,
    mlflow_name: str | None = None,
) -> None:
    """Render a side-by-side comparison of prediction and ground truth."""
    # Create the figure with two subplots
    fig = plt.figure(figsize=(20, 10))
    ax_pred = fig.add_subplot(121, projection="3d", title="Prediction")
    ax_gt = fig.add_subplot(122, projection="3d", title="Ground Truth")

    # Plot the prediction
    plot_3d_view(
        ax=ax_pred,
        binary_volume=prediction_volume,
        skeleton=prediction_skeleton,
        graph=prediction_graph,
        voxel_color="orange",
        skeleton_color="red",
        node_color="blue",
        edge_color="green",
    )

    # Plot the ground truth
    plot_3d_view(
        ax=ax_gt,
        binary_volume=ground_truth_volume,
        skeleton=ground_truth_skeleton,
        graph=ground_truth_graph,
        voxel_color="purple",
        skeleton_color="cyan",
        node_color="yellow",
        edge_color="black",
    )

    if mlflow_name:
        # Check if there is an active MLflow run
        active_run = mlflow.active_run()
        if active_run:
            mlflow.log_figure(fig, mlflow_name)
            msg = f"Figure logged to MLflow with name: {mlflow_name}"
            logger.info(msg)
        else:
            logger.info("No active MLflow run. Please start an MLflow run to log the figure.")
        plt.close(fig)  # Close the figure to free up memory
    else:
        # Display the plots
        plt.show()


# TODO: does currently only work with binary volumes
# Update so the skeleton and graph can be displayed too.
def plot_3d_volume(binary_volume, voxel_limit=100_000_000):
    if binary_volume is not None and binary_volume.ndim != 3:
        msg = f"Input binary volume must be 3D, but got shape: {binary_volume.shape}"
        raise ValueError(msg)

    # Calculate the number of voxels in the original volume
    binary_volume = downsample_volume(binary_volume, voxel_limit)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    plot_3d_view(
        ax=ax,
        binary_volume=binary_volume,
        voxel_color="orange",
        skeleton_color="red",
        node_color="blue",
        edge_color="green",
    )


def plot_3d_comparison(
    mask: np.ndarray,
    pred: np.ndarray,
    voxel_limit: int = 100_000,
    mlflow_name: str | None = None,
):
    """Plots 3D binary volumes for the mask (reference) and pred (prediction) side by side.

    Parameters:
        mask: 3D binary NumPy array representing the reference mask.
        pred: 3D binary NumPy array representing the predicted mask.
        voxel_limit: Maximum number of voxels allowed for visualization. Volumes exceeding this
                     limit will be downsampled.
    """
    assert mask.ndim == 3, "Input mask must be a 3D binary volume."
    assert pred.ndim == 3, "Input prediction must be a 3D binary volume."
    assert mask.shape == pred.shape, "Mask and prediction volumes must have the same shape."

    # Downsample if necessary
    mask_downsampled = downsample_volume(mask, voxel_limit)
    pred_downsampled = downsample_volume(pred, voxel_limit)

    # Create a figure with two subplots for side-by-side comparison
    fig = plt.figure(figsize=(20, 10))

    # Plot the mask
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_title("Reference Mask")
    plot_3d_view(
        ax=ax1,
        binary_volume=mask_downsampled,
        voxel_color="blue",
    )

    # Plot the prediction
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_title("Prediction")
    plot_3d_view(
        ax=ax2,
        binary_volume=pred_downsampled,
        voxel_color="orange",
    )

    plt.tight_layout()
    if mlflow_name:
        # Check if there is an active MLflow run
        active_run = mlflow.active_run()
        if active_run:
            mlflow.log_figure(fig, mlflow_name)
            msg = f"Figure logged to MLflow with name: {mlflow_name}"
            logger.info(msg)
        else:
            logger.info("No active MLflow run. Please start an MLflow run to log the figure.")
        plt.close(fig)  # Close the figure to free up memory
    else:
        # Display the plot
        plt.show()


def run_inference_and_plot(
    data_loader: DataLoader,
    inferer: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    n: int = 1,
    threshold: float = 0.5,
    sigmoid: bool = True,
    plot_3d: bool = True,
    tag: str = "training",
):
    for batch_idx, (img, mask) in enumerate(data_loader):
        if batch_idx >= n:
            break
        img = img.to(device)  # Move to the correct device
        mask = mask.to(device)
        pred = inferer(img)
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
                mlflow_name=f"{tag}_batch_{batch_idx}_sample_{i}.png",
            )

            display_projection_comparison(
                mask_numpy,
                binary_pred,
                mlflow_name=f"{tag}_batch_{batch_idx}_sample_{i}_projection.png",
            )

            if plot_3d is not None:
                binary_pred_volume, binary_mask_volume = binary_pred > 0, mask_numpy > 0

                logger.info("Plotting 3D comparison ...")
                plot_3d_comparison(
                    binary_mask_volume,
                    binary_pred_volume,
                    mlflow_name=f"{tag}_batch_{batch_idx}_sample_{i}_3d.png",
                )


# TODO: this function should also work without Mlflow for easy testing
@torch.no_grad()
def visualize_model(
    val_data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    inference_cfg: trainer.InferenceConfig,
    val_batches: int = 1,
    threshold: float = 0.5,
    sigmoid: bool = True,
    plot_3d: bool = True,
    train_data_loader: DataLoader | None = None,
    train_batches: int = 4,
    epoch: int | None = None,
    extract_graph: Callable[[np.ndarray], tuple[nx.Graph, np.ndarray]] | None = None,
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

    # Run inference and plot visualizations
    logger.info("Starting inference and visualization for validation data ...")
    run_inference_and_plot(
        data_loader=val_data_loader,
        inferer=lambda img: trainer.inference(img, model, inference_cfg),
        device=device,
        n=val_batches,
        threshold=threshold,
        sigmoid=sigmoid,
        plot_3d=plot_3d,
        tag=f"validation_epoch_{epoch}" if epoch is not None else "validation",
    )

    if train_data_loader is not None:
        logger.info("Starting inference and visualization for training data ...")
        run_inference_and_plot(
            data_loader=train_data_loader,
            inferer=model,
            device=device,
            n=train_batches,
            threshold=threshold,
            sigmoid=sigmoid,
            plot_3d=plot_3d,
            tag=f"training_epoch_{epoch}" if epoch is not None else "training",
        )
