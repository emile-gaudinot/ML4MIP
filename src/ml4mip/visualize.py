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
def plot_3d_volume(binary_volume):
    if binary_volume is not None and binary_volume.ndim != 3:
        msg = f"Input binary volume must be 3D, but got shape: {binary_volume.shape}"
        raise ValueError(msg)

    # Define the voxel limit
    VOXEL_LIMIT = 100_000
    # Calculate the number of voxels in the original volume
    num_voxels = np.prod(binary_volume.shape)
    if num_voxels > VOXEL_LIMIT:
        msg = f"Volume exceeds the voxel limit ({VOXEL_LIMIT}). Downsampling for visualization."
        print(msg)
        # Calculate the downsampling factor
        downsample_factor = (VOXEL_LIMIT / num_voxels) ** (1 / 3)
        # Compute the target shape
        target_shape = tuple(int(s * downsample_factor) for s in binary_volume.shape)
        # Downsample the volume
        binary_volume = resize(
            binary_volume,
            output_shape=target_shape,
            preserve_range=True,
            anti_aliasing=False,
            mode="constant",
        )

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


# TODO: this function should also work without Mlflow for easy testing
@torch.no_grad()
def visualize_model(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    n: int = 1,
    threshold: float = 0.5,
    sigmoid: bool = True,
    plot_3d: bool = True,
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

            if plot_3d is not None:
                binary_pred_volume, binary_mask_volume = binary_pred > 0, mask_numpy > 0
                pred_graph, pred_skeleton, mask_graph, mask_skeleton = (
                    None,
                    None,
                    None,
                    None,
                )
                if extract_graph is not None:
                    msg = f"Extracting graph for batch {batch_idx}, sample {i}"
                    logger.info(msg)
                    pred_graph, pred_skeleton = extract_graph(binary_pred_volume)
                    mask_graph, mask_skeleton = extract_graph(binary_mask_volume)

                msg = f"visualize 3d plot for batch {batch_idx}, sample {i}"
                logger.info(msg)
                plot_comparison(
                    prediction_volume=binary_pred_volume,
                    prediction_skeleton=pred_skeleton,
                    prediction_graph=pred_graph,
                    ground_truth_volume=binary_mask_volume,
                    ground_truth_skeleton=mask_skeleton,
                    ground_truth_graph=mask_graph,
                    mlflow_name=f"batch_{batch_idx}_sample_{i}_graph.png",
                )
