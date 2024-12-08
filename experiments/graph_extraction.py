# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python (cake)
#     language: python
#     name: cake
# ---

# %%

import matplotlib.pyplot as plt
import numpy as np

from ml4mip.dataset import DatasetConfig, TransformType, get_dataset
from ml4mip.graph_extraction import extract_graph
from ml4mip.visualize import plot_3d_view


# %%
def plot_3d_volume(binary_volume=None, skeleton=None, graph=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    plot_3d_view(
        ax=ax,
        binary_volume=binary_volume,
        skeleton=skeleton,
        graph=graph,
        voxel_color="orange",
        skeleton_color="red",
        node_color="blue",
        edge_color="green",
    )


# %%
# Load the NIfTI file
cfg_dataset = DatasetConfig(
    data_dir="/data/training_data",
    image_suffix="img.nii.gz",
    mask_suffix="label.nii.gz",
    transform=TransformType.RESIZE_96,
    train=False,
)

ds = get_dataset(cfg_dataset)

# %%
img, mask = ds[0][0].squeeze().numpy(), ds[0][1].squeeze().numpy()


# %%
# Example usage
binary_mask = mask > 0
graph, skeleton = extract_graph(binary_mask)
plot_3d_volume(
    binary_volume=binary_mask,
    skeleton=skeleton,
    # graph=graph,
)


# %% [markdown]
# # Visualization Functions

# %%
def filter_and_rescale(
    image: np.ndarray,
    value_range: tuple[float, float],
) -> np.ndarray:
    """Filters an image to keep only values within a specified range.

    Parameters:
        image: 2D or 3D image with normalized values in [0, 1].
        value_range: (min_value, max_value) range to retain.

    Returns:
        Processed image with values outside the range set to 0,
                       and remaining values rescaled to [0, 1].
    """
    # Ensure input values are normalized
    if np.any(image < 0) or np.any(image > 1):
        raise ValueError("Input image must have normalized values in the range [0, 1].")

    # Extract range limits
    min_val, max_val = value_range
    if not (0 <= min_val <= max_val <= 1):
        raise ValueError("Value range must be within [0, 1].")

    # Create a mask for the specified range
    mask = (image >= min_val) & (image <= max_val)

    # Set values outside the range to 0
    filtered_image = np.zeros_like(image)
    filtered_image[mask] = image[mask]

    # Rescale the remaining values to [0, 1]
    if mask.any():  # Avoid division by zero if the mask is empty
        filtered_image = normalize_img(filtered_image)

    return filtered_image


def normalize_img(image: np.ndarray) -> np.ndarray:
    """Normalizes an image to have values in the range [0, 1].

    Parameters:
        image: 2D or 3D image with arbitrary values.

    Returns:
        Normalized image with values rescaled to [0, 1].
    """
    # Extract the minimum and maximum values
    min_val, max_val = np.min(image), np.max(image)

    # Normalize the image to the range [0, 1]
    if min_val != max_val:
        image = (image - min_val) / (max_val - min_val)

    return image

def get_value_range(img: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Computes the value range of an image within the mask.

    Parameters:
        img: 2D or 3D image with arbitrary values.
        mask: 2D or 3D binary mask.

    Returns:
        Value range of the image within the mask.
    """
    # Filter img values where mask == 1
    masked_values = img[mask == 1]

    # Calculate the value range
    min_value = masked_values.min()
    max_value = masked_values.max()

    return min_value, max_value
