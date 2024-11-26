import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact


def display_slices(img: np.ndarray, mask: np.ndarray, slice_indices: list[int] | None = None):
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
