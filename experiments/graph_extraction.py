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
from skimage.morphology import skeletonize

from ml4mip.dataset import DatasetConfig, TransformType, get_dataset


# %%
def plot_3d_volume_with_skeleton(
    binary_volume, skeleton=None, voxel_color="blue", skeleton_color="red"
):
    """Render a 3D volume using Matplotlib voxels and overlay skeleton points.

    Parameters:
        volume (numpy.ndarray): 3D volume data.
        skeleton (numpy.ndarray): 3D binary skeleton array (same shape as volume).
        voxel_color (str): Color for the volume voxels.
        skeleton_color (str): Color for the skeleton points.
    """
    # Get skeleton coordinates


    # Create the figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the voxels
    ax.voxels(binary_volume, facecolors=voxel_color, edgecolor="k", alpha=0.2)

    # Overlay the skeleton points
    if skeleton is not None:
        x, y, z = np.nonzero(skeleton)
        ax.scatter(x, y, z, color=skeleton_color, marker="o", s=5, label="Skeleton")

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("3D Volume Rendering with Skeleton")
    plt.legend()
    plt.show()

def visualize_graph(skeleton, G):
    """Visualize the skeleton and its graph representation.

    Parameters:
        skeleton (numpy.ndarray): 3D binary array of the skeleton.
        G (networkx.Graph): Graph representation of the skeleton.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the skeleton points
    z, y, x = np.nonzero(skeleton)
    ax.scatter(x, y, z, c="blue", s=1, label="Skeleton")

    # Plot the graph nodes (add label only once)
    has_node_label = False
    for node, data in G.nodes(data=True):
        coord = data["coordinate"]
        ax.scatter(
            coord[2], coord[1], coord[0], c="red", s=2, label="Nodes" if not has_node_label else None
        )
        has_node_label = True

    # Plot the graph edges
    has_edge_label = False
    for u, v in G.edges():
        coord_u = G.nodes[u]["coordinate"]
        coord_v = G.nodes[v]["coordinate"]
        ax.plot(
            [coord_u[2], coord_v[2]],
            [coord_u[1], coord_v[1]],
            [coord_u[0], coord_v[0]],
            c="green",
            linewidth=2,
            label="Edges" if not has_edge_label else None,
        )
        has_edge_label = True

    # Set labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best")

    # Adjust aspect ratio for better visibility
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()



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
skeleton = skeletonize(binary_mask)
plot_3d_volume_with_skeleton(binary_mask, skeleton)

# %%
import networkx as nx


def skeleton_to_graph(skeleton):
    """Convert a 3D skeleton into a graph where each skeleton point is a node.

    Parameters:
        skeleton (numpy.ndarray): 3D binary array of the skeleton.

    Returns:
        G (networkx.Graph): Graph representation of the skeleton.
    """
    # Define a 3x3x3 neighborhood kernel to find neighbors
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = 0  # Exclude the center point

    # Identify all skeleton points
    skeleton_coords = np.argwhere(skeleton > 0)

    # Initialize the graph
    G = nx.Graph()

    # Add all skeleton points as nodes
    for idx, coord in enumerate(skeleton_coords):
        G.add_node(idx, coordinate=tuple(coord))

    # Create a map of coordinates to node indices for fast lookup
    coord_to_index = {tuple(coord): idx for idx, coord in enumerate(skeleton_coords)}

    # Find neighbors for each skeleton point and add edges
    for coord in skeleton_coords:
        # Get neighbors within the kernel
        neighbors = np.argwhere(kernel) - 1  # Offsets for 3x3x3 neighborhood
        for offset in neighbors:
            neighbor_coord = tuple(coord + offset)
            if neighbor_coord in coord_to_index:  # Check if neighbor exists in skeleton
                G.add_edge(coord_to_index[tuple(coord)], coord_to_index[neighbor_coord])

    return G


def reduce_graph(graph):
    """Reduce the graph by keeping only nodes that are endpoints or connecting nodes.

    - Endpoints (nodes with exactly one neighbor).
    - Connecting nodes (nodes with more than two neighbors).

    Parameters:
        graph (networkx.Graph): Input graph.

    Returns:
        reduced_graph (networkx.Graph): Reduced graph.
    """
    # Initialize the reduced graph
    reduced_graph = nx.Graph()

    # Iterate over the nodes in the original graph
    for node in graph.nodes():
        degree = graph.degree[node]  # Number of neighbors

        # Retain the node if it is an endpoint or a connecting node
        if degree == 1 or degree > 2:
            # Add the node to the reduced graph with its attributes
            reduced_graph.add_node(node, **graph.nodes[node])  # Copy node attributes

    for primary_node in reduced_graph.nodes():
        for primary_neighbor in graph.neighbors(primary_node):
            visited = set()
            current_node = primary_neighbor
            while current_node not in reduced_graph.nodes():
                visited.add(current_node)
                current_node_neighbors = set(graph.neighbors(current_node))
                assert len(current_node_neighbors) == 2, f"{len(current_node_neighbors)=}"
                diff = current_node_neighbors - visited - {primary_node}
                assert len(diff) == 1, f"{len(diff)=}"
                current_node = diff.pop()

            reduced_graph.add_edge(primary_node, current_node)

    return reduced_graph



# %%
graph = skeleton_to_graph(skeleton)
reduced_graph = reduce_graph(graph)
visualize_graph(
    skeleton, reduced_graph
)  # TODO for some reason it looks like some edges are missing

# %%

# %%
visualize_graph(skeleton, graph)

# %%
reduced_graph = reduce_graph(graph)
visualize_graph(skeleton, reduced_graph)
# for u, v in reduced_graph.edges():
#     print(f"Edge: {u} -> {v}")

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
