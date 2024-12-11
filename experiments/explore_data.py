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

import random
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import networkx as nx
import numpy as np
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    MapTransform,
    Randomizable,
    RandSpatialCropd,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    Spacingd,
    ToTensord,
)
from scipy.stats import truncnorm
from skimage.transform import resize
from tqdm.notebook import tqdm

from ml4mip.dataset import NiftiDataset

mplstyle.use("fast")
mplstyle.use(["dark_background", "ggplot", "fast"])


# %%
def run_dataset_sampling(dataset,analyze_sample, num_samples=100, multi_threaded=False):
    indices = random.sample(range(len(dataset)), num_samples)
    results = []
    if multi_threaded:
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(analyze_sample, dataset, idx): idx for idx in indices}

            # Use tqdm to display the progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading samples"):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    msg = f"Error processing index {futures[future]}: {e}"
                    print(msg)
    else:
        for idx in tqdm(indices):
            res = analyze_sample(dataset, idx)
            results.append(res)

    return results


# %%
class TruncatedGaussianRandomCrop(Randomizable, MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        roi_size: Sequence[int] | int,
        sigma_ratio: float = 0.1,
        allow_missing_keys: bool = False,
    ):
        """Transform to crop a random region from a volume based on a truncated Gaussian distribution.

        Args:
            keys: Keys of the corresponding items to be transformed.
            roi_size: Desired output size of the crop (e.g., Depth, Height, Width for 3D).
                      Can handle 2D, 3D, or higher-dimensional data.
            sigma_ratio: Ratio to determine the standard deviation of the Gaussian
                         distribution relative to volume dimensions.
            allow_missing_keys: Don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.roi_size = np.array(roi_size if isinstance(roi_size, Sequence) else [roi_size])
        self.sigma_ratio = sigma_ratio

    def truncated_normal(self, mean, std, lower, upper):
        """Generate a sample from a truncated normal distribution."""
        a, b = (lower - mean) / std, (upper - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std, random_state=self.R)

    def sample_center(self, img_shape):
        """Sample the center of the crop based on the truncated Gaussian distribution."""
        img_shape = np.array(img_shape)
        center = img_shape // 2
        sigma = img_shape * self.sigma_ratio

        crop_center = []
        for i in range(len(img_shape)):
            lower_bound = self.roi_size[i] // 2
            upper_bound = img_shape[i] - self.roi_size[i] // 2
            crop_center.append(
                int(self.truncated_normal(center[i], sigma[i], lower_bound, upper_bound))
            )

        return np.array(crop_center)

    def plot_distributions(self, img_shape, num_samples=10000):
        """Plot the distributions of the sampled crop center for each dimension.

        Note:
        - For 2D data, generates a heatmap for the sampled (x, y) coordinates.
        - For higher-dimensional data, generates histograms for each dimension.
        """
        img_shape = np.array(img_shape)
        centers = np.array([self.sample_center(img_shape) for _ in range(num_samples)])

        if len(img_shape) == 2:  # Special case: 2D data
            # Generate 2D histogram for x, y distribution
            heatmap, xedges, yedges = np.histogram2d(
                centers[:, 0], centers[:, 1], bins=(50, 50), range=[[0, img_shape[0]], [0, img_shape[1]]]
            )

            # Plot heatmap
            plt.figure(figsize=(8, 6))
            plt.imshow(
                heatmap.T, origin="lower", aspect="auto",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap="viridis"
            )
            plt.colorbar(label="Frequency")
            plt.title("2D Heatmap of Sampled Crop Centers (x, y)")
            plt.xlabel("Height (y)")
            plt.ylabel("Width (x)")
            plt.grid(False)
            plt.show()

        else:  # Default case: N-Dimensional data
            fig, axes = plt.subplots(1, len(img_shape), figsize=(5 * len(img_shape), 5))
            if len(img_shape) == 1:
                axes = [axes]  # Ensure axes is iterable for 1D case
            axis_labels = [f"Dim {i}" for i in range(len(img_shape))]

            for i, ax in enumerate(axes):
                ax.hist(centers[:, i], bins=50, alpha=0.7, color="blue", edgecolor="black")
                ax.set_title(f"Distribution of {axis_labels[i]} Sampled Crop Centers")
                ax.set_xlabel(f"{axis_labels[i]} Coordinate")
                ax.set_ylabel("Frequency")
                ax.grid(True)

            plt.tight_layout()
            plt.show()


    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            img = d[key]
            img_shape = np.array(img.shape[1:])  # Assuming channel-first format
            crop_center = self.sample_center(img_shape)

            # Calculate the start and end points for cropping
            start = crop_center - self.roi_size // 2
            end = start + self.roi_size

            # Extract the crop
            slices = tuple(slice(s, e) for s, e in zip(start, end, strict=False))
            d[key] = img[(slice(None), *slices)]

        return d


transform = TruncatedGaussianRandomCrop(
    keys=["image", "label"],
    roi_size=(96, 96),
    sigma_ratio=0.2,
)
img_shape = np.array([512, 448])  # Depth, Height, Width
transform.plot_distributions(img_shape, num_samples=10000)



# %% [markdown]
# # Pipeline definition

# %%
resample_transform = Spacingd(
        keys=["image", "mask"],  # Keys to apply the transform
        pixdim=(0.5, 0.5, 0.5),  # Desired voxel spacing in mm
        mode=("bilinear", "nearest"),  # Interpolation modes for image and mask
    )

# this was calculated after observing the maximum size of the images after the resampling
# That way we don't need to add too much padding.
# TODO: make the experiment reproducible
target_spatial_size = (512, 448, 256)  # (X, Y, Z) in voxels


resize_transform = ResizeWithPadOrCropd(
    keys=["image", "mask"],
    spatial_size=target_spatial_size
)

ensure_channel_first = EnsureChannelFirstd(keys=["image", "mask"])

# this really depends on the input range. it could happen that the range is not meaningful
scale_transform = ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)

# So this could be a good transforming approach
# Then based on this we could perform the two other transforming operations
size = 96
pipeline = Compose([
    ensure_channel_first,
    resample_transform,
    resize_transform,
    scale_transform,
    TruncatedGaussianRandomCrop(
        keys=["image", "mask"],
        roi_size=(size, size, size),
        sigma_ratio=0.1,
    ),
    ToTensord(keys=["image", "mask"])
])

# %% [markdown]
# # Experiment, compare different preprocessing steps with each other
#
# - [x] Setting, sample 100 indices and analyze:
#     - contains positive examples
#     - what is the true positive portion
#
#     - Configuration:
#         - Random Cropping
#         - TruncatedGaussianCropping (high ratio)
#         - TruncatedGaussianCropping (mid ratio)
#         - TruncatedGaussianCropping (low ratio)
#
#
# # Next steps
#
# - Data Preprocessing: how to scale values, maybe filter values that aren't of interest?
#
# ## Patch
# - what is a good patch size 96 x 96 x 96?
#     => could be handled by unetr
#     => test this here
#
# - training then with random crops
# - inference with GridOverlap
#   
# - What happens if there are empty patches?
# - How does the model calculate these?
# => smooth dice loss (monai)
#
# - what is a good global training view: it should respect the proportions.
#     => cannt be handled by unetr
#     
#     if equally dimension are required:
#     (then either resize or CropOrPad ...)
#
# - how to validate then?
#     => then upscale the images again (F.interpolate(predictions, size=original_size, mode='trilinear', align_corners=False))
#
#
# **Integrate these steps in the Pipeline.**

# %%
size = 96
pipeline_rand = Compose([
    ensure_channel_first,
    resample_transform,
    resize_transform,
    scale_transform,
    RandSpatialCropd(
                keys=["image", "mask"],
                roi_size=(size, size, size),  # Size of the patches
                random_size=False,  # Ensure fixed-size patches
            ),
])

pipeline_gaus_high = Compose([
    ensure_channel_first,
    resample_transform,
    resize_transform,
    scale_transform,
    TruncatedGaussianRandomCrop(
        keys=["image", "mask"],
        roi_size=(size, size, size),
        sigma_ratio=0.2,
    ),
])

pipeline_gaus_mid = Compose([
    ensure_channel_first,
    resample_transform,
    resize_transform,
    scale_transform,
    TruncatedGaussianRandomCrop(
        keys=["image", "mask"],
        roi_size=(size, size, size),
        sigma_ratio=0.1,
    ),
])

pipeline_gaus_low = Compose([
    ensure_channel_first,
    resample_transform,
    resize_transform,
    scale_transform,
    TruncatedGaussianRandomCrop(
        keys=["image", "mask"],
        roi_size=(size, size, size),
        sigma_ratio=0.05,
    ),
])






# %%
def analyze_sample(dataset, idx):
    img, mask = dataset[idx]

    binary_volume = mask.squeeze().numpy()
    positive_count = (binary_volume > 0).sum()
    has_positive = positive_count > 0

    portion_positive = positive_count / binary_volume.size

    image_shape = img.shape
    voxel_spacing = img.pixdim.tolist()

    return {
        "index": idx,
        "image_shape": image_shape,
        "voxel_spacing": voxel_spacing,
        "has_positive": has_positive,
        "portion_positive": portion_positive,
    }

pipelines = {
    "Random Crop": pipeline_rand,
    "Gaussian Crop (High Sigma)": pipeline_gaus_high,
    "Gaussian Crop (Mid Sigma)": pipeline_gaus_mid,
    "Gaussian Crop (Low Sigma)": pipeline_gaus_low,
}
NUM_SAMPLES = 50
results = {}
for name, pipeline in pipelines.items():
    ds = NiftiDataset(
        data_dir="/data/training_data",
        transform=pipeline,
    )
    results[name] = run_dataset_sampling(ds, analyze_sample, num_samples=NUM_SAMPLES)

count_positive = lambda items: sum(res["has_positive"] for res in items)
avg_portion_positive = lambda items: sum(res["portion_positive"] for res in items) / len(items)

mapped_results = {name: {"count_positive": count_positive(res), "avg_portion_positive": avg_portion_positive(res)} for name, res in results.items()}

# %%
result_copy = mapped_results.copy()
rename_map = {
    "Random Crop": "Random Crop",
    "Gaussian Crop (High Sigma)": "High Sigma (0.2)",
    "Gaussian Crop (Mid Sigma)": "Mid Sigma (0.1)",
    "Gaussian Crop (Low Sigma)": "Low Sigma (0.05)",
}
mapped_results = {rename_map[k]: v for k, v in result_copy.items()}

# %%
# Extract data for plotting
pipelines = list(mapped_results.keys())
count_positive_values = [mapped_results[name]["count_positive"] for name in pipelines]
avg_portion_positive_values = [mapped_results[name]["avg_portion_positive"] for name in pipelines]

# # Plot 1: Count of Positive Samples
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.bar(pipelines, count_positive_values, color='blue', alpha=0.7, edgecolor='black')
# plt.title("Count of Positive Samples")
# plt.ylabel("Count Positive")
# plt.xticks(rotation=15, ha="right")

# # Plot 2: Average Portion of Positive Voxels
# plt.subplot(1, 2, 2)
# plt.bar(pipelines, avg_portion_positive_values, color='orange', alpha=0.7, edgecolor='black')
# plt.title("Average Portion of Positive Voxels")
# plt.ylabel("Avg Portion Positive")
# plt.xticks(rotation=15, ha="right")

# plt.tight_layout()
# plt.show()

fig, ax1 = plt.subplots(figsize=(12, 6))

# Create bar positions
x = np.arange(len(pipelines))
width = 0.35  # Width of the bars

# Plot count_positive on the left y-axis
ax1.bar(x - width / 2, count_positive_values, width, label=f"Count Positive x/{NUM_SAMPLES}", color="blue", alpha=0.7, edgecolor="black")
ax1.set_ylabel("Count Positive", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.set_xticks(x)
ax1.set_xticklabels(pipelines, rotation=15, ha="right")

# Add the second y-axis for avg_portion_positive
ax2 = ax1.twinx()
ax2.bar(x + width / 2, avg_portion_positive_values, width, label="Avg Portion Positive", color="orange", alpha=0.7, edgecolor="black")
ax2.set_ylabel("Avg Portion Positive", color="orange")
ax2.tick_params(axis="y", labelcolor="orange")

# Add title and legend
# fig.suptitle("Comparison of Count Positive and Avg Portion Positive with Dual Scales")
ax1.legend(loc="upper left", bbox_to_anchor=(0, 1), labelcolor="black")
ax2.legend(loc="upper right", bbox_to_anchor=(1, 1), labelcolor="black")
plt.title("Comparison of Count Positive and Avg Portion Positive with Dual Scales")
plt.tight_layout()
plt.show()

# %%
# Logic for printing Image shapes etc, was useful for logging
# # Analyze the shapes
# if shapes:
#     # Convert shapes to a tensor for easier analysis
#     shape_tensor = torch.tensor(shapes)

#     # Compute min and max for each dimension
#     min_dims = shape_tensor.min(dim=0).values
#     max_dims = shape_tensor.max(dim=0).values

#     print("Shape ranges across samples:")
#     for i, (min_val, max_val) in enumerate(zip(min_dims.tolist(), max_dims.tolist(), strict=False)):
#         print(f"Dimension {i}: Min = {min_val}, Max = {max_val}")
# else:
#     print("No shapes were analyzed.")

# # Analyze the voxel spacings
# if spacings:
#     # Convert spacings to a tensor for easier analysis
#     spacing_tensor = torch.tensor(spacings)

#     # Compute min and max for each dimension
#     min_spacings = spacing_tensor.min(dim=0).values
#     max_spacings = spacing_tensor.max(dim=0).values

#     print("\nVoxel spacing ranges across samples:")
#     for i, (min_val, max_val) in enumerate(zip(min_spacings.tolist(), max_spacings.tolist(), strict=False)):
#         print(f"Axis {i}: Min = {min_val:.6f} mm, Max = {max_val:.6f} mm")
# else:
#     print("No voxel spacings were analyzed.")


# %% [markdown]
# **Shape ranges across samples:**  
# - Dimension 0: Min = 512, Max = 512  
# - Dimension 1: Min = 512, Max = 512  
# - Dimension 2: Min = 206, Max = 277  
#
# **Voxel spacing ranges across samples:**  
# - Axis 0: Min = 0.316406 mm, Max = 0.433594 mm  
# - Axis 1: Min = 0.316406 mm, Max = 0.433594 mm  
# - Axis 2: Min = 0.500000 mm, Max = 0.500000 mm  
#
# => use unified voxel spacing: 0.5 / 0.5 / 0.5
#
# After Update:  
# **Shape ranges across samples:**  
# - Dimension 0: Min = 512, Max = 512
# - Dimension 1: Min = 324, Max = 444
# - Dimension 2: Min = 132, Max = 239 
#
# **Voxel spacing ranges across samples:**  
# - Axis 0: Min = 0.500000 mm, Max = 0.500000 mm  
# - Axis 1: Min = 0.500000 mm, Max = 0.500000 mm  
# - Axis 2: Min = 0.500000 mm, Max = 0.500000 mm  

# %% [markdown]
# sigma_ratio: 0.2
#
# True values: 0, False values: 884736
# True values: 0, False values: 884736
# True values: 0, False values: 884736
# True values: 0, False values: 884736
# True values: 2617, False values: 882119
# True values: 0, False values: 884736
# True values: 8064, False values: 876672
# True values: 0, False values: 884736
# True values: 313, False values: 884423
# True values: 6645, False values: 878091

# %%
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
img, mask = ds[0]
print(f"{mask.shape=}")
binary_volume = mask.squeeze().numpy()
print(f"{binary_volume.shape=}")
# Count the number of True and False values in the binary_volume
true_count = (binary_volume > 0).sum()
false_count = binary_volume.size - true_count

print(f"True values: {true_count}, False values: {false_count}")
downsample_factor = 0.25
target_shape = tuple(int(s * downsample_factor) for s in binary_volume.shape)

downsampled_volume = resize(
    binary_volume,
    output_shape=target_shape,
    preserve_range=True,
    anti_aliasing=False,
    mode="constant"
)
print(f"{downsampled_volume.shape=}")

# %%
plot_3d_volume(binary_volume=downsampled_volume)

# %%
import plotly.graph_objects as go

# Assume `binary_volume` is a 3D numpy array
fig = go.Figure(data=go.Volume(
    x=np.arange(downsampled_volume.shape[0]),
    y=np.arange(downsampled_volume.shape[1]),
    z=np.arange(downsampled_volume.shape[2]),
    value=downsampled_volume.flatten(),
    opacity=0.1,
    surface_count=20
))
fig.show()

# would be nice to have something interactive that is a bit faster, why doesn't it work?
# TODO: here also check provided examples

