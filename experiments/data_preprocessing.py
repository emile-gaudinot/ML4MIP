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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%

import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    RandSpatialCropd,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    Spacingd,
    ToTensord,
)
from skimage.transform import resize
from tqdm.notebook import tqdm

from ml4mip.dataset import NiftiDataset, TruncatedGaussianRandomCrop
from ml4mip.visualize import plot_3d_volume

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


fig, ax1 = plt.subplots(figsize=(12, 6))

# Create bar positions
x = np.arange(len(pipelines))
width = 0.35  # Width of the bars

# Plot count_positive on the left y-axis
ax1.bar(
    x - width / 2,
    count_positive_values,
    width,
    label=f"Count Positive x/{NUM_SAMPLES}",
    color="blue",
    alpha=0.7,
    edgecolor="black",
)
ax1.set_ylabel("Count Positive", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.set_xticks(x)
ax1.set_xticklabels(pipelines, rotation=15, ha="right")

# Add the second y-axis for avg_portion_positive
ax2 = ax1.twinx()
ax2.bar(
    x + width / 2,
    avg_portion_positive_values,
    width,
    label="Avg Portion Positive",
    color="orange",
    alpha=0.7,
    edgecolor="black",
)
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

