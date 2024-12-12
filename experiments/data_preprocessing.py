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
import os
import pathlib
import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from monai.data.utils import affine_to_spacing, compute_shape_offset, to_affine_nd, zoom_affine
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    RandSpatialCropd,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    Spacing,
    Spacingd,
    ToTensord,
)
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm

from ml4mip.dataset import (
    NiftiDataset,
    PositiveBiasedRandomCrop,
    TruncatedGaussianRandomCrop,
    get_patch_center_gaussian_transform,
    get_patch_positive_center_transform,
)
from ml4mip.visualize import plot_3d_volume


# %%
# Utility functions for collection processing

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


def process_directory_for_resampling(directory_path, target_pixdim, predict_shape: Callable):
    """Process all NIfTI files in a directory and predict their new shape."""
    nifti_files = [f for f in os.listdir(directory_path) if f.endswith(".nii.gz")]
    resampling_predictions = []

    for nifti_file in nifti_files:
        file_path = os.path.join(directory_path, nifti_file)
        current_shape, current_pixdim, new_shape = predict_shape(file_path, target_pixdim)
        resampling_predictions.append({
            "filename": nifti_file,
            "current_shape": tuple(current_shape),
            "current_pixdim": tuple(current_pixdim),
            "new_shape": tuple(new_shape),
            "target_pixdim": target_pixdim,
        })

    return resampling_predictions


# %%
def predict_new_shape(file_path, target_pixdim, diagonal=False, scale_extent=False):
    """Predict the new shape of a NIfTI (emulating MONAI's Spacing)."""
    nii = nib.load(file_path)
    header = nii.header
    affine = nii.affine
    current_shape = np.array(header.get_data_shape())

    # Convert the affine matrix to n-dimensional affine
    sr = len(current_shape)  # Spatial rank (dimensions)
    affine_nd = to_affine_nd(sr, affine)  # Convert to spatial affine matrix

    # Extract voxel spacing from n-dimensional affine matrix
    original_spacing = affine_to_spacing(affine_nd, sr)

    # Compute new affine matrix for target spacing
    new_affine = zoom_affine(affine_nd, target_pixdim, diagonal=diagonal)

    # Compute output shape and offset
    output_shape, offset = compute_shape_offset(
        current_shape,
        affine_nd,
        new_affine,
        scale_extent,
    )

    # Round output shape to integers
    new_shape = np.round(output_shape).astype(int)
    return current_shape, original_spacing, new_shape

# MONAI pipeline to resample image
def monai_resample_image(file_path, target_pixdim):
    # Load the image
    loader = LoadImage(image_only=True)
    ensure_channel_first = EnsureChannelFirst()

    image = loader(file_path)
    image = ensure_channel_first(image)

    # Apply Spacing transform
    spacing_transform = Spacing(pixdim=target_pixdim, mode="bilinear")
    resampled_image = spacing_transform(image)

    # Get the new shape
    new_shape = resampled_image.shape[1:]
    return new_shape



# Target pixel dimensions
target_pixel_dimensions = (0.3, 0.3, 0.5)

# Directory containing your NIfTI files
nifti_directory = pathlib.Path("/data/training_data")
mask_file = nifti_directory / "188c1f.label.nii.gz"
image_file = nifti_directory / "188c1f.img.nii.gz"


# Test if new shape prediction is correct and works as the MONAI pipeline
_,_, new_shape = predict_new_shape(image_file, target_pixel_dimensions)
new_shape_monai = monai_resample_image(image_file, target_pixel_dimensions)
assert np.allclose(new_shape, new_shape_monai), "Shapes do not match!"


# %%

def plot_shape_distribution_with_percentile(resampling_predictions, percentile=90, key="new_shape"):
    # Extract x, y, z dimensions from the new shape data
    x_dims = [pred[key][0] for pred in resampling_predictions]  # X-dimension
    y_dims = [pred[key][1] for pred in resampling_predictions]  # Y-dimension
    z_dims = [pred[key][2] for pred in resampling_predictions]  # Z-dimension

    # Convert to DataFrame for easier plotting
    data = pd.DataFrame({"X-Dimension": x_dims, "Y-Dimension": y_dims, "Z-Dimension": z_dims})

    # Create a single figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    fig.suptitle(
        f"Distribution of {key} (X, Y, Z Dimensions) - {percentile}% Range Included",
        fontsize=16,
    )

    # Plot histograms for each dimension and calculate the range
    for dimension, ax in zip(data.columns, axes, strict=False):
        lower_bound = data[dimension].quantile((100 - percentile) / 200)
        upper_bound = data[dimension].quantile(1 - (100 - percentile) / 200)
        mean = data[dimension].mean()
        ax.hist(data[dimension], bins=20, alpha=0.7, edgecolor="black")
        ax.axvline(lower_bound, color="red", linestyle="--", label=f"Lower: {lower_bound:.2f}")
        ax.axvline(upper_bound, color="blue", linestyle="--", label=f"Upper: {upper_bound:.2f}")
        ax.set_title(
            f"{dimension}: [{lower_bound:.2f}, {upper_bound:.2f}] #{mean:.2f}", color="black"
        )
        ax.set_xlabel(f"{dimension} ({key})", color="black")
        ax.set_ylabel("Frequency", color="black")
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


TARGET_PIXEL_DIM = (0.35, 0.35, 0.5)
TARGET_SPATIAL_SIZE = (600, 600, 280)  # new Target spatial size (600 x 600 x 280)

resampling_metadata = process_directory_for_resampling(
    nifti_directory,
    target_pixel_dimensions,
    predict_new_shape,
)
# Example: Call the function with your data
plot_shape_distribution_with_percentile(
    resampling_metadata,
    percentile=90,
    key="current_pixdim",
    # key="new_shape",
)


# %%
transform = TruncatedGaussianRandomCrop(
    keys=["image", "label"],
    roi_size=(96, 96),
    sigma_ratio=0.1,
)
img_shape = np.array([600, 600])  # Depth, Height, Width
transform.plot_distributions(img_shape, num_samples=10000)

# %% [markdown]
# # Pipeline definition

# %%
# this really depends on the input range. it could happen that the range is not meaningful
scale_transform = ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)

loader = LoadImage(image_only=True, ensure_channel_first=True)
mask = loader(mask_file)
img = loader(image_file)

volume = mask.as_tensor().squeeze().numpy() > 0
print(mask.pixdim, mask.shape)
positive_count = (volume > 0).sum()
total_count = volume.size
print(f"Positive count: {positive_count} / {total_count} ({positive_count / total_count:.2%})")
plot_3d_volume(volume, voxel_limit=100_000)

# %%
resample_transform = Spacingd(
    keys=["image", "mask"],  # Keys to apply the transform
    pixdim=TARGET_PIXEL_DIM,  # Desired voxel spacing in mm
    mode=("bilinear", "nearest"),  # Interpolation modes for image and mask
)

pipeline_no_resize = Compose(
    [
        resample_transform,
        scale_transform,
        ToTensord(keys=["image", "mask"]),
    ]
)

output = pipeline_no_resize({"image": img, "mask": mask})
volume = output["mask"].squeeze().numpy() > 0
print(output["mask"].pixdim, output["mask"].shape)
positive_count = (volume > 0).sum()
total_count = volume.size
print(f"Positive count: {positive_count} / {total_count} ({positive_count / total_count:.2%})")
plot_3d_volume(volume, voxel_limit=100_000)

# %%
resize_transform = ResizeWithPadOrCropd(
    keys=["image", "mask"],
    spatial_size=TARGET_SPATIAL_SIZE,
)

pipeline_std = Compose(
    [
        resample_transform,
        resize_transform,
        scale_transform,
        ToTensord(keys=["image", "mask"]),
    ]
)

output = pipeline_std({"image": img, "mask": mask})
volume = output["mask"].squeeze().numpy() > 0
print(output["mask"].pixdim, output["mask"].shape)
positive_count = (volume > 0).sum()
total_count = volume.size
print(f"Positive count: {positive_count} / {total_count} ({positive_count / total_count:.2%})")
plot_3d_volume(volume, voxel_limit=500_000)

# %%
size = 96
pipeline = Compose(
    [
        resample_transform,
        resize_transform,
        scale_transform,
        TruncatedGaussianRandomCrop(
            keys=["image", "mask"],
            roi_size=(size, size, size),
            sigma_ratio=0.1,
        ),
        ToTensord(keys=["image", "mask"]),
    ]
)

output = pipeline({"image": img, "mask": mask})
volume = output["mask"].squeeze().numpy() > 0
print(output["mask"].pixdim, output["mask"].shape)
positive_count = (volume > 0).sum()
total_count = volume.size
print(f"Positive count: {positive_count} / {total_count} ({positive_count / total_count:.2%})")
plot_3d_volume(volume, voxel_limit=100_000)

# %%
size = 96
pipeline = Compose(
    [
        resample_transform,
        resize_transform,
        scale_transform,
        PositiveBiasedRandomCrop(
                keys=["image", "mask"],
                positive_key="mask",
                roi_size=(size, size, size),
                positive_probability=1,
            ),
        ToTensord(keys=["image", "mask"]),
    ]
)

output = pipeline({"image": img, "mask": mask})
volume = output["mask"].squeeze().numpy() > 0
print(output["mask"].pixdim, output["mask"].shape)
positive_count = (volume > 0).sum()
total_count = volume.size
print(f"Positive count: {positive_count} / {total_count} ({positive_count / total_count:.2%})")
plot_3d_volume(volume, voxel_limit=100_000)

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
#         - PositiveBiasedRandomCrop (p=0.75)
#         - PositiveBiasedRandomCrop (p=1.)

# %%
size = 96
pipeline_rand = Compose(
    [
        resample_transform,
        resize_transform,
        scale_transform,
        RandSpatialCropd(
            keys=["image", "mask"],
            roi_size=(size, size, size),  # Size of the patches
            random_size=False,  # Ensure fixed-size patches
        ),
    ]
)


pipeline_gaus_high = get_patch_center_gaussian_transform(
    size=(size, size, size),
    target_pixel_dim=TARGET_PIXEL_DIM,
    target_spatial_size=TARGET_SPATIAL_SIZE,
    sigma_ratio=0.2,
)

pipeline_gaus_mid = get_patch_center_gaussian_transform(
    size=(size, size, size),
    target_pixel_dim=TARGET_PIXEL_DIM,
    target_spatial_size=TARGET_SPATIAL_SIZE,
    sigma_ratio=0.1,
)

pipeline_gaus_low = get_patch_center_gaussian_transform(
    size=(size, size, size),
    target_pixel_dim=TARGET_PIXEL_DIM,
    target_spatial_size=TARGET_SPATIAL_SIZE,
    sigma_ratio=0.05,
)

pipeline_positive_center_mid = get_patch_positive_center_transform(
    size=(size, size, size),
    target_pixel_dim=TARGET_PIXEL_DIM,
    target_spatial_size=TARGET_SPATIAL_SIZE,
    pos_center_prob=0.75,
)

pipeline_positive_center_high = get_patch_positive_center_transform(
    size=(size, size, size),
    target_pixel_dim=TARGET_PIXEL_DIM,
    target_spatial_size=TARGET_SPATIAL_SIZE,
    pos_center_prob=1.,
)



def run_dataset_sampling(
    dataset,
    analyze_sample,
    num_samples=100,
    batch_size=1,
    num_workers=0,
    pin_memory=False,
):
    # Randomly sample indices from the dataset
    indices = random.sample(range(len(dataset)), num_samples)
    subset = Subset(dataset, indices)  # Subset the dataset for selected indices

    # Create a DataLoader for the subset
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    results = []

    # Use DataLoader to fetch data and analyze
    idx = 0
    for batch in tqdm(dataloader, desc="Analyzing samples"):
        for img, mask in zip(*batch, strict=True):
            res = analyze_sample({"image": img, "mask": mask}, idx)
            results.append(res)
            idx += 1

    return results


def analyze_sample(data, idx):
    img = data["image"]
    mask = data["mask"]

    binary_volume = mask.squeeze().numpy()
    positive_count = (binary_volume > 0).sum()
    has_positive = positive_count > 0

    portion_positive = positive_count / binary_volume.size

    image_shape = img.shape
    voxel_spacing = img.pixdim.tolist() if hasattr(img, "pixdim") else None

    return {
        "index": idx,
        "image_shape": image_shape,
        "voxel_spacing": voxel_spacing,
        "has_positive": has_positive,
        "portion_positive": portion_positive,
    }


pipelines = {
    "Random Crop": pipeline_rand,
    # "Gaussian Crop (High Sigma)": pipeline_gaus_high,
    "Gaussian Crop (sigma=0.1)": pipeline_gaus_mid,
    "Gaussian Crop (sigma=0.05)": pipeline_gaus_low,
    "Positive Centered Crop (p=0.75)": pipeline_positive_center_mid,
    "Positive Centered Crop (p=1.0)": pipeline_positive_center_high,
}
NUM_SAMPLES = 1
results = {}
for name, pipeline in pipelines.items():
    ds = NiftiDataset(
        data_dir="/data/training_data",
        transform=pipeline,
    )
    results[name] = run_dataset_sampling(
        ds,
        analyze_sample,
        batch_size=4,
        num_samples=NUM_SAMPLES,
        num_workers=1,
        pin_memory=False,
    )


# %% [markdown]
# # Plot Experiment Results

# %%
def count_positive(items):
    return sum(res["has_positive"] for res in items)


def avg_portion_positive(items):
    return sum(res["portion_positive"] for res in items) / len(items)


mapped_results = {
    name: {"count_positive": count_positive(res), "avg_portion_positive": avg_portion_positive(res)}
    for name, res in results.items()
}

# Extract data for plotting
pipelines = list(mapped_results.keys())
count_positive_values = [mapped_results[name]["count_positive"] for name in pipelines]
avg_portion_positive_values = [
    mapped_results[name]["avg_portion_positive"] * 100 for name in pipelines
]  # Convert to percent


fig, ax1 = plt.subplots(figsize=(12, 6))

# Create bar positions
x = np.arange(len(pipelines))
width = 0.2  # Width of the bars
gap = 0.01  # Add a gap between the bars

# Plot count_positive on the left y-axis
ax1.bar(
    x - width - gap / 2,
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
    x + width + gap / 2,
    avg_portion_positive_values,
    width,
    label="Avg Portion Positive",
    color="orange",
    alpha=0.7,
    edgecolor="black",
)
ax2.set_ylabel("Avg Portion Positive (%)", color="orange")
ax2.tick_params(axis="y", labelcolor="orange")

# Add title and legend
fig.suptitle(
    "Comparison of Count Positive and Avg Portion Positive with Dual Scales", color="black"
)
ax1.legend(loc="upper left", bbox_to_anchor=(0, 1), labelcolor="black")
ax2.legend(loc="upper right", bbox_to_anchor=(1, 1), labelcolor="black")
plt.tight_layout()
plt.show()
