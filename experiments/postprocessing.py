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
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml4mip.dataset import DataLoaderConfig, DatasetConfig, TransformType, get_dataset
from ml4mip.models import ModelConfig, ModelType, get_model
from ml4mip.trainer import InferenceConfig, InferenceMode, inference
from ml4mip.visualize import (
    display_comparison_slices,
    display_projection_comparison,
    plot_3d_comparison,
)

logger = logging.getLogger(__name__)
device = torch.device("cuda")


# %%
data_cfg = DataLoaderConfig(
    train=DatasetConfig(
        # data_dir = '/Users/pvmeng/Documents/ML4MIP/data/tiny_data',
        transform=TransformType.RESIZE,
        # size=24,
        # image_suffix="avg.nii.gz",
        # mask_suffix="avg_seg.nii.gz",
        # target_pixel_dim=(1.0, 1.0, 1.0),
        # target_spatial_size=(128, 128, 128),
    ),
    val=DatasetConfig(
        transform=TransformType.STD,
        max_samples=20,
    ),
)
_, ds_val = get_dataset(data_cfg)
val_loader = DataLoader(
    ds_val, batch_size=1, shuffle=False, pin_memory=torch.cuda.is_available()
)

print(len(ds_val))

model_cfg = ModelConfig(
    model_type=ModelType.UNETMONAI2,
    model_path="/group/cake/ML4MIP/models/unetmonai2_resize192_192_96_dice/final_model.pt",
)
model = get_model(model_cfg)
_ = model.eval()
model = model.to(device)

OUTPUT_DIR = Path("/group/cake/ML4MIP/data/inference/unetmonai2_resize192_192_96_dice")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# %% [markdown]
# # Steps
#
# - Load model
# - apply inference
# - save output
# - apply post processing method
# - validate with actual validation example

# %%
with torch.no_grad():
    for idx, (img, mask) in enumerate(val_loader):
        img = img.to(device)
        batch = img.to(device)
        float_mask = inference(
            images=batch,
            model=model,
            cfg=InferenceConfig(
                mode=InferenceMode.RESCALE_BINARY, model_input_size=(192, 192, 96), sw_batch_size=1
            ),
        )
        pred_mask = (float_mask >= 0.5).squeeze(0).squeeze(0).cpu().numpy()
        # Save as a .npy file
        pred_path = OUTPUT_DIR / f"pred_{idx}.npy"  # Unique filename for each image
        np.save(pred_path, pred_mask)
        print(f"Saved prediction mask as {pred_path}")

        mask_path = OUTPUT_DIR / f"mask_{idx}.npy"
        np.save(mask_path, mask.squeeze(0).squeeze(0).cpu().numpy())
        print(f"Saved mask as {mask_path}")


# %%
idx = 19
# get output mask from file
pred = np.load(OUTPUT_DIR / f"pred_{idx}.npy")
mask = np.load(OUTPUT_DIR / f"mask_{idx}.npy")
display_projection_comparison(mask, pred)
print("Shape of pred: ", pred.shape)
print("Unique values in pred: ", np.unique(pred))

# %%
plot_3d_comparison(mask, pred, voxel_limit=500000)

# %%
img = ds_val[0][0].squeeze(0).cpu().numpy()
display_comparison_slices(img, mask, pred)


# %%
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, pred_dir, mask_dir=None):
        if mask_dir is None:
            mask_dir = pred_dir
        self.pred_files = sorted(Path(pred_dir).glob("pred_*.npy"), key=lambda x: int(x.stem.split("_")[1]))
        print(self.pred_files)
        self.mask_files = sorted(Path(mask_dir).glob("mask_*.npy"), key=lambda x: int(x.stem.split("_")[1]))
        assert len(self.pred_files) == len(self.mask_files)

    def __len__(self):
        return len(self.pred_files)

    def __getitem__(self, idx):
        return np.load(self.pred_files[idx]), np.load(self.mask_files[idx])


ds = SegmentationDataset(OUTPUT_DIR)
len(ds)

# %%
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree


def connected_component_filter(pred, min_size=100):
    # Perform connected component analysis
    structure = np.ones((3, 3, 3), dtype=np.int64)  # Define connectivity (6, 18, or 26)
    labeled_array, num_features = label(pred, structure=structure)

    # Calculate sizes of connected components
    component_sizes = np.bincount(labeled_array.ravel())

    # Remove the background size (component 0)
    component_sizes[0] = 0

    # Create a mask to keep components larger than the threshold
    filtered_components = (
        np.isin(labeled_array, np.where(component_sizes >= min_size)[0]) if min_size > 0 else pred
    )

    # Count the number of connected components that are larger than the threshold
    num_filtered_components = np.sum(component_sizes >= min_size)

    return filtered_components, num_filtered_components


def compute_minimum_distances(component_points, valid_labels):
    num_labels = len(valid_labels)
    distances = np.inf * np.ones(
        (num_labels, num_labels)
    )  # Initialize distance matrix with infinity

    for i in range(num_labels):
        points_a = component_points[valid_labels[i]]
        tree_a = cKDTree(points_a)
        for j in range(i + 1, num_labels):
            points_b = component_points[valid_labels[j]]
            distances[i, j] = distances[j, i] = tree_a.query(points_b, k=1)[0].min()

    return distances


def connected_component_distance_filter(pred, min_size=300, max_dist=50):
    structure = np.ones((3, 3, 3), dtype=np.int64)  # Define connectivity (6, 18, or 26)
    labeled_array, _ = label(pred, structure=structure)
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0  # Remove background size
    valid_labels = np.where(component_sizes >= min_size)[0]
    component_points = {label: np.argwhere(labeled_array == label) for label in valid_labels}
    distances = compute_minimum_distances(component_points, valid_labels)
    min_distances = distances.min(axis=1) <= max_dist

    to_keep = valid_labels[min_distances]
    # Create a mask with the filtered components
    filtered_mask = np.isin(labeled_array, list(to_keep)).astype(np.uint8)

    # Count the final number of components
    final_num_components = len(to_keep)

    return filtered_mask, final_num_components


# Load prediction mask
idx = 19
pred, mask = ds[idx]
filtered_pred, n_components = connected_component_distance_filter(pred, min_size=100, max_dist=100)
# Display comparison (optional)
display_projection_comparison(mask, filtered_pred), n_components


# %%
import gc
import statistics
from collections.abc import Callable

from tqdm import tqdm

from ml4mip.utils.metrics import get_metrics


def run_function_on_ds(
    ds: torch.utils.data.Dataset,
    eval_function: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    function: Callable,
    **kwargs,
):
    results = []
    for idx in tqdm(range(len(ds)), leave=False):
        pred, mask = ds[idx]
        output, n_components = function(pred, **kwargs)
        score = eval_function(mask, output)
        score["n_components"] = n_components
        print(score)
        results.append(score)
        # Run garbage collection
        gc.collect()
        # Clear the GPU cache
        torch.cuda.empty_cache()

    return {key: statistics.mean([result[key] for result in results]) for key in results[0].keys()}


def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2.0 * intersection) / (np.sum(y_true) + np.sum(y_pred))


metrics = get_metrics()
metrics.sigmoid = False


def eval_function(y_true, y_pred):
    torch_y_true = torch.tensor(y_true).unsqueeze(0).unsqueeze(0).float()
    torch_y_pred = torch.tensor(y_pred).unsqueeze(0).unsqueeze(0).float()
    metrics(torch_y_true, torch_y_pred)
    scores = metrics.aggregate()
    metrics.reset()
    return scores



# %%
# distance_values = [50, 60, 70, 80, 90, 100]
size_values = [0, 100, 200, 300, 400, 500, 600, 700]
for size in size_values:
    print(f"Size: {size}")
    res = run_function_on_ds(
        ds,
        eval_function,
        connected_component_filter,
        # connected_component_distance_filter,
        min_size=size,
        # max_dist=d,
    )
    print(f"Size: {size}, Scores: {res}")

# %%

thresholds = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for t in distance_values:
    res = run_function_on_ds(
        ds,
        dice_coefficient,
        connected_component_filter,
        min_size=t,
    )
    print(f"Size: {t}, Dice: {res}")

# %% [markdown]
# # Next steps
#
# - Nick stops process and generates some outputs
# - So maybe sliding window inference on 10 samples
