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
from ml4mip.dataset import DatasetConfig, TransformType, get_dataset, MaskOperations
from ml4mip.visualize import project_mask_2d, plot_3d_view, plot_3d_volume
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

# %%
cfg_cache = DatasetConfig(
    data_dir="/data/ML4MIP-Data/preprocessed_data/training/",
    mask_dir="/data/ML4MIP-Data/preprocessed_data/training/",
    image_suffix=".img.nii.gz",
    mask_suffix=".label.nii.gz",
    image_prefix="hepaticvessel_",
    mask_prefix="hepaticvessel_",
    split_ratio=0.9,
    use_preprocessed_dataset=True,
)

# %%
cfg_online = DatasetConfig(
    data_dir="/data/ML4MIP-Data/Task08_HepaticVessel/imagesTr",
    mask_dir="/data/ML4MIP-Data/Task08_HepaticVessel/labelsTr",
    image_suffix=".nii.gz",
    mask_suffix=".nii.gz",
    image_prefix="hepaticvessel_",
    mask_prefix="hepaticvessel_",
    transform=TransformType.PATCH_CENTER_GAUSSIAN,
    split_ratio=0.9,
    target_spatial_size=[128, 128, 128],
    target_pixel_dim=[1.0, 1.0, 1.0],
    mask_operation=MaskOperations.BINARY_CLASS
)

ds, _ = get_dataset(cfg_cache)
train_loader = DataLoader(
    ds, batch_size=16, shuffle=True, pin_memory=torch.cuda.is_available()
)

ds_online, _ = get_dataset(cfg_online)

if False:
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")

    for batch in progress_bar:
        images, masks = batch
        print('mask')
        print(f"shape: {masks.shape}")
        print(f"max: {masks.max()}")
        print(f"min: {masks.min()}")
        print(f"count: {masks.sum()}")
        print(torch.histogram(masks, bins=10, density=True))

        print('image')
        print(f"shape: {images.shape}")
        print(f"max: {images.max()}")
        print(f"min: {images.min()}")
        print(f"count: {images.sum()}")
        print(torch.histogram(images, bins=10, density=True))
        break

# %%
img, mask = ds_online[54]
plot_3d_volume(mask.squeeze().numpy(), voxel_limit=100_000)

# %%
print('mask 9')
print(torch.histogram(mask, bins=10, density=True))
#project_mask_2d(img.squeeze().numpy())
project_mask_2d((mask==0).squeeze().numpy())
project_mask_2d((mask==1).squeeze().numpy())
project_mask_2d((mask==2).squeeze().numpy())