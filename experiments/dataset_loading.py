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

# %% [markdown]
# # Goal:
#
# - load the train/val dataset for resize and patch.

# %%
from ml4mip.dataset import DatasetConfig, TransformType, get_dataset
from ml4mip.visualize import plot_3d_volume

# %%
cfg = DatasetConfig(
    # data_dir = '/Users/pvmeng/Documents/ML4MIP/data/tiny_data',
    transform=TransformType.RESIZE,
    # size=24,
    # image_suffix="avg.nii.gz",
    # mask_suffix="avg_seg.nii.gz",
    # target_pixel_dim=(1.0, 1.0, 1.0),
    # target_spatial_size=(128, 128, 128),
)

ds_train_resize, _ = get_dataset(cfg)
img, mask = ds_train_resize[0]
print(img.shape, type(img), mask.shape, type(mask))
binary_volume = mask.squeeze().numpy().astype(bool)
plot_3d_volume(binary_volume)

# %%
cfg.transform = TransformType.PATCH_POS_CENTER
ds_train_patch, ds_val = get_dataset(cfg)
img, mask = ds_train_patch[0]
print(img.shape, type(img), mask.shape, type(mask))
binary_volume = mask.squeeze().numpy().astype(bool)
plot_3d_volume(binary_volume)

# %%
img, mask = ds_val[0]
print(img.shape, mask.shape)
binary_volume = mask.squeeze().numpy().astype(bool)
plot_3d_volume(binary_volume)
