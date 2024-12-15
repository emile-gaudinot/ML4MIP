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
from ml4mip.dataset import get_dataset, DatasetConfig,TransformType
from ml4mip.visualize import project_mask_2d

# %%
cfg = DatasetConfig(
    mask_dir = None,
    transform=TransformType.RESIZE,
)

ds_train_resize, _ = get_dataset(cfg)

# %%
img, mask = ds_train_resize[0]

# %%
project_mask_2d(mask.squeeze().numpy())

# %%
empty_mask = mask * 0
project_mask_2d(empty_mask.squeeze().numpy())

# %%
mask_with_single_one = empty_mask.clone()
mask_with_single_one[0, 0, 48:52, 48:52] = 1
project_mask_2d(mask_with_single_one.squeeze().numpy())

# %%
mask_with_all_ones = empty_mask.clone()
mask_with_all_ones[:, :, :, :] = 1
project_mask_2d(mask_with_all_ones.squeeze().numpy())


# %%
# add batch dim, the batch size is 1
mask = mask.unsqueeze(0) if mask.dim() < 5 else mask
empty_mask = empty_mask.unsqueeze(0) if empty_mask.dim() < 5 else empty_mask
mask_with_all_ones = mask_with_all_ones.unsqueeze(0) if mask_with_all_ones.dim() < 5 else mask_with_all_ones
print(mask.shape, empty_mask.shape, mask_with_all_ones.shape)

# %%
from monai.losses import DiceLoss


loss = DiceLoss(include_background=True)
pred = empty_mask
loss(pred, mask), loss(pred, empty_mask)

# %%
loss = DiceLoss(include_background=False)
pred = empty_mask
loss(pred, mask), loss(pred, empty_mask)

# %%
pred = mask_with_single_one
loss(pred, mask), loss(pred, empty_mask)

# %%
pred = mask_with_all_ones
loss(pred, mask), loss(pred, empty_mask)

# %% [markdown]
# here there is no difference between 'mask_with_single_one' and 'mask_with_all_ones'

# %%
from monai.losses import DiceCELoss

# Initialize Dice Cross Entropy Loss
loss = DiceCELoss(
    include_background=True,  # Whether to include background in the loss computation
    to_onehot_y=False,         # Convert target to one-hot encoding
    softmax=False,             # Apply softmax to predictions
    lambda_dice=0.7,          # Weight for the Dice Loss component
    lambda_ce=0.3             # Weight for the Cross Entropy Loss component
)

pred = mask
loss(pred, mask), loss(pred, empty_mask)

# %%
pred = mask_with_all_ones
loss(pred, mask), loss(pred, empty_mask)

# %%
pred = mask_with_single_one
loss(pred, mask), loss(pred, empty_mask)

# %%
pred = empty_mask
loss(pred, mask), loss(pred, empty_mask)
