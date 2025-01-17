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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import logging
from pathlib import Path

import numpy as np
import torch
from monai.transforms import Compose, ResizeWithPadOrCrop, SaveImage, Spacing

from ml4mip.dataset import GraphDataset, get_scaling_transform
from ml4mip.models import ModelConfig, ModelType, get_model
from ml4mip.trainer import InferenceConfig, InferenceMode, inference
from ml4mip.visualize import (
    display_projection_comparison,
)

logger = logging.getLogger(__name__)
device = torch.device("cuda")

# %% [markdown]
# ### 1. Load Model and Dataset

# %%
model_cfg = ModelConfig(
    model_type=ModelType.UNETMONAI2,
    model_path="/group/cake/ML4MIP/models/unetmonai2_resize192_192_96_dice/final_model.pt",
)
model = get_model(model_cfg)
_ = model.eval()
model = model.to(device)

OUTPUT_DIR = Path("/group/cake/ML4MIP/data/inference/unetmonai2_resize192_192_96_dice")

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# important for this model, bc it was trained min max scaled data.
transform = get_scaling_transform()
ds = GraphDataset(transform=transform)


# %% [markdown]
# ### 2. Reshape to original space

# %%
def reshape_to_original(mask):
    original_shape = img.meta.get("spatial_shape")
    original_pixel_spacing = img.meta.get("pixdim")
    if original_shape is None:
        msg = "Original shape information is not available in the metadata."
        raise ValueError(msg)
    if original_pixel_spacing is None:
        msg = "Original pixel spacing information is not available in the metadata."
        raise ValueError(msg)
    return Compose(
        [
            Spacing(
                pixdim=original_pixel_spacing[1:4].tolist(),
                mode="nearest",
            ),
            ResizeWithPadOrCrop(
                spatial_size=original_shape.tolist(),
                mode="nearest",  # Use 'nearest' for binary segmentation masks
            ),
        ]
    )(mask)



# %%
item = ds[0]
img = item["image"]
msk = item["mask"]

reshaped_msk = reshape_to_original(msk)
mask = reshaped_msk.squeeze(0).cpu().numpy()
raw_mask = item["raw_mask"].squeeze(0).cpu().numpy()

display_projection_comparison(mask, raw_mask)

# %% [markdown]
# ### 3. Run inference and save predictions

# %%


# Initialize the SaveImage transform
save_pred = SaveImage(
    output_dir=OUTPUT_DIR,
    output_ext="pred.nii.gz",
    separate_folder=False,
    print_log=True,
)

save_label = SaveImage(
    output_dir=OUTPUT_DIR,
    output_ext="label.nii.gz",
    separate_folder=False,
    print_log=True,
)

# img.shape, msk.shape, graph
with torch.no_grad():
    n_samples = 20
    for idx in range(n_samples):
        item = ds[idx]
        img, raw_msk, graph_file = item["image"], item["raw_mask"], item["graph"]
        img = img.unsqueeze(0).to(device)
        float_mask = inference(
                images=img,
                model=model,
                cfg=InferenceConfig(
                    mode=InferenceMode.RESCALE_BINARY, model_input_size=(192, 192, 96), sw_batch_size=1
                ),
            )
        pred_mask = (float_mask >= 0.5).squeeze(0)

        reshaped_mask = reshape_to_original(pred_mask)
        reshaped_mask = reshaped_mask.squeeze(0).cpu()

        # Save the prediction mask
        save_pred(reshaped_mask, meta_data=reshaped_mask.meta)
        save_label(raw_msk, meta_data=raw_msk.meta)

        # id_ = graph_file.stem.split(".")[0]
        # pred_path = OUTPUT_DIR / f"{id_}.pred.npy"  # Unique filename for each image
        # np.save(pred_path, reshaped_mask)
        # print(f"Saved prediction mask as {pred_path}")

        # mask_path = OUTPUT_DIR / f"{id_}.mask.npy"
        # np.save(mask_path, raw_msk.squeeze(0).cpu().numpy())
        # print(f"Saved mask as {mask_path}")

for file in OUTPUT_DIR.glob("*_trans*"):
    # Remove 'label_trans.' from the filename
    new_name = file.name.replace("img_trans.", "").replace("label_trans.", "")
    new_path = file.with_name(new_name)
    file.rename(new_path)
    print(f"Renamed {file.name} to {new_name}")

# %%
import nibabel as nib

# Load the NIfTI file
id_ = "000f21"
id_ = "00328c"

pred = nib.load(OUTPUT_DIR / f"{id_}.pred.nii.gz")
mask = nib.load(OUTPUT_DIR / f"{id_}.label.nii.gz")
print(f"{pred.shape=}, {mask.shape=}")
print(f"{pred.header['pixdim'][1:4]=}, {mask.header['pixdim'][1:4]=}")

# convert to numpy
pred_data = pred.get_fdata()
mask_data = mask.get_fdata()

display_projection_comparison(mask_data, pred_data)
