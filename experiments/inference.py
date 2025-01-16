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

from ml4mip.dataset import GraphDataset, get_scaling_transform
from ml4mip.models import ModelConfig, ModelType, get_model
from ml4mip.trainer import InferenceConfig, InferenceMode, inference
from ml4mip.visualize import (
    display_projection_comparison,
)

logger = logging.getLogger(__name__)
device = torch.device("cuda")

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

# %%
# important for this model, bc it was trained min max scaled data.
transform = get_scaling_transform()
ds = GraphDataset(transform=transform)

# img.shape, msk.shape, graph
with torch.no_grad():
    n_samples = 20
    for idx in range(n_samples):
        img, msk, graph_file = ds[idx]
        print(img.shape, msk.shape, graph_file)
        img = img.unsqueeze(0).to(device)
        float_mask = inference(
                images=img,
                model=model,
                cfg=InferenceConfig(
                    mode=InferenceMode.RESCALE_BINARY, model_input_size=(192, 192, 96), sw_batch_size=1
                ),
            )
        pred_mask = (float_mask >= 0.5).squeeze(0).squeeze(0).cpu().numpy()

        id_ = graph_file.stem.split(".")[0]
        pred_path = OUTPUT_DIR / f"{id_}.pred.npy"  # Unique filename for each image
        np.save(pred_path, pred_mask)
        print(f"Saved prediction mask as {pred_path}")

        mask_path = OUTPUT_DIR / f"{id_}.mask.npy"
        np.save(mask_path, msk.squeeze(0).cpu().numpy())
        print(f"Saved mask as {mask_path}")

# %%


# list files in output dir
print("Output dir contents:")
print(len(list(OUTPUT_DIR.glob("*"))))

id_ = "00328c"
id_ = "000f21"

pred = np.load(OUTPUT_DIR / f"{id_}.pred.npy")
mask = np.load(OUTPUT_DIR / f"{id_}.mask.npy")
display_projection_comparison(mask, pred)



