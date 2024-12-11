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
import torch

# Define dimensions
batch_size = 2
channels = 1
depth, height, width = 128, 128, 128

# Create a random tensor with the specified dimensions
tensor = torch.randn(batch_size, channels, depth, height, width)

# Desired output dimensions for Depth, Height, and Width
target_depth, target_height, target_width = [96, 96, 96]


import torch.nn.functional as F

# Use trilinear interpolation to resize the tensor
resized_tensor = F.interpolate(
    tensor,
    size=(target_depth, target_height, target_width),
    mode='trilinear',
    align_corners=False
)

print(f"Original shape: {tensor.shape}")
print(f"Resized shape: {resized_tensor.shape}")



