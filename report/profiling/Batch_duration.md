# Batch duration analysis

batch_size: 16
model_type: UNETMONAI1
transform: PATCH_CENTER_GAUSSIAN
size:
  - 96
  - 96
  - 96
target_pixel_dim:
  - 1.0
  - 1.0
  - 1.0
target_spatial_size:
  - 128
  - 128
  - 128

ssd live load and trasnform:
per batch: 120s 

ram cached:
per batch: 13s
