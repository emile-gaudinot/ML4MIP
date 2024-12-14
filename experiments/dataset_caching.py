from ml4mip.dataset import DatasetConfig, TransformType, get_dataset
from ml4mip.visualize import plot_3d_volume

# %%
cfg = DatasetConfig(
    data_dir="/data/ML4MIP-Data/Task08_HepaticVessel/imagesTr",
    mask_dir="/data/ML4MIP-Data/Task08_HepaticVessel/labelsTr",
    image_suffix=".nii.gz",
    mask_suffix=".nii.gz",
    image_prefix="hepaticvessel_",
    mask_prefix="hepaticvessel_",
    split_ratio=0.9,
    target_spatial_size=[128, 128, 128],
    target_pixel_dim=[1.0, 1.0, 1.0],
    transform=TransformType.STD,
    max_train_samples=2,
    max_val_samples=2,
    cache_train_dataset=True,
    cache_val_dataset=True,
    cache_pooling=12
)

ds_train_resize, _ = get_dataset(cfg)
ds_train_resize.save_cache()
