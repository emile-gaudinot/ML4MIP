import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    Randomizable,
    Resized,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    Spacingd,
    ToTensord,
)
from scipy.stats import truncnorm
from torch.utils.data import Dataset


class TransformType(Enum):
    RESIZE = "resize"
    PATCH = "patch"
    STD = "std"


# this was calculated after observing the maximum size of the images after the resampling
TARGET_PIXEL_DIM = (0.5, 0.5, 0.5)
# That way we don't need to add too much padding.
TARGET_SPATIAL_SIZE = (512, 448, 256)
# This was determined with some experiments. It's a good value for the sigma.
GOOD_SIGMA_RATIO = 0.1


@dataclass
class DatasetConfig:
    # Don't change these values unless you know what you are doing:
    data_dir: str = "/data/training_data"  # path to the data in the directory
    image_suffix: str = ".img.nii.gz"
    mask_suffix: str = ".label.nii.gz"
    transform: TransformType = TransformType.RESIZE
    size: tuple[int, int, int] = (96, 96, 96)
    split_ratio: float = 0.9
    target_pixel_dim: tuple[float, float, float] = TARGET_PIXEL_DIM
    target_spatial_size: tuple[int, int, int] = TARGET_SPATIAL_SIZE
    sigma_ratio: float = GOOD_SIGMA_RATIO
    max_train_samples: int | None = None
    max_val_samples: int | None = None


_cs = ConfigStore.instance()
_cs.store(
    group="dataset",
    name="base_dataset",
    node=DatasetConfig,
)


# TODO: we also need to create training data for the 2D model.
# This requires then additional data augmentation.
def get_dataset(cfg: DatasetConfig) -> tuple[Dataset, Dataset]:
    """Return the training and validation datasets."""
    train_dataset = NiftiDataset(
        cfg.data_dir,
        image_suffix=cfg.image_suffix,
        mask_suffix=cfg.mask_suffix,
        transform=get_transform(
            type_=cfg.transform,
            size=cfg.size,
            target_pixel_dim=cfg.target_pixel_dim,
            target_spatial_size=cfg.target_spatial_size,
            sigma_ratio=cfg.sigma_ratio,
        ),
        train=True,
        split_ratio=cfg.split_ratio,
        max_samples=cfg.max_train_samples,
    )

    val_dataset = NiftiDataset(
        cfg.data_dir,
        image_suffix=cfg.image_suffix,
        mask_suffix=cfg.mask_suffix,
        transform=get_transform(
            type_=TransformType.STD,
            target_pixel_dim=cfg.target_pixel_dim,
            target_spatial_size=cfg.target_spatial_size,
        ),
        train=False,
        split_ratio=cfg.split_ratio,
        max_samples=cfg.max_val_samples,
    )
    return train_dataset, val_dataset


##### Datasets #####


class NiftiDataset(Dataset):
    """PyTorch Dataset for loading 3D NIfTI images and masks from a directory.

    Parameters:
        data_dir: Path to the directory containing image and mask files.
        image_suffix: Suffix or pattern to identify image files (default: '.img.nii.gz').
        mask_suffix: Suffix or pattern to identify mask files (default: '.label.nii.gz').
        transform: A function/transform to apply to both images and masks.
    """

    def __init__(
        self,
        data_dir: str | Path,
        image_suffix: str = ".img.nii.gz",
        mask_suffix: str = ".label.nii.gz",
        transform: Callable | None = None,
        train: bool = True,
        split_ratio: float = 0.9,
        max_samples: int | None = None,
    ) -> None:
        self.data_dir: Path = Path(data_dir)
        self.image_suffix: str = image_suffix
        self.mask_suffix: str = mask_suffix
        self.transform: Callable | None = transform
        self.loader = LoadImaged(keys=["image", "mask"])

        # Collect image and mask file paths
        image_files: list[Path] = sorted(self.data_dir.glob(f"*{self.image_suffix}"))
        mask_files: list[Path] = sorted(self.data_dir.glob(f"*{self.mask_suffix}"))

        if len(image_files) == 0 or len(mask_files) == 0:
            msg = "No image or mask files found. Verify the data directory and image and mask suffixes."
            raise ValueError(msg)

        # Split the dataset into training and validation sets
        data_files = list(zip(image_files, mask_files, strict=True))
        data_files = self.get_sample(data_files, train=train, split_ratio=split_ratio)
        self.image_files, self.mask_files = zip(*data_files, strict=True)

        if max_samples is not None:
            self.image_files = self.image_files[:max_samples]
            self.mask_files = self.mask_files[:max_samples]

        # Ensure image and mask files match
        assert len(self.image_files) == len(
            self.mask_files
        ), "Number of image files and mask files must match."
        for img, mask in zip(self.image_files, self.mask_files, strict=False):
            assert img.name.replace(image_suffix, "") == mask.name.replace(
                mask_suffix, ""
            ), f"Image file {img} and mask file {mask} do not match."

    @staticmethod
    def get_sample(
        data_files: list[tuple[Path, Path]],
        train: bool = True,
        split_ratio: float = 0.9,
    ):
        random.seed(42)
        num_train = int(len(data_files) * split_ratio)
        all_indices = list(range(len(data_files)))
        train_indices = random.sample(all_indices, num_train)
        val_indices = list(set(all_indices) - set(train_indices))
        return [data_files[i] for i in (train_indices if train else val_indices)]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data_dict = {
            "image": self.image_files[idx],
            "mask": self.mask_files[idx],
        }

        # Load images and metadata
        loaded_data = self.loader(data_dict)

        # Apply additional transformations if provided
        if self.transform:
            loaded_data = self.transform(loaded_data)

        # Extract the transformed image and mask
        img = loaded_data["image"]
        mask = loaded_data["mask"]

        return img, mask


##### Transformations #####


class TruncatedGaussianRandomCrop(Randomizable, MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        roi_size: Sequence[int],
        sigma_ratio: float = 0.1,
        allow_missing_keys: bool = False,
    ):
        """Transform to crop a random region from a volume based on a truncated Gaussian distribution.

        Args:
            keys: Keys of the corresponding items to be transformed.
            roi_size: Desired output size of the crop (e.g., Depth, Height, Width for 3D).
                      Can handle 2D, 3D, or higher-dimensional data.
            sigma_ratio: Ratio to determine the standard deviation of the Gaussian
                         distribution relative to volume dimensions.
            allow_missing_keys: Don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.roi_size = np.array(roi_size if isinstance(roi_size, Sequence) else [roi_size])
        self.sigma_ratio = sigma_ratio

    def truncated_normal(self, mean, std, lower, upper):
        """Generate a sample from a truncated normal distribution."""
        a, b = (lower - mean) / std, (upper - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std, random_state=self.R)

    def sample_center(self, img_shape):
        """Sample the center of the crop based on the truncated Gaussian distribution."""
        img_shape = np.array(img_shape)
        center = img_shape // 2
        sigma = img_shape * self.sigma_ratio

        crop_center = []
        for i in range(len(img_shape)):
            lower_bound = self.roi_size[i] // 2
            upper_bound = img_shape[i] - self.roi_size[i] // 2
            crop_center.append(
                int(self.truncated_normal(center[i], sigma[i], lower_bound, upper_bound))
            )

        return np.array(crop_center)

    def plot_distributions(self, img_shape, num_samples=10000):
        """Plot the distributions of the sampled crop center for each dimension.

        Note:
        - For 2D data, generates a heatmap for the sampled (x, y) coordinates.
        - For higher-dimensional data, generates histograms for each dimension.
        """
        img_shape = np.array(img_shape)
        centers = np.array([self.sample_center(img_shape) for _ in range(num_samples)])

        if len(img_shape) == 2:  # Special case: 2D data
            # Generate 2D histogram for x, y distribution
            heatmap, xedges, yedges = np.histogram2d(
                centers[:, 0],
                centers[:, 1],
                bins=(50, 50),
                range=[[0, img_shape[0]], [0, img_shape[1]]],
            )

            # Plot heatmap
            plt.figure(figsize=(8, 6))
            plt.imshow(
                heatmap.T,
                origin="lower",
                aspect="auto",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap="viridis",
            )
            plt.colorbar(label="Frequency")
            plt.title("2D Heatmap of Sampled Crop Centers (x, y)")
            plt.xlabel("Height (y)")
            plt.ylabel("Width (x)")
            plt.grid(False)
            plt.show()

        else:  # Default case: N-Dimensional data
            fig, axes = plt.subplots(1, len(img_shape), figsize=(5 * len(img_shape), 5))
            if len(img_shape) == 1:
                axes = [axes]  # Ensure axes is iterable for 1D case
            axis_labels = [f"Dim {i}" for i in range(len(img_shape))]

            for i, ax in enumerate(axes):
                ax.hist(centers[:, i], bins=50, alpha=0.7, color="blue", edgecolor="black")
                ax.set_title(f"Distribution of {axis_labels[i]} Sampled Crop Centers")
                ax.set_xlabel(f"{axis_labels[i]} Coordinate")
                ax.set_ylabel("Frequency")
                ax.grid(True)

            plt.tight_layout()
            plt.show()

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        d = dict(data)
        if not self.allow_missing_keys and not all(k in d for k in self.keys):
            msg = f"Not all keys {self.keys} not found in data dictionary"
            raise KeyError(msg)

        for key in self.keys:
            img = d[key]
            img_shape = np.array(img.shape[1:])  # Assuming channel-first format
            crop_center = self.sample_center(img_shape)

            # Calculate the start and end points for cropping
            start = crop_center - self.roi_size // 2
            end = start + self.roi_size

            # Extract the crop
            slices = tuple(slice(s, e) for s, e in zip(start, end, strict=False))
            d[key] = img[(slice(None), *slices)]

        return d


def get_default_transforms(
    target_pixel_dim: tuple[float, float, float],
    target_spatial_size: tuple[int, int, int],
) -> list[MapTransform]:
    return [
        # 1) Ensure Channel first
        EnsureChannelFirstd(keys=["image", "mask"]),
        # 2) Resample the image and mask to have the same voxel spacing
        Spacingd(
            keys=["image", "mask"],
            pixdim=target_pixel_dim,
            mode=("bilinear", "nearest"),
        ),
        # 3) Resize the image and mask to a target spatial size without distorting the aspect ratio
        ResizeWithPadOrCropd(
            keys=["image", "mask"],
            spatial_size=target_spatial_size,
        ),
        # 4) Scale the intensity of the image to [0, 1]
        # this really depends on the input range. it could happen that the range is not meaningful
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ToTensord(keys=["image", "mask"]),
    ]


# DEFAULT_TRANSFORM_COMPONENTS = [
#     # 1) Ensure Channel first
#     EnsureChannelFirstd(keys=["image", "mask"]),
#     # 2) Resample the image and mask to have the same voxel spacing
#     Spacingd(
#         keys=["image", "mask"],
#         pixdim=TARGET_PIXEL_DIM,
#         mode=("bilinear", "nearest"),
#     ),
#     # 3) Resize the image and mask to a target spatial size without distorting the aspect ratio
#     ResizeWithPadOrCropd(
#         keys=["image", "mask"],
#         spatial_size=TARGET_SPATIAL_SIZE,
#     ),
#     # 4) Scale the intensity of the image to [0, 1]
#     # this really depends on the input range. it could happen that the range is not meaningful
#     ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
#     ToTensord(keys=["image", "mask"]),
# ]


def get_resize_transform(
    size: Sequence[int] = TARGET_SPATIAL_SIZE,
    target_pixel_dim: tuple[float, float, float] = TARGET_PIXEL_DIM,
    target_spatial_size: tuple[int, int, int] = TARGET_SPATIAL_SIZE,
):
    # Make a copy of the pipeline transforms
    transforms = get_default_transforms(target_pixel_dim, target_spatial_size)
    return Compose(
        transforms[:-1]
        + [
            Resized(
                keys=["image", "mask"],
                spatial_size=size,
                mode=("bilinear", "nearest"),
            )
        ]
        + transforms[-1:]
    )


def get_patch_transform(
    size: Sequence[int] = (96, 96, 96),
    target_pixel_dim: tuple[float, float, float] = TARGET_PIXEL_DIM,
    target_spatial_size: tuple[int, int, int] = TARGET_SPATIAL_SIZE,
    sigma_ratio: float = GOOD_SIGMA_RATIO,
):
    # Make a copy of the pipeline transforms
    transforms = get_default_transforms(target_pixel_dim, target_spatial_size)
    return Compose(
        transforms[:-1]
        + [
            TruncatedGaussianRandomCrop(
                keys=["image", "mask"],
                roi_size=size,
                sigma_ratio=sigma_ratio,
            )
        ]
        + transforms[-1:]
    )


def get_std_transform(
    target_pixel_dim: tuple[float, float, float] = TARGET_PIXEL_DIM,
    target_spatial_size: tuple[int, int, int] = TARGET_SPATIAL_SIZE,
):
    transforms = get_default_transforms(target_pixel_dim, target_spatial_size)
    return Compose(transforms)


def get_transform(
    type_: TransformType,
    size: Sequence[int] | int = 96,
    target_pixel_dim: tuple[float, float, float] = TARGET_PIXEL_DIM,
    target_spatial_size: tuple[int, int, int] = TARGET_SPATIAL_SIZE,
    sigma_ratio: float = GOOD_SIGMA_RATIO,
) -> Callable:
    """Get the transformation function based on the type."""
    size = [size] * 3 if isinstance(size, int) else size
    match type_:
        case TransformType.RESIZE:
            return get_resize_transform(size, target_pixel_dim, target_spatial_size)
        case TransformType.PATCH:
            return get_patch_transform(size, target_pixel_dim, target_spatial_size, sigma_ratio)
        case TransformType.STD:
            return get_std_transform(target_pixel_dim, target_spatial_size)
        case _:
            msg = f"Invalid transform type: {type_}"
            raise ValueError(msg)
