import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from multiprocessing import Pool
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    LoadImaged,
    MapTransform,
    Randomizable,
    Resized,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    Spacingd,
    ToTensord,
    RandSpatialCropd,
)
from scipy.stats import truncnorm
from torch.utils.data import Dataset


class TransformType(Enum):
    RESIZE = "resize"
    PATCH_CENTER_GAUSSIAN = "patch_center_gaussian"
    PATCH_POS_CENTER = "patch_pos"
    PATCH_UNIFORM = "patch_uniform"
    STD = "std"

class MaskOperations(Enum):
    BINARY_CLASS = "binary"
    STD = "std"


# this was calculated after observing the maximum size of the images after the resampling
TARGET_PIXEL_DIM = (0.35, 0.35, 0.5)
# That way we don't need to add too much padding.
TARGET_SPATIAL_SIZE = (600, 600, 280)
# This was determined with some experiments. It's a good value for the sigma.
GOOD_SIGMA_RATIO = 0.1
POS_CENTER_PROB = 0.75


@dataclass
class DatasetConfig:
    # Don't change these values unless you know what you are doing:
    data_dir: str = "/data/training_data"  # path to the data in the directory
    mask_dir: str = "/data/training_data"  # path to the data in the directory
    image_suffix: str = ".img.nii.gz"
    mask_suffix: str = ".label.nii.gz"
    image_prefix: str = ""
    mask_prefix: str = ""
    val_data_dir: str | None = None
    val_mask_dir: str | None = None
    val_image_suffix: str | None = None
    val_mask_suffix: str | None = None
    val_image_prefix: str | None = None
    val_mask_prefix: str | None = None
    transform: TransformType = TransformType.PATCH_POS_CENTER
    size: tuple[int, int, int] = (96, 96, 96)
    split_ratio: float = 0.9
    target_pixel_dim: tuple[float, float, float] = TARGET_PIXEL_DIM
    target_spatial_size: tuple[int, int, int] = TARGET_SPATIAL_SIZE
    sigma_ratio: float = GOOD_SIGMA_RATIO
    pos_center_prob: float = POS_CENTER_PROB
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    cache_train_dataset: bool = False
    cache_val_dataset: bool = False
    cache_pooling: int = 0
    use_preprocessed_dataset: bool = False
    mask_operation: MaskOperations = MaskOperations.STD
    multipatch: bool = False
    max_epochs: int = 1


_cs = ConfigStore.instance()
_cs.store(
    group="dataset",
    name="base_dataset",
    node=DatasetConfig,
)

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
        mask_dir: str | Path | None = None,
        image_suffix: str = ".img.nii.gz",
        mask_suffix: str = ".label.nii.gz",
        image_prefix: str = "",
        mask_prefix: str = "",
        transform: Callable | None = None,
        train: bool = True,
        split_ratio: float = 0.9,
        max_samples: int | None = None,
        cache: bool = False,
        cache_pooling: int = 0,
        mask_operation: MaskOperations = MaskOperations.STD
    ) -> None:
        self.use_cache = cache
        self.image_cache = []
        self.mask_cache = []
        self.data_dir: Path = Path(data_dir)
        self.mask_dir: Path = self.data_dir
        self.mask_operation = mask_operation

        if mask_dir is not None:
            self.mask_dir = Path(mask_dir)

        self.image_suffix: str = image_suffix
        self.mask_suffix: str = mask_suffix
        self.image_prefix: str = image_prefix
        self.mask_prefix: str = mask_prefix

        self.transform: Callable | None = transform
        # Initialize the loader, very import to ensure channel first!
        self.loader = LoadImaged(keys=["image", "mask"], ensure_channel_first=True)

        # Collect image and mask file paths
        image_files: list[Path] = sorted(
            self.data_dir.glob(f"{self.image_prefix}*{self.image_suffix}")
        )
        mask_files: list[Path] = sorted(
            self.mask_dir.glob(f"{self.mask_prefix}*{self.mask_suffix}")
        )

        if len(image_files) == 0:
            msg = f"No image files found. {self.data_dir}"
            raise ValueError(msg)

        if len(mask_files) == 0:
            msg = f"No mask files found. (mask_dir: {self.mask_dir})"
            raise ValueError(msg)

        # Split the dataset into training and validation sets
        data_files = list(zip(image_files, mask_files, strict=True))
        # print(data_files) # check if mapping is correct
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

        if cache:
            if cache_pooling != 0:
                result = np.array_split(range(len(self.image_files)), cache_pooling)
                with Pool(processes=cache_pooling) as pool: 
                    pooled_samples = pool.map(self.preprocess_samples, [list(part) for part in result])
                unpacked_samples = list(zip(*pooled_samples))
                self.image_cache = sum(unpacked_samples[0], [])
                self.mask_cache = sum(unpacked_samples[1], [])

            else:
                images, masks = self.preprocess_samples(range(len(self.image_files)))
                self.image_cache = images
                self.mask_cache = masks

    def preprocess_samples(
        self,
        sample_list: list
    ):
        images = []
        masks = []

        for img_num in sample_list:
            data_dict = {
                "image": self.image_files[img_num],
                "mask": self.mask_files[img_num],
            }   

            # Load images and metadata
            loaded_data = self.loader(data_dict)

            # Apply additional transformations if provided
            if self.transform:
                loaded_data = self.transform(loaded_data)

            images.append(loaded_data["image"])
            masks.append(loaded_data["mask"])

        return (images, masks)


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
        if self.use_cache:  
            img = self.image_cache[idx]
            mask = self.mask_cache[idx]
            return img, mask
        else:
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
            mask = perform_mask_transformation(loaded_data["mask"], self.mask_operation)

            return img, mask

class ProprocessedNiftiDataset(Dataset):
    """PyTorch Dataset for loading preprocessed 3D NIfTI images and masks from a directory.

    Parameters:
        data_dir: Path to the directory containing image and mask files.
        image_suffix: Suffix or pattern to identify image files (default: '.img.nii.gz').
        mask_suffix: Suffix or pattern to identify mask files (default: '.label.nii.gz').
        transform: A function/transform to apply to both images and masks.
    """

    def __init__(
        self,
        data_dir: str | Path,
        mask_dir: str | Path | None = None,
        image_suffix: str = ".img.nii.gz",
        mask_suffix: str = ".label.nii.gz",
        image_prefix: str = "",
        mask_prefix: str = "",
        max_samples: int | None = None,
        cache: bool = False,
        cache_pooling: int = 0,
        mask_operation: MaskOperations = MaskOperations.STD,
        multipatch: bool = False,
        max_epochs: int = 1,
    ) -> None:
        self.use_cache = cache
        self.image_cache = []
        self.mask_cache = []
        self.data_dir: Path = Path(data_dir)
        self.mask_dir: Path = self.data_dir
        self.epoch_counter = 0
        self.max_epoch = max_epochs
        self.mask_operation = mask_operation

        if mask_dir is not None:
            self.mask_dir = Path(mask_dir)

        self.image_suffix: str = image_suffix
        self.mask_suffix: str = mask_suffix
        self.image_prefix: str = image_prefix
        self.mask_prefix: str = mask_prefix
        self.image_files = []
        self.mask_files = []

        # Initialize the loader, very import to ensure channel first!
        self.loader = LoadImaged(keys=["image", "mask"], ensure_channel_first=True)

        if self.max_epoch >= 1:
            if multipatch: 
                raise NotImplementedError

        for epoch_num in range(self.max_epoch):
            if multipatch:            
                image_file_pattern = f"{self.image_prefix}*{self.image_suffix}"
                mask_file_pattern = f"{self.mask_prefix}*{self.mask_suffix}"
            else:
                image_file_pattern = f"{self.image_prefix}*_patch[[]{epoch_num}[]]{self.image_suffix}"
                mask_file_pattern = f"{self.mask_prefix}*_patch[[]{epoch_num}[]]{self.mask_suffix}"

            # Collect image and mask file paths
            image_files: list[Path] = sorted(
                self.data_dir.glob(image_file_pattern)
            )
            mask_files: list[Path] = sorted(
                self.mask_dir.glob(mask_file_pattern)
            )

            if len(image_files) == 0:
                msg = f"No image files found. {self.data_dir}"
                raise ValueError(msg)

            if len(mask_files) == 0:
                msg = f"No mask files found. (mask_dir: {self.mask_dir})"
                raise ValueError(msg)

            # Split the dataset into training and validation sets
            data_files = list(zip(image_files, mask_files, strict=True))
            # print(data_files) # check if mapping is correct
            data_files = self.get_sample(data_files)
            zipped_image_files, zipped_mask_files = zip(*data_files, strict=True)
            self.image_files.append(zipped_image_files)
            self.mask_files.append(zipped_mask_files)

            if max_samples is not None:
                self.image_files[epoch_num] = self.image_files[epoch_num][:max_samples]
                self.mask_files[epoch_num] = self.mask_files[epoch_num][:max_samples]

            # Ensure image and mask files match
            assert len(self.image_files[epoch_num]) == len(
                self.mask_files[epoch_num]
            ), "Number of image files and mask files must match."
            for img, mask in zip(self.image_files[epoch_num], self.mask_files[epoch_num], strict=False):
                assert img.name.replace(image_suffix, "") == mask.name.replace(
                    mask_suffix, ""
                ), f"Image file {img} and mask file {mask} do not match."

            #print(f"Epoch {epoch_num} data checked.")

        if cache:
            if self.max_epoch != 1:
                #TODO: multiepoch caching
                raise NotImplementedError

            if cache_pooling != 0:
                result = np.array_split(range(len(self.image_files[0])), cache_pooling)
                with Pool(processes=cache_pooling) as pool: 
                    pooled_samples = pool.map(self.preprocess_samples, [list(part) for part in result])
                unpacked_samples = list(zip(*pooled_samples))
                self.image_cache[0] = sum(unpacked_samples[0], [])
                self.mask_cache[0] = sum(unpacked_samples[1], [])

            else:
                images, masks = self.preprocess_samples(range(len(self.image_files)))
                self.image_cache[0] = images
                self.mask_cache[0] = masks

    def preprocess_samples(
        self,
        sample_list: list
    ):
        images = []
        masks = []

        for img_num in sample_list:
            data_dict = {
                "image": self.image_files[0][img_num],
                "mask": self.mask_files[0][img_num],
            }   

            # Load images and metadata
            loaded_data = self.loader(data_dict)
            loaded_data = ToTensord(keys=["image", "mask"])(loaded_data)

            images.append(loaded_data["image"])
            masks.append(loaded_data["mask"])

        return (images, masks)


    @staticmethod
    def get_sample(
        data_files: list[tuple[Path, Path]]
    ):
        random.seed(42)
        num_train = int(len(data_files))
        all_indices = list(range(len(data_files)))
        train_indices = random.sample(all_indices, num_train)
        return [data_files[i] for i in train_indices]

    def __len__(self) -> int:
        return len(self.image_files[self.epoch_counter % self.max_epoch])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_cache:  
            img = self.image_cache[idx]
            mask = self.mask_cache[idx]
            return img, mask
        else:
            data_dict = {
                "image": self.image_files[self.epoch_counter % self.max_epoch][idx],
                "mask": self.mask_files[self.epoch_counter % self.max_epoch][idx],
            }

            # Load images and metadata
            loaded_data = self.loader(data_dict)

            loaded_data = ToTensord(keys=["image", "mask"])(loaded_data)

            # Extract the transformed image and mask
            img = loaded_data["image"]
            mask = perform_mask_transformation(loaded_data["mask"], self.mask_operation)

            return img, mask
        
    def next_epoch(self):
        self.epoch_counter += 1

        if self.use_cache:
            #TODO: load cache for next epoch
            pass


# TODO: we also need to create training data for the 2D model.
# This requires then additional data augmentation.
def get_dataset(cfg: DatasetConfig) -> tuple[NiftiDataset, NiftiDataset] | tuple[ProprocessedNiftiDataset, NiftiDataset]:
    """Return the training and validation datasets."""

    if cfg.use_preprocessed_dataset:
        train_dataset = ProprocessedNiftiDataset(
            data_dir=cfg.data_dir,
            mask_dir=cfg.mask_dir,
            image_prefix=cfg.image_prefix,
            mask_prefix=cfg.mask_prefix,
            image_suffix=cfg.image_suffix,
            mask_suffix=cfg.mask_suffix,
            max_samples=cfg.max_train_samples,
            cache=cfg.cache_train_dataset,
            cache_pooling=cfg.cache_pooling,
            mask_operation=cfg.mask_operation,
            multipatch=cfg.multipatch,
            max_epochs=cfg.max_epochs
        )
    else:
        if cfg.multipatch:
            print('No preprocessed patched dataset. multipatch will be ignored.')

        train_dataset = NiftiDataset(
            cfg.data_dir,
            mask_dir=cfg.mask_dir,
            image_prefix=cfg.image_prefix,
            mask_prefix=cfg.mask_prefix,
            image_suffix=cfg.image_suffix,
            mask_suffix=cfg.mask_suffix,
            transform=get_transform(
                type_=cfg.transform,
                size=cfg.size,
                target_pixel_dim=cfg.target_pixel_dim,
                target_spatial_size=cfg.target_spatial_size,
                sigma_ratio=cfg.sigma_ratio,
                pos_center_prob=cfg.pos_center_prob,
            ),
            train=True,
            split_ratio=cfg.split_ratio,
            max_samples=cfg.max_train_samples,
            cache=cfg.cache_train_dataset,
            cache_pooling=cfg.cache_pooling,
            mask_operation=cfg.mask_operation,
        )

    val_dataset = NiftiDataset(
        data_dir=(cfg.val_data_dir if cfg.val_data_dir else cfg.data_dir),
        mask_dir=(cfg.val_mask_dir if cfg.val_mask_dir else cfg.mask_dir),
        image_prefix=(cfg.val_image_prefix if cfg.val_image_prefix else cfg.image_prefix),
        mask_prefix=(cfg.val_mask_prefix if cfg.val_mask_prefix else cfg.mask_prefix),
        image_suffix=(cfg.val_image_suffix if cfg.val_image_suffix else cfg.image_suffix),
        mask_suffix=(cfg.val_mask_suffix if cfg.val_mask_suffix else cfg.mask_suffix),
        transform=get_transform(
            type_=TransformType.STD,
            target_pixel_dim=cfg.target_pixel_dim,
            target_spatial_size=cfg.target_spatial_size,
        ),
        train=False,
        split_ratio=cfg.split_ratio,
        max_samples=cfg.max_val_samples,
        cache=cfg.cache_val_dataset,
        cache_pooling=cfg.cache_pooling,
        mask_operation=cfg.mask_operation,
    )
    return train_dataset, val_dataset

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


class PositiveBiasedRandomCrop(Randomizable, MapTransform):
    def __init__(
        self,
        keys,
        roi_size,
        positive_key,
        positive_probability=POS_CENTER_PROB,
        allow_missing_keys=False,
    ):
        """Randomly crops with a specified probability of centering the crop on a positive voxel.

        Args:
            keys: Keys of the corresponding items to be transformed.
            roi_size: Desired output size of the crop (e.g., Depth, Height, Width for 3D).
                      Can handle 2D, 3D, or higher-dimensional data.
            positive_key: Key for the segmentation mask used to determine positive voxels.
            positive_probability: Probability that the crop will center on a positive voxel.
            allow_missing_keys: Don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.roi_size = np.array(roi_size if isinstance(roi_size, list | tuple) else [roi_size])
        self.positive_key = positive_key
        self.positive_probability = positive_probability

    def sample_center_positive(self, mask):
        """Sample a crop center from positive voxels."""
        img_shape = np.array(mask.shape[1:])
        positive_mask = mask.squeeze() > 0  # Shape: (600, 600, 280)

        # Create boundary masks for each axis
        z_valid = np.arange(mask.shape[1]) >= self.roi_size[0] // 2
        z_valid &= np.arange(mask.shape[1]) < img_shape[0] - self.roi_size[0] // 2

        y_valid = np.arange(mask.shape[2]) >= self.roi_size[1] // 2
        y_valid &= np.arange(mask.shape[2]) < img_shape[1] - self.roi_size[1] // 2

        x_valid = np.arange(mask.shape[3]) >= self.roi_size[2] // 2
        x_valid &= np.arange(mask.shape[3]) < img_shape[2] - self.roi_size[2] // 2

        # Combine the positive mask with valid bounds for each axis
        valid_mask = (
            positive_mask & z_valid[:, None, None] & y_valid[None, :, None] & x_valid[None, None, :]
        )

        # Get positive voxel coordinates
        positive_voxels = np.argwhere(valid_mask)

        if len(positive_voxels) == 0:
            # No positive voxels found, sample randomly
            return self.sample_center_random(img_shape)

        return random.choice(positive_voxels)

    def sample_center_random(self, img_shape):
        """Sample a random crop center."""
        crop_center = [
            self.R.randint(self.roi_size[i] // 2, img_shape[i] - self.roi_size[i] // 2)
            for i in range(len(img_shape))
        ]
        return np.array(crop_center)

    def __call__(self, data):
        """Apply the crop transform to the input data."""
        d = dict(data)

        if not self.allow_missing_keys and not all(k in d for k in self.keys):
            msg = f"Keys {self.keys} not found in data dictionary"
            raise KeyError(msg)

        mask = d[self.positive_key]
        img_shape = np.array(mask.shape[1:])

        crop_center = (
            self.sample_center_positive(mask)
            if self.R.random() < self.positive_probability
            else self.sample_center_random(img_shape)
        )

        for key in self.keys:
            img = d[key]

            # Calculate start and end indices for cropping
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
        # 1) Resample the image and mask to have the same voxel spacing
        Spacingd(
            keys=["image", "mask"],
            pixdim=target_pixel_dim,
            mode=("bilinear", "nearest"),
        ),
        # 2) Scale the intensity of the image to [0, 1]
        # this really depends on the input range. it could happen that the range is not meaningful
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        # 3) Resize the image and mask to a target spatial size without distorting the aspect ratio
        ResizeWithPadOrCropd(
            keys=["image", "mask"],
            spatial_size=target_spatial_size,
        ),
        ToTensord(keys=["image", "mask"]),
    ]


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


def get_patch_center_gaussian_transform(
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


def get_patch_positive_center_transform(
    size: Sequence[int] = (96, 96, 96),
    target_pixel_dim: tuple[float, float, float] = TARGET_PIXEL_DIM,
    target_spatial_size: tuple[int, int, int] = TARGET_SPATIAL_SIZE,
    pos_center_prob: float = POS_CENTER_PROB,
):
    # Make a copy of the pipeline transforms
    transforms = get_default_transforms(target_pixel_dim, target_spatial_size)
    return Compose(
        transforms[:-1]
        + [
            PositiveBiasedRandomCrop(
                keys=["image", "mask"],
                positive_key="mask",
                roi_size=size,
                positive_probability=pos_center_prob,
            )
        ]
        + transforms[-1:]
    )

def get_patch_uniform_transform(
    size: Sequence[int] = (96, 96, 96),
    target_pixel_dim: tuple[float, float, float] = TARGET_PIXEL_DIM,
    target_spatial_size: tuple[int, int, int] = TARGET_SPATIAL_SIZE,
):
    # Make a copy of the pipeline transforms
    transforms = get_default_transforms(target_pixel_dim, target_spatial_size)
    return Compose(
        transforms[:-1]
        + [
            RandSpatialCropd(
                keys=["image", "mask"],
                roi_size=size,
                random_size=False,
                random_center=True,
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
    pos_center_prob: float = POS_CENTER_PROB,
) -> Callable:
    """Get the transformation function based on the type."""
    size = [size] * 3 if isinstance(size, int) else size
    match type_:
        case TransformType.RESIZE:
            return get_resize_transform(size, target_pixel_dim, target_spatial_size)
        case TransformType.PATCH_CENTER_GAUSSIAN:
            return get_patch_center_gaussian_transform(
                size,
                target_pixel_dim,
                target_spatial_size,
                sigma_ratio,
            )
        case TransformType.PATCH_POS_CENTER:
            return get_patch_positive_center_transform(
                size,
                target_pixel_dim,
                target_spatial_size,
                pos_center_prob,
            )
        case TransformType.PATCH_UNIFORM:
            return get_patch_uniform_transform(
                size, 
                target_pixel_dim, 
                target_spatial_size
            )
        case TransformType.STD:
            return get_std_transform(target_pixel_dim, target_spatial_size)
        case _:
            msg = f"Invalid transform type: {type_}"
            raise ValueError(msg)

def perform_mask_transformation(mask, mask_operation: MaskOperations):
    if mask_operation == MaskOperations.BINARY_CLASS:
        mask[mask!=1] = 0
        return mask
    else:
        return mask