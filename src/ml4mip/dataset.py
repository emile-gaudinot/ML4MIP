import random
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
from hydra.core.config_store import ConfigStore
from monai.data import GridPatchDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandSpatialCropd,
    Resized,
    ScaleIntensityd,
    ToTensord,
)
from omegaconf import MISSING
from torch.utils.data import Dataset


class TransformType(Enum):
    RESIZE = "resize"
    PATCH = "patch"


@dataclass
class DatasetConfig:
    data_dir: str = MISSING
    image_suffix: str = ".img.nii.gz"
    mask_suffix: str = ".label.nii.gz"
    transform: TransformType = MISSING
    size: int = 96
    train: bool = True
    split_ratio: float = 0.9


_cs = ConfigStore.instance()
_cs.store(
    group="dataset",
    name="base_dataset",
    node=DatasetConfig,
)


def get_dataset(cfg: DatasetConfig):
    return NiftiDataset(
        cfg.data_dir,
        image_suffix=cfg.image_suffix,
        mask_suffix=cfg.mask_suffix,
        transform=get_transform(
            cfg.transform,
            cfg.size,
        ),
        train=cfg.train,
        split_ratio=cfg.split_ratio,
    )


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
    ) -> None:
        self.data_dir: Path = Path(data_dir)
        self.image_suffix: str = image_suffix
        self.mask_suffix: str = mask_suffix
        self.transform: Callable | None = transform
        self.loader = LoadImaged(keys=["image", "mask"])

        # Collect image and mask file paths
        image_files: list[Path] = sorted(self.data_dir.glob(f"*{self.image_suffix}"))
        mask_files: list[Path] = sorted(self.data_dir.glob(f"*{self.mask_suffix}"))

        # Split the dataset into training and validation sets
        data_files = list(zip(image_files, mask_files, strict=True))
        data_files = self.get_sample(data_files, train=train, split_ratio=split_ratio)
        self.image_files, self.mask_files = zip(*data_files, strict=True)

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


class NiftiGridPatchDataset(NiftiDataset):
    """Subclass of NiftiDataset to handle grid-based patch extraction for large volumes.

    Loads the entire dataset into memory, should be mainly used for validation. For training
    use the the regular NiftiDataset with random patch cropping.

    Parameters:
        data_dir: Path to the directory containing image and mask files.
        image_suffix: Suffix or pattern to identify image files (default: '.img.nii.gz').
        mask_suffix: Suffix or pattern to identify mask files (default: '.label.nii.gz').
        transform: A function/transform to apply to both images and masks.
        patch_size: Size of the patches to extract (e.g., (96, 96, 96)).
        patch_overlap: Overlap between patches (e.g., (16, 16, 16)).
        train: Whether to split data for training or validation.
        split_ratio: Ratio for splitting training and validation data.
    """

    def __init__(
        self,
        data_dir: str | Path,
        image_suffix: str = ".img.nii.gz",
        mask_suffix: str = ".label.nii.gz",
        transform: Callable | None = None,
        patch_size: tuple[int, int, int] = (96, 96, 96),
        patch_overlap: tuple[int, int, int] = (0, 0, 0),
        train: bool = False,
        split_ratio: float = 0.9,
    ) -> None:
        super().__init__(data_dir, image_suffix, mask_suffix, transform, train, split_ratio)
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.grid_datasets = self._create_grid_datasets()

    def _create_grid_datasets(self):
        """Create GridPatchDatasets for all volumes."""
        grid_datasets = []
        for img_path, mask_path in zip(self.image_files, self.mask_files, strict=False):
            data_dict = {"image": img_path, "mask": mask_path}
            loaded_data = self.loader(data_dict)
            grid_dataset = GridPatchDataset(
                data=loaded_data,
                patch_size=self.patch_size,
                overlap=self.patch_overlap,
            )
            grid_datasets.append(grid_dataset)
        return grid_datasets

    def __len__(self):
        """Return the total number of patches across all volumes."""
        return sum(len(grid) for grid in self.grid_datasets)

    def __getitem__(self, idx: int):
        """Fetch a patch by index."""
        for grid_dataset in self.grid_datasets:
            if idx < len(grid_dataset):
                patch = grid_dataset[idx]
                if self.transform:
                    patch = self.transform(patch)
                return patch["image"], patch["mask"]
            idx -= len(grid_dataset)

        msg = "Index out of range."
        raise IndexError(msg)


def get_transform_resize(size: int):
    return Compose(
        [
            EnsureChannelFirstd(keys=["image", "mask"]),
            # Spacingd(keys=["image", "mask"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            ScaleIntensityd(keys=["image"]),
            Resized(
                keys=["image", "mask"],
                spatial_size=(size, size, size),
                mode=("trilinear", "nearest"),  # 'trilinear' for image, 'nearest' for mask
                align_corners=(True, None),  # 'align_corners' is only relevant for 'trilinear' mode
            ),
            ToTensord(keys=["image", "mask"]),
        ]
    )


def get_transform_patch(size: int):
    return Compose(
        [
            EnsureChannelFirstd(keys=["image", "mask"]),
            ScaleIntensityd(keys=["image"]),
            RandSpatialCropd(
                keys=["image", "mask"],
                roi_size=(size, size, size),  # Size of the patches
                random_size=False,  # Ensure fixed-size patches
            ),
            ToTensord(keys=["image", "mask"]),
        ]
    )


# TODO: add transform config
def get_transform(type_: TransformType, size: int) -> Callable:
    """Get the transformation function based on the type."""
    match type_:
        case TransformType.RESIZE:
            return get_transform_resize(size)
        case TransformType.PATCH:
            return get_transform_patch(size)
        case _:
            msg = f"Invalid transform type: {type_}"
            raise ValueError(msg)
