from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Resized,
    ScaleIntensityd,
    ToTensord,
)
from torch.utils.data import Dataset


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
    ) -> None:
        self.data_dir: Path = Path(data_dir)
        self.image_suffix: str = image_suffix
        self.mask_suffix: str = mask_suffix
        self.transform: Callable | None = transform
        self.loader = LoadImaged(keys=["image", "mask"])

        # Collect image and mask file paths
        self.image_files: list[Path] = sorted(self.data_dir.glob(f"*{self.image_suffix}"))
        self.mask_files: list[Path] = sorted(self.data_dir.glob(f"*{self.mask_suffix}"))

        # Ensure image and mask files match
        assert len(self.image_files) == len(
            self.mask_files
        ), "Number of image files and mask files must match."
        for img, mask in zip(self.image_files, self.mask_files, strict=False):
            # print(img.name.replace(image_suffix, ""))
            # print(mask.name.replace(mask_suffix, ""))
            assert img.name.replace(image_suffix, "") == mask.name.replace(
                mask_suffix, ""
            ), f"Image file {img} and mask file {mask} do not match."

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


transforms = Compose(
    [
        EnsureChannelFirstd(keys=["image", "mask"]),
        # Spacingd(keys=["image", "mask"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityd(keys=["image"]),
        Resized(
            keys=["image", "mask"],
            spatial_size=(96, 96, 96),
            mode=("trilinear", "nearest"),  # 'trilinear' for image, 'nearest' for mask
            align_corners=(True, None),  # 'align_corners' is only relevant for 'trilinear' mode
        ),
        ToTensord(keys=["image", "mask"]),
    ]
)


def transform_resize(
    data: dict[str, np.ndarray], target_shape=(96, 96, 96)
) -> dict[str, np.ndarray]:
    """Apply transformations to the image and mask."""
    # Resize to a target shape

    if "image" not in data or "mask" not in data:
        msg = "Data must contain both 'image' and 'mask' keys."
        raise ValueError(msg)

    image = resize(data["image"], target_shape=target_shape)
    mask = resize(data["mask"], target_shape=target_shape, binary=True)

    return {"image": image, "mask": mask}


def resize(
    data: np.ndarray,
    target_shape: tuple = (96, 96, 96),
    binary: bool = False,
    threshold: float = 0.5,
) -> np.ndarray:
    """Resize a 3D NumPy array to a target shape using trilinear interpolation.

    Parameters:
        data: 3D NumPy array to be resized.
        target_shape: Target shape for the resized array.

    Returns:
        Resized 3D NumPy array.
    """
    # Convert to PyTorch tensor and add batch and channel dimensions
    data_tensor = torch.tensor(data).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, D, H, W]

    # Resize to target shape
    resized_tensor = F.interpolate(
        data_tensor, size=target_shape, mode="trilinear", align_corners=False
    )

    if binary:
        resized_tensor = resized_tensor >= threshold  # Apply thresholding

    # Convert back to NumPy array
    return resized_tensor.squeeze(0).squeeze(0).numpy()  # Shape: [D, H, W]
