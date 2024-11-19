from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
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

        # Collect image and mask file paths
        self.image_files: list[Path] = sorted(self.data_dir.glob(f"*{self.image_suffix}"))
        self.mask_files: list[Path] = sorted(self.data_dir.glob(f"*{self.mask_suffix}"))

        # Ensure image and mask files match
        assert len(self.image_files) == len(
            self.mask_files
        ), "Number of image files and mask files must match."
        for img, mask in zip(self.image_files, self.mask_files, strict=False):
            assert (
                img.name.split(".")[0] == mask.name.split(".")[0]
            ), f"Image file {img} and mask file {mask} do not match."

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Load image and mask
        img: np.ndarray = nib.load(self.image_files[idx]).get_fdata()
        mask: np.ndarray = nib.load(self.mask_files[idx]).get_fdata()

        # Ensure correct shapes
        img = np.asarray(img, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)

        # Apply transforms if provided
        if self.transform:
            transformed: dict = self.transform(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]

        # Convert to PyTorch tensors
        img_tensor: torch.Tensor = torch.tensor(img, dtype=torch.float32)
        mask_tensor: torch.Tensor = torch.tensor(mask, dtype=torch.float32)

        return img_tensor, mask_tensor


def transform(image: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Apply transformations to the image and mask."""
    # Resize to a target shape
    image = resize(image, target_shape=(96, 96, 96))
    mask = resize(mask, target_shape=(96, 96, 96), binary=True)

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
