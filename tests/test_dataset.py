from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest
import torch

from ml4mip.dataset import (
    ABCNiftiDataset,
    GroupedNifitDataset,  # Update this with your module name
    NiftiDataset,
)


def test_load_image_mask_files():
    with patch("ml4mip.dataset.Path.glob") as mock_glob:
        mock_glob.side_effect = [
            [Path("image1.nii"), Path("image2.nii")],
            [Path("mask1.nii"), Path("mask2.nii")],
        ]
        image_files, mask_files = NiftiDataset.load_image_mask_files(
            "*.nii", "*.nii", Path("/images"), Path("/masks")
        )
        assert len(image_files) == 2
        assert len(mask_files) == 2


def test_check_img_mask_files():
    image_files = [Path("/image1.img.nii.gz"), Path("/image2.img.nii.gz")]
    mask_files = [Path("/image1.label.nii.gz"), Path("/image2.label.nii.gz")]

    NiftiDataset.check_img_mask_files(
        image_files,
        mask_files,
        Path("/images"),
        Path("/masks"),
        ("", ".img.nii.gz"),
        ("", ".label.nii.gz"),
    )


def test_split_samples():
    image_files = [Path(f"/image{i}.nii") for i in range(10)]
    mask_files = [Path(f"/mask{i}.nii") for i in range(10)]

    train_images, train_masks = NiftiDataset.get_sample(
        image_files, mask_files, train=True, split_ratio=0.8
    )
    val_images, val_masks = NiftiDataset.get_sample(
        image_files, mask_files, train=False, split_ratio=0.8
    )

    assert len(train_images) == 8
    assert len(val_images) == 2


def test_process_samples():
    with (
        patch("ml4mip.dataset.ABCNiftiDataset.load_image_mask_files") as mock_load_files,
        patch("ml4mip.dataset.LoadImaged.__call__") as mock_loader,
    ):
        n_samples = 10
        mock_load_files.return_value = (
            [Path(f"/image{i}.img.nii.gz") for i in range(1, n_samples + 1)],  # 10 examples
            [Path(f"/image{i}.label.nii.gz") for i in range(1, n_samples + 1)],
        )

        mock_loader.side_effect = [
            {"image": torch.tensor([i]), "mask": torch.tensor([i])} for i in range(n_samples)
        ]

        dataset = NiftiDataset("", train=True, split_ratio=0.8)
        assert len(dataset) == 8, "Expected 8 samples in training set"
        images, masks = dataset.process_samples([0, 4, 6])
        assert len(images) == 3, "Expected 3 image samples in result"
        assert len(masks) == 3, "Expected 3 mask samples in result"
        assert images[0].item() == 0
        assert masks[-1].item() == 2  # This is because of the mock_loader.side_effect

        image, mask = dataset[0]
        assert image.item() == 3
        assert mask.item() == 3


def test_init():
    with (
        patch("ml4mip.dataset.NiftiDataset.load_image_mask_files") as mock_load_files,
        patch("ml4mip.dataset.LoadImaged.__call__") as mock_loader,
        patch("ml4mip.dataset.random.sample") as mock_sample,
    ):
        n_samples = 10
        mock_load_files.return_value = (
            [Path(f"/image{i}.img.nii.gz") for i in range(1, n_samples + 1)],  # 10 examples
            [Path(f"/image{i}.label.nii.gz") for i in range(1, n_samples + 1)],
        )
        mock_sample.return_value = [0, 1, 2]
        mock_loader.side_effect = [
            {"image": torch.tensor([i]), "mask": torch.tensor([i])} for i in range(n_samples)
        ]

        dataset = NiftiDataset("", cache=False)
        assert len(dataset.image_files) == 3
        assert len(dataset.mask_files) == 3
        assert not dataset.use_cache

        dataset = NiftiDataset("", cache=True)
        assert len(dataset.image_files) == 3
        assert len(dataset.mask_files) == 3
        assert dataset.use_cache

        image, mask = dataset[0]
        assert image.item() == 0
        assert mask.item() == 0

        # This would fail, because new Processes are spawned that do not have the mock_loader
        # dataset = NiftiDataset("", cache=True, cache_pooling=2)
        # assert len(dataset.image_files) == 3
        # assert len(dataset.mask_files) == 3
        # assert dataset.use_cache


def test_override_methods():
    class InvalidDataset(ABCNiftiDataset):
        pass

    with pytest.raises(TypeError):
        _ = InvalidDataset("/data")


@pytest.fixture
def mock_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    mask_dir = tmp_path / "mask"
    data_dir.mkdir()
    mask_dir.mkdir()

    # Create mock NIfTI image and mask files with small dummy data
    for e_idx in range(5):  # Epoch index
        for i in range(10):  # File index
            image_data = np.random.rand(10, 10, 10).astype(np.float32)
            mask_data = np.random.randint(0, 2, size=(10, 10, 10), dtype=np.uint8)

            image_nii = nib.Nifti1Image(image_data, affine=np.eye(4))
            mask_nii = nib.Nifti1Image(mask_data, affine=np.eye(4))

            nib.save(image_nii, data_dir / f"image_{i}_patch[{e_idx}].img.nii.gz")
            nib.save(mask_nii, mask_dir / f"image_{i}_patch[{e_idx}].label.nii.gz")

    return data_dir, mask_dir


def test_initialization(
    mock_data_dir,
):
    data_dir, mask_dir = mock_data_dir

    dataset = GroupedNifitDataset(
        data_dir=data_dir,
        mask_dir=mask_dir,
        image_affix=("", ".img.nii.gz"),
        mask_affix=("", ".label.nii.gz"),
        cache=False,
        max_samples=5,
        max_epoch=2,
    )

    assert len(dataset.image_files) == 2
    assert len(dataset.image_files[0]) == 5
    assert len(dataset.mask_files[0]) == 5


def test_len(mock_data_dir):
    data_dir, mask_dir = mock_data_dir

    dataset = GroupedNifitDataset(
        data_dir=data_dir,
        mask_dir=mask_dir,
        max_samples=10,
        max_epoch=1,
    )

    assert len(dataset) == 10


def test_epoch_switching(mock_data_dir):
    data_dir, mask_dir = mock_data_dir

    dataset = GroupedNifitDataset(
        data_dir=data_dir,
        mask_dir=mask_dir,
        max_samples=10,
        max_epoch=3,
    )

    image_1, mask_1 = dataset[0]
    assert dataset.epoch_counter == 0
    dataset.next_epoch()
    assert dataset.epoch_counter == 1
    image_2, mask_2 = dataset[0]
    assert not torch.equal(image_1, image_2)
    assert not torch.equal(mask_1, mask_2)

    dataset.next_epoch()
    assert dataset.epoch_counter == 2
    dataset.next_epoch()
    assert dataset.epoch_counter == 3

    image_1_b, mask_1_b = dataset[0]
    assert torch.equal(image_1, image_1_b), "Images do not match"
    assert torch.equal(mask_1, mask_1_b), "Masks do not match"


def test_caching_behavior(mock_data_dir):
    data_dir, mask_dir = mock_data_dir

    dataset = GroupedNifitDataset(
        data_dir=data_dir,
        mask_dir=mask_dir,
        max_samples=10,
        max_epoch=1,
        cache=True,
        cache_pooling=2,
    )

    assert hasattr(dataset, "image_cache")
    assert hasattr(dataset, "mask_cache")
    assert len(dataset.image_cache) == 10
    assert len(dataset.mask_cache) == 10


def test_invalid_files():
    with pytest.raises(ValueError):
        GroupedNifitDataset.check_img_mask_files(
            [],
            [],
            data_dir=Path("/invalid/data_dir"),
            mask_dir=Path("/invalid/mask_dir"),
            image_affix=("", ".img.nii.gz"),
            mask_affix=("", ".label.nii.gz"),
        )
