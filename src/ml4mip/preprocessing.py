import logging
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from monai.transforms import Compose, SaveImaged
from omegaconf import OmegaConf

from ml4mip.dataset import DatasetConfig, NiftiDataset, get_transform

logger = logging.getLogger(__name__)


def create_patches(
    image,
    mask,
    transform,
    n_patches,
    output_dir,
    image_affix,
    mask_affix,
):
    # 1) get name and remove affixes for clean output names
    if image.meta.get("filename_or_obj", None) is None:
        msg = "Image does not have a filename"
        raise ValueError(msg)
    image_prefix, image_suffix = image_affix
    image.meta["filename_or_obj"] = (
        Path(image.meta["filename_or_obj"]).name.replace(image_prefix, "").replace(image_suffix, "")
    )

    if mask.meta.get("filename_or_obj", None) is None:
        msg = "Mask does not have a filename"
        raise ValueError(msg)
    mask_prefix, mask_suffix = mask_affix
    mask.meta["filename_or_obj"] = (
        Path(mask.meta["filename_or_obj"]).name.replace(mask_prefix, "").replace(mask_suffix, "")
    )

    # 2) create directory with output_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 3) repeat n_patches times:
    for i in range(n_patches):
        image_saver = SaveImaged(
            keys=["image"],  # Keys of the items to be saved
            output_dir=output_dir,  # Directory where the files will be saved
            output_postfix=f"patch[{i}].img",  # Postfix for the saved file names
            output_ext=".nii.gz",  # File extension
            resample=False,  # Whether to resample the image
            print_log=True,  # Whether to print log messages
            separate_folder=False,
        )

        mask_saver = SaveImaged(
            keys=["mask"],  # Keys of the items to be saved
            output_dir=output_dir,  # Directory where the files will be saved
            output_postfix=f"patch[{i}].label",  # Postfix for the saved file names
            output_ext=".nii.gz",  # File extension
            resample=False,  # Whether to resample the image
            print_log=True,  # Whether to print log messages
            separate_folder=False,
        )
        # 3.1) apply transform and save
        patch = transform({"image": image, "mask": mask})
        image_saver(patch)
        mask_saver(patch)


def process_subset(
    index_subset: list[int],
    base_dataset,
    post_transforms,
    n_patches,
    output_dir,
    image_affix,
    mask_affix,
):
    logger.info("Processing subset: %s", index_subset)
    for i in range(len(index_subset)):
        local_img, local_msk = base_dataset[index_subset[i]]
        create_patches(
            local_img,
            local_msk,
            post_transforms,
            n_patches,
            output_dir,
            image_affix,
            mask_affix,
        )


@dataclass
class PreprocessingConfig:
    n_patches: int = 10
    # current working directory
    output_dir: str = Path.cwd() / "preprocessed_data"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    computation_pool_size: int = 1


_cs = ConfigStore.instance()
_cs.store(
    name="base_preprocessing_config",
    node=PreprocessingConfig,
)


@hydra.main(version_base=None, config_path="conf", config_name="preprocessing_config")
def main(cfg: PreprocessingConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(
        cfg
    )  # this is important, so the values are treated as in the workflow.Config object
    # transform should be standard transform
    transforms = get_transform(
        type_=cfg.dataset.transform,
        size=cfg.dataset.size,
        target_pixel_dim=cfg.dataset.target_pixel_dim,
        target_spatial_size=cfg.dataset.target_spatial_size,
        sigma_ratio=cfg.dataset.sigma_ratio,
        pos_center_prob=cfg.dataset.pos_center_prob,
    )

    base_transforms = Compose(transforms.transforms[:3])

    post_transforms = Compose(transforms.transforms[3:])

    base_dataset = NiftiDataset(
        data_dir=cfg.dataset.data_dir,
        mask_dir=cfg.dataset.mask_dir,
        image_affix=cfg.dataset.image_affix,
        mask_affix=cfg.dataset.mask_affix,
        transform=base_transforms,
        train=True,
        split_ratio=cfg.dataset.split_ratio,
        max_samples=cfg.dataset.max_samples,
        cache=cfg.dataset.cache,
        cache_pooling=cfg.dataset.cache_pooling,
    )

    # run process_subset in computation pool
    subsets = np.array_split(range(len(base_dataset)), cfg.computation_pool_size)
    with Pool(processes=cfg.computation_pool_size) as pool:
        pool.starmap(
            process_subset,
            [
                (
                    list(part),
                    base_dataset,
                    post_transforms,
                    cfg.n_patches,
                    cfg.output_dir,
                    cfg.dataset.image_affix,
                    cfg.dataset.mask_affix,
                )
                for part in subsets
            ],
        )
