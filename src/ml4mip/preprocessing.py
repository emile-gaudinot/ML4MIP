import logging
from dataclasses import dataclass, field
from pathlib import Path

import hydra
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
):
    # 1) get name
    image_filename = image.meta.get("filename_or_obj", None)
    if image_filename is None:
        raise ValueError("Image does not have a filename")
    # 2) create directory with output_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    # 3) repeat n_patches times:
    for i in range(n_patches):
        saver = SaveImaged(
            keys=["image", "mask"],  # Keys of the items to be saved
            output_dir=output_dir,  # Directory where the files will be saved
            output_postfix=f"patch[{i}]",  # Postfix for the saved file names
            output_ext=".nii.gz",  # File extension
            resample=False,  # Whether to resample the image
            print_log=True,  # Whether to print log messages
        )
        # 3.1) apply transform and save
        patch = transform({"image": image, "mask": mask})
        saver(patch)


@dataclass
class PreprocessingConfig:
    n_patches: int = 10
    # current working directory
    output_dir: str = Path.cwd() / "preprocessed_data"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


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
        image_prefix=cfg.dataset.image_prefix,
        mask_prefix=cfg.dataset.mask_prefix,
        image_suffix=cfg.dataset.image_suffix,
        mask_suffix=cfg.dataset.mask_suffix,
        transform=base_transforms,
        train=True,
        split_ratio=cfg.dataset.split_ratio,
        max_samples=cfg.dataset.max_train_samples,
        cache=cfg.dataset.cache_train_dataset,
        cache_pooling=cfg.dataset.cache_pooling,
    )

    # TODO: multithreading
    for i in range(len(base_dataset)):
        image, mask = base_dataset[i]
        create_patches(image, mask, post_transforms, cfg.n_patches, cfg.output_dir)
