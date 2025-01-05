import logging
import pathlib
from dataclasses import dataclass
from enum import Enum

import torch
from hydra.core.config_store import ConfigStore
from monai.networks.nets import UNet
from omegaconf import MISSING

from ml4mip.segment_anything import sam_model_registry

logger = logging.getLogger(__name__)


class ModelType(Enum):
    UNETR_PTR = "unetr_ptr"
    UNETR = "unetr"
    UNET = "unet"
    UNETMONAI1 = "unet_monai_1"
    UNETMONAI2 = "unet_monai_2"
    UNETMONAI3 = "unet_monai_3"
    MEDSAM = "medsam"


@dataclass
class ModelConfig:
    model_type: ModelType = MISSING
    # this should be interpreted as either a path to a state dict
    # or to a check point directory
    model_path: str | None = None
    base_model_jit_path: str | None = None
    # TODO add more config values for other model classes:
    # maybe nested classes are better for model specific config values
    checkpoint_path: str | None = None


_cs = ConfigStore.instance()
_cs.store(
    group="model",
    name="base_model",
    node=ModelConfig,
)


class UnetrPtrJitWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, selected_channels: tuple[int, ...] = (0,)):
        super().__init__()
        self.model = model
        self.selected_channels = selected_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_output = self.model(
            x
        )  # Forward pass through the original model BS x 14 x 96 x 96 x 96
        return full_output[:, self.selected_channels, ...]  # BS x C x D x H x W


class UNetWrapper(torch.nn.Module):
    def __init__(self):
        super(UNetWrapper, self).__init__()

        # First block without checkpointing
        self.firstBlock = torch.nn.Sequential(
            torch.nn.Conv3d(1, 24, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(24),
            torch.nn.ReLU(),
        )

        # Sequentials for Checkpointin

        self.en1 = torch.nn.Sequential(
            torch.nn.Conv3d(24, 24, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(24),
            torch.nn.ReLU(),
        )

        self.en2 = torch.nn.Sequential(
            torch.nn.MaxPool3d(2),
            torch.nn.Conv3d(24, 48, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(48),
            torch.nn.ReLU(),
            torch.nn.Conv3d(48, 48, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(48),
            torch.nn.ReLU(),
        )

        self.en3 = torch.nn.Sequential(
            torch.nn.MaxPool3d(2),
            torch.nn.Conv3d(48, 96, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(96),
            torch.nn.ReLU(),
            torch.nn.Conv3d(96, 96, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(96),
            torch.nn.ReLU(),
        )

        self.valley = torch.nn.Sequential(
            torch.nn.MaxPool3d(2),
            torch.nn.Conv3d(96, 192, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(192),
            torch.nn.ReLU(),
            torch.nn.Conv3d(192, 192, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(192),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(192, 96, 2, 2),
        )

        self.dec1 = torch.nn.Sequential(
            torch.nn.Conv3d(192, 96, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(96),
            torch.nn.ReLU(),
            torch.nn.Conv3d(96, 96, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(96),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(96, 48, 2, 2),
        )

        self.dec2 = torch.nn.Sequential(
            torch.nn.Conv3d(96, 48, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(48),
            torch.nn.ReLU(),
            torch.nn.Conv3d(48, 48, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(48),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(48, 24, 2, 2),
        )

        self.dec3 = torch.nn.Sequential(
            torch.nn.Conv3d(48, 24, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(24),
            torch.nn.ReLU(),
            torch.nn.Conv3d(24, 24, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(24),
            torch.nn.ReLU(),
            torch.nn.Conv3d(24, 1, kernel_size=3, padding=1),
        )

        # Sigmoid for output
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        # Skip-Connections
        skip = []

        # First Convolution
        x = self.firstBlock(x)

        # Encoding
        x = self.en1(x)
        skip.append(torch.clone(x))
        x = self.en2(x)
        skip.append(torch.clone(x))
        x = self.en3(x)
        skip.append(torch.clone(x))

        # Bottleneck
        x = self.valley(x)

        # Decoding
        x = torch.cat((x, skip[-1]), 1)
        x = self.dec1(x)
        x = torch.cat((x, skip[-2]), 1)
        x = self.dec2(x)
        x = torch.cat((x, skip[-3]), 1)
        x = self.dec3(x)

        # x = self.sig(x)
        return x


class MedSamWrapper(torch.nn.Module):
    def __init__(self, checkpoint_path: str | pathlib.Path):
        super().__init__()
        self.model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        self.model.load_state_dict(torch.load(checkpoint_path, weights_only=False))

    # TODO:
    def forward(self, images: torch.Tensor, masks) -> torch.Tensor:
        # Reshaping images: treat each z-slice image independantly
        images = images.permute(0, 4, 1, 2, 3)
        sh = images.shape
        images = images.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4])
        # Same for masks
        masks = masks.permute(0, 4, 1, 2, 3)
        sh = masks.shape
        masks = masks.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4])
        # Create "image", "boxes", "point_coords", "mask_inputs" and
        # "original_size" attributes to 'x'
        images = [
            {
                "image": img,
                "boxes": torch.tensor([[[0, 0, 95, 95]]], device="cuda"),
                # "point_coords": None,
                "mask_inputs": mask[None],
                "original_size": (96, 96),
            }
            for img, mask in zip(images, masks, strict=False)
        ]
        # del masks ?
        outputs = []
        bs = 2
        for i in range(len(images) // bs):
            single_batch_output = self.model(images[i : i + bs])
            outputs += [single_output["masks"][0] for single_output in single_batch_output]
        outputs = torch.stack(outputs)
        # Add the batch_size dimension
        outputs_sh = outputs.shape
        outputs = outputs.reshape(
            outputs_sh[0] // 96, 96, outputs_sh[1], outputs_sh[2], outputs_sh[3]
        )
        masks_sh = masks.shape
        masks = masks.reshape(masks_sh[0] // 96, 96, masks_sh[1], masks_sh[2], masks_sh[3])
        # Permute the z index to the end
        outputs = outputs.permute(0, 2, 3, 4, 1).to(dtype=torch.float)
        masks = masks.permute(0, 2, 3, 4, 1)

        return outputs


def get_model(cfg: ModelConfig) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    match cfg.model_type:
        case ModelType.UNETR_PTR:
            model = torch.jit.load(cfg.base_model_jit_path, map_location=device)
            model = UnetrPtrJitWrapper(model)
        case ModelType.UNET:
            model = UNetWrapper()
        case ModelType.UNETMONAI1:
            model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
        case ModelType.UNETMONAI2:
            model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=(32, 64, 128, 256, 320, 320),
                strides=(2, 2, 2, 2, 2),
                num_res_units=2,
            )
        case ModelType.UNETMONAI3:
            model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=(64, 128, 256, 512, 512, 512),
                strides=(2, 2, 2, 2, 2),
                num_res_units=2,
            )
        case ModelType.MEDSAM:
            MedSAM_CKPT_PATH = cfg.checkpoint_path
            model = sam_model_registry["vit_b"](checkpoint=MedSAM_CKPT_PATH)
            model.load_state_dict(torch.load(MedSAM_CKPT_PATH, weights_only=False))
        case _:
            msg = f"Model type {cfg.model_type} not implemented."
            raise NotImplementedError(msg)

    if cfg.model_path:
        msg = f"Loading model from {cfg.model_path} ..."
        logger.info(msg)
        state_dict = torch.load(cfg.model_path)
        model.load_state_dict(state_dict)

    return model
