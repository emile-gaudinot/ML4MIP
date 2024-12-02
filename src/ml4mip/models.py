from dataclasses import dataclass
from enum import Enum

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class ModelType(Enum):
    UNETR_PTR = "unetr_ptr"
    UNETR = "unetr"
    UNET = "unet"
    MEDSAM = "medsam"


@dataclass
class ModelConfig:
    model_type: ModelType = MISSING
    model_path: str | None = None
    base_model_jit_path: str | None = None
    # TODO add more config values for other model classes:
    # maybe nested classes are better for model specific config values


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
    from torch.utils.checkpoint import checkpoint, checkpoint_sequential

    def __init__(self):
        super(UNetWrapper, self).__init__()
        
        #First block without checkpointing
        self.firstBlock = torch.nn.Sequential(*[
            torch.nn.Conv3d(1, 12, kernel_size=3, padding=1), torch.nn.ReLU()
        ])
        
        #Sequentials for Checkpointin
        
        self.en1 = torch.nn.Sequential(*[
            torch.nn.Conv3d(12, 12, kernel_size=3, padding=1), torch.nn.ReLU()
        ])
        
        self.en2 = torch.nn.Sequential(*[
            torch.nn.MaxPool3d(2),
            torch.nn.Conv3d(12, 24, kernel_size=3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv3d(24, 24, kernel_size=3, padding=1), torch.nn.ReLU()
        ])
        
        self.en3 = torch.nn.Sequential(*[
            torch.nn.MaxPool3d(2),
            torch.nn.Conv3d(24, 48, kernel_size=3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv3d(48, 48, kernel_size=3, padding=1), torch.nn.ReLU()
        ])
        
        self.valley = torch.nn.Sequential(*[
            torch.nn.MaxPool3d(2),
            torch.nn.Conv3d(48, 96, kernel_size=3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv3d(96, 96, kernel_size=3, padding=1), torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(96, 48, 2, 2)
        ])
        
        self.dec1 = torch.nn.Sequential(*[
            torch.nn.Conv3d(96, 48, kernel_size=3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv3d(48, 48, kernel_size=3, padding=1), torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(48, 24, 2, 2)
        ])
        
        self.dec2 = torch.nn.Sequential(*[
            torch.nn.Conv3d(48, 24, kernel_size=3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv3d(24, 24, kernel_size=3, padding=1), torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(24, 12, 2, 2)
        ])
        
        self.dec3 = torch.nn.Sequential(*[
            torch.nn.Conv3d(24, 12, kernel_size=3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv3d(12, 12, kernel_size=3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv3d(12, 1, kernel_size=3, padding=1), torch.nn.ReLU()
        ])
        
        #Sigmoid for output
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        # Skip-Connections
        skip = []

        # First Convolution
        x = self.firstBlock(x)

        # Encoder
        x = self.en1(x)
        skip.append(torch.clone(x))
        x = self.en2(x)
        skip.append(torch.clone(x))
        x = self.en3(x)
        skip.append(torch.clone(x))

        x = self.valley(x)

        # Decoder
        x = torch.cat((x, skip[-1]), 1)
        x = self.dec1(x)
        x = torch.cat((x, skip[-2]), 1)
        x = self.dec2(x)
        x = torch.cat((x, skip[-3]), 1)
        x = self.dec3(x)

        x = self.sig(x)
        return x


def get_model(cfg: ModelConfig) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    match cfg.model_type:
        case ModelType.UNETR_PTR:
            model = torch.jit.load(cfg.base_model_jit_path, map_location=device)
            model = UnetrPtrJitWrapper(model)
        case ModelType.UNET:
            model = UNetWrapper()
        case _:
            msg = f"Model type {cfg.model_type} not implemented."
            raise NotImplementedError(msg)

    if cfg.model_path:
        state_dict = torch.load(cfg.model_path)
        model.load_state_dict(state_dict)

    return model
