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
        full_output = self.model(x)  # Forward pass through the original model
        return full_output[:, self.selected_channels, ...]


def get_model(cfg: ModelConfig) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    match cfg.model_type:
        case ModelType.UNETR_PTR:
            model = torch.jit.load(cfg.base_model_jit_path, map_location=device)
            model = UnetrPtrJitWrapper(model)
        case _:
            msg = f"Model type {cfg.model_type} not implemented."
            raise NotImplementedError(msg)

    if cfg.model_path:
        state_dict = torch.load(cfg.model_path)
        model.load_state_dict(state_dict)

    return model
