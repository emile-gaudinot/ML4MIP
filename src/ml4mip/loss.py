from dataclasses import dataclass
from enum import Enum

from monai.losses import BCELoss, CrossEntropyLoss, DiceCELoss, DiceLoss, FocalLoss, TverskyLoss


class LossType(Enum):
    DICE = "dice"
    BCE = "bce"
    BCE_DICE = "bce_dice"
    CE = "ce"
    FOCAL = "focal"
    TVERSKY = "tversky"


@dataclass
class LossConfig:
    loss_type: LossType = LossType.BCE_DICE
    lambda_dice: float = 0.7
    lambda_ce: float = 0.3
    sigmoid = True


def get_loss(cfg: LossConfig):
    match cfg.loss_type:
        case LossType.DICE:
            return DiceLoss(include_background=True, sigmoid=cfg.sigmoid)
        case LossType.BCE:
            return BCELoss(sigma=cfg.sigmoid)
        case LossType.BCE_DICE:
            return DiceCELoss(
                include_background=True,
                to_onehot_y=False,
                softmax=False,
                sigmoid=cfg.sigmoid,
                lambda_dice=cfg.lambda_dice,
                lambda_ce=cfg.lambda_ce,
            )
        case LossType.CE:
            return CrossEntropyLoss(sigmoid=cfg.sigmoid)
        case LossType.FOCAL:
            return FocalLoss(sigmoid=cfg.sigmoid)
        case LossType.TVERSKY:
            return TverskyLoss(sigmoid=cfg.sigmoid)
        case _:
            msg = f"Loss type {cfg.loss_type} not supported"
            raise ValueError(msg)
