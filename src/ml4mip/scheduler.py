from dataclasses import dataclass
from enum import Enum

from torch.optim import lr_scheduler

class SchedulerType(Enum):
    LINEARLR = "linearlr"


@dataclass
class SchedulerConfig:	
    scheheduler_type: SchedulerType = SchedulerType.LINEARLR
    linear_start_factor: float = 1
    linear_end_factor: float = 0.01
    linear_total_iters: int | None = None
    resume_schedule: bool = True


def get_scheduler(cfg: SchedulerConfig, optimizer):
    match cfg.scheheduler_type:
        case SchedulerType.LINEARLR:
            return lr_scheduler.LinearLR(
                        optimizer, start_factor=cfg.linear_start_factor, end_factor=cfg.linear_end_factor, total_iters=cfg.linear_total_iters
                    )
        case _:
            msg = f"Scheduler type {cfg.scheheduler_type} not supported"
            raise ValueError(msg)
