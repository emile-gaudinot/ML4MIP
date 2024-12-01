import logging

import hydra
from omegaconf import OmegaConf

from ml4mip.workflows import (
    Config,
    Mode,
    run_evaluation,
    run_training,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config):
    logger.info(OmegaConf.to_yaml(cfg))
    match cfg.mode:
        case Mode.TRAIN:
            run_training(cfg)
        case Mode.EVAL:
            run_evaluation(cfg)
        case _:
            msg = f"Unsupported mode {cfg.mode}"
            raise ValueError(msg)


if __name__ == "__main__":
    main()
