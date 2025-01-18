import logging
import sys
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import hydra
import mlflow
import mlflow.pytorch
import torch
from hydra.core.config_store import ConfigStore
from monai.transforms import SaveImage
from omegaconf import MISSING, OmegaConf
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml4mip import trainer
from ml4mip.dataset import (
    DataLoaderConfig,
    ImageMaskDataset,
    UnlabeledDataset,
    get_dataset,
    get_scaling_transform,
    reshape_to_original,
)
from ml4mip.graph_extraction import ExtractionConfig, extract_graph
from ml4mip.loss import LossConfig, get_loss
from ml4mip.models import ModelConfig, get_model
from ml4mip.scheduler import SchedulerConfig, get_scheduler
from ml4mip.utils.logging import log_hydra_config_to_mlflow, log_metrics
from ml4mip.utils.metrics import MetricType, get_metrics
from ml4mip.utils.torch import load_checkpoint, save_model
from ml4mip.visualize import visualize_model

logger = logging.getLogger(__name__)


@dataclass
class Config:
    ml_flow_uri: str = str(Path.cwd() / "runs")
    model_dir: str = MISSING
    model_tag: str = MISSING
    batch_size: int = 1
    lr: float = 1e-4
    num_epochs: int = 10
    model: ModelConfig = MISSING
    dataset: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    visualize_model: bool = False
    visualize_model_val_batches: int = 1
    visualize_model_train_batches: int = 4
    plot_3d: bool = False
    extract_graph: bool = False
    epoch_profiling_torch: bool = False
    epoch_profiling_cpy: bool = False
    inference: trainer.InferenceConfig = field(default_factory=trainer.InferenceConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


_cs = ConfigStore.instance()
_cs.store(
    name="base_config",
    node=Config,
)


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config",
)
def run_training(cfg: Config) -> None:
    """Prepare data, model, and training loop for fine-tuning."""
    logger.info(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(
        cfg
    )  # this is important, so the values are treated as in the workflow.Config object

    logger.info("Starting model training script")

    # TODO: checks don't make sense with preprocessed datasets, update:
    # if cfg.inference.mode == trainer.InferenceMode.SLIDING_WINDOW and cfg.dataset.train.transform not in (
    #     TransformType.PATCH_POS_CENTER,
    #     TransformType.PATCH_CENTER_GAUSSIAN,
    # ):
    #     msg = (
    #         "Sliding window validation is only supported for patch-based datasets. "
    #         "Please set the dataset.transform to 'PATCH' in the configuration file."
    #     )
    #     raise ValueError(msg)

    # if (
    #     cfg.inference.mode == trainer.InferenceMode.RESCALE
    #     and cfg.dataset.transform != TransformType.RESIZE
    # ):
    #     msg = (
    #         "Rescale validation is only supported for resize-based datasets. "
    #         "Please set the dataset.transform to 'RESIZE' in the configuration file."
    #     )
    #     raise ValueError(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds = get_dataset(cfg.dataset)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=torch.cuda.is_available()
    )

    msg = f"Training on {len(train_ds)} samples"
    logger.info(msg)

    # Model and optimizer
    model = get_model(cfg.model)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    if cfg.scheduler.linear_total_iters is None:
        cfg.scheduler.linear_total_iters = cfg.num_epochs
    scheduler = get_scheduler(cfg.scheduler, optimizer)

    checkpoint_dir = (Path(cfg.model_dir) / f"{cfg.model_tag}").with_suffix("")
    current_epoch = 0
    if checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir()):
        prev_epochs = load_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_dir=checkpoint_dir,
            scheduler=(scheduler if cfg.scheduler.resume_schedule else None),
        )
        current_epoch = prev_epochs + 1
        logger.info(
            "Resuming training from epoch %d/%d with %d epochs remaining",
            current_epoch,
            cfg.num_epochs,
            cfg.num_epochs - current_epoch,
        )
    elif checkpoint_dir.exists() and not checkpoint_dir.is_dir():
        msg = (
            f"Checkpoint directory {checkpoint_dir} already exists and is not a directory."
            "The checkpoint dir is model_dir/model_tag."
            "Please remove the file or specify a different model_tag."
        )
        logger.error(msg)
        sys.exit(1)
    else:
        msg = f"Starting training from scratch with {current_epoch} epochs (no checkpoint found)"
        logger.info(msg)

    loss_fn = get_loss(cfg.loss)
    metrics = get_metrics(metric_types=[MetricType.DICE])
    metrics_val = get_metrics()

    # Initialize MLflow
    mlflow.set_tracking_uri(cfg.ml_flow_uri)  # Update path as needed
    mlflow.set_experiment("model_training")

    try:
        with mlflow.start_run(run_name="training_run"):
            # Log configuration parameters
            log_hydra_config_to_mlflow(cfg)
            # Train the model and log metrics
            trainer.train(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                metrics=metrics,
                metrics_val=metrics_val,
                device=device,
                current_epoch=current_epoch,
                num_epochs=cfg.num_epochs,
                val_loader=val_loader,
                inference_cfg=cfg.inference,
                checkpoint_dir=checkpoint_dir,
                scheduler=scheduler,
                torch_profiling=cfg.epoch_profiling_torch,
                cpython_profiling=cfg.epoch_profiling_cpy,
            )

            # Save and log the final model
            save_model(
                model,
                checkpoint_dir / "final_model",
            )
            if cfg.visualize_model:
                visualize_model(
                    val_loader,
                    model,
                    device,
                    val_batches=cfg.visualize_model_val_batches,
                    sigmoid=True,
                    plot_3d=cfg.plot_3d,
                    extract_graph=(extract_graph if cfg.extract_graph else None),
                    train_data_loader=train_loader,
                    train_batches=cfg.visualize_model_train_batches,
                    inference_cfg=cfg.inference,
                )
    except KeyboardInterrupt:
        # Handle manual stopping
        logger.info("Manual stopping detected. Saving model state...")
        # Save and log the final model
        save_model(
            model,
            checkpoint_dir / "manual_stop_model",
        )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_validation(cfg: Config):
    logger.info(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(
        cfg
    )  # this is important, so the values are treated as in the workflow.Config object
    logger.info("Starting validation script")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_ds = get_dataset(cfg.dataset)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=torch.cuda.is_available()
    )

    msg = f"Validation on {len(val_ds)} samples"
    logger.info(msg)

    # Model
    model = get_model(cfg.model)
    model = model.to(device)

    loss_fn = get_loss(cfg.loss)
    metrics = get_metrics()

    # Initialize MLflow
    mlflow.set_tracking_uri(cfg.ml_flow_uri)  # Update path as needed
    mlflow.set_experiment("model_evaluation")

    with mlflow.start_run(run_name="evaluation_run"):
        # Log configuration details as parameters
        log_hydra_config_to_mlflow(cfg)

        val_result = trainer.validate(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            metrics=metrics,
            device=device,
            inference_cfg=cfg.inference,
        )
        log_metrics(
            "val",
            val_result,
            step=0,
            logger=logger,
        )
        if cfg.visualize_model:
            visualize_model(
                val_loader,
                model,
                device,
                val_batches=cfg.visualize_model_val_batches,
                sigmoid=True,
                plot_3d=cfg.plot_3d,
                extract_graph=(extract_graph if cfg.extract_graph else None),
                inference_cfg=cfg.inference,
            )


@dataclass
class RunInferenceConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: trainer.InferenceConfig = field(default_factory=trainer.InferenceConfig)
    batch_size: int = 1
    input_dir: str = MISSING
    output_dir: str = MISSING
    num_workers: int = 4

_cs.store(
    name="base_inference_config",
    node=RunInferenceConfig,
)

@hydra.main(version_base=None, config_name="base_inference_config")
def run_inference(cfg: RunInferenceConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("Starting inference script")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = get_model(cfg.model)
    model = model.to(device)

    transform = get_scaling_transform()

    ds = ImageMaskDataset(
        data_dir=cfg.input_dir,
        transform=transform,
    )
    dataloader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    save_output = SaveImage(
        output_dir=cfg.output_dir,
        output_ext="label.nii.gz",
        separate_folder=False,
        print_log=True,
    )

    for images, _ in tqdm(dataloader):
        images = images.to(device)
        with torch.no_grad():
            output = trainer.inference(
                images=images,
                model=model,
                inference_cfg=cfg.inference,
            )

            # output is of shape (bs, c, h, w, d)
            # iterate over batch size
            for j in range(output.shape[0]):
                pred = output[j]
                binary_mask = pred >= 0.5
                reshaped_mask = reshape_to_original(binary_mask)
                save_output(reshaped_mask.squeeze(0).cpu(), meta_data=pred.meta)

    for file in Path(cfg.output_dir).glob("*_trans*"):
        # Remove 'label_trans.' from the filename
        new_name = file.name.replace("img_trans.", "").replace("label_trans.", "")
        new_path = file.with_name(new_name)
        file.rename(new_path)
        logger.info("Renamed %s to %s", file.name, new_name)

@dataclass
class RunGraphExtractionConfig:
    input_dir: str = MISSING
    output_dir: str = MISSING
    num_workers: int = 4
    extraction_cfg: ExtractionConfig = field(default_factory=ExtractionConfig)
    max_samples: int | None = None


_cs.store(
    name="base_extraction_config",
    node=RunGraphExtractionConfig,
)

@hydra.main(version_base=None, config_name="base_extraction_config")
def run_graph_extraction(cfg: RunGraphExtractionConfig):
    ds = UnlabeledDataset(
        input_dir=cfg.input_dir,
    )

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    def handle_idx(idx):
        nifti_obj = ds[idx]
        binary_volume = nifti_obj.get_fdata()
        file_id = nifti_obj.get_filename().split(".")[0]
        path = Path(cfg.output_dir) / f"{file_id}.graph.json"
        extract_graph(binary_volume, cfg.extraction_cfg, path=path)

    indices = range(len(ds)) if cfg.max_samples is None else range(cfg.max_samples)
    with Pool(processes=cfg.num_workers) as pool:
        pool.map(handle_idx, indices)
