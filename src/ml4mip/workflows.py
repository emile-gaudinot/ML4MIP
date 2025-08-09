import gc
import logging
import resource
import sys
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import hydra
import mlflow
import mlflow.pytorch
import nibabel as nib
import torch
from graph_extraction import connected_component_distance_filter
from hydra.core.config_store import ConfigStore
from monai.transforms import (
    Compose,
    ResizeWithPadOrCrop,
    SaveImage,
    ScaleIntensity,
    Spacing,
    ToTensor,
)
from omegaconf import MISSING, OmegaConf
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml4mip import trainer
from ml4mip.dataset import (
    TARGET_PIXEL_DIM,
    TARGET_SPATIAL_SIZE,
    DataLoaderConfig,
    ImageDataset,
    UnlabeledDataset,
    get_dataset,
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
    """
    Prepare data, model, and training loop for fine-tuning.
    Args:
        cfg: Config object containing all training parameters and settings.
    """
    logger.info(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(
        cfg
    )  # this is important, so the values are treated as in the workflow.Config object

    logger.info("Starting model training script")

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
    """
    Run model validation on the validation dataset and log results.
    Args:
        cfg: Config object containing all validation parameters and settings.
    """
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


def log_memory_usage():
    """
    Logs the current CPU and GPU memory usage.
    """
    # CPU memory usage
    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    memory_usage_mb = memory_usage / 1024
    logger.info(f"CPU memory usage: {memory_usage_mb:.2f} MB")

    # GPU memory usage (if CUDA is available)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        max_reserved = torch.cuda.max_memory_reserved() / 1024**2

        logger.info(
            f"GPU memory usage - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB"
        )
        logger.info(
            f"GPU memory peak - Max Allocated: {max_allocated:.2f} MB, Max Reserved: {max_reserved:.2f} MB"
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
    """
    Run inference on a dataset using a trained model and save the predictions.
    Args:
        cfg: RunInferenceConfig object containing inference parameters and settings.
    """
    logger.info(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(cfg)
    logger.info("Starting inference script")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = get_model(cfg.model)
    model = model.to(device)

    ds = ImageDataset(
        data_dir=cfg.input_dir,
        transform=Compose(
            [
                Spacing(
                    pixdim=TARGET_PIXEL_DIM,
                    mode="bilinear",
                ),
                ScaleIntensity(minv=0.0, maxv=1.0),
                ResizeWithPadOrCrop(
                    spatial_size=TARGET_SPATIAL_SIZE,
                    mode="edge",
                ),
                ToTensor(),
            ]
        ),
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

    for images in tqdm(dataloader):
        log_memory_usage()
        images = images.to(device)
        with torch.no_grad():
            output = trainer.inference(
                images=images,
                model=model,
                cfg=cfg.inference,
            )

            # output is of shape (bs, c, h, w, d)
            # iterate over batch size
            for j in range(output.shape[0]):
                pred = output[j]
                binary_mask = pred >= 0.5
                reshaped_mask = reshape_to_original(binary_mask)
                reshaped_mask = reshaped_mask >= 0.5
                save_output(reshaped_mask.squeeze(0).cpu().detach(), meta_data=pred.meta)

                del reshaped_mask, pred, binary_mask

        del output, images
        # run python garbage collection and empty gpu cache to prevent full memory training stops
        gc.collect()
        torch.cuda.empty_cache()

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
    batch_size: int = 5
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    max_samples: int | None = None


_cs.store(
    name="base_extraction_config",
    node=RunGraphExtractionConfig,
)


def handle_idx(idx, ds, cfg):
    """
    Extract a graph from a NIfTI object and save it as a JSON file.
    Args:
        idx: Index of the sample in the dataset.
        ds: Dataset object.
        cfg: RunGraphExtractionConfig object with extraction settings.
    """
    nifti_obj = ds[idx]
    file_id = Path(nifti_obj.get_filename()).stem.split(".")[0]
    path = Path(cfg.output_dir) / f"{file_id}.graph.json"
    extract_graph(nifti_obj, cfg.extraction, path=path)


@hydra.main(version_base=None, config_name="base_extraction_config")
def run_graph_extraction(cfg: RunGraphExtractionConfig):
    """
    Run graph extraction on a dataset and save the results as JSON files.
    Args:
        cfg: RunGraphExtractionConfig object containing extraction parameters and settings.
    """
    logger.info(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(cfg)
    ds = UnlabeledDataset(
        data_dir=cfg.input_dir,
    )

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    indices = range(len(ds)) if cfg.max_samples is None else range(cfg.max_samples)

    # Use functools.partial to pass `ds` and `cfg` to the function
    handle_idx_partial = partial(handle_idx, ds=ds, cfg=cfg)

    with (
        Pool(processes=cfg.num_workers) as pool,
        tqdm(total=len(indices), desc="Processing") as pbar,
    ):
        for _ in pool.imap_unordered(handle_idx_partial, indices, chunksize=cfg.batch_size):
            pbar.update()

    logger.info("Done")


@dataclass
class PostprocessingConfig:
    input_dir: str = MISSING
    output_dir: str = MISSING
    num_workers: int = 4
    batch_size: int = 5
    min_size: int = 1
    max_distance: int = 0
    n_largest: int = 3
    max_samples: int | None = None


_cs.store(
    name="base_postprocessing_config",
    node=PostprocessingConfig,
)


def postprocess_segmentation(idx, ds, cfg):
    """
    Apply postprocessing to a segmentation mask and save the processed result.
    Args:
        idx: Index of the sample in the dataset.
        ds: Dataset object.
        cfg: PostprocessingConfig object with postprocessing settings.
    """
    nifti_obj = ds[idx]
    file_id = Path(nifti_obj.get_filename()).stem.split(".")[0]

    binary_volume = nifti_obj.get_fdata()
    processed_volume = connected_component_distance_filter(
        binary_volume,
        cfg.min_size,
        cfg.max_distance,
        cfg.n_largest,
    )

    # Create a new NIfTI image with the processed data
    new_nifti_obj = nib.Nifti1Image(processed_volume, nifti_obj.affine, nifti_obj.header)

    # Save the new NIfTI file
    output_path = Path(cfg.output_dir) / f"{file_id}.label.nii.gz"
    nib.save(new_nifti_obj, str(output_path))


def run_post_processing(cfg: PostprocessingConfig):
    """
    Run postprocessing on a dataset and save the processed segmentation masks.
    Args:
        cfg: PostprocessingConfig object containing postprocessing parameters and settings.
    """
    logger.info(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(cfg)
    ds = UnlabeledDataset(
        data_dir=cfg.input_dir,
    )

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    indices = range(len(ds)) if cfg.max_samples is None else range(cfg.max_samples)

    # Use functools.partial to pass `ds` and `cfg` to the function
    handle_idx_partial = partial(postprocess_segmentation, ds=ds, cfg=cfg)

    with (
        Pool(processes=cfg.num_workers) as pool,
        tqdm(total=len(indices), desc="Processing") as pbar,
    ):
        for _ in pool.imap_unordered(handle_idx_partial, indices, chunksize=cfg.batch_size):
            pbar.update()

    logger.info("Done")
