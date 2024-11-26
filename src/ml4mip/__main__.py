import logging
from pathlib import Path

# import sys

LOG_DIR = Path("/group/cake/ML4MIP/logs")
# Configure logging to output to stdout
# Configure logging to write to a file
logging.basicConfig(
    filename=LOG_DIR / "training.log",  # Name of the log file
    filemode="a",  # Append mode (use 'w' for overwrite)
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
)


import argparse

import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, EnsureChannelFirstd, Resized, ScaleIntensityd, ToTensord
from torch import optim
from torch.utils.data import DataLoader

from ml4mip import trainer
from ml4mip.dataset import NiftiDataset


def finetune_unetr():
    logging.info("Starting unetr finetuning script")
    model_dir = Path("/group/cake/ML4MIP/models")

    model_path = model_dir / "UNETR_model_best_acc.pt"
    # !ls data/training_data/
    data_dir = Path("/data/training_data")
    print(model_dir, model_dir.exists())
    print(model_path, model_path.exists())
    print(data_dir, data_dir.exists())

    batch_size = 4
    lr = 1e-4
    num_epochs = 10

    transforms = Compose(
        [
            EnsureChannelFirstd(keys=["image", "mask"]),
            # Spacingd(keys=["image", "mask"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            ScaleIntensityd(keys=["image"]),
            Resized(
                keys=["image", "mask"],
                spatial_size=(96, 96, 96),
                mode=("trilinear", "nearest"),  # 'trilinear' for image, 'nearest' for mask
                align_corners=(True, None),  # 'align_corners' is only relevant for 'trilinear' mode
            ),
            ToTensord(keys=["image", "mask"]),
        ]
    )

    train_ds = NiftiDataset(
        data_dir,
        transform=transforms,
        train=True,
    )
    val_ds = NiftiDataset(
        data_dir,
        transform=transforms,
        train=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    msg = f"Training on {len(train_ds)} samples"
    logging.info(msg)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    # Model and optimizer
    model = torch.jit.load(model_path, map_location=device)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    # Fine-tuning
    trainer.finetune(
        model,
        train_loader,
        optimizer,
        loss_fn,
        dice_metric,
        device,
        num_epochs,
        val_loader=val_loader,
    )
    # Save the final model
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "fine_tuned_model.pt")


def evaluate_unetr():
    logging.info("Starting unetr evaluation script")
    model_dir = Path("/group/cake/ML4MIP/models")
    msg = f"Model directory: {model_dir} ({model_dir.exists()})"

    model_path = model_dir / "fine_tuned_model.pt"
    msg = f"Model path: {model_path} ({model_path.exists()})"

    model_path_basic = model_dir / "UNETR_model_best_acc.pt"
    msg = f"Model path basic: {model_path_basic} ({model_path_basic.exists()})"
    logging.info(msg)

    data_dir = Path("/data/training_data")
    msg = f"Data directory: {data_dir} ({data_dir.exists()})"

    batch_size = 4

    transforms = Compose(
        [
            EnsureChannelFirstd(keys=["image", "mask"]),
            # Spacingd(keys=["image", "mask"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            ScaleIntensityd(keys=["image"]),
            Resized(
                keys=["image", "mask"],
                spatial_size=(96, 96, 96),
                mode=("trilinear", "nearest"),  # 'trilinear' for image, 'nearest' for mask
                align_corners=(True, None),  # 'align_corners' is only relevant for 'trilinear' mode
            ),
            ToTensord(keys=["image", "mask"]),
        ]
    )

    val_ds = NiftiDataset(
        data_dir,
        transform=transforms,
        train=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    msg = f"Evaluating on {len(val_ds)} samples"
    logging.info(msg)

    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Model and optimizer

    model = torch.jit.load(model_path_basic, map_location=device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    # Evaluation
    result = trainer.validate(model, loader, loss_fn, dice_metric, device)
    msg = f"Validation Dice: {result['avg_dice']:.4f} | Validation Loss: {result['val_loss']:.4f}"
    logging.info(msg)


def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Choose to train or evaluate the model.")

    # Add arguments for mode
    parser.add_argument(
        "mode",
        choices=["train", "evaluate"],
        help="Select 'train' to train the model or 'evaluate' to evaluate the model.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the appropriate function based on the mode
    if args.mode == "train":
        finetune_unetr()
    elif args.mode == "evaluate":
        evaluate_unetr()


if __name__ == "__main__":
    main()
