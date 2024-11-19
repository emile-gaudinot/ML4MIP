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


import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, EnsureChannelFirstd, Resized, ScaleIntensityd, ToTensord
from torch import optim
from torch.utils.data import DataLoader, Subset

from ml4mip import trainer
from ml4mip.dataset import NiftiDataset


def main():
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
        # image_suffix="_avg.nii.gz",
        # mask_suffix="_avg_seg.nii.gz",
        transform=transforms,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset_size = len(train_ds)
    # subset_size = int(0.1 * dataset_size)  # Use 10% of the dataset
    # subset_indices = torch.randperm(dataset_size)[:subset_size]
    # train_subset = Subset(train_ds, subset_indices)
    
    ds = train_ds
    # ds = train_subset
    logging.info(f"Training on {len(ds)} samples")

    
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True)

    # val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    # Model and optimizer
    model = torch.jit.load(model_path, map_location=device)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    # Fine-tuning
    trainer.finetune(model, train_loader, optimizer, loss_fn, dice_metric, device, num_epochs)
    # Save the final model
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "fine_tuned_model.pt")


if __name__ == "__main__":
    main()
