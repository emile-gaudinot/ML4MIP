# %%
from ml4mip.models import get_model, ModelConfig, ModelType
from ml4mip.dataset import get_transform, TransformType
from ml4mip.trainer import inference, InferenceConfig
from ml4mip.utils.metrics import get_metrics, MetricType
from monai.transforms import (
    Compose,
    LoadImaged,
    MapTransform,
    Randomizable,
    RandSpatialCropd,
    Resized,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    NormalizeIntensityd,
    Spacingd,
    ToTensord,
)
import torch

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#/group/cake/ML4MIP/models/UNETMONAI2_128_finetuned_1/final_model.pt
#/group/cake/ML4MIP/models/UNETMONAI2_192_2/manual_stop_model.pt
model_cfg = ModelConfig(model_type=ModelType.UNETMONAI2, model_path='/group/cake/ML4MIP/models/UNETMONAI2_128_finetuned_1/final_model.pt')
model = get_model(model_cfg)
model.to(device)
model.eval()

# %%
path_img = '/data/training_data/5d8f6c.img.nii.gz'
path_msk = '/data/training_data/5d8f6c.label.nii.gz'
image_loader = LoadImaged(keys=["image", "mask"], ensure_channel_first=True)

path_list = {"image": path_img, "mask": path_msk}
image = image_loader(path_list)
transform = get_transform(TransformType.STD)
image = transform(image)

# %%
image['image'].shape

# %%
import gc
gc.collect()
torch.cuda.empty_cache()

# %%
inference_cfg = InferenceConfig(sw_size=(128, 128, 128), sw_batch_size=1)
metrics_manager = get_metrics([MetricType.DICE])
metrics_manager.reset()

with torch.no_grad():
    inference_image = torch.unsqueeze(image["image"], 0).to(device)
    out = inference(inference_image, model, inference_cfg)
    metrics_manager(y=torch.unsqueeze(image['mask'], 0).to(device), y_pred=out)
    
metrics_manager.aggregate()

# %%
transform.inverse(out)

# %%



