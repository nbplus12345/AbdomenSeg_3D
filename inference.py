import glob
import os
import sys

import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    SaveImage,
    ScaleIntensityRanged,
    Spacingd,
)
from tqdm import tqdm
from utils.config_utils import get_args, load_config

# ==========================================
# 1. 初始化配置
# ==========================================
config = load_config(get_args().config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=== Pre-flight Checklist ==✈")
print(f"[INFO] Device set to: {device}")

# ==========================================
# 2. 构建专属的推理数据流
# ==========================================
# 假设要预测的图片放在一个专用的 inference 文件夹，如果没有，暂时回退到 val 文件夹
inference_images_dir = getattr(
    config.paths, "inference_images", config.paths.val_images
)
image_paths = sorted(glob.glob(os.path.join(inference_images_dir, "*.nii.gz")))

# 构建字典，只有 "image" 键，没有 "label" 键
infer_files = [{"image": img_path} for img_path in image_paths]
print(
    f"[INFO] Found {len(infer_files)} images to predict in {inference_images_dir}"
)

# 推理专属的预处理流水线
infer_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=tuple(config.data.spacing),
            mode="bilinear",
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=config.data.a_min,
            a_max=config.data.a_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]
)

infer_ds = CacheDataset(data=infer_files, transform=infer_transforms, cache_rate=1.0)
infer_loader = DataLoader(infer_ds, batch_size=1, shuffle=False)

# 设置保存预测结果的输出目录
output_dir = os.path.join(config.paths.output_root, "predictions")
os.makedirs(output_dir, exist_ok=True)
print(f"[INFO] Predictions will be saved to: {output_dir}")

# ==========================================
# 3. 初始化模型并加载权重
# ==========================================
print("[INFO] Initializing MONAI 3D UNet architecture...")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=config.data.num_classes,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

weight_path = config.paths.weight_path
if os.path.exists(weight_path):
    print(f"[INFO] Loading weights from: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
else:
    print(f"[ERROR] Weight file not found at {weight_path}")
    sys.exit(1)

# ==========================================
# 4. 初始化后处理与保存工具
# ==========================================
# 预测时我们只需要每个像素属于哪个类别 (argmax)，不需要转成计算 Dice 时的 One-Hot 编码
post_pred = AsDiscrete(argmax=True)

# SaveImage 工具：负责把 Tensor 转换为 NIfTI 格式并写入硬盘
# output_postfix="seg" 会让保存的文件名自动加上后缀，如 original_name_seg.nii.gz
# separate_folder=False 防止为每个文件单独创建一个同名子文件夹
saver = SaveImage(
    output_dir=output_dir,
    output_postfix="seg",
    output_ext=".nii.gz",
    resample=False,  # 保持与当前预处理后相同的分辨率
    separate_folder=False,
    print_log=False,
)

print("===========================")
print("")
print("===== Prediction Started ====✈")
print("")

# ==========================================
# 5. 推理循环 (Inference Loop)
# ==========================================
with torch.no_grad():
    for infer_data in tqdm(infer_loader, desc="[Predict]", dynamic_ncols=True):
        infer_images = infer_data["image"].to(device)

        # 混合精度滑动窗口推理
        with torch.amp.autocast("cuda"):
            infer_outputs = sliding_window_inference(
                inputs=infer_images,
                roi_size=config.data.patch_size,
                sw_batch_size=4,
                predictor=model,
                overlap=0.25,
            )

        # 1. 拆解 Batch 维度
        # MONAI 的 decollate_batch 可以把 [1, Channels, X, Y, Z] 的大张量
        # 安全地拆成 1 个 [Channels, X, Y, Z] 的列表，防止维度错乱
        infer_outputs_list = decollate_batch(infer_outputs)
        infer_data_list = decollate_batch(infer_data)

        for pred_tensor, original_data in zip(infer_outputs_list, infer_data_list):
            # 2. 提取最大概率通道 (得到单通道的类别索引 0~13)
            pred_mask = post_pred(pred_tensor)

            # 3. 继承元数据 (Meta Data) 【极其关键】
            # 将原始图像的空间信息（原点坐标、物理间距、方向矩阵）硬拷贝给预测的掩码
            # 如果没有这一步，保存出的图像在 3D 软件里会发生严重的空间偏移或变形
            if isinstance(original_data["image"], torch.Tensor) and hasattr(
                original_data["image"], "meta"
            ):
                pred_mask.meta = original_data["image"].meta
                # 确保 Applied_operations 被正确继承，用于可能的逆向变换
                pred_mask.applied_operations = original_data["image"].applied_operations

            # 4. 保存到硬盘
            saver(pred_mask)

print(f"[INFO] Prediction finished! All masks are saved in: {output_dir}")
