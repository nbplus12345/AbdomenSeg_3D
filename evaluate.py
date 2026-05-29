import os
import sys
import time
import warnings

import torch
from dataset.dataset_3d import get_test_dataloader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import AsDiscrete
from tqdm import tqdm
from utils.config_utils import get_args, load_config
from utils.logger_utils import Logger

warnings.filterwarnings(
    "ignore",
    category=UserWarning or FutureWarning,
    message="Using a non-tuple sequence for multidimensional indexing is deprecated",
)

# ==========================================
# 1. 初始化配置与日志系统
# ==========================================
config = load_config(get_args().config)
current_time = time.strftime("%Y%m%d_%H%M")

logger = Logger(
    logger_name="AbdomenSeg_3D_Tester",
    log_file=os.path.join(config.paths.log_dir, f"AbdomenSeg_3D_Tester_{current_time}.log"),
).get_logger()

# 进度条样式配置
TQDM_BASE_CONFIG = {
    "file": sys.stdout,
    "colour": "white",
    "disable": not sys.stdout.isatty(),
    "leave": False,
    "dynamic_ncols": True,
}

logger.info("")
logger.info("=== Pre-flight Checklist ==✈")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"[INFO] Device set to: {device}")

# ==========================================
# 2. 数据加载与模型初始化 (支持多模型架构分支)
# ==========================================
test_loader = get_test_dataloader(config)
logger.info(f"[INFO] Test dataset loaded. Samples: {len(test_loader.dataset)}")

logger.info("[INFO] Initializing MONAI 3D UNet architecture...")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=config.data.num_classes,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

# 3DUNet 的最佳权重路径 (根据你的 config.yaml)
weight_path = config.paths.weight_path

# ==========================================
# 3. 加载预训练权重
# ==========================================
if os.path.exists(weight_path):
    logger.info(f"[INFO] Loading weights from: {weight_path}")
    # 仅加载模型参数字典
    model.load_state_dict(torch.load(weight_path, map_location=device))

    # 进入测试阶段，必须冻结网络，关闭 Dropout 和 BatchNorm 的动态更新
    model.eval()
    logger.info("[INFO] Model weights loaded successfully. Ready for inference.")
else:
    logger.error(f"[ERROR] Weight file not found at {weight_path}")
    sys.exit(1)

# ==========================================
# 4. 初始化指标计算器与后处理
# ==========================================
# 将网络输出的概率图转为确定的 One-Hot 编码
post_pred = AsDiscrete(argmax=True, to_onehot=config.data.num_classes)
post_label = AsDiscrete(to_onehot=config.data.num_classes)

# 计算全局平均 Dice 和 各类别平均 Dice (均排除背景 0 类)
dice_metric_mean = DiceMetric(include_background=False, reduction="mean")
dice_metric_classes = DiceMetric(include_background=False, reduction="mean_batch")

logger.info("===========================")
logger.info("")
logger.info("===== Testing Started ====✈")
logger.info("")
logger.info("========================================")
# ==========================================
# 5. 测试循环 (Test Loop)
# ==========================================
# 测试阶段严禁计算梯度，释放显存
with torch.no_grad():
    test_pbar = tqdm(test_loader, desc="[Test]", **TQDM_BASE_CONFIG)

    for test_data in test_pbar:
        test_images = test_data["image"].to(device)
        test_labels = test_data["label"].to(device)

        # 开启 AMP 混合精度推理加速
        with torch.amp.autocast("cuda"):
            # 滑动窗口推理
            test_outputs = sliding_window_inference(
                inputs=test_images,
                roi_size=config.data.patch_size,
                sw_batch_size=4,
                predictor=model,
                overlap=0.25,
            )

        # 剥离 batch 维度并清洗为 One-Hot 格式
        test_outputs_list = [post_pred(i) for i in test_outputs]
        test_labels_list = [post_label(i) for i in test_labels]

        # 喂给计算器累加当前 batch 的结果
        dice_metric_mean(y_pred=test_outputs_list, y=test_labels_list)
        dice_metric_classes(y_pred=test_outputs_list, y=test_labels_list)

# ==========================================
# 6. 汇总与输出测试结果
# ==========================================
# 触发 aggregate() 开始结算最终得分
mean_dice = dice_metric_mean.aggregate().item()
class_dice = dice_metric_classes.aggregate()  # 提取出各个类别的得分张量

# 算完之后清空计算器状态
dice_metric_mean.reset()
dice_metric_classes.reset()

logger.info(f"[RESULT] Overall Mean Dice Score: {mean_dice * 100:.2f}%")
logger.info("----------------------------------------")
logger.info("[RESULT] Per-class Dice Scores:")

# BTCV 等腹部数据集常见的 13 个器官标签名称
organ_names = [
    "Spleen",
    "Right Kidney",
    "Left Kidney",
    "Gallbladder",
    "Esophagus",
    "Liver",
    "Stomach",
    "Aorta",
    "IVC",
    "Portal Vein",
    "Pancreas",
    "Right Adrenal",
    "Left Adrenal",
]

for i, organ_name in enumerate(organ_names):
    # 如果该类别在测试集中存在（非 NaN），则打印分数
    score = class_dice[i].item()
    if not torch.isnan(class_dice[i]):
        logger.info(f"         {organ_name:<15}: {score * 100:.2f}%")
    else:
        logger.info(f"         {organ_name:<15}: N/A (Not in dataset)")

logger.info("========================================")
