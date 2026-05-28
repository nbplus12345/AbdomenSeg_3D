import os
import sys
import time
import warnings

import torch
from dataset.dataset_3d import get_3d_dataloaders
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import AsDiscrete
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.config_utils import get_args, load_config
from utils.logger_utils import Logger

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Using a non-tuple sequence for multidimensional indexing is deprecated",
)
config = load_config(get_args().config)
total_start_time = time.time()
current_time = time.strftime("%Y%m%d_%H%M")
logger = Logger(
    logger_name="AbdomenSeg_3DUNet_Trainer",
    log_file=f"output/logs/AbdomenSeg_3DUNet_Trainer_{current_time}.log",
).get_logger()
TQDM_BASE_CONFIG = {
    "file": sys.stdout,
    "colour": "white",
    "disable": not sys.stdout.isatty(),
    "leave": False,
    "dynamic_ncols": True,
}

logger.info("")
logger.info("=== Pre-flight Checklist ===✈")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"[INFO] Device set to: {device}")

# 获取 DataLoader
train_loader, val_loader = get_3d_dataloaders(config)
logger.info(
    f"[INFO] Dataset loaded. Training samples: {len(train_loader.dataset)} | Validation samples: {len(val_loader.dataset)}"
)

# 初始化 3D U-Net 模型
model = UNet(
    spatial_dims=3,  # 3D 网络
    in_channels=1,  # 输入 1 通道 (单模态 CT)
    out_channels=14,  # 输出 14 个通道 (1背景 + 13器官)
    channels=(
        16,
        32,
        64,
        128,
        256,
    ),  # 为了节省 3D 显存，基础通道数比 2D 缩小了 (16起步)
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)
logger.info(
    f"[INFO] MONAI 3D UNet Initialized. Total params: {sum(p.numel() for p in model.parameters()):,}"
)

# 配置多分类损失函数与优化器
criterion = DiceCELoss(
    to_onehot_y=True,  # 【关键参数】标签原本是0~13的整数，计算Dice前需要自动转换为14通道的One-Hot编码
    softmax=True,  # 【关键参数】多分类网络输出必须用 Softmax，绝不能用 Sigmoid
    include_background=False,  # 计算指标时不考虑背景类（0类），只关注13个器官的分割质量
)
logger.info(
    "[INFO] Criterion: MONAI DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)"
)

# 使用 AdamW 优化器 (对 3D 任务通常比 SGD 收敛更快)
optimizer = optim.AdamW(
    model.parameters(),
    lr=config.train.lr,
    weight_decay=float(config.train.weight_decay),
)
logger.info(
    f"[INFO] Optimizer configured: AdamW (lr={config.train.lr}, weight_decay={config.train.weight_decay})"
)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=config.train.scheduler_factor,
    patience=config.train.scheduler_patience,
    min_lr=1e-5,
)
logger.info(
    f"[INFO] Scheduler: ReduceLROnPlateau (factor={config.train.scheduler_factor}, patience={config.train.scheduler_patience})"
)

# tensorboard 日志目录
tb_log_dir = os.path.join("./output", f"tensorboard/Board_{current_time}")
board_writer = SummaryWriter(log_dir=tb_log_dir)

# AMP 梯度缩放器（Automatic Mixed Precision）
scaler = torch.amp.GradScaler()  # 初始化梯度缩放器，防止 float16 梯度下溢出

max_epochs = config.train.epochs
val_interval = config.train.val_interval
best_val_dice = 0.0
global_step = 0
counter = 0
logger.info(
    f"[INFO] Early Stopping configured. Patience: {config.train.patience} epochs, max_epochs: {max_epochs}"
)

# 断点续训
start_epoch = 0
if config.train.resume_training and os.path.exists(config.paths.checkpoint_path):
    # 1. 把包裹取回来
    checkpoint = torch.load(config.paths.checkpoint_path, map_location=device)
    # 2. 依次把记忆注入到对应的身体里
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1  # 从断点的下一轮开始
    best_val_dice = checkpoint["best_val_dice"]
    counter = checkpoint["counter"]
    global_step = checkpoint.get("global_step", 0)
    logger.info(
        f"[INFO] Find Checkpoint {config.paths.checkpoint_path}, Continue training from epoch {start_epoch + 1} ."
    )

# 初始化验证指标计算器
# 遇到多分类问题，AsDiscrete 是必不可少的清洗工具
post_pred = AsDiscrete(
    argmax=True, to_onehot=14
)  # 将概率最大值提取出来，转为14通道one-hot
post_label = AsDiscrete(to_onehot=14)  # 标签本身是0~13的整数，也转为14通道one-hot
dice_metric = DiceMetric(
    include_background=False, reduction="mean"
)  # 计算所有 batch 的平均 Dice (不算背景)

logger.info("============================")
logger.info("")
logger.info("\n===== Training Started ====✈\n")
logger.info("==========================================================")


# ETA 计算函数
def format_eta(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s"


avg_epoch_time = 0.0
completed_epochs = 0

# 开始 Epoch 循环
for epoch in range(start_epoch, max_epochs):
    epoch_start_time = time.time()
    model.train()
    epoch_loss = 0.0
    step = 0  # 记录当前 epoch 的训练步数，用于计算平均 loss 和日志记录

    # ==========================================
    # 训练阶段 (Training Loop)
    # ==========================================
    # 使用 tqdm 进度条包裹训练加载器
    train_pbar = tqdm(train_loader, desc="[Train]", **TQDM_BASE_CONFIG)
    for batch_data in train_pbar:
        step += 1
        global_step += 1  # 训练步数递增，用于全局日志记录（跨 epoch 累积）

        # 将 3D 图像和标签推入显卡
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        optimizer.zero_grad()

        # 开启 AMP 上下文管理器
        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # AMP 的反向传播三步曲
        # 1. 放大 loss 并反向传播
        scaler.scale(loss).backward()
        # 2. 缩放回梯度并更新权重
        scaler.step(optimizer)
        # 3. 更新 scaler 的内部倍率
        scaler.update()

        # 记录与打印日志
        epoch_loss += loss.item()
        board_writer.add_scalar("Train/Step_Loss", loss.item(), global_step)
        train_pbar.set_postfix(
            {"Loss": f"{loss.item():.4f}"}
        )  # 在 tqdm 进度条上显示当前 step 的 loss

    epoch_train_avg_loss = epoch_loss / step
    board_writer.add_scalar("Train/Epoch_Loss", epoch_train_avg_loss, epoch)

    # ==========================================
    # 验证阶段 (Validation Loop)
    # ==========================================
    if (epoch + 1) % val_interval == 0:
        model.eval()
        val_total_loss = 0.0
        val_step = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="[Val]", **TQDM_BASE_CONFIG)
            for val_data in val_pbar:
                val_step += 1
                val_images = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)

                # 【核心知识】：滑动窗口推理
                # roi_size: 窗口大小，必须和训练时的 patch_size 一致
                # sw_batch_size: 每次并行跑几个窗口（根据显存决定，一般设为 4 没问题）
                # overlap: 窗口重叠率 25%，保证拼接边缘平滑
                with torch.amp.autocast("cuda"):
                    val_outputs = sliding_window_inference(
                        inputs=val_images,
                        roi_size=config.data.patch_size,
                        sw_batch_size=4,
                        predictor=model,
                        overlap=0.25,
                    )
                    loss = criterion(val_outputs, val_labels)
                    val_total_loss += loss.item()

                # 【核心知识】：离散化并计算 Dice
                # 把每个 batch 的输出和标签变成 list，分别进行 AsDiscrete 清洗
                val_outputs = [post_pred(i) for i in val_outputs]
                val_labels = [post_label(i) for i in val_labels]
                # 将清洗后的数据喂给计算器
                dice_metric(y_pred=val_outputs, y=val_labels)

            # 汇总所有验证集病人的平均 Dice 分数
            val_avg_dice = dice_metric.aggregate().item()
            dice_metric.reset()  # 算完之后清空，留给下一个 Epoch
            val_avg_loss = val_total_loss / len(val_loader)
            board_writer.add_scalar("Val/Epoch_Loss", val_avg_loss, epoch)
            board_writer.add_scalar("Val/Epoch_Dice", val_avg_dice, epoch)

            # 注意：因为调度器监控的是 Dice (越高越好)，你要确保初始化时设的是 mode="max"
            scheduler.step(val_avg_dice)
            current_lr = optimizer.param_groups[0]["lr"]

            epoch_time = time.time() - epoch_start_time
            completed_epochs += 1
            avg_epoch_time += (epoch_time - avg_epoch_time) / completed_epochs
            remaining_epochs = max_epochs - epoch - 1
            eta = avg_epoch_time * remaining_epochs
            logger.info("----------------------------------------------------------")
            logger.info(
                f"[Epoch {epoch + 1:03d}/{max_epochs:03d}] | Train Loss: {epoch_train_avg_loss:.4f}  |   Time    : {int(epoch_time // 60)}m {int(epoch_time % 60):02d}s | ETA: {format_eta(eta)}"
            )
            logger.info(
                f"   Lr : {current_lr:8f} |  Val Loss: {val_avg_loss:.4f}   | Val Dice  : {val_avg_dice:.4f}"
            )
            logger.info("==========================================================")

            # 早停与保存最佳权重
            if val_avg_dice > best_val_dice:
                best_val_dice = val_avg_dice
                counter = 0

                # 保存最佳模型
                torch.save(model.state_dict(), config.paths.weight_path)
                logger.info(
                    f"[SAVE] New best record ： {val_avg_dice * 100:.2f}% ! Model saved!"
                )
                logger.info("")
                logger.info(
                    "=========================================================="
                )
            else:
                counter += 1
                logger.info(
                    f"[WARN] No improvement. Patience: {counter}/{config.train.patience}"
                )
                logger.info("")
                logger.info(
                    "=========================================================="
                )

                if counter >= config.train.patience:
                    board_writer.close()
                    total_time = time.time() - total_start_time
                    logger.info("[STOP] Early stopping triggered! Training halted.")
                    logger.info(
                        f"       Final best Val Dice: {best_val_dice:.4f} | Time: {int(total_time // 60)}m {int(total_time % 60):02d}s"
                    )
                    logger.info(
                        "=========================================================="
                    )
                    break
    else:
        epoch_time = time.time() - epoch_start_time
        completed_epochs += 1
        avg_epoch_time += (epoch_time - avg_epoch_time) / completed_epochs
        remaining_epochs = max_epochs - epoch - 1
        eta = avg_epoch_time * remaining_epochs
        logger.info(
            f"[Epoch {epoch + 1:03d}/{max_epochs:03d}] | Train Loss: {epoch_train_avg_loss:.4f}  |   Time    : {int(epoch_time // 60)}m {int(epoch_time % 60):02d}s | ETA: {format_eta(eta)}"
        )
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_dice": best_val_dice,
        "global_step": global_step,
        "counter": counter,
    }
    torch.save(checkpoint, config.paths.checkpoint_path)
