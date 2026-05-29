import glob
import os

from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)


def get_3d_transforms(config):
    """
    定义 3D 图像的预处理与数据增强流水线
    """
    patch_size = tuple(config.data.patch_size)
    spacing = tuple(config.data.spacing)
    a_min = config.data.a_min
    a_max = config.data.a_max
    num_samples = config.data.num_samples

    # --- 训练集专属 Pipeline ---
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(
                keys=["image", "label"]
            ),  # 把读取的数据变成 (Channel, X, Y, Z) 的格式
            Orientationd(
                keys=["image", "label"], axcodes="RAS"
            ),  # 强制统一重定向到 RAS（Right, Anterior, Superior）标准解剖学坐标系
            Spacingd(  # 非常重要！ 把不同切片厚度的 CT 强制插值重采样到统一的物理分辨率
                keys=["image", "label"],
                pixdim=spacing,
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(  # 窗宽窗位归一化
                keys=["image"],
                a_min=a_min,
                a_max=a_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(  # 裁剪掉 CT 图像周围的全黑背景，减少无效计算
                keys=["image", "label"], source_key="image"
            ),
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=patch_size,
                mode="constant",
            ),
            # --- 核心：3D Patch 随机裁剪 ---
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            # --- 3D 数据增强 ---
            RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.5,
                spatial_size=patch_size,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
            RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.5,
                spatial_size=patch_size,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            # --- 3D 数据增强 (强度与噪声) ---
            # 适当调高高斯噪声的触发概率到 0.2
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
            # 随机偏移图像整体亮度，模拟不同CT机的本底差异
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.5,
            ),
            # 随机 Gamma 对比度变换，增强模型对模糊边界器官的辨识力
            RandAdjustContrastd(
                keys=["image"],
                prob=0.5,
                gamma=(0.5, 2.0),
            ),
        ]
    )

    # --- 验证集 Pipeline ---
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=spacing,
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=a_min,
                a_max=a_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )

    return train_transforms, val_transforms


def get_3d_dataloaders(config):
    """
    构建 DataLoader
    """
    train_images = sorted(
        glob.glob(os.path.join(config.paths.train_images, "*.nii.gz"))
    )
    train_labels = sorted(
        glob.glob(os.path.join(config.paths.train_labels, "*.nii.gz"))
    )

    val_images = sorted(glob.glob(os.path.join(config.paths.val_images, "*.nii.gz")))
    val_labels = sorted(glob.glob(os.path.join(config.paths.val_labels, "*.nii.gz")))

    # 构建 MONAI 需要的字典格式
    train_files = [
        {"image": img, "label": lbl} for img, lbl in zip(train_images, train_labels)
    ]
    val_files = [
        {"image": img, "label": lbl} for img, lbl in zip(val_images, val_labels)
    ]

    print(f"[INFO] Training samples: {len(train_files)}")
    print(f"[INFO] Validation samples: {len(val_files)}")

    train_transforms, val_transforms = get_3d_transforms(config)

    print("[INFO] Building Train CacheDataset...")
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=config.train.num_worker,
    )

    print("[INFO] Building Validation CacheDataset...")
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=config.train.num_worker,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_worker,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.train.num_worker,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(config):
    """专门用于测试集的 DataLoader，跳过训练集加载"""
    # 如果配置文件中有单独的 test_images，则用 test，否则暂时回退使用 val 作为测试集
    test_images_path = getattr(config.paths, "test_images", config.paths.val_images)
    test_labels_path = getattr(config.paths, "test_labels", config.paths.val_labels)

    test_images = sorted(glob.glob(os.path.join(test_images_path, "*.nii.gz")))
    test_labels = sorted(glob.glob(os.path.join(test_labels_path, "*.nii.gz")))

    test_files = [
        {"image": img, "label": lbl} for img, lbl in zip(test_images, test_labels)
    ]

    # 获取验证集/测试集通用的预处理 Pipeline
    _, val_transforms = get_3d_transforms(config)

    test_ds = CacheDataset(
        data=test_files,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=config.train.num_worker,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,  # 3D 测试 batch_size 必须为 1
        shuffle=False,
        num_workers=config.train.num_worker,
        pin_memory=True,
    )

    return test_loader
