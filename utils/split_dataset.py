import os
import random
import re
import shutil

from config_utils import get_args, load_config
from tqdm import tqdm


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_medical_dataset(
    src_image_dir, src_label_dir, output_root, split_ratios=(0.7, 0.1, 0.2), seed=42
):
    """
    Split medical image dataset based on proportions.
    :param src_image_dir: Path to raw images.
    :param src_label_dir: Path to raw labels.
    :param output_root: Output root directory.
    :param split_ratios: Proportions for (Train, Val, Test). Sum must be 1.0.
    :param seed: Random seed for reproducibility.
    """
    # 锁定随机种子
    random.seed(seed)

    # 1. 获取所有图像文件
    image_files = sorted(
        [f for f in os.listdir(src_image_dir) if f.endswith(".nii.gz")]
    )
    total_files = len(image_files)

    print(f"[INFO] Found a total of {total_files} CT images.")

    if total_files == 0:
        print("[ERROR] No .nii.gz files found. Please check the directory path!")
        return

    # 2. 校验 Label 是否一一对应
    label_files = sorted([f for f in os.listdir(src_label_dir) if f.endswith(".nii.gz")])

    def extract_case_id(filename, prefix):
        match = re.fullmatch(rf"{re.escape(prefix)}(\d+)\.nii\.gz", filename)
        if match is None:
            raise ValueError(
                f"[ERROR] Invalid {prefix} file name: {filename}. "
                f"Expected format like {prefix}0001.nii.gz"
            )
        return match.group(1)

    label_by_id = {}
    for lbl_name in label_files:
        case_id = extract_case_id(lbl_name, "label")
        if case_id in label_by_id:
            raise ValueError(
                f"[ERROR] Duplicate label id {case_id}: "
                f"{label_by_id[case_id]} and {lbl_name}"
            )
        label_by_id[case_id] = lbl_name

    image_ids = set()
    label_by_image = {}
    for img_name in image_files:
        case_id = extract_case_id(img_name, "img")
        image_ids.add(case_id)
        lbl_name = label_by_id.get(case_id)
        if lbl_name is None:
            raise FileNotFoundError(
                f"[ERROR] Corresponding label file not found for {img_name}. "
                f"Expected label id: {case_id}"
            )
        label_by_image[img_name] = lbl_name

    extra_label_ids = sorted(set(label_by_id) - image_ids)
    if extra_label_ids:
        raise FileNotFoundError(
            f"[ERROR] Label files without corresponding images: {extra_label_ids}"
        )

    # 3. 打乱数据
    random.shuffle(image_files)

    # 4. 比例校验与切分点计算
    if abs(sum(split_ratios) - 1.0) > 1e-5:
        print(
            f"[WARN] Split ratios {split_ratios} do not sum to 1.0! The results might be unexpected."
        )

    train_end = int(total_files * split_ratios[0])
    val_end = train_end + int(total_files * split_ratios[1])

    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    # 剩下的全部给测试集，防止除不尽
    test_files = image_files[val_end:]

    print(
        f"[INFO] Split plan -> Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}"
    )

    # 5. 创建目标文件夹结构
    subsets = ["train", "val", "test"]
    for subset in subsets:
        create_dir(os.path.join(output_root, subset, "images"))
        create_dir(os.path.join(output_root, subset, "labels"))

    # 6. 定义复制函数
    def copy_files(file_list, subset_name):
        for f in tqdm(file_list, desc=f"[INFO] Copying {subset_name} data"):
            # 原路径
            src_img = os.path.join(src_image_dir, f)
            label_name = label_by_image[f]
            src_lbl = os.path.join(src_label_dir, label_name)

            # 目标路径
            dst_img = os.path.join(output_root, subset_name, "images", f)
            dst_lbl = os.path.join(output_root, subset_name, "labels", label_name)

            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)

    # 7. 开始复制
    print("\n --- Starting file copy ---")
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

    print(
        f"\n[SUCCESS] Dataset split and copy completed! Data saved to: {os.path.abspath(output_root)}"
    )


if __name__ == "__main__":
    config = load_config(get_args().config)
    # ==========================
    # 在这里修改为你的实际路径
    # ==========================
    SOURCE_IMAGES = config.paths.source_images
    SOURCE_LABELS = config.paths.source_labels

    # 我们希望输出到项目里的 dataset 目录
    OUTPUT_ROOT = "./data"

    # Train比例, Val比例, Test比例 （如果是整数，表示具体数量；如果是小数，表示比例）
    SPLIT_PLAN = (0.7, 0.1, 0.2)

    split_medical_dataset(
        src_image_dir=SOURCE_IMAGES,
        src_label_dir=SOURCE_LABELS,
        output_root=OUTPUT_ROOT,
        split_ratios=SPLIT_PLAN,
        seed=config.common.seed,  # 锁定种子，防止不小心运行两次切出不一样的数据
    )
