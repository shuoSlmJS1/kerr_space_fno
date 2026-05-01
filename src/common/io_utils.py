# ==========================================================
# File: src/common/io_utils.py
#
# 功能简介：
# 1. 提供统一的文件读写工具；
# 2. 提供 JSON / NPZ / NPY / TXT 的保存与读取函数；
# 3. 提供目录创建辅助函数；
# 4. 提供数据集 train/val/test 切分工具；
# 5. 提供统一日志打印辅助。
#
# 依赖关系：
# - 被 dataset_saver.py 使用
# - 被后续训练/分析模块使用
#
# 重要说明：
# - 本文件负责“怎么写文件”，不负责“写到哪里”；
# - 路径由 common/paths.py 管理；
# - 保存时默认同路径覆盖旧文件；
# - 不会自动另起新文件名。
# ==========================================================
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# ==========================================================
# 一、目录相关
# ==========================================================

def ensure_dir(path: str | Path) -> Path:
    """
    确保目录存在，不存在则自动创建。

    参数：
    - path: 目录路径

    返回：
    - Path 对象

    说明：
    - 如果目录已存在，不报错；
    - 这是后面所有保存逻辑都会频繁调用的基础函数。
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(file_path: str | Path) -> Path:
    """
    确保某个文件路径的父目录存在。

    参数：
    - file_path: 文件路径

    返回：
    - 文件路径对应的 Path 对象

    说明：
    - 只创建父目录，不创建文件本身；
    - 适合在写文件前调用。
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


# ==========================================================
# 二、JSON 读写
# ==========================================================

def save_json(data: Any, file_path: str | Path, indent: int = 2) -> Path:
    """
    将 Python 对象保存为 JSON 文件。

    参数：
    - data: 任意可 JSON 序列化对象
    - file_path: 目标文件路径
    - indent: JSON 缩进

    返回：
    - Path 对象

    覆盖行为：
    - 若 file_path 已存在，会直接覆盖旧文件；
    - 这是 Python 的标准写文件行为（mode='w'）。
    """
    file_path = ensure_parent_dir(file_path)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

    return file_path


def load_json(file_path: str | Path) -> Any:
    """
    读取 JSON 文件并返回对应 Python 对象。
    """
    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================================================
# 三、NumPy 数据读写
# ==========================================================

def save_npz(file_path: str | Path, **arrays: Any) -> Path:
    """
    将多个数组保存为压缩 npz 文件。

    参数：
    - file_path: 目标文件路径
    - arrays: 关键字数组内容，例如：
        save_npz(path, x_train=x_train, y_train=y_train)

    返回：
    - Path 对象

    覆盖行为：
    - 若 file_path 已存在，会直接覆盖旧文件；
    - np.savez_compressed 写入时默认重新写整个文件。
    """
    file_path = ensure_parent_dir(file_path)
    np.savez_compressed(file_path, **arrays)
    return file_path


def load_npz(file_path: str | Path) -> dict[str, np.ndarray]:
    """
    读取 npz 文件，并转成普通字典返回。

    返回：
    - dict[str, np.ndarray]
    """
    file_path = Path(file_path)

    with np.load(file_path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def save_npy(array: np.ndarray, file_path: str | Path) -> Path:
    """
    保存单个 numpy 数组为 .npy 文件。

    覆盖行为：
    - 若 file_path 已存在，会直接覆盖旧文件。
    """
    file_path = ensure_parent_dir(file_path)
    np.save(file_path, array)
    return file_path


def load_npy(file_path: str | Path) -> np.ndarray:
    """
    读取单个 .npy 文件。
    """
    file_path = Path(file_path)
    return np.load(file_path, allow_pickle=False)


# ==========================================================
# 四、纯文本读写
# ==========================================================

def save_text(text: str, file_path: str | Path) -> Path:
    """
    保存纯文本文件。

    覆盖行为：
    - 若 file_path 已存在，会直接覆盖旧文件。
    """
    file_path = ensure_parent_dir(file_path)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

    return file_path


def load_text(file_path: str | Path) -> str:
    """
    读取纯文本文件。
    """
    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# ==========================================================
# 五、通用摘要打印
# ==========================================================

def print_section(title: str, width: int = 70) -> None:
    """
    打印简单分隔标题。

    用途：
    - 命令行日志更清晰
    """
    line = "=" * width
    print(line)
    print(title)
    print(line)


def print_kv(key: str, value: Any, key_width: int = 24) -> None:
    """
    打印键值对风格日志。

    示例：
        输出目录                : data/tasks/...
        成功样本数              : 240
    """
    print(f"{key:<{key_width}s} : {value}")


# ==========================================================
# 六、数据切分辅助
# ==========================================================

def split_indices(
    n: int,
    split_ratios: tuple[float, float, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将样本索引随机打乱并切分为 train / val / test。

    参数：
    - n: 样本总数
    - split_ratios: (train, val, test)
    - seed: 随机种子

    返回：
    - train_idx, val_idx, test_idx

    说明：
    - 这里使用 numpy 随机打乱；
    - 后面 dataset_saver.py 会直接调用它。
    """
    if n <= 0:
        raise ValueError(f"样本总数 n 必须 > 0，当前得到：{n}")

    train_ratio, val_ratio, test_ratio = split_ratios
    s = train_ratio + val_ratio + test_ratio
    if abs(s - 1.0) > 1e-12:
        raise ValueError(f"split_ratios 三者之和必须为 1，当前得到：{split_ratios}")

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    # 防止边界情况下切分为空
    n_train = min(max(n_train, 1), n - 2) if n >= 3 else max(1, n - 1)
    n_val = min(max(n_val, 1), n - n_train - 1) if n - n_train >= 2 else max(0, n - n_train)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx