import argparse
import os

import yaml


class Map(dict):
    """
    让字典支持点号访问。
    把 config['train']['lr'] 变成 config.train.lr
    """

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = self._wrap(v)
        if kwargs:
            for k, v in kwargs.items():
                self[k] = self._wrap(v)

    def _wrap(self, value):
        if isinstance(value, dict):
            return Map(value)
        elif isinstance(value, list):
            return [self._wrap(v) for v in value]
        return value

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self[key] = value


def load_config(config_path="../config/config.yaml"):
    """加载 YAML 配置文件并返回可点号访问的对象"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] DO NOT FIND Config File: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return Map(config_dict)


def get_args():
    # 用 argparse 打造一个“钥匙孔”
    parser = argparse.ArgumentParser(description="config 启动器")
    parser.add_argument(
        "--config",
        type=str,
        default="./config/config.yaml",
        help="你要使用的 YAML 配置文件路径",
    )
    return parser.parse_args()
