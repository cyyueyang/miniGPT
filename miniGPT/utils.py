import os
import json
import random
import sys
import torch
import numpy as np
from ast import literal_eval


def set_seed(seed):
    """设置随机种子，保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_logging(config):
    """设置日志目录和配置文件"""
    work_dir = config.system.work_dir
    os.makedirs(work_dir, exist_ok=True)

    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))

    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))


class CfgNode:
    """轻量级配置类，支持嵌套"""
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def __repr__(self):
        return self.__str__()

    def _str_helper(self, indent=0):
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append(f"{' ' * indent}{k}:")
                parts.append(v._str_helper(indent + 2))
            else:
                parts.append(f"{' ' * indent}{k}: {v}")
        return '\n'.join(parts)

    def to_dict(self):
        """转换为字典"""
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result

    def merge_from_dict(self, d):
        """从字典合并配置"""
        for k, v in d.items():
            if isinstance(v, dict):
                if not hasattr(self, k):
                    setattr(self, k, CfgNode())
                getattr(self, k).merge_from_dict(v)
            else:
                setattr(self, k, v)

    def merge_from_args(self, args):
        """从命令行参数合并配置，格式: --key=value"""
        for arg in args:
            keyval = arg.split('=')
            assert len(keyval) == 2, f"参数格式错误: {arg}, 应为 --key=value"
            key, val = keyval[0], keyval[1]

            try:
                val = literal_eval(val)
            except:
                pass

            assert key[:2] == "--", f"参数应以 -- 开头: {arg}"
            key = key[2:]
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            assert hasattr(obj, leaf_key), f"{key} 不是有效配置项"
            print(f"{key}: {val} 覆盖配置")
            setattr(obj, leaf_key, val)

    def get(self, key, default=None):
        """获取配置项，支持嵌套 key，如 'model.n_layer'"""
        keys = key.split('.')
        obj = self
        for k in keys:
            if isinstance(obj, dict):
                obj = obj.get(k)
            else:
                obj = getattr(obj, k, None)
            if obj is None:
                return default
        return obj if obj is not None else default


def dict_to_cfgnode(d):
    """递归将字典转换为 CfgNode"""
    if isinstance(d, dict):
        node = CfgNode()
        for k, v in d.items():
            setattr(node, k, dict_to_cfgnode(v))
        return node
    return d
