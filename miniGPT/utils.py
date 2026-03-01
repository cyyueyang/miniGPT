import os
import json
import random
import sys
import torch
import numpy as np
from ast import literal_eval

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def setup_logging(config):
    work_dir = config.system.work_dir
    os.makedirs(work_dir, exist_ok=True)

    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))

    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))

class CfgNode:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent=0):
        parts = []

        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append(f"{k}:\n")
                parts.append(v._str_helper(indent+2))
            else:
                parts.append(f"{k}:{v}\n")
        parts = [' ' * (indent+4) + p for p in parts]
        return ''.join(parts)

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items()}

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        for arg in args:

            keyval = arg.split('=')
            assert len(keyval) == 2
            key, val = keyval[0], keyval[1]

            try:
                val = literal_eval(val)
            except:
                pass

            assert key[:2] == "--"
            key = key[2:]
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            assert hasattr(obj, leaf_key), f"{key} is not an attribute"

            print(f"{key}: {val} is overwrite into config")
            setattr(obj, leaf_key, val)


