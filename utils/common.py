import torch
import random
import json
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def checkSameConfigs(config1, config2):
    ignored_keys = {"exp_dir"}

    c1 = {k: v for k, v in config1.items() if k not in ignored_keys}
    c2 = {k: v for k, v in config2.items() if k not in ignored_keys}

    return c1 == c2

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return json.JSONEncoder.default(self, obj)

