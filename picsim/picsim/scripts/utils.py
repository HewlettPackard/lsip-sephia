"""
Author: Matěj Hejda (matej.hejda@hpe.com), based on pyutils library by Jiaqi Gu (jqgu@utexas.edu)
Date: 2024
"""

import hashlib, torch, json, yaml, os, json, yaml, numpy as np, re
from ast import literal_eval
from typing import Any, Dict, List, Tuple, Union, Optional
import torch.autograd as autograd
from multimethod import multimethod

__all__ = [
    "Config",
    "AdaptiveLossSoft",
    "KDLoss",
    "CosineAnnealingWarmupRestarts",
    "RAdam",
    "SAM",
    "MovingAverage",
]

######## CONFIG LOADERS ###########


# Enabling tuples: https://stackoverflow.com/questions/9169025/how-can-i-add-a-python-tuple-to-a-yaml-file-using-pyyaml
class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))
    
    # Override the scalar constructor to handle scientific notation
    def construct_scalar(self, node):
        value = super().construct_scalar(node)

        # Check if it's "None" (case-sensitive)
        if value == "None":
            return None
                
        # Check if it looks like scientific notation (e.g., 1e2, 1.5e-3)
        if isinstance(value, str):
            sci_notation_pattern = r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)$'
            if re.match(sci_notation_pattern, value):
                try:
                    return float(value)  # Convert to float
                except ValueError:
                    pass  # If conversion fails, return as string
        
        return value

PrettySafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", PrettySafeLoader.construct_python_tuple
)


class Config(dict):
    def __getattr__(self, key: str) -> Any:
        if key not in self:
            d = self
            ## try hierarchical access
            keys = key.split(".")
            for k in keys:
                if k not in d:
                    raise AttributeError(key)
                d = d[k]
            return d
        else:
            return self[key]

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self[key]

    # def __init__(self, logger = None):
    #     super().__init__()
    #     self.set_logger(logger)

    def load(self, fpath: str, *, recursive: bool = False) -> None:
        if not os.path.exists(fpath):
            raise FileNotFoundError(fpath)
        fpaths = [fpath]
        if recursive:
            while fpath:
                fpath = os.path.dirname(fpath)
                for fname in ["default.yaml", "default.yml"]:
                    fpaths.append(os.path.join(fpath, fname))
        for fpath in reversed(fpaths):
            if os.path.exists(fpath):
                with open(fpath, "r") as f:
                    cfg_dict = yaml.load(f, Loader=PrettySafeLoader)
                self.update(cfg_dict)

    def reload(self, fpath: str, *, recursive: bool = False) -> None:
        self.clear()
        self.load(fpath, recursive=recursive)

    @multimethod
    def update(self, other: Dict) -> None:
        for key, value in other.items():
            if isinstance(value, dict):
                if key not in self or not isinstance(self[key], Config):
                    self[key] = Config()
                self[key].update(value)
            else:
                self[key] = value

    @multimethod
    def update(self, opts: Union[List, Tuple]) -> None:
        index = 0
        while index < len(opts):
            opt = opts[index]
            if opt.startswith("--"):
                opt = opt[2:]
            if "=" in opt:
                key, value = opt.split("=", 1)
                index += 1
            else:
                key, value = opt, opts[index + 1]
                index += 2
            current = self
            subkeys = key.split(".")
            try:
                value = literal_eval(value)
            except:
                pass
            for subkey in subkeys[:-1]:
                current = current.setdefault(subkey, Config())
            current[subkeys[-1]] = value

    def dict(self) -> Dict[str, Any]:
        configs = dict()
        for key, value in self.items():
            if isinstance(value, Config):
                value = value.dict()
            configs[key] = value
        return configs

    def flat_dict(self) -> Dict[str, Any]:
        def _flatten_dict(dd, separator: str = "_", prefix: str = ""):
            return (
                {
                    prefix + separator + k if prefix else k: v
                    for kk, vv in dd.items()
                    for k, v in _flatten_dict(vv, separator, kk).items()
                }
                if isinstance(dd, dict)
                else {prefix: dd}
            )

        return _flatten_dict(self.dict(), separator=".")

    def hash(self) -> str:
        buffer = json.dumps(self.dict(), sort_keys=True)
        return hashlib.sha256(buffer.encode()).hexdigest()

    def dump_to_yml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.dict(), f)

    def __str__(self) -> str:
        texts = []
        for key, value in self.items():
            if isinstance(value, Config):
                seperator = "\n"
            else:
                seperator = " "

            text = str(key) + ":" + seperator + str(value)
            lines = text.split("\n")
            for k, line in enumerate(lines[1:]):
                lines[k + 1] = (" " * 2) + line
            texts.extend(lines)
        return "\n".join(texts)

    ## Extra function to merge and override any existing entries
    def merge_and_override(self, 
                           other: Dict, 
                           logger = None,
                           logger_msg = "") -> None:
        for key, new_value in other.items():
            if key in self:
                if isinstance(new_value, dict):
                    if key not in self or not isinstance(self[key], Config):
                        self[key] = Config()
                        if logger is not None and self[key] != new_value:
                            logger.debug(f"Platform param override ({logger_msg}): {key} ({self[key]}->{new_value})")
                    self[key].merge_and_override(new_value, 
                                                 logger = logger,
                                                 logger_msg = logger_msg)
                else:
                    self[key] = new_value
                    if logger is not None and self[key] != new_value:
                        logger.debug(f"Platform param override ({logger_msg}): {key} ({self[key]}->{new_value})")
            else:
                self[key] = new_value

# Utility function for debugging torch.Sequential models
class SequentialPrint(torch.nn.Module):
    def __init__(self, breakpoint_name):
        super(SequentialPrint, self).__init__()
        self.breakpoint_name = breakpoint_name

    def forward(self, x):
        print(f"{self.breakpoint_name} --> {x.shape}, {type(x)}")
        return x

## METRICS
class Metric:
    """Computes and stores the average and current value of a ML process metric (accuracy, loss etc.)"""

    def __init__(
        self, name: str, fmt: str = ":f", round: Optional[int] = None, history=True
    ) -> None:
        self.name = name
        self.fmt = fmt
        self.round = round
        self.reset()

    def reset(self):
        self.val = None  # most recent update
        self.sum = 0  # sum from all updates
        self.count = 0  # total n from all updates
        self.avg = 0
        self.history = []
        self.history_epochs = []

    def update(self, val, n=1):
        if val is not None:
            self.val = val
            self.history.append(val)
            if n > 0:
                self.sum = self.type_as(self.sum, val) + (val * n)
                self.count = self.type_as(self.count, n) + n
        self.avg = self.sum / self.count if self.count > 0 else self.val

    def start_epoch(self):
        self.sum = 0  # sum from all updates
        self.count = 0  # total n from all updates

    def update_epochs_end(self):
        self.history_epochs.append(self.avg)

    def state_dict(self):
        return {
            "val": self.val,
            "sum": self.sum,
            "count": self.count,
            "round": self.round,
        }

    def get_history(self):
        return np.asarray(self.history)

    def get_history_epochs(self):
        return np.asarray(self.history_epochs)

    def load_state_dict(self, state_dict):
        self.val = state_dict["val"]
        self.sum = state_dict["sum"]
        self.count = state_dict["count"]
        self.round = state_dict.get("round", None)

    @property
    def smoothed_value(self) -> float:
        val = self.avg
        if self.round is not None and val is not None:
            val = self.safe_round(val, self.round)
        return val

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} (avg: {avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def type_as(self, a, b):
        if torch.is_tensor(a) and torch.is_tensor(b):
            return a.to(b)
        else:
            return a

    def safe_round(self, number, ndigits):
        if hasattr(number, "__round__"):
            return round(number, ndigits)
        elif torch is not None and torch.is_tensor(number) and number.numel() == 1:
            return self.safe_round(number.item(), ndigits)
        elif np is not None and np.ndim(number) == 0 and hasattr(number, "item"):
            return self.safe_round(number.item(), ndigits)
        else:
            return number


# Modified Metric() for accuracy tracking
class AccuracyMetric(Metric):
    def __init__(
        self,
        configs,
        name: str,
        fmt: str = ":f",
        round: Optional[int] = None,
        history=True,
        device="cuda:0",
    ) -> None:
        super().__init__(name, fmt, round, history)

        self.device = device
        self.configs = configs

        if self.configs.criterion.name == "ce":
            self.mode = "categorical"
            self.dim = -1
        else:
            raise NotImplementedError

        if self.configs.run.save_acc_confusion_matrix:
            self.reset_confusion_matrix()

    def reset_confusion_matrix(self):
        self.n_outputs = self.configs.model.layer_widths[-1][1]
        self.confusion_matrix = torch.zeros(
            (self.n_outputs, self.n_outputs), device=self.device
        )  # (predicitons, labels)

    def acc_categorical(self, pred, target):
        with torch.no_grad():
            pred_argmax = torch.argmax(pred, dim=self.dim)
            correct = 0
            correct += torch.sum(pred_argmax == target).item()

            # Expects block_compatible batched data (that is, [batch, blockx, blocky, logits])
            if self.configs.run.save_acc_confusion_matrix:
                for batch_idx in range(pred_argmax.shape[0]):
                    self.confusion_matrix[
                        int(pred_argmax[batch_idx]),
                        int(target[batch_idx]),
                    ] += 1

            return correct / len(target)

            

    def acc_onehot(self, pred, target):
        with torch.no_grad():
            pred = torch.argmax(pred, dim=self.dim)
            target_onehot = torch.argmax(target, dim=self.dim)
            assert pred.shape[0] == len(target)
            correct = 0
            correct += torch.sum(pred == target_onehot).item()
        return correct / len(target)

    def calculate_acc(
        self,  # default is prediction, label
        pred,
        target,
    ):
        if self.mode == "categorical":
            return self.acc_categorical(pred, target)
        else:
            raise NotImplementedError

    def get_confusion_matrix(self):
        # Returns the confusion matrix
        return self.confusion_matrix


## MATH
def MovingAverage(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def NumToLetter(num):
    if num < 26:
        return chr(num + 65)
    else:
        return NumToLetter(num // 25-1) + NumToLetter(num % 25-1)

def compare_param_dict_with_regular_dict(param_dict, regular_dict):
    """Compare a torch.ParameterDict with a regular dict by comparing tensor values.
    Used for the purposes of caching the model parameters in the regular dict."""
    if set(param_dict.keys()) != set(regular_dict.keys()):
        return False
    
    for key in param_dict:
        # Check if the parameter tensor equals the dictionary value
        if isinstance(regular_dict[key], torch.Tensor):
            if not torch.equal(param_dict[key], regular_dict[key]):
                return False
        else:
            # If regular_dict contains non-tensor values
            return False
    
    return True