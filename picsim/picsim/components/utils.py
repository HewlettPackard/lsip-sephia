from typing import Optional, Tuple, Union
import torch
import random
import logging
import numpy as np
from torch import Tensor

### Convenience fucntions
def dB_to_absolute(dB):
    return torch.pow(10, dB / 10)

def dB_to_absolute_sqrt(dB: Tensor):
    return torch.pow(10, dB / 20)


def dB_to_absolute_sqrt_retCompl(dB: Tensor):
    return torch.pow(10, dB / 20).to(torch.complex64)
