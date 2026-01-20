import torch
from torch import nn
from torch import Tensor
from torch.types import Device
from ..config import DEFAULT_DEVICE, get_global_wavelengths, LOGGER, get_global_configs

# Simple nn.Module for when out_features < in_features. This selects which ports (features in last tensor dimension) are used, and which dropped)
class DimReduction(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        out_features_loc="center",
        device: Device = torch.device("cpu"),
    ):
        super().__init__()

        # the out_features_loc parameter probably does not make a significant difference, therefore its pre-defined as center.
        # Might be interesting to explore for different types of meshes.

        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_loc = out_features_loc

        self.get_dimreduction_params()

    def get_dimreduction_params(self):
        # out_features_loc: "center" or anything else. Determines where the output features are taken from in the input tensor.
        if self.out_features_loc == "center":
            self.dimred_from = (self.in_features - self.out_features) // 2
            self.dimred_to = self.dimred_from + self.out_features
        elif self.out_features_loc == "top":
            self.dimred_from = 0
            self.dimred_to = -(self.in_features - self.out_features)
        else:
            raise Warning(
                f'out_features_loc for DimReduction() must be "center" or "top", was provided as {self.out_features_loc}.'
            )

    def forward(self, x: Tensor) -> Tensor:
        return x[..., self.dimred_from : self.dimred_to]


# Simple instance (sample) normalizer, without learnable parameters or affine transformation.
class InstanceNormSimple(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.normalize(x, p=2.0, dim=self.dim, eps=1e-12)