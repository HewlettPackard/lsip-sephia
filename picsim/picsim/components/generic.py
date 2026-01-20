import torch
import torch.nn as nn
import numpy as np
from ..config import DEFAULT_DEVICE, get_global_wavelengths, LOGGER, get_global_configs, get_global_opt_power_unit

class BaseComponentModule(nn.Module):
    def __init__(self):
        super().__init__()

    # @property
    # def arch_space(self):
    #     return None

    def build_transform(self):
        raise NotImplementedError

class EOConverter(nn.Module):
    def __init__(self, 
                 inp_format = "bdws",
                 inp_type = "power", # "power" or "amplitude"
                 wavelengths = None,
                 normalize_over_spatial=False, # Prior to entering the O domain, normalize the input to have a max value of 1
                 unity_opt_P_per_WDM_channel_Watt = 1e-3, # Sets the optical power (in watt!) of a single WDM channel for the value '1' encoded in the channel
                 spatial_dim_broadcast_n = None,
                 spatial_dim_broadcast_divide_power=False,
                 neg_into_pi_shift = True,
                 eps=1e-6,
                 device=DEFAULT_DEVICE,
                ):
        super().__init__()
        self.__name__ = "EOConverter"

        self.device = device
        self._eps = eps

        _allowed_inp_formats = ["bdws", "bws", "bw", "bds", "bs",]
        assert inp_format in _allowed_inp_formats, f"Invalid input format provided. Expected one of {_allowed_inp_formats}, got {inp_format}."
        self.inp_format = inp_format

        if inp_format == "bw":
            if spatial_dim_broadcast_n is not None:
                assert isinstance(spatial_dim_broadcast_n, int), "Spatial dimension broadcast factor must be an integer."
                self.set_spatial_dim_broadcast(n=spatial_dim_broadcast_n, divide_power=spatial_dim_broadcast_divide_power)
            else:
                self.set_spatial_dim_broadcast(n=1, divide_power=spatial_dim_broadcast_divide_power) # by default, no expansion is applied (n=1)

        self.inp_type = inp_type

        self.normalize = normalize_over_spatial
        if wavelengths is None:
            self.wavelengths = get_global_wavelengths()
        else:
            self.set_wavelengths(wavelengths)

        self.unity_opt_P_per_WDM_channel = unity_opt_P_per_WDM_channel_Watt
        self.sgn_into_phase_flag = neg_into_pi_shift

        # assert output_unit in ["mW", "W"], "Invalid output unit provided. Must be either 'mW' or 'W'."
        # self.output_unit = output_unit
        if get_global_opt_power_unit() == "mW":
            self.opt_P_unit_scaling = 1e3
        elif get_global_opt_power_unit() == "W":
            self.opt_P_unit_scaling = 1
        elif get_global_opt_power_unit() == "uW":
            self.opt_P_unit_scaling = 1e6
        else:
            raise ValueError(f"Invalid (global) optical power unit in EOConv(). Must be one of ['mW', 'W', 'uW'].")

    def set_wavelengths(self, wavelengths):
        assert torch.is_tensor(wavelengths), "Wavelengths must be a tensor."
        assert len(wavelengths.shape) == 1, "Wavelengths must be a 1D tensor."
        self.wavelengths = wavelengths

    def normalize_tensor(self, tensor, dim=-1):
        min_val, _ = torch.min(tensor, dim=dim, keepdim=True)
        max_val, _ = torch.max(tensor, dim=dim, keepdim=True)
        
        # Check if max equals min (including the all-zeros case)
        range_is_zero = (max_val == min_val) < self._eps
        
        # Where range is zero, return zeros instead of attempting division
        # Otherwise perform the normalization as usual
        scaled_tensor = torch.where(
            range_is_zero,
            torch.ones_like(tensor),
            (tensor - min_val) / (max_val - min_val + self._eps)
        )
        
        return scaled_tensor

    def set_spatial_dim_broadcast(self, 
                                  n = 1,
                                  divide_power=False):
        assert self.inp_format == "bw", "Spatial dimension broadcasting is only applicable when the input format is 'bw' (batch, wavelength)."
        assert n>= 1, "Spatial dimension broadcast factor must be greater than or equal to 1."

        self.broadcast_spatial_dim_factor = n
        if n > 1:
            if divide_power:
                self.broadcast_spatial_power_division = (1/n)
                LOGGER.debug(f"EOConv spatial dimension broadcasting: active to {n} spatial outputs. divide_power is active -> inputs multiplied by (1/{n}) = {self.broadcast_spatial_power_division:.4f}.")
            else:
                self.broadcast_spatial_power_division = 1
                LOGGER.debug(f"EOConv spatial dimension broadcasting: active to {n} spatial outputs. divide_power is inactive.")
        else:
            self.broadcast_spatial_power_division = 1

    def forward(self, x):
        source_device = x.device
        assert torch.is_tensor(x), "Input to EOConverter must be a tensor."

        if self.normalize:
            x = self.normalize_tensor(x)

        if self.inp_format == "bdws":
            assert len(x.shape) == 4, f'Input to EOConverter must be a 4D tensor: ["batch", "decomp", "wl", "spatial_feature"], got {x.shape=}.'
            assert x.shape[-2] == self.wavelengths.shape[0], f"Size mismatch between input and provided wavelengths. Got {x.shape[-2]=}, expected {self.wavelengths.shape[0]=}. Make sure to initiate with the wavelengths arg, or use set_wavelengths()"
        elif self.inp_format == "bws":
            assert len(x.shape) == 3, f'Input to EOConverter must be a 3D tensor: ["batch", "wl", "spatial_feature"], got {x.shape=}.'
            assert x.shape[-2] == self.wavelengths.shape[0], f"Size mismatch between input and provided wavelengths. Got {x.shape[-2]=}, expected {self.wavelengths.shape[0]=}. Make sure to initiate with the wavelengths arg, or use set_wavelengths()"
            x = x.unsqueeze(1)
        elif self.inp_format == "bds":
            assert len(x.shape) == 3, f'Input to EOConverter must be a 3D tensor: ["batch", "decomp", "spatial_feature"], got {x.shape=}.'
            #assert x.shape[-2] == self.wavelengths.shape[0], f"Size mismatch between input and provided wavelengths. Got {x.shape[-2]=}, expected {self.wavelengths.shape[0]=}. Make sure to initiate with the wavelengths arg, or use set_wavelengths()"
            x = x.unsqueeze(-2)
        elif self.inp_format == "bw":
            assert len(x.shape) == 2, f'Input to EOConverter must be a 2D tensor: [batch, "wl"], got {x.shape=}.'
            assert x.shape[-1] == self.wavelengths.shape[0], f"Size mismatch between input and provided wavelengths. Got {x.shape[-1]=}, expected {self.wavelengths.shape[0]=}. Make sure to initiate with the wavelengths arg, or use set_wavelengths()"
            x = x.unsqueeze(1).unsqueeze(-1).repeat(1, 1, 1, self.broadcast_spatial_dim_factor)*self.broadcast_spatial_power_division
        elif self.inp_format == "bs":
            assert len(x.shape) == 2, f'Input to EOConverter must be a 2D tensor: [batch, "spatial_feature"], got {x.shape=}.'
            x = x.unsqueeze(1).unsqueeze(-2)

        if not torch.is_complex(x):

            # Perform sqrt when input corresponds to power
            if self.inp_type == "power":
                x = x.sqrt()

            # Applies scaling to the desired power units (per channel)
            y = (x*np.sqrt(self.unity_opt_P_per_WDM_channel*self.opt_P_unit_scaling)).type(torch.complex64)

            return y.type(torch.complex64).to(self.device)
        
        else:
            raise Warning("Input tensor to EOConverter is complex, while its expected to be real. The convention is that complex tensors represent electric fields (optical signals) only.")

        
class QuantizeToNBits(nn.Module):
    def __init__(self, n_bits, min_val=0.0, max_val=1.0):
        """
        Initialize the ADCQuantizer module.
        
        Args:
            n_bits (int): Number of bits for quantization.
            min_val (float): Minimum input value for the ADC.
            max_val (float): Maximum input value for the ADC.
        """
        super().__init__()
        self.__name__ = "QuantizeToNBits"

        self.n_bits = n_bits
        self.min_val = min_val
        self.max_val = max_val
        
        # Calculate the number of quantization levels.
        # For n_bits, levels go from 0 to 2^n_bits - 1.
        self.q_levels = 2**n_bits - 1
        
        # Precompute the scale factor.
        # Maps the analog range [min_val, max_val] to [0, q_levels].
        self.scale = self.q_levels / (max_val - min_val)
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass for the ADC quantization.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: The quantized and reconstructed tensor.
        """

        # 1. Clamp the input to the ADC's valid range.
        x_clamped = torch.clamp(x, self.min_val, self.max_val)
        
        # 2. Scale the clamped value to the digital range.
        x_scaled = (x_clamped - self.min_val) * self.scale
        
        # 3. Quantize using a rounding operation with a straight-through estimator.
        x_quantized = self.ste_round(x_scaled)
        
        # 4. Map back to the original analog range.
        x_reconstructed = x_quantized / self.scale + self.min_val
        
        return x_reconstructed

    @staticmethod
    def ste_round(x):
        """
        Rounds the input using a straight-through estimator (STE).
        
        The forward pass uses standard rounding, but the backward pass
        passes the gradient as if the function were the identity.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Rounded tensor with gradients flowing through.
        """
        # The trick: x - x.detach() has the same gradient as x.
        # Adding x.detach().round() applies the rounding in the forward pass.
        return (x - x.detach()) + x.detach().round()
    
class Diagonal(nn.Module):
    def __init__(self, 
                 in_features, 
                 trainable=True, 
                 decomp=1,
                 single_value = False, # The whole diagonal consists of the same, single (trainable) value
                 device=DEFAULT_DEVICE):
        super(Diagonal, self).__init__()
        self.__name__ = "Diagonal"
        self.decomp = decomp
        self.device = device

        if single_value:
            self.S = nn.Parameter(
                torch.ones(1, requires_grad=trainable, device=self.device)
            )
        elif decomp == 1:
            self.S = nn.Parameter(
                torch.ones(in_features, requires_grad=trainable, device=self.device)
            )
        else:
            self.S = nn.Parameter(
                torch.ones(decomp, in_features, requires_grad=trainable, device=self.device)
            )

    # def forward(self, *args):
    #     return self.S.mul(*args)
    def forward(self, x):
        # if isinstance(x, TensorWDM):
        #     # Handle TensorWDM input
        #     ret = TensorWDM(self.S.mul(x))
        #     ret.set_wavelengths(x.wavelengths)
        #     ret.set_power_unit(x.power_unit)
        #     return ret
        # else:
        # Handle regular tensor input
        if self.decomp == 1:
            return self.S.mul(x)  
        else:
            return torch.einsum("bds, ds->bds", x, self.S)  

class Reroute(nn.Module):
    def __init__(
        self,
        n_waveguides,
        routes: list,  # List of routings: [[2,4], .. ] → routes inp2 to out4
        passthrough: bool = True,  # If True, the initial state of the transfer matrix is a diag mat with 1s on the diagonal. If False, its a matrix of zeros
        enforce_permutation: bool = True,
        share_uv="none",
        device=DEFAULT_DEVICE,
    ):
        super(Reroute, self).__init__()
        self.share_uv = share_uv
        self.device = device

        if passthrough:
            self.P = torch.eye(n_waveguides, device=self.device)
        else:
            self.P = torch.zeros(n_waveguides, n_waveguides, device=self.device)

        assert all([len(i) == 2 for i in routes]), (
            "Wrong list of routes provided to Permutation(): each list element must be a pair of values in a list or tuple."
        )

        if enforce_permutation:
            inps = {i[0] for i in routes}
            outs = {i[1] for i in routes}
            assert inps == outs, (
                "The provided set of routings does not satisfy a permutation (1-to-1) routing."
            )

        for pair in routes:
            inp, out = pair
            self.P[inp, inp] = 0
            self.P[out, out] = 0
            self.P[inp, out] = 1
        self.P = self.P.to(torch.complex64)

    def forward(self, x):
        if self.share_uv == "none":
            return torch.einsum("abcd,de->abce", x, self.P)
        else:
            raise NotImplementedError()
            # return torch.matmul(self.P,x)

class Sqrt(nn.Module):
    def __init__(self, support_negatives=False):
        super().__init__()
        self.__name__ = "Sqrt"

        self.support_negatives = support_negatives
    
    def forward(self, x):
        if self.support_negatives:
            return x.abs().sqrt() * x.sign()
        else:
            return x.sqrt()