import torch
import copy
import pandas as pd
from torch import nn
from matrepr import mprint
from ..config import DEFAULT_DEVICE, get_global_wavelengths, LOGGER, get_global_configs, get_global_wavelengths, get_global_platform
from ..components.actives import (
    Microrings,
)
from ..scripts.utils import  compare_param_dict_with_regular_dict


class _GenericPIC2(nn.Module):
    def __init__(self,
                 precompute_tm = True,
                 decomp_size = 1,
                 trainable = True):
        super().__init__()

        self.circuit_name = None
        self.decomp_size = decomp_size
        self.precompute_tm = precompute_tm
        self.trainable=trainable

        self.check_ports()

        if self.precompute_tm:
            # Builds the tensor with model weights inside this class
            self.build_weights()
            self.enable_caching(True) # Currently seems problematic!
        else:
            # Builds the torch.sequential model
            self.build_sequential()

    def enable_caching(self, enable_caching):
        if self.precompute_tm:
            self.cache_enabled = enable_caching
            # if enable_caching:
            #     self._cached_tm = None # Initializes cache for the transfer matrix
            #     self._cached_weights = None # Initializes cache for the weights (memory overhead of this should be okay for small models)
        else:
            raise AssertionError(
                "The precompute_tm flag is set to False, but an attempt to active enable_caching has been made. Please check the configuration."
            )
    
    def check_ports(self):
        assert self.block_list, "The block_list is empty. Prior to calling super().__init__(), please create a block_list variable with component blocks."
        assert self.ports_in, "The ports_in is empty. Prior to calling super().__init__(), please create a ports_in variable with used port numbers."
        assert self.ports_out, "The ports_out is empty. Prior to calling super().__init__(), please create a ports_out variable with used port numbers."

        if isinstance(self.ports_in, list):
            self.ports_in = torch.tensor(self.ports_in, dtype=torch.long, device=self.device)
        if isinstance(self.ports_out, list):
            self.ports_out = torch.tensor(self.ports_out, dtype=torch.long, device=self.device)

        assert self.ports_out.shape[0] <= self.n_waveguides, (
            f"The number of output ports ({self.ports_out}) cannot be greater than the number of waveguides ({self.n_waveguides})."
        )

    def build_weights(self):
        if self.precompute_tm:
            if self.trainable:
                self.weights = torch.nn.ParameterDict()
            else:
                self.weights = {}

            for block in self.block_list:
                if isinstance(block, Microrings):
                    assert block.label not in self.weights, (
                        f"Label {block.label} already exists in the weights dictionary."
                    )
                    self.weights[block.label] = torch.zeros(
                        self.decomp_size,
                        block.n_mrrs,
                        dtype=torch.float32,
                        requires_grad=self.trainable,
                        device=self.device,
                    )
                else:
                    raise NotImplementedError("The desired component is currently not implemented in picsim.circuit builder.")
        else:
            raise AssertionError(
                "The precompute_tm flag is set to False, but the circuit model is being built with weights. Please check the configuration.")
    
    def build_sequential(self):
        if not self.precompute_tm:
            self.model = torch.nn.Sequential(*self.block_list)
        else:
            raise AssertionError(
                "The precompute_tm flag is set to True, but the circuit model is being built as a Sequential model. Please check the configuration.")

    def build_transform(
        self,
        wls,
    ):
        wls_count = wls.shape[0]

        # M has three dimensions: [decomp, wavelength, n_waveguides]
        M = torch.ones(
            self.decomp_size,
            wls_count,
            self.n_waveguides,
            dtype=torch.complex64,
            device=self.device,
        )

        M = torch.diag_embed(M)
        
        # compute the product of all matrices in the sequence in the *reverse* order of application
        for block in self.block_list[::-1]:
            if isinstance(block, Microrings):
                M = (
                    block.build_transform(wls=wls, 
                                          vals=self.weights[block.label])
                    @ M
                )
            else:
                raise NotImplementedError("The desired component is currently not implemented in this version.")
        
        # For cases where only few output ports are required, we can return reduced matrix. Should reduce memory load. Hopefully this will speed-up the computations as well.
        return M[:, :, self.ports_out, :]

    def _get_tm_for_forward(self, wls):
        """Cache-aware transfer matrix calculation"""
        # !! During training!! always rebuild the TM to maintain gradient flow
        if self.training or not hasattr(self, 'cache_enabled') or not self.cache_enabled:
            return self.build_transform(wls)
        
        # Only use caching during evaluation (no gradients needed)
        if not hasattr(self, '_cached_tm') or self._cached_tm is None:
            self._cached_tm = None
            self._cached_weights = None
            self._cached_wls_hash = None
        
        # Some LLM-proposed wizardry here
        current_wls_hash = hash(wls.cpu().numpy().tobytes())
        
        cache_valid = (
            self._cached_tm is not None and 
            self._cached_wls_hash == current_wls_hash and
            compare_param_dict_with_regular_dict(self.weights, self._cached_weights)
        )
        
        if not cache_valid:
            self._cached_tm = self.build_transform(wls)
            self._cached_wls_hash = current_wls_hash
            
            self._cached_weights = {}
            for key, param in self.weights.items():
                self._cached_weights[key] = param.detach().clone()
        
        return self._cached_tm

    def forward(self, x):

        n_wls = x.shape[-2]
        wls = get_global_wavelengths()[0:n_wls].to(self.device)

        if self.n_waveguides > self.n_inputs:

            x_expanded = torch.zeros(
                *x.shape[:-1],
                self.n_waveguides,
                dtype=torch.complex64,
                device=self.device,
            )
            x_expanded[..., self.ports_in] = x

        
            if self.precompute_tm:
                M = self._get_tm_for_forward(wls)
                return torch.einsum("dwij, bdwj -> bdwi", M, x_expanded)
            else:
                ret = self.model(x_expanded)
                # perform output port remaping
                return ret[..., self.ports_out]
            
        else:
            if self.precompute_tm:
                M = self._get_tm_for_forward(wls)
                return torch.einsum("dwij, bdwj -> bdwi", M, x)
            else:
                return self.model(x)

    def get_info(self):
        print(f"{self.circuit_name} info:")
        print(f"-- n_inputs = {self.n_inputs}")
        print(f"-- (n_waveguides = {self.n_waveguides})")
        print(f"-- depth = {self.depth}")
        print(
            f"-- number of trainable parameters: {sum(p.numel() for p in self.model.parameters())}"
        )

        for key in self.info:
            print(f"-- {key} = {self.info[key]}")

        # If there is a _get_info() function specified locally, it also calls it. If not, it skips it.
        try:
            self._get_info()
        except:
            pass
