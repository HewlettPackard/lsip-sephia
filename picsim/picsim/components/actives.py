import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.nn as nn
from torch.types import Device
from .utils import dB_to_absolute, dB_to_absolute_sqrt
from .generic import BaseComponentModule
from ..config import (
    DEFAULT_DEVICE,
    get_global_wavelengths,
    get_global_current_unit,
    get_global_opt_power_unit,
    LOGGER,
)

class Microrings(BaseComponentModule):
    def __init__(
        self,
        layout: Tensor,  # 1D Tensor that specifies the placement of PSs, and total number of WGs (=len)
        resonance_wls: Tensor,  # 1D Tensor that specifies the resonance wavelengths of the MRRs
        specs: dict,
        precompute_tm: bool = False,  # whether enable the regime of precomputing the transfer matrix. When True, the module is stateless, and the transfer matrix is computed in the build_transform() method. When False, the module is stateful, and the transfer matrix is computed in the forward() method.
        decomp_size: int = 1,
        apply_sigmoid_clamp: bool = True,  # whether to apply sigmoid based clamping to the weights (when creating transform)
        allow_fewer_wls: bool = True,  # whether to allow wavelength dim of input < len(global_wavelengths)
        label="MRR",
        trainable=True,
        device: Device = DEFAULT_DEVICE,
    ):
        super().__init__()
        self.device = device
        self.__name__ = "Microrings"
        self.label = label

        self.precompute_tm = precompute_tm
        self.decomp_size = decomp_size
        self.apply_sigmoid_clamp = apply_sigmoid_clamp
        self.allow_fewer_wls = allow_fewer_wls
        self.trainable = trainable

        self.set_layout(layout)
        self.set_resonances_static(resonance_wls)
        self.set_specs(specs)

        if not self.precompute_tm:
            self.build_weights()

    def set_layout(self, layout):
        assert torch.is_tensor(layout), (
            "Wrong layout argument provided to Microrings(), expecting a Tensor."
        )
        assert len(layout.shape) == 1, (
            f"Wrong shape of layout argument provided to Microrings(), expected 1, got {len(self.layout.shape)}."
        )
        self.layout = layout
        self.n_waveguides = int(layout.shape[0])
        self.n_mrrs = torch.sum(layout).item()
        self.layout_mrr_positions = torch.nonzero(layout).flatten()

    def set_specs(
        self,
        specs,
    ):
        if "mrr_Q" in specs:
            self.MRR_Q_factor = specs["mrr_Q"]
        else:
            self.MRR_Q_factor = 15000.0
            #LOGGER.debug(f"Microrings(label={self.label}) + : Using default MRR_Q_factor of {self.MRR_Q_factor}")

        if "mrr_ER_db" in specs:
            self.MRR_ER_db = specs["mrr_ER_db"]
        else:
            self.MRR_ER_db = 20.0
            #LOGGER.debug(f"Microrings(label={self.label}) + : Using default MRR_ER_db of {self.MRR_Q_factor}")
        assert self.MRR_ER_db > 0, (
            "Invalid MRR_IL_db provided for Microrings(). Must be > 0."
        )

        if "mrr_IL_db" in specs:
            self.MRR_IL_db = specs["mrr_IL_db"]
        else:
            self.MRR_IL_db = 0.0
        assert self.MRR_IL_db >= 0, (
            "Invalid MRR_IL_db provided for Microrings(). Must be >= 0."
        )

        if "mrr_coupling_factor" in specs:
            self.MRR_coupling_factor = torch.tensor(
                specs["mrr_coupling_factor"], requires_grad=False
            ).to(self.device)
        else:
            self.MRR_coupling_factor = torch.tensor(0.85, requires_grad=False).to(
                self.device
            )

        if "mrr_reso_shift_range_pm" in specs:
            self.MRR_reso_shift_range_pm = specs[
                "mrr_reso_shift_range_pm"
            ]  # picometres, per unit of weight
        else:
            self.MRR_reso_shift_range_pm = -250

    def set_resonances_static(self, reso_wls=None):
        if reso_wls is not None:
            assert torch.all(reso_wls > 100) and torch.all(reso_wls < 2000), (
                "Invalid resonance wavelengths provided for Microrings(). Expecting wavelengths in nm (currently bounded between 100 and 2000 nm)."
            )
            assert reso_wls.shape[0] == self.n_mrrs, (
                f"Invalid number of provided MRR static resonance wavelengths (must match number of MRRs). Expected {self.n_mrrs}, got {reso_wls.shape[0]}."
            )
            self.mrr_resonances_static = reso_wls.to(torch.float32).to(self.device)
        else:
            self.mrr_resonances_static = (
                torch.ones(self.n_mrrs, dtype=torch.float32, device=self.device)
                * 1310.0
            )

    def mrr_phaseshift(self, wl, reso_wls):
        wl = wl.unsqueeze(-1).tile((1, self.n_mrrs))
        reso_wls = reso_wls.unsqueeze(0).tile((wl.shape[0], 1))
        wldiff = wl - reso_wls
        return torch.arctan(
            (2 * self.Q * (wldiff) / reso_wls)
            / (1 + torch.pow((2 * self.Q * (wldiff) / reso_wls), 2))
        )

    def build_weights(self):
        if self.precompute_tm is False:
            if self.trainable:
                self.weights = nn.Parameter(
                    torch.zeros(
                        self.decomp_size,  # self.decomp_dim,
                        self.n_mrrs,
                        device=self.device,
                    ),
                    requires_grad=True,
                )
            else:
                self.weights = torch.zeros(
                    self.decomp_size,  # self.decomp_dim,
                    self.n_mrrs,
                    device=self.device,
                    requires_grad=True, # this should help, at least in some of the tested cases, but is probably not strictly required
                )
        else:
            raise AssertionError(
                "Microrings() with precompute_tm=True does not store weights inside the module."
            )

    def build_transform(self, wls=None, vals=None):
        # Expects vals to be a 2D tensor of shape (n_decomp, n_mrrs)
        if vals is None and not self.precompute_tm:
            vals = self.weights

        if self.apply_sigmoid_clamp:
            vals = torch.sigmoid(vals)

        if True:
            # Expects vals to be a 2D tensor of shape [decomp_size, n_mrrs]
            if vals.ndim == 2:
                # vals.unsqueeze_(0)  # Add a decomp dimension if vals is 1D
                mrr_resonances_static = self.mrr_resonances_static.unsqueeze(
                    0
                )  # to ensure compatibility with vals shape: [decomp_size, n_mrrs]

            assert vals.ndim == mrr_resonances_static.ndim, "Mismatch in dimensions of vals and mrr_resonances_static."

        arguments = (
            wls,
            vals,
            self.mrr_resonances_static,
            self.MRR_reso_shift_range_pm,
            self.decomp_size,
            self.n_waveguides,
            self.layout_mrr_positions,
            self.MRR_Q_factor,
            self.MRR_ER_db,
            self.MRR_IL_db,
        )

        t_elems = _mrr_build_transform_real_compiled(*arguments)

        transform = t_elems.to(torch.complex64) # Converting to complex outside of the torchscripted function, as that tends to cause many issues with gradient propagation


        if self.precompute_tm:
            # Returns a tensor of size [decomp, n_wavelengths, n_waveguides, n_waveguides]
            return torch.diag_embed(transform)
        else:
            return transform

    def forward(self, x, weights = None):
        if self.precompute_tm:
            raise NotImplementedError(
                "Microrings() with precompute_tm=True does not implement a forward pass. Obtain the transform matrix using build_transform(), then apply it to the input tensor."
            )
        else:
            wls = get_global_wavelengths()
            if self.allow_fewer_wls:
                if x.shape[-2] <= len(wls):
                    wls = wls[: x.shape[-2]]
                else:
                    raise ValueError(
                        f"Microrings() have received input with {x.shape=}, where {x.shape[-2]=} corresponds to WL. Since allow_fewer_wls=true, {x.shape[-2]=} must be <= {len(wls)}."
                    )
            else:
                if x.shape[-2] != len(wls):
                    raise ValueError(
                        f"Microrings() have received input with {x.shape=}, where {x.shape[-2]=} corresponds to WL. Since allow_fewer_wls=false, {x.shape[-2]=} must be == {len(wls)}."
                    )

            return torch.mul(x, self.build_transform(wls=wls, vals=weights).unsqueeze(0))


def _mrr_build_transform_real(
    wls,
    vals,
    mrr_resonances_static: Tensor,
    MRR_reso_shift_range_pm: float,
    decomp_size: int,
    n_waveguides: int,
    layout_mrr_positions: Tensor,
    MRR_Q_factor: float,
    MRR_ER_db: float,
    MRR_IL_db: float,
) -> Tensor:
    
    # Workaround for getting the device
    device = vals.device

    mrr_resonances_dynamic = torch.add(
        mrr_resonances_static,
        (MRR_reso_shift_range_pm * 1e-3) # 1e-3 factor to go from pm to nm, the standard unit of wavelength
        * vals,  
    )

    transform = torch.ones(
        decomp_size,
        wls.shape[0],
        n_waveguides,
        #dtype=torch.complex64,
        dtype=torch.float32,  # Use float32 for physical calculations
        device=device,
    )

    c = 299792458  # Speed of light in m/s
    wl_m = wls * 1e-9
    res_freqs = c / (mrr_resonances_dynamic * 1e-9)
    linewidths_pow2 = torch.pow(res_freqs / MRR_Q_factor, 2)

    # Reshape for broadcasting
    wl_expanded = wl_m.unsqueeze(-1).unsqueeze(-1)  # [n_wavelengths, 1, 1]
    res_freqs_expanded = res_freqs.unsqueeze(0)  # [1, decomp_size, n_mrrs]
    linewidths_pow2_expanded = linewidths_pow2.unsqueeze(
        0
    )  # [1, decomp_size, n_mrrs]

    # Calculate frequency difference squared
    freq_diff_squared = torch.pow(c / wl_expanded - res_freqs_expanded, 2)

    # Calculate transmission
    mrr_transmissions = (
        -(linewidths_pow2_expanded / (freq_diff_squared + linewidths_pow2_expanded))
        * MRR_ER_db
    )

    # Adjust dimensions to [decomp_size, n_wavelengths, n_mrrs] AND apply insertion loss (in dB)
    t = mrr_transmissions.permute(1, 0, 2) - MRR_IL_db

    t_elems = dB_to_absolute_sqrt(t)
    transform[:, :, layout_mrr_positions] = t_elems

    return transform

# Compiled version of the (real-only) transmissiom matrix generator
_mrr_build_transform_real_compiled = torch.compile(_mrr_build_transform_real)

class Photodetectors(nn.Module):
    def __init__(
        self,
        specs,
        use_as_pairwise_balanced=False,
        clamp_output=True,  # Clamps the output to always have only positive values
        device: Device = DEFAULT_DEVICE,
    ):
        super().__init__()
        self.__name__ = "Photodetectors"
        self.device = device
        self.fresh_init = True  # Used to print out the settings once at the beginning

        # if output_unit is not N
        # assert output_unit in ["uA", "A", "mA"], (
        #     "Invalid output_unit provided for Photodetectors(). Must be 'A' or 'mA', got {output_unit}."
        # )
        self.output_unit = get_global_current_unit()

        self.responsivity = specs["pds_responsivity"]
        self.thermal_noise_enabled = specs["pds_noise_thermal_enabled"]
        self.shot_noise_enabled = specs["pds_noise_shot_enabled"]
        self.fx_pds_floor_cutoff = specs["pds_floor_cutoff"]
        self.Idark = specs["pds_Idark"]
        self.R_load = specs["pds_R_load"]
        self.T = specs["pds_T"]
        self.f_cut = specs["pds_f_cut"]

        if self.thermal_noise_enabled:
            self.build_noise_thermal()

        self.clamp_output = clamp_output

        self.use_as_pairwise_balanced = use_as_pairwise_balanced

        self.calculate_power_scaling_coeffs()

    def calculate_power_scaling_coeffs(self):
        """Calculates the scaling coefficients for the power conversion to current.
        This is done by taking into account the responsivity of the photodetector and the load resistance.
        """
        if self.output_unit == "mA":
            self.current_multiplier = 1e3
            if get_global_opt_power_unit() == "mW":
                self.power_rescaling_factor = 1
            elif get_global_opt_power_unit() == "W":
                self.power_rescaling_factor = 1e3
            elif get_global_opt_power_unit() == "uW":
                self.power_rescaling_factor = 1e-3
        elif self.output_unit == "A":
            self.current_multiplier = 1
            if get_global_opt_power_unit() == "mW":
                self.power_rescaling_factor = 1e-3
            elif get_global_opt_power_unit() == "W":
                self.power_rescaling_factor = 1
            elif get_global_opt_power_unit() == "uW":
                self.power_rescaling_factor = 1e-6
        elif self.output_unit == "uA":
            self.current_multiplier = 1e6
            if get_global_opt_power_unit() == "mW":
                self.power_rescaling_factor = 1e-3
            elif get_global_opt_power_unit() == "W":
                self.power_rescaling_factor = 1e-6
            elif get_global_opt_power_unit() == "uW":
                self.power_rescaling_factor = 1
        else:
            raise ValueError(
                f"Invalid output_unit provided for Photodetectors(). Must be 'A' or 'mA', got {self.output_unit}."
            )

    def build_noise_thermal(
        self,
    ):  # Johnson noise or Nyquist noise is the consequence of thermal fluctuations and is directly associated with thermal radiation
        Kb = 1.38e-23  # Boltzmann constant

        self.noise_thermal_variance = torch.tensor((4 * Kb * self.T * self.f_cut) / (self.R_load))

    def build_noise_shot(self,
                         pd_current):
        Q = 1.6e-19  # Elementary charge
        self.I_tensor = torch.add(
            input=pd_current/self.current_multiplier,  # Convert to A
            other=torch.ones(pd_current.size(), device=pd_current.device),
            alpha=self.Idark,
        )
        noise_shot_variance = 2 * Q * self.f_cut * self.I_tensor  # Schottky formula     
        return noise_shot_variance   

    def Calculate_NEP(
        self,
        incident_opt_power_W = 0, #Calculates NEP under no optical input
        one_over_f_noise_mean_sqrt = 0,
        amplifier_noise_mean_sqrt = 0,
        dynamic_range_db=None, # Optional, for calculating the number of resolvable states
    ):
        Q = 1.6e-19  # Elementary charge
        # mean square value of signal is equal to its variance 

        # Shot noise calculation
        # https://www.fiberoptics4sale.com/blogs/wave-optics/photodetector-noise?srsltid=AfmBOooMSWnZhbPK4km_BtJyZxPPTJrtUE5fdAy_vjzbngok_WK2Mh0c
        
        pds_responsivity = self.responsivity
        pds_f_cut = self.f_cut
        pds_Idark=self.Idark

        all_currents = incident_opt_power_W*pds_responsivity +  pds_Idark
        shot_noise_variance = 2 * Q * pds_f_cut * all_currents
        print(f"Shot noise    (variance) = {shot_noise_variance:.3e} A")
        print(f"Thermal noise (variance) = {self.noise_thermal_variance.item():.3e} A")


        total_spectral_density = np.sqrt(shot_noise_variance + self.noise_thermal_variance.item() + one_over_f_noise_mean_sqrt**2 + amplifier_noise_mean_sqrt**2)
        NEP = total_spectral_density / pds_responsivity
        print(f"NEP = {NEP:.3e} W")
        # print NEP_per_Hz in dBm instead of W
        NEP_dBm = 10 * np.log10(NEP / 1e-3)
        print(f"NEP = {NEP_dBm:.3f} dBm")

        # NEP normalized
        NEP_normalized = NEP / np.sqrt(pds_f_cut)
        print(f"NEP normalized = {NEP_normalized:.3e} W/Hz^0.5")

        # Number of states
        if dynamic_range_db is not None:
            max_power = NEP_dBm + dynamic_range_db
            max_power_W = 10**((max_power) / 10) * 1e-3
            number_of_states = max_power_W/NEP
            print(f"For dynamic_range = {dynamic_range_db}dB:")
            print(f"-> NEP-limited number of resolvable states = {number_of_states}")
            print(f"-> {np.log2(number_of_states):.3f} bits")
        else:
            print("Dynamic range not provided, skipping number of states calculation.")    

        return NEP_dBm

    # @torch.jit.script
    def forward(self, x):

        power_rescaling_factor = self.power_rescaling_factor
        self.fresh_init = False

        # P is proportional to |E|^2
        # converts to power, applies scaling to correct units (like mW -> W), then applies responsivity to obtain currents
        pd_current = (
            torch.square(torch.abs(x))  # .as_tensor()
            * power_rescaling_factor
            * self.responsivity
        )

        # Since we perform incoherent detection, we sum over individual powers of all the wavelength channels of the [batch, decomp, wavelength, spatial]
        pd_current = torch.sum(pd_current, dim=-2)

        if self.fx_pds_floor_cutoff is not None:
            cond = torch.as_tensor(
                (pd_current - self.fx_pds_floor_cutoff * self.current_multiplier) > 0,
                dtype=torch.bool,
            )
            pd_current = torch.where(cond, pd_current, 0)

        if self.thermal_noise_enabled and self.shot_noise_enabled:
            # The noise sources mentioned above are incoherent and the total noise in a system is the square root of the sum of the squares of all the incoherent noise sources.
            cumulative_noise = torch.randn_like(pd_current) * torch.sqrt(
                torch.add(self.noise_thermal_variance, self.build_noise_shot(pd_current))
            )
        elif self.thermal_noise_enabled:
            cumulative_noise = (
                torch.randn_like(pd_current) * torch.sqrt(self.noise_thermal_variance)
            )
        elif self.shot_noise_enabled:
            cumulative_noise = torch.randn_like(pd_current) * torch.sqrt(
                self.build_noise_shot(pd_current)
            )
        else:
            cumulative_noise = torch.zeros_like(pd_current)

        # Adds the noise effect(s) to the current signal
        pd_current = torch.add(
            input=pd_current,
            other=cumulative_noise,
            alpha=self.current_multiplier,
        )

        if self.clamp_output:
            pd_current = torch.clamp(pd_current, min=0)

        # Allows to use adjacent PDs as paired BPDs -> reduces size of final dimension by half!
        if self.use_as_pairwise_balanced:
            assert pd_current.shape[-1] % 2 == 0, (
                "Photodetectors() -> Spatial dimension in the pairwise_balanced regime must be of even number size."
            )
            # Creates the "balanced" mask (alternating 1s and -1s)
            mask = torch.ones(pd_current.shape[-1], device=self.device)
            mask[..., 1::2] = -1
            pd_current = torch.mul(pd_current, mask.view(1, 1, -1))

            # Reshape to group pairs of elements into a new dim
            new_shape = list(pd_current.shape[:-1]) + [pd_current.shape[-1] // 2, 2]

            # Sum along the new last dimension containing the pairs
            pd_current = torch.sum(pd_current.reshape(*new_shape), dim=-1).float()

        return pd_current