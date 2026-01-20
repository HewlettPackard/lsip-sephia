import traceback, sys
import numpy as np
from scipy.signal import find_peaks
from matrepr import mprint, mdisplay
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
import picsim
from picsim.config import (
    DEFAULT_DEVICE,
    get_global_wavelengths,
    LOGGER,
    get_global_configs,
    get_global_platform,
)
from picsim.components.generic import EOConverter
from picsim.components.actives import Photodetectors
from picsim.circuits.mrr_based import WeightBank_Allpass, MRRModArray_Allpass
from picsim.components.generic import Diagonal
from picsim.layers.linear import OELayer_MRRWeightBank


def dbm_to_watt(power_dbm):
    return 10 ** (power_dbm / 10) / 1000


class OESNN_SEPhIA_MultiTiled2(nn.Module):
    def __init__(
        self,
        comb_class,
        num_tsteps=32,
        layer_widths=(
            (36, 18),
            (18, 8),
        ),  # this is a list of tuples, where each tuple is the input and output neural count of the layer
        tile_counts=(2,1),
        lif_thresholds=(0.55, 0.25),
        lif_betas=(0.95, 0.95),
        reset_mechanism="zero",
        ref_period_enabled=False,
        ref_period_timesteps=1,
        clip_weights=False,
        eoconv_unity_opt_P_per_WDM_channel_dbm=-10,
        eoconv_spatial_dim_broadcast_divide_power=False,
        sephia_specs={
            "mrr_Q": 15000,
            "mrr_ER_db": 20.0,
            "mrr_IL_db": 0.0,
            "mrr_reso_shift_range_pm": -250.0,
        },
        sephia_post_divide_power=False,
        disable_sephia_at_output=True,
        dropout=(False, 0.15),
        use_bpds=True,
        batch_size=16,
        configs_override = None,
        device="cpu",
    ):
        super().__init__()
        self.num_tsteps = num_tsteps
        self.num_layers = len(layer_widths)
        self.layer_widths = layer_widths
        self.tile_counts = tile_counts
        self.lif_thresholds = lif_thresholds

        self._maximum_layer_inp_width = max([x[0] for x in layer_widths])
        self._maximum_layer_outp_width = max([x[1] for x in layer_widths])
        self.used_specs = picsim.get_global_platform()
        assert len(tile_counts) == self.num_layers, f"Tile counts must match number of layers."

        self.reset_mechanism = reset_mechanism
        self.ref_period_enabled = ref_period_enabled
        self.ref_period_timesteps = ref_period_timesteps
        self.clip_weights = clip_weights
        self.sephia_post_divide_power = sephia_post_divide_power
        self.disable_sephia_at_output = disable_sephia_at_output
        self.dropout = dropout
        self.use_bpds = use_bpds
        self.batch_size = batch_size
        self.configs_override = configs_override
        self.device = device

        picsim.set_global_wavelengths(comb_class.peak_locations)

        for specs_key in self.configs_override["specs"].keys():
            self.used_specs.specs[specs_key] = self.configs_override["specs"][specs_key]
            LOGGER.debug(f"Overriding global_specs '{specs_key}' with value: {self.configs_override['specs'][specs_key]}")

        self.eo_c = EOConverter(
            inp_format="bw",
            inp_type="power",  # "power" or "amplitude"
            wavelengths=comb_class.peak_locations[0 : self.layer_widths[0][0]],
            normalize_over_spatial=False,
            unity_opt_P_per_WDM_channel_Watt=dbm_to_watt(
                eoconv_unity_opt_P_per_WDM_channel_dbm
            ),
            spatial_dim_broadcast_n=self.layer_widths[0][1] * 2 // self.tile_counts[0]
            if self.use_bpds
            else self.layer_widths[0][1] // self.tile_counts[0],
            spatial_dim_broadcast_divide_power=eoconv_spatial_dim_broadcast_divide_power,  # Be mindful of this setting when discussing this in the future!
            device=self.device,
        )

        _fcs = {}
        self.lifs = {}
        self.sephia_blocks = {}
        self.sephia_inputs = {}

        for i_l, (n_inp, n_outp) in enumerate(self.layer_widths):

            tiling_factor = self.tile_counts[i_l]

            if self.use_bpds:  # and i_l < (self.num_layers - 1):
                balanced_pds = Photodetectors(
                    specs=self.used_specs.specs,
                    use_as_pairwise_balanced=True,
                    device=device,
                )

                _fcs[f"{i_l}"] = OELayer_MRRWeightBank(
                    in_features=n_outp
                    * 2 // tiling_factor,  # this is quite unique for the WDMBank = the total spatial channels are set by the output
                    out_features=n_outp * 2 // tiling_factor,
                    decomp = tiling_factor,
                    module_sequence=[  #'Norm',  #bw
                        "M",  # bdws
                        balanced_pds,  #'Pd', #bds
                        #'Sq',
                        Diagonal(in_features=n_outp//tiling_factor, 
                                trainable=True, 
                                decomp=tiling_factor,
                                device=self.device),  #'Sigma'
                    ],  # Configure this to change the layer composition
                    wavelengths=comb_class.peak_locations[0:n_inp// tiling_factor],
                    configs_override = self.configs_override,
                    device=device,
                )
            else:
                raise NotImplementedError(
                    "Non-BPDs version of OESNN_SEPhIA_MultiTiled2 is not implemented yet!"
                )

            self.lifs[str(i_l)] = snn.Leaky(
                beta=lif_betas[i_l],
                threshold=self.lif_thresholds[i_l],
                reset_mechanism=self.reset_mechanism,
                reset_delay=True,
            ).to(device)

            if (i_l < (self.num_layers - 1)) or (
                i_l == (self.num_layers - 1) and not self.disable_sephia_at_output
            ):

                self.sephia_blocks[f"{i_l}"] = MRRModArray_Allpass(
                    specs=sephia_specs,
                    mrr_reso_wavelengths=comb_class.peak_locations[0:n_outp],
                    decomp_size=self.batch_size,  # somewhat unusually, decomp is used to process a batch_size in this case
                    device=self.device,
                )

                self.sephia_inputs[f"{i_l}"] = (
                    torch.sqrt(
                        dbm_to_watt(
                            comb_class.get_peak_powers_of_channels(as_tensor=True)
                        )
                        * 1e6
                    )
                    .to(torch.complex64)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .repeat(1, self.batch_size, 1, 1)
                    .to(self.device)
                )  # [BDWS]

        self.fcs = nn.ModuleDict(_fcs)
        del _fcs

    def Refractoriness(
        self,
        inp,
        mem_rec,
        ref_period=1,
        threshold=1.0,
    ):

        history_length = mem_rec.shape[0]

        check_from = max(0, history_length - ref_period)

        lookback_check = (
            ~torch.any(mem_rec[check_from:history_length, ...] > threshold, dim=0)
        ).long()

        return torch.mul(inp, lookback_check), torch.mul(mem_rec[-1], lookback_check)

    def post_SEPhIA_converter(self, x, n_spatial_channels, divide_power=False):
        if divide_power:
            factor = (1 / n_spatial_channels) ** 0.5
        else:
            factor = 1

        x = x.repeat(1, 1, 1, n_spatial_channels) * factor

        return x

    def forward(self, x_in):
        """
        Forward pass with correct implementation of BPTT.

        Args:
            x_in: Input tensor with shape [time, batch, feature]

        Returns:
            tuple: (spikes, membrane_potentials) across all timesteps
        """
        assert len(x_in.shape) == 3, (
            f"Input tensor must be 3D: (time, batch, features), got {x_in.shape=}"
        )

        num_timesteps, batch_size, features = x_in.shape
        assert num_timesteps == self.num_tsteps, (
            f"Number of time steps must match model config: expected {self.num_tsteps}, got {num_timesteps}"
        )


        mems_recs_all = []
        spks_recs_all = []

        for i_l in range(self.num_layers):
            spks_recs_all.append([])

            mems_recs_all.append(
                []
            )  # torch.zeros(self.num_tsteps, batch_size, self.layer_widths[i_l][1], device=self.device, requires_grad=True))
            mems_recs_all[i_l].append(
                torch.zeros(batch_size, self.layer_widths[i_l][1], device=self.device)
            )

        for tstep in range(self.num_tsteps):

            layer_input = self.eo_c(x_in[tstep, ...])

            for i_l in range(self.num_layers):
                if i_l > 0:
                    mult = 1 if not self.use_bpds else 2
                    mult = mult//self.tile_counts[i_l]  # Multiply by tile count, since we have multiple tiles in the layer
                    layer_input = self.post_SEPhIA_converter(
                        layer_output,
                        n_spatial_channels=self.layer_widths[i_l][1]
                        * mult,  # if not self.use_bpds else self.layer_widths[i_l][1]*2,
                        divide_power=self.sephia_post_divide_power,
                    )


                _B, _D, _W, _S = layer_input.shape
                layer_input = layer_input.reshape(_B, 1*self.tile_counts[i_l], _W//self.tile_counts[i_l], _S)

                cur = self.fcs[f"{i_l}"](layer_input)
                cur = cur.reshape(_B ,1,-1)  # Reshape back, now into spatial dim
                cur.squeeze_(1)  # Remove singleton dimension corresponding to decomp
                # Tensor shape: [BS]     

                if self.dropout[0]:
                    cur = F.dropout(cur, p=self.dropout[1], training=self.training)

                if self.ref_period_enabled:
                    mem_history_up_to_tstep = torch.stack(mems_recs_all[i_l])
                    spk, mem = self.lifs[str(i_l)](
                        *self.Refractoriness(
                            inp=cur,
                            mem_rec=mem_history_up_to_tstep,
                            ref_period=self.ref_period_timesteps,
                            threshold=self.lifs[str(i_l)].threshold,
                        )
                    )
                else:
                    spk, mem = self.lifs[str(i_l)](cur, mems_recs_all[i_l][-1])

                mems_recs_all[i_l].append(mem)

                if self.disable_sephia_at_output and i_l == (self.num_layers - 1):
                    layer_output = spk
                    spks_recs_all[i_l].append(
                        layer_output[..., 0 : self.layer_widths[i_l][1]]
                    )
                else:

                    # def hook_fn(grad):


                    layer_output = self.sephia_blocks[
                        f"{i_l}"
                    ].forward(
                        spk, self.sephia_inputs[f"{i_l}"]
                    )[
                        :, :, 0 : self.layer_widths[i_l][1], :
                    ]  # Since the MRRArray considers that input has all WLs from global wavelengths, we have to remove those now manually, otherwise there will unfiltered WDM channels entering next layer
                    spks_recs_all[i_l].append(
                        torch.square(
                            torch.abs(
                                layer_output.detach()[
                                    :,  # B
                                    0,  # D
                                    0 : self.layer_widths[i_l][1],  # W
                                    0,  # S
                                ]
                            )
                        )
                    )  # Saved tensor shape: [BW]

        for i_l in range(self.num_layers):
            spks_recs_all[i_l] = torch.stack(spks_recs_all[i_l])
            mems_recs_all[i_l] = torch.stack(mems_recs_all[i_l])
            mems_recs_all[i_l] = mems_recs_all[i_l][
                1:, ...
            ]  # remove the first timestep, which is all zeros

        return spks_recs_all, mems_recs_all
