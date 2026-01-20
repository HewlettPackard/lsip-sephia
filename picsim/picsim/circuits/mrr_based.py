import torch
from .generic import _GenericPIC2
from matrepr import mdisplay
from ..components.actives import (
    Microrings,
)

class WeightBank_Allpass(_GenericPIC2):
    r"""Pass-through weight bank with MRRs. 
    Created for the compiled version.

    """

    def __init__(
        self,
        n_inputs: int,
        specs,
        mrr_reso_wavelengths: torch.Tensor = torch.tensor([1550,1551,1552], dtype=torch.float32),
        #depth=1, # to be removed later!
        decomp_size=1,
        precompute_tm=True,
        trainable=True,
        device="cuda:1",
    ):
        self.device = device
        self.circuit_name = "WeightBank_Allpass"

        assert n_inputs > 0, "n_inputs must be non-zero and even."

        self.precompute_tm = precompute_tm
        self.trainable = trainable
        self.n_inputs = n_inputs
        self.mrr_reso_wavelengths = mrr_reso_wavelengths.to(torch.float32).to(self.device)
        self.depth = self.mrr_reso_wavelengths.shape[0]
        self.decomp_size = decomp_size
        # For trainable Weight Bank, it's good to have sigmoid clamping. However, if the module is not trainable, sigmoid clamping is likely to cause more issues than benefits.
        # Therefore, it is directly linked to trainability for this architecture. Feel free to override if this is not desired.
        self.apply_sigmoid_clamp = trainable


        self.n_waveguides = n_inputs
        self.ports_in = [inp for inp in range(self.n_inputs)]
        self.ports_out = [out for out in range(self.n_inputs)]

        self.block_layouts = {}
        self.block_layouts["MRR_Full"] = torch.tensor(
            [1 for i in range(self.n_waveguides)],
            dtype=torch.bool,
        )

        self.block_list = [Microrings(layout=self.block_layouts["MRR_Full"],
                                      resonance_wls=torch.ones(self.n_inputs).to(self.device) * self.mrr_reso_wavelengths[i],
                                      precompute_tm = precompute_tm,
                                      decomp_size=self.decomp_size,
                                      apply_sigmoid_clamp=self.apply_sigmoid_clamp,
                                      specs=specs,
                                      label=str(i).zfill(2)+"__MRRs",
                                      trainable=self.trainable,
                                      device=self.device,
                        ) for i in range(self.depth)]
        

        super().__init__(precompute_tm = self.precompute_tm,
                         decomp_size=decomp_size,
                         trainable=self.trainable,
                        )

    def _get_info(self):
        mdisplay(self.mrr_resonances_all, 
                 title="Resonant wavelengths of all MRRs in the circuit (row=inp, col=wavelength)")
        

class MRRModArray_Allpass(WeightBank_Allpass):
    def __init__(
        self,
        specs,
        mrr_reso_wavelengths: torch.Tensor = torch.tensor([1550,1551,1552], dtype=torch.float32),
        #depth=1, # to be removed later!
        decomp_size=1,
        device="cuda:1",
    ):
        super().__init__(
            n_inputs=1,
            specs=specs,
            mrr_reso_wavelengths=mrr_reso_wavelengths,
            decomp_size=decomp_size,
            precompute_tm=False,
            trainable=False,
            device=device,
        )
        self.circuit_name = "MRRModArray_Allpass"
    
    # Override the forward method to handle the special case in this module, having both input data and optical (carrier) inputs
    def forward(self, 
                inp, # Input values (data)
                inp_optical # Carrier optical signal
                ): 
        # This block embeds the batch dimension into the decomp dimension 
        stop_idx = inp.shape[-1]

        #x = inp_optical
        for i_m, module in enumerate(self.model):
            if i_m < stop_idx:
                module.weights = inp[:,i_m].unsqueeze(1)
                # x = module(x, weights=inp[:,i_m].unsqueeze(1))
            else:
                module.weights = torch.zeros_like(inp[:,0]).unsqueeze(1)
        #return x.permute(1,0,2,3)  # [decomp, batch, wl, wg] -> [batch, decomp, wl, wg]
        return self.model(inp_optical).permute(1,0,2,3)  # [decomp, batch, wl, wg] -> [batch, decomp, wl, wg]
