from collections import OrderedDict

import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from torch.types import Device
from ..config import DEFAULT_DEVICE, get_global_wavelengths, LOGGER, get_global_configs, get_global_wavelengths, get_global_platform

from ..components.actives import Photodetectors as PD
from ..components.generic import EOConverter, Sqrt, Diagonal

from ..circuits.mrr_based import WeightBank_Allpass as mrrWB_AP # This is the compiled version!
from ..layers.generic import DimReduction, InstanceNormSimple
from ..scripts.utils import NumToLetter, SequentialPrint

class _GenericOELayer(nn.Module):

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        module_sequence: list = ['Norm', 'EOConv', 'M', 'Pd', 'Sigma'], # Can be a list of strings (module names), or a list of classes (module instances)
        out_features_loc: str = "center",
        decomp=1,
        bias: bool = False,
        mode: str = "phase",
        configs_override = None,  # Can take a dictionary with values that will locally override the global platform + global configs
        device: Device = DEFAULT_DEVICE,
    ):
        super().__init__()
        self.device = device
        self.build_configs(configs_override)

        self.in_features = in_features
        self.out_features = out_features
        self.out_features_loc = out_features_loc
        self.decomp = decomp
        self.bias_flag = bias
        self.mode = mode

        self.check_module_sequence(module_sequence)

        self.mesh_model_built_flag = False
        self.model_built_flag = False

        assert out_features <= in_features, "out_features must be equal or smaller than in_features."
        assert mode in {"phase"}, f"Mode not supported. Expected one from (weight, usv, phase, voltage) but got {mode}."

    def initialize_sigmas_to_unity(self):
        # Initialize all Sigma parameters to 1.0 = no gain, no loss
        assert self.model_built_flag, "Model not built. Run build_model() first."
        for block in self.model.children():
            if isinstance(block, Diagonal):
                for name, param in block.named_parameters():
                    if "Sigma" in name:
                        torch.nn.init.constant_(param, 1.0)
        
    def initialize_weights_rand_uniform(self,
                                        min=0.0, 
                                        max=1.0):
        # Initialize all weights to identity matrix
        assert self.model_built_flag, "Model not built. Run build_model() first."
        for block in self.model.children():
            for name, param in block.named_parameters():
                if "weight" in name:
                    torch.nn.init.uniform_(param, min, max)

    def initialize_weights_to_unity(self):
        # Initialize all weights to identity matrix
        assert self.model_built_flag, "Model not built. Run build_model() first."
        for block in self.model.children():
            for name, param in block.named_parameters():
                if "weight" in name:
                    torch.nn.init.constant_(param, 1.0)

    def check_module_sequence(self, module_sequence):
        # Check if all elements are strings
        module_sequence_str = ""
        for module in module_sequence:
            if isinstance(module, (str)): 
                module_sequence_str += module + "-"
            else: 
                module_sequence_str += module.__name__ + "-"

            self.module_sequence = module_sequence
            self.name = f"OELayer_({module_sequence_str})"
        

    def build_configs(self,
                      configs_override):
        self.configs = get_global_platform()

        if get_global_configs() is not None: 
            self.configs.merge_and_override(get_global_configs(),
                                        logger = LOGGER,
                                        logger_msg = "(from global config)")
        
        if configs_override is not None:
            self.configs.merge_and_override(configs_override,
                                        logger = LOGGER,
                                        logger_msg = "(from local configs_override)") 

    def build_eo_parameters(self, 
                            wavelengths=None, 
                            inp_format="bdws"):
        
        self.wavelengths = get_global_wavelengths() if wavelengths is None else wavelengths
        self.inp_format = inp_format

    def build_model(self,
                    PIC_model):

        _debug_sequential = False # Can be switched locally to print tensor info between Sequential steps. For debugging purposes...
        self._actual_layer_output_size = self.out_features

        module_seq = OrderedDict()

        i = 0
        for module in self.module_sequence:
            if isinstance(module, str):
                if module == 'Norm':
                    module_seq[f"{NumToLetter(i)}_Norm"] = InstanceNormSimple(dim=-1)
                    if _debug_sequential: module_seq[f"{NumToLetter(i)}_out"] = SequentialPrint(
                        f"{NumToLetter(i)}_Norm"
                    )
                    i += 1

                elif module == 'EOConv':
                    module_seq[f"{NumToLetter(i)}_EOConv"] = EOConverter(
                                        inp_format = self.inp_format,
                                        wavelengths = self.wavelengths,
                                        device=self.device
                                        )
                    if _debug_sequential: module_seq[f"{NumToLetter(i)}_out"] = SequentialPrint(
                        f"{NumToLetter(i)}_EOConv"
                    )
                    i += 1

                elif module == 'M':
                    module_seq[f"{NumToLetter(i)}_PIC"] = PIC_model
                    if _debug_sequential: module_seq[f"{NumToLetter(i)}_out"] = SequentialPrint(
                        f"{NumToLetter(i)}_PIC"
                    )
                    i += 1

                    if self.in_features != self.out_features:
                        module_seq[f"{NumToLetter(i)}_DimReduction"] = DimReduction(
                            in_features=self.in_features,
                            out_features=self.out_features,
                            out_features_loc=self.out_features_loc,
                            device=self.device,
                        )
                        i += 1

                elif module == 'Pd':
                    self._actual_layer_output_size = self.out_features
                    module_seq[f"{NumToLetter(i)}_Pd"] = PD(
                        specs=self.configs.specs,
                        use_as_pairwise_balanced=False,
                        clamp_output = getattr(self.configs.specs, "pds_clamp_output", None),
                        device = self.device,

                    )
                    if _debug_sequential: module_seq[f"{NumToLetter(i)}_out"] = SequentialPrint(
                        f"{NumToLetter(i)}_Pd"
                    )
                    i += 1

                elif module == 'Sq':
                    module_seq[f"{NumToLetter(i)}_Sqrt"] = Sqrt()
                    if _debug_sequential: module_seq[f"{NumToLetter(i)}_out"] = SequentialPrint(
                        f"{NumToLetter(i)}_Sqrt"
                    )
                    i += 1

                elif module == 'Sigma':
                    module_seq[f"{NumToLetter(i)}_Sigma"] = Diagonal(
                        in_features=self._actual_layer_output_size, 
                        trainable=True, 
                        decomp=self.decomp,
                        device=self.device
                    )
                    if _debug_sequential: module_seq[f"{NumToLetter(i)}_out"] = SequentialPrint(
                        f"{NumToLetter(i)}_Sigma"
                    )
                    i += 1
            else: # check_module_sequence verifies that module_sequence is a list of classes or strings -> therefore, if its not string, we assume its a class
                module_seq[f"{NumToLetter(i)}_{module.__name__}"] = module
                if _debug_sequential: module_seq[f"{NumToLetter(i)}_out"] = SequentialPrint(
                    f"{NumToLetter(i)}_{module.__name__}"
                )
                i += 1   

        # The addition of balanced PDs means that sometimes, a layer has half of the expected inputs
        # This is because the PDs are used as pairwise balanced PDs, which means that the output of the layer is halved
        # Below is a workaround to make sure that the bias term is correctly scaled (it causes issues in fwd pass if this is not done)
        for module in module_seq.values():
            if isinstance(module, PD):
                if module.use_as_pairwise_balanced:
                    self._actual_layer_output_size = self.out_features // 2

        # Defines bias either as a model parameter (trainable), or as a simple, fixed tensor of zeros (no bias)
        if self.bias_flag:
            self.bias = Parameter(
                torch.zeros(
                    self._actual_layer_output_size,
                    requires_grad=True,
                    device=self.device,
                )
            )
        else:
            self.bias = torch.zeros(
                self._actual_layer_output_size, 
                requires_grad=False, 
                device=self.device
            )
            # self.register_parameter("bias", None)

        self.model = nn.Sequential(module_seq)
        self.model_built_flag = True

    def forward(self, x: Tensor) -> Tensor:
        if not self.model_built_flag:
            raise Warning(f"{self.name} model not built. Run build_model() first.")

        # if self.in_bit < 16:
        #     x = self.input_quantizer(x)

        x = self.model(x)
        return torch.add(x, self.bias)

class OELayer_MRRWeightBank(_GenericOELayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        decomp=1,
        module_sequence=['Norm', 
                    'EOConv', 
                    'M', 
                    'Pd', 
                    #'Sq', 
                    'Sigma'], # Configure this to change the layer composition
        inp_format: str ="bws", # Format of input data, defaults to "bs": 3D Tensor of dims (batch, wavelength, spatial)
        bias: bool = False,
        mode: str = "phase",
        wavelengths = None,
        configs_override = None,
        device: Device = DEFAULT_DEVICE,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            module_sequence=module_sequence,
            out_features_loc="center",
            decomp=decomp,
            bias=bias,
            mode=mode,
            configs_override = configs_override,
            device=device,
        )

        self.build_eo_parameters(
            wavelengths=wavelengths,
            inp_format=inp_format,
        )

        self.build_model(mrrWB_AP(
                n_inputs=in_features,
                specs=self.configs.specs,
                mrr_reso_wavelengths=wavelengths,
                decomp_size=decomp,
                precompute_tm=True,
                device=device,
            )
            )  
        self.initialize_weights_rand_uniform(min=-2, max=2) # Initialize weights to random values = this is prior to the sigmoid transform!