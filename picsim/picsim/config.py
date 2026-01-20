import torch
import copy
import logging
import os
import random
from .scripts.utils import Config
from .scripts.logging import Logger

# Global configuration variables
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WAVELENGTHS = torch.Tensor(
    [
        1550,
    ]
).to(DEFAULT_DEVICE)  # nm
OPT_POWER_UNIT = "uW"
CURRENT_UNIT = "mA"
LOGGER = Logger(verbose=True, logfile=None, console=True)
print(
    f"picsim: Logger is by default set to verbose (debug) mode. Use picsim.set_logger_verbose(False) to disable this."
)

CONFIGS = None
PLATFORM = Config(logger=LOGGER)

platform_fpath = os.path.join(os.path.dirname(__file__), "platforms", "default.yml")
PLATFORM.reload(platform_fpath)
print(f"picsim: Loaded default material platform configuration from {platform_fpath}")
RANDOM = random.Random(42)  # Global random number generator


# Update functions
def set_default_device(device):
    """Set the default device for all picsim components."""
    global DEFAULT_DEVICE
    DEFAULT_DEVICE = (
        device if isinstance(device, torch.device) else torch.device(device)
    )


def set_config(fpath):
    """Set the configuration from a file."""
    global CONFIGS
    CONFIGS = Config()
    CONFIGS.reload(fpath)
    LOGGER.debug(f"picsim: Loaded default config from {fpath}")


def set_platform(fpath):
    """Set the configuration from a file."""
    global PLATFORM
    PLATFORM.reload(fpath)


def set_global_wavelengths(wavelengths: torch.Tensor):
    # Alias for function above
    global WAVELENGTHS
    WAVELENGTHS = copy.deepcopy(wavelengths)


def set_opt_power_unit(unit):
    """Set the optical power unit."""
    global OPT_POWER_UNIT
    _allowed_units = ["uW", "mW", "W"]
    if unit not in _allowed_units:
        raise ValueError(
            f"Invalid optical power unit: {unit}. Must be one of {_allowed_units}."
        )
    OPT_POWER_UNIT = unit


def set_current_unit(unit):
    """Set the current unit."""
    global CURRENT_UNIT
    _allowed_units = ["uA", "mA", "A"]
    if unit not in _allowed_units:
        raise ValueError(
            f"Invalid current unit: {unit}. Must be one of {_allowed_units}."
        )
    CURRENT_UNIT = unit


def set_logger_verbose(is_verbose=False):
    global LOGGER
    LOGGER.logger.setLevel(logging.DEBUG if is_verbose else logging.INFO)


def set_global_random_seed(seed):
    """Set the global random seed."""
    global RANDOM
    if not isinstance(seed, int):
        raise ValueError(f"Seed must be an integer, got {type(seed)}")
    RANDOM = random.Random(seed)
    LOGGER.debug(f"picsim: Set global random seed to {seed}")


# Read functions
def get_global_configs():
    """Get the CONFIGS configuration object."""
    return CONFIGS


def get_global_platform():
    """Get the PLATFORM configuration object."""
    return PLATFORM


def get_global_device():
    """Get the default CUDA device."""
    return DEFAULT_DEVICE


def get_global_wavelengths():
    """Get the default wavelengths."""
    return WAVELENGTHS


def get_global_opt_power_unit():
    """Get the default optical power unit."""
    return OPT_POWER_UNIT


def get_global_current_unit():
    """Get the default current unit."""
    return CURRENT_UNIT


def get_global_random():
    """Get the global random number generator."""
    return RANDOM


def disable_logging_into_file():
    """Disable logging into file."""
    LOGGER.disable_logging_into_file()


def enable_logging_into_file(logfile="main.log"):
    """Enable logging into file."""
    LOGGER.enable_logging_into_file(logfile=logfile)