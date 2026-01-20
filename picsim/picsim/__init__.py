from .config import (
    DEFAULT_DEVICE, 
    WAVELENGTHS,
    LOGGER,
    PLATFORM,
    get_global_configs,
    get_global_wavelengths,
    get_global_opt_power_unit,
    get_global_current_unit,
    get_global_platform,
    get_global_random,
    set_default_device,
    set_global_wavelengths,
    set_config,
    set_logger_verbose,
    set_global_random_seed,
    disable_logging_into_file,
    enable_logging_into_file,
)

# root_folder = str(Path(os.getcwd()).parents[0])
# script_name = str(os.path.basename(__file__))[:-3]

# Import other modules
from .components import *
from .circuits import *