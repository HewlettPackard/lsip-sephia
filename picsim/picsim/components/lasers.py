import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from ..config import (
    DEFAULT_DEVICE,
    LOGGER,
)


# Idealized frequency comb spectrum generator
class _CombLaser_Synthetic:
    def __init__(
        self,
        amplitudes_dbm,
        center_wl_nm=1550,
        peak_channel_power_dbm=-10,
        power_range_db=3, # Describes the range of powers (below peak) where the powers will be distributed
        power_floor_dbm=-50,
        spacing_hz=100e9,
        gamma=8e9,
        noise_level=1e-7,  # in milliwatts
        WDM_channel_halfwidth_arglen=45,
        spectral_precision=0.005e-9,  # nm
        device=DEFAULT_DEVICE,
    ):
        # Parameters for the Lorentzian frequency comb
        self.device = device
        self.power_floor = power_floor_dbm
        self.power_range = power_range_db
        self.WDM_channel_halfwidth_arglen = WDM_channel_halfwidth_arglen
        channel_spacing_hz = spacing_hz

        self.absolute_amplitudes_dbm = amplitudes_dbm + peak_channel_power_dbm
        self.total_lines = len(self.absolute_amplitudes_dbm)

        center_frequency = self.wl_freq_conv(center_wl_nm * 1e-9)  # 100 THz
        self.f0s = center_frequency + channel_spacing_hz * (
            np.arange(self.total_lines) - self.total_lines // 2
        )

        self.peak_locations = (
            self.wl_freq_conv(self.f0s[amplitudes_dbm > -self.power_range]) * 1e9
        )  # Convert to nm

        self.peak_amplitudes_dbm = self.absolute_amplitudes_dbm[
            amplitudes_dbm > -self.power_range
        ]

        self.n = len(self.peak_locations)
        self.generate_spectrum(
            center_frequency=center_frequency,
            f0s=self.f0s,
            spacing=channel_spacing_hz,
            gamma=gamma,
            noise_level=noise_level,
            spectral_precision=spectral_precision,
        )

        # Store as a tensor
        self.peak_locations = torch.tensor(
            self.peak_locations,
            requires_grad=False,
            device=self.device,
            dtype=torch.float32,
        )
        # This is just for compatibility with some older code
        self.amplitudes_dbm = self.peak_amplitudes_dbm

    def generate_spectrum(
        self, center_frequency, f0s, spacing, gamma, noise_level, spectral_precision
    ):
        # # Use reduced resolution for spectrum generation
        # wl_range = self.wl_freq_conv(center_frequency)
        nm = np.arange(
            self.wl_freq_conv(center_frequency + (self.total_lines // 2 + 2) * spacing),
            self.wl_freq_conv(center_frequency - (self.total_lines // 2 + 2) * spacing),
            spectral_precision,
        )

        f = self.wl_freq_conv(nm)
        x_np = self.wl_freq_conv(f) * 1e9  # in nm

        self.x = torch.tensor(
            x_np,
            requires_grad=False,
            device=self.device,
            dtype=torch.float32,
        )

        self.x = torch.tensor(
            x_np, requires_grad=False, device=self.device, dtype=torch.float32
        )  # Use float32 instead of default float64

        spectrum = torch.zeros(len(f), device=self.device, dtype=torch.float32)
        for i, f0 in enumerate(f0s):
            # Add directly to spectrum without creating intermediate arrays
            spectrum += self.lorentzian(
                torch.tensor(f, device=self.device), f0, gamma
            ) * (self.absolute_amplitudes_dbm[i] - self.power_floor)

        # Add floor and convert to torch tensor directly
        self._y_ideal = spectrum.cpu().numpy() + self.power_floor

        # Store as numpy for compatibility with other methods
        self.y_np = self._y_ideal.copy()
        self.x_np = self.x.cpu().numpy()

        # Create tensor once
        self.y = torch.tensor(
            self._y_ideal,
            requires_grad=False,
            device=self.device,
            dtype=torch.float32,
        )

        # Add noise directly to torch tensor
        self._add_noise_torch(noise_level)

        self.peak_locations_args = self._find_nearest_indices(self.peak_locations, nm)
        self._create_wdm_channel_width_starting_args()

    def _find_nearest_indices(self, N, X):
        # Find index of minimum difference for each value in N
        return np.argmin(np.abs(N[:, np.newaxis] - X), axis=1)

    def lorentzian(self, f, f0, gamma):
        """
        Calculate the Lorentzian line shape.

        Parameters:
        - f: Frequency array.
        - f0: Center frequency of the Lorentzian.
        - gamma: Full width at half maximum (FWHM) of the Lorentzian.

        Returns:
        - Lorentzian line shape values.
        """
        spectrum = (gamma / 2) / ((f - f0) ** 2 + (gamma / 2) ** 2)
        return spectrum / max(spectrum)

    def generate_lorentzian_comb(
        self, center_frequency, spacing, num_lines, amplitudes, gamma
    ):
        """
        Generate an idealized Lorentzian frequency comb spectrum.

        Parameters:
        - center_frequency: The central frequency of the comb (in Hz).
        - spacing: The spacing between the comb lines (in Hz).
        - num_lines: The number of comb lines to generate.
        - gamma: Full width at half maximum (FWHM) of the Lorentzian lines.
        - amplitude: The amplitude of the comb lines.

        Returns:
        - f: Frequency array.
        - spectrum: Spectrum values.
        """

        nm = np.arange(
            self.wl_freq_conv(center_frequency) - (num_lines // 4 + 3) * 1e-9,
            self.wl_freq_conv(center_frequency) + (num_lines // 4 + 3) * 1e-9,
            0.002e-9,
        )
        f = self.wl_freq_conv(nm)

        spectrum = np.zeros_like(f)
        for i in range(num_lines):
            f0 = center_frequency + spacing * (i - num_lines // 2)
            spectrum += self.lorentzian(f, f0, gamma) * amplitudes[i]
        return f, spectrum

    def wl_freq_conv(self, frequency):
        c = 299792458
        return c / frequency

    def dBm_to_mW(self, dBm):
        return 10 ** (dBm / 10)

    def add_noise(self, noise_level):
        y_mW = self.dBm_to_mW(self._y_ideal)

        y_mW += np.random.normal(0, 1, len(self.y)) * noise_level
        y_mW = np.clip(
            y_mW, a_min=0 + 1e-9, a_max=None
        )  # small shift to ensure non-zero values

        self.y_np = np.clip(10 * np.log10(y_mW), -80, 0)

        self.y = torch.tensor(
            self.y_np,
            requires_grad=False,
            device=self.device,
            dtype=torch.float32,
        )

    def _add_noise_torch(self, noise_level):
        """
        Add noise directly to torch tensor to avoid numpy conversion
        """
        # Convert dBm to mW
        y_mW = torch.pow(10, self.y / 10)

        # Add noise
        noise = torch.randn_like(y_mW) * noise_level
        y_mW.add_(noise)

        # Ensure non-zero values
        y_mW = torch.clamp(y_mW, min=1e-9)

        # Convert back to dBm and update tensors
        self.y = torch.clamp(10 * torch.log10(y_mW), min=-80, max=0)

        # Update numpy version if needed elsewhere
        self.y_np = self.y.cpu().numpy()

    def _create_wdm_channel_width_starting_args(self):
        # Create a list of starting indices for each WDM channel
        starting_indices = []
        for i in range(self.n):
            start_idx = max(
                0, self.peak_locations_args[i] - self.WDM_channel_halfwidth_arglen
            )
            starting_indices.append(start_idx)
        self.WDM_channels_starting_arg = starting_indices

    def plot(self, ax=None, rainbow=False):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 3))
        comb_y_np_mW = np.power(10, self.y_np / 10)
        if rainbow:
            for i in range(len(self.x_np) - 1):
                ax.plot(self.x_np[i : i + 2], 
                        self.y_np[i : i + 2], 
                        color=plt.cm.rainbow(i / len(self.x_np)))            
        else:
            ax.plot(self.x_np, self.y_np, label="Ideal Comb")
        ax.set_xlim((self.x_np[0], self.x_np[-1]))
        print(f"{np.sum(comb_y_np_mW)} mW")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Power (dBm)")

        ax.fill_between(
            self.x_np,
            max(self.peak_amplitudes_dbm) - self.power_range,
            max(self.peak_amplitudes_dbm),
            color="grey",
            alpha=0.15,
            label=f"{self.power_range}dB band",
        )

        ax.scatter(
            self.peak_locations,
            self.peak_amplitudes_dbm + 2,
            marker="v",
            color="red",
            label=f"Peaks within {self.power_range}dB band",
            s=5,
        )

    def get_wavelengths(self):
        return torch.tensor(self.peak_locations)

    def get_peak_powers_of_channels(self,
                                    as_tensor=True):
        if as_tensor:
            return torch.tensor(self.peak_amplitudes_dbm)
        return self.peak_amplitudes_dbm

# Specific case of a comb source, using amplitudes calculated from mean-field laser theory
class CombLaser_MFLT(_CombLaser_Synthetic):
    def __init__(
        self,
        center_wl_nm=1550,
        peak_channel_power_dbm=-10,
        power_range_db=3,
        power_floor_dbm=-50,
        spacing_hz=100e9,
        gamma=8e9,
        noise_level=1e-7,  # in milliwatts
        WDM_channel_halfwidth_arglen=45,
        spectral_precision=0.005e-9,  # nm
        device=DEFAULT_DEVICE,
    ):
        filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "platforms",
            "resources",
            "CombAmplitudes_dB__RayMFL_a=0.00_b=-282.74__[2025-05-30].npy",
        )
        amplitudes_dbm = np.load(filepath)

        super().__init__(
            amplitudes_dbm=amplitudes_dbm,
            center_wl_nm=center_wl_nm,
            peak_channel_power_dbm=peak_channel_power_dbm,
            power_range_db=power_range_db,
            power_floor_dbm=power_floor_dbm,
            spacing_hz=spacing_hz,
            gamma=gamma,
            noise_level=noise_level,
            WDM_channel_halfwidth_arglen=WDM_channel_halfwidth_arglen,
            spectral_precision=spectral_precision,
            device=device,
        )

# Specific case of a comb source, using amplitudes calculated from mean-field laser theory
class CombLaser_Ideal(_CombLaser_Synthetic):
    def __init__(
        self,
        num_lines=36,
        center_wl_nm=1550,
        peak_channel_power_dbm=-10,
        power_range_db=2,
        power_floor_dbm=-70,
        spacing_hz=100e9,
        gamma=8e9,
        noise_level=1e-7,  # in milliwatts
        WDM_channel_halfwidth_arglen=45,
        spectral_precision=0.005e-9,  # nm
        device=DEFAULT_DEVICE,
    ):
        try:
            filepath = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "platforms",
                "resources",
                f"CombAmplitudes_dBm__Ideal_unif{int(power_range_db)}dB-band_n={int(num_lines)}.npy",
            )
            amplitudes_dbm = np.load(filepath)
            LOGGER.debug(
                f"CombLaser_Ideal: Amplitude distribution for num_lines={num_lines},power_range_db={power_range_db} loaded from: {filepath}"
            )
        except FileNotFoundError:
            folderpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                "..",
                "platforms",
                "resources",)
            raise FileNotFoundError(f"There's no file with amplitudes for {num_lines} lines in {power_range_db}dB band. Please make sure it exists in {folderpath}.")

        super().__init__(
            amplitudes_dbm=amplitudes_dbm,
            center_wl_nm=center_wl_nm,
            peak_channel_power_dbm=peak_channel_power_dbm,
            power_range_db=power_range_db,
            power_floor_dbm=power_floor_dbm,
            spacing_hz=spacing_hz,
            gamma=gamma,
            noise_level=noise_level,
            WDM_channel_halfwidth_arglen=WDM_channel_halfwidth_arglen,
            spectral_precision=spectral_precision,
            device=device,
        )