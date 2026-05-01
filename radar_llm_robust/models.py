from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Tuple
import math
import numpy as np

MODULATIONS = ("LFM", "BPSK", "Barker", "Costas")


@dataclass(frozen=True)
class Waveform:
    carrier_freq_ghz: float
    bandwidth_mhz: float
    prf_khz: float
    pulse_width_us: float
    modulation: str = "LFM"
    n_pulses: int = 32

    def to_dict(self) -> Dict[str, float | str | int]:
        return asdict(self)

    def to_vector(self) -> np.ndarray:
        mod_idx = MODULATIONS.index(self.modulation) if self.modulation in MODULATIONS else 0
        return np.array([
            self.carrier_freq_ghz,
            self.bandwidth_mhz,
            self.prf_khz,
            self.pulse_width_us,
            float(self.n_pulses),
            float(mod_idx),
        ], dtype=float)

    @staticmethod
    def from_vector(x: np.ndarray) -> "Waveform":
        x = np.asarray(x, dtype=float)
        mod_idx = int(np.clip(round(x[5]), 0, len(MODULATIONS) - 1))
        return Waveform(
            carrier_freq_ghz=float(x[0]),
            bandwidth_mhz=float(x[1]),
            prf_khz=float(x[2]),
            pulse_width_us=float(x[3]),
            n_pulses=int(np.clip(round(x[4]), 8, 128)),
            modulation=MODULATIONS[mod_idx],
        )


@dataclass(frozen=True)
class Environment:
    snr_db: float
    clutter_to_noise_db: float
    doppler_spread_hz: float
    range_spread_m: float
    clutter_type: str = "gaussian"  # gaussian, sea_k, ground_weibull, urban
    jammer_to_noise_db: float = -80.0
    max_range_km: float = 15.0
    desired_range_resolution_m: float = 3.0
    mission: str = "detection"  # detection, tracking, high_resolution, anti_jamming

    def to_dict(self) -> Dict[str, float | str]:
        return asdict(self)

    def to_feature_vector(self) -> np.ndarray:
        clutter_code = {
            "gaussian": 0.0,
            "sea_k": 1.0,
            "ground_weibull": 2.0,
            "urban": 3.0,
        }.get(self.clutter_type, 0.0)
        mission_code = {
            "detection": 0.0,
            "tracking": 1.0,
            "high_resolution": 2.0,
            "anti_jamming": 3.0,
        }.get(self.mission, 0.0)
        return np.array([
            self.snr_db,
            self.clutter_to_noise_db,
            self.doppler_spread_hz,
            self.range_spread_m,
            clutter_code,
            self.jammer_to_noise_db,
            self.max_range_km,
            self.desired_range_resolution_m,
            mission_code,
        ], dtype=float)

    def describe(self) -> str:
        return (
            f"mission={self.mission}; clutter={self.clutter_type}; "
            f"SNR={self.snr_db:.1f} dB; CNR={self.clutter_to_noise_db:.1f} dB; "
            f"doppler_spread={self.doppler_spread_hz:.1f} Hz; "
            f"range_spread={self.range_spread_m:.1f} m; "
            f"jammer={self.jammer_to_noise_db:.1f} dB; max_range={self.max_range_km:.1f} km; "
            f"desired_range_resolution={self.desired_range_resolution_m:.1f} m"
        )


@dataclass(frozen=True)
class WaveformConstraints:
    fc_min: float = 8.0
    fc_max: float = 12.0
    bandwidth_min: float = 10.0
    bandwidth_max: float = 500.0
    prf_min: float = 1.0
    prf_max: float = 100.0
    pulse_width_min: float = 0.1
    pulse_width_max: float = 50.0
    n_pulses_min: int = 8
    n_pulses_max: int = 128
    max_duty_cycle: float = 0.20
    min_time_bandwidth_product: float = 16.0
    min_unambiguous_range_factor: float = 1.10

    def bounds(self) -> np.ndarray:
        return np.array([
            [self.fc_min, self.fc_max],
            [self.bandwidth_min, self.bandwidth_max],
            [self.prf_min, self.prf_max],
            [self.pulse_width_min, self.pulse_width_max],
            [self.n_pulses_min, self.n_pulses_max],
            [0.0, len(MODULATIONS) - 1.0],
        ], dtype=float)

    def repair(self, w: Waveform, env: Environment | None = None) -> Waveform:
        fc = float(np.clip(w.carrier_freq_ghz, self.fc_min, self.fc_max))
        b = float(np.clip(w.bandwidth_mhz, self.bandwidth_min, self.bandwidth_max))
        prf = float(np.clip(w.prf_khz, self.prf_min, self.prf_max))
        tau = float(np.clip(w.pulse_width_us, self.pulse_width_min, self.pulse_width_max))
        n = int(np.clip(round(w.n_pulses), self.n_pulses_min, self.n_pulses_max))
        mod = w.modulation if w.modulation in MODULATIONS else "LFM"

        if b * tau < self.min_time_bandwidth_product:
            tau = min(self.pulse_width_max, self.min_time_bandwidth_product / max(b, 1e-9))
        duty = prf * tau * 1e-3
        if duty > self.max_duty_cycle:
            tau = max(self.pulse_width_min, self.max_duty_cycle / max(prf, 1e-9) * 1e3)
        if env is not None and env.max_range_km > 0:
            c = 3e8
            max_prf_for_range = c / (2.0 * env.max_range_km * 1e3 * self.min_unambiguous_range_factor) / 1e3
            if max_prf_for_range >= self.prf_min:
                prf = min(prf, max_prf_for_range)
                duty = prf * tau * 1e-3
                if duty > self.max_duty_cycle:
                    tau = max(self.pulse_width_min, self.max_duty_cycle / max(prf, 1e-9) * 1e3)
        return Waveform(fc, b, prf, tau, mod, n)

    def check(self, w: Waveform, env: Environment | None = None) -> Tuple[bool, List[str]]:
        issues: List[str] = []
        if not self.fc_min <= w.carrier_freq_ghz <= self.fc_max:
            issues.append("carrier frequency out of X-band bounds")
        if not self.bandwidth_min <= w.bandwidth_mhz <= self.bandwidth_max:
            issues.append("bandwidth out of bounds")
        if not self.prf_min <= w.prf_khz <= self.prf_max:
            issues.append("PRF out of bounds")
        if not self.pulse_width_min <= w.pulse_width_us <= self.pulse_width_max:
            issues.append("pulse width out of bounds")
        if w.modulation not in MODULATIONS:
            issues.append("unsupported modulation")
        if w.bandwidth_mhz * w.pulse_width_us < self.min_time_bandwidth_product:
            issues.append("time-bandwidth product below minimum")
        if w.prf_khz * w.pulse_width_us * 1e-3 > self.max_duty_cycle:
            issues.append("duty cycle exceeds limit")
        if env is not None and env.max_range_km > 0:
            unamb_km = 3e8 / (2.0 * w.prf_khz * 1e3) / 1e3
            if unamb_km < self.min_unambiguous_range_factor * env.max_range_km:
                issues.append("unambiguous range below mission requirement")
        return len(issues) == 0, issues


@dataclass
class SimulationMetrics:
    pd: float
    pfa: float
    effective_snr_db: float
    range_resolution_m: float
    velocity_resolution_mps: float
    unambiguous_range_km: float
    unambiguous_velocity_mps: float
    ambiguity_score: float
    energy_score: float
    constraint_margin: float
    scalar_score: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
