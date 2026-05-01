from __future__ import annotations

import math
from typing import Dict
import numpy as np

try:
    from scipy.stats import chi2, ncx2
except Exception:  # pragma: no cover
    chi2 = None
    ncx2 = None

from .models import Waveform, Environment, WaveformConstraints, SimulationMetrics


class RadarSimulator:
    """Physics-inspired radar simulator.

    The model is deliberately lightweight so that algorithms can be compared over
    many Monte-Carlo uncertainty samples. It is not a replacement for a radar
    range or RF hardware testbed; it is a reproducible verification layer.
    """

    def __init__(self, constraints: WaveformConstraints | None = None, pfa: float = 1e-6):
        self.c = 3e8
        self.constraints = constraints or WaveformConstraints()
        self.pfa = pfa

    def evaluate(self, waveform: Waveform, env: Environment) -> SimulationMetrics:
        w = self.constraints.repair(waveform, env)
        valid, issues = self.constraints.check(w, env)

        fc_hz = w.carrier_freq_ghz * 1e9
        b_hz = w.bandwidth_mhz * 1e6
        prf_hz = w.prf_khz * 1e3
        tau_s = w.pulse_width_us * 1e-6
        lam = self.c / fc_hz

        range_resolution_m = self.c / (2.0 * b_hz)
        cpi_s = max(w.n_pulses / prf_hz, 1e-9)
        velocity_resolution_mps = lam / (2.0 * cpi_s)
        unambiguous_range_km = self.c / (2.0 * prf_hz) / 1e3
        unambiguous_velocity_mps = lam * prf_hz / 4.0

        tbp = max(b_hz * tau_s, 1.0)
        coherent_gain_db = 10.0 * math.log10(max(w.n_pulses, 1))
        processing_gain_db = 0.55 * 10.0 * math.log10(tbp) + 0.35 * coherent_gain_db
        duty_cycle = w.prf_khz * w.pulse_width_us * 1e-3
        duty_penalty_db = 25.0 * max(0.0, duty_cycle - self.constraints.max_duty_cycle)

        mod_gain_db = {
            "LFM": 0.5,
            "BPSK": 1.5,
            "Barker": 2.0,
            "Costas": 2.5,
        }.get(w.modulation, 0.0)
        sidelobe = {
            "LFM": 0.22,
            "BPSK": 0.16,
            "Barker": 0.10,
            "Costas": 0.08,
        }.get(w.modulation, 0.22)
        doppler_tolerance = {
            "LFM": 0.75,
            "BPSK": 1.00,
            "Barker": 0.90,
            "Costas": 1.15,
        }.get(w.modulation, 0.75)

        clutter_shape_penalty_db = {
            "gaussian": 0.0,
            "sea_k": 3.5,
            "ground_weibull": 2.5,
            "urban": 2.0,
        }.get(env.clutter_type, 0.0)
        clutter_suppression_db = 7.5 * math.log10(max(w.prf_khz, 1.0) / 10.0 + 1.0)
        bandwidth_clutter_gain_db = 2.0 * math.log10(max(w.bandwidth_mhz, 10.0) / 100.0 + 1.0)
        residual_clutter_db = max(
            0.0,
            env.clutter_to_noise_db + clutter_shape_penalty_db - clutter_suppression_db - bandwidth_clutter_gain_db,
        )

        jammer_loss_db = 0.0
        if env.jammer_to_noise_db > -20.0:
            spread_gain_db = 10.0 * math.log10(max(w.bandwidth_mhz / 10.0, 1.0))
            jammer_loss_db = max(0.0, env.jammer_to_noise_db - spread_gain_db - 8.0)

        effective_snr_db = env.snr_db + processing_gain_db + mod_gain_db - residual_clutter_db - jammer_loss_db - duty_penalty_db
        pd = self._pd_from_snr(effective_snr_db)

        desired_rr = max(env.desired_range_resolution_m, 0.1)
        range_score = math.exp(-max(0.0, range_resolution_m - desired_rr) / desired_rr)
        doppler_score = 1.0 / (1.0 + env.doppler_spread_hz / max(prf_hz * doppler_tolerance, 1.0))
        range_match = 1.0 / (1.0 + abs(range_resolution_m - max(env.range_spread_m, 0.1)) / max(env.range_spread_m, 0.1))
        ambiguity_score = float(np.clip(0.45 * range_match + 0.35 * doppler_score + 0.20 * (1.0 - sidelobe), 0.0, 1.0))
        energy_score = float(np.clip(1.0 - duty_cycle / max(self.constraints.max_duty_cycle, 1e-9), 0.0, 1.0))

        range_margin = (unambiguous_range_km / max(env.max_range_km, 1e-9)) - 1.0
        tbp_margin = (w.bandwidth_mhz * w.pulse_width_us / self.constraints.min_time_bandwidth_product) - 1.0
        duty_margin = (self.constraints.max_duty_cycle - duty_cycle) / max(self.constraints.max_duty_cycle, 1e-9)
        constraint_margin = float(min(range_margin, tbp_margin, duty_margin))
        feasibility_score = 1.0 if valid else 0.55

        mission_weights = {
            "detection": (0.42, 0.18, 0.15, 0.15, 0.10),
            "tracking": (0.32, 0.18, 0.25, 0.15, 0.10),
            "high_resolution": (0.30, 0.32, 0.12, 0.16, 0.10),
            "anti_jamming": (0.38, 0.16, 0.16, 0.20, 0.10),
        }.get(env.mission, (0.42, 0.18, 0.15, 0.15, 0.10))
        w_pd, w_rr, w_dop, w_amb, w_energy = mission_weights
        scalar_score = (
            w_pd * pd +
            w_rr * range_score +
            w_dop * doppler_score +
            w_amb * ambiguity_score +
            w_energy * energy_score
        ) * feasibility_score
        if constraint_margin < 0:
            scalar_score *= max(0.0, 1.0 + 0.5 * constraint_margin)
        scalar_score = float(np.clip(scalar_score, 0.0, 1.0))

        return SimulationMetrics(
            pd=float(np.clip(pd, 0.0, 1.0)),
            pfa=self.pfa,
            effective_snr_db=float(effective_snr_db),
            range_resolution_m=float(range_resolution_m),
            velocity_resolution_mps=float(velocity_resolution_mps),
            unambiguous_range_km=float(unambiguous_range_km),
            unambiguous_velocity_mps=float(unambiguous_velocity_mps),
            ambiguity_score=ambiguity_score,
            energy_score=energy_score,
            constraint_margin=constraint_margin,
            scalar_score=scalar_score,
        )

    def _pd_from_snr(self, snr_db: float) -> float:
        snr_lin = max(10.0 ** (snr_db / 10.0), 1e-12)
        if chi2 is not None and ncx2 is not None:
            threshold = chi2.isf(self.pfa, df=2)
            pd = float(ncx2.sf(threshold, df=2, nc=1.35 * snr_lin))
            # Swerling fluctuation and model-mismatch loss keep the benchmark non-saturated.
            cal = float(1.0 / (1.0 + math.exp(-(snr_db - 15.0) / 3.2)))
            return 0.65 * pd + 0.35 * cal
        # Smooth fallback calibrated to give a threshold-like curve.
        return float(1.0 / (1.0 + math.exp(-(snr_db - 15.0) / 3.2)))
