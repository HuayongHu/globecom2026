from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List
import numpy as np

from .models import Environment, Waveform
from .simulator import RadarSimulator


@dataclass(frozen=True)
class UncertaintySpec:
    """Reproducible sample-based perturbation model.

    The paper uses this object to avoid an ambiguous phrase such as
    "distributional uncertainty". Defaults reproduce the v4.4.x experiments.
    Values describe one-sigma Gaussian/lognormal perturbations, not formal
    distribution-free safety bounds.
    """

    snr_std_db: float = 1.5
    cnr_std_db: float = 2.5
    doppler_log_std: float = 0.18
    range_spread_log_std: float = 0.12
    jnr_std_db: float = 2.0
    max_range_log_std: float = 0.05
    desired_resolution_log_std: float = 0.05

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def to_rows(self) -> List[Dict[str, str]]:
        """Human-readable table rows for paper/report generation."""
        return [
            {"Variable": "SNR", "Perturbation": f"Gaussian, sigma={self.snr_std_db:g} dB"},
            {"Variable": "CNR", "Perturbation": f"Gaussian, sigma={self.cnr_std_db:g} dB"},
            {"Variable": "Doppler spread", "Perturbation": f"Lognormal multiplier, log-sigma={self.doppler_log_std:g}"},
            {"Variable": "Clutter range spread", "Perturbation": f"Lognormal multiplier, log-sigma={self.range_spread_log_std:g}"},
            {"Variable": "JNR", "Perturbation": f"Gaussian, sigma={self.jnr_std_db:g} dB when a jammer is present"},
            {"Variable": "Required range", "Perturbation": f"Lognormal multiplier, log-sigma={self.max_range_log_std:g}"},
            {"Variable": "Desired range resolution", "Perturbation": f"Lognormal multiplier, log-sigma={self.desired_resolution_log_std:g}"},
            {"Variable": "Clutter type", "Perturbation": "Sampled from nearby clutter families"},
        ]


@dataclass
class RobustResult:
    robust_score: float
    mean_score: float
    worst_score: float
    cvar_score: float
    lcb_score: float
    mean_pd: float
    worst_pd: float
    cvar_pd: float
    risk_violation_rate: float
    mean_effective_snr_db: float
    mean_range_resolution_m: float
    mean_ambiguity_score: float
    n_samples: int
    simulator_calls: int

    def to_dict(self) -> Dict[str, float | int]:
        return asdict(self)


class EnvironmentUncertaintySampler:
    def __init__(self, seed: int = 0, spec: UncertaintySpec | None = None):
        self.rng = np.random.default_rng(seed)
        self.spec = spec or UncertaintySpec()

    def sample(self, env: Environment, n: int) -> List[Environment]:
        samples: List[Environment] = []
        clutter_options = [env.clutter_type]
        if env.clutter_type == "gaussian":
            clutter_options += ["ground_weibull"]
        elif env.clutter_type == "sea_k":
            clutter_options += ["gaussian", "ground_weibull"]
        elif env.clutter_type == "urban":
            clutter_options += ["ground_weibull"]
        else:
            clutter_options += ["gaussian"]

        sp = self.spec
        for _ in range(n):
            snr = env.snr_db + self.rng.normal(0.0, sp.snr_std_db)
            cnr = env.clutter_to_noise_db + self.rng.normal(0.0, sp.cnr_std_db)
            dop = max(1.0, env.doppler_spread_hz * np.exp(self.rng.normal(0.0, sp.doppler_log_std)))
            rsp = max(0.2, env.range_spread_m * np.exp(self.rng.normal(0.0, sp.range_spread_log_std)))
            jam = env.jammer_to_noise_db
            if env.jammer_to_noise_db > -40:
                jam = env.jammer_to_noise_db + self.rng.normal(0.0, sp.jnr_std_db)
            clutter = str(self.rng.choice(clutter_options, p=None))
            samples.append(Environment(
                snr_db=float(snr),
                clutter_to_noise_db=float(cnr),
                doppler_spread_hz=float(dop),
                range_spread_m=float(rsp),
                clutter_type=clutter,
                jammer_to_noise_db=float(jam),
                max_range_km=float(max(1.0, env.max_range_km * np.exp(self.rng.normal(0.0, sp.max_range_log_std)))),
                desired_range_resolution_m=float(max(0.2, env.desired_range_resolution_m * np.exp(self.rng.normal(0.0, sp.desired_resolution_log_std)))),
                mission=env.mission,
            ))
        return samples


def lower_tail_mean(values: np.ndarray, alpha: float) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(float(alpha) * len(arr))))
    return float(np.mean(np.sort(arr)[:k]))


def normal_mean_ci95(values: np.ndarray) -> tuple[float, float, float]:
    """Return mean and a simple normal-approximation 95% confidence interval."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, mean, mean
    half = 1.96 * float(np.std(arr, ddof=1)) / np.sqrt(arr.size)
    return mean, mean - half, mean + half


class RobustObjective:
    def __init__(
        self,
        simulator: RadarSimulator | None = None,
        n_samples: int = 32,
        cvar_alpha: float = 0.20,
        risk_pd_threshold: float = 0.80,
        seed: int = 0,
        uncertainty_spec: UncertaintySpec | None = None,
    ):
        self.simulator = simulator or RadarSimulator()
        self.n_samples = int(n_samples)
        self.cvar_alpha = float(cvar_alpha)
        self.risk_pd_threshold = float(risk_pd_threshold)
        self.uncertainty_spec = uncertainty_spec or UncertaintySpec()
        self.sampler = EnvironmentUncertaintySampler(seed, self.uncertainty_spec)
        self.eval_count = 0

    def evaluate(self, waveform: Waveform, env: Environment) -> RobustResult:
        samples = self.sampler.sample(env, self.n_samples)
        scores = []
        pds = []
        snrs = []
        rr = []
        amb = []
        violations = 0
        for e in samples:
            m = self.simulator.evaluate(waveform, e)
            self.eval_count += 1
            scores.append(m.scalar_score)
            pds.append(m.pd)
            snrs.append(m.effective_snr_db)
            rr.append(m.range_resolution_m)
            amb.append(m.ambiguity_score)
            if m.pd < self.risk_pd_threshold or m.constraint_margin < 0:
                violations += 1
        scores_arr = np.asarray(scores, dtype=float)
        pds_arr = np.asarray(pds, dtype=float)
        mean = float(np.mean(scores_arr))
        cvar = lower_tail_mean(scores_arr, self.cvar_alpha)
        worst = float(np.min(scores_arr))
        std = float(np.std(scores_arr, ddof=1)) if len(scores_arr) > 1 else 0.0
        lcb = float(mean - 1.96 * std / np.sqrt(max(len(scores_arr), 1)))

        # The selection score is deliberately scalar-score based. Detection-tail
        # metrics are reported separately so that readers can see where the
        # risk-aware selector helps and where classical optimizers remain strong.
        violation_rate = float(violations / max(len(samples), 1))
        robust_score = 0.35 * mean + 0.35 * cvar + 0.20 * worst + 0.10 * lcb - 0.10 * violation_rate
        return RobustResult(
            robust_score=float(np.clip(robust_score, 0.0, 1.0)),
            mean_score=mean,
            worst_score=worst,
            cvar_score=cvar,
            lcb_score=lcb,
            mean_pd=float(np.mean(pds_arr)),
            worst_pd=float(np.min(pds_arr)),
            cvar_pd=lower_tail_mean(pds_arr, self.cvar_alpha),
            risk_violation_rate=violation_rate,
            mean_effective_snr_db=float(np.mean(snrs)),
            mean_range_resolution_m=float(np.mean(rr)),
            mean_ambiguity_score=float(np.mean(amb)),
            n_samples=len(samples),
            simulator_calls=self.eval_count,
        )

    def scalar(self, waveform: Waveform, env: Environment) -> float:
        return self.evaluate(waveform, env).robust_score
