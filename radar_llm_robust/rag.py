from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np

from .models import Environment, Waveform, WaveformConstraints, MODULATIONS
from .simulator import RadarSimulator
from .robust import RobustObjective


def normalize_env_feature(x: np.ndarray) -> np.ndarray:
    scale = np.array([20.0, 35.0, 600.0, 60.0, 3.0, 80.0, 80.0, 10.0, 3.0], dtype=float)
    shift = np.array([5.0, 10.0, 250.0, 20.0, 1.0, -50.0, 25.0, 3.0, 1.0], dtype=float)
    return (x - shift) / scale


@dataclass
class LibraryEntry:
    env: Environment
    waveform: Waveform
    score: float
    rationale: str

    def to_json(self) -> dict:
        return {
            "env": self.env.to_dict(),
            "waveform": self.waveform.to_dict(),
            "score": float(self.score),
            "rationale": self.rationale,
        }

    @staticmethod
    def from_json(obj: dict) -> "LibraryEntry":
        return LibraryEntry(
            env=Environment(**obj["env"]),
            waveform=Waveform(**obj["waveform"]),
            score=float(obj.get("score", 0.0)),
            rationale=str(obj.get("rationale", "retrieved design")),
        )


class WaveformLibrary:
    def __init__(self, entries: List[LibraryEntry]):
        self.entries = entries
        if entries:
            self.features = np.vstack([normalize_env_feature(e.env.to_feature_vector()) for e in entries])
        else:
            self.features = np.zeros((0, 9), dtype=float)

    @staticmethod
    def load(path: str | Path) -> "WaveformLibrary":
        p = Path(path)
        if not p.exists():
            return WaveformLibrary([])
        entries: List[LibraryEntry] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(LibraryEntry.from_json(json.loads(line)))
        return WaveformLibrary(entries)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for e in self.entries:
                f.write(json.dumps(e.to_json(), ensure_ascii=False) + "\n")

    def retrieve(self, env: Environment, k: int = 8) -> List[LibraryEntry]:
        if len(self.entries) == 0:
            return []
        q = normalize_env_feature(env.to_feature_vector())
        d = np.linalg.norm(self.features - q[None, :], axis=1)
        idx = np.argsort(d)[: min(k, len(d))]
        return [self.entries[int(i)] for i in idx]


class RuleBasedDesigner:
    def __init__(self, constraints: WaveformConstraints | None = None, seed: int = 0):
        self.constraints = constraints or WaveformConstraints()
        self.rng = np.random.default_rng(seed)

    def propose(self, env: Environment, n: int = 8) -> List[Waveform]:
        base = self._base(env)
        waves = [base]
        for _ in range(max(0, n - 1)):
            x = base.to_vector()
            x[0] += self.rng.normal(0, 0.35)
            x[1] *= float(np.exp(self.rng.normal(0, 0.22)))
            x[2] *= float(np.exp(self.rng.normal(0, 0.25)))
            x[3] *= float(np.exp(self.rng.normal(0, 0.25)))
            x[4] += self.rng.integers(-16, 17)
            if self.rng.random() < 0.35:
                x[5] = self.rng.integers(0, len(MODULATIONS))
            waves.append(self.constraints.repair(Waveform.from_vector(x), env))
        return waves

    def _base(self, env: Environment) -> Waveform:
        desired_rr = max(env.desired_range_resolution_m, 0.3)
        b_needed = 3e8 / (2.0 * desired_rr) / 1e6
        b = float(np.clip(1.2 * b_needed, 50.0, 500.0))
        if env.mission == "high_resolution":
            b = max(b, 320.0)
        if env.clutter_to_noise_db > 18 or env.clutter_type in ("sea_k", "urban"):
            prf = 45.0 + 0.10 * env.doppler_spread_hz
        elif env.max_range_km > 40:
            prf = 6.0
        else:
            prf = 18.0 + 0.04 * env.doppler_spread_hz
        prf = float(np.clip(prf, 1.0, 100.0))
        if env.snr_db < 3:
            tau = 18.0
            n = 96
        elif env.snr_db < 10:
            tau = 10.0
            n = 64
        else:
            tau = 4.0
            n = 32
        if env.mission == "tracking":
            n = min(128, int(n * 1.4))
        if env.jammer_to_noise_db > -20:
            mod = "Costas"
            b = max(b, 250.0)
        elif env.clutter_type == "sea_k":
            mod = "Barker"
        elif env.doppler_spread_hz > 300:
            mod = "BPSK"
        else:
            mod = "LFM"
        fc = 10.0 if env.mission != "high_resolution" else 11.0
        return self.constraints.repair(Waveform(fc, b, prf, tau, mod, n), env)


def random_environment(rng: np.random.Generator) -> Environment:
    mission = str(rng.choice(["detection", "tracking", "high_resolution", "anti_jamming"]))
    clutter = str(rng.choice(["gaussian", "sea_k", "ground_weibull", "urban"], p=[0.35, 0.25, 0.25, 0.15]))
    if clutter == "gaussian":
        cnr = rng.uniform(-5, 8)
    elif clutter == "sea_k":
        cnr = rng.uniform(15, 35)
    elif clutter == "urban":
        cnr = rng.uniform(10, 28)
    else:
        cnr = rng.uniform(8, 24)
    snr = rng.uniform(-5, 22)
    dop = rng.uniform(30, 650)
    rsp = rng.uniform(1.0, 60.0)
    max_range = rng.uniform(5, 80)
    desired_rr = rng.uniform(0.5, 8.0) if mission == "high_resolution" else rng.uniform(2.0, 15.0)
    jam = rng.uniform(0, 25) if mission == "anti_jamming" and rng.random() < 0.7 else -80.0
    return Environment(float(snr), float(cnr), float(dop), float(rsp), clutter, float(jam), float(max_range), float(desired_rr), mission)


def build_bootstrap_library(n_envs: int = 200, candidates_per_env: int = 32, seed: int = 0) -> WaveformLibrary:
    rng = np.random.default_rng(seed)
    constraints = WaveformConstraints()
    simulator = RadarSimulator(constraints)
    designer = RuleBasedDesigner(constraints, seed=seed)
    entries: List[LibraryEntry] = []
    for i in range(n_envs):
        env = random_environment(rng)
        candidates = designer.propose(env, n=6)
        bounds = constraints.bounds()
        for _ in range(max(0, candidates_per_env - len(candidates))):
            x = rng.uniform(bounds[:, 0], bounds[:, 1])
            candidates.append(constraints.repair(Waveform.from_vector(x), env))
        best_w = None
        best_score = -1.0
        for w in candidates:
            score = simulator.evaluate(w, env).scalar_score
            if score > best_score:
                best_score = score
                best_w = w
        rationale = "bootstrap expert library: rule-based proposal plus random feasibility search"
        entries.append(LibraryEntry(env=env, waveform=best_w, score=float(best_score), rationale=rationale))
    return WaveformLibrary(entries)
