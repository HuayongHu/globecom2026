from __future__ import annotations

from pathlib import Path
from typing import List
import csv
import numpy as np

from .models import Environment


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def scenario_record_to_env(row: dict) -> Environment:
    return Environment(
        snr_db=float(row["snr_db"]),
        clutter_to_noise_db=float(row["clutter_to_noise_db"]),
        doppler_spread_hz=float(row["doppler_spread_hz"]),
        range_spread_m=float(row["range_spread_m"]),
        clutter_type=str(row["clutter_type"]),
        jammer_to_noise_db=float(row["jammer_to_noise_db"]),
        max_range_km=float(row["max_range_km"]),
        desired_range_resolution_m=float(row["desired_range_resolution_m"]),
        mission=str(row["mission"]),
    )


def load_scenario_dataset(csv_path: str | Path) -> List[dict]:
    path = Path(csv_path)
    if not path.is_absolute():
        path = _package_root() / path
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = dict(row)
            rec["env"] = scenario_record_to_env(row)
            records.append(rec)
    return records


def bootstrap_random_scenarios(n: int = 80, seed: int = 0) -> List[dict]:
    rng = np.random.default_rng(seed)
    clutter_types = ["gaussian", "sea_k", "ground_weibull", "urban"]
    missions = ["detection", "tracking", "high_resolution", "anti_jamming"]
    records: List[dict] = []
    for i in range(n):
        clutter = clutter_types[i % len(clutter_types)]
        mission = missions[(i * 3) % len(missions)]
        env = Environment(
            snr_db=float(rng.uniform(-5, 20)),
            clutter_to_noise_db=float(rng.uniform(-5, 35)),
            doppler_spread_hz=float(rng.uniform(40, 650)),
            range_spread_m=float(rng.uniform(1, 45)),
            clutter_type=clutter,
            jammer_to_noise_db=float(-80 if rng.random() < 0.65 else rng.uniform(8, 24)),
            max_range_km=float(rng.uniform(8, 70)),
            desired_range_resolution_m=float(rng.uniform(0.4, 8.0)),
            mission=mission,
        )
        records.append({"scenario_id": f"bootstrap_{i:03d}", "family": f"{clutter}_{mission}", "difficulty": "bootstrap", "env": env})
    return records
