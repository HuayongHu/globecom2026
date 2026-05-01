
from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import time
import traceback
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .models import Environment, Waveform, WaveformConstraints, MODULATIONS
from .simulator import RadarSimulator
from .robust import RobustObjective
from .rag import RuleBasedDesigner
from .llm_client import LLMWaveformClient
from .v41_plots import setup_style, save


def pkg() -> Path:
    return Path(__file__).resolve().parents[1]


def load_rows(path: str | Path) -> List[dict]:
    p = Path(path)
    if not p.is_absolute():
        p = pkg() / p
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row: dict, key: str, default: float) -> float:
    val = row.get(key, "")
    if val is None or str(val).strip() == "":
        return float(default)
    return float(val)


def s(row: dict, key: str, default: str = "") -> str:
    val = row.get(key, default)
    if val is None:
        return default
    return str(val).strip()


def b(row: dict, key: str, default: bool = False) -> bool:
    val = row.get(key, "")
    if str(val).strip() == "":
        return bool(default)
    return str(val).strip().lower() in {"1", "true", "yes", "y"}


def env(row: dict) -> Environment:
    return Environment(
        snr_db=float(row["snr_db"]),
        clutter_to_noise_db=float(row["clutter_to_noise_db"]),
        doppler_spread_hz=float(row["doppler_spread_hz"]),
        range_spread_m=float(row["range_spread_m"]),
        clutter_type=row["clutter_type"],
        jammer_to_noise_db=float(row["jammer_to_noise_db"]),
        max_range_km=float(row["max_range_km"]),
        desired_range_resolution_m=float(row["desired_range_resolution_m"]),
        mission=row["mission"],
    )


def physical_repair(w: Waveform, e: Environment) -> Waveform:
    return WaveformConstraints().repair(w, e)


def semantic_hard_repair(w: Waveform, row: dict, e: Environment) -> Waveform:
    """Apply only hard semantic constraints.

    v4.2 intentionally does NOT force soft preferences such as preferred_modulation.
    This makes semantic compliance measurable and prevents a rule-only method from
    trivially receiving a perfect score by post-hoc repair.
    """
    c = WaveformConstraints()
    wr = c.repair(w, e)
    bw = float(np.clip(wr.bandwidth_mhz, c.bandwidth_min, min(c.bandwidth_max, f(row, "max_bandwidth_mhz", c.bandwidth_max))))
    bw = max(bw, f(row, "min_bandwidth_mhz", c.bandwidth_min))
    prf = float(np.clip(wr.prf_khz, c.prf_min, min(c.prf_max, f(row, "max_prf_khz", c.prf_max))))

    # Avoid a forbidden PRF interval if specified by coexistence constraints.
    lo = f(row, "forbidden_prf_min_khz", -1.0)
    hi = f(row, "forbidden_prf_max_khz", -1.0)
    if lo >= 0 and hi > lo and lo <= prf <= hi:
        left = max(c.prf_min, lo - 0.5)
        right = min(f(row, "max_prf_khz", c.prf_max), hi + 0.5)
        # Choose the nearest side that remains valid for long-range requirements.
        prf = left if abs(prf - left) <= abs(prf - right) else right

    tau = wr.pulse_width_us
    max_duty = min(c.max_duty_cycle, f(row, "max_duty_cycle", c.max_duty_cycle))
    if prf * tau * 1e-3 > max_duty:
        tau = max(c.pulse_width_min, max_duty / max(prf, 1e-9) * 1e3)

    n = int(np.clip(wr.n_pulses, max(c.n_pulses_min, int(f(row, "min_n_pulses", c.n_pulses_min))), min(c.n_pulses_max, int(f(row, "max_n_pulses", c.n_pulses_max)))))

    mod = wr.modulation if wr.modulation in MODULATIONS else "LFM"
    req = s(row, "required_modulation")
    avoid = s(row, "avoid_modulation")
    if req in MODULATIONS:
        mod = req
    if avoid in MODULATIONS and mod == avoid:
        # Select a robust non-forbidden alternative without forcing the soft preferred_modulation.
        for alt in ["Costas", "LFM", "BPSK", "Barker"]:
            if alt != avoid:
                mod = alt
                break

    return c.repair(Waveform(wr.carrier_freq_ghz, bw, prf, tau, mod, n), e)


def hard_soft_compliance(w: Waveform, row: dict, e: Environment | None = None) -> tuple[float, float, float, dict]:
    """Return hard satisfaction, soft preference score, overall compliance and details."""
    hard: List[float] = []
    soft: List[float] = []
    det: Dict[str, float] = {}

    def add_h(k: str, ok: bool):
        det[k] = 1.0 if ok else 0.0
        hard.append(det[k])

    def add_s(k: str, ok: bool):
        det[k] = 1.0 if ok else 0.0
        soft.append(det[k])

    duty = w.prf_khz * w.pulse_width_us * 1e-3
    max_bw = f(row, "max_bandwidth_mhz", 500.0)
    min_bw = f(row, "min_bandwidth_mhz", 10.0)
    max_prf = f(row, "max_prf_khz", 100.0)
    max_duty = f(row, "max_duty_cycle", 0.20)
    min_p = int(f(row, "min_n_pulses", 8))
    max_p = int(f(row, "max_n_pulses", 128))
    avoid = s(row, "avoid_modulation")
    required = s(row, "required_modulation")
    preferred = s(row, "preferred_modulation")
    forbidden_lo = f(row, "forbidden_prf_min_khz", -1.0)
    forbidden_hi = f(row, "forbidden_prf_max_khz", -1.0)

    add_h("hard_max_bandwidth", w.bandwidth_mhz <= max_bw + 1e-9)
    add_h("hard_min_bandwidth", w.bandwidth_mhz >= min_bw - 1e-9)
    add_h("hard_max_prf", w.prf_khz <= max_prf + 1e-9)
    add_h("hard_max_duty", duty <= max_duty + 1e-9)
    add_h("hard_min_pulses", w.n_pulses >= min_p)
    add_h("hard_max_pulses", w.n_pulses <= max_p)
    if forbidden_lo >= 0 and forbidden_hi > forbidden_lo:
        add_h("hard_forbidden_prf_band", not (forbidden_lo <= w.prf_khz <= forbidden_hi))
    if avoid in MODULATIONS:
        add_h("hard_avoid_modulation", w.modulation != avoid)
    if required in MODULATIONS:
        add_h("hard_required_modulation", w.modulation == required)

    # Soft preferences: these are intentionally not fully repaired for baselines.
    if preferred in MODULATIONS:
        add_s("soft_preferred_modulation", w.modulation == preferred)
    if b(row, "lpi_priority"):
        add_s("soft_lpi_low_duty", duty <= 0.65 * max_duty + 1e-9)
        add_s("soft_lpi_moderate_pulses", w.n_pulses <= max(min_p + 16, 0.75 * max_p))
    if b(row, "anti_jam_priority"):
        add_s("soft_antijam_wideband", w.bandwidth_mhz >= max(min_bw, min(max_bw, 0.65 * max_bw)))
        add_s("soft_antijam_coding", w.modulation in {"Costas", "BPSK"})
    if b(row, "high_resolution_priority") or str(row.get("mission", "")) == "high_resolution":
        add_s("soft_high_resolution_bandwidth", w.bandwidth_mhz >= max(min_bw, 0.75 * max_bw))
    if b(row, "long_range_priority"):
        add_s("soft_long_range_low_prf", w.prf_khz <= 0.70 * max_prf + 1e-9)
        add_s("soft_long_range_pulses", w.n_pulses >= min(max_p, max(min_p, 48)))
    if b(row, "low_power_priority"):
        add_s("soft_low_power_duty", duty <= 0.60 * max_duty + 1e-9)
        add_s("soft_low_power_pulses", w.n_pulses <= max(min_p, 0.70 * max_p))

    hard_score = float(np.mean(hard)) if hard else 1.0
    soft_score = float(np.mean(soft)) if soft else 1.0
    overall = 0.70 * hard_score + 0.30 * soft_score
    det["hard_satisfaction"] = hard_score
    det["soft_preference_score"] = soft_score
    det["semantic_compliance"] = overall
    det["duty_cycle"] = duty
    return hard_score, soft_score, overall, det


def llm_prompt(row: dict, e: Environment, n: int) -> str:
    hard_keys = [
        "max_bandwidth_mhz", "min_bandwidth_mhz", "max_prf_khz", "forbidden_prf_min_khz", "forbidden_prf_max_khz",
        "max_duty_cycle", "min_n_pulses", "max_n_pulses", "required_modulation", "avoid_modulation",
    ]
    soft_keys = [
        "preferred_modulation", "anti_jam_priority", "lpi_priority", "low_power_priority", "high_resolution_priority", "long_range_priority",
    ]
    hard = {k: row.get(k, "") for k in hard_keys if str(row.get(k, "")).strip() != ""}
    soft = {k: row.get(k, "") for k in soft_keys if str(row.get(k, "")).strip() != ""}
    schema = {
        "candidates": [
            {
                "carrier_freq_ghz": 10.0,
                "bandwidth_mhz": 100.0,
                "prf_khz": 10.0,
                "pulse_width_us": 4.0,
                "modulation": "Costas",
                "n_pulses": 64,
                "design_intent": "one of: compliant_tail_risk, lpi, anti_jam, high_resolution, long_range, low_power",
                "rationale": "short reason"
            }
        ]
    }
    return (
        "Current radar environment:\n" + json.dumps(e.to_dict(), indent=2) +
        "\n\nNatural-language operator requirement:\n" + s(row, "natural_language_requirement") +
        "\n\nHard semantic constraints that should be satisfied:\n" + json.dumps(hard, indent=2) +
        "\n\nSoft semantic preferences to optimize when possible:\n" + json.dumps(soft, indent=2) +
        f"\n\nReturn exactly {n} diverse waveform candidates. Cover multiple strategies: compliant_tail_risk, lpi, anti_jam, high_resolution, long_range, low_power. "
        "Output one strict JSON object and no markdown. Schema:\n" + json.dumps(schema, indent=2)
    )


def llm_candidates(row: dict, e: Environment, config, trace_dir: Path, n: int) -> tuple[List[Waveform], dict]:
    if not bool(getattr(config, "USE_API", True)) and bool(getattr(config, "SEMANTIC_ALLOW_RULE_FALLBACK", False)):
        rule = RuleBasedDesigner(WaveformConstraints(), seed=2026)
        return rule.propose(e, n=n), {"fallback_count": 1, "llm_calls": 0, "parsed_candidate_count": 0}
    client = LLMWaveformClient(
        config.MODEL_ID,
        config.API_URL,
        config.API_KEY,
        use_api=bool(getattr(config, "USE_API", True)),
        require_api=not bool(getattr(config, "SEMANTIC_ALLOW_RULE_FALLBACK", False)),
        allow_rule_fallback=bool(getattr(config, "SEMANTIC_ALLOW_RULE_FALLBACK", False)),
        max_tokens=int(getattr(config, "LLM_MAX_TOKENS", 3000)),
        timeout_sec=int(getattr(config, "REQUEST_TIMEOUT_SEC", 180)),
        trace_dir=trace_dir,
        trace_tag="semantic_v42",
        max_retries=int(getattr(config, "API_MAX_RETRIES", 8)),
        parse_retry_max=int(getattr(config, "API_PARSE_RETRY_MAX", 2)),
    )
    payload = {
        "model": config.MODEL_ID,
        "messages": [
            {"role": "system", "content": "You are a cognitive radar waveform designer. Return only valid JSON."},
            {"role": "user", "content": llm_prompt(row, e, n)},
        ],
        "temperature": float(getattr(config, "SEMANTIC_LLM_TEMPERATURE", 0.30)),
        "max_tokens": int(getattr(config, "LLM_MAX_TOKENS", 3000)),
    }
    if bool(getattr(config, "USE_RESPONSE_FORMAT_JSON", True)):
        payload["response_format"] = {"type": "json_object"}
    data, status, raw, attempts = client._post_with_retry(payload)
    client.usage.calls += 1
    u = data.get("usage", {}) if isinstance(data, dict) else {}
    client.usage.prompt_tokens += int(u.get("prompt_tokens", 0) or 0)
    client.usage.completion_tokens += int(u.get("completion_tokens", 0) or 0)
    choice = data["choices"][0]
    finish_reason = str(choice.get("finish_reason", "") or "")
    if finish_reason in {"length", "max_tokens"}:
        client.usage.truncated_or_max_tokens_count += 1
    content = choice["message"].get("content", "") or ""
    waves = client._parse(content)
    client.usage.parsed_candidate_count += len(waves)
    if waves:
        client.usage.parse_success_count += 1
    else:
        client.usage.parse_error_count += 1
    client._write_trace(payload, status, raw, content, None, len(waves), finish_reason, attempts)
    return waves[:n], client.usage.to_dict()


def select_waveform(cands: List[Waveform], row: dict, e: Environment, objective: RobustObjective, *, method: str, compliance_weight: float, tail_weight: float) -> tuple[Waveform, object, float, dict]:
    if not cands:
        raise RuntimeError("No candidates available for semantic stress method")
    best = None
    for w0 in cands:
        # The crucial v4.2 distinction:
        # - rule_no_semantic does not receive semantic repair or semantic-aware selection.
        # - semantic_rule and semantic_llm are allowed to use formalized semantic constraints.
        if method == "rule_no_semantic":
            w = physical_repair(w0, e)
        else:
            w = semantic_hard_repair(w0, row, e)
        rr = objective.evaluate(w, e)
        hard, soft, comp, det = hard_soft_compliance(w, row, e)
        if method == "rule_no_semantic":
            score = rr.robust_score
        else:
            score = rr.robust_score + compliance_weight * comp + tail_weight * (0.5 * rr.cvar_pd + 0.5 * rr.worst_pd) - 0.10 * rr.risk_violation_rate
        if best is None or score > best[0]:
            best = (score, w, rr, comp, det)
    return best[1], best[2], best[3], best[4]


def run_semantic_stress(config) -> pd.DataFrame:
    out = Path(config.SEMANTIC_OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    trace = out / "api_traces"
    trace.mkdir(exist_ok=True)
    rows = load_rows(config.SEMANTIC_SCENARIOS_FILE)
    rule = RuleBasedDesigner(WaveformConstraints(), seed=2026)
    records: List[dict] = []
    methods = ["semantic_llm", "semantic_rule", "rule_no_semantic"]
    for i, row in enumerate(rows):
        e = env(row)
        print(f"[semantic-v4.2] [{i+1}/{len(rows)}] {row['scenario_id']}", flush=True)
        for method in methods:
            t0 = time.perf_counter()
            rob = RobustObjective(RadarSimulator(), n_samples=int(config.SEMANTIC_ROBUST_SAMPLES), seed=2026 + i)
            usage: dict = {}
            status = "ok"
            error_message = ""
            cands: List[Waveform] = []
            try:
                if method == "semantic_llm":
                    cands, usage = llm_candidates(row, e, config, trace, int(config.SEMANTIC_LLM_CANDIDATES))
                    # Add deterministic semantic anchors so the LLM cannot be penalized solely for one malformed design region.
                    if bool(getattr(config, "SEMANTIC_LLM_INCLUDE_RULE_ANCHORS", True)):
                        cands = cands + rule.propose(e, n=max(3, int(config.SEMANTIC_RULE_CANDIDATES) // 3))
                elif method == "semantic_rule":
                    cands = rule.propose(e, n=int(config.SEMANTIC_RULE_CANDIDATES))
                else:
                    cands = rule.propose(e, n=int(config.SEMANTIC_RULE_CANDIDATES))
                w, rr, comp, det = select_waveform(
                    cands,
                    row,
                    e,
                    rob,
                    method=method,
                    compliance_weight=float(config.SEMANTIC_COMPLIANCE_WEIGHT),
                    tail_weight=float(getattr(config, "SEMANTIC_TAIL_WEIGHT", 0.15)),
                )
            except Exception as exc:
                status = "failed"
                error_message = f"{type(exc).__name__}: {exc}"
                w = Waveform(10.0, 100.0, 10.0, 5.0, "LFM", 32)
                rr = rob.evaluate(physical_repair(w, e), e)
                comp, det = 0.0, {"hard_satisfaction": 0.0, "soft_preference_score": 0.0, "semantic_compliance": 0.0}
                usage = dict(usage)
                usage["error_traceback"] = traceback.format_exc()
            rec = {
                "scenario_id": row["scenario_id"],
                "method": method,
                "status": status,
                "error_message": error_message,
                "runtime_sec": time.perf_counter() - t0,
                "candidate_count": len(cands),
                "semantic_viol_free_score": 1.0 - float(rr.risk_violation_rate),
                "semantic_compliance": comp,
                **e.to_dict(),
                **w.to_dict(),
                **rr.to_dict(),
                **det,
                **usage,
            }
            records.append(rec)
            print(
                f"  {method}: robust={rr.robust_score:.3f}, hard={rec.get('hard_satisfaction', 0):.3f}, "
                f"soft={rec.get('soft_preference_score', 0):.3f}, cvar={rr.cvar_pd:.3f}, viol={rr.risk_violation_rate:.3f}",
                flush=True,
            )
            pd.DataFrame(records).to_csv(out / "semantic_stress_results.csv", index=False)
    df = pd.DataFrame(records)
    ok = df[df["status"].eq("ok")].copy()
    summary = ok.groupby("method").agg(
        robust=("robust_score", "mean"),
        semantic_compliance=("semantic_compliance", "mean"),
        hard_satisfaction=("hard_satisfaction", "mean"),
        soft_preference=("soft_preference_score", "mean"),
        cvar_pd=("cvar_pd", "mean"),
        worst_pd=("worst_pd", "mean"),
        violation=("risk_violation_rate", "mean"),
        no_violation=("semantic_viol_free_score", "mean"),
        llm_calls=("llm_calls", "mean"),
        parsed_candidates=("parsed_candidate_count", "mean"),
        parse_success=("parse_success_count", "mean"),
        n=("scenario_id", "count"),
    ).reset_index()
    summary.to_csv(out / "semantic_stress_summary.csv", index=False)
    write_paired_deltas(ok, out)
    plot_semantic_results(out, bool(getattr(config, "SAVE_PDF_FIGURES", True)))
    write_notes(ok, out)
    return df


def write_paired_deltas(df: pd.DataFrame, out: Path):
    rows = []
    metrics = ["robust_score", "semantic_compliance", "hard_satisfaction", "soft_preference_score", "cvar_pd", "worst_pd", "risk_violation_rate"]
    piv = df.pivot_table(index="scenario_id", columns="method", values=metrics, aggfunc="mean")
    for baseline in ["semantic_rule", "rule_no_semantic"]:
        if ("robust_score", "semantic_llm") not in piv.columns or ("robust_score", baseline) not in piv.columns:
            continue
        for metric in metrics:
            pair = piv[[(metric, "semantic_llm"), (metric, baseline)]].dropna()
            if pair.empty:
                continue
            d = pair[(metric, "semantic_llm")] - pair[(metric, baseline)]
            rows.append({
                "baseline": baseline,
                "metric": metric,
                "n_pairs": len(d),
                "mean_delta_llm_minus_baseline": float(d.mean()),
                "median_delta": float(d.median()),
                "win_rate": float((d > 0).mean()) if metric != "risk_violation_rate" else float((d < 0).mean()),
            })
    pd.DataFrame(rows).to_csv(out / "semantic_paired_deltas.csv", index=False)


def plot_semantic_results(out_dir: str | Path, save_pdf: bool = True):
    setup_style()
    out = Path(out_dir)
    df = pd.read_csv(out / "semantic_stress_results.csv")
    df = df[df.get("status", "ok").astype(str).eq("ok")].copy() if "status" in df.columns else df
    s = df.groupby("method").agg(
        robust=("robust_score", "mean"),
        hard=("hard_satisfaction", "mean"),
        soft=("soft_preference_score", "mean"),
        cvar=("cvar_pd", "mean"),
        worst=("worst_pd", "mean"),
        viol=("risk_violation_rate", "mean"),
    ).reset_index()
    order = [m for m in ["semantic_llm", "semantic_rule", "rule_no_semantic"] if m in set(s["method"])]
    s = s.set_index("method").loc[order].reset_index()
    labels = [m.replace("_", " ") for m in s["method"]]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    x = np.arange(len(s))
    width = 0.18
    ax.bar(x - 1.5 * width, s["robust"], width, label="Robust")
    ax.bar(x - 0.5 * width, s["hard"], width, label="Hard sat.")
    ax.bar(x + 0.5 * width, s["soft"], width, label="Soft pref.")
    ax.bar(x + 1.5 * width, s["cvar"], width, label="CVaR Pd")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean metric")
    ax.set_title("v4.2 semantic-constrained stress test")
    ax.legend(frameon=False, ncol=2)
    save(fig, out / "fig_semantic_v42_metrics", save_pdf)

    # Paired deltas against the semantic rule baseline.
    if "semantic_rule" in set(df["method"]) and "semantic_llm" in set(df["method"]):
        p = df.pivot_table(index="scenario_id", columns="method", values=["robust_score", "hard_satisfaction", "soft_preference_score", "cvar_pd", "worst_pd", "risk_violation_rate"], aggfunc="mean")
        vals = []
        names = []
        for metric, name, lower in [
            ("robust_score", "Robust", False),
            ("hard_satisfaction", "Hard sat.", False),
            ("soft_preference_score", "Soft pref.", False),
            ("cvar_pd", "CVaR Pd", False),
            ("worst_pd", "Worst Pd", False),
            ("risk_violation_rate", "Violation", True),
        ]:
            pair = p[[(metric, "semantic_llm"), (metric, "semantic_rule")]].dropna()
            if pair.empty:
                continue
            d = pair[(metric, "semantic_llm")] - pair[(metric, "semantic_rule")]
            vals.append(-float(d.mean()) if lower else float(d.mean()))
            names.append(("-" if lower else "") + name)
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        xx = np.arange(len(vals))
        ax.bar(xx, vals)
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(xx)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel("Mean paired improvement of LLM over semantic rule")
        ax.set_title("LLM semantic stress paired improvements")
        for xi, v in zip(xx, vals):
            ax.text(xi, v + (0.01 if v >= 0 else -0.025), f"{v:+.3f}", ha="center", fontsize=8)
        save(fig, out / "fig_semantic_v42_paired_delta", save_pdf)


def write_notes(df: pd.DataFrame, out: Path):
    lines = ["# v4.2 semantic stress notes", ""]
    if df.empty:
        lines.append("No successful semantic stress rows were found.")
        (out / "semantic_v42_notes.md").write_text("\n".join(lines), encoding="utf-8")
        return
    s = df.groupby("method").agg(
        robust=("robust_score", "mean"),
        hard=("hard_satisfaction", "mean"),
        soft=("soft_preference_score", "mean"),
        cvar=("cvar_pd", "mean"),
        worst=("worst_pd", "mean"),
        viol=("risk_violation_rate", "mean"),
    )
    rounded = s.round(4)
    try:
        table_text = rounded.to_markdown()
    except Exception:
        table_text = rounded.to_string()
    lines.append(table_text)
    lines.append("")
    lines.append("Claim guidance:")
    lines.append("- Use this test to discuss semantic-constrained waveform design, not ordinary numerical optimization.")
    lines.append("- The rule_no_semantic baseline is deliberately blind to semantic constraints; it is evaluated against them post hoc.")
    lines.append("- Report hard satisfaction, soft preference, CVaR/Worst Pd and violation rate separately.")
    (out / "semantic_v42_notes.md").write_text("\n".join(lines), encoding="utf-8")
