from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import math
import traceback
import time
import pandas as pd

from .models import WaveformConstraints
from .simulator import RadarSimulator
from .robust import RobustObjective, UncertaintySpec
from .rag import WaveformLibrary, build_bootstrap_library
from .llm_client import LLMWaveformClient
from .optimizers import make_designer, run_method, DesignResult
from .scenarios import load_scenario_dataset
from .utils import ensure_dir, write_json, memory_tracker


LLM_METHODS = {"rag_cra", "direct_llm", "rag_cra_no_refine", "rag_cra_no_robust"}
NO_API_METHODS = {"rag_cra_no_api", "rag_only"}


def load_or_build_library(path: Path, size: int, seed: int) -> WaveformLibrary:
    if path.exists():
        lib = WaveformLibrary.load(path)
        if len(lib.entries) > 0:
            return lib
    lib = build_bootstrap_library(n_envs=size, candidates_per_env=24, seed=seed)
    lib.save(path)
    return lib


def make_llm_client_for_method(method: str, config_module, constraints: WaveformConstraints, trace_dir: Path, trace_tag: str, seed: int) -> LLMWaveformClient:
    is_llm_method = method in LLM_METHODS
    is_no_api = method in NO_API_METHODS
    use_api = bool(config_module.USE_API and is_llm_method and not is_no_api)
    require_api = bool(config_module.REQUIRE_API_FOR_LLM_METHODS and is_llm_method and config_module.USE_API)
    allow_fallback = bool(getattr(config_module, "ALLOW_RULE_FALLBACK_WHEN_API_FAILS", False))
    return LLMWaveformClient(
        model=str(config_module.MODEL_ID),
        api_url=str(config_module.API_URL),
        api_key=str(config_module.API_KEY),
        use_api=use_api,
        require_api=require_api,
        allow_rule_fallback=allow_fallback,
        max_tokens=int(config_module.LLM_MAX_TOKENS),
        timeout_sec=int(config_module.REQUEST_TIMEOUT_SEC),
        use_response_format_json=bool(config_module.USE_RESPONSE_FORMAT_JSON),
        constraints=constraints,
        seed=seed,
        trace_dir=trace_dir if bool(config_module.SAVE_API_TRACES) and is_llm_method else None,
        trace_tag=trace_tag,
        max_retries=int(getattr(config_module, "API_MAX_RETRIES", 8)),
        retry_initial_delay=float(getattr(config_module, "API_RETRY_INITIAL_DELAY", 2.0)),
        retry_max_delay=float(getattr(config_module, "API_RETRY_MAX_DELAY", 120.0)),
        retry_exp_base=float(getattr(config_module, "API_RETRY_EXP_BASE", 2.0)),
        retry_jitter=float(getattr(config_module, "API_RETRY_JITTER", 0.25)),
        retry_status_codes=list(getattr(config_module, "API_RETRY_STATUS_CODES", [408, 409, 425, 429, 500, 502, 503, 504])),
        parse_retry_max=int(getattr(config_module, "API_PARSE_RETRY_MAX", 2)),
    )


def make_uncertainty_spec_from_config(config_module) -> UncertaintySpec:
    return UncertaintySpec(
        snr_std_db=float(getattr(config_module, "UNCERTAINTY_SNR_STD_DB", 1.5)),
        cnr_std_db=float(getattr(config_module, "UNCERTAINTY_CNR_STD_DB", 2.5)),
        doppler_log_std=float(getattr(config_module, "UNCERTAINTY_DOPPLER_LOG_STD", 0.18)),
        range_spread_log_std=float(getattr(config_module, "UNCERTAINTY_RANGE_SPREAD_LOG_STD", 0.12)),
        jnr_std_db=float(getattr(config_module, "UNCERTAINTY_JNR_STD_DB", 2.0)),
        max_range_log_std=float(getattr(config_module, "UNCERTAINTY_MAX_RANGE_LOG_STD", 0.05)),
        desired_resolution_log_std=float(getattr(config_module, "UNCERTAINTY_DESIRED_RESOLUTION_LOG_STD", 0.05)),
    )


def final_verify(result: DesignResult, env, config_module, constraints: WaveformConstraints, seed: int, robust_samples: int, final_robust_samples: int | None = None) -> tuple[DesignResult, float, int]:
    simulator = RadarSimulator(constraints=constraints, pfa=config_module.PFA)
    n_samples = int(final_robust_samples or robust_samples)
    objective = RobustObjective(
        simulator,
        n_samples=n_samples,
        cvar_alpha=config_module.CVAR_ALPHA,
        risk_pd_threshold=config_module.RISK_PD_THRESHOLD,
        seed=seed,
        uncertainty_spec=make_uncertainty_spec_from_config(config_module),
    )
    t0 = time.perf_counter()
    final_r = objective.evaluate(result.waveform, env)
    dt = time.perf_counter() - t0
    result.robust = final_r
    return result, dt, objective.eval_count



def _as_float(x) -> float:
    """Convert pandas/scalar values to float; return nan for missing/unparseable."""
    try:
        if pd.isna(x):
            return math.nan
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return math.nan


def _is_blank(x) -> bool:
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    if x is None:
        return True
    return str(x).strip().lower() in {"", "nan", "none", "null"}


def _row_has_complete_final_metrics(row: dict) -> bool:
    # These are the minimum metrics needed for a completed row to be useful in
    # the paper. Legacy v3 rows sometimes have complete metrics but no status.
    required = ["robust_score", "cvar_pd", "worst_pd", "risk_violation_rate"]
    return all(math.isfinite(_as_float(row.get(k))) for k in required)


def normalize_result_row(row: dict) -> dict:
    """Normalize a result row across v3/v3.1/v4 schemas.

    v3 rows did not always contain a populated status field. Treat a blank/nan
    status as successful only when final verification metrics are finite and no
    error fields are present. This prevents RESUME=True and the integrity checker
    from incorrectly rerunning or flagging valid legacy rows.
    """
    row = dict(row)
    status_raw = row.get("status", "")
    err_type = row.get("error_type", "")
    err_msg = row.get("error_message", "")
    try:
        existing_repaired = int(float(row.get("status_repaired_from_legacy", 0) or 0))
    except Exception:
        existing_repaired = 0
    repaired = existing_repaired

    if _is_blank(status_raw):
        if _row_has_complete_final_metrics(row) and _is_blank(err_type) and _is_blank(err_msg):
            row["status"] = "ok"
            row["status_note"] = "legacy_missing_status_repaired"
            repaired = 1
        else:
            row["status"] = "incomplete"
            row["status_note"] = "missing_status_without_complete_metrics"
            repaired = 1
    else:
        status = str(status_raw).strip().lower()
        if status in {"ok", "success", "complete", "completed"}:
            row["status"] = "ok"
        elif status in {"failed", "failure", "error"}:
            row["status"] = "failed"
        elif status in {"incomplete", "partial"}:
            row["status"] = "incomplete"
        else:
            # Unknown status: keep it visible and let integrity_report flag it.
            row["status"] = status
        row.setdefault("status_note", "")

    row["status_repaired_from_legacy"] = int(repaired)
    for k in ["error_type", "error_message", "error_traceback", "last_error"]:
        if k not in row or _is_blank(row.get(k)):
            row[k] = ""
    return row


TELEMETRY_DEFAULTS = {
    "llm_calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "llm_latency_sec": 0.0,
    "parse_success_count": 0,
    "parsed_candidate_count": 0,
    "fallback_count": 0,
    "api_error_count": 0,
    "retry_count": 0,
    "api_attempt_count": 0,
    "parse_error_count": 0,
    "truncated_or_max_tokens_count": 0,
    "last_status_code": 0,
    "last_error": "",
    "llm_enabled": 0,
}


def normalize_results_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = pd.DataFrame([normalize_result_row(r) for r in df.to_dict("records")])
    # Non-LLM baselines such as PSO/GA/DE legitimately do not populate the
    # LLM/API telemetry fields. Later summary aggregation expects these
    # columns to exist, so create zero-valued defaults instead of failing with
    # KeyError. This also keeps legacy completed rows reportable.
    for col, default in TELEMETRY_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
    return out


def _is_ok_row(row: dict) -> bool:
    return normalize_result_row(row).get("status") == "ok"


def _load_existing_suite_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
        return [normalize_result_row(r) for r in df.to_dict("records")]
    except Exception:
        return []


def _write_suite_checkpoint(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    pd.DataFrame([normalize_result_row(r) for r in rows]).to_csv(path, index=False)


def _row_key(row: dict) -> tuple:
    return (str(row.get("suite")), str(row.get("scenario_id")), int(row.get("seed")), str(row.get("method")))


def _nan_metrics() -> dict:
    return {
        "robust_score": math.nan,
        "mean_score": math.nan,
        "cvar_score": math.nan,
        "worst_score": math.nan,
        "lcb_score": math.nan,
        "mean_pd": math.nan,
        "cvar_pd": math.nan,
        "worst_pd": math.nan,
        "risk_violation_rate": math.nan,
    }


def _failure_row(suite_name: str, sid: str, seed: int, method: str, env, family: str, difficulty: str, exc: Exception, llm_client: LLMWaveformClient | None, runtime_sec: float, peak_memory_mb: float) -> dict:
    row = {
        "suite": suite_name,
        "scenario_id": sid,
        "seed": seed,
        "method": method,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:1000],
        "error_traceback": traceback.format_exc()[-4000:],
        "family": family,
        "difficulty": difficulty,
        "runtime_sec": runtime_sec,
        "eval_count": 0,
        "selection_runtime_sec": runtime_sec,
        "selection_eval_count": 0,
        "selection_score_before_final_verification": math.nan,
        "final_verify_runtime_sec": 0.0,
        "final_verify_eval_count": 0,
        "total_runtime_sec": runtime_sec,
        "total_eval_count": 0,
        "peak_memory_mb": peak_memory_mb,
        "flop_proxy": math.nan,
    }
    row.update(env.to_dict())
    row.update(_nan_metrics())
    if llm_client is not None:
        row.update(llm_client.usage.to_dict())
    else:
        row.update({
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "llm_latency_sec": 0.0,
            "parse_success_count": 0,
            "parsed_candidate_count": 0,
            "fallback_count": 0,
            "api_error_count": 0,
            "retry_count": 0,
            "api_attempt_count": 0,
            "parse_error_count": 0,
            "truncated_or_max_tokens_count": 0,
            "last_status_code": 0,
            "last_error": "",
        })
    row["flop_proxy"] = float(row.get("prompt_tokens", 0) or 0) * 2.0e5
    return row


def run_suite(out_dir: Path, suite_name: str, scenario_records: List[dict], methods: List[str], *, library: WaveformLibrary, constraints: WaveformConstraints, config_module, robust_samples: int, local_budget: int, baseline_budget: int, seed: int, final_robust_samples: int | None = None) -> pd.DataFrame:
    suite_csv = out_dir / f"{suite_name}_results.csv"
    rows: List[dict] = []
    histories: Dict[str, list] = {}

    existing_rows = _load_existing_suite_rows(suite_csv) if bool(getattr(config_module, "RESUME", True)) else []
    completed = {_row_key(r) for r in existing_rows if _is_ok_row(r)}
    rows.extend(existing_rows)

    hist_path = out_dir / f"{suite_name}_histories.json"
    if hist_path.exists() and bool(getattr(config_module, "RESUME", True)):
        try:
            import json
            histories.update(json.loads(hist_path.read_text(encoding="utf-8")))
        except Exception:
            pass

    trace_dir = ensure_dir(out_dir / "api_traces" / suite_name)
    for si, rec in enumerate(scenario_records):
        env = rec["env"]
        sid = rec["scenario_id"]
        family = rec.get("family", sid.split("_")[0])
        difficulty = rec.get("difficulty", "unknown")
        print(f"[{suite_name}] [{si+1}/{len(scenario_records)}] {sid}")
        for mi, method in enumerate(methods):
            key = (suite_name, sid, seed, method)
            if key in completed:
                print(f"  method={method}  [resume: already completed]")
                continue

            print(f"  method={method}")
            t_method0 = time.perf_counter()
            llm_client = None
            mem_peak = 0.0
            try:
                simulator = RadarSimulator(constraints=constraints, pfa=config_module.PFA)
                selection_objective = RobustObjective(simulator, n_samples=robust_samples, cvar_alpha=config_module.CVAR_ALPHA, risk_pd_threshold=config_module.RISK_PD_THRESHOLD, seed=seed + si * 100 + mi, uncertainty_spec=make_uncertainty_spec_from_config(config_module))
                trace_tag = f"{suite_name}_{sid}_seed{seed}_{method}".replace("/", "_")
                llm_client = make_llm_client_for_method(method, config_module, constraints, trace_dir, trace_tag, seed + si * 1000 + mi)
                designer = make_designer(method, library, constraints, llm_client=llm_client, seed=seed + 10 * mi)
                with memory_tracker() as mem:
                    result = run_method(method, designer, env, selection_objective, budget=baseline_budget, local_budget=local_budget, llm_candidates=int(config_module.LLM_CANDIDATES))
                mem_peak = mem.peak_mb

                selection_runtime = result.runtime_sec
                selection_eval_count = result.eval_count
                selection_score = float(result.robust.robust_score)
                result, verify_runtime, verify_eval_count = final_verify(result, env, config_module, constraints, seed + 99991 + si * 100 + mi, robust_samples, final_robust_samples)
                row = result.row(sid, env)
                row.update({
                    "status": "ok",
                    "error_type": "",
                    "error_message": "",
                    "error_traceback": "",
                    "suite": suite_name,
                    "seed": seed,
                    "family": family,
                    "difficulty": difficulty,
                    "peak_memory_mb": mem_peak,
                    "selection_runtime_sec": selection_runtime,
                    "selection_eval_count": selection_eval_count,
                    "selection_score_before_final_verification": selection_score,
                    "final_verify_runtime_sec": verify_runtime,
                    "final_verify_eval_count": verify_eval_count,
                    "selection_robust_samples": robust_samples,
                    "final_robust_samples": int(final_robust_samples or robust_samples),
                    "total_runtime_sec": selection_runtime + verify_runtime,
                    "total_eval_count": selection_eval_count + verify_eval_count,
                })
                row["flop_proxy"] = float(row["total_eval_count"]) * 2.5e4 + float(row.get("prompt_tokens", 0) or 0) * 2.0e5
                histories[f"{suite_name}/{sid}/seed{seed}/{method}"] = result.history
                print(f"    robust={row['robust_score']:.4f}, cvar_pd={row['cvar_pd']:.4f}, llm_calls={row.get('llm_calls',0)}, parsed={row.get('parsed_candidate_count',0)}, retries={row.get('retry_count',0)}, fallback={row.get('fallback_count',0)}")
            except Exception as exc:
                runtime_sec = time.perf_counter() - t_method0
                row = _failure_row(suite_name, sid, seed, method, env, family, difficulty, exc, llm_client, runtime_sec, mem_peak)
                print(f"    FAILED after {runtime_sec:.1f}s: {type(exc).__name__}: {str(exc)[:220]}")
                if bool(getattr(config_module, "STOP_ON_METHOD_ERROR", False)):
                    rows.append(row)
                    _write_suite_checkpoint(suite_csv, rows)
                    write_json(hist_path, histories)
                    raise

            # Replace any older failed row for the same key, then append the newest attempt.
            rows = [r for r in rows if _row_key(r) != key]
            rows.append(row)
            if _is_ok_row(row):
                completed.add(key)

            if bool(getattr(config_module, "SAVE_PARTIAL_EVERY_ROW", True)):
                _write_suite_checkpoint(suite_csv, rows)
                write_json(hist_path, histories)

    df = pd.DataFrame(rows)
    df.to_csv(suite_csv, index=False)
    write_json(hist_path, histories)
    return df



def _safe_int_value(x) -> int:
    try:
        if pd.isna(x):
            return 0
        return int(float(x))
    except Exception:
        return 0


def integrity_report(df: pd.DataFrame, config_module, out_dir: Path) -> None:
    df = normalize_results_df(df)
    records = []
    for _, row in df.iterrows():
        method = row["method"]
        status = str(row.get("status", "ok")).strip().lower()
        requires_llm = bool(config_module.USE_API and method in LLM_METHODS)
        is_no_api = method in NO_API_METHODS or method not in LLM_METHODS
        ok = True
        reasons = []
        warnings = []

        if status != "ok":
            ok = False
            reasons.append(f"status={status}; {row.get('error_type','')}: {str(row.get('error_message',''))[:160]}")

        if requires_llm and status == "ok":
            if _safe_int_value(row.get("llm_enabled", 0)) != 1:
                ok = False; reasons.append("llm_enabled is not 1")
            if _safe_int_value(row.get("llm_calls", 0)) < 1:
                ok = False; reasons.append("llm_calls < 1")
            if _safe_int_value(row.get("parsed_candidate_count", 0)) < 1:
                ok = False; reasons.append("no parsed LLM candidates")
            if _safe_int_value(row.get("fallback_count", 0)) != 0:
                ok = False; reasons.append("fallback occurred in API-required LLM method")
            # v3 legacy rows may not have api_attempt_count, so missing telemetry is a warning, not a failure.
            if _safe_int_value(row.get("api_attempt_count", 0)) == 0 and _safe_int_value(row.get("llm_calls", 0)) > 0:
                warnings.append("api_attempt_count missing in legacy completed row")

        if is_no_api and status == "ok":
            if _safe_int_value(row.get("llm_calls", 0)) != 0:
                ok = False; reasons.append("non-LLM/no-api method has llm_calls")
            if method == "rag_cra_no_api" and _safe_int_value(row.get("fallback_count", 0)) > 0:
                warnings.append("legacy no-api row used rule fallback via LLM client; v4 disables this")

        records.append({
            "suite": row["suite"],
            "scenario_id": row["scenario_id"],
            "seed": row["seed"],
            "method": method,
            "status": status,
            "status_repaired_from_legacy": _safe_int_value(row.get("status_repaired_from_legacy", 0)),
            "requires_llm": requires_llm,
            "ok": ok,
            "telemetry_warning": "; ".join(warnings),
            "retry_count": _safe_int_value(row.get("retry_count", 0)),
            "api_attempt_count": _safe_int_value(row.get("api_attempt_count", 0)),
            "last_status_code": _safe_int_value(row.get("last_status_code", 0)),
            "reason": "; ".join(reasons),
        })
    rep = pd.DataFrame(records)
    rep.to_csv(out_dir / "api_integrity_report.csv", index=False)
    summary = {
        "rows": int(len(rep)),
        "failed_rows": int((~rep["ok"]).sum()),
        "true_failed_or_incomplete_rows": int((rep["status"] != "ok").sum()),
        "schema_repaired_legacy_rows": int(rep["status_repaired_from_legacy"].sum()),
        "llm_required_rows": int(rep["requires_llm"].sum()),
        "transient_retry_rows": int((rep["retry_count"] > 0).sum()),
        "telemetry_warning_rows": int((rep["telemetry_warning"].astype(str).str.len() > 0).sum()),
        "all_ok": bool(rep["ok"].all()),
    }
    write_json(out_dir / "api_integrity_summary.json", summary)
    if not rep["ok"].all():
        failed = rep[~rep["ok"]].head(10)
        print("\nAPI integrity check found incomplete/failed rows. First failed rows:")
        print(failed.to_string(index=False))
        print("Rerun the same command with RESUME=True after fixing the underlying issue.")
        if bool(getattr(config_module, "RAISE_ON_INTEGRITY_FAILURE", False)):
            raise RuntimeError("API integrity check failed. See api_integrity_report.csv.")
    elif summary["telemetry_warning_rows"] > 0:
        print(f"\nAPI integrity check passed, with {summary['telemetry_warning_rows']} legacy telemetry/methodology warnings.")
        print("See api_integrity_report.csv and diagnostic_findings.md before using legacy results in the paper.")


def _ok_only(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_results_df(df)
    if "status" not in df.columns:
        return df
    return df[df["status"] == "ok"].copy()


def run_paper_experiment(config_module) -> pd.DataFrame:
    out_dir = ensure_dir(config_module.OUTPUT_DIR)
    constraints = WaveformConstraints()

    if getattr(config_module, "FAST_MODE", False):
        seeds = list(config_module.RANDOM_SEEDS[:1])
        robust_main = min(8, config_module.ROBUST_SAMPLES_MAIN)
        robust_ood = min(12, config_module.ROBUST_SAMPLES_OOD)
        local_budget = min(32, config_module.LOCAL_BUDGET)
        baseline_budget = min(48, config_module.BASELINE_BUDGET)
        library_size = min(80, config_module.LIBRARY_BUILD_SIZE)
    else:
        seeds = list(config_module.RANDOM_SEEDS)
        robust_main = int(config_module.ROBUST_SAMPLES_MAIN)
        robust_ood = int(config_module.ROBUST_SAMPLES_OOD)
        local_budget = int(config_module.LOCAL_BUDGET)
        baseline_budget = int(config_module.BASELINE_BUDGET)
        library_size = int(config_module.LIBRARY_BUILD_SIZE)

    def _final_samples(name: str, fallback: int) -> int:
        value = getattr(config_module, name, None)
        return int(fallback if value is None else value)

    final_main = _final_samples("FINAL_VERIFY_SAMPLES_MAIN", robust_main)
    final_ood = _final_samples("FINAL_VERIFY_SAMPLES_OOD", robust_ood)
    final_ablation = _final_samples("FINAL_VERIFY_SAMPLES_ABLATION", robust_main)

    nominal = load_scenario_dataset(config_module.NOMINAL_SCENARIOS_FILE)
    ood = load_scenario_dataset(config_module.OOD_SCENARIOS_FILE)
    _ = load_scenario_dataset(config_module.LIBRARY_SCENARIOS_FILE)

    library = load_or_build_library(out_dir / "waveform_library.jsonl", size=library_size, seed=seeds[0])
    all_frames = []
    for seed in seeds:
        seed_dir = ensure_dir(out_dir / f"seed_{seed}")
        all_frames.append(run_suite(seed_dir, "main_nominal", nominal, list(config_module.MAIN_METHODS), library=library, constraints=constraints, config_module=config_module, robust_samples=robust_main, final_robust_samples=final_main, local_budget=local_budget, baseline_budget=baseline_budget, seed=seed))
        all_frames.append(run_suite(seed_dir, "ood_stress", ood, list(config_module.MAIN_METHODS), library=library, constraints=constraints, config_module=config_module, robust_samples=robust_ood, final_robust_samples=final_ood, local_budget=local_budget, baseline_budget=baseline_budget, seed=seed))
        all_frames.append(run_suite(seed_dir, "ablation", nominal[: max(8, len(nominal)//2)], list(config_module.ABLATION_METHODS), library=library, constraints=constraints, config_module=config_module, robust_samples=robust_main, final_robust_samples=final_ablation, local_budget=local_budget, baseline_budget=baseline_budget, seed=seed))

    all_df = normalize_results_df(pd.concat(all_frames, ignore_index=True))
    all_df.to_csv(out_dir / "all_results.csv", index=False)
    ok_df = _ok_only(all_df)

    if not ok_df.empty:
        summary = ok_df.groupby(["suite", "method"]).agg(
            robust_score_mean=("robust_score", "mean"),
            robust_score_std=("robust_score", "std"),
            cvar_pd_mean=("cvar_pd", "mean"),
            worst_pd_mean=("worst_pd", "mean"),
            violation_rate_mean=("risk_violation_rate", "mean"),
            total_runtime_sec_mean=("total_runtime_sec", "mean"),
            total_eval_count_mean=("total_eval_count", "mean"),
            peak_memory_mb_mean=("peak_memory_mb", "mean"),
            flop_proxy_mean=("flop_proxy", "mean"),
            llm_calls_mean=("llm_calls", "mean"),
            parsed_candidate_count_mean=("parsed_candidate_count", "mean"),
            fallback_count_mean=("fallback_count", "mean"),
            api_error_count_mean=("api_error_count", "mean"),
            retry_count_mean=("retry_count", "mean"),
            api_attempt_count_mean=("api_attempt_count", "mean"),
        ).reset_index()
    else:
        summary = pd.DataFrame()
    summary.to_csv(out_dir / "summary_all_suites.csv", index=False)
    write_json(out_dir / "run_metadata.json", {
        "seeds": seeds,
        "api_enabled": bool(config_module.USE_API),
        "api_url": config_module.API_URL,
        "model": config_module.MODEL_ID,
        "require_api_for_llm_methods": bool(config_module.REQUIRE_API_FOR_LLM_METHODS),
        "allow_rule_fallback_when_api_fails": bool(config_module.ALLOW_RULE_FALLBACK_WHEN_API_FAILS),
        "api_max_retries": int(getattr(config_module, "API_MAX_RETRIES", 8)),
        "resume": bool(getattr(config_module, "RESUME", True)),
        "stop_on_method_error": bool(getattr(config_module, "STOP_ON_METHOD_ERROR", False)),
        "pipeline_version": str(getattr(config_module, "PIPELINE_VERSION", "v4.5")),
        "final_unified_robust_verification": True,
        "selection_robust_samples_main": int(robust_main),
        "selection_robust_samples_ood": int(robust_ood),
        "final_verify_samples_main": int(final_main),
        "final_verify_samples_ood": int(final_ood),
        "final_verify_samples_ablation": int(final_ablation),
        "uncertainty_spec": make_uncertainty_spec_from_config(config_module).to_dict(),
        "status_ok_rows": int(len(ok_df)),
        "status_failed_rows": int(len(all_df) - len(ok_df)),
        "status_repaired_from_legacy_rows": int(all_df.get("status_repaired_from_legacy", pd.Series([0] * len(all_df))).fillna(0).astype(int).sum()),
    })
    integrity_report(all_df, config_module, out_dir)
    return all_df
