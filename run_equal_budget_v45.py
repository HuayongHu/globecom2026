from __future__ import annotations

"""Run equal-online-budget PSO/GA/DE baselines for v4.5.

The default full pipeline gives PSO/GA/DE roughly twice the online simulator
calls of RAG-CRA. This script reruns only the population baselines with
BASELINE_BUDGET = LOCAL_BUDGET, so their final total_eval_count matches the
RAG-CRA online evaluation budget under the same robust sample counts.
"""

from pathlib import Path
from types import SimpleNamespace
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from radar_llm_robust import config as base_config
from radar_llm_robust.experiments import run_paper_experiment, normalize_results_df


METHOD_LABELS = {
    "rag_cra": "RAG-CRA",
    "rag_cra_no_api": "RAG-CRA w/o LLM",
    "pso": "PSO-equal",
    "ga": "GA-equal",
    "de": "DE-equal",
}

SUITE_LABELS = {
    "main_nominal": "Nominal",
    "ood_stress": "OOD stress",
}


def clone_config(output_dir: str, methods: list[str], budget: int | None, include_random: bool) -> SimpleNamespace:
    data = {k: v for k, v in vars(base_config).items() if k.isupper()}
    if include_random and "random" not in methods:
        methods = methods + ["random"]
    data.update({
        "PIPELINE_VERSION": str(data.get("PIPELINE_VERSION", "v4.5")) + "+equal-budget",
        "OUTPUT_DIR": output_dir,
        "USE_API": False,
        "REQUIRE_API_FOR_LLM_METHODS": False,
        "ALLOW_RULE_FALLBACK_WHEN_API_FAILS": False,
        "MAIN_METHODS": methods,
        "ABLATION_METHODS": [],
        "BASELINE_BUDGET": int(budget if budget is not None else data.get("LOCAL_BUDGET", 96)),
        "RESUME": True,
        "RUN_SEMANTIC_STRESS": False,
    })
    return SimpleNamespace(**data)


def mean_ci(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) <= 1:
        return 0.0
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def summarize(df: pd.DataFrame, suite: str) -> pd.DataFrame:
    df = normalize_results_df(df)
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().eq("ok")].copy()
    if df.empty:
        return pd.DataFrame()
    part = df[df["suite"].eq(suite)].copy()
    if part.empty:
        return pd.DataFrame()
    time_col = "total_runtime_sec" if "total_runtime_sec" in part.columns else "runtime_sec"
    eval_col = "total_eval_count" if "total_eval_count" in part.columns else "eval_count"
    out = part.groupby("method").agg(
        robust_score=("robust_score", "mean"),
        robust_ci95=("robust_score", mean_ci),
        cvar_pd=("cvar_pd", "mean"),
        worst_pd=("worst_pd", "mean"),
        violation=("risk_violation_rate", "mean"),
        runtime_sec=(time_col, "mean"),
        eval_count=(eval_col, "mean"),
        n=("robust_score", "count"),
    ).reset_index()
    return out


def load_original_rag_rows(original_dir: Path) -> pd.DataFrame:
    p = original_dir / "all_results.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df = df[df["method"].isin(["rag_cra", "rag_cra_no_api"])].copy()
    return df


def format_table(df: pd.DataFrame, suite: str) -> pd.DataFrame:
    cols = ["method", "robust_score", "robust_ci95", "cvar_pd", "worst_pd", "violation", "runtime_sec", "eval_count", "n"]
    out = df[cols].copy()
    order = {"rag_cra": 0, "rag_cra_no_api": 1, "pso": 2, "ga": 3, "de": 4, "random": 5}
    out["order"] = out["method"].map(order).fillna(99)
    out = out.sort_values("order").drop(columns=["order"])
    out.insert(0, "Suite", SUITE_LABELS.get(suite, suite))
    out["Method"] = out["method"].map(lambda m: METHOD_LABELS.get(m, str(m)))
    out = out.drop(columns=["method"])
    out = out[["Suite", "Method", "robust_score", "robust_ci95", "cvar_pd", "worst_pd", "violation", "runtime_sec", "eval_count", "n"]]
    return out


def write_table(table: pd.DataFrame, base: Path) -> None:
    pretty = table.copy()
    rename = {
        "robust_score": "Robust",
        "robust_ci95": "CI95",
        "cvar_pd": "CVaR Pd",
        "worst_pd": "Worst Pd",
        "violation": "Viol.",
        "runtime_sec": "Time (s)",
        "eval_count": "Evals",
        "n": "n",
    }
    pretty = pretty.rename(columns=rename)
    for col in ["Robust", "CI95", "CVaR Pd", "Worst Pd", "Viol."]:
        pretty[col] = pretty[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    pretty["Time (s)"] = pretty["Time (s)"].map(lambda x: "" if pd.isna(x) else f"{float(x):.2f}")
    pretty["Evals"] = pretty["Evals"].map(lambda x: "" if pd.isna(x) else f"{float(x):.1f}")
    pretty["n"] = pretty["n"].map(lambda x: "" if pd.isna(x) else f"{int(float(x))}")
    pretty.to_csv(base.with_suffix(".csv"), index=False)
    base.with_suffix(".tex").write_text(pretty.to_latex(index=False, escape=False), encoding="utf-8")
    try:
        base.with_suffix(".md").write_text(pretty.to_markdown(index=False), encoding="utf-8")
    except Exception:
        base.with_suffix(".md").write_text(pretty.to_string(index=False), encoding="utf-8")


def plot_table(table: pd.DataFrame, suite: str, out: Path) -> None:
    if table.empty:
        return
    names = table["Method"].tolist()
    robust = table["robust_score"].astype(float).to_numpy()
    ci = table["robust_ci95"].astype(float).to_numpy()
    x = np.arange(len(names))
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": True, "grid.alpha": 0.18})
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    ax.bar(x, robust, yerr=ci, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Robust score")
    ax.set_title(f"Equal-budget comparison: {SUITE_LABELS.get(suite, suite)}")
    if len(robust):
        ax.set_ylim(max(0, np.nanmin(robust) - 0.06), min(1.02, np.nanmax(robust) + 0.06))
    fig.tight_layout()
    fig.savefig(out / f"fig_equal_budget_{suite}.png", bbox_inches="tight", dpi=600)
    fig.savefig(out / f"fig_equal_budget_{suite}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_equal_budget_report(equal_dir: Path, original_dir: Path) -> None:
    out = equal_dir / "paper_equal_budget"
    out.mkdir(parents=True, exist_ok=True)
    eq_path = equal_dir / "all_results.csv"
    if not eq_path.exists():
        raise FileNotFoundError(f"Missing {eq_path}")
    equal_df = pd.read_csv(eq_path)
    original_rag = load_original_rag_rows(original_dir)
    combined = pd.concat([original_rag, equal_df], ignore_index=True, sort=False) if not original_rag.empty else equal_df
    for suite in ["main_nominal", "ood_stress"]:
        s = summarize(combined, suite)
        if s.empty:
            continue
        table = format_table(s, suite)
        write_table(table, out / f"table_equal_budget_{suite}")
        plot_table(table, suite, out)
    meta = {
        "equal_budget_dir": str(equal_dir),
        "original_results_dir": str(original_dir),
        "interpretation": "PSO/GA/DE were rerun with BASELINE_BUDGET=LOCAL_BUDGET so online simulator calls match RAG-CRA under the same final verification sample count.",
    }
    (out / "equal_budget_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "README_equal_budget.md").write_text(
        "# Equal-budget reporting\n\n"
        "Use these tables to compare RAG-CRA against PSO/GA/DE under approximately the same online simulator-call budget. "
        "The RAG-CRA rows are read from the existing main run; PSO/GA/DE rows come from this equal-budget rerun.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run equal-budget PSO/GA/DE baselines.")
    parser.add_argument("--output-dir", default="outputs/equal_budget")
    parser.add_argument("--original-results-dir", default="outputs/paper_run")
    parser.add_argument("--budget", type=int, default=None, help="Baseline search budget. Default: config.LOCAL_BUDGET.")
    parser.add_argument("--include-random", action="store_true")
    args = parser.parse_args()

    methods = ["pso", "ga", "de"]
    cfg = clone_config(args.output_dir, methods, args.budget, args.include_random)
    print("=" * 78)
    print("Running equal-budget baselines")
    print("Methods:", cfg.MAIN_METHODS)
    print("Output dir:", cfg.OUTPUT_DIR)
    print("LOCAL_BUDGET:", cfg.LOCAL_BUDGET)
    print("BASELINE_BUDGET:", cfg.BASELINE_BUDGET)
    print("=" * 78)
    run_paper_experiment(cfg)
    make_equal_budget_report(Path(args.output_dir), Path(args.original_results_dir))
    print("Equal-budget report written to", Path(args.output_dir) / "paper_equal_budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
