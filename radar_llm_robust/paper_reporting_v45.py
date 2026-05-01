from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .robust import UncertaintySpec

METHOD_LABELS = {
    "rag_cra": "RAG-CRA",
    "rag_cra_no_api": "RAG-CRA w/o LLM",
    "pso": "PSO",
    "ga": "GA",
    "de": "DE",
    "random": "Random",
    "ml_policy": "ML Policy",
    "direct_llm": "Direct LLM",
    "rag_only": "RAG only",
    "rag_cra_no_robust": "w/o robust objective",
    "rag_cra_no_refine": "w/o refinement",
    "rule_no_semantic": "Rule w/o Semantic",
    "semantic_rule": "Semantic Rule",
    "semantic_llm": "Semantic LLM",
}

# Standard numerical baselines plus the no-LLM diagnostic row when available.
PRIMARY_METHODS = ["rag_cra", "rag_cra_no_api", "pso", "ga", "de", "random", "ml_policy"]
ABLATION_METHODS = ["direct_llm", "rag_only", "rag_cra_no_robust", "rag_cra_no_refine", "rag_cra_no_api", "rag_cra"]
SEMANTIC_METHODS = ["rule_no_semantic", "semantic_rule", "semantic_llm"]

PRETTY = {
    "robust_score": "Robust",
    "cvar_pd": "CVaR Pd",
    "worst_pd": "Worst Pd",
    "risk_violation_rate": "Viol.",
    "no_violation": "No Viol.",
    "runtime_sec": "Time (s)",
    "total_runtime_sec": "Time (s)",
    "eval_count": "Evals",
    "total_eval_count": "Evals",
    "semantic_compliance": "Semantic",
    "hard_satisfaction": "Hard",
    "soft_preference_score": "Soft",
}

SEMANTIC_RENAME = {
    "robust": "robust_score",
    "soft_preference": "soft_preference_score",
    "violation": "risk_violation_rate",
}


def _style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _label(method: str) -> str:
    return METHOD_LABELS.get(str(method), str(method))


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "status" in out.columns:
        out = out[out["status"].fillna("ok").astype(str).str.lower().eq("ok")]
    if "robust_score" in out.columns:
        out = out[out["robust_score"].notna()]
    return out.copy()


def _mean_ci(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) <= 1:
        return 0.0
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def _summary(df: pd.DataFrame, suite: str, methods: list[str]) -> pd.DataFrame:
    part = _ok(df)
    part = part[(part["suite"] == suite) & (part["method"].isin(methods))].copy()
    if part.empty:
        return pd.DataFrame()
    time_col = "total_runtime_sec" if "total_runtime_sec" in part.columns else "runtime_sec"
    eval_col = "total_eval_count" if "total_eval_count" in part.columns else "eval_count"
    agg = part.groupby("method").agg(
        robust_score=("robust_score", "mean"),
        cvar_pd=("cvar_pd", "mean"),
        worst_pd=("worst_pd", "mean"),
        risk_violation_rate=("risk_violation_rate", "mean"),
        runtime_sec=(time_col, "mean"),
        eval_count=(eval_col, "mean"),
        n=("robust_score", "count"),
        robust_ci95=("robust_score", _mean_ci),
    ).reset_index()
    order = {m: i for i, m in enumerate(methods)}
    agg["order"] = agg["method"].map(order)
    agg = agg.sort_values("order").drop(columns=["order"])
    agg.insert(0, "Method", agg["method"].map(_label))
    return agg


def _normalize_semantic_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "method" not in df.columns and len(df.columns):
        df = df.rename(columns={df.columns[0]: "method"})
    df = df.rename(columns={k: v for k, v in SEMANTIC_RENAME.items() if k in df.columns})
    for col in ["robust_score", "semantic_compliance", "hard_satisfaction", "soft_preference_score", "cvar_pd", "worst_pd", "risk_violation_rate", "no_violation"]:
        if col not in df.columns:
            df[col] = np.nan
    if df["no_violation"].isna().all() and "risk_violation_rate" in df.columns:
        df["no_violation"] = 1.0 - pd.to_numeric(df["risk_violation_rate"], errors="coerce")
    df = df[df["method"].isin(SEMANTIC_METHODS)].copy()
    order = {m: i for i, m in enumerate(SEMANTIC_METHODS)}
    df["order"] = df["method"].map(order)
    df = df.sort_values("order").drop(columns=["order"])
    df.insert(0, "Method", df["method"].map(_label))
    return df


def _fmt_table(table: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    available = [c for c in cols if c in table.columns]
    out = table[["Method"] + available].copy().rename(columns={c: PRETTY.get(c, c) for c in available})
    for c in out.columns:
        if c == "Method":
            continue
        if c in {"Evals", "n"}:
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
        elif c == "Time (s)":
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
        else:
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    return out


def _write(table: pd.DataFrame, base: Path, cols: list[str]) -> None:
    if table.empty:
        return
    out = _fmt_table(table, cols)
    out.to_csv(base.with_suffix(".csv"), index=False)
    base.with_suffix(".tex").write_text(out.to_latex(index=False, escape=False), encoding="utf-8")
    try:
        base.with_suffix(".md").write_text(out.to_markdown(index=False), encoding="utf-8")
    except Exception:
        base.with_suffix(".md").write_text(out.to_string(index=False), encoding="utf-8")


def _bar(table: pd.DataFrame, base: Path, metric: str, title: str, ylabel: str) -> None:
    if table.empty or metric not in table.columns:
        return
    x = np.arange(len(table))
    y = table[metric].astype(float).to_numpy()
    err = table["robust_ci95"].to_numpy() if metric == "robust_score" and "robust_ci95" in table.columns else None
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(x, y, yerr=err, capsize=3 if err is not None else 0)
    ax.set_xticks(x)
    ax.set_xticklabels(table["Method"], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if metric in {"robust_score", "cvar_pd", "worst_pd"} and len(y):
        ax.set_ylim(max(0, float(np.nanmin(y)) - 0.08), min(1.02, float(np.nanmax(y)) + 0.08))
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _paired_delta(df: pd.DataFrame, suite: str, out_dir: Path) -> pd.DataFrame:
    part = _ok(df)
    part = part[part["suite"] == suite].copy()
    rows = []
    for baseline in ["rag_cra_no_api", "pso", "ga", "de", "random", "ml_policy"]:
        a = part[part["method"] == "rag_cra"][["scenario_id", "seed", "robust_score"]].rename(columns={"robust_score": "rag"})
        b = part[part["method"] == baseline][["scenario_id", "seed", "robust_score"]].rename(columns={"robust_score": "base"})
        merged = a.merge(b, on=["scenario_id", "seed"], how="inner")
        if merged.empty:
            continue
        delta = merged["rag"] - merged["base"]
        rows.append({
            "Baseline": _label(baseline),
            "mean_delta": float(delta.mean()),
            "ci95": _mean_ci(delta),
            "median_delta": float(delta.median()),
            "win_rate": float((delta > 0).mean()),
            "n": int(len(delta)),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out.to_csv(out_dir / f"v45_paired_delta_{suite}.csv", index=False)
    pretty = out.copy()
    pretty["Mean Delta"] = pretty["mean_delta"].map(lambda x: f"{x:+.3f}")
    pretty["CI95"] = pretty["ci95"].map(lambda x: f"{x:.3f}")
    pretty["Median"] = pretty["median_delta"].map(lambda x: f"{x:+.3f}")
    pretty["Win Rate"] = pretty["win_rate"].map(lambda x: f"{100*x:.1f}\\%")
    pretty = pretty[["Baseline", "Mean Delta", "CI95", "Median", "Win Rate", "n"]]
    pretty.to_csv(out_dir / f"table_paired_delta_{suite}.csv", index=False)
    (out_dir / f"table_paired_delta_{suite}.tex").write_text(pretty.to_latex(index=False, escape=False), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    x = np.arange(len(out))
    ax.bar(x, out["mean_delta"], yerr=out["ci95"], capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(out["Baseline"], rotation=20, ha="right")
    ax.set_ylabel("Mean paired delta Robust")
    ax.set_title(f"RAG-CRA vs baselines ({suite.replace('_', ' ')})")
    fig.tight_layout()
    fig.savefig(out_dir / f"v45_paired_delta_{suite}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"v45_paired_delta_{suite}.pdf", bbox_inches="tight")
    plt.close(fig)
    return out


def _write_uncertainty_table(out_dir: Path) -> None:
    df = pd.DataFrame(UncertaintySpec().to_rows())
    df.to_csv(out_dir / "table_uncertainty_model.csv", index=False)
    (out_dir / "table_uncertainty_model.tex").write_text(df.to_latex(index=False, escape=False), encoding="utf-8")
    try:
        (out_dir / "table_uncertainty_model.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    except Exception:
        (out_dir / "table_uncertainty_model.md").write_text(df.to_string(index=False), encoding="utf-8")


def _write_llm_telemetry(df: pd.DataFrame, out_dir: Path) -> None:
    part = _ok(df)
    if part.empty or "llm_calls" not in part.columns:
        return
    d = part[pd.to_numeric(part.get("llm_calls", 0), errors="coerce").fillna(0) > 0].copy()
    if d.empty:
        return
    d["tokens"] = pd.to_numeric(d.get("prompt_tokens", 0), errors="coerce").fillna(0) + pd.to_numeric(d.get("completion_tokens", 0), errors="coerce").fillna(0)
    g = d.groupby(["suite", "method"]).agg(
        llm_calls=("llm_calls", "mean"),
        parse_success=("parse_success_count", "mean"),
        parsed_candidates=("parsed_candidate_count", "mean"),
        retries=("retry_count", "mean"),
        api_errors=("api_error_count", "mean"),
        fallbacks=("fallback_count", "mean"),
        latency=("llm_latency_sec", "mean"),
        tokens=("tokens", "mean"),
        n=("scenario_id", "count"),
    ).reset_index()
    g.insert(0, "Method", g["method"].map(_label))
    g["Suite"] = g["suite"].str.replace("_", " ")
    g["Parse/call"] = g.apply(lambda r: np.nan if r["llm_calls"] <= 0 else r["parse_success"] / r["llm_calls"], axis=1)
    out = g[["Suite", "Method", "llm_calls", "Parse/call", "parsed_candidates", "retries", "api_errors", "fallbacks", "latency", "tokens", "n"]].copy()
    out.columns = ["Suite", "Method", "Calls", "Parse/call", "Parsed", "Retries", "API err.", "Fallback", "Latency", "Tokens", "n"]
    for c in ["Calls", "Parse/call", "Parsed", "Retries", "API err.", "Fallback", "Latency", "Tokens"]:
        out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    out.to_csv(out_dir / "table_llm_telemetry.csv", index=False)
    (out_dir / "table_llm_telemetry.tex").write_text(out.to_latex(index=False, escape=False), encoding="utf-8")


def _semantic_summary(semantic_dir: Path) -> pd.DataFrame:
    path = semantic_dir / "semantic_stress_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return _normalize_semantic_summary(pd.read_csv(path))


def _plot_semantic(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    metrics = ["robust_score", "semantic_compliance", "hard_satisfaction", "soft_preference_score", "cvar_pd", "worst_pd", "no_violation"]
    labels = [PRETTY.get(m, m) for m in metrics]
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.3, 3.8))
    for i, (_, row) in enumerate(summary.iterrows()):
        vals = [float(row.get(m, np.nan)) for m in metrics]
        ax.bar(x + (i - 1) * width, vals, width=width, label=row["Method"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title("Semantic-constrained stress test")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(out_dir / "v45_semantic_stress_summary.png", bbox_inches="tight")
    fig.savefig(out_dir / "v45_semantic_stress_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_paper_outputs(results_dir: str | Path, semantic_dir: str | Path | None = None, output_subdir: str = "paper_v45") -> None:
    _style()
    results_dir = Path(results_dir)
    out_dir = results_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results_path = results_dir / "all_results.csv"
    if not all_results_path.exists():
        raise FileNotFoundError(f"Missing {all_results_path}. Run the main experiment first or set --results-dir correctly.")
    df = _ok(pd.read_csv(all_results_path))

    main = _summary(df, "main_nominal", PRIMARY_METHODS)
    ood = _summary(df, "ood_stress", PRIMARY_METHODS)
    ablation = _summary(df, "ablation", ABLATION_METHODS)

    _write(main, out_dir / "table_I_main_nominal_v45", ["robust_score", "cvar_pd", "worst_pd", "risk_violation_rate", "runtime_sec", "eval_count"])
    _write(ood, out_dir / "table_II_ood_v45", ["robust_score", "cvar_pd", "worst_pd", "risk_violation_rate", "runtime_sec", "eval_count"])
    _write(ablation, out_dir / "table_III_ablation_v45", ["robust_score", "cvar_pd", "worst_pd", "risk_violation_rate", "runtime_sec", "eval_count"])

    _bar(main, out_dir / "fig_v45_main_robust", "robust_score", "Main nominal benchmark", "Robust score")
    _bar(ood, out_dir / "fig_v45_ood_robust", "robust_score", "OOD stress benchmark", "Robust score")
    _bar(ablation, out_dir / "fig_v45_ablation_robust", "robust_score", "Ablation study", "Robust score")
    _paired_delta(df, "main_nominal", out_dir)
    _paired_delta(df, "ood_stress", out_dir)
    _write_uncertainty_table(out_dir)
    _write_llm_telemetry(df, out_dir)

    semantic_note = "Semantic stress directory not provided or not found."
    if semantic_dir is not None:
        semantic_dir = Path(semantic_dir)
        if semantic_dir.exists():
            semantic = _semantic_summary(semantic_dir)
            if not semantic.empty:
                _write(semantic, out_dir / "table_IV_semantic_stress_v45", ["robust_score", "semantic_compliance", "hard_satisfaction", "soft_preference_score", "cvar_pd", "worst_pd", "risk_violation_rate"])
                _plot_semantic(semantic, out_dir)
                semantic_note = f"Semantic stress table generated from {semantic_dir}."

    notes = [
        "# v4.5 paper reporting notes",
        "",
        "This layer adds reviewer-facing reporting without changing the v4.4.1 benchmark defaults.",
        "",
        "Claim boundary:",
        "- Use 'risk-aware' or 'sample-risk-verified' rather than unqualified 'risk-certified'.",
        "- The risk report is sample based under the prescribed perturbation model.",
        "- The no-LLM diagnostic row is included when present to expose how much the LLM contributes beyond retrieval, repair, and refinement.",
        "- Paired-delta tables report mean delta, CI, median delta, and win rate rather than claiming per-scenario dominance.",
        "",
        semantic_note,
    ]
    (out_dir / "v45_reporting_notes.md").write_text("\n".join(notes), encoding="utf-8")
    print(f"v4.5 paper outputs written to: {out_dir}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate v4.5 paper-oriented tables/figures without rerunning experiments.")
    parser.add_argument("--results-dir", default="outputs/paper_run")
    parser.add_argument("--semantic-dir", default="outputs/semantic_stress")
    parser.add_argument("--output-subdir", default="paper_v45")
    args = parser.parse_args()
    semantic_dir = Path(args.semantic_dir)
    generate_paper_outputs(args.results_dir, semantic_dir if semantic_dir.exists() else None, args.output_subdir)


if __name__ == "__main__":
    main()
