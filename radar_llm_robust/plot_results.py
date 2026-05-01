from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .experiments import normalize_results_df


def _setup_style():
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
        "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "figure.dpi": 140, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": True,
        "grid.alpha": 0.22, "lines.linewidth": 2.0,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _save(fig, path_base: Path, save_pdf: bool = True):
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), bbox_inches="tight")
    if save_pdf:
        fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_all_results(out_dir: Path) -> tuple[pd.DataFrame, dict]:
    df = normalize_results_df(pd.read_csv(out_dir / "all_results.csv"))
    histories = {}
    for p in sorted(out_dir.glob("seed_*/*_histories.json")):
        try:
            histories.update(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return df, histories


def summarize(df: pd.DataFrame, suite: str) -> pd.DataFrame:
    df = normalize_results_df(df)
    part = df[df["suite"] == suite].copy()
    if "status" in part.columns:
        part = part[part["status"] == "ok"].copy()
    if part.empty:
        return pd.DataFrame(columns=["method","robust_mean","robust_std","cvar_pd_mean","worst_pd_mean","viol_mean","time_mean","eval_mean","mem_mean","flop_mean","llm_calls","parsed","fallback","n","robust_ci"])
    grp = part.groupby("method").agg(
        robust_mean=("robust_score", "mean"),
        robust_std=("robust_score", "std"),
        cvar_pd_mean=("cvar_pd", "mean"),
        worst_pd_mean=("worst_pd", "mean"),
        viol_mean=("risk_violation_rate", "mean"),
        time_mean=("total_runtime_sec", "mean"),
        eval_mean=("total_eval_count", "mean"),
        mem_mean=("peak_memory_mb", "mean"),
        flop_mean=("flop_proxy", "mean"),
        llm_calls=("llm_calls", "mean"),
        parsed=("parsed_candidate_count", "mean"),
        fallback=("fallback_count", "mean"),
        n=("robust_score", "count"),
    ).reset_index()
    grp["robust_ci"] = 1.96 * grp["robust_std"].fillna(0) / np.sqrt(np.maximum(grp["n"], 1))
    return grp.sort_values("robust_mean", ascending=False)


def plot_bar(df, suite, out_dir, name, title, save_pdf):
    s = summarize(df, suite)
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    x = np.arange(len(s))
    ax.bar(x, s["robust_mean"], yerr=s["robust_ci"], capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(s["method"], rotation=25, ha="right")
    ax.set_ylabel("Unified final robust score")
    ax.set_title(title)
    y0 = max(0.0, s["robust_mean"].min() - 0.04)
    y1 = min(1.02, s["robust_mean"].max() + 0.04)
    if y1 - y0 > 0.08:
        ax.set_ylim(y0, y1)
    _save(fig, out_dir / name, save_pdf)


def plot_ablation(df, out_dir, save_pdf):
    s = summarize(df, "ablation")
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(len(s)); width = 0.36
    ax.bar(x - width/2, s["robust_mean"], width=width, label="Robust score")
    ax.bar(x + width/2, s["cvar_pd_mean"], width=width, label="CVaR Pd")
    ax.set_xticks(x); ax.set_xticklabels(s["method"], rotation=28, ha="right")
    ax.set_ylabel("Unified final metric")
    ax.set_title("Ablation under the same final robust evaluator")
    ax.legend(frameon=False)
    _save(fig, out_dir / "fig3_ablation_unified", save_pdf)


def plot_convergence(histories, df, suite, out_dir, save_pdf):
    part = df[df["suite"] == suite]
    curves = {}
    for key, hist in histories.items():
        pieces = key.split("/")
        if len(pieces) < 4 or pieces[0] != suite or not hist:
            continue
        method = pieces[-1]
        xs = np.array([h.get("eval", 0.0) for h in hist], dtype=float)
        ys = np.maximum.accumulate(np.array([h.get("score", 0.0) for h in hist], dtype=float))
        if len(xs) < 2:
            continue
        grid = np.linspace(0, xs.max(), 100)
        curves.setdefault(method, []).append((grid, np.interp(grid, xs, ys)))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method, vals in curves.items():
        if not vals:
            continue
        max_x = max(v[0][-1] for v in vals)
        grid = np.linspace(0, max_x, 140)
        data = np.vstack([np.interp(grid, g, y) for g, y in vals])
        median = np.nanmedian(data, axis=0)
        q25 = np.nanpercentile(data, 25, axis=0)
        q75 = np.nanpercentile(data, 75, axis=0)
        ax.plot(grid, median, label=method, linewidth=2.0)
        # v4.3/v4.4: IQR ribbon is intentionally lighter/narrower than mean +/- std.
        ax.fill_between(grid, q25, q75, alpha=0.07, linewidth=0)
    ax.set_xlabel("Robust simulator calls during selection")
    ax.set_ylabel("Median best selection score")
    ax.set_title("Convergence versus simulator calls")
    ax.legend(ncol=2, frameon=False, loc="lower right")
    _save(fig, out_dir / "fig4_convergence_calls", save_pdf)


def plot_complexity(df, out_dir, save_pdf):
    s = summarize(df, "main_nominal")
    if s.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6))
    metrics = [("time_mean", "Total time (s)"), ("mem_mean", "Peak memory (MB)"), ("eval_mean", "Total simulator calls"), ("flop_mean", "FLOP/token proxy")]
    for ax, (col, title) in zip(axes.ravel(), metrics):
        ax.bar(np.arange(len(s)), s[col])
        ax.set_xticks(np.arange(len(s))); ax.set_xticklabels(s["method"], rotation=25, ha="right")
        ax.set_title(title); ax.yaxis.set_major_locator(MaxNLocator(5))
    fig.suptitle("Complexity and resource analysis", y=1.02)
    _save(fig, out_dir / "fig5_complexity", save_pdf)


def plot_api_integrity(df, out_dir, save_pdf):
    s = df.groupby("method").agg(llm_calls=("llm_calls","mean"), parsed=("parsed_candidate_count","mean"), fallback=("fallback_count","mean"), retries=("retry_count","mean"), attempts=("api_attempt_count","mean")).reset_index()
    # v4.3/v4.4: keep only methods with actual API activity; otherwise the chart is mostly empty.
    s = s[(s["llm_calls"] > 0) | (s["parsed"] > 0) | (s["fallback"] > 0) | (s["retries"] > 0)].copy()
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8,4.0))
    x=np.arange(len(s)); width=0.22
    ax.bar(x-width, s["llm_calls"], width=width, label="LLM calls")
    ax.bar(x, s["parsed"], width=width, label="Parsed candidates")
    ax.bar(x+width, s["fallback"], width=width, label="Fallback count")
    ax.set_xticks(x); ax.set_xticklabels(s["method"], rotation=20, ha="right")
    ax.set_title("API integrity statistics")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    _save(fig, out_dir / "fig6_api_integrity", save_pdf)


def plot_heatmap(df, out_dir, save_pdf):
    df = normalize_results_df(df)
    part = df[df["suite"] == "main_nominal"].copy()
    if "status" in part.columns:
        part = part[part["status"] == "ok"].copy()
    if part.empty:
        return
    table = part.pivot_table(index="family", columns="method", values="robust_score", aggfunc="mean")
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    im = ax.imshow(table.values, aspect="auto")
    ax.set_xticks(np.arange(table.shape[1])); ax.set_xticklabels(table.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(table.shape[0])); ax.set_yticklabels(table.index)
    ax.set_title("Scenario-family robustness heatmap")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cbar.set_label("Robust score")
    _save(fig, out_dir / "fig7_family_heatmap", save_pdf)


def make_tables(df, out_dir):
    for suite in ["main_nominal", "ood_stress", "ablation"]:
        s = summarize(df, suite)
        if s.empty:
            continue
        pretty = s[["method","robust_mean","robust_ci","cvar_pd_mean","worst_pd_mean","viol_mean","time_mean","eval_mean","mem_mean","llm_calls","parsed","fallback"]].copy()
        pretty.columns = ["Method","Robust","CI95","CVaR_Pd","Worst_Pd","Viol","Time_s","Evals","Mem_MB","LLM_calls","Parsed","Fallback"]
        pretty = pretty.round({"Robust":3,"CI95":3,"CVaR_Pd":3,"Worst_Pd":3,"Viol":3,"Time_s":2,"Evals":1,"Mem_MB":1,"LLM_calls":1,"Parsed":1,"Fallback":1})
        (out_dir / f"table_{suite}.tex").write_text(pretty.to_latex(index=False), encoding="utf-8")
        pretty.to_csv(out_dir / f"table_{suite}.csv", index=False)


def build_report_summary(df, out_dir):
    df = normalize_results_df(df) if len(df) else df
    failed = int((df.get("status", pd.Series(["ok"] * len(df))) != "ok").sum()) if len(df) else 0
    text=["# Experiment summary", "", "All reported scores use the same final robust evaluator.", f"Failed/incomplete rows: {failed}. Rerun with RESUME=True to retry them."]
    for suite in ["main_nominal", "ood_stress", "ablation"]:
        s=summarize(df,suite)
        if not s.empty:
            best=s.iloc[0]
            text.append(f"- {suite}: best method **{best['method']}**, robust score {best['robust_mean']:.3f}.")
    (out_dir/"report_summary.md").write_text("\n".join(text), encoding="utf-8")


def make_all_figures(out_dir: str | Path, save_pdf: bool = True):
    _setup_style()
    out_dir = Path(out_dir)
    df, histories = load_all_results(out_dir)
    plot_bar(df, "main_nominal", out_dir, "fig1_main_performance_unified", "Main benchmark: unified robust performance", save_pdf)
    plot_bar(df, "ood_stress", out_dir, "fig2_ood_performance_unified", "OOD stress benchmark: unified robust performance", save_pdf)
    plot_ablation(df, out_dir, save_pdf)
    plot_convergence(histories, df, "main_nominal", out_dir, save_pdf)
    plot_complexity(df, out_dir, save_pdf)
    plot_api_integrity(df, out_dir, save_pdf)
    plot_heatmap(df, out_dir, save_pdf)
    make_tables(df, out_dir)
    build_report_summary(df, out_dir)
