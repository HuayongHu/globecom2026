from __future__ import annotations

from pathlib import Path
import json
import math
import statistics
import pandas as pd

from .experiments import normalize_results_df


def _ci95(xs):
    vals = []
    for x in xs:
        try:
            fx = float(x)
        except Exception:
            continue
        if math.isfinite(fx):
            vals.append(fx)
    if len(vals) <= 1:
        return 0.0
    return 1.96 * statistics.stdev(vals) / math.sqrt(len(vals))


def _safe_float(x):
    try:
        if pd.isna(x):
            return math.nan
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return math.nan


def diagnose_results(out_dir: str | Path) -> dict:
    """Create paper-readiness diagnostics for a completed experiment folder.

    Outputs:
    - diagnostic_method_ranking.csv
    - diagnostic_paired_deltas.csv
    - diagnostic_winner_counts.csv
    - diagnostic_family_means.csv
    - diagnostic_findings.md
    - diagnostic_findings.json
    """
    out_dir = Path(out_dir)
    all_path = out_dir / "all_results.csv"
    if not all_path.exists():
        raise FileNotFoundError(f"Missing {all_path}")

    df = normalize_results_df(pd.read_csv(all_path))
    ok = df[df["status"] == "ok"].copy()

    ranking_rows = []
    for suite, part in ok.groupby("suite"):
        for method, g in part.groupby("method"):
            ranking_rows.append({
                "suite": suite,
                "method": method,
                "n": int(len(g)),
                "robust_mean": float(g["robust_score"].mean()),
                "robust_ci95": float(_ci95(g["robust_score"].tolist())),
                "cvar_pd_mean": float(g["cvar_pd"].mean()),
                "worst_pd_mean": float(g["worst_pd"].mean()),
                "violation_rate_mean": float(g["risk_violation_rate"].mean()),
                "llm_calls_mean": float(g["llm_calls"].mean()) if "llm_calls" in g else math.nan,
                "parsed_candidates_mean": float(g["parsed_candidate_count"].mean()) if "parsed_candidate_count" in g else math.nan,
                "fallback_count_mean": float(g["fallback_count"].mean()) if "fallback_count" in g else math.nan,
            })
    ranking = pd.DataFrame(ranking_rows).sort_values(["suite", "robust_mean"], ascending=[True, False])
    ranking.to_csv(out_dir / "diagnostic_method_ranking.csv", index=False)

    paired_rows = []
    for suite, part in ok.groupby("suite"):
        pivot = part.pivot_table(index=["scenario_id", "seed"], columns="method", values="robust_score", aggfunc="mean")
        if "rag_cra" not in pivot.columns:
            continue
        for other in pivot.columns:
            if other == "rag_cra":
                continue
            d = (pivot["rag_cra"] - pivot[other]).dropna()
            if len(d) == 0:
                continue
            paired_rows.append({
                "suite": suite,
                "comparison": f"rag_cra - {other}",
                "n_pairs": int(len(d)),
                "mean_delta": float(d.mean()),
                "median_delta": float(d.median()),
                "ci95_delta": float(_ci95(d.tolist())),
                "rag_cra_win_rate": float((d > 0).mean()),
            })
    paired = pd.DataFrame(paired_rows).sort_values(["suite", "mean_delta"], ascending=[True, False])
    paired.to_csv(out_dir / "diagnostic_paired_deltas.csv", index=False)

    winner_rows = []
    for suite, part in ok.groupby("suite"):
        for key, g in part.groupby(["scenario_id", "seed"]):
            best = g.sort_values("robust_score", ascending=False).iloc[0]
            winner_rows.append({"suite": suite, "scenario_id": key[0], "seed": key[1], "winner": best["method"], "winner_robust": float(best["robust_score"])})
    winners = pd.DataFrame(winner_rows)
    if not winners.empty:
        winners.groupby(["suite", "winner"]).size().reset_index(name="wins").to_csv(out_dir / "diagnostic_winner_counts.csv", index=False)
    else:
        pd.DataFrame(columns=["suite", "winner", "wins"]).to_csv(out_dir / "diagnostic_winner_counts.csv", index=False)

    fam = ok[ok["suite"] == "main_nominal"].pivot_table(index="family", columns="method", values="robust_score", aggfunc="mean")
    fam.to_csv(out_dir / "diagnostic_family_means.csv")

    risks = []
    def get_mean(suite, method):
        r = ranking[(ranking["suite"] == suite) & (ranking["method"] == method)]
        return float(r.iloc[0]["robust_mean"]) if len(r) else math.nan

    integrity_path = out_dir / "api_integrity_summary.json"
    integrity = {}
    if integrity_path.exists():
        try:
            integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        except Exception:
            integrity = {}
        if int(integrity.get("failed_rows", 0) or 0) > 0:
            risks.append("Integrity report still has failed rows; do not use results directly in the paper.")
        if int(integrity.get("schema_repaired_legacy_rows", 0) or 0) > 0:
            risks.append("Some rows were repaired from legacy blank status fields; this is okay for completeness but should be mentioned in lab notes.")
        if int(integrity.get("telemetry_warning_rows", 0) or 0) > 0:
            risks.append("Some legacy rows have missing telemetry or legacy no-api fallback warnings.")

    for suite in ["main_nominal", "ood_stress", "ablation"]:
        sub = ranking[ranking["suite"] == suite]
        if len(sub):
            best = sub.sort_values("robust_mean", ascending=False).iloc[0]
            if suite in {"main_nominal", "ood_stress"} and best["method"] != "rag_cra":
                risks.append(f"{suite}: rag_cra is not ranked first by robust mean; best is {best['method']}.")

    d_main = get_mean("main_nominal", "rag_cra") - get_mean("main_nominal", "rag_cra_no_api")
    if math.isfinite(d_main) and d_main < 0.01:
        risks.append(f"main_nominal: rag_cra advantage over no-api is small ({d_main:+.4f}).")
    d_abl = get_mean("ablation", "rag_cra") - get_mean("ablation", "rag_cra_no_api")
    if math.isfinite(d_abl) and d_abl <= 0:
        risks.append(f"ablation: no-api is at least as strong as rag_cra ({d_abl:+.4f}); v4 clean no-api ablation should be rerun.")

    no_api_fb = ok[ok["method"] == "rag_cra_no_api"].get("fallback_count")
    if no_api_fb is not None and len(no_api_fb) and _safe_float(no_api_fb.mean()) > 0:
        risks.append("rag_cra_no_api has nonzero fallback_count in existing results, meaning the v3 no-api baseline was overpowered by rule-fallback proposals.")

    findings = {
        "rows": int(len(df)),
        "ok_rows": int(len(ok)),
        "non_ok_rows": int(len(df) - len(ok)),
        "integrity": integrity,
        "risks": risks,
    }
    (out_dir / "diagnostic_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")

    lines = ["# v4 Diagnostic Findings", "", f"Rows: {len(df)}; usable ok rows: {len(ok)}; non-ok rows: {len(df)-len(ok)}.", ""]
    if risks:
        lines.append("## Warnings / paper-readiness risks")
        lines.extend([f"- {r}" for r in risks])
    else:
        lines.append("No major paper-readiness risks were detected by the automatic checks.")
    lines.append("")
    lines.append("## Best method by suite")
    for suite in ["main_nominal", "ood_stress", "ablation"]:
        sub = ranking[ranking["suite"] == suite].sort_values("robust_mean", ascending=False)
        if len(sub):
            b = sub.iloc[0]
            lines.append(f"- {suite}: {b['method']} robust={b['robust_mean']:.4f} +/- {b['robust_ci95']:.4f}")
    lines.append("")
    lines.append("See diagnostic_method_ranking.csv, diagnostic_paired_deltas.csv, diagnostic_winner_counts.csv, and diagnostic_family_means.csv for details.")
    (out_dir / "diagnostic_findings.md").write_text("\n".join(lines), encoding="utf-8")
    return findings


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default="outputs/paper_run")
    args = ap.parse_args()
    diagnose_results(args.out_dir)
    print(f"Wrote diagnostics to {args.out_dir}")
