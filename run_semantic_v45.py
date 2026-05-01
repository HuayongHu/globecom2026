from __future__ import annotations

"""Run the semantic-constrained stress benchmark and refresh paper tables.

This script exists because run_full_v45_pipeline.py only runs the semantic
benchmark when config.RUN_SEMANTIC_STRESS is True. You can run this script
after the main numerical pipeline has finished.
"""

from pathlib import Path
import sys
import pandas as pd

from radar_llm_robust import config
from radar_llm_robust.semantic_stress import run_semantic_stress
from radar_llm_robust.paper_reporting_v45 import generate_paper_outputs


def _check_api_config() -> None:
    if not bool(getattr(config, "USE_API", True)):
        raise RuntimeError("USE_API is False. Set USE_API=True before semantic LLM runs.")
    api_key = str(getattr(config, "API_KEY", "") or "")
    if not api_key or api_key == "PASTE_YOUR_API_KEY_HERE":
        raise RuntimeError("API_KEY is not configured in radar_llm_robust/config.py.")
    if bool(getattr(config, "SEMANTIC_ALLOW_RULE_FALLBACK", False)):
        raise RuntimeError("SEMANTIC_ALLOW_RULE_FALLBACK must be False for publishable semantic LLM results.")


def main() -> int:
    print("=" * 78)
    print("Running v4.5 semantic-constrained stress benchmark")
    print("=" * 78)
    _check_api_config()

    df = run_semantic_stress(config)
    out_dir = Path(getattr(config, "SEMANTIC_OUTPUT_DIR", "outputs/semantic_stress"))
    csv_path = out_dir / "semantic_stress_results.csv"
    summary_path = out_dir / "semantic_stress_summary.csv"

    failed = df[df.get("status", "ok").astype(str).str.lower().ne("ok")].copy() if len(df) else pd.DataFrame()
    if not failed.empty:
        failed.to_csv(out_dir / "semantic_failed_rows.csv", index=False)
        print("ERROR: semantic stress contains failed rows. See", out_dir / "semantic_failed_rows.csv")
        return 2

    main_results = Path(getattr(config, "OUTPUT_DIR", "outputs/paper_run")) / "all_results.csv"
    if main_results.exists():
        generate_paper_outputs(
            getattr(config, "OUTPUT_DIR", "outputs/paper_run"),
            getattr(config, "SEMANTIC_OUTPUT_DIR", "outputs/semantic_stress"),
        )
        print("Refreshed paper_v45 tables/figures, including semantic table if available.")
    else:
        print("Main numerical all_results.csv not found; skipped paper_v45 refresh.")

    print("Semantic results:", csv_path)
    print("Semantic summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
