from __future__ import annotations

"""RAG-CRA v4.5 configuration.

Edit API_URL, API_KEY and MODEL_ID, then run:
    python run_paper_experiment.py

The v4.5 defaults preserve the v4.4.1 numerical protocol. Optional
FINAL_VERIFY_* settings can be increased for stronger sample-risk reporting.
"""

PIPELINE_VERSION = "v4.5"

# API settings
USE_API = True
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY = "PASTE_YOUR_API_KEY_HERE"
MODEL_ID = "openai/gpt-oss-120b"
REQUEST_TIMEOUT_SEC = 180
USE_RESPONSE_FORMAT_JSON = True
SAVE_API_TRACES = True

# Important safety switches
# If True, LLM methods will stop with an error when API is unavailable.
# This prevents silently reporting no-API results as LLM results.
REQUIRE_API_FOR_LLM_METHODS = True
ALLOW_RULE_FALLBACK_WHEN_API_FAILS = False

# Experiment settings
OUTPUT_DIR = "outputs/paper_run"
RANDOM_SEEDS = [2026, 2027, 2028]
FAST_MODE = False
SAVE_PDF_FIGURES = True

# LLM and optimization settings
LLM_CANDIDATES = 6
LLM_MAX_TOKENS = 2500
LIBRARY_BUILD_SIZE = 220
ROBUST_SAMPLES_MAIN = 32
ROBUST_SAMPLES_OOD = 48
# By default final reporting uses the same sample count as selection,
# preserving v4.4.1 numbers. Set these to 256 or 512 for tighter
# sample-risk estimates when regenerating results.
FINAL_VERIFY_SAMPLES_MAIN = None
FINAL_VERIFY_SAMPLES_OOD = None
FINAL_VERIFY_SAMPLES_ABLATION = None
LOCAL_BUDGET = 96
BASELINE_BUDGET = 192
PFA = 1e-6
CVAR_ALPHA = 0.20
RISK_PD_THRESHOLD = 0.80

# Methods
MAIN_METHODS = ["rag_cra", "rag_cra_no_api", "ml_policy", "pso", "ga", "de", "random"]
ABLATION_METHODS = [
    "direct_llm",
    "rag_only",
    "rag_cra_no_api",
    "rag_cra",
    "rag_cra_no_refine",
    "rag_cra_no_robust",
]

# Dataset files
LIBRARY_SCENARIOS_FILE = "datasets/library_seed_scenarios.csv"
NOMINAL_SCENARIOS_FILE = "datasets/nominal_scenarios.csv"
OOD_SCENARIOS_FILE = "datasets/ood_scenarios.csv"

# Retry and resilience settings added in v3.1 and hardened in v4.0
API_MAX_RETRIES = 8
API_RETRY_INITIAL_DELAY = 2.0
API_RETRY_MAX_DELAY = 120.0
API_RETRY_EXP_BASE = 2.0
API_RETRY_JITTER = 0.25
API_RETRY_STATUS_CODES = [408, 409, 425, 429, 500, 502, 503, 504]
API_PARSE_RETRY_MAX = 2

# Long-run resilience. Failed rows are recorded and retried on the next run when RESUME=True.
RESUME = True
SAVE_PARTIAL_EVERY_ROW = True
STOP_ON_METHOD_ERROR = False
RAISE_ON_INTEGRITY_FAILURE = False

# v4.0 result-schema compatibility and paper-readiness diagnostics
NORMALIZE_LEGACY_STATUS = True
RUN_RESULT_DIAGNOSTICS = True
# v4 clean ablation: rag_cra_no_api no longer obtains extra candidates through the LLM-client fallback path.
NO_API_BASELINE_USES_LLM_FALLBACK = False


# v4.1+ extra figures and optional semantic constraint stress test
RUN_V41_EXTRA_FIGURES = True
RUN_SEMANTIC_STRESS = False
SEMANTIC_OUTPUT_DIR = "outputs/semantic_stress"
SEMANTIC_SCENARIOS_FILE = "datasets/semantic_stress_scenarios_v42.csv"
SEMANTIC_ROBUST_SAMPLES = 32
FINAL_VERIFY_SAMPLES_SEMANTIC = None
SEMANTIC_LLM_CANDIDATES = 6
SEMANTIC_RULE_CANDIDATES = 12
SEMANTIC_COMPLIANCE_WEIGHT = 0.35
SEMANTIC_TAIL_WEIGHT = 0.15
SEMANTIC_LLM_TEMPERATURE = 0.30
SEMANTIC_LLM_INCLUDE_RULE_ANCHORS = True
SEMANTIC_ALLOW_RULE_FALLBACK = False


# v4.5 explicit uncertainty model documentation. These values match
# radar_llm_robust.robust.UncertaintySpec defaults.
UNCERTAINTY_SNR_STD_DB = 1.5
UNCERTAINTY_CNR_STD_DB = 2.5
UNCERTAINTY_DOPPLER_LOG_STD = 0.18
UNCERTAINTY_RANGE_SPREAD_LOG_STD = 0.12
UNCERTAINTY_JNR_STD_DB = 2.0
UNCERTAINTY_MAX_RANGE_LOG_STD = 0.05
UNCERTAINTY_DESIRED_RESOLUTION_LOG_STD = 0.05
