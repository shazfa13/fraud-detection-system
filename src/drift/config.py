"""
Phase 8 configuration — Drift Detection.
===========================================

Every tunable knob Phase 8's own code needs lives here, exactly once (same convention as
src/streaming/config.py). Nothing here changes Model V1, its 34-feature contract, its threshold,
or any Phase 3/5/7 logic — those are read-only inputs to drift detection, never written.

Resolution note: this file resolves the project root from its own location
(src/drift/config.py -> parents[2] is the project root), matching src/streaming/config.py.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---- Existing Phase 1-7 artifacts (READ ONLY - Phase 8 never writes to these) -----------------
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "featured_transactions.parquet"
EXPECTED_FULL_PAYSIM_ROWS = 6_362_620
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model_v1.json"
METADATA_PATH = PROJECT_ROOT / "models" / "model_v1_metadata.json"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "model_v1_threshold.json"
FE_SUMMARY_PATH = PROJECT_ROOT / "reports" / "feature_engineering_summary.json"

# ---- Phase 8 outputs ------------------------------------------------------------------------
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DRIFT_FEATURE_SUMMARY_PATH = REPORTS_DIR / "drift_feature_summary.csv"
DRIFT_REPORT_PATH = REPORTS_DIR / "drift_report.json"
FIG_DRIFT_SUMMARY = FIGURES_DIR / "drift_feature_summary.png"
FIG_TOP_DRIFTED = FIGURES_DIR / "drift_top_features.png"
FIG_DIST_COMPARISON_PREFIX = "drift_distribution_"   # + feature name + ".png"

# ---- Reference / recent window selection -------------------------------------------------------
# Resolved dynamically from the REAL model_v1_metadata.json (see drift_detector.resolve_windows) -
# never hardcoded here, so this stays correct however Model V1 was actually trained.
#   REFERENCE = the chronological TRAINING period (step <= train_end_step): the distribution Model
#               V1 was actually fitted on - the natural, already-existing "baseline".
#   RECENT    = the chronological TEST period (step > validation_end_step): transactions Model V1
#               never trained on, the same period Phase 6/7 already treat as "live/recent" data.
#   EXCLUDED  = the validation period in between (train_end_step < step <= validation_end_step) is
#               deliberately excluded from BOTH windows, so reference and recent never overlap and
#               are never mixed - a clean gap, not a random split.

# ---- PSI (Population Stability Index) ---------------------------------------------------------
PSI_BINS = 10                 # deciles of the REFERENCE distribution (continuous features)
PSI_EPSILON = 1e-6            # floor for zero expected/actual bin proportions (avoids log(0) / div-by-0)
PSI_WATCH_THRESHOLD = 0.10    # PSI < 0.10            -> little/no evidence of meaningful drift
PSI_SIGNIFICANT_THRESHOLD = 0.20   # 0.10 <= PSI < 0.20 -> moderate/watch; PSI >= 0.20 -> significant
# These bands are the commonly cited industry convention for PSI monitoring, used here as a
# configurable, documented decision rule - NOT an absolute statistical law.

# ---- KS test (continuous features only) -------------------------------------------------------
KS_ALPHA = 0.05                       # significance level BEFORE multiple-testing correction
MULTIPLE_TESTING_METHOD = "benjamini_hochberg"   # FDR control across all features tested together

# ---- Feature-level eligibility -----------------------------------------------------------------
MIN_SAMPLE_SIZE = 30           # minimum non-NaN observations required, in EACH window, to test a feature
# Features are classified SAFE/POTENTIAL_RISK candidates already screened by Phase 3's leakage audit;
# Phase 8 additionally excludes the label and pure metadata columns explicitly (see drift_detector.
# EXCLUDED_FROM_DRIFT) regardless of what else is present in the source data.
EXCLUDED_FROM_DRIFT = {"isFraud", "event_id", "step"}

# ---- Overall drift decision --------------------------------------------------------------------
# overall_drift_rate = n_features_with_drift / n_features_tested (features with an insufficient
# sample are excluded from BOTH the numerator and the denominator - documented, not silently guessed).
OVERALL_DRIFT_RATE_THRESHOLD = 0.15   # >15% of TESTED features drifting -> overall DRIFT DETECTED = YES
# Also a configurable monitoring convention, not an absolute law - see the notebook's Section 11.

# ---- Optional controlled demonstration (Section 13 of the master prompt) -----------------------
CONTROLLED_DEMO_SHIFT_MULTIPLIER = 3.0   # synthetic amount-inflation factor, demonstration ONLY
