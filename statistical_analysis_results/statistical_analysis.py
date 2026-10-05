from pathlib import Path

import pandas as pd


# ============================================================
# Statistical analysis: data preparation layer
# ============================================================
#
# Step 4 of the revision workflow.
#
# This script does NOT yet perform:
#   - confidence intervals / standard errors
#   - normality tests
#   - outlier analysis
#   - significance tests
#
# Those are added in the next steps.
#
# Its purpose is to create one clean, validated analysis dataset from:
#   1. the main 700-run experiment
#   2. the paired r/s sensitivity experiment
#
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

# Change this if your definitive 700-run output directory has another name.
BASELINE_RESULTS_CSV = Path("experiment_results") / "run_status.csv"

# Output produced by run_rs_sensitivity.py
SENSITIVITY_RESULTS_CSV = (
    Path("rs_sensitivity_results") / "rs_sensitivity_all_runs.csv"
)

OUTPUT_DIR = Path("statistical_analysis_results")


# ------------------------------------------------------------
# Expected design
# ------------------------------------------------------------

EXPECTED_BASELINE_CONFIGURATIONS = {
    ("Slow", 32),
    ("Slow", 43),
    ("Moderate", 32),
    ("Moderate", 62),
    ("Moderate", 160),
    ("Fast", 32),
    ("Fast", 62),
}

EXPECTED_RS_SETTINGS = {
    (1, 3),
    (2, 3),
    (3, 3),
    (1, 4),
    (1, 5),
}

VALID_STATUSES = {"OK", "TIMEOUT", "ERROR"}


# ============================================================
# Utility functions
# ============================================================

def require_columns(df, required, dataset_name):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}"
        )


def coerce_numeric(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def validate_status_values(df, dataset_name):
    observed = set(df["Status"].dropna().astype(str))
    unexpected = observed - VALID_STATUSES

    if unexpected:
        raise ValueError(
            f"{dataset_name} contains unexpected status values: "
            f"{sorted(unexpected)}"
        )


# ============================================================
# Baseline 700-run data
# ============================================================

def load_baseline_results():
    if not BASELINE_RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Baseline results not found: {BASELINE_RESULTS_CSV}\n"
            "Set BASELINE_RESULTS_CSV to the run_status.csv file from "
            "your definitive experiment."
        )

    df = pd.read_csv(BASELINE_RESULTS_CSV)

    required = [
        "Perfil",
        "SMF",
        "L",
        "Run",
        "Seed",
        "Status",
        "Tempo (s)",
        "Wall time (s)",
        "Número de subciclos",
        "Qualidade",
    ]
    require_columns(df, required, "Baseline results")

    # Standardize names for the statistical scripts that follow.
    df = df.rename(
        columns={
            "Perfil": "Profile",
            "Tempo (s)": "Time_s",
            "Wall time (s)": "Wall_time_s",
            "Número de subciclos": "Subtours",
            "Qualidade": "Quality",
        }
    )

    df = coerce_numeric(
        df,
        [
            "SMF",
            "L",
            "Run",
            "Seed",
            "Time_s",
            "Wall_time_s",
            "Subtours",
            "Quality",
        ],
    )

    validate_status_values(df, "Baseline results")

    # Baseline values used in the submitted model.
    df["r"] = 1
    df["s"] = 3

    # Useful flags for all later analyses.
    df["Solved"] = df["Status"].eq("OK")
    df["TimedOut"] = df["Status"].eq("TIMEOUT")
    df["Errored"] = df["Status"].eq("ERROR")

    # Quality, solver time and subtour count are meaningful for proven-optimal
    # observations only. Keep the original columns, but make this explicit in
    # dedicated analysis columns.
    df["Quality_optimal"] = df["Quality"].where(df["Solved"])
    df["Time_optimal_s"] = df["Time_s"].where(df["Solved"])
    df["Subtours_optimal"] = df["Subtours"].where(df["Solved"])

    # Wall time remains available for every termination outcome.
    # This will later be useful when handling timeouts explicitly.
    df["Termination_time_s"] = df["Wall_time_s"]

    # Check for duplicated run identifiers.
    duplicate_mask = df.duplicated(
        subset=["Profile", "SMF", "L", "Run"],
        keep=False,
    )
    if duplicate_mask.any():
        duplicates = df.loc[
            duplicate_mask,
            ["Profile", "SMF", "L", "Run", "Seed", "Status"],
        ]
        raise ValueError(
            "Duplicate baseline run identifiers detected:\n"
            + duplicates.to_string(index=False)
        )

    return df


# ============================================================
# Paired r/s sensitivity data
# ============================================================

def load_sensitivity_results():
    if not SENSITIVITY_RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Sensitivity results not found: {SENSITIVITY_RESULTS_CSV}\n"
            "Run run_rs_sensitivity.py first, or update "
            "SENSITIVITY_RESULTS_CSV."
        )

    df = pd.read_csv(SENSITIVITY_RESULTS_CSV)

    required = [
        "Profile",
        "SMF",
        "L",
        "b",
        "Run",
        "Seed",
        "r",
        "s",
        "Status",
        "Quality",
        "Time_s",
        "Wall_time_s",
        "Subtours",
    ]
    require_columns(df, required, "Sensitivity results")

    df = coerce_numeric(
        df,
        [
            "SMF",
            "L",
            "b",
            "Run",
            "Seed",
            "r",
            "s",
            "Quality",
            "Time_s",
            "Wall_time_s",
            "Subtours",
        ],
    )

    validate_status_values(df, "Sensitivity results")

    df["Solved"] = df["Status"].eq("OK")
    df["TimedOut"] = df["Status"].eq("TIMEOUT")
    df["Errored"] = df["Status"].eq("ERROR")

    df["Quality_optimal"] = df["Quality"].where(df["Solved"])
    df["Time_optimal_s"] = df["Time_s"].where(df["Solved"])
    df["Subtours_optimal"] = df["Subtours"].where(df["Solved"])
    df["Termination_time_s"] = df["Wall_time_s"]

    duplicate_mask = df.duplicated(
        subset=["Profile", "SMF", "L", "Run", "r", "s"],
        keep=False,
    )
    if duplicate_mask.any():
        duplicates = df.loc[
            duplicate_mask,
            ["Profile", "SMF", "L", "Run", "Seed", "r", "s", "Status"],
        ]
        raise ValueError(
            "Duplicate sensitivity run identifiers detected:\n"
            + duplicates.to_string(index=False)
        )

    return df


# ============================================================
# Design checks
# ============================================================

def check_baseline_design(df):
    observed = set(
        zip(
            df["Profile"].astype(str),
            df["L"].astype(int),
        )
    )

    missing = EXPECTED_BASELINE_CONFIGURATIONS - observed
    extra = observed - EXPECTED_BASELINE_CONFIGURATIONS

    if missing:
        print(
            "WARNING: baseline configurations missing:",
            sorted(missing),
        )

    if extra:
        print(
            "WARNING: unexpected baseline configurations present:",
            sorted(extra),
        )


def check_sensitivity_design(df):
    observed = set(
        zip(
            df["r"].astype(int),
            df["s"].astype(int),
        )
    )

    missing = EXPECTED_RS_SETTINGS - observed
    extra = observed - EXPECTED_RS_SETTINGS

    if missing:
        print(
            "WARNING: sensitivity settings missing:",
            sorted(missing),
        )

    if extra:
        print(
            "WARNING: unexpected sensitivity settings present:",
            sorted(extra),
        )

    # Pairing check:
    # each Run/Seed should ideally appear once for every requested (r,s).
    expected_per_instance = len(EXPECTED_RS_SETTINGS)

    pairing = (
        df.groupby(["Run", "Seed"], dropna=False)
        .size()
        .rename("N_settings")
        .reset_index()
    )

    incomplete = pairing[pairing["N_settings"] != expected_per_instance]

    if not incomplete.empty:
        print(
            "WARNING: some sensitivity instances do not contain all "
            f"{expected_per_instance} r/s settings."
        )
        print(incomplete.to_string(index=False))


# ============================================================
# Simple inventory tables
# ============================================================

def baseline_inventory(df):
    return (
        df.groupby(["Profile", "SMF", "L", "Status"], dropna=False)
        .size()
        .rename("N")
        .reset_index()
        .sort_values(["Profile", "L", "Status"])
    )


def sensitivity_inventory(df):
    return (
        df.groupby(["Profile", "L", "r", "s", "Status"], dropna=False)
        .size()
        .rename("N")
        .reset_index()
        .sort_values(["r", "s", "Status"])
    )


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    baseline = load_baseline_results()
    sensitivity = load_sensitivity_results()

    check_baseline_design(baseline)
    check_sensitivity_design(sensitivity)

    baseline_counts = baseline_inventory(baseline)
    sensitivity_counts = sensitivity_inventory(sensitivity)

    baseline.to_csv(
        OUTPUT_DIR / "baseline_analysis_ready.csv",
        index=False,
    )

    sensitivity.to_csv(
        OUTPUT_DIR / "rs_sensitivity_analysis_ready.csv",
        index=False,
    )

    baseline_counts.to_csv(
        OUTPUT_DIR / "baseline_status_counts.csv",
        index=False,
    )

    sensitivity_counts.to_csv(
        OUTPUT_DIR / "rs_sensitivity_status_counts.csv",
        index=False,
    )

    print("\n" + "=" * 78)
    print("BASELINE DATA")
    print("=" * 78)
    print(f"Rows: {len(baseline)}")
    print(
        baseline_counts.to_string(index=False)
    )

    print("\n" + "=" * 78)
    print("PAIRED r/s SENSITIVITY DATA")
    print("=" * 78)
    print(f"Rows: {len(sensitivity)}")
    print(
        sensitivity_counts.to_string(index=False)
    )

    print("\n" + "=" * 78)
    print("ANALYSIS-READY FILES CREATED")
    print("=" * 78)
    print(OUTPUT_DIR / "baseline_analysis_ready.csv")
    print(OUTPUT_DIR / "rs_sensitivity_analysis_ready.csv")
    print(OUTPUT_DIR / "baseline_status_counts.csv")
    print(OUTPUT_DIR / "rs_sensitivity_status_counts.csv")


if __name__ == "__main__":
    main()
