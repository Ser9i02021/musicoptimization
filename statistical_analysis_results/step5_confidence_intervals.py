from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Step 5: standard errors and 95% confidence intervals
# ============================================================
#
# This script uses the analysis-ready files created in Step 4.
#
# For each experimental configuration it reports:
#   - total number of observations
#   - number of OK / TIMEOUT / ERROR observations
#   - mean
#   - sample standard deviation
#   - standard error of the mean
#   - 95% percentile-bootstrap confidence interval for the mean
#   - median
#
# IMPORTANT:
# Quality, optimization time, and subtour statistics are calculated only
# from runs with a proven optimum (Status == "OK"). TIMEOUT and ERROR
# observations are counted separately and are not silently treated as
# successful observations.
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

INPUT_DIR = Path("statistical_analysis_results")

BASELINE_CSV = INPUT_DIR / "baseline_analysis_ready.csv"
SENSITIVITY_CSV = INPUT_DIR / "rs_sensitivity_analysis_ready.csv"

OUTPUT_DIR = INPUT_DIR / "step5_confidence_intervals"


# ------------------------------------------------------------
# Bootstrap configuration
# ------------------------------------------------------------

CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_RESAMPLES = 10_000
RANDOM_SEED = 20261002


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

# These columns were created in Step 4 and are populated only for Status == OK.
METRICS = {
    "Time_optimal_s": "Time_s",
    "Subtours_optimal": "Subtours",
    "Quality_optimal": "Quality",
}


# ============================================================
# Helpers
# ============================================================

def load_csv(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} file not found: {path}\n"
            "Run Step 4 (statistical_analysis.py) first."
        )
    return pd.read_csv(path)


def require_columns(df, columns, label):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}"
        )


def percentile_bootstrap_ci(
    values,
    confidence_level=CONFIDENCE_LEVEL,
    n_resamples=BOOTSTRAP_RESAMPLES,
    rng=None,
):
    """
    Percentile-bootstrap confidence interval for the sample mean.

    Returns (lower, upper). For n < 2, returns (NaN, NaN).
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)
    if n < 2:
        return np.nan, np.nan

    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    # Sample indices with replacement.
    indices = rng.integers(
        low=0,
        high=n,
        size=(n_resamples, n),
    )

    bootstrap_means = x[indices].mean(axis=1)

    alpha = 1.0 - confidence_level
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    lower, upper = np.percentile(
        bootstrap_means,
        [lower_q, upper_q],
    )

    return float(lower), float(upper)


def metric_statistics(values, rng):
    """
    Compute descriptive statistics, SE, and bootstrap CI for one metric.

    Standard error:
        SE = sample SD / sqrt(n)

    The confidence interval is a percentile-bootstrap CI for the mean.
    """
    x = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).to_numpy(dtype=float)

    x = x[np.isfinite(x)]
    n = len(x)

    if n == 0:
        return {
            "N_used": 0,
            "Mean": np.nan,
            "SD": np.nan,
            "SE": np.nan,
            "CI95_low": np.nan,
            "CI95_high": np.nan,
            "Median": np.nan,
        }

    mean = float(np.mean(x))
    median = float(np.median(x))

    if n >= 2:
        sd = float(np.std(x, ddof=1))
        se = float(sd / np.sqrt(n))
        ci_low, ci_high = percentile_bootstrap_ci(
            x,
            rng=rng,
        )
    else:
        sd = np.nan
        se = np.nan
        ci_low = np.nan
        ci_high = np.nan

    return {
        "N_used": n,
        "Mean": mean,
        "SD": sd,
        "SE": se,
        "CI95_low": ci_low,
        "CI95_high": ci_high,
        "Median": median,
    }


def status_counts(group):
    status = group["Status"].astype(str)

    return {
        "N_total": int(len(group)),
        "N_OK": int((status == "OK").sum()),
        "N_TIMEOUT": int((status == "TIMEOUT").sum()),
        "N_ERROR": int((status == "ERROR").sum()),
    }


def summarize_dataset(df, group_columns, dataset_label):
    required = (
        group_columns
        + ["Status"]
        + list(METRICS.keys())
    )
    require_columns(df, required, dataset_label)

    rows = []

    # One deterministic random-number stream makes the full analysis
    # exactly reproducible.
    rng = np.random.default_rng(RANDOM_SEED)

    grouped = df.groupby(
        group_columns,
        dropna=False,
        sort=True,
    )

    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = {
            column: value
            for column, value in zip(group_columns, group_key)
        }

        row.update(status_counts(group))

        for source_column, short_name in METRICS.items():
            stats = metric_statistics(
                group[source_column],
                rng=rng,
            )

            for statistic_name, value in stats.items():
                row[f"{short_name}_{statistic_name}"] = value

        rows.append(row)

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(group_columns).reset_index(drop=True)

    return result


def make_paper_friendly_table(summary_df, group_columns):
    """
    Compact table retaining the quantities most likely to be reported
    in the manuscript.
    """
    output = summary_df[group_columns + [
        "N_total",
        "N_OK",
        "N_TIMEOUT",
        "N_ERROR",
        "Time_s_Mean",
        "Time_s_SE",
        "Time_s_CI95_low",
        "Time_s_CI95_high",
        "Time_s_Median",
        "Subtours_Mean",
        "Subtours_SE",
        "Subtours_CI95_low",
        "Subtours_CI95_high",
        "Subtours_Median",
        "Quality_Mean",
        "Quality_SE",
        "Quality_CI95_low",
        "Quality_CI95_high",
        "Quality_Median",
    ]].copy()

    return output


def print_summary(title, df):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    if df.empty:
        print("No rows.")
        return

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.float_format", lambda x: f"{x:.4f}",
    ):
        print(df.to_string(index=False))


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = load_csv(
        BASELINE_CSV,
        "Baseline analysis-ready",
    )

    sensitivity = load_csv(
        SENSITIVITY_CSV,
        "Sensitivity analysis-ready",
    )

    baseline_summary = summarize_dataset(
        baseline,
        group_columns=["Profile", "SMF", "L"],
        dataset_label="Baseline dataset",
    )

    sensitivity_summary = summarize_dataset(
        sensitivity,
        group_columns=["Profile", "SMF", "L", "r", "s"],
        dataset_label="Sensitivity dataset",
    )

    baseline_paper = make_paper_friendly_table(
        baseline_summary,
        ["Profile", "SMF", "L"],
    )

    sensitivity_paper = make_paper_friendly_table(
        sensitivity_summary,
        ["Profile", "SMF", "L", "r", "s"],
    )

    # Full machine-readable output.
    baseline_summary.to_csv(
        OUTPUT_DIR / "baseline_se_ci_full.csv",
        index=False,
    )

    sensitivity_summary.to_csv(
        OUTPUT_DIR / "rs_sensitivity_se_ci_full.csv",
        index=False,
    )

    # Compact tables for manuscript preparation.
    baseline_paper.to_csv(
        OUTPUT_DIR / "baseline_se_ci_paper.csv",
        index=False,
    )

    sensitivity_paper.to_csv(
        OUTPUT_DIR / "rs_sensitivity_se_ci_paper.csv",
        index=False,
    )

    # Record the statistical protocol in plain text for reproducibility.
    method_text = f"""Step 5 statistical protocol

For each experimental configuration, means, sample standard deviations,
standard errors, medians, and {int(CONFIDENCE_LEVEL * 100)}% confidence
intervals were calculated for optimization time, subtour count, and
objective value.

Standard error:
    SE = sample SD / sqrt(n)

Confidence interval:
    percentile bootstrap confidence interval for the mean
    resamples = {BOOTSTRAP_RESAMPLES}
    confidence level = {CONFIDENCE_LEVEL}
    random seed = {RANDOM_SEED}

Only runs with Status == "OK" (proven optimality) contribute to the
optimization-time, subtour, and objective-value estimates. TIMEOUT and
ERROR observations are counted and reported separately rather than
silently discarded or treated as optimal solutions.
"""

    with open(
        OUTPUT_DIR / "step5_method.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(method_text)

    print_summary(
        "BASELINE: STANDARD ERRORS AND 95% BOOTSTRAP CONFIDENCE INTERVALS",
        baseline_paper,
    )

    print_summary(
        "r/s SENSITIVITY: STANDARD ERRORS AND 95% BOOTSTRAP CONFIDENCE INTERVALS",
        sensitivity_paper,
    )

    print("\nFiles created:")
    print(OUTPUT_DIR / "baseline_se_ci_full.csv")
    print(OUTPUT_DIR / "baseline_se_ci_paper.csv")
    print(OUTPUT_DIR / "rs_sensitivity_se_ci_full.csv")
    print(OUTPUT_DIR / "rs_sensitivity_se_ci_paper.csv")
    print(OUTPUT_DIR / "step5_method.txt")


if __name__ == "__main__":
    main()
