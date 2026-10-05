from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Step 7: outlier analysis
# ============================================================
#
# This script uses the analysis-ready files created in Step 4.
#
# For each configuration and outcome, it applies Tukey's 1.5 x IQR rule:
#
#   lower fence = Q1 - 1.5 * IQR
#   upper fence = Q3 + 1.5 * IQR
#
# It reports:
#   - N used
#   - Q1
#   - median
#   - Q3
#   - IQR
#   - lower and upper fences
#   - number and percentage of outliers
#   - minimum and maximum observed values
#   - minimum and maximum outlier values, when present
#
# IMPORTANT:
# Outliers are IDENTIFIED, not deleted.
# TIMEOUT and ERROR observations are not treated as outliers because they are
# different termination outcomes rather than unusually large/small proven-optimal
# observations.
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

INPUT_DIR = Path("statistical_analysis_results")

BASELINE_CSV = INPUT_DIR / "baseline_analysis_ready.csv"
SENSITIVITY_CSV = INPUT_DIR / "rs_sensitivity_analysis_ready.csv"

OUTPUT_DIR = INPUT_DIR / "step7_outliers"
PLOTS_DIR = OUTPUT_DIR / "boxplots"


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

METRICS = {
    "Time_optimal_s": "Time",
    "Subtours_optimal": "Subtours",
    "Quality_optimal": "Quality",
}

IQR_MULTIPLIER = 1.5


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


def clean_numeric(series):
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return x[np.isfinite(x)]


def tukey_outlier_statistics(values):
    x = clean_numeric(values)
    n = len(x)

    result = {
        "N_used": n,
        "Q1": np.nan,
        "Median": np.nan,
        "Q3": np.nan,
        "IQR": np.nan,
        "Lower_fence": np.nan,
        "Upper_fence": np.nan,
        "N_outliers": 0,
        "Outlier_percent": np.nan,
        "Observed_min": np.nan,
        "Observed_max": np.nan,
        "Outlier_min": np.nan,
        "Outlier_max": np.nan,
    }

    if n == 0:
        return result, np.array([], dtype=float)

    q1 = float(np.percentile(x, 25))
    median = float(np.percentile(x, 50))
    q3 = float(np.percentile(x, 75))
    iqr = q3 - q1

    lower_fence = q1 - IQR_MULTIPLIER * iqr
    upper_fence = q3 + IQR_MULTIPLIER * iqr

    mask = (x < lower_fence) | (x > upper_fence)
    outliers = x[mask]

    result.update(
        {
            "Q1": q1,
            "Median": median,
            "Q3": q3,
            "IQR": iqr,
            "Lower_fence": float(lower_fence),
            "Upper_fence": float(upper_fence),
            "N_outliers": int(len(outliers)),
            "Outlier_percent": float(100.0 * len(outliers) / n),
            "Observed_min": float(np.min(x)),
            "Observed_max": float(np.max(x)),
            "Outlier_min": (
                float(np.min(outliers))
                if len(outliers) > 0
                else np.nan
            ),
            "Outlier_max": (
                float(np.max(outliers))
                if len(outliers) > 0
                else np.nan
            ),
        }
    )

    return result, outliers


def analyze_dataset(
    df,
    group_columns,
    dataset_label,
):
    required = (
        group_columns
        + ["Run", "Seed", "Status"]
        + list(METRICS.keys())
    )
    require_columns(df, required, dataset_label)

    summary_rows = []
    outlier_rows = []

    grouped = df.groupby(
        group_columns,
        dropna=False,
        sort=True,
    )

    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_info = {
            column: value
            for column, value in zip(group_columns, group_key)
        }

        for source_column, metric_name in METRICS.items():
            numeric = pd.to_numeric(
                group[source_column],
                errors="coerce",
            )

            valid_mask = numeric.notna()
            valid_values = numeric.loc[valid_mask]

            stats_row, _ = tukey_outlier_statistics(valid_values)

            row = dict(group_info)
            row["Metric"] = metric_name
            row.update(stats_row)
            summary_rows.append(row)

            if stats_row["N_used"] == 0:
                continue

            lower_fence = stats_row["Lower_fence"]
            upper_fence = stats_row["Upper_fence"]

            outlier_mask = valid_mask & (
                (numeric < lower_fence)
                | (numeric > upper_fence)
            )

            if outlier_mask.any():
                for idx in group.index[outlier_mask]:
                    outlier_row = dict(group_info)
                    outlier_row.update(
                        {
                            "Metric": metric_name,
                            "Run": group.loc[idx, "Run"],
                            "Seed": group.loc[idx, "Seed"],
                            "Status": group.loc[idx, "Status"],
                            "Value": float(numeric.loc[idx]),
                            "Lower_fence": lower_fence,
                            "Upper_fence": upper_fence,
                            "Direction": (
                                "Low"
                                if numeric.loc[idx] < lower_fence
                                else "High"
                            ),
                        }
                    )
                    outlier_rows.append(outlier_row)

    summary_df = pd.DataFrame(summary_rows)
    outliers_df = pd.DataFrame(outlier_rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            group_columns + ["Metric"]
        ).reset_index(drop=True)

    if not outliers_df.empty:
        outliers_df = outliers_df.sort_values(
            group_columns + ["Metric", "Run"]
        ).reset_index(drop=True)

    return summary_df, outliers_df


def make_compact_table(df, group_columns):
    return df[
        group_columns
        + [
            "Metric",
            "N_used",
            "Q1",
            "Median",
            "Q3",
            "IQR",
            "Lower_fence",
            "Upper_fence",
            "N_outliers",
            "Outlier_percent",
            "Outlier_min",
            "Outlier_max",
        ]
    ].copy()


def save_boxplots(
    df,
    group_columns,
    dataset_name,
):
    """
    Save one boxplot per metric.

    Compatible with current Matplotlib versions, which use tick_labels
    rather than the older labels keyword.
    """
    for source_column, metric_name in METRICS.items():
        data = []
        tick_labels = []

        grouped = df.groupby(
            group_columns,
            dropna=False,
            sort=True,
        )

        for group_key, group in grouped:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            values = clean_numeric(group[source_column])

            if len(values) == 0:
                continue

            info = {
                column: value
                for column, value in zip(group_columns, group_key)
            }

            data.append(values)
            tick_labels.append(
                "\n".join(
                    f"{column}={value}"
                    for column, value in info.items()
                )
            )

        if not data:
            continue

        fig = plt.figure(
            figsize=(max(8, 1.8 * len(data)), 6)
        )
        ax = fig.add_subplot(111)

        ax.boxplot(
            data,
            tick_labels=tick_labels,
            showfliers=True,
        )

        ax.set_title(
            f"{dataset_name}: {metric_name} — Tukey boxplots"
        )
        ax.set_ylabel(metric_name)
        ax.tick_params(axis="x", labelrotation=45)

        fig.tight_layout()

        output_path = (
            PLOTS_DIR
            / f"{dataset_name.lower()}_{metric_name.lower()}_boxplot.png"
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def print_summary(title, df):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    if df.empty:
        print("No rows.")
        return

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 250,
        "display.float_format", lambda x: f"{x:.4f}",
    ):
        print(df.to_string(index=False))


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline = load_csv(
        BASELINE_CSV,
        "Baseline analysis-ready",
    )

    sensitivity = load_csv(
        SENSITIVITY_CSV,
        "Sensitivity analysis-ready",
    )

    baseline_summary, baseline_outliers = analyze_dataset(
        baseline,
        group_columns=["Profile", "SMF", "L"],
        dataset_label="Baseline dataset",
    )

    sensitivity_summary, sensitivity_outliers = analyze_dataset(
        sensitivity,
        group_columns=["Profile", "SMF", "L", "r", "s"],
        dataset_label="Sensitivity dataset",
    )

    baseline_paper = make_compact_table(
        baseline_summary,
        ["Profile", "SMF", "L"],
    )

    sensitivity_paper = make_compact_table(
        sensitivity_summary,
        ["Profile", "SMF", "L", "r", "s"],
    )

    baseline_summary.to_csv(
        OUTPUT_DIR / "baseline_outlier_summary_full.csv",
        index=False,
    )

    baseline_paper.to_csv(
        OUTPUT_DIR / "baseline_outlier_summary_paper.csv",
        index=False,
    )

    baseline_outliers.to_csv(
        OUTPUT_DIR / "baseline_outlier_observations.csv",
        index=False,
    )

    sensitivity_summary.to_csv(
        OUTPUT_DIR / "rs_sensitivity_outlier_summary_full.csv",
        index=False,
    )

    sensitivity_paper.to_csv(
        OUTPUT_DIR / "rs_sensitivity_outlier_summary_paper.csv",
        index=False,
    )

    sensitivity_outliers.to_csv(
        OUTPUT_DIR / "rs_sensitivity_outlier_observations.csv",
        index=False,
    )

    save_boxplots(
        baseline,
        group_columns=["Profile", "L"],
        dataset_name="Baseline",
    )

    save_boxplots(
        sensitivity,
        group_columns=["r", "s"],
        dataset_name="Sensitivity",
    )

    method_text = f"""Step 7 outlier-analysis protocol

Outliers were identified separately within each experimental configuration
and outcome using Tukey's {IQR_MULTIPLIER} x IQR rule:

    IQR = Q3 - Q1
    lower fence = Q1 - {IQR_MULTIPLIER} * IQR
    upper fence = Q3 + {IQR_MULTIPLIER} * IQR

An observation below the lower fence or above the upper fence was flagged as
an outlier.

The analysis was applied to:
    - optimization time,
    - number of subtours eliminated, and
    - optimal objective value.

Only runs with proven optimality contribute to these distributions because
the Step 4 analysis-ready columns contain values only for Status == "OK".

TIMEOUT and ERROR observations are not classified as statistical outliers.
They are separate computational outcomes and should be reported separately.

Outliers are NOT removed from the principal analysis. Their number and
percentage are reported to characterize distributional heterogeneity. Any
analysis excluding them should be described explicitly as a secondary
robustness analysis rather than as the primary result.
"""

    with open(
        OUTPUT_DIR / "step7_method.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(method_text)

    print_summary(
        "BASELINE OUTLIER ANALYSIS",
        baseline_paper,
    )

    print_summary(
        "r/s SENSITIVITY OUTLIER ANALYSIS",
        sensitivity_paper,
    )

    print("\nFiles created:")
    print(OUTPUT_DIR / "baseline_outlier_summary_full.csv")
    print(OUTPUT_DIR / "baseline_outlier_summary_paper.csv")
    print(OUTPUT_DIR / "baseline_outlier_observations.csv")
    print(OUTPUT_DIR / "rs_sensitivity_outlier_summary_full.csv")
    print(OUTPUT_DIR / "rs_sensitivity_outlier_summary_paper.csv")
    print(OUTPUT_DIR / "rs_sensitivity_outlier_observations.csv")
    print(OUTPUT_DIR / "step7_method.txt")
    print(f"Boxplots: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
