from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats


# ============================================================
# Step 6: normality checks
# ============================================================

INPUT_DIR = Path("statistical_analysis_results")

BASELINE_CSV = INPUT_DIR / "baseline_analysis_ready.csv"
SENSITIVITY_CSV = INPUT_DIR / "rs_sensitivity_analysis_ready.csv"

OUTPUT_DIR = INPUT_DIR / "step6_normality"
BASELINE_PLOTS_DIR = OUTPUT_DIR / "baseline_qq_plots"
SENSITIVITY_PLOTS_DIR = OUTPUT_DIR / "sensitivity_qq_plots"

ALPHA = 0.05

METRICS = {
    "Time_optimal_s": "Time",
    "Subtours_optimal": "Subtours",
    "Quality_optimal": "Quality",
}


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
        raise ValueError(f"{label} is missing required columns: {missing}")


def clean_numeric(series):
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return x[np.isfinite(x)]


def safe_filename_component(value):
    text = str(value)
    for old, new in [
        (" ", "_"),
        ("/", "_"),
        ("\\", "_"),
        (":", "_"),
        (".", "p"),
    ]:
        text = text.replace(old, new)
    return text


def normality_statistics(values):
    x = clean_numeric(values)
    n = len(x)

    result = {
        "N_used": n,
        "Shapiro_W": np.nan,
        "Shapiro_p": np.nan,
        "Skewness": np.nan,
        "Excess_kurtosis": np.nan,
        "Normality_at_0.05": "Not tested",
    }

    if n < 3:
        result["Normality_at_0.05"] = "Insufficient N"
        return result

    W, p = stats.shapiro(x)

    result["Shapiro_W"] = float(W)
    result["Shapiro_p"] = float(p)
    result["Skewness"] = float(stats.skew(x, bias=False))
    result["Excess_kurtosis"] = float(
        stats.kurtosis(x, fisher=True, bias=False)
    )

    if p < ALPHA:
        result["Normality_at_0.05"] = "Reject normality"
    else:
        result["Normality_at_0.05"] = "Do not reject normality"

    return result


def save_qq_plot(values, title, output_path):
    x = clean_numeric(values)

    if len(x) < 3:
        return False

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    stats.probplot(x, dist="norm", plot=ax)

    ax.set_title(title)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Ordered observed values")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return True


def summarize_normality(df, group_columns, dataset_label, plots_dir):
    required = group_columns + list(METRICS.keys())
    require_columns(df, required, dataset_label)

    rows = []

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

        group_label = ", ".join(
            f"{column}={value}"
            for column, value in group_info.items()
        )

        filename_prefix = "_".join(
            f"{safe_filename_component(column)}-"
            f"{safe_filename_component(value)}"
            for column, value in group_info.items()
        )

        for source_column, metric_name in METRICS.items():
            stats_row = normality_statistics(group[source_column])

            row = dict(group_info)
            row["Metric"] = metric_name
            row.update(stats_row)
            rows.append(row)

            plot_path = (
                plots_dir
                / f"{filename_prefix}_{metric_name}_qq.png"
            )

            save_qq_plot(
                group[source_column],
                title=f"Q-Q plot: {metric_name}\n{group_label}",
                output_path=plot_path,
            )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            group_columns + ["Metric"]
        ).reset_index(drop=True)

    return result


def make_compact_table(df, group_columns):
    return df[
        group_columns
        + [
            "Metric",
            "N_used",
            "Shapiro_W",
            "Shapiro_p",
            "Skewness",
            "Excess_kurtosis",
            "Normality_at_0.05",
        ]
    ].copy()


def print_summary(title, df):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)

    if df.empty:
        print("No rows.")
        return

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.float_format", lambda x: f"{x:.6f}",
    ):
        print(df.to_string(index=False))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    SENSITIVITY_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline = load_csv(
        BASELINE_CSV,
        "Baseline analysis-ready",
    )

    sensitivity = load_csv(
        SENSITIVITY_CSV,
        "Sensitivity analysis-ready",
    )

    baseline_results = summarize_normality(
        baseline,
        group_columns=["Profile", "SMF", "L"],
        dataset_label="Baseline dataset",
        plots_dir=BASELINE_PLOTS_DIR,
    )

    sensitivity_results = summarize_normality(
        sensitivity,
        group_columns=["Profile", "SMF", "L", "r", "s"],
        dataset_label="Sensitivity dataset",
        plots_dir=SENSITIVITY_PLOTS_DIR,
    )

    baseline_compact = make_compact_table(
        baseline_results,
        ["Profile", "SMF", "L"],
    )

    sensitivity_compact = make_compact_table(
        sensitivity_results,
        ["Profile", "SMF", "L", "r", "s"],
    )

    baseline_results.to_csv(
        OUTPUT_DIR / "baseline_normality_full.csv",
        index=False,
    )

    sensitivity_results.to_csv(
        OUTPUT_DIR / "rs_sensitivity_normality_full.csv",
        index=False,
    )

    baseline_compact.to_csv(
        OUTPUT_DIR / "baseline_normality_paper.csv",
        index=False,
    )

    sensitivity_compact.to_csv(
        OUTPUT_DIR / "rs_sensitivity_normality_paper.csv",
        index=False,
    )

    method_text = f'''Step 6 normality-check protocol

For each experimental configuration, the distributions of:
- optimization time,
- number of subtours eliminated, and
- optimal objective value

were assessed using the Shapiro-Wilk test.

Significance level:
alpha = {ALPHA}

Interpretation:
p < {ALPHA}: reject the null hypothesis of normality.
p >= {ALPHA}: do not reject the null hypothesis of normality.

Skewness and excess kurtosis are also reported as descriptive measures of
distributional shape. A Q-Q plot against the normal distribution is generated
for each configuration and outcome.

Only runs with proven optimality (Status == "OK") are included in these
normality checks. TIMEOUT and ERROR observations are not treated as observed
optimal objective values or solution times.

Failure to reject normality does not prove that a distribution is normal; it
only indicates that the sample does not provide sufficient evidence to reject
the normality assumption.
'''

    with open(
        OUTPUT_DIR / "step6_method.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(method_text)

    print_summary(
        "BASELINE NORMALITY CHECKS",
        baseline_compact,
    )

    print_summary(
        "r/s SENSITIVITY NORMALITY CHECKS",
        sensitivity_compact,
    )

    print("\nFiles created:")
    print(OUTPUT_DIR / "baseline_normality_full.csv")
    print(OUTPUT_DIR / "baseline_normality_paper.csv")
    print(OUTPUT_DIR / "rs_sensitivity_normality_full.csv")
    print(OUTPUT_DIR / "rs_sensitivity_normality_paper.csv")
    print(OUTPUT_DIR / "step6_method.txt")
    print(f"Baseline Q-Q plots: {BASELINE_PLOTS_DIR}/")
    print(f"Sensitivity Q-Q plots: {SENSITIVITY_PLOTS_DIR}/")


if __name__ == "__main__":
    main()
