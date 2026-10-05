from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# Step 8: significance tests for the original L-scaling experiment
# ============================================================
#
# Baseline comparisons are made WITHIN each profile:
#
#   Slow:      L = 32 vs 43
#   Moderate:  L = 32 vs 62 vs 160
#   Fast:      L = 32 vs 62
#
# Outcomes:
#   - optimization time
#   - number of subtours
#   - optimal objective value (quality)
#
# Because the configurations are based on independently sampled candidate
# sets and the distributions may be non-normal, the analysis uses:
#
#   * Mann-Whitney U for two-group profile comparisons
#   * Kruskal-Wallis for Moderate (three L levels)
#   * pairwise Mann-Whitney U after Kruskal-Wallis
#   * Holm correction for the three Moderate pairwise p-values
#
# Effect sizes:
#   * rank-biserial correlation for Mann-Whitney comparisons
#   * epsilon-squared for Kruskal-Wallis
#
# Only proven-optimal runs are used for the three outcome variables.
# TIMEOUT and ERROR counts are reported separately.
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

INPUT_DIR = Path("statistical_analysis_results")
BASELINE_CSV = INPUT_DIR / "baseline_analysis_ready.csv"

OUTPUT_DIR = INPUT_DIR / "step8_significance_L"


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

ALPHA = 0.05

PROFILE_L_VALUES = {
    "Slow": [32, 43],
    "Moderate": [32, 62, 160],
    "Fast": [32, 62],
}

METRICS = {
    "Time_optimal_s": "Time",
    "Subtours_optimal": "Subtours",
    "Quality_optimal": "Quality",
}


# ============================================================
# Helpers
# ============================================================

def load_data():
    if not BASELINE_CSV.exists():
        raise FileNotFoundError(
            f"Baseline analysis-ready file not found: {BASELINE_CSV}\n"
            "Run Step 4 first."
        )

    df = pd.read_csv(BASELINE_CSV)

    required = {
        "Profile",
        "L",
        "Run",
        "Seed",
        "Status",
        *METRICS.keys(),
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Baseline analysis-ready file is missing columns: {missing}"
        )

    return df


def clean_numeric(values):
    """
    Convert either a pandas Series/list or an already-created NumPy array
    to a finite float NumPy array.
    """
    x = np.asarray(
        pd.to_numeric(pd.Series(values), errors="coerce"),
        dtype=float,
    )
    return x[np.isfinite(x)]


def holm_adjust(p_values):
    """
    Holm step-down family-wise error-rate correction.

    Returns adjusted p-values in the original order.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)

    if m == 0:
        return np.array([], dtype=float)

    order = np.argsort(p)
    sorted_p = p[order]

    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0

    for i, p_i in enumerate(sorted_p):
        multiplier = m - i
        adjusted = min(1.0, multiplier * p_i)
        running_max = max(running_max, adjusted)
        adjusted_sorted[i] = running_max

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted

    return adjusted


def rank_biserial_from_u(u_statistic, n1, n2):
    """
    Rank-biserial correlation based on Mann-Whitney U.

    Positive values mean observations in group 1 tend to be larger than
    observations in group 2; negative values mean group 1 tends to be smaller.
    """
    if n1 == 0 or n2 == 0:
        return np.nan

    return float((2.0 * u_statistic) / (n1 * n2) - 1.0)


def epsilon_squared_kruskal(h_statistic, n_total, k_groups):
    """
    Epsilon-squared effect-size estimate for Kruskal-Wallis.
    """
    if n_total <= k_groups:
        return np.nan

    value = (h_statistic - k_groups + 1.0) / (n_total - k_groups)
    return float(max(0.0, value))


def group_descriptives(values):
    x = clean_numeric(values)

    if len(x) == 0:
        return {
            "N": 0,
            "Mean": np.nan,
            "Median": np.nan,
            "SD": np.nan,
        }

    return {
        "N": int(len(x)),
        "Mean": float(np.mean(x)),
        "Median": float(np.median(x)),
        "SD": float(np.std(x, ddof=1)) if len(x) >= 2 else np.nan,
    }


def mann_whitney_result(profile, metric_name, l1, x1, l2, x2):
    n1 = len(x1)
    n2 = len(x2)

    if n1 == 0 or n2 == 0:
        return {
            "Profile": profile,
            "Metric": metric_name,
            "L1": l1,
            "L2": l2,
            "N1": n1,
            "N2": n2,
            "Median1": np.nan if n1 == 0 else float(np.median(x1)),
            "Median2": np.nan if n2 == 0 else float(np.median(x2)),
            "U": np.nan,
            "p_raw": np.nan,
            "p_Holm": np.nan,
            "Rank_biserial": np.nan,
            "Significant_0.05": "Not tested",
        }

    result = stats.mannwhitneyu(
        x1,
        x2,
        alternative="two-sided",
        method="asymptotic",
    )

    u = float(result.statistic)
    p = float(result.pvalue)

    return {
        "Profile": profile,
        "Metric": metric_name,
        "L1": l1,
        "L2": l2,
        "N1": n1,
        "N2": n2,
        "Median1": float(np.median(x1)),
        "Median2": float(np.median(x2)),
        "U": u,
        "p_raw": p,
        "p_Holm": p,  # unchanged for a single comparison
        "Rank_biserial": rank_biserial_from_u(u, n1, n2),
        "Significant_0.05": "Yes" if p < ALPHA else "No",
    }


# ============================================================
# Main analysis
# ============================================================

def run_tests(df):
    omnibus_rows = []
    pairwise_rows = []
    descriptive_rows = []
    status_rows = []

    for profile, l_values in PROFILE_L_VALUES.items():
        profile_df = df[df["Profile"] == profile].copy()

        # Status counts are useful context, especially if a configuration has
        # TIMEOUT observations.
        for L in l_values:
            subset = profile_df[profile_df["L"] == L]
            statuses = subset["Status"].astype(str)

            status_rows.append(
                {
                    "Profile": profile,
                    "L": L,
                    "N_total": int(len(subset)),
                    "N_OK": int((statuses == "OK").sum()),
                    "N_TIMEOUT": int((statuses == "TIMEOUT").sum()),
                    "N_ERROR": int((statuses == "ERROR").sum()),
                }
            )

        for source_column, metric_name in METRICS.items():
            data_by_L = {}

            for L in l_values:
                values = clean_numeric(
                    profile_df.loc[
                        profile_df["L"] == L,
                        source_column,
                    ]
                )
                data_by_L[L] = values

                desc = group_descriptives(values)
                descriptive_rows.append(
                    {
                        "Profile": profile,
                        "Metric": metric_name,
                        "L": L,
                        **desc,
                    }
                )

            # ----------------------------------------------------
            # Two L levels: direct Mann-Whitney U
            # ----------------------------------------------------
            if len(l_values) == 2:
                l1, l2 = l_values

                pairwise_rows.append(
                    mann_whitney_result(
                        profile,
                        metric_name,
                        l1,
                        data_by_L[l1],
                        l2,
                        data_by_L[l2],
                    )
                )

            # ----------------------------------------------------
            # Three L levels: Kruskal-Wallis + post-hoc pairwise
            # ----------------------------------------------------
            elif len(l_values) >= 3:
                groups = [data_by_L[L] for L in l_values]

                if any(len(group) == 0 for group in groups):
                    omnibus_rows.append(
                        {
                            "Profile": profile,
                            "Metric": metric_name,
                            "K_groups": len(groups),
                            "N_total": int(sum(len(g) for g in groups)),
                            "H": np.nan,
                            "p": np.nan,
                            "Epsilon_squared": np.nan,
                            "Significant_0.05": "Not tested",
                        }
                    )
                    continue

                kw = stats.kruskal(*groups)
                h = float(kw.statistic)
                p_kw = float(kw.pvalue)

                omnibus_rows.append(
                    {
                        "Profile": profile,
                        "Metric": metric_name,
                        "K_groups": len(groups),
                        "N_total": int(sum(len(g) for g in groups)),
                        "H": h,
                        "p": p_kw,
                        "Epsilon_squared": epsilon_squared_kruskal(
                            h,
                            sum(len(g) for g in groups),
                            len(groups),
                        ),
                        "Significant_0.05": (
                            "Yes" if p_kw < ALPHA else "No"
                        ),
                    }
                )

                # Pairwise comparisons.
                metric_pairwise = []

                for l1, l2 in combinations(l_values, 2):
                    metric_pairwise.append(
                        mann_whitney_result(
                            profile,
                            metric_name,
                            l1,
                            data_by_L[l1],
                            l2,
                            data_by_L[l2],
                        )
                    )

                raw_ps = [
                    row["p_raw"]
                    for row in metric_pairwise
                ]

                adjusted_ps = holm_adjust(raw_ps)

                for row, p_adj in zip(metric_pairwise, adjusted_ps):
                    row["p_Holm"] = float(p_adj)
                    row["Significant_0.05"] = (
                        "Yes" if p_adj < ALPHA else "No"
                    )
                    pairwise_rows.append(row)

    return (
        pd.DataFrame(descriptive_rows),
        pd.DataFrame(omnibus_rows),
        pd.DataFrame(pairwise_rows),
        pd.DataFrame(status_rows),
    )


def print_table(title, df):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    if df.empty:
        print("No rows.")
        return

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 260,
        "display.float_format", lambda x: f"{x:.6g}",
    ):
        print(df.to_string(index=False))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    (
        descriptives,
        omnibus,
        pairwise,
        status_counts,
    ) = run_tests(df)

    descriptives.to_csv(
        OUTPUT_DIR / "L_descriptives.csv",
        index=False,
    )

    omnibus.to_csv(
        OUTPUT_DIR / "L_kruskal_wallis.csv",
        index=False,
    )

    pairwise.to_csv(
        OUTPUT_DIR / "L_pairwise_mann_whitney.csv",
        index=False,
    )

    status_counts.to_csv(
        OUTPUT_DIR / "L_status_counts.csv",
        index=False,
    )

    method_text = f"""Step 8 significance-testing protocol for candidate-set size L

Tests are conducted separately within each lick-profile category.

Slow:
    L = 32 versus L = 43

Moderate:
    L = 32, 62, and 160

Fast:
    L = 32 versus L = 62

Outcomes:
    - optimization time
    - number of subtours eliminated
    - optimal objective value

Because the samples at different L values are independently generated and
the outcome distributions need not be normal, nonparametric tests are used.

For profiles with two L levels:
    two-sided Mann-Whitney U test

For Moderate, which has three L levels:
    Kruskal-Wallis omnibus test
    followed by all three pairwise two-sided Mann-Whitney U tests

The three Moderate pairwise p-values are corrected using Holm's family-wise
error-rate procedure.

Significance level:
    alpha = {ALPHA}

Effect sizes:
    - pairwise comparisons: rank-biserial correlation
    - Kruskal-Wallis: epsilon-squared

For rank-biserial correlation, a positive value indicates that values in L1
tend to be larger than values in L2, and a negative value indicates that
values in L1 tend to be smaller.

Only runs with Status == "OK" contribute to optimization-time, subtour, and
objective-value tests. TIMEOUT and ERROR observations are counted separately
in L_status_counts.csv.

Therefore, if timeouts occur, runtime comparisons describe the distribution
of time-to-proven-optimality among successfully solved instances rather than
the complete censored runtime distribution.
"""

    with open(
        OUTPUT_DIR / "step8_method.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(method_text)

    print_table(
        "DESCRIPTIVE STATISTICS BY L",
        descriptives,
    )

    print_table(
        "KRUSKAL-WALLIS TESTS (MODERATE PROFILE)",
        omnibus,
    )

    print_table(
        "PAIRWISE MANN-WHITNEY TESTS",
        pairwise,
    )

    print_table(
        "STATUS COUNTS",
        status_counts,
    )

    print("\nFiles created:")
    print(OUTPUT_DIR / "L_descriptives.csv")
    print(OUTPUT_DIR / "L_kruskal_wallis.csv")
    print(OUTPUT_DIR / "L_pairwise_mann_whitney.csv")
    print(OUTPUT_DIR / "L_status_counts.csv")
    print(OUTPUT_DIR / "step8_method.txt")


if __name__ == "__main__":
    main()
