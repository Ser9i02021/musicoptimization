from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# Step 9: paired significance tests for r/s sensitivity
# ============================================================
#
# The same candidate instance is solved repeatedly under different
# parameter settings, so these comparisons are PAIRED.
#
# r sensitivity (s fixed at 3):
#     (r, s) = (1,3), (2,3), (3,3)
#
# s sensitivity (r fixed at 1):
#     (r, s) = (1,3), (1,4), (1,5)
#
# Outcomes:
#     - optimization time
#     - number of subtours
#     - optimal objective value (quality)
#
# Tests:
#     - Friedman omnibus test across the three paired settings
#     - pairwise Wilcoxon signed-rank tests
#     - Holm correction within each metric/family
#
# Effect sizes:
#     - Kendall's W for Friedman
#     - matched-pairs rank-biserial correlation for Wilcoxon
#
# IMPORTANT:
# Only complete paired instances with proven-optimal values under all
# three settings in the relevant family are used in inferential tests.
# TIMEOUT and ERROR outcomes are counted separately.
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

INPUT_DIR = Path("statistical_analysis_results")
SENSITIVITY_CSV = INPUT_DIR / "rs_sensitivity_analysis_ready.csv"

OUTPUT_DIR = INPUT_DIR / "step9_significance_rs"


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

ALPHA = 0.05

R_SETTINGS = [(1, 3), (2, 3), (3, 3)]
S_SETTINGS = [(1, 3), (1, 4), (1, 5)]

METRICS = {
    "Time_optimal_s": "Time",
    "Subtours_optimal": "Subtours",
    "Quality_optimal": "Quality",
}


# ============================================================
# Utility functions
# ============================================================

def load_data():
    if not SENSITIVITY_CSV.exists():
        raise FileNotFoundError(
            f"Sensitivity analysis-ready file not found: {SENSITIVITY_CSV}\n"
            "Run Step 4 first."
        )

    df = pd.read_csv(SENSITIVITY_CSV)

    required = {
        "Profile",
        "SMF",
        "L",
        "Run",
        "Seed",
        "r",
        "s",
        "Status",
        *METRICS.keys(),
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Sensitivity analysis-ready file is missing columns: {missing}"
        )

    return df


def holm_adjust(p_values):
    """
    Holm step-down family-wise error-rate correction.
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
        adjusted = min(1.0, (m - i) * p_i)
        running_max = max(running_max, adjusted)
        adjusted_sorted[i] = running_max

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted

    return adjusted


def setting_label(setting):
    r, s = setting
    return f"r={r},s={s}"


def kendalls_w_from_friedman(chi2, n, k):
    """
    Kendall's W for a Friedman test:
        W = chi-square / [n * (k - 1)]
    """
    if n <= 0 or k <= 1:
        return np.nan

    return float(chi2 / (n * (k - 1)))


def matched_rank_biserial(x1, x2):
    """
    Matched-pairs rank-biserial correlation.

    Difference is defined as:
        d = x2 - x1

    Positive effect:
        values in setting 2 tend to be larger.

    Negative effect:
        values in setting 2 tend to be smaller.

    Zero differences are omitted, matching Wilcoxon's default
    zero_method='wilcox'.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    d = x2 - x1
    d = d[np.isfinite(d)]
    d = d[d != 0]

    if len(d) == 0:
        return 0.0

    ranks = stats.rankdata(np.abs(d), method="average")

    w_pos = float(ranks[d > 0].sum())
    w_neg = float(ranks[d < 0].sum())

    denom = w_pos + w_neg
    if denom == 0:
        return 0.0

    return float((w_pos - w_neg) / denom)


def safe_wilcoxon(x1, x2):
    """
    Two-sided Wilcoxon signed-rank test with graceful handling when
    all paired differences are exactly zero.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    d = x2 - x1

    if np.all(d == 0):
        return 0.0, 1.0

    result = stats.wilcoxon(
        x1,
        x2,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method="auto",
    )

    return float(result.statistic), float(result.pvalue)


def describe(values):
    x = np.asarray(values, dtype=float)

    return {
        "N": int(len(x)),
        "Mean": float(np.mean(x)) if len(x) else np.nan,
        "Median": float(np.median(x)) if len(x) else np.nan,
        "SD": (
            float(np.std(x, ddof=1))
            if len(x) >= 2
            else np.nan
        ),
    }


# ============================================================
# Data preparation for one paired family
# ============================================================

def prepare_family(df, settings, metric_column):
    """
    Return a complete-case wide table for one family and metric.

    Every returned row corresponds to one candidate instance identified by
    (Run, Seed), and contains a proven-optimal value for every setting.
    """
    family = df[
        df.apply(
            lambda row: (int(row["r"]), int(row["s"])) in settings,
            axis=1,
        )
    ].copy()

    family["Setting"] = family.apply(
        lambda row: setting_label(
            (int(row["r"]), int(row["s"]))
        ),
        axis=1,
    )

    # There should be at most one observation per instance/setting.
    duplicates = family.duplicated(
        subset=["Run", "Seed", "Setting"],
        keep=False,
    )

    if duplicates.any():
        bad = family.loc[
            duplicates,
            ["Run", "Seed", "r", "s", "Status"],
        ]
        raise ValueError(
            "Duplicate sensitivity observations detected:\n"
            + bad.to_string(index=False)
        )

    wide = family.pivot(
        index=["Run", "Seed"],
        columns="Setting",
        values=metric_column,
    )

    expected_columns = [setting_label(x) for x in settings]

    for column in expected_columns:
        if column not in wide.columns:
            wide[column] = np.nan

    wide = wide[expected_columns]

    complete = wide.dropna(axis=0, how="any").copy()

    return family, wide, complete


# ============================================================
# Status summaries
# ============================================================

def make_status_counts(df):
    rows = []

    grouped = df.groupby(
        ["r", "s"],
        dropna=False,
        sort=True,
    )

    for (r, s), group in grouped:
        statuses = group["Status"].astype(str)

        rows.append(
            {
                "r": int(r),
                "s": int(s),
                "N_total": int(len(group)),
                "N_OK": int((statuses == "OK").sum()),
                "N_TIMEOUT": int((statuses == "TIMEOUT").sum()),
                "N_ERROR": int((statuses == "ERROR").sum()),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Statistical analysis for one family
# ============================================================

def analyze_family(df, family_name, settings):
    omnibus_rows = []
    pairwise_rows = []
    descriptive_rows = []
    completeness_rows = []

    setting_names = [setting_label(x) for x in settings]

    for metric_column, metric_name in METRICS.items():
        family_df, wide, complete = prepare_family(
            df,
            settings,
            metric_column,
        )

        n_available = len(wide)
        n_complete = len(complete)

        completeness_rows.append(
            {
                "Family": family_name,
                "Metric": metric_name,
                "N_instances_available": int(n_available),
                "N_complete_pairs": int(n_complete),
                "N_excluded_incomplete": int(n_available - n_complete),
            }
        )

        # Descriptives based on exactly the same complete paired sample
        # used by the inferential tests.
        for setting in setting_names:
            desc = describe(
                complete[setting].to_numpy(dtype=float)
                if n_complete
                else np.array([], dtype=float)
            )

            r_value, s_value = map(
                int,
                setting.replace("r=", "").replace("s=", "").split(","),
            )

            descriptive_rows.append(
                {
                    "Family": family_name,
                    "Metric": metric_name,
                    "r": r_value,
                    "s": s_value,
                    **desc,
                }
            )

        if n_complete < 2:
            omnibus_rows.append(
                {
                    "Family": family_name,
                    "Metric": metric_name,
                    "K_settings": len(settings),
                    "N_complete_pairs": n_complete,
                    "Friedman_chi2": np.nan,
                    "p": np.nan,
                    "Kendalls_W": np.nan,
                    "Significant_0.05": "Not tested",
                }
            )
            continue

        arrays = [
            complete[setting].to_numpy(dtype=float)
            for setting in setting_names
        ]

        # Friedman can fail when every value is identical across all
        # settings. In that degenerate case there is no evidence of a
        # difference, so report chi-square=0 and p=1.
        try:
            friedman = stats.friedmanchisquare(*arrays)
            chi2 = float(friedman.statistic)
            p_friedman = float(friedman.pvalue)
        except ValueError:
            stacked = np.column_stack(arrays)
            if np.all(stacked == stacked[:, [0]]):
                chi2 = 0.0
                p_friedman = 1.0
            else:
                raise

        omnibus_rows.append(
            {
                "Family": family_name,
                "Metric": metric_name,
                "K_settings": len(settings),
                "N_complete_pairs": n_complete,
                "Friedman_chi2": chi2,
                "p": p_friedman,
                "Kendalls_W": kendalls_w_from_friedman(
                    chi2,
                    n_complete,
                    len(settings),
                ),
                "Significant_0.05": (
                    "Yes" if p_friedman < ALPHA else "No"
                ),
            }
        )

        metric_pairwise = []

        for setting1, setting2 in combinations(setting_names, 2):
            x1 = complete[setting1].to_numpy(dtype=float)
            x2 = complete[setting2].to_numpy(dtype=float)

            statistic, p_raw = safe_wilcoxon(x1, x2)

            delta = x2 - x1

            metric_pairwise.append(
                {
                    "Family": family_name,
                    "Metric": metric_name,
                    "Setting1": setting1,
                    "Setting2": setting2,
                    "N_pairs": int(len(delta)),
                    "Median1": float(np.median(x1)),
                    "Median2": float(np.median(x2)),
                    "Median_delta_2_minus_1": float(
                        np.median(delta)
                    ),
                    "Mean_delta_2_minus_1": float(
                        np.mean(delta)
                    ),
                    "Wilcoxon_statistic": statistic,
                    "p_raw": p_raw,
                    "p_Holm": np.nan,
                    "Rank_biserial_2_minus_1":
                        matched_rank_biserial(x1, x2),
                    "Significant_0.05": "Pending",
                }
            )

        adjusted = holm_adjust(
            [row["p_raw"] for row in metric_pairwise]
        )

        for row, p_adj in zip(metric_pairwise, adjusted):
            row["p_Holm"] = float(p_adj)
            row["Significant_0.05"] = (
                "Yes" if p_adj < ALPHA else "No"
            )
            pairwise_rows.append(row)

    return (
        pd.DataFrame(descriptive_rows),
        pd.DataFrame(omnibus_rows),
        pd.DataFrame(pairwise_rows),
        pd.DataFrame(completeness_rows),
    )


# ============================================================
# Output
# ============================================================

def print_table(title, df):
    print("\n" + "=" * 125)
    print(title)
    print("=" * 125)

    if df.empty:
        print("No rows.")
        return

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 280,
        "display.float_format", lambda x: f"{x:.6g}",
    ):
        print(df.to_string(index=False))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    status_counts = make_status_counts(df)

    (
        r_desc,
        r_omnibus,
        r_pairwise,
        r_complete,
    ) = analyze_family(
        df,
        family_name="r sensitivity (s=3)",
        settings=R_SETTINGS,
    )

    (
        s_desc,
        s_omnibus,
        s_pairwise,
        s_complete,
    ) = analyze_family(
        df,
        family_name="s sensitivity (r=1)",
        settings=S_SETTINGS,
    )

    descriptives = pd.concat(
        [r_desc, s_desc],
        ignore_index=True,
    )

    omnibus = pd.concat(
        [r_omnibus, s_omnibus],
        ignore_index=True,
    )

    pairwise = pd.concat(
        [r_pairwise, s_pairwise],
        ignore_index=True,
    )

    completeness = pd.concat(
        [r_complete, s_complete],
        ignore_index=True,
    )

    descriptives.to_csv(
        OUTPUT_DIR / "rs_paired_descriptives.csv",
        index=False,
    )

    omnibus.to_csv(
        OUTPUT_DIR / "rs_friedman_tests.csv",
        index=False,
    )

    pairwise.to_csv(
        OUTPUT_DIR / "rs_pairwise_wilcoxon.csv",
        index=False,
    )

    completeness.to_csv(
        OUTPUT_DIR / "rs_complete_case_counts.csv",
        index=False,
    )

    status_counts.to_csv(
        OUTPUT_DIR / "rs_status_counts.csv",
        index=False,
    )

    method_text = f"""Step 9 paired significance-testing protocol for r and s

The sensitivity experiment evaluates the same candidate instances repeatedly
under different parameter settings. Statistical comparisons are therefore
paired rather than independent.

r sensitivity:
    s = 3 fixed
    r = 1, 2, 3

s sensitivity:
    r = 1 fixed
    s = 3, 4, 5

Outcomes:
    - optimization time
    - number of subtours eliminated
    - optimal objective value

For each family and outcome, only instances with a proven-optimal value under
all three compared settings are retained. Thus, the Friedman and pairwise
Wilcoxon tests use the same complete paired sample.

Omnibus test:
    Friedman test

Pairwise post-hoc tests:
    two-sided Wilcoxon signed-rank tests for all three pairs

Multiplicity correction:
    Holm family-wise error-rate correction applied separately within each
    family and outcome.

Significance level:
    alpha = {ALPHA}

Effect sizes:
    - Friedman: Kendall's W
    - Wilcoxon: matched-pairs rank-biserial correlation

For pairwise results, delta is defined as:
    setting 2 minus setting 1

Accordingly, for objective value, a negative delta means that setting 2
produced a lower (better) objective value. For runtime, a negative delta
means that setting 2 was faster.

TIMEOUT and ERROR outcomes are not converted to artificial numeric values.
They are reported separately in rs_status_counts.csv. If such outcomes cause
an instance to lack a proven-optimal value under one of the three settings,
that instance is excluded from the complete-case inferential test for that
family/outcome and the exclusion count is reported explicitly.
"""

    with open(
        OUTPUT_DIR / "step9_method.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(method_text)

    print_table(
        "PAIRED DESCRIPTIVE STATISTICS",
        descriptives,
    )

    print_table(
        "FRIEDMAN OMNIBUS TESTS",
        omnibus,
    )

    print_table(
        "PAIRWISE WILCOXON SIGNED-RANK TESTS",
        pairwise,
    )

    print_table(
        "COMPLETE-CASE COUNTS",
        completeness,
    )

    print_table(
        "STATUS COUNTS BY r/s SETTING",
        status_counts,
    )

    print("\nFiles created:")
    print(OUTPUT_DIR / "rs_paired_descriptives.csv")
    print(OUTPUT_DIR / "rs_friedman_tests.csv")
    print(OUTPUT_DIR / "rs_pairwise_wilcoxon.csv")
    print(OUTPUT_DIR / "rs_complete_case_counts.csv")
    print(OUTPUT_DIR / "rs_status_counts.csv")
    print(OUTPUT_DIR / "step9_method.txt")


if __name__ == "__main__":
    main()
