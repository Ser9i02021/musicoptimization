from pathlib import Path

from matplotlib import lines
import numpy as np
import pandas as pd


# ============================================================
# Step 11: consolidate statistical outputs for the manuscript
# ============================================================
#
# This script gathers the outputs from Steps 5-10 and creates:
#
#   1. manuscript-ready summary tables
#   2. a concise statistical Methods draft
#   3. a Results draft based on the actual computed values
#   4. a draft response to the referee
#
# It does NOT rerun any statistical test.
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE = Path("statistical_analysis_results")

STEP5 = BASE / "step5_confidence_intervals"
STEP6 = BASE / "step6_normality"
STEP7 = BASE / "step7_outliers"
STEP8 = BASE / "step8_significance_L"
STEP9 = BASE / "step9_significance_rs"
STEP10 = BASE / "step10_timeout_censoring"

OUTPUT_DIR = BASE / "step11_manuscript_package"


# ------------------------------------------------------------
# General settings
# ------------------------------------------------------------

ALPHA = 0.05
LOWER_OBJECTIVE_IS_BETTER = True


# ============================================================
# Helpers
# ============================================================

def load_required(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n  {path}\n\n"
            "Complete the preceding statistical-analysis steps first."
        )
    return pd.read_csv(path)


def fmt(x, digits=2):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def fmt_p(x):
    if pd.isna(x):
        return "NA"
    x = float(x)
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}"


def yes_no_p(x):
    if pd.isna(x):
        return "NA"
    return "Yes" if float(x) < ALPHA else "No"


def ci_text(mean, se, low, high, digits=2):
    if any(pd.isna(v) for v in [mean, se, low, high]):
        return "NA"
    return (
        f"{float(mean):.{digits}f} "
        f"(SE {float(se):.{digits}f}; "
        f"95% CI {float(low):.{digits}f} to {float(high):.{digits}f})"
    )


def markdown_table(df):
    """
    Small dependency-free Markdown table renderer.
    """
    if df.empty:
        return "_No data._"

    cols = list(df.columns)

    def cell(value):
        if pd.isna(value):
            return "NA"
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]

    for _, row in df.iterrows():
        lines.append(
            "| " + " | ".join(cell(row[c]) for c in cols) + " |"
        )

    return "\n".join(lines)


def save_markdown_table(df, path, title):
    text = f"# {title}\n\n{markdown_table(df)}\n"
    path.write_text(text, encoding="utf-8")


def metric_lookup(df, metric):
    return df[df["Metric"] == metric].copy()


# ============================================================
# 1. Baseline manuscript table
# ============================================================

def build_baseline_table(ci, normality, outliers, status_counts):
    rows = []

    ci = ci.copy()
    normality = normality.copy()
    outliers = outliers.copy()
    status_counts = status_counts.copy()

    for _, row in ci.iterrows():
        profile = row["Profile"]
        smf = int(row["SMF"])
        L = int(row["L"])

        base = {
            "Profile": profile,
            "SMF": smf,
            "L": L,
            "N_OK": int(row["N_OK"]),
            "Timeouts": int(row["N_TIMEOUT"]),
            "Errors": int(row["N_ERROR"]),
        }

        for metric, ci_prefix in [
            ("Time", "Time_s"),
            ("Subtours", "Subtours"),
            ("Quality", "Quality"),
        ]:
            nrow = normality[
                (normality["Profile"] == profile)
                & (normality["SMF"] == smf)
                & (normality["L"] == L)
                & (normality["Metric"] == metric)
            ]

            orow = outliers[
                (outliers["Profile"] == profile)
                & (outliers["SMF"] == smf)
                & (outliers["L"] == L)
                & (outliers["Metric"] == metric)
            ]

            base[f"{metric}: mean (SE; 95% CI)"] = ci_text(
                row[f"{ci_prefix}_Mean"],
                row[f"{ci_prefix}_SE"],
                row[f"{ci_prefix}_CI95_low"],
                row[f"{ci_prefix}_CI95_high"],
            )

            base[f"{metric}: median"] = fmt(
                row[f"{ci_prefix}_Median"]
            )

            if not nrow.empty:
                base[f"{metric}: Shapiro p"] = fmt_p(
                    nrow.iloc[0]["Shapiro_p"]
                )
            else:
                base[f"{metric}: Shapiro p"] = "NA"

            if not orow.empty:
                base[f"{metric}: outliers"] = (
                    f"{int(orow.iloc[0]['N_outliers'])} "
                    f"({fmt(orow.iloc[0]['Outlier_percent'], 1)}%)"
                )
            else:
                base[f"{metric}: outliers"] = "NA"

        rows.append(base)

    result = pd.DataFrame(rows)
    return result.sort_values(["Profile", "L"]).reset_index(drop=True)


# ============================================================
# 2. L-scaling test tables
# ============================================================

def build_L_omnibus_table(kw):
    if kw.empty:
        return kw.copy()

    out = kw.copy()

    out["p"] = out["p"].map(fmt_p)
    out["Epsilon_squared"] = out["Epsilon_squared"].map(
        lambda x: fmt(x, 3)
    )

    return out[
        [
            "Profile",
            "Metric",
            "K_groups",
            "N_total",
            "H",
            "p",
            "Epsilon_squared",
            "Significant_0.05",
        ]
    ]


def build_L_pairwise_table(pairwise):
    out = pairwise.copy()

    out["p_raw"] = out["p_raw"].map(fmt_p)
    out["p_Holm"] = out["p_Holm"].map(fmt_p)
    out["Rank_biserial"] = out["Rank_biserial"].map(
        lambda x: fmt(x, 3)
    )

    return out[
        [
            "Profile",
            "Metric",
            "L1",
            "L2",
            "N1",
            "N2",
            "Median1",
            "Median2",
            "p_raw",
            "p_Holm",
            "Rank_biserial",
            "Significant_0.05",
        ]
    ]


# ============================================================
# 3. r/s sensitivity tables
# ============================================================

def build_rs_summary_table(ci, normality, outliers):
    rows = []

    for _, row in ci.iterrows():
        profile = row["Profile"]
        smf = int(row["SMF"])
        L = int(row["L"])
        r = int(row["r"])
        s = int(row["s"])

        base = {
            "Profile": profile,
            "L": L,
            "r": r,
            "s": s,
            "N_OK": int(row["N_OK"]),
            "Timeouts": int(row["N_TIMEOUT"]),
            "Errors": int(row["N_ERROR"]),
        }

        for metric, prefix in [
            ("Time", "Time_s"),
            ("Subtours", "Subtours"),
            ("Quality", "Quality"),
        ]:
            nrow = normality[
                (normality["Profile"] == profile)
                & (normality["SMF"] == smf)
                & (normality["L"] == L)
                & (normality["r"] == r)
                & (normality["s"] == s)
                & (normality["Metric"] == metric)
            ]

            orow = outliers[
                (outliers["Profile"] == profile)
                & (outliers["SMF"] == smf)
                & (outliers["L"] == L)
                & (outliers["r"] == r)
                & (outliers["s"] == s)
                & (outliers["Metric"] == metric)
            ]

            base[f"{metric}: mean (SE; 95% CI)"] = ci_text(
                row[f"{prefix}_Mean"],
                row[f"{prefix}_SE"],
                row[f"{prefix}_CI95_low"],
                row[f"{prefix}_CI95_high"],
            )

            base[f"{metric}: median"] = fmt(
                row[f"{prefix}_Median"]
            )

            base[f"{metric}: Shapiro p"] = (
                fmt_p(nrow.iloc[0]["Shapiro_p"])
                if not nrow.empty
                else "NA"
            )

            base[f"{metric}: outliers"] = (
                f"{int(orow.iloc[0]['N_outliers'])} "
                f"({fmt(orow.iloc[0]['Outlier_percent'], 1)}%)"
                if not orow.empty
                else "NA"
            )

        rows.append(base)

    return pd.DataFrame(rows).sort_values(
        ["r", "s"]
    ).reset_index(drop=True)


def build_rs_omnibus_table(friedman):
    out = friedman.copy()

    out["p"] = out["p"].map(fmt_p)
    out["Kendalls_W"] = out["Kendalls_W"].map(
        lambda x: fmt(x, 3)
    )

    return out[
        [
            "Family",
            "Metric",
            "N_complete_pairs",
            "Friedman_chi2",
            "p",
            "Kendalls_W",
            "Significant_0.05",
        ]
    ]


def build_rs_pairwise_table(wilcoxon):
    out = wilcoxon.copy()

    out["p_raw"] = out["p_raw"].map(fmt_p)
    out["p_Holm"] = out["p_Holm"].map(fmt_p)

    out["Median_delta_2_minus_1"] = out[
        "Median_delta_2_minus_1"
    ].map(lambda x: fmt(x, 3))

    out["Mean_delta_2_minus_1"] = out[
        "Mean_delta_2_minus_1"
    ].map(lambda x: fmt(x, 3))

    out["Rank_biserial_2_minus_1"] = out[
        "Rank_biserial_2_minus_1"
    ].map(lambda x: fmt(x, 3))

    return out[
        [
            "Family",
            "Metric",
            "Setting1",
            "Setting2",
            "N_pairs",
            "Median_delta_2_minus_1",
            "Mean_delta_2_minus_1",
            "p_raw",
            "p_Holm",
            "Rank_biserial_2_minus_1",
            "Significant_0.05",
        ]
    ]


# ============================================================
# 4. Timeout/censoring table
# ============================================================

def build_timeout_table(baseline_survival):
    out = baseline_survival.copy()

    for col in [
        "Timeout_percent",
        "KM_median_time_s",
        "RMST_0_300_s",
    ]:
        out[col] = out[col].map(lambda x: fmt(x, 2))

    return out[
        [
            "Profile",
            "L",
            "N_total",
            "N_OK",
            "N_TIMEOUT",
            "N_ERROR",
            "Timeout_percent",
            "KM_median_time_s",
            "RMST_0_300_s",
        ]
    ]


# ============================================================
# 5. Automatic narrative
# ============================================================

def normality_sentence(normality):
    tested = normality.dropna(subset=["Shapiro_p"]).copy()

    if tested.empty:
        return (
            "Normality could not be assessed because no valid "
            "Shapiro-Wilk results were available."
        )

    rejected = tested[tested["Shapiro_p"] < ALPHA]

    return (
        f"Shapiro-Wilk tests were available for {len(tested)} "
        f"configuration-outcome combinations; normality was rejected at "
        f"the 5% level in {len(rejected)} of them. Q-Q plots, skewness, "
        "and excess kurtosis were also examined, so the normality test was "
        "used as a diagnostic rather than as the sole basis for test selection."
    )


def outlier_sentence(outliers):
    if outliers.empty:
        return "No outlier-summary results were available."

    n_examined = len(outliers)

    total_outliers = int(
        pd.to_numeric(
            outliers["N_outliers"],
            errors="coerce",
        ).fillna(0).sum()
    )

    n_affected = int(
        (
            pd.to_numeric(
                outliers["N_outliers"],
                errors="coerce",
            ).fillna(0)
            > 0
        ).sum()
    )

    return (
        f"Tukey's 1.5×IQR rule was applied to all {n_examined} "
        f"configuration-outcome combinations. It identified "
        f"{total_outliers} outlying observations, with at least one "
        f"outlier occurring in {n_affected} of the {n_examined} "
        f"combinations. These observations were retained in the "
        "principal analysis; the outlier analysis is descriptive "
        "rather than a deletion rule."
    )


def significant_pairwise_text(pairwise, label):
    if pairwise.empty:
        return f"No {label} pairwise tests were available."

    # Use numeric p_Holm from the original dataframe.
    sig = pairwise[
        pd.to_numeric(
            pairwise["p_Holm"],
            errors="coerce",
        ) < ALPHA
    ]

    if sig.empty:
        return (
            f"None of the Holm-adjusted {label} pairwise comparisons "
            "was significant at the 5% level."
        )

    pieces = []

    for _, row in sig.iterrows():
        if "L1" in row.index:
            comparison = (
                f"{row['Profile']} profile, "
                f"L={int(row['L1'])} vs L={int(row['L2'])}"
            )
        else:
            comparison = f"{row['Setting1']} vs {row['Setting2']}"

        pieces.append(
            f"{row['Metric']} ({comparison}, adjusted p={fmt_p(row['p_Holm'])})"
        )

    return (
        f"Significant Holm-adjusted {label} pairwise differences were found "
        "for: " + "; ".join(pieces) + "."
    )


def parameter_direction_text(wilcoxon):
    """
    Generate cautious direction summaries for baseline-vs-relaxed settings.
    """
    if wilcoxon.empty:
        return ""

    lines = []

    baseline = "r=1,s=3"

    relevant = wilcoxon[
        wilcoxon["Setting1"].eq(baseline)
        | wilcoxon["Setting2"].eq(baseline)
    ].copy()

    for _, row in relevant.iterrows():
        delta = pd.to_numeric(
            pd.Series([row["Median_delta_2_minus_1"]]),
            errors="coerce",
        ).iloc[0]

        if pd.isna(delta):
            continue

        metric = row["Metric"]
        setting1 = row["Setting1"]
        setting2 = row["Setting2"]

        p_adj = pd.to_numeric(
        pd.Series([row["p_Holm"]]),
        errors="coerce",
    ).iloc[0]

    effect = pd.to_numeric(
        pd.Series([row["Rank_biserial_2_minus_1"]]),
        errors="coerce",
    ).iloc[0]

    if np.isclose(delta, 0.0):

        if not pd.isna(p_adj) and p_adj < ALPHA:

            if metric == "Quality":
                if effect < 0:
                    direction = (
                        "had a median paired difference of zero, "
                        "but the nonzero paired differences were "
                        "systematically toward lower (better) objective values"
                    )
                elif effect > 0:
                    direction = (
                        "had a median paired difference of zero, "
                        "but the nonzero paired differences were "
                        "systematically toward higher (worse) objective values"
                    )
                else:
                    direction = (
                        "had a median paired difference of zero despite "
                        "a statistically significant paired distributional shift"
                    )

            elif metric == "Time":
                if effect < 0:
                    direction = (
                        "had a median paired difference of zero, "
                        "but the nonzero paired differences tended toward "
                        "lower solution times"
                    )
                elif effect > 0:
                    direction = (
                        "had a median paired difference of zero, "
                        "but the nonzero paired differences tended toward "
                        "higher solution times"
                    )
                else:
                    direction = (
                        "had a median paired difference of zero despite "
                        "a statistically significant paired distributional shift"
                    )

            else:
                direction = (
                    "had a median paired difference of zero despite "
                    "a statistically significant paired distributional shift"
                )

        else:
            direction = "showed no detectable paired change"

    elif metric == "Quality":
        direction = (
            "showed a lower (better) median objective"
            if delta < 0
            else "showed a higher (worse) median objective"
        )

    elif metric == "Time":
        direction = (
            "showed a lower median solution time"
            if delta < 0
            else "showed a higher median solution time"
        )

    else:
        direction = (
            "showed a lower median value"
            if delta < 0
            else "showed a higher median value"
        )

        lines.append(
        f"For {metric}, {setting2} relative to {setting1} {direction} "
        f"(median paired difference {fmt(delta, 3)}; "
        f"matched-pairs rank-biserial correlation "
        f"{fmt(effect, 3)}; "
        f"Holm-adjusted p={fmt_p(row['p_Holm'])})."
    )

    return " ".join(lines)


def make_methods_draft():
    return """## Statistical analysis

For each experimental configuration, we report the mean, median, sample standard deviation, standard error, and a 95% percentile-bootstrap confidence interval for the mean based on 10,000 resamples. Distributional shape was assessed using Shapiro-Wilk tests together with Q-Q plots, skewness, and excess kurtosis. Outliers were identified within each configuration and outcome using Tukey's 1.5×IQR rule; flagged observations were retained in the principal analyses.

Because the outcome distributions were not assumed to be Gaussian, comparisons across candidate-set sizes were conducted using nonparametric procedures. Two-level comparisons used two-sided Mann-Whitney U tests, while the three-level Moderate-profile comparison used a Kruskal-Wallis omnibus test followed by pairwise Mann-Whitney U tests. Holm correction was applied to families of pairwise comparisons. Rank-biserial correlation and epsilon-squared were reported as effect-size measures.

For the Moderate profile with L=62, the sensitivity analysis varied the repetition limit r and pause limit s while keeping the same candidate instances across parameter settings. These repeated evaluations were therefore analyzed as paired observations. Friedman tests were used as omnibus tests and were followed by paired Wilcoxon signed-rank tests with Holm correction. Kendall's W and matched-pairs rank-biserial correlation were reported as effect-size measures.

Runs that reached the fixed 300-s limit without a proof of optimality were treated as right-censored rather than as exact 300-s observations. Kaplan-Meier summaries and restricted mean time to proven optimality through 300 s were used for censoring-aware runtime summaries. Errors were reported separately and were not converted into numeric runtime or objective observations.

The sensitivity analysis was conducted for the Moderate profile with
L=62. The repetition limit r and pause limit s were varied while keeping
the same candidate instances across parameter settings. These repeated
evaluations were therefore analyzed as paired observations. Friedman
tests were used as omnibus tests and were followed by paired Wilcoxon
signed-rank tests with Holm correction. Kendall's W and matched-pairs
rank-biserial correlation were reported as effect-size measures.
"""


def make_results_draft(
    baseline_normality,
    baseline_outliers,
    L_kw,
    L_pairwise,
    rs_normality,
    rs_outliers,
    rs_friedman,
    rs_wilcoxon,
    baseline_survival,
):
    paragraphs = []

    paragraphs.append(
        "## Statistical robustness results\n\n"
        + normality_sentence(baseline_normality)
        + " "
        + outlier_sentence(baseline_outliers)
    )

    if not L_kw.empty:
        significant_kw = L_kw[
            pd.to_numeric(L_kw["p"], errors="coerce") < ALPHA
        ]

        if significant_kw.empty:
            omnibus_text = (
                "For the Moderate profile, none of the Kruskal-Wallis "
                "omnibus tests across L=32, 62, and 160 was significant "
                "at the 5% level."
            )
        else:
            items = [
                f"{row['Metric']} (p={fmt_p(row['p'])}, "
                f"epsilon-squared={fmt(row['Epsilon_squared'], 3)})"
                for _, row in significant_kw.iterrows()
            ]
            omnibus_text = (
                "For the Moderate profile, significant Kruskal-Wallis "
                "differences across L were observed for "
                + "; ".join(items)
                + "."
            )

        paragraphs.append(
                omnibus_text
                + " "
                + significant_pairwise_text(
                    L_pairwise,
                    "L-scaling",
                )
            )

        paragraphs.append(
        "## Parameter-sensitivity results\n\n"
        "Parameter sensitivity was examined for the Moderate profile "
        "with L=62, using the same candidate instances under every "
        "parameter setting. The repetition limit was varied over "
        "r=1,2,3 with s=3, while the pause limit was varied over "
        "s=3,4,5 with r=1. "
        + normality_sentence(rs_normality)
        + " "
        + outlier_sentence(rs_outliers)
    )

    if not rs_friedman.empty:
        significant_f = rs_friedman[
            pd.to_numeric(
                rs_friedman["p"],
                errors="coerce",
            ) < ALPHA
        ]

        if significant_f.empty:
            friedman_text = (
                "None of the paired Friedman omnibus tests for r or s "
                "was significant at the 5% level."
            )
        else:
            items = [
                f"{row['Family']} / {row['Metric']} "
                f"(p={fmt_p(row['p'])}, "
                f"Kendall's W={fmt(row['Kendalls_W'], 3)})"
                for _, row in significant_f.iterrows()
            ]

            friedman_text = (
                "Significant paired omnibus effects were observed for "
                + "; ".join(items)
                + "."
            )

        paragraphs.append(
            friedman_text
            + " "
            + significant_pairwise_text(
                rs_wilcoxon,
                "r/s",
            )
            + " "
            + parameter_direction_text(rs_wilcoxon)
        )

    if not baseline_survival.empty:
        n_timeouts = int(
            pd.to_numeric(
                baseline_survival["N_TIMEOUT"],
                errors="coerce",
            ).fillna(0).sum()
        )

        if n_timeouts == 0:
            timeout_text = (
                "No baseline run reached the 300-s time limit, so censoring "
                "did not affect the baseline runtime summaries."
            )
        else:
            timeout_text = (
                f"Across the baseline experiment, {n_timeouts} run(s) reached "
                "the 300-s time limit. These observations were treated as "
                "right-censored in the censoring-aware runtime analysis rather "
                "than as exact 300-s solution times."
            )

        paragraphs.append(
            "## Timeout-aware runtime analysis\n\n" + timeout_text
        )

    return "\n\n".join(paragraphs) + "\n"


def make_referee_response(
    baseline_normality,
    baseline_outliers,
    L_pairwise,
    rs_friedman,
    rs_wilcoxon,
    baseline_survival,
):
    n_normality = len(
        baseline_normality.dropna(subset=["Shapiro_p"])
    )

    n_outlier_tests = len(baseline_outliers)

    total_outliers = int(
        pd.to_numeric(
            baseline_outliers["N_outliers"],
            errors="coerce",
        ).fillna(0).sum()
    )

    n_outlier_affected = int(
        (
            pd.to_numeric(
                baseline_outliers["N_outliers"],
                errors="coerce",
            ).fillna(0)
            > 0
        ).sum()
    )

    n_timeout = int(
        pd.to_numeric(
            baseline_survival["N_TIMEOUT"],
            errors="coerce",
        ).fillna(0).sum()
    )

    return f"""## Draft response to the referee

**Comment:** There are still no confidence intervals, standard errors, normality checks, outlier analysis, or significance tests. The parameters r and s are fixed and not varied; their effect on quality and time is not investigated.

**Response:** We thank the reviewer for this suggestion. We have expanded the statistical analysis substantially. For every experimental configuration, we now report standard errors and 95% bootstrap confidence intervals for the principal outcomes. We also added Shapiro-Wilk normality diagnostics ({n_normality} configuration-outcome tests in the baseline analysis), together with Q-Q plots, skewness, and excess kurtosis. Outlier screening using Tukey's 1.5×IQR rule was applied to all {n_outlier_tests} baseline configuration-outcome combinations. It identified {total_outliers} outlying observations, with at least one outlier occurring in {n_outlier_affected} of the {n_outlier_tests} combinations. These observations are reported and retained rather than automatically removed.

We also added formal significance tests for the candidate-set-size experiments. Two-level comparisons use two-sided Mann-Whitney U tests, while the three-level Moderate-profile comparison uses a Kruskal-Wallis omnibus test followed by pairwise Mann-Whitney U tests. Holm correction is applied to multiple pairwise comparisons, and effect sizes are reported.

In addition, we introduced a dedicated parameter-sensitivity experiment for the Moderate profile with L=62, in which the repetition limit r and pause limit s are varied on the same candidate instances. Specifically, r is varied over 1, 2, and 3 with s fixed at 3, while s is varied over 3, 4, and 5 with r fixed at 1. Because the same instances are evaluated repeatedly, these comparisons are paired. We therefore use Friedman omnibus tests followed by paired Wilcoxon signed-rank tests with Holm correction, together with Kendall's W and matched-pairs rank-biserial effect sizes. This directly evaluates the effect of relaxing r and s on objective quality and computational time while avoiding confounding from different randomly sampled candidate sets.

Finally, the 300-s computational limit is handled explicitly. Across the baseline experiment, {n_timeout} run(s) reached the time limit. Such runs are treated as right-censored rather than as exact 300-s observations, and censoring-aware runtime summaries are reported separately.

The corresponding methodological description, tables, and discussion have been added to the revised manuscript.
"""


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 5
    baseline_ci = load_required(
        STEP5 / "baseline_se_ci_paper.csv"
    )
    rs_ci = load_required(
        STEP5 / "rs_sensitivity_se_ci_paper.csv"
    )

    # Step 6
    baseline_norm = load_required(
        STEP6 / "baseline_normality_paper.csv"
    )
    rs_norm = load_required(
        STEP6 / "rs_sensitivity_normality_paper.csv"
    )

    # Step 7
    baseline_outliers = load_required(
        STEP7 / "baseline_outlier_summary_paper.csv"
    )
    rs_outliers = load_required(
        STEP7 / "rs_sensitivity_outlier_summary_paper.csv"
    )

    # Step 8
    L_kw = load_required(
        STEP8 / "L_kruskal_wallis.csv"
    )
    L_pairwise = load_required(
        STEP8 / "L_pairwise_mann_whitney.csv"
    )
    L_status = load_required(
        STEP8 / "L_status_counts.csv"
    )

    # Step 9
    rs_friedman = load_required(
        STEP9 / "rs_friedman_tests.csv"
    )
    rs_wilcoxon = load_required(
        STEP9 / "rs_pairwise_wilcoxon.csv"
    )

    # Step 10
    baseline_survival = load_required(
        STEP10 / "baseline_survival_summary.csv"
    )
    baseline_logrank = load_required(
        STEP10 / "baseline_logrank_pairwise.csv"
    )
    rs_survival = load_required(
        STEP10 / "rs_survival_summary.csv"
    )
    rs_censor_pairwise = load_required(
        STEP10 / "rs_paired_censoring_sign_tests.csv"
    )

    # --------------------------------------------------------
    # Build manuscript-ready tables
    # --------------------------------------------------------

    baseline_table = build_baseline_table(
        baseline_ci,
        baseline_norm,
        baseline_outliers,
        L_status,
    )

    L_omnibus_table = build_L_omnibus_table(L_kw)
    L_pairwise_table = build_L_pairwise_table(L_pairwise)

    rs_summary_table = build_rs_summary_table(
        rs_ci,
        rs_norm,
        rs_outliers,
    )

    rs_omnibus_table = build_rs_omnibus_table(rs_friedman)
    rs_pairwise_table = build_rs_pairwise_table(rs_wilcoxon)

    timeout_table = build_timeout_table(
        baseline_survival
    )

    # CSV outputs
    baseline_table.to_csv(
        OUTPUT_DIR / "Table_A_baseline_statistics.csv",
        index=False,
    )
    L_omnibus_table.to_csv(
        OUTPUT_DIR / "Table_B_L_omnibus_tests.csv",
        index=False,
    )
    L_pairwise_table.to_csv(
        OUTPUT_DIR / "Table_C_L_pairwise_tests.csv",
        index=False,
    )
    rs_summary_table.to_csv(
        OUTPUT_DIR / "Table_D_rs_sensitivity_statistics.csv",
        index=False,
    )
    rs_omnibus_table.to_csv(
        OUTPUT_DIR / "Table_E_rs_omnibus_tests.csv",
        index=False,
    )
    rs_pairwise_table.to_csv(
        OUTPUT_DIR / "Table_F_rs_pairwise_tests.csv",
        index=False,
    )
    timeout_table.to_csv(
        OUTPUT_DIR / "Table_G_timeout_analysis.csv",
        index=False,
    )

    # Markdown versions
    save_markdown_table(
        baseline_table,
        OUTPUT_DIR / "Table_A_baseline_statistics.md",
        "Baseline statistical summary",
    )
    save_markdown_table(
        L_omnibus_table,
        OUTPUT_DIR / "Table_B_L_omnibus_tests.md",
        "Candidate-set-size omnibus tests",
    )
    save_markdown_table(
        L_pairwise_table,
        OUTPUT_DIR / "Table_C_L_pairwise_tests.md",
        "Candidate-set-size pairwise tests",
    )
    save_markdown_table(
        rs_summary_table,
        OUTPUT_DIR / "Table_D_rs_sensitivity_statistics.md",
        "r/s sensitivity statistical summary",
    )
    save_markdown_table(
        rs_omnibus_table,
        OUTPUT_DIR / "Table_E_rs_omnibus_tests.md",
        "r/s sensitivity omnibus tests",
    )
    save_markdown_table(
        rs_pairwise_table,
        OUTPUT_DIR / "Table_F_rs_pairwise_tests.md",
        "r/s sensitivity pairwise tests",
    )
    save_markdown_table(
        timeout_table,
        OUTPUT_DIR / "Table_G_timeout_analysis.md",
        "Timeout-aware runtime summary",
    )

    # --------------------------------------------------------
    # Draft manuscript prose
    # --------------------------------------------------------

    methods = make_methods_draft()

    results = make_results_draft(
        baseline_norm,
        baseline_outliers,
        L_kw,
        L_pairwise,
        rs_norm,
        rs_outliers,
        rs_friedman,
        rs_wilcoxon,
        baseline_survival,
    )

    referee = make_referee_response(
        baseline_norm,
        baseline_outliers,
        L_pairwise,
        rs_friedman,
        rs_wilcoxon,
        baseline_survival,
    )

    (OUTPUT_DIR / "manuscript_methods_draft.md").write_text(
        methods,
        encoding="utf-8",
    )

    (OUTPUT_DIR / "manuscript_results_draft.md").write_text(
        results,
        encoding="utf-8",
    )

    (OUTPUT_DIR / "response_to_referee_draft.md").write_text(
        referee,
        encoding="utf-8",
    )

    # Keep raw censoring comparison tables in the package as well.
    baseline_logrank.to_csv(
        OUTPUT_DIR / "Supplement_baseline_logrank.csv",
        index=False,
    )

    rs_survival.to_csv(
        OUTPUT_DIR / "Supplement_rs_survival_summary.csv",
        index=False,
    )

    rs_censor_pairwise.to_csv(
        OUTPUT_DIR / "Supplement_rs_censoring_pairwise.csv",
        index=False,
    )

    print("\n" + "=" * 90)
    print("STEP 11 COMPLETE")
    print("=" * 90)
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nMain manuscript files:")
    print("  Table_A_baseline_statistics.csv / .md")
    print("  Table_B_L_omnibus_tests.csv / .md")
    print("  Table_C_L_pairwise_tests.csv / .md")
    print("  Table_D_rs_sensitivity_statistics.csv / .md")
    print("  Table_E_rs_omnibus_tests.csv / .md")
    print("  Table_F_rs_pairwise_tests.csv / .md")
    print("  Table_G_timeout_analysis.csv / .md")
    print("  manuscript_methods_draft.md")
    print("  manuscript_results_draft.md")
    print("  response_to_referee_draft.md")
    print("\nReview the automatically drafted prose before inserting it into the paper.")


if __name__ == "__main__":
    main()
