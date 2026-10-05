from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# Step 10: timeout / right-censoring analysis for runtime
# ============================================================
#
# The experiment uses a fixed 300-second cap. Therefore a TIMEOUT does not
# mean that the true time to proven optimality equals 300 seconds; it means
# only that it exceeds the observation window.
#
# This script treats:
#   OK       -> observed event (proven optimum reached)
#   TIMEOUT  -> right-censored at 300 s
#   ERROR    -> excluded from survival analysis and counted separately
#
# Baseline L-scaling experiment:
#   - Kaplan-Meier estimates
#   - restricted mean time to proven optimality (RMST) through 300 s
#   - pairwise log-rank tests within profile, Holm corrected
#
# Paired r/s sensitivity experiment:
#   - Kaplan-Meier/RMST summaries for descriptive purposes
#   - paired censoring-aware sign tests for pairwise setting comparisons
#
# The paired sign test uses the fact that all settings have the same 300 s
# cap. An OK solve before 300 s is known to be faster than a TIMEOUT at
# 300 s, while two TIMEOUTs are treated as a tie.
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

INPUT_DIR = Path("statistical_analysis_results")

BASELINE_CSV = INPUT_DIR / "baseline_analysis_ready.csv"
SENSITIVITY_CSV = INPUT_DIR / "rs_sensitivity_analysis_ready.csv"

OUTPUT_DIR = INPUT_DIR / "step10_timeout_censoring"
PLOTS_DIR = OUTPUT_DIR / "km_plots"


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

CENSOR_TIME = 300.0
ALPHA = 0.05

PROFILE_L_VALUES = {
    "Slow": [32, 43],
    "Moderate": [32, 62, 160],
    "Fast": [32, 62],
}

R_SETTINGS = [(1, 3), (2, 3), (3, 3)]
S_SETTINGS = [(1, 3), (1, 4), (1, 5)]


# ============================================================
# Generic helpers
# ============================================================

def load_csv(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} file not found: {path}\n"
            "Run Step 4 first."
        )

    return pd.read_csv(path)


def require_columns(df, columns, label):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}"
        )


def holm_adjust(p_values):
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


# ============================================================
# Runtime / censoring preparation
# ============================================================

def prepare_runtime_data(df, label):
    required = [
        "Status",
        "Time_optimal_s",
    ]
    require_columns(df, required, label)

    out = df.copy()

    out["Status"] = out["Status"].astype(str)

    out["Survival_time_s"] = np.nan
    out["Event_optimal"] = np.nan

    ok = out["Status"].eq("OK")
    timeout = out["Status"].eq("TIMEOUT")

    ok_times = pd.to_numeric(
        out.loc[ok, "Time_optimal_s"],
        errors="coerce",
    )

    if ok_times.isna().any():
        raise ValueError(
            f"{label}: at least one OK row has no Time_optimal_s value."
        )

    if (ok_times < 0).any():
        raise ValueError(
            f"{label}: negative optimization times were found."
        )

    if (ok_times > CENSOR_TIME).any():
        count = int((ok_times > CENSOR_TIME).sum())
        print(
            f"WARNING: {label} contains {count} OK run(s) with "
            f"Time_optimal_s > {CENSOR_TIME:.0f} s. "
            "Their observed times are retained."
        )

    out.loc[ok, "Survival_time_s"] = ok_times
    out.loc[ok, "Event_optimal"] = 1

    out.loc[timeout, "Survival_time_s"] = CENSOR_TIME
    out.loc[timeout, "Event_optimal"] = 0

    # ERROR rows remain NaN and are excluded from time-to-optimality analysis.
    return out


# ============================================================
# Kaplan-Meier and RMST
# ============================================================

def km_curve(times, events, tau=CENSOR_TIME):
    """
    Kaplan-Meier curve for time to proven optimality.

    Survival here means:
        P(not yet proven optimal by time t)

    Returns arrays suitable for plt.step(..., where="post").
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)

    valid = (
        np.isfinite(times)
        & np.isfinite(events)
        & (times >= 0)
    )

    times = times[valid]
    events = events[valid]

    if len(times) == 0:
        return np.array([0.0]), np.array([1.0])

    unique_times = np.sort(
        np.unique(times[times <= tau])
    )

    curve_t = [0.0]
    curve_s = [1.0]

    survival = 1.0

    for t in unique_times:
        at_risk = int(np.sum(times >= t))
        d = int(np.sum((times == t) & (events == 1)))

        if at_risk > 0 and d > 0:
            survival *= (1.0 - d / at_risk)

        curve_t.append(float(t))
        curve_s.append(float(survival))

    if curve_t[-1] < tau:
        curve_t.append(float(tau))
        curve_s.append(float(survival))

    return np.asarray(curve_t), np.asarray(curve_s)


def rmst(times, events, tau=CENSOR_TIME):
    """
    Restricted mean time to proven optimality through tau:
        integral_0^tau S(t) dt
    """
    curve_t, curve_s = km_curve(times, events, tau=tau)

    area = 0.0

    for i in range(len(curve_t) - 1):
        left = curve_t[i]
        right = min(curve_t[i + 1], tau)

        if right > left:
            area += curve_s[i] * (right - left)

        if right >= tau:
            break

    return float(area)


def km_median_time(times, events, tau=CENSOR_TIME):
    curve_t, curve_s = km_curve(times, events, tau=tau)

    hits = np.where(curve_s <= 0.5)[0]

    if len(hits) == 0:
        return np.nan

    return float(curve_t[hits[0]])


def survival_summary(group):
    usable = group[
        group["Survival_time_s"].notna()
        & group["Event_optimal"].notna()
    ].copy()

    times = usable["Survival_time_s"].to_numpy(dtype=float)
    events = usable["Event_optimal"].to_numpy(dtype=int)

    statuses = group["Status"].astype(str)

    return {
        "N_total": int(len(group)),
        "N_analyzed": int(len(usable)),
        "N_OK": int((statuses == "OK").sum()),
        "N_TIMEOUT": int((statuses == "TIMEOUT").sum()),
        "N_ERROR": int((statuses == "ERROR").sum()),
        "Timeout_percent": (
            float(100.0 * (statuses == "TIMEOUT").sum() / len(group))
            if len(group)
            else np.nan
        ),
        "KM_median_time_s": (
            km_median_time(times, events)
            if len(usable)
            else np.nan
        ),
        "RMST_0_300_s": (
            rmst(times, events)
            if len(usable)
            else np.nan
        ),
    }


# ============================================================
# Log-rank test for independent groups
# ============================================================

def logrank_two_sample(times1, events1, times2, events2):
    """
    Standard two-sample log-rank test.
    """
    times1 = np.asarray(times1, dtype=float)
    events1 = np.asarray(events1, dtype=int)
    times2 = np.asarray(times2, dtype=float)
    events2 = np.asarray(events2, dtype=int)

    event_times = np.sort(
        np.unique(
            np.concatenate(
                [
                    times1[events1 == 1],
                    times2[events2 == 1],
                ]
            )
        )
    )

    observed1 = 0.0
    expected1 = 0.0
    variance = 0.0

    for t in event_times:
        n1 = np.sum(times1 >= t)
        n2 = np.sum(times2 >= t)
        n = n1 + n2

        d1 = np.sum((times1 == t) & (events1 == 1))
        d2 = np.sum((times2 == t) & (events2 == 1))
        d = d1 + d2

        if n <= 1 or d == 0:
            continue

        expected = d * n1 / n

        var = (
            n1
            * n2
            * d
            * (n - d)
            / (n * n * (n - 1))
        )

        observed1 += d1
        expected1 += expected
        variance += var

    if variance <= 0:
        return np.nan, 1.0

    z = (observed1 - expected1) / np.sqrt(variance)
    chi2 = float(z * z)
    p = float(stats.chi2.sf(chi2, df=1))

    return chi2, p


# ============================================================
# Baseline censoring analysis
# ============================================================

def analyze_baseline(df):
    summary_rows = []
    pairwise_rows = []

    for profile, l_values in PROFILE_L_VALUES.items():
        profile_df = df[df["Profile"] == profile].copy()

        for L in l_values:
            group = profile_df[profile_df["L"] == L]

            row = {
                "Profile": profile,
                "L": L,
            }
            row.update(survival_summary(group))
            summary_rows.append(row)

        profile_pairwise = []

        for l1, l2 in combinations(l_values, 2):
            g1 = profile_df[
                (profile_df["L"] == l1)
                & profile_df["Survival_time_s"].notna()
            ]
            g2 = profile_df[
                (profile_df["L"] == l2)
                & profile_df["Survival_time_s"].notna()
            ]

            chi2, p = logrank_two_sample(
                g1["Survival_time_s"].to_numpy(dtype=float),
                g1["Event_optimal"].to_numpy(dtype=int),
                g2["Survival_time_s"].to_numpy(dtype=float),
                g2["Event_optimal"].to_numpy(dtype=int),
            )

            profile_pairwise.append(
                {
                    "Profile": profile,
                    "L1": l1,
                    "L2": l2,
                    "N1": int(len(g1)),
                    "N2": int(len(g2)),
                    "Logrank_chi2": chi2,
                    "p_raw": p,
                    "p_Holm": np.nan,
                    "Significant_0.05": "Pending",
                }
            )

        adjusted = holm_adjust(
            [row["p_raw"] for row in profile_pairwise]
        )

        for row, p_adj in zip(profile_pairwise, adjusted):
            row["p_Holm"] = float(p_adj)
            row["Significant_0.05"] = (
                "Yes" if p_adj < ALPHA else "No"
            )
            pairwise_rows.append(row)

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(pairwise_rows),
    )


def save_baseline_km_plots(df):
    for profile, l_values in PROFILE_L_VALUES.items():
        profile_df = df[df["Profile"] == profile]

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)

        plotted = False

        for L in l_values:
            group = profile_df[
                (profile_df["L"] == L)
                & profile_df["Survival_time_s"].notna()
            ]

            if group.empty:
                continue

            t, s = km_curve(
                group["Survival_time_s"].to_numpy(dtype=float),
                group["Event_optimal"].to_numpy(dtype=int),
            )

            ax.step(
                t,
                s,
                where="post",
                label=f"L={L}",
            )
            plotted = True

        if plotted:
            ax.set_title(
                f"{profile}: time to proven optimality"
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Probability not yet proven optimal")
            ax.set_xlim(0, CENSOR_TIME)
            ax.set_ylim(0, 1.02)
            ax.legend()
            fig.tight_layout()

            fig.savefig(
                PLOTS_DIR / f"baseline_{profile.lower()}_km.png",
                dpi=300,
                bbox_inches="tight",
            )

        plt.close(fig)


# ============================================================
# Sensitivity censoring analysis
# ============================================================

def sensitivity_survival_summary(df):
    rows = []

    grouped = df.groupby(
        ["Profile", "SMF", "L", "r", "s"],
        dropna=False,
        sort=True,
    )

    for key, group in grouped:
        profile, smf, L, r, s = key

        row = {
            "Profile": profile,
            "SMF": int(smf),
            "L": int(L),
            "r": int(r),
            "s": int(s),
        }
        row.update(survival_summary(group))
        rows.append(row)

    return pd.DataFrame(rows)


def paired_censored_comparison(df, setting1, setting2):
    """
    Compare two settings while retaining paired structure and respecting
    the common 300-second censoring limit.

    Faster/slower determination:
      OK vs OK       -> compare observed optimization times
      OK vs TIMEOUT  -> OK is faster
      TIMEOUT vs OK  -> OK is faster
      TIMEOUT vs TIMEOUT -> tie
      any ERROR/missing -> excluded
    """
    r1, s1 = setting1
    r2, s2 = setting2

    cols = [
        "Run",
        "Seed",
        "r",
        "s",
        "Status",
        "Time_optimal_s",
    ]

    a = df[
        (df["r"] == r1)
        & (df["s"] == s1)
    ][cols].copy()

    b = df[
        (df["r"] == r2)
        & (df["s"] == s2)
    ][cols].copy()

    a = a.rename(
        columns={
            "Status": "Status1",
            "Time_optimal_s": "Time1",
        }
    )

    b = b.rename(
        columns={
            "Status": "Status2",
            "Time_optimal_s": "Time2",
        }
    )

    merged = a[
        ["Run", "Seed", "Status1", "Time1"]
    ].merge(
        b[["Run", "Seed", "Status2", "Time2"]],
        on=["Run", "Seed"],
        how="outer",
        validate="one_to_one",
    )

    wins1 = 0
    wins2 = 0
    ties = 0
    excluded = 0

    for _, row in merged.iterrows():
        status1 = row["Status1"]
        status2 = row["Status2"]

        if pd.isna(status1) or pd.isna(status2):
            excluded += 1
            continue

        if status1 == "ERROR" or status2 == "ERROR":
            excluded += 1
            continue

        if status1 not in {"OK", "TIMEOUT"}:
            excluded += 1
            continue

        if status2 not in {"OK", "TIMEOUT"}:
            excluded += 1
            continue

        if status1 == "TIMEOUT" and status2 == "TIMEOUT":
            ties += 1
            continue

        if status1 == "OK" and status2 == "TIMEOUT":
            wins1 += 1
            continue

        if status1 == "TIMEOUT" and status2 == "OK":
            wins2 += 1
            continue

        # Both solved.
        time1 = float(row["Time1"])
        time2 = float(row["Time2"])

        if np.isclose(time1, time2, rtol=1e-12, atol=1e-12):
            ties += 1
        elif time1 < time2:
            wins1 += 1
        else:
            wins2 += 1

    informative = wins1 + wins2

    if informative == 0:
        p = 1.0
    else:
        # Exact two-sided sign test under equal probability of either
        # setting being faster.
        p = float(
            stats.binomtest(
                wins2,
                informative,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    return {
        "Setting1": setting_label(setting1),
        "Setting2": setting_label(setting2),
        "N_matched": int(len(merged)),
        "Setting1_faster": int(wins1),
        "Setting2_faster": int(wins2),
        "Ties": int(ties),
        "Excluded": int(excluded),
        "N_informative": int(informative),
        "Setting2_win_fraction": (
            float(wins2 / informative)
            if informative
            else np.nan
        ),
        "p_raw": p,
        "p_Holm": np.nan,
        "Significant_0.05": "Pending",
    }


def analyze_paired_censoring_family(df, family_name, settings):
    rows = []

    for setting1, setting2 in combinations(settings, 2):
        row = paired_censored_comparison(
            df,
            setting1,
            setting2,
        )
        row["Family"] = family_name
        rows.append(row)

    adjusted = holm_adjust(
        [row["p_raw"] for row in rows]
    )

    for row, p_adj in zip(rows, adjusted):
        row["p_Holm"] = float(p_adj)
        row["Significant_0.05"] = (
            "Yes" if p_adj < ALPHA else "No"
        )

    return rows


def save_sensitivity_km_plots(df):
    families = [
        ("r_sensitivity", R_SETTINGS),
        ("s_sensitivity", S_SETTINGS),
    ]

    for family_name, settings in families:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)

        plotted = False

        for r, s in settings:
            group = df[
                (df["r"] == r)
                & (df["s"] == s)
                & df["Survival_time_s"].notna()
            ]

            if group.empty:
                continue

            t, surv = km_curve(
                group["Survival_time_s"].to_numpy(dtype=float),
                group["Event_optimal"].to_numpy(dtype=int),
            )

            ax.step(
                t,
                surv,
                where="post",
                label=setting_label((r, s)),
            )
            plotted = True

        if plotted:
            ax.set_title(
                family_name.replace("_", " ")
                + ": time to proven optimality"
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Probability not yet proven optimal")
            ax.set_xlim(0, CENSOR_TIME)
            ax.set_ylim(0, 1.02)
            ax.legend()
            fig.tight_layout()

            fig.savefig(
                PLOTS_DIR / f"{family_name}_km.png",
                dpi=300,
                bbox_inches="tight",
            )

        plt.close(fig)


# ============================================================
# Printing
# ============================================================

def print_table(title, df):
    print("\n" + "=" * 130)
    print(title)
    print("=" * 130)

    if df.empty:
        print("No rows.")
        return

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 300,
        "display.float_format", lambda x: f"{x:.6g}",
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

    require_columns(
        baseline,
        ["Profile", "L", "Status", "Time_optimal_s"],
        "Baseline dataset",
    )

    require_columns(
        sensitivity,
        [
            "Profile",
            "SMF",
            "L",
            "Run",
            "Seed",
            "r",
            "s",
            "Status",
            "Time_optimal_s",
        ],
        "Sensitivity dataset",
    )

    baseline = prepare_runtime_data(
        baseline,
        "Baseline dataset",
    )

    sensitivity = prepare_runtime_data(
        sensitivity,
        "Sensitivity dataset",
    )

    baseline_summary, baseline_logrank = analyze_baseline(
        baseline
    )

    sensitivity_summary = sensitivity_survival_summary(
        sensitivity
    )

    paired_rows = []

    paired_rows.extend(
        analyze_paired_censoring_family(
            sensitivity,
            "r sensitivity (s=3)",
            R_SETTINGS,
        )
    )

    paired_rows.extend(
        analyze_paired_censoring_family(
            sensitivity,
            "s sensitivity (r=1)",
            S_SETTINGS,
        )
    )

    sensitivity_paired = pd.DataFrame(paired_rows)

    baseline_summary.to_csv(
        OUTPUT_DIR / "baseline_survival_summary.csv",
        index=False,
    )

    baseline_logrank.to_csv(
        OUTPUT_DIR / "baseline_logrank_pairwise.csv",
        index=False,
    )

    sensitivity_summary.to_csv(
        OUTPUT_DIR / "rs_survival_summary.csv",
        index=False,
    )

    sensitivity_paired.to_csv(
        OUTPUT_DIR / "rs_paired_censoring_sign_tests.csv",
        index=False,
    )

    save_baseline_km_plots(baseline)
    save_sensitivity_km_plots(sensitivity)

    method_text = f"""Step 10 timeout and censoring protocol

The computational experiment uses a fixed time limit of {CENSOR_TIME:.0f} s.

A TIMEOUT therefore does not imply that the true time to proven optimality
equals {CENSOR_TIME:.0f} s. Instead, it is treated as right-censored at
{CENSOR_TIME:.0f} s.

Coding:
    OK      = event observed (proven optimum reached)
    TIMEOUT = right-censored at {CENSOR_TIME:.0f} s
    ERROR   = excluded from survival analysis and reported separately

Baseline L-scaling experiment:
    Kaplan-Meier curves are used to describe the probability that a problem
    instance has not yet reached proven optimality by time t.

    Restricted mean time to proven optimality (RMST) is calculated through
    tau = {CENSOR_TIME:.0f} s.

    Pairwise L comparisons within each profile use two-sample log-rank tests.
    Pairwise p-values within each profile are Holm corrected.

Paired r/s sensitivity experiment:
    Kaplan-Meier and RMST values are reported descriptively for each setting.

    Because the same candidate instance is solved under every parameter
    setting, independence-based log-rank tests are not used for inferential
    comparisons among r/s settings.

    Instead, pairwise censoring-aware sign tests preserve the pairing:
        - if both settings solve, observed solution times are compared;
        - if one solves and the other times out, the solved setting is known
          to be faster under the common {CENSOR_TIME:.0f} s cap;
        - two timeouts are treated as a tie;
        - ERROR/missing pairs are excluded.

    Exact two-sided binomial sign tests are applied to the non-tied pairs,
    with Holm correction within the r family and within the s family.

This analysis complements the solved-only runtime tests from Steps 8 and 9.
It avoids treating censored TIMEOUT observations as if their exact runtime
were known.
"""

    with open(
        OUTPUT_DIR / "step10_method.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(method_text)

    print_table(
        "BASELINE CENSORING-AWARE RUNTIME SUMMARY",
        baseline_summary,
    )

    print_table(
        "BASELINE PAIRWISE LOG-RANK TESTS",
        baseline_logrank,
    )

    print_table(
        "r/s CENSORING-AWARE RUNTIME SUMMARY",
        sensitivity_summary,
    )

    print_table(
        "PAIRED r/s CENSORING-AWARE SIGN TESTS",
        sensitivity_paired,
    )

    print("\nFiles created:")
    print(OUTPUT_DIR / "baseline_survival_summary.csv")
    print(OUTPUT_DIR / "baseline_logrank_pairwise.csv")
    print(OUTPUT_DIR / "rs_survival_summary.csv")
    print(OUTPUT_DIR / "rs_paired_censoring_sign_tests.csv")
    print(OUTPUT_DIR / "step10_method.txt")
    print(f"Kaplan-Meier plots: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
