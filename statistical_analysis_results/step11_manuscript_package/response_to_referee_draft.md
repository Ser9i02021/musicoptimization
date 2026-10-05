## Draft response to the referee

**Comment:** There are still no confidence intervals, standard errors, normality checks, outlier analysis, or significance tests. The parameters r and s are fixed and not varied; their effect on quality and time is not investigated.

**Response:** We thank the reviewer for this suggestion. We have expanded the statistical analysis substantially. For every experimental configuration, we now report standard errors and 95% bootstrap confidence intervals for the principal outcomes. We also added Shapiro-Wilk normality diagnostics (21 configuration-outcome tests in the baseline analysis), together with Q-Q plots, skewness, and excess kurtosis. Outlier screening using Tukey's 1.5×IQR rule was applied to all 21 baseline configuration-outcome combinations. It identified 161 outlying observations, with at least one outlier occurring in 20 of the 21 combinations. These observations are reported and retained rather than automatically removed.

We also added formal significance tests for the candidate-set-size experiments. Two-level comparisons use two-sided Mann-Whitney U tests, while the three-level Moderate-profile comparison uses a Kruskal-Wallis omnibus test followed by pairwise Mann-Whitney U tests. Holm correction is applied to multiple pairwise comparisons, and effect sizes are reported.

In addition, we introduced a dedicated parameter-sensitivity experiment for the Moderate profile with L=62, in which the repetition limit r and pause limit s are varied on the same candidate instances. Specifically, r is varied over 1, 2, and 3 with s fixed at 3, while s is varied over 3, 4, and 5 with r fixed at 1. Because the same instances are evaluated repeatedly, these comparisons are paired. We therefore use Friedman omnibus tests followed by paired Wilcoxon signed-rank tests with Holm correction, together with Kendall's W and matched-pairs rank-biserial effect sizes. This directly evaluates the effect of relaxing r and s on objective quality and computational time while avoiding confounding from different randomly sampled candidate sets.

Finally, the 300-s computational limit is handled explicitly. Across the baseline experiment, 6 run(s) reached the time limit. Such runs are treated as right-censored rather than as exact 300-s observations, and censoring-aware runtime summaries are reported separately.

The corresponding methodological description, tables, and discussion have been added to the revised manuscript.
