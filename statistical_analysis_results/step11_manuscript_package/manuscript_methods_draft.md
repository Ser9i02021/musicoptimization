## Statistical analysis

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
