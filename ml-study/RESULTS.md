# Results: leaf classifier study

Setup as in [SPEC.md](SPEC.md) and [PLAN.md](PLAN.md): 492 rows after removing exact duplicates, a stratified split into 393 training rows and 99 test rows (seed 42), every model tuned with 5-fold stratified CV on the training rows only, balanced accuracy as the metric. The test set was used once, after the model was chosen.

Reproduce from the repository root with `uv run python ml-study/train_models.py --search`, then `--final` (which refuses to run a second time), then `--ci` (confidence intervals from the saved test predictions).

## Model choice (cross-validation, training rows only)

| Model | CV balanced accuracy (mean ± std) | Best settings |
|---|---|---|
| baseline (always "maple") | 0.333 ± 0.000 | – |
| logistic regression | 0.929 ± 0.043 | C = 1 |
| decision tree | 0.931 ± 0.018 | max_depth = 5 |
| **k-NN** | **0.946 ± 0.030** | n_neighbors = 9 |
| SVM (RBF) | 0.942 ± 0.030 | C = 100, gamma = 0.1 |
| random forest | 0.938 ± 0.027 | max_depth = 8, min_samples_leaf = 1 |

Chosen: **k-NN with k = 9**. It has the best mean; SVM and random forest are within 0.01, and k-NN is the simplest of the three (the "simpler" order is set in PLAN.md, not in SPEC.md). Every grid point is in `results/cv_*.csv`.

The five real models are practically tied: their means (0.929–0.946) differ by less than one CV standard deviation (0.02–0.04), and the ranking of k-NN and SVM depends on the CV seed (with seeds 0–4 instead of 42, SVM is ahead once). k-NN is the choice by the rule in SPEC.md, not clearly the best model.

## Test set (99 rows, used once)

| Model | Balanced accuracy | Accuracy | Macro F1 |
|---|---|---|---|
| baseline | 0.333 | 0.424 | 0.199 |
| **k-NN (k = 9)** | **0.884** | **0.899** | **0.887** |
| k-NN, 95 % CI | 0.813–0.948 | 0.838–0.949 | 0.816–0.948 |

The confidence intervals come from a percentile bootstrap of the saved test predictions (`results/test_predictions.csv`): 10,000 resamples, drawn within each class so the class counts stay fixed, seed 42 (`train_models.py --ci`, saved to `results/test_ci.csv`). No model was refitted or re-run on the test rows.

Confusion matrix of k-NN (rows = true class):

| | pred cherry | pred maple | pred walnut | recall |
|---|---|---|---|---|
| **cherry** (33) | 30 | 1 | 2 | 0.91 |
| **maple** (42) | 1 | 40 | 1 | 0.95 |
| **walnut** (24) | 4 | 1 | 19 | 0.79 |

## Test vs. cross-validation

- The test balanced accuracy (0.884) is **0.061 below the CV estimate** (0.946 ± 0.030), about two CV standard deviations.
- With only 99 test rows, one wrong prediction moves the balanced accuracy by 0.008–0.014 (depending on the class). The 95 % CI of the test balanced accuracy is 0.813–0.948 (bootstrap standard error 0.034), so the CV estimate of 0.946 lies just inside it: the gap is consistent with noise from a small test set, though at the edge of it.
- The test set also caught **3 of the 8 suspicious rows** from the data report (about 1.6 expected), and k-NN got all three wrong:
  - test lines 386 and 497 have the same measurements as training lines 511 and 504, but the opposite label. A model that learned the training label predicts the test row wrong, so these two rows are expected errors for any model that fits the training data closely;
  - line 216 is a "cherry" inside the maple cluster and was predicted maple.
- The weakest class is walnut (recall 0.79): 4 walnuts were called cherry, all in the region where the two classes overlap (height 10–12, width 5–6).

## Files in `results/`

- `cv_<model>.csv`, `cv_summary.csv`: the search.
- `test_predictions.csv`: every test row with its file line, measurements, true class, k-NN prediction, baseline prediction and whether k-NN was right.
- `test_score.csv`: test scores of k-NN and the baseline next to their CV scores.
- `test_confusion.csv`: the k-NN confusion matrix.
- `test_ci.csv`: 95 % bootstrap intervals of the k-NN test scores.
