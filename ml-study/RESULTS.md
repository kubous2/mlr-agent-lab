# Results: leaf classifier study

Setup as in [SPEC.md](SPEC.md) and [PLAN.md](PLAN.md): 492 rows after removing exact duplicates, a stratified split into 393 training rows and 99 test rows (seed 42), every model tuned with 5-fold stratified CV on the training rows only, balanced accuracy as the metric. The test set was used once, after the model was chosen.

Reproduce from the repository root with `uv run python ml-study/train_models.py --search`, then `--final` (which refuses to run a second time).

## Model choice (cross-validation, training rows only)

| Model | CV balanced accuracy (mean ± std) | Best settings |
|---|---|---|
| baseline (always "maple") | 0.333 ± 0.000 | – |
| logistic regression | 0.929 ± 0.043 | C = 1 |
| decision tree | 0.931 ± 0.018 | max_depth = 5 |
| **k-NN** | **0.946 ± 0.030** | n_neighbors = 9 |
| SVM (RBF) | 0.942 ± 0.030 | C = 100, gamma = 0.1 |
| random forest | 0.938 ± 0.027 | max_depth = 8, min_samples_leaf = 1 |

Chosen: **k-NN with k = 9**. It has the best mean; SVM and random forest are within 0.01, and k-NN is the simplest of the three. Every grid point is in `results/cv_*.csv`.

## Test set (99 rows, used once)

| Model | Balanced accuracy | Accuracy | Macro F1 |
|---|---|---|---|
| baseline | 0.333 | 0.424 | 0.199 |
| **k-NN (k = 9)** | **0.884** | **0.899** | **0.887** |

Confusion matrix of k-NN (rows = true class):

| | pred cherry | pred maple | pred walnut | recall |
|---|---|---|---|---|
| **cherry** (33) | 30 | 1 | 2 | 0.91 |
| **maple** (42) | 1 | 40 | 1 | 0.95 |
| **walnut** (24) | 4 | 1 | 19 | 0.79 |

## Test vs. cross-validation

- The test balanced accuracy (0.884) is **0.061 below the CV estimate** (0.946 ± 0.030), about two CV standard deviations.
- With only 99 test rows, one wrong prediction moves the balanced accuracy by 0.008–0.014 (depending on the class), and the standard error of the test score is about 0.03. So part of the gap is noise from a small test set.
- The test set also caught **3 of the 8 suspicious rows** from the data report (about 1.6 expected), and k-NN got all three wrong:
  - lines 386 and 497 have an identical twin in the training rows with the other label (511 and 504), so no model could get both twin rows right;
  - line 216 is a "cherry" inside the maple cluster and was predicted maple.
- Leaving those 3 rows out (a diagnostic only, not used for any choice) gives a balanced accuracy of 0.915 and accuracy of 0.927 on the remaining 96 rows.
- The weakest class is walnut (recall 0.79): 4 walnuts were called cherry, all in the region where the two classes overlap (height 10–12, width 5–6).

## Files in `results/`

- `cv_<model>.csv`, `cv_summary.csv`: the search.
- `test_predictions.csv`: every test row with its file line, measurements, true class, k-NN prediction, baseline prediction and whether k-NN was right.
- `test_score.csv`: test scores of k-NN and the baseline next to their CV scores.
- `test_confusion.csv`: the k-NN confusion matrix.
