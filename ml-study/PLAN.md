# Plan: leaf classifier study (ml-study)

## Context

`ml-study/SPEC.md` defines the task: find the classifier that best predicts the tree type (maple, cherry or walnut) from leaf height and width in `data/leaves.csv`, and measure it once on held-out data. Before running the full search, we start with the baseline and a small trial run with one model and a tiny search, end to end, to check the pipeline. On approval, this plan is saved as `ml-study/PLAN.md`.

## Decisions for the open points in SPEC.md

SPEC.md leaves 12 points open. Unless you change them, these are the choices:

- **Output:** one script, `ml-study/train_models.py`. It prints to the console, and the full run also writes `ml-study/RESULTS.md`. No figures.
- **"Simpler" order for the 0.01 rule:** baseline < logistic regression < decision tree < k-NN < SVM < random forest. Every model within 0.01 of the best mean CV score counts as tied, and the simplest of them wins.
- **No class weights.** scikit-learn defaults.
- **Features:** raw `Leaf height` and `Leaf width` only.
- **Scaling:** `StandardScaler` in every pipeline, including trees and the baseline, where it has no effect.
- **Model settings:** defaults for everything not in the spec (k-NN uniform weights, Euclidean distance; logistic regression `max_iter=1000`).
- **Refit:** the chosen settings are refitted on the whole training part (the GridSearchCV default).
- **CV report:** mean ± standard deviation per model; selection uses the mean only.
- **Ties within one grid:** scikit-learn keeps the first.
- **Label noise:** list the 3 conflicting cherry/walnut pairs and the 2 cherry rows inside the maple cluster (file lines 216 and 392), say whether each landed in train or test, and give the prediction for those in test.

## Keeping the test set unused until the end

The spec says the test set is used once, at the end. "End to end" in the trial therefore stops short of it: the trial does the same split, but evaluates on a **validation slice** cut from the training part (20 %, stratified, seed 42). The 99 test rows are touched only by the final run.

## Steps

1. ✅ **Save the plan** to `ml-study/PLAN.md`.
2. ✅ **Data and split** in `ml-study/train_models.py`:
   - Reuse `load()`, `FEATURES`, `LABEL` and `HERE` from `ml-study/explore_leaves.py`. `load()` already sets the index to the file line number, which is used to trace the label-noise rows.
   - `deduplicate(df)`: `drop_duplicates(keep="first")`. Check: 492 rows (maple 208, cherry 162, walnut 122).
   - `split(df)`: `train_test_split(test_size=0.2, stratify=y, random_state=42)`. Check: 393 train / 99 test (scikit-learn rounds the 98.4-row test part up), with class shares within one row of the full data.
3. ✅ **Search builder:** `make_search(model, grid)` returns a `GridSearchCV` with `Pipeline([("scale", StandardScaler()), ("model", model)])`, `StratifiedKFold(5, shuffle=True, random_state=42)` and `scoring="balanced_accuracy"`. Every model goes through this one function.
4. ✅ **Baseline:** `DummyClassifier(strategy="most_frequent")` through `make_search` with an empty grid. CV balanced accuracy is exactly 0.333.
5. ✅ **Trial run** (`--trial`): baseline plus logistic regression with `C` in {0.1, 1, 10}, fitted on 314 rows and checked on a 79-row validation slice of the training part. CV 0.922, validation slice 0.946. Console output only.
6. ✅ **Checkpoint:** trial output reviewed; test size kept at 20 % (99 rows).
7. ✅ **Full search** (`uv run python ml-study/train_models.py --search`): all six models with the grids from SPEC.md, tuned with 5-fold CV on the full 393-row training part; the validation slice is not used and the test set is not touched. Every grid point is saved to `ml-study/results/cv_<model>.csv` (parameters, mean, std, rank and all five fold scores), and the per-model summary with the chosen model to `ml-study/results/cv_summary.csv`. Runs in about 45 s.
8. ✅ **Final evaluation** (`--final`): repeats the search on the training part (checked against `cv_summary.csv`), refits k-NN (k = 9) on the 393 training rows and evaluates it and the baseline once on the 99 test rows. Saves `results/test_predictions.csv`, `results/test_score.csv` and `results/test_confusion.csv`; reports the suspicious rows; full write-up in `ml-study/RESULTS.md`. The script refuses a second `--final` once `test_score.csv` exists.

**Test result:** k-NN balanced accuracy 0.884 (accuracy 0.899, macro F1 0.887) vs. 0.946 ± 0.030 in CV; baseline 0.333. See `RESULTS.md` for the comparison.

## Cross-validation results (training part only)

| Model | Grid points | CV balanced accuracy (mean ± std) | Best settings | vs. baseline |
|---|---|---|---|---|
| baseline | 1 | 0.333 ± 0.000 | – | – |
| logistic regression | 7 | 0.929 ± 0.043 | C = 1 | +0.596 |
| decision tree | 10 | 0.931 ± 0.018 | max_depth = 5 | +0.598 |
| **k-NN** | 30 | **0.946 ± 0.030** | n_neighbors = 9 | +0.613 |
| SVM (RBF) | 36 | 0.942 ± 0.030 | C = 100, gamma = 0.1 | +0.609 |
| random forest | 24 | 0.938 ± 0.027 | max_depth = 8, min_samples_leaf = 1 | +0.605 |

- **Chosen: k-NN with k = 9.** It has the best mean, and SVM and random forest are within 0.01 of it, so the tie rule picks the simplest of the three, which is k-NN again.
- The gaps between the five real models (0.929–0.946) are smaller than one CV standard deviation (0.02–0.04), so they are practically equivalent on this data.
- No best setting sits on the edge of its grid, so the ranges in SPEC.md were wide enough.

## Files

- New: `ml-study/train_models.py`, `ml-study/PLAN.md`, `ml-study/results/*.csv`, `ml-study/RESULTS.md`.
- Reused, unchanged: `ml-study/explore_leaves.py` (`load`, `FEATURES`, `LABEL`, `HERE`).
- Unchanged: `data/leaves.csv`. `ml-study/SPEC.md` changed only to correct the test size to 99 rows.
- Environment: the existing root uv project (pandas, scikit-learn, matplotlib); no new packages.

## Verification

- Assertions inside the script: 492 rows after deduplication; 393/99 split; train and test indices disjoint; in `--trial` mode, no test row is passed to any fit, predict or score. In `--search` mode the test rows are excluded by construction: only the training part is passed to the searches, and nothing is predicted or scored outside CV.
- Baseline CV balanced accuracy = 0.333 (checked in the trial and the full search).
- CSV hash is the same before and after each run.
- Running `--trial` twice gives identical numbers (fixed seeds). In the result CSVs only `mean_fit_time` changes between runs.
- Nothing is committed or pushed unless you ask.
