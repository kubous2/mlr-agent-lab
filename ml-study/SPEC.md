Goal: Which classifier best predicts the tree type (maple, cherry or walnut) from leaf height and width, and how well does it do on data it has never seen?

Context: `data/leaves.csv` (from <https://tomasvicar.github.io/MLR-public/data/leaves.csv>, retrieved 2026-10-14) has 518 rows and 3 columns: `Leaf height` (float), `Leaf width` (float) and `Tree type` (`maple` 216, `cherry` 173, `walnut` 129). There are no missing values. 46 rows form 20 groups of identical rows (26 extra copies), and every group has a single class; the first three rows (lines 2–4) reappear at lines 26–28 and 65–67. Separately, 3 pairs of rows have the same measurements but different classes (cherry vs. walnut: lines 386/511, 497/504, 416/456), and two cherry rows sit inside the maple cluster (lines 216 and 392). Maple separates well by width; cherry and walnut overlap around height 11–16, width 5–8. Full report: `ml-study/explore_leaves.py` and `ml-study/README.md`.

Constraints:

- Duplicates: remove exact duplicate rows before splitting, keeping the first occurrence (in memory only; the CSV is never modified). This leaves 492 rows: maple 208, cherry 162, walnut 122. Why: the repeated block at lines 2–4 / 26–28 / 65–67 is clearly a copy artefact, so the copies carry no extra information, and keeping them would let the same row sit in both training and test data and inflate the score. Removing them costs only 5 % of the rows and allows ordinary stratified splitting, which is simpler than group-aware splitting. The 3 cherry/walnut pairs with conflicting labels and the two suspicious cherry rows are kept unchanged: they are not identical rows, and there is no way to know which label is right. They are listed in the results as known label noise.
- Test set: 20 % of the deduplicated data (99 rows; scikit-learn rounds 98.4 up), stratified by class, `random_state=42`, split once at the start and used once at the end, after the final model has been chosen.
- Models: majority-class baseline (`DummyClassifier`), k-NN, logistic regression, decision tree, SVM (RBF kernel) and random forest.
- Hyperparameters:
  - k-NN: `n_neighbors` 1–30.
  - Logistic regression: `C` in 10⁻³ … 10³ (7 values, log scale).
  - Decision tree: `max_depth` 1–10.
  - SVM (RBF): `C` in 10⁻² … 10³ and `gamma` in 10⁻³ … 10² (6 × 6 values, log scale).
  - Random forest: 300 trees, `max_depth` in {2, 4, 6, 8, 10, None} and `min_samples_leaf` in {1, 2, 5, 10}.
  - All models with a random component use `random_state=42`.
- Tuning: 5-fold stratified cross-validation (shuffled, `random_state=42`) on the training part only, with `StandardScaler` inside a `Pipeline` so scaling is fitted on each training fold. Exhaustive grid search; every grid above has at most 36 combinations, within a budget of 50 per model.
- Metric: balanced accuracy, because the classes are unbalanced (42 % / 33 % / 25 % after deduplication); the baseline scores 0.33. The final model is the one with the best mean cross-validated balanced accuracy, never chosen by the test score; if two models are within 0.01, pick the simpler one. On the test set, report balanced accuracy, accuracy, macro F1 and the confusion matrix for the chosen model and the baseline.
