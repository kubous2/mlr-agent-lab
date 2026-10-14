"""Train and compare leaf classifiers as described in ml-study/SPEC.md and ml-study/PLAN.md.

Run from the repository root:
    uv run python ml-study/train_models.py --trial   # baseline + tiny logistic-regression search;
                                                     # evaluated on a validation slice, test set untouched
    uv run python ml-study/train_models.py --search  # full grid search of all models on the training part;
                                                     # every grid point saved to ml-study/results/, test set untouched
    uv run python ml-study/train_models.py --final   # the single test-set evaluation of the chosen model
                                                     # (refuses to run again once test_score.csv exists)
    uv run python ml-study/train_models.py --ci      # bootstrap confidence intervals of the test scores,
                                                     # from the saved test predictions only (no refit, no predict)
The CSV is only read, never modified.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from explore_leaves import FEATURES, HERE, LABEL, load

RESULTS = HERE / "results"
SEED = 42
TEST_SIZE = 0.2
CLASSES = ["cherry", "maple", "walnut"]


# ---------- Data ----------

def deduplicate(df):
    """Drop exact duplicate rows, keeping the first; the index (file line number) is preserved."""
    out = df.drop_duplicates(keep="first")
    assert len(out) == 492, len(out)
    assert out[LABEL].value_counts().to_dict() == {"maple": 208, "cherry": 162, "walnut": 122}
    return out


def split(df):
    """Stratified split into (train, held-out) parts."""
    train, held = train_test_split(df, test_size=TEST_SIZE, stratify=df[LABEL], random_state=SEED)
    assert train.index.intersection(held.index).empty
    full_share = df[LABEL].value_counts(normalize=True)
    for part in (train, held):
        counts = part[LABEL].value_counts()
        assert ((counts - full_share * len(part)).abs() <= 1).all(), counts
    return train, held


# ---------- Models ----------

def make_search(model, grid):
    """Grid search with scaling inside the pipeline and stratified 5-fold CV on balanced accuracy."""
    pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    param_grid = {f"model__{name}": values for name, values in grid.items()}
    return GridSearchCV(pipe, param_grid, cv=cv, scoring="balanced_accuracy")


TRIAL_MODELS = {
    "baseline": (DummyClassifier(strategy="most_frequent"), {}),
    "logistic regression": (LogisticRegression(max_iter=1000), {"C": [0.1, 1, 10]}),
}

# Grids from SPEC.md. Insertion order is the "simpler first" order used for tie-breaking.
FULL_MODELS = {
    "baseline": (DummyClassifier(strategy="most_frequent"), {}),
    "logistic regression": (LogisticRegression(max_iter=1000),
                            {"C": np.logspace(-3, 3, 7)}),
    "decision tree": (DecisionTreeClassifier(random_state=SEED),
                      {"max_depth": list(range(1, 11))}),
    "k-NN": (KNeighborsClassifier(),
             {"n_neighbors": list(range(1, 31))}),
    "SVM (RBF)": (SVC(kernel="rbf"),
                  {"C": np.logspace(-2, 3, 6), "gamma": np.logspace(-3, 2, 6)}),
    "random forest": (RandomForestClassifier(n_estimators=300, random_state=SEED),
                      {"max_depth": [2, 4, 6, 8, 10, None], "min_samples_leaf": [1, 2, 5, 10]}),
}
TIE_MARGIN = 0.01


def tune(models, train):
    """Fit every search on the training rows; return {name: fitted search}."""
    X, y = train[FEATURES], train[LABEL]
    searches = {}
    for name, (model, grid) in models.items():
        searches[name] = make_search(model, grid).fit(X, y)
    return searches


def cv_summary(searches):
    rows = []
    for name, s in searches.items():
        i = s.best_index_
        params = {k.removeprefix("model__"): (v.item() if hasattr(v, "item") else v)  # numpy -> plain number
                  for k, v in s.best_params_.items()}
        rows.append({"model": name,
                     "cv_mean": s.cv_results_["mean_test_score"][i],
                     "cv_std": s.cv_results_["std_test_score"][i],
                     "best_params": params or "-"})
    return pd.DataFrame(rows).set_index("model")


def choose(summary):
    """SPEC rule: best mean CV score; any model within TIE_MARGIN of it counts as tied, simplest wins."""
    best = summary["cv_mean"].max()
    tied = [name for name in summary.index if summary.loc[name, "cv_mean"] >= best - TIE_MARGIN]
    return tied[0], tied  # summary rows follow the simpler-first order of FULL_MODELS


def save_trials(name, search):
    """Write every grid point of one search (all CV fold scores) to ml-study/results/cv_<model>.csv."""
    RESULTS.mkdir(exist_ok=True)
    res = pd.DataFrame(search.cv_results_)
    params = pd.DataFrame(list(res["params"])).rename(columns=lambda c: c.removeprefix("model__"))
    folds = [c for c in res.columns if c.startswith("split") and c.endswith("_test_score")]
    out = pd.concat([params, res[["mean_test_score", "std_test_score", "rank_test_score", *folds,
                                  "mean_fit_time"]]], axis=1)
    out.insert(0, "model", name)
    path = RESULTS / f"cv_{name.split(' (')[0].lower().replace(' ', '_').replace('-', '')}.csv"
    out.sort_values("rank_test_score").to_csv(path, index=False)
    return path


def evaluate(search, rows):
    """Scores of a refitted search on the given rows."""
    y_true = rows[LABEL]
    y_pred = search.predict(rows[FEATURES])
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "confusion": pd.DataFrame(confusion_matrix(y_true, y_pred, labels=CLASSES),
                                  index=[f"true {c}" for c in CLASSES],
                                  columns=[f"pred {c}" for c in CLASSES]),
    }


def print_scores(name, scores):
    print(f"\n{name}: balanced accuracy {scores['balanced_accuracy']:.3f}, "
          f"accuracy {scores['accuracy']:.3f}, macro F1 {scores['macro_f1']:.3f}")
    print(scores["confusion"].to_string())


# ---------- Runs ----------

def run_trial():
    df = deduplicate(load())
    train, test = split(df)
    assert (len(train), len(test)) == (393, 99)  # 20 % of 492 = 98.4, rounded up for the test part
    print(f"Rows after deduplication: {len(df)}; train {len(train)}, test {len(test)} (test not used in trial)")

    # Validation slice cut from the training part only, so the test rows stay unseen.
    fit_rows, val_rows = split(train)
    used = fit_rows.index.union(val_rows.index)
    assert used.intersection(test.index).empty
    print(f"Trial: fit on {len(fit_rows)} rows, validate on {len(val_rows)} rows")

    searches = tune(TRIAL_MODELS, fit_rows)
    summary = cv_summary(searches)
    print("\nCross-validated balanced accuracy (5-fold, fit rows):")
    print(summary.to_string(float_format="{:.3f}".format))
    assert abs(summary.loc["baseline", "cv_mean"] - 1 / 3) < 1e-9

    print("\nAll CV scores of the logistic-regression grid:")
    lr = searches["logistic regression"].cv_results_
    for params, mean, std in zip(lr["params"], lr["mean_test_score"], lr["std_test_score"]):
        print(f"  C={params['model__C']:<5} {mean:.3f} ± {std:.3f}")

    for name, s in searches.items():
        print_scores(f"Validation slice, {name}", evaluate(s, val_rows))


def search_all(train):
    """Tune every model on the training rows only; return (searches, summary, chosen, tied)."""
    searches = tune(FULL_MODELS, train)
    summary = cv_summary(searches)
    summary.insert(0, "grid_points", [len(s.cv_results_["params"]) for s in searches.values()])
    summary["vs_baseline"] = summary["cv_mean"] - summary.loc["baseline", "cv_mean"]
    chosen, tied = choose(summary)
    summary["chosen"] = ["yes" if n == chosen else "" for n in summary.index]
    return searches, summary, chosen, tied


def run_search():
    df = deduplicate(load())
    train, test = split(df)
    assert (len(train), len(test)) == (393, 99)
    print(f"Rows after deduplication: {len(df)}; tuning on {len(train)} training rows, "
          f"test set ({len(test)} rows) not used")

    searches, summary, chosen, tied = search_all(train)  # only training rows are passed in
    for name, s in searches.items():
        print(f"  {name:20} {len(s.cv_results_['params']):3} grid points -> {save_trials(name, s)}")
    summary.to_csv(RESULTS / "cv_summary.csv")

    print("\nBest cross-validated balanced accuracy per model (5-fold, training part):")
    print(summary.to_string(float_format="{:.3f}".format))
    print(f"\nWithin {TIE_MARGIN} of the best: {', '.join(tied)} -> chosen (simplest): {chosen}")
    print(f"Saved {RESULTS / 'cv_summary.csv'}")


TEST_SCORE = RESULTS / "test_score.csv"
# File lines of the suspicious rows from the data report (see SPEC.md).
NOISE_ROWS = {386: "same measurements as 511, labels differ", 511: "same measurements as 386, labels differ",
              497: "same measurements as 504, labels differ", 504: "same measurements as 497, labels differ",
              416: "same measurements as 456, labels differ", 456: "same measurements as 416, labels differ",
              216: "cherry inside the maple cluster", 392: "cherry inside the maple cluster"}


def run_final(force=False):
    """The one evaluation on the test set, of the chosen model and the baseline."""
    if TEST_SCORE.exists() and not force:
        raise SystemExit(f"{TEST_SCORE} exists: the test set has already been used once (SPEC.md). "
                         "Use --force only if you really mean to evaluate again.")
    raw = load()
    df = deduplicate(raw)
    train, test = split(df)
    assert (len(train), len(test)) == (393, 99)

    # Selection uses the training part only; it must agree with the saved search results.
    searches, summary, chosen, _ = search_all(train)
    saved = pd.read_csv(RESULTS / "cv_summary.csv", index_col="model")
    assert saved.index[saved["chosen"] == "yes"].tolist() == [chosen], "choice differs from cv_summary.csv"
    assert np.allclose(saved["cv_mean"], summary["cv_mean"]), "CV scores differ from cv_summary.csv"
    print(f"Chosen by cross-validation: {chosen} {summary.loc[chosen, 'best_params']}, "
          f"refitted on {len(train)} training rows")

    # ---- the single use of the test rows ----
    results = {name: evaluate(searches[name], test) for name in ("baseline", chosen)}
    pred = test[FEATURES + [LABEL]].rename(columns={LABEL: "true"})
    pred.insert(0, "line", pred.index)
    pred["predicted"] = searches[chosen].predict(test[FEATURES])
    pred["baseline_predicted"] = searches["baseline"].predict(test[FEATURES])
    pred["correct"] = pred["true"] == pred["predicted"]

    scores = pd.DataFrame([{"model": name, "params": summary.loc[name, "best_params"],
                            "test_balanced_accuracy": r["balanced_accuracy"],
                            "test_accuracy": r["accuracy"], "test_macro_f1": r["macro_f1"],
                            "cv_mean": summary.loc[name, "cv_mean"], "cv_std": summary.loc[name, "cv_std"],
                            "test_rows": len(test)} for name, r in results.items()])
    scores["test_minus_cv"] = scores["test_balanced_accuracy"] - scores["cv_mean"]

    pred.to_csv(RESULTS / "test_predictions.csv", index=False)
    results[chosen]["confusion"].to_csv(RESULTS / "test_confusion.csv")
    scores.to_csv(TEST_SCORE, index=False)

    for name, r in results.items():
        print_scores(f"Test set ({len(test)} rows), {name}", r)
    print("\nTest vs. cross-validation (balanced accuracy):")
    print(scores.set_index("model")[["cv_mean", "cv_std", "test_balanced_accuracy", "test_minus_cv"]]
          .to_string(float_format="{:.3f}".format))

    print("\nWrong predictions on the test set:")
    print(pred[~pred["correct"]].to_string(index=False))

    print("\nSuspicious rows from the data report:")
    for line, why in NOISE_ROWS.items():
        if line in test.index:
            p = pred.loc[line]
            where = f"test, predicted {p['predicted']} ({'correct' if p['correct'] else 'wrong'})"
        else:
            where = "train" if line in train.index else "dropped as duplicate"
        print(f"  line {line}: {raw.loc[line, FEATURES].tolist()} {raw.loc[line, LABEL]:6} ({why}) -> {where}")

    print(f"\nSaved {TEST_SCORE}, {RESULTS / 'test_predictions.csv'}, {RESULTS / 'test_confusion.csv'}")


TEST_CI = RESULTS / "test_ci.csv"
N_BOOT = 10_000


def run_ci():
    """95 % percentile-bootstrap intervals of the chosen model's test scores, read from test_predictions.csv.

    Resamples are drawn within each true class, so every resample has the class counts of the test set.
    """
    pred = pd.read_csv(RESULTS / "test_predictions.csv")
    scores = pd.read_csv(TEST_SCORE, index_col="model")
    chosen = scores.index[scores.index != "baseline"][0]
    metrics = {"balanced_accuracy": balanced_accuracy_score, "accuracy": accuracy_score,
               "macro_f1": lambda t, p: f1_score(t, p, average="macro")}

    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(pred["true"] == c) for c in CLASSES]
    true, predicted = pred["true"].to_numpy(), pred["predicted"].to_numpy()
    boot = {name: [] for name in metrics}
    for _ in range(N_BOOT):
        i = np.concatenate([rng.choice(g, len(g)) for g in groups])
        for name, fn in metrics.items():
            boot[name].append(fn(true[i], predicted[i]))

    rows = []
    for name, fn in metrics.items():
        score = fn(true, predicted)
        assert np.isclose(score, scores.loc[chosen, f"test_{name}"]), name  # same predictions as --final
        low, high = np.percentile(boot[name], [2.5, 97.5])
        rows.append({"model": chosen, "metric": name, "score": score, "ci_low": low, "ci_high": high,
                     "bootstrap_se": np.std(boot[name]), "resamples": N_BOOT})
    ci = pd.DataFrame(rows)
    ci.to_csv(TEST_CI, index=False)

    cv_mean = scores.loc[chosen, "cv_mean"]
    print(f"Test scores of {chosen} with 95 % bootstrap intervals ({N_BOOT} stratified resamples, seed {SEED}):")
    print(ci.set_index("metric")[["score", "ci_low", "ci_high", "bootstrap_se"]]
          .to_string(float_format="{:.3f}".format))
    print(f"\nShare of resamples with balanced accuracy >= the CV mean ({cv_mean:.3f}): "
          f"{np.mean(np.array(boot['balanced_accuracy']) >= cv_mean):.3f}")
    print(f"Saved {TEST_CI}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--trial", action="store_true", help="baseline + tiny search, test set untouched")
    mode.add_argument("--search", action="store_true", help="full grid search, test set untouched")
    mode.add_argument("--final", action="store_true", help="the one evaluation of the chosen model on the test set")
    mode.add_argument("--ci", action="store_true", help="bootstrap intervals from the saved test predictions")
    parser.add_argument("--force", action="store_true", help="allow --final to run again")
    args = parser.parse_args()
    if args.trial:
        run_trial()
    elif args.search:
        run_search()
    elif args.ci:
        run_ci()
    else:
        run_final(force=args.force)


if __name__ == "__main__":
    main()
