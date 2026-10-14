"""Facts and tables for report.html and slides.qmd, read only from saved files (no retyped numbers).

Sources: data/leaves.csv, ml-study/results/*.csv (including test_ci.csv), ml-study/figures/scatter.png.
"""
import ast
import base64

import pandas as pd

from explore_leaves import CSV, FIGURE, HERE, LABEL, load
from train_models import NOISE_ROWS, SEED, TEST_SIZE, TIE_MARGIN

RESULTS = HERE / "results"
SCATTER = FIGURE


def f3(x):
    return f"{x:.3f}"


def params_text(raw):
    """"{'n_neighbors': 9}" -> "n_neighbors = 9"; "-" stays "–"."""
    if not isinstance(raw, str) or not raw.startswith("{"):
        return "–"
    return ", ".join(f"{k} = {v}" for k, v in ast.literal_eval(raw).items())


def load_all():
    """Everything the report and the slides need, as one dict."""
    raw = load()
    cv = pd.read_csv(RESULTS / "cv_summary.csv", index_col="model")
    test = pd.read_csv(RESULTS / "test_score.csv", index_col="model")
    conf = pd.read_csv(RESULTS / "test_confusion.csv", index_col=0)
    pred = pd.read_csv(RESULTS / "test_predictions.csv")
    ci = pd.read_csv(RESULTS / "test_ci.csv", index_col="metric")

    chosen = cv.index[cv["chosen"] == "yes"][0]
    best = cv["cv_mean"].max()
    margin = TIE_MARGIN
    tied = [m for m in cv.index if cv.loc[m, "cv_mean"] >= best - margin]
    real = cv.drop(index="baseline")

    totals = conf.sum(axis=1)
    recall = pd.Series([conf.iloc[i, i] / totals.iloc[i] for i in range(len(conf))],
                       index=[i.removeprefix("true ") for i in conf.index])

    n_dedup = len(raw.drop_duplicates())
    n_test = int(test.loc[chosen, "test_rows"])
    folds = [c for c in pd.read_csv(RESULTS / "cv_baseline.csv", nrows=0).columns
             if c.startswith("split") and c.endswith("_test_score")]
    noise_in_test = pred[pred["line"].isin(NOISE_ROWS)]
    return {
        "seed": SEED, "test_pct": round(TEST_SIZE * 100), "n_folds": len(folds),
        "n_noise": len(NOISE_ROWS), "noise_in_test": noise_in_test,
        "noise_wrong": int((~noise_in_test["correct"]).sum()),
        "n_models": len(cv) - 1, "n_grid": int(cv["grid_points"].sum()),
        "raw": raw, "cv": cv, "test": test, "conf": conf, "pred": pred,
        "csv_name": f"{CSV.parent.name}/{CSV.name}",
        "n_rows": len(raw), "n_dupes": len(raw) - n_dedup, "n_dedup": n_dedup,
        "n_test": n_test, "n_train": n_dedup - n_test,
        "class_counts": raw[LABEL].value_counts(),
        "chosen": chosen, "chosen_params": params_text(cv.loc[chosen, "best_params"]),
        "tied": tied, "margin": margin,
        "cv_mean": cv.loc[chosen, "cv_mean"], "cv_std": cv.loc[chosen, "cv_std"],
        "real_min": real["cv_mean"].min(), "real_max": real["cv_mean"].max(),
        "real_std_min": real["cv_std"].min(), "real_std_max": real["cv_std"].max(),
        "base_cv": cv.loc["baseline", "cv_mean"],
        "base_test": test.loc["baseline", "test_balanced_accuracy"],
        "base_acc": test.loc["baseline", "test_accuracy"],
        "test_ba": test.loc[chosen, "test_balanced_accuracy"],
        "test_acc": test.loc[chosen, "test_accuracy"],
        "test_f1": test.loc[chosen, "test_macro_f1"],
        "gap": test.loc[chosen, "test_minus_cv"],
        "ci": ci, "n_boot": int(ci["resamples"].iloc[0]),
        "ci_ba": (ci.loc["balanced_accuracy", "ci_low"], ci.loc["balanced_accuracy", "ci_high"]),
        "n_wrong": int((~pred["correct"]).sum()),
        "recall": recall,
        "weakest": recall.idxmin(),
        "weakest_as": weakest_confusion(conf, recall.idxmin()),
    }


def weakest_confusion(conf, cls):
    """(other class, count) the given true class is most often predicted as."""
    row = conf.loc[f"true {cls}"].drop(f"pred {cls}")
    return row.idxmax().removeprefix("pred "), int(row.max())


def cv_table(d):
    """Model comparison: CV mean ± std, best settings, gain over the baseline."""
    cv = d["cv"]
    return pd.DataFrame({
        "Model": cv.index,
        "Grid points": cv["grid_points"].values,
        "CV balanced accuracy": [f"{f3(m)} ± {f3(s)}" for m, s in zip(cv["cv_mean"], cv["cv_std"])],
        "Best settings": [params_text(p) for p in cv["best_params"]],
        "vs. baseline": ["–" if m == "baseline" else f"+{f3(v)}" for m, v in zip(cv.index, cv["vs_baseline"])],
        "Chosen": ["✔" if c == "yes" else "" for c in cv["chosen"].fillna("")],
    })


def test_table(d):
    """Test scores next to the CV scores, chosen model and baseline."""
    t = d["test"]
    return pd.DataFrame({
        "Model": t.index,
        "CV balanced accuracy": [f"{f3(m)} ± {f3(s)}" for m, s in zip(t["cv_mean"], t["cv_std"])],
        "Test balanced accuracy": [f3(v) for v in t["test_balanced_accuracy"]],
        "95 % CI": [ci_text(d, m) for m in t.index],
        "Test accuracy": [f3(v) for v in t["test_accuracy"]],
        "Test macro F1": [f3(v) for v in t["test_macro_f1"]],
        "Test − CV": [f"{v:+.3f}" for v in t["test_minus_cv"]],
    })


def ci_text(d, model):
    """Bootstrap interval of the test balanced accuracy; only computed for the chosen model."""
    if model != d["chosen"]:
        return "–"
    low, high = d["ci_ba"]
    return f"{f3(low)}–{f3(high)}"


def confusion_table(d):
    """Confusion matrix with row totals and recall."""
    conf = d["conf"].copy()
    conf.index = [i.removeprefix("true ") + f" ({n})" for i, n in zip(conf.index, conf.sum(axis=1))]
    conf.columns = [c.replace("pred ", "predicted ") for c in conf.columns]
    conf["recall"] = [f3(r) for r in d["recall"]]
    conf.index.name = "true class (rows)"
    return conf


def html_table(df):
    """DataFrame -> HTML table without the index."""
    return df.to_html(index=False, escape=True, border=0, classes="tbl")


def confusion_html(d):
    """Confusion matrix as HTML; correct predictions (the diagonal) are highlighted."""
    t = confusion_table(d)
    head = "".join(f"<th>{c}</th>" for c in [t.index.name, *t.columns])
    rows = []
    for i, (name, row) in enumerate(t.iterrows()):
        cells = [f'<td class="hit">{v}</td>' if j == i else f"<td>{v}</td>" for j, v in enumerate(row)]
        rows.append(f"<tr><th>{name}</th>{''.join(cells)}</tr>")
    return f'<table class="tbl confusion"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def image_data_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
