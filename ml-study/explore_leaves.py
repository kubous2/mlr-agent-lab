"""Explore data/leaves.csv: columns, class counts, duplicates, outliers, scatter plot.

Run from anywhere with:  uv run python ml-study/explore_leaves.py
The CSV is only read, never modified.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "data" / "leaves.csv"
FIGURE = HERE / "figures" / "scatter.png"
FEATURES = ["Leaf height", "Leaf width"]
LABEL = "Tree type"
COLORS = {"maple": "#d1495b", "cherry": "#8e44ad", "walnut": "#2e8b57"}


def load():
    df = pd.read_csv(CSV)
    df.index = df.index + 2  # index = line number in the file (line 1 is the header)
    return df


def report_overview(df):
    print(f"File: {CSV}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumns:")
    for col in df.columns:
        print(f"  {col!r:16} dtype={df[col].dtype}, missing={df[col].isna().sum()}")
    print("\nLeaves per type:")
    print(df[LABEL].value_counts().to_string())
    print("\nFeature summary per type:")
    print(df.groupby(LABEL)[FEATURES].describe().round(2).T.to_string())


def report_duplicates(df):
    print("\n--- Duplicates ---")
    full = df[df.duplicated(keep=False)]
    groups = full.groupby(list(df.columns))
    extra = df.duplicated().sum()
    print(f"Identical rows (all 3 columns): {len(full)} rows in {groups.ngroups} groups, "
          f"{extra} of them are extra copies")
    for key, g in groups:
        print(f"  {key}  x{len(g)}  lines {list(g.index)}")

    feat = df[df.duplicated(FEATURES, keep=False)]
    by_feat = feat.groupby(FEATURES)[LABEL].agg(lambda s: sorted(set(s)))
    conflicts = by_feat[by_feat.map(len) > 1]
    print(f"\nRows sharing the same height+width: {len(feat)} rows in {len(by_feat)} groups")
    if conflicts.empty:
        print("Every such group has a single class: duplicates are always the same class.")
    else:
        print(f"{len(conflicts)} groups have DIFFERENT classes for the same measurements:")
        for (h, w), labels in conflicts.items():
            lines = list(feat[(feat[FEATURES[0]] == h) & (feat[FEATURES[1]] == w)].index)
            print(f"  height={h}, width={w}: {labels}  lines {lines}")


def report_unusual(df):
    print("\n--- Other checks ---")
    print("Non-positive values:", int((df[FEATURES] <= 0).sum().sum()))
    print("Class labels:", sorted(df[LABEL].unique()))

    # Rows far from their own class centre (robust z-score using median and MAD).
    z = df.groupby(LABEL)[FEATURES].transform(
        lambda s: (s - s.median()) / (1.4826 * (s - s.median()).abs().median()))
    far = df[(z.abs() > 4).any(axis=1)]
    print(f"Rows > 4 robust SD from their class median: {len(far)}")
    print(far.assign(z_height=z.loc[far.index, FEATURES[0]].round(1),
                     z_width=z.loc[far.index, FEATURES[1]].round(1)).to_string())

    decimals = df[FEATURES].astype(str).apply(
        lambda col: col.str.split(".").str[1].str.len()).max(axis=1)
    high = decimals[decimals >= 4]
    print(f"\nRows with 4+ decimals: {len(high)} (lines {high.index.min()}-{high.index.max()}); "
          f"other rows have at most {decimals[decimals < 4].max()} decimals")


def plot(df):
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for kind, g in df.groupby(LABEL):
        ax.scatter(g[FEATURES[1]], g[FEATURES[0]], s=18, alpha=0.65,
                   color=COLORS.get(kind), label=f"{kind} ({len(g)})", edgecolors="none")
    ax.set_xlabel("Leaf width")
    ax.set_ylabel("Leaf height")
    ax.set_title("Leaves: height vs. width by tree type")
    ax.legend(title="Tree type")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=120)
    print(f"\nSaved {FIGURE}")


def main():
    df = load()
    report_overview(df)
    report_duplicates(df)
    report_unusual(df)
    plot(df)


if __name__ == "__main__":
    main()
