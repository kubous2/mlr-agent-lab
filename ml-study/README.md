# ml-study

Study of the leaves dataset.

## Data

- File: [`../data/leaves.csv`](../data/leaves.csv)
- Source: <https://tomasvicar.github.io/MLR-public/data/leaves.csv>
- Retrieved: 2026-10-14

### Structure

CSV with a header row, comma-separated, UTF-8 without BOM, LF line endings.
518 rows, 3 columns, no missing values.

| Column | Type | Range | Notes |
|---|---|---|---|
| `Leaf height` | float | 3.84 – 25.50 | unit not stated in the source |
| `Leaf width` | float | 2.50 – 29.70 | unit not stated in the source |
| `Tree type` | string | `maple`, `cherry`, `walnut` | class label |

Class counts (not balanced):

| Tree type | Rows | Height mean ± std | Width mean ± std |
|---|---|---|---|
| maple | 216 | 13.48 ± 3.41 | 15.24 ± 4.05 |
| cherry | 173 | 9.33 ± 2.18 | 5.33 ± 1.92 |
| walnut | 129 | 15.32 ± 2.40 | 7.57 ± 2.13 |

Things to keep in mind:

- Maple leaves are wide, walnut leaves are tall and narrow, and cherry leaves are small,
  so the two measurements should separate the classes well.
- Precision is mixed: most rows have 1–2 decimals, but roughly file lines 102–303
  have 5–7 significant digits, which suggests the data was assembled from more than one source.
- 26 rows are exact duplicates of another row.
- A few cherry rows lie far from the rest of their class, e.g. `16.9,21.6,cherry`
  (width 21.6 vs. a cherry mean of 5.3), so they may be outliers or mislabelled.

## Environment

This folder is a [uv](https://docs.astral.sh/uv/) project (Python 3.12, pinned in `.python-version`).

```sh
cd ml-study
uv sync               # create .venv from uv.lock
uv run python ...     # run code in the environment
uv add <package>      # add a dependency
```
