# pandas-eda-check

`pandas-eda-check` is a lightweight utility for inspecting one pandas
DataFrame and comparing meaningful EDA profile changes between two DataFrames.

## Features

- `check(df)` creates a one-row-per-column data-quality report.
- `compare(reference, current)` reports structural, quality, and profile changes.
- Detects schema, missing-data, duplicate-rate, numeric-profile, datetime-range,
  and categorical changes.
- Does not align rows, compare individual cells, require matching indexes or
  shapes, depend on row order, or require a join key.
- Returns pandas DataFrames for straightforward programmatic use.
- Safely handles empty DataFrames, nullable dtypes, mixed object values,
  unhashable values, infinities, and all-null columns.

## Installation

```bash
pip install pandas-eda-check
```

Python 3.9 or newer and pandas 1.5 or newer are required.

## Inspect one DataFrame

```python
import pandas as pd

from pandas_eda_check import check

df = pd.DataFrame(
    {
        "name": ["Ada", "Bob", "Bob"],
        "age": [36, None, 29],
        "city": ["London", "Paris", None],
    }
)

report = check(df)
print(report)
```

Console summary:

```text
Data Shape: (3, 3)
Total Missing Cells: 2
Rows With Missing Values: 2
Overall Missing Percentage: 22.22%
```

Report:

```text
     Data Type  Unique Values  Values Present  Missing Count  Missing %
name    object              2               3              0       0.00
age    float64              2               2              1      33.33
city    object              2               2              1      33.33
```

The original DataFrame column names are used as the report index.

### `check()` parameters

```python
check(
    data,
    include_dtypes=True,
    include_complete=True,
    sort_by=None,
    ascending=False,
    round_digits=2,
    display=True,
)
```

| Parameter | Description |
| --- | --- |
| `data` | pandas DataFrame to inspect. |
| `include_dtypes` | Include the `Data Type` report column. |
| `include_complete` | Include columns that have no missing values. |
| `sort_by` | Sort by `missing_pct`, `missing_count`, `unique`, or `dtype`. |
| `ascending` | Use ascending order when sorting. |
| `round_digits` | Non-negative number of decimal places for percentages. |
| `display` | Print the dataset-level summary. |

```python
check(df, sort_by="missing_pct")
missing_columns = check(df, include_complete=False)
quiet_report = check(df, display=False)

quiet_report.attrs["shape"]
quiet_report.attrs["total_missing_cells"]
quiet_report.attrs["rows_with_missing"]
quiet_report.attrs["overall_missing_percent"]
```

## Compare two DataFrames

`compare()` answers: “How did the structure, data quality, and statistical
profile of the current dataset change compared with the reference dataset?”

It compares profiles by column name. It does not inspect matching-row cell
differences and does not use either DataFrame's index or row order. The inputs
may have different row counts, columns, shapes, indexes, or dtypes, and neither
input is mutated.

```python
import pandas as pd
from pandas_eda_check import compare

reference = pd.DataFrame({
    "age": [20, 25, 30, 35],
    "status": ["active", "active", "inactive", "active"],
})

current = pd.DataFrame({
    "age": [20, None, 42, 50, 55],
    "status": ["active", "pending", "pending", "active", "pending"],
    "source": ["web", "web", "mobile", "web", "mobile"],
})

report = compare(reference, current, display=False)

print(report["overview"])
print(report["schema_changes"])
print(report["column_changes"])
print(report["category_changes"])
```

Set `display=True` (the default) to print all sections in a fixed, plain-text
format. It works in terminals and notebooks without IPython or Jupyter.

### `compare()` parameters

```python
compare(
    reference,
    current,
    *,
    display=True,
    include_stable=False,
    missing_change_threshold=5.0,
    numeric_change_threshold=10.0,
    unique_change_threshold=20.0,
    category_limit=100,
    round_digits=2,
)
```

| Parameter | Description |
| --- | --- |
| `reference` | Baseline pandas DataFrame. |
| `current` | Newer pandas DataFrame compared with the baseline. |
| `display` | Print all report sections when `True`; never changes the returned report. |
| `include_stable` | Include stable rows in detailed sections. Overview counts always include them. |
| `missing_change_threshold` | Percentage-point threshold for missing, duplicate, and infinite-value rates. |
| `numeric_change_threshold` | Relative-percent threshold for numeric statistics and date ranges; percentage-point threshold for zero, negative, and dominant-category rates. |
| `unique_change_threshold` | Relative-percent threshold for unique metrics and category-set severity. |
| `category_limit` | Maximum unique count on each side for complete category-set comparison. |
| `round_digits` | Decimal places in report values. Status and severity use unrounded values. |

Examples:

```python
# Suppress output and access individual report DataFrames.
report = compare(reference, current, display=False)
dataset_changes = report["dataset_summary"]
schema_changes = report["schema_changes"]

# Include findings that remained below their applicable thresholds.
full_report = compare(
    reference,
    current,
    include_stable=True,
    display=False,
)
print(full_report["column_changes"])
```

### Report sections and stable columns

The returned dictionary always has exactly these five keys, in this order.
Every value is a pandas DataFrame, even when the section is empty.

| Section | Purpose | Columns |
| --- | --- | --- |
| `overview` | Counts all findings before stable rows are filtered, including statuses, severities, and schema-change totals. | `Metric`, `Value` |
| `dataset_summary` | Shape, total/overall missingness, and duplicate count/rate comparisons. | `Metric`, `Reference`, `Current`, `Absolute Change`, `Percent Change`, `Status`, `Severity`, `Note` |
| `schema_changes` | Added, removed, unchanged, and exact dtype-changed columns. | `Column`, `Change Type`, `Reference Dtype`, `Current Dtype`, `Status`, `Severity`, `Note` |
| `column_changes` | Long-format generic and type-specific metrics for common columns. | `Column`, `Column Type`, `Metric`, `Reference`, `Current`, `Absolute Change`, `Percent Change`, `Status`, `Severity`, `Note` |
| `category_changes` | New/removed values, dominant values and rates, and category-set completion status. | `Column`, `New Values`, `Removed Values`, `Reference Top Value`, `Current Top Value`, `Reference Top Percentage`, `Current Top Percentage`, `Dominant Percentage Point Change`, `Set Comparison`, `Status`, `Severity`, `Note` |

A finding is one row in a detailed section, not an overview row. Related
profile changes may appear in different sections. For example, `column_changes`
can report a categorical unique-count change while `category_changes` lists the
actual new or removed values.

### Status and severity rules

Statuses have these meanings:

- `Improved`: an objectively undesirable percentage decreased by at least its
  threshold (missing, duplicate, or infinite percentage).
- `Worsened`: one of those percentages increased by at least its threshold.
- `Changed`: a meaningful non-directional change, such as a schema, count,
  statistic, date, or category change.
- `Stable`: equal or below the applicable threshold.

Severities are `None`, `Low`, `Medium`, or `High`. For threshold-based metrics:

- Below 1× the threshold: `None`
- From 1× to below 2×: `Low`
- From 2× to below 4×: `Medium`
- At least 4×: `High`

Schema severities are fixed: added columns are `Medium`; removed columns and
exact dtype changes are `High`. A changed dominant category is at least
`Medium`. When a reference value is zero and the current value is nonzero,
relative percent change is undefined, the report stores `pd.NA`, adds a note,
and uses `Low` unless a more specific objective rule applies.

Calculations use full precision and are rounded only for report output.

- A percentage-point change compares rates directly: 10% to 18% is an
  8-percentage-point increase.
- Relative percent change is `(current - reference) / abs(reference) * 100`:
  10 to 18 is an 80% relative increase.
- Zero and negative-value percentages use percentage-point changes against
  `numeric_change_threshold`.

### Type-specific profiles

Every common column is compared for missing count/rate, present count, and
unique count/rate. Exact dtype changes are reported separately.

- Numeric columns add mean, median, sample standard deviation, minimum,
  maximum, zero/negative percentages, and infinite count/rate. Missing values
  and infinities are excluded from finite statistics.
- Datetime columns add earliest/latest date and range in days.
- Categorical-like columns (object, string, category, and boolean) add the
  first-seen most frequent value and its percentage. First-seen order also
  resolves frequency ties.

When exact dtypes differ but both have the same broad type (for example,
`int64` and `Int64`, or `object` and `string`), type-specific metrics are still
compared. When broad types differ, only the five common metrics are compared,
with an explanatory note.

Full new/removed category sets are calculated only when both unique counts are
less than or equal to `category_limit`. If either exceeds the limit, set
comparison is marked `Skipped`, values are not silently truncated, and
dominant-value metrics are still reported. This keeps comparisons lightweight
for high-cardinality columns.

### Conceptual differences

| Function | Purpose |
| --- | --- |
| `check(df)` | Profiles the structure and quality of one dataset. |
| `compare(reference, current)` | Profiles meaningful structural, quality, and statistical change between dataset versions. |
| `pandas.DataFrame.compare()` | Shows individual cell differences between similarly labeled DataFrames. |

These APIs serve different use cases: `compare()` in this package is intended
for dataset-level and column-profile EDA comparison without row matching.

## Development

Install the package and development tools in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Run the tests:

```bash
python -m pytest -q
```

Build and validate the distribution:

```bash
python -m build
python -m twine check dist/*
```

## License

MIT License. See [LICENSE](LICENSE).

## Author

**Ponkoj Shill**  
AI/ML researcher and Ph.D. candidate in Computer Science

- [GitHub](https://github.com/CS-Ponkoj)
- [Portfolio](https://ponkoj.com)
---------------
