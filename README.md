# pandas-eda-check

`pandas-eda-check` is a lightweight utility for quickly summarizing data types,
unique values, completeness, and missingness in a pandas DataFrame.

## Features

- One-row-per-column data quality report
- Dataset-level missing-data summary
- Optional filtering to columns with missing values
- Sorting by missing percentage, missing count, unique count, or data type
- Report metadata stored in `DataFrame.attrs`
- Safe handling of empty DataFrames and duplicate column names

## Installation

```bash
pip install pandas-eda-check
```

Python 3.9 or newer is required.

## Usage

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

## Parameters

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

### Sorting

```python
check(df, sort_by="missing_pct")
check(df, sort_by="missing_count")
check(df, sort_by="unique")
check(df, sort_by="dtype", ascending=True)
```

### Columns with missing values only

```python
missing_columns = check(df, include_complete=False)
```

### Suppress console output

```python
report = check(df, display=False)
```

### Report metadata

```python
report.attrs["shape"]
report.attrs["total_missing_cells"]
report.attrs["rows_with_missing"]
report.attrs["overall_missing_percent"]
```

## Development

Install the package and development tools in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Run the tests:

```bash
python -m pytest
```

Build and validate the distribution:

```bash
python -m build
python -m twine check dist/*
```

## License

MIT License. See [LICENSE](LICENSE).
