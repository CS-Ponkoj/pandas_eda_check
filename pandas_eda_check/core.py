"""Core DataFrame summary functionality."""

from numbers import Integral
from typing import Optional

import pandas as pd


_SORT_COLUMNS = {
    "missing_pct": "Missing %",
    "missing_count": "Missing Count",
    "unique": "Unique Values",
    "dtype": "Data Type",
}


def check(
    data: pd.DataFrame,
    include_dtypes: bool = True,
    include_complete: bool = True,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    round_digits: int = 2,
    display: bool = True,
) -> pd.DataFrame:
    """Summarize data types, unique values, completeness, and missingness.

    Parameters
    ----------
    data:
        The pandas DataFrame to summarize.
    include_dtypes:
        Include the ``Data Type`` column in the returned report.
    include_complete:
        Include columns that contain no missing values.
    sort_by:
        Optional metric used to sort the report. Supported values are
        ``"missing_pct"``, ``"missing_count"``, ``"unique"``, and ``"dtype"``.
    ascending:
        Sort in ascending order when ``sort_by`` is provided.
    round_digits:
        Non-negative number of decimal places used for percentages.
    display:
        Print the dataset-level summary when true.

    Returns
    -------
    pandas.DataFrame
        A column-level data quality report. Dataset-level summary values are
        available in ``report.attrs``.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("check() expects a pandas DataFrame.")

    if isinstance(round_digits, bool) or not isinstance(round_digits, Integral):
        raise TypeError("round_digits must be a non-negative integer.")
    if round_digits < 0:
        raise ValueError("round_digits must be a non-negative integer.")

    if sort_by is not None and (
        not isinstance(sort_by, str) or sort_by not in _SORT_COLUMNS
    ):
        valid_values = ", ".join(repr(value) for value in _SORT_COLUMNS)
        raise ValueError(f"sort_by must be one of None, {valid_values}.")

    row_count = len(data)
    report_columns = [
        "Data Type",
        "Unique Values",
        "Values Present",
        "Missing Count",
        "Missing %",
    ]
    records = []

    # Iterate by position so duplicate column labels remain independent rows.
    for position in range(data.shape[1]):
        series = data.iloc[:, position]
        missing_count = int(series.isna().sum())
        missing_percent = missing_count * 100.0 / row_count if row_count else 0.0
        records.append(
            {
                "Data Type": str(series.dtype),
                "Unique Values": int(series.nunique(dropna=True)),
                "Values Present": int(series.notna().sum()),
                "Missing Count": missing_count,
                "Missing %": missing_percent,
            }
        )

    report = pd.DataFrame(
        records,
        index=pd.Index(list(data.columns)),
        columns=report_columns,
    )

    if not include_complete:
        report = report.iloc[report["Missing Count"].to_numpy() > 0]

    if sort_by is not None:
        report = report.sort_values(
            by=_SORT_COLUMNS[sort_by],
            ascending=ascending,
            kind="mergesort",
        )

    if not include_dtypes:
        report = report.drop(columns="Data Type")

    report["Missing %"] = report["Missing %"].round(round_digits)

    missing_mask = data.isna()
    total_missing_cells = int(missing_mask.to_numpy().sum())
    rows_with_missing = int(missing_mask.any(axis=1).sum())
    overall_missing_percent = (
        round(total_missing_cells * 100.0 / data.size, round_digits)
        if data.size
        else 0.0
    )

    report.attrs.update(
        {
            "shape": data.shape,
            "total_missing_cells": total_missing_cells,
            "rows_with_missing": rows_with_missing,
            "overall_missing_percent": overall_missing_percent,
        }
    )

    if display:
        print(f"Data Shape: {data.shape}")
        print(f"Total Missing Cells: {total_missing_cells}")
        print(f"Rows With Missing Values: {rows_with_missing}")
        print(
            "Overall Missing Percentage: "
            f"{overall_missing_percent:.{round_digits}f}%"
        )

    return report
