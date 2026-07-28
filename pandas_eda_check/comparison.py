"""Profile-based comparison for pandas DataFrames."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as ptypes


OVERVIEW_COLUMNS = ["Metric", "Value"]
DATASET_COLUMNS = [
    "Metric",
    "Reference",
    "Current",
    "Absolute Change",
    "Percent Change",
    "Status",
    "Severity",
    "Note",
]
SCHEMA_COLUMNS = [
    "Column",
    "Change Type",
    "Reference Dtype",
    "Current Dtype",
    "Status",
    "Severity",
    "Note",
]
COLUMN_COLUMNS = [
    "Column",
    "Column Type",
    "Metric",
    "Reference",
    "Current",
    "Absolute Change",
    "Percent Change",
    "Status",
    "Severity",
    "Note",
]
CATEGORY_COLUMNS = [
    "Column",
    "New Values",
    "Removed Values",
    "Reference Top Value",
    "Current Top Value",
    "Reference Top Percentage",
    "Current Top Percentage",
    "Dominant Percentage Point Change",
    "Set Comparison",
    "Status",
    "Severity",
    "Note",
]
REPORT_ORDER = [
    "overview",
    "dataset_summary",
    "schema_changes",
    "column_changes",
    "category_changes",
]
SEVERITY_RANK = {"None": 0, "Low": 1, "Medium": 2, "High": 3}


def _is_missing(value: Any) -> bool:
    """Return whether a single Python or pandas value is missing."""
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _canonical_key(value: Any) -> Tuple[Any, ...]:
    """Create a hashable, type-aware key for arbitrary object values."""
    if _is_missing(value):
        return ("missing",)
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    if isinstance(value, dict):
        items = [(_canonical_key(key), _canonical_key(item)) for key, item in value.items()]
        return (type_name, tuple(sorted(items, key=repr)))
    if isinstance(value, (set, frozenset)):
        return (type_name, tuple(sorted((_canonical_key(item) for item in value), key=repr)))
    if isinstance(value, (list, tuple)):
        return (type_name, tuple(_canonical_key(item) for item in value))
    try:
        hash(value)
    except TypeError:
        return (type_name, repr(value))
    return (type_name, repr(value))


def _readable_value(value: Any) -> Any:
    """Preserve simple values and use repr for unhashable values."""
    if _is_missing(value):
        return pd.NA
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _present_values(series: pd.Series) -> Iterable[Any]:
    for value in series.array:
        if not _is_missing(value):
            yield value


def _safe_nunique(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return len({_canonical_key(value) for value in _present_values(series)})


def _ordered_unique_values(series: pd.Series) -> List[Tuple[Tuple[Any, ...], Any]]:
    seen = set()
    values: List[Tuple[Tuple[Any, ...], Any]] = []
    for value in _present_values(series):
        key = _canonical_key(value)
        if key not in seen:
            seen.add(key)
            values.append((key, value))
    return values


def _safe_top_value(series: pd.Series) -> Tuple[Any, Any, int]:
    """Return readable top value, canonical key, and count with first-seen ties."""
    counts: dict[Tuple[Any, ...], int] = {}
    originals: dict[Tuple[Any, ...], Any] = {}
    order: List[Tuple[Any, ...]] = []
    for value in _present_values(series):
        key = _canonical_key(value)
        if key not in counts:
            counts[key] = 0
            originals[key] = value
            order.append(key)
        counts[key] += 1
    if not order:
        return pd.NA, None, 0
    winning_key = max(order, key=lambda key: counts[key])
    return _readable_value(originals[winning_key]), winning_key, counts[winning_key]


def _safe_duplicate_count(data: pd.DataFrame) -> int:
    if data.shape[1] == 0:
        return 0
    try:
        return int(data.duplicated().sum())
    except TypeError:
        seen = set()
        duplicates = 0
        for row in data.itertuples(index=False, name=None):
            key = tuple(_canonical_key(value) for value in row)
            if key in seen:
                duplicates += 1
            else:
                seen.add(key)
        return duplicates


def _safe_percentage(numerator: Real, denominator: Real) -> float:
    return float(numerator) * 100.0 / float(denominator) if denominator else 0.0


def _values_equal(reference: Any, current: Any) -> bool:
    ref_missing = _is_missing(reference)
    cur_missing = _is_missing(current)
    if ref_missing or cur_missing:
        return ref_missing and cur_missing
    try:
        result = reference == current
    except (TypeError, ValueError):
        return _canonical_key(reference) == _canonical_key(current)
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _safe_changes(reference: Any, current: Any) -> Tuple[Any, Any, str]:
    """Return absolute change, relative percent change, and any explanatory note."""
    if _is_missing(reference) or _is_missing(current):
        return pd.NA, pd.NA, ""
    try:
        absolute = float(current) - float(reference)
    except (TypeError, ValueError, OverflowError):
        return pd.NA, pd.NA, ""
    if not math.isfinite(absolute):
        return pd.NA, pd.NA, ""
    ref_value = float(reference)
    cur_value = float(current)
    if ref_value == 0:
        if cur_value == 0:
            return absolute, 0.0, ""
        return (
            absolute,
            pd.NA,
            "Relative percent change is undefined because the reference value is zero.",
        )
    percent = absolute / abs(ref_value) * 100.0
    return absolute, percent if math.isfinite(percent) else pd.NA, ""


def _threshold_severity(magnitude: Any, threshold: float) -> str:
    if _is_missing(magnitude) or float(magnitude) == 0:
        return "None"
    if float(magnitude) < threshold:
        return "None"
    if threshold == 0:
        return "High"
    if float(magnitude) < 2 * threshold:
        return "Low"
    if float(magnitude) < 4 * threshold:
        return "Medium"
    return "High"


def _max_severity(*values: str) -> str:
    return max(values, key=SEVERITY_RANK.__getitem__)


def _relative_status(
    reference: Any, current: Any, threshold: float
) -> Tuple[str, str, Any, Any, str]:
    absolute, percent, note = _safe_changes(reference, current)
    ref_missing = _is_missing(reference)
    cur_missing = _is_missing(current)
    if ref_missing and cur_missing:
        return "Stable", "None", absolute, percent, note
    if ref_missing != cur_missing:
        return "Changed", "Low", absolute, percent, note
    if _values_equal(reference, current):
        return "Stable", "None", absolute, percent, note
    if _is_missing(percent):
        return "Changed", "Low", absolute, percent, note
    severity = _threshold_severity(abs(float(percent)), threshold)
    status = "Stable" if severity == "None" else "Changed"
    return status, severity, absolute, percent, note


def _point_status(
    reference: Any,
    current: Any,
    threshold: float,
    *,
    directional: bool,
) -> Tuple[str, str, Any, Any, str]:
    absolute, percent, note = _safe_changes(reference, current)
    if _is_missing(reference) and _is_missing(current):
        return "Stable", "None", absolute, percent, note
    if _is_missing(reference) != _is_missing(current):
        return "Changed", "Low", absolute, percent, note
    magnitude = abs(float(absolute))
    severity = _threshold_severity(magnitude, threshold)
    if severity == "None":
        status = "Stable"
    elif directional:
        status = "Improved" if float(absolute) < 0 else "Worsened"
    else:
        status = "Changed"
    return status, severity, absolute, percent, note


def _exact_status(reference: Any, current: Any) -> Tuple[str, str, Any, Any, str]:
    absolute, percent, note = _safe_changes(reference, current)
    if _values_equal(reference, current):
        return "Stable", "None", absolute, percent, note
    return "Changed", "Low", absolute, percent, note


def _classify_series(series: pd.Series) -> str:
    dtype = series.dtype
    if ptypes.is_bool_dtype(dtype):
        return "boolean"
    if ptypes.is_numeric_dtype(dtype):
        return "numeric"
    if ptypes.is_datetime64_any_dtype(dtype):
        return "datetime"
    if (
        ptypes.is_object_dtype(dtype)
        or ptypes.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    ):
        return "categorical"
    return "other"


def _validate_numeric(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative number.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite.")
    if converted < 0:
        raise ValueError(f"{name} must be non-negative.")
    return converted


def _duplicated_labels(columns: pd.Index) -> List[Any]:
    duplicated: List[Any] = []
    seen = set()
    for label in columns[columns.duplicated(keep=False)]:
        key = _canonical_key(label)
        if key not in seen:
            seen.add(key)
            duplicated.append(label)
    return duplicated


def _validate_compare_inputs(
    reference: Any,
    current: Any,
    display: Any,
    include_stable: Any,
    missing_change_threshold: Any,
    numeric_change_threshold: Any,
    unique_change_threshold: Any,
    category_limit: Any,
    round_digits: Any,
) -> Tuple[float, float, float]:
    if not isinstance(reference, pd.DataFrame):
        raise TypeError(
            f"reference must be a pandas DataFrame, got {type(reference).__name__}"
        )
    if not isinstance(current, pd.DataFrame):
        raise TypeError(f"current must be a pandas DataFrame, got {type(current).__name__}")
    if not isinstance(display, bool):
        raise TypeError("display must be a bool.")
    if not isinstance(include_stable, bool):
        raise TypeError("include_stable must be a bool.")
    missing_threshold = _validate_numeric(
        "missing_change_threshold", missing_change_threshold
    )
    numeric_threshold = _validate_numeric(
        "numeric_change_threshold", numeric_change_threshold
    )
    unique_threshold = _validate_numeric(
        "unique_change_threshold", unique_change_threshold
    )
    if isinstance(category_limit, bool) or not isinstance(category_limit, Integral):
        raise TypeError("category_limit must be a positive integer.")
    if category_limit <= 0:
        raise ValueError("category_limit must be a positive integer.")
    if isinstance(round_digits, bool) or not isinstance(round_digits, Integral):
        raise TypeError("round_digits must be a non-negative integer.")
    if round_digits < 0:
        raise ValueError("round_digits must be a non-negative integer.")
    for name, frame in (("reference", reference), ("current", current)):
        duplicated = _duplicated_labels(frame.columns)
        if duplicated:
            raise ValueError(f"{name} contains duplicate column names: {duplicated!r}")
    return missing_threshold, numeric_threshold, unique_threshold


def _dataset_profile(data: pd.DataFrame) -> Mapping[str, float]:
    rows, columns = data.shape
    total_missing = int(data.isna().to_numpy().sum())
    duplicate_count = _safe_duplicate_count(data)
    return {
        "Row Count": rows,
        "Column Count": columns,
        "Total Missing Values": total_missing,
        "Overall Missing Percentage": _safe_percentage(total_missing, data.size),
        "Duplicate Row Count": duplicate_count,
        "Duplicate Row Percentage": _safe_percentage(duplicate_count, rows),
    }


def _dataset_summary(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    missing_threshold: float,
) -> List[dict[str, Any]]:
    ref_profile = _dataset_profile(reference)
    cur_profile = _dataset_profile(current)
    point_metrics = {"Overall Missing Percentage", "Duplicate Row Percentage"}
    records = []
    for metric in ref_profile:
        ref_value = ref_profile[metric]
        cur_value = cur_profile[metric]
        if metric in point_metrics:
            status, severity, absolute, percent, note = _point_status(
                ref_value, cur_value, missing_threshold, directional=True
            )
        else:
            status, severity, absolute, percent, note = _exact_status(
                ref_value, cur_value
            )
        records.append(
            {
                "Metric": metric,
                "Reference": ref_value,
                "Current": cur_value,
                "Absolute Change": absolute,
                "Percent Change": percent,
                "Status": status,
                "Severity": severity,
                "Note": note,
            }
        )
    return records


def _schema_changes(
    reference: pd.DataFrame, current: pd.DataFrame
) -> List[dict[str, Any]]:
    records = []
    current_columns = set(current.columns)
    reference_columns = set(reference.columns)
    for column in reference.columns:
        if column not in current_columns:
            change_type, current_dtype, status, severity = (
                "Removed",
                pd.NA,
                "Changed",
                "High",
            )
        elif reference[column].dtype != current[column].dtype:
            change_type, current_dtype, status, severity = (
                "Data Type Changed",
                str(current[column].dtype),
                "Changed",
                "High",
            )
        else:
            change_type, current_dtype, status, severity = (
                "Unchanged",
                str(current[column].dtype),
                "Stable",
                "None",
            )
        records.append(
            {
                "Column": column,
                "Change Type": change_type,
                "Reference Dtype": str(reference[column].dtype),
                "Current Dtype": current_dtype,
                "Status": status,
                "Severity": severity,
                "Note": (
                    "Exact dtype metadata differs."
                    if change_type == "Data Type Changed"
                    and str(reference[column].dtype) == str(current[column].dtype)
                    else ""
                ),
            }
        )
    for column in current.columns:
        if column not in reference_columns:
            records.append(
                {
                    "Column": column,
                    "Change Type": "Added",
                    "Reference Dtype": pd.NA,
                    "Current Dtype": str(current[column].dtype),
                    "Status": "Changed",
                    "Severity": "Medium",
                    "Note": "",
                }
            )
    return records


def _common_profile(series: pd.Series) -> Mapping[str, float]:
    row_count = len(series)
    missing = int(series.isna().sum())
    present = row_count - missing
    unique = _safe_nunique(series)
    return {
        "Missing Count": missing,
        "Missing Percentage": _safe_percentage(missing, row_count),
        "Present Count": present,
        "Unique Count": unique,
        "Unique Percentage": _safe_percentage(unique, present),
    }


def _numeric_profile(series: pd.Series) -> Mapping[str, Any]:
    values = series.to_numpy(dtype="float64", na_value=np.nan)
    non_missing = values[~np.isnan(values)]
    finite = non_missing[np.isfinite(non_missing)]
    infinite_count = int(np.isinf(non_missing).sum())
    if finite.size:
        mean = float(np.mean(finite))
        median = float(np.median(finite))
        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
        standard_deviation = (
            float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
        )
    else:
        mean = median = minimum = maximum = standard_deviation = float("nan")
    return {
        "Mean": mean,
        "Median": median,
        "Standard Deviation": standard_deviation,
        "Minimum": minimum,
        "Maximum": maximum,
        "Zero Percentage": _safe_percentage(int((finite == 0).sum()), finite.size),
        "Negative Percentage": _safe_percentage(int((finite < 0).sum()), finite.size),
        "Infinite Count": infinite_count,
        "Infinite Percentage": _safe_percentage(infinite_count, non_missing.size),
    }


def _datetime_profile(series: pd.Series) -> Mapping[str, Any]:
    present = series.dropna()
    if present.empty:
        return {
            "Earliest Date": pd.NaT,
            "Latest Date": pd.NaT,
            "Date Range Days": pd.NA,
        }
    earliest = present.min()
    latest = present.max()
    return {
        "Earliest Date": earliest,
        "Latest Date": latest,
        "Date Range Days": (latest - earliest).total_seconds() / 86400.0,
    }


def _categorical_profile(series: pd.Series) -> Mapping[str, Any]:
    present_count = int(series.notna().sum())
    top_value, top_key, top_count = _safe_top_value(series)
    return {
        "Most Frequent Value": top_value,
        "Most Frequent Value Percentage": _safe_percentage(top_count, present_count),
        "_Top Key": top_key,
    }


def _metric_record(
    column: Any,
    column_type: str,
    metric: str,
    reference: Any,
    current: Any,
    status_data: Tuple[str, str, Any, Any, str],
    extra_note: str = "",
) -> dict[str, Any]:
    status, severity, absolute, percent, note = status_data
    notes = [value for value in (note, extra_note) if value]
    return {
        "Column": column,
        "Column Type": column_type,
        "Metric": metric,
        "Reference": reference,
        "Current": current,
        "Absolute Change": absolute,
        "Percent Change": percent,
        "Status": status,
        "Severity": severity,
        "Note": " ".join(notes),
    }


def _column_changes(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    missing_threshold: float,
    numeric_threshold: float,
    unique_threshold: float,
) -> List[dict[str, Any]]:
    records = []
    current_columns = set(current.columns)
    for column in reference.columns:
        if column not in current_columns:
            continue
        ref_series, cur_series = reference[column], current[column]
        ref_type, cur_type = _classify_series(ref_series), _classify_series(cur_series)
        column_type = ref_type if ref_type == cur_type else f"{ref_type} → {cur_type}"
        mismatch_note = (
            ""
            if ref_type == cur_type
            else "Type-specific comparison skipped because broad column types differ."
        )
        ref_common, cur_common = _common_profile(ref_series), _common_profile(cur_series)
        for metric in ref_common:
            if metric in {"Missing Count", "Present Count"}:
                status_data = _exact_status(ref_common[metric], cur_common[metric])
            elif metric == "Missing Percentage":
                status_data = _point_status(
                    ref_common[metric],
                    cur_common[metric],
                    missing_threshold,
                    directional=True,
                )
            else:
                status_data = _relative_status(
                    ref_common[metric], cur_common[metric], unique_threshold
                )
            records.append(
                _metric_record(
                    column,
                    column_type,
                    metric,
                    ref_common[metric],
                    cur_common[metric],
                    status_data,
                    mismatch_note,
                )
            )
        if ref_type != cur_type:
            continue
        if ref_type == "numeric":
            ref_profile, cur_profile = _numeric_profile(ref_series), _numeric_profile(
                cur_series
            )
            for metric in ref_profile:
                if metric == "Infinite Count":
                    status_data = _exact_status(ref_profile[metric], cur_profile[metric])
                elif metric == "Infinite Percentage":
                    status_data = _point_status(
                        ref_profile[metric],
                        cur_profile[metric],
                        missing_threshold,
                        directional=True,
                    )
                elif metric in {"Zero Percentage", "Negative Percentage"}:
                    status_data = _point_status(
                        ref_profile[metric],
                        cur_profile[metric],
                        numeric_threshold,
                        directional=False,
                    )
                else:
                    status_data = _relative_status(
                        ref_profile[metric], cur_profile[metric], numeric_threshold
                    )
                records.append(
                    _metric_record(
                        column,
                        column_type,
                        metric,
                        ref_profile[metric],
                        cur_profile[metric],
                        status_data,
                    )
                )
        elif ref_type == "datetime":
            ref_profile, cur_profile = _datetime_profile(ref_series), _datetime_profile(
                cur_series
            )
            for metric in ref_profile:
                if metric in {"Earliest Date", "Latest Date"}:
                    ref_value, cur_value = ref_profile[metric], cur_profile[metric]
                    if _values_equal(ref_value, cur_value):
                        status_data = ("Stable", "None", 0.0, pd.NA, "")
                    elif _is_missing(ref_value) != _is_missing(cur_value):
                        status_data = ("Changed", "Low", pd.NA, pd.NA, "")
                    else:
                        days = (cur_value - ref_value).total_seconds() / 86400.0
                        status_data = ("Changed", "Low", days, pd.NA, "")
                else:
                    status_data = _relative_status(
                        ref_profile[metric], cur_profile[metric], numeric_threshold
                    )
                records.append(
                    _metric_record(
                        column,
                        column_type,
                        metric,
                        ref_profile[metric],
                        cur_profile[metric],
                        status_data,
                    )
                )
        elif ref_type in {"boolean", "categorical"}:
            ref_profile = _categorical_profile(ref_series)
            cur_profile = _categorical_profile(cur_series)
            top_equal = ref_profile["_Top Key"] == cur_profile["_Top Key"]
            top_status = (
                ("Stable", "None", pd.NA, pd.NA, "")
                if top_equal
                else ("Changed", "Medium", pd.NA, pd.NA, "")
            )
            records.append(
                _metric_record(
                    column,
                    column_type,
                    "Most Frequent Value",
                    ref_profile["Most Frequent Value"],
                    cur_profile["Most Frequent Value"],
                    top_status,
                )
            )
            records.append(
                _metric_record(
                    column,
                    column_type,
                    "Most Frequent Value Percentage",
                    ref_profile["Most Frequent Value Percentage"],
                    cur_profile["Most Frequent Value Percentage"],
                    _point_status(
                        ref_profile["Most Frequent Value Percentage"],
                        cur_profile["Most Frequent Value Percentage"],
                        numeric_threshold,
                        directional=False,
                    ),
                )
            )
    return records


def _category_changes(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    category_limit: int,
    numeric_threshold: float,
    unique_threshold: float,
) -> List[dict[str, Any]]:
    records = []
    current_columns = set(current.columns)
    for column in reference.columns:
        if column not in current_columns:
            continue
        ref_series, cur_series = reference[column], current[column]
        if _classify_series(ref_series) not in {"boolean", "categorical"}:
            continue
        if _classify_series(cur_series) not in {"boolean", "categorical"}:
            continue
        ref_values = _ordered_unique_values(ref_series)
        cur_values = _ordered_unique_values(cur_series)
        ref_top, ref_top_key, ref_count = _safe_top_value(ref_series)
        cur_top, cur_top_key, cur_count = _safe_top_value(cur_series)
        ref_percentage = _safe_percentage(ref_count, int(ref_series.notna().sum()))
        cur_percentage = _safe_percentage(cur_count, int(cur_series.notna().sum()))
        dominant_change = cur_percentage - ref_percentage
        top_changed = ref_top_key != cur_top_key
        note = ""
        if len(ref_values) <= category_limit and len(cur_values) <= category_limit:
            set_comparison = "Completed"
            ref_keys = {key for key, _ in ref_values}
            cur_keys = {key for key, _ in cur_values}
            new_values = tuple(
                _readable_value(value) for key, value in cur_values if key not in ref_keys
            )
            removed_values = tuple(
                _readable_value(value) for key, value in ref_values if key not in cur_keys
            )
            changed_count = len(new_values) + len(removed_values)
            category_change_percentage = (
                changed_count / max(len(ref_values), 1) * 100.0
            )
            set_severity = _threshold_severity(
                category_change_percentage, unique_threshold
            )
            set_changed = changed_count > 0
        else:
            set_comparison = "Skipped"
            new_values = removed_values = pd.NA
            set_severity = "None"
            set_changed = False
            note = (
                "Skipped full category-set comparison because cardinality exceeded "
                "category_limit."
            )
        dominant_severity = _threshold_severity(
            abs(dominant_change), numeric_threshold
        )
        severity = _max_severity(
            set_severity,
            dominant_severity,
            "Medium" if top_changed else "None",
        )
        status = (
            "Changed"
            if set_changed or top_changed or dominant_severity != "None"
            else "Stable"
        )
        records.append(
            {
                "Column": column,
                "New Values": new_values,
                "Removed Values": removed_values,
                "Reference Top Value": ref_top,
                "Current Top Value": cur_top,
                "Reference Top Percentage": ref_percentage,
                "Current Top Percentage": cur_percentage,
                "Dominant Percentage Point Change": dominant_change,
                "Set Comparison": set_comparison,
                "Status": status,
                "Severity": severity,
                "Note": note,
            }
        )
    return records


def _round_report(frame: pd.DataFrame, round_digits: int) -> pd.DataFrame:
    for column in frame.columns:
        frame[column] = frame[column].map(
            lambda value: (
                round(float(value), round_digits)
                if isinstance(value, (float, np.floating))
                and math.isfinite(float(value))
                else value
            )
        )
    return frame


def _build_overview(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    statuses: List[str] = []
    severities: List[str] = []
    for frame in frames:
        statuses.extend(frame["Status"].tolist())
        severities.extend(frame["Severity"].tolist())
    schema = frames[1]
    metrics = [
        ("Total Evaluated Findings", len(statuses)),
        ("Changed Findings", statuses.count("Changed")),
        ("Improved Findings", statuses.count("Improved")),
        ("Worsened Findings", statuses.count("Worsened")),
        ("Stable Findings", statuses.count("Stable")),
        ("High Severity Findings", severities.count("High")),
        ("Medium Severity Findings", severities.count("Medium")),
        ("Low Severity Findings", severities.count("Low")),
        ("Added Columns", int((schema["Change Type"] == "Added").sum())),
        ("Removed Columns", int((schema["Change Type"] == "Removed").sum())),
        (
            "Data Type Changes",
            int((schema["Change Type"] == "Data Type Changed").sum()),
        ),
    ]
    return pd.DataFrame(metrics, columns=OVERVIEW_COLUMNS)


def _display_report(report: Mapping[str, pd.DataFrame]) -> None:
    headings = {
        "overview": "Overview",
        "dataset_summary": "Dataset Summary",
        "schema_changes": "Schema Changes",
        "column_changes": "Column Changes",
        "category_changes": "Category Changes",
    }
    for key in REPORT_ORDER:
        print(f"=== {headings[key]} ===")
        frame = report[key]
        print(frame.to_string(index=False) if not frame.empty else "No changes detected.")


def compare(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    display: bool = True,
    include_stable: bool = False,
    missing_change_threshold: float = 5.0,
    numeric_change_threshold: float = 10.0,
    unique_change_threshold: float = 20.0,
    category_limit: int = 100,
    round_digits: int = 2,
) -> dict[str, pd.DataFrame]:
    """Compare structural, data-quality, and statistical DataFrame profiles.

    ``reference`` is the baseline dataset and ``current`` is the newer dataset.
    Unlike cell-level comparison, this function never aligns rows, compares
    indexes, requires a key, or depends on row order. Neither input is mutated.

    Parameters
    ----------
    reference, current:
        pandas DataFrames whose profiles are compared by column name.
    display:
        Print the five report sections using plain text when true.
    include_stable:
        Include stable rows in detailed sections. Overview counts always include
        stable findings.
    missing_change_threshold:
        Percentage-point threshold for missing, duplicate, and infinity rates.
    numeric_change_threshold:
        Relative-percent threshold for numeric statistics and date ranges;
        percentage-point threshold for zero, negative, and dominant-value rates.
    unique_change_threshold:
        Relative-percent threshold for unique metrics and category-set severity.
    category_limit:
        Maximum unique values allowed on each side for full category-set
        comparison. Higher-cardinality sets are reported as skipped, not truncated.
    round_digits:
        Decimal places used in returned display values. Decisions use unrounded data.

    Returns
    -------
    dict[str, pandas.DataFrame]
        ``overview`` contains finding counts; ``dataset_summary`` contains shape,
        missingness, and duplicate metrics; ``schema_changes`` contains column and
        dtype changes; ``column_changes`` contains long-format common-column
        metrics; and ``category_changes`` contains category-set and dominant-value
        changes. Status values are Improved, Worsened, Changed, and Stable.
        Severity values are None, Low, Medium, and High. Absolute changes for rates
        are percentage points; Percent Change is relative to the absolute reference
        value and is missing for a nonzero current value with a zero reference.

    Examples
    --------
    >>> from pandas_eda_check import compare
    >>> report = compare(reference_df, current_df, display=False)
    >>> print(report["schema_changes"])
    >>> print(report["column_changes"])
    """
    (
        missing_threshold,
        numeric_threshold,
        unique_threshold,
    ) = _validate_compare_inputs(
        reference,
        current,
        display,
        include_stable,
        missing_change_threshold,
        numeric_change_threshold,
        unique_change_threshold,
        category_limit,
        round_digits,
    )

    dataset = pd.DataFrame(
        _dataset_summary(reference, current, missing_threshold),
        columns=DATASET_COLUMNS,
    )
    schema = pd.DataFrame(
        _schema_changes(reference, current),
        columns=SCHEMA_COLUMNS,
    )
    columns = pd.DataFrame(
        _column_changes(
            reference,
            current,
            missing_threshold,
            numeric_threshold,
            unique_threshold,
        ),
        columns=COLUMN_COLUMNS,
    )
    categories = pd.DataFrame(
        _category_changes(
            reference,
            current,
            int(category_limit),
            numeric_threshold,
            unique_threshold,
        ),
        columns=CATEGORY_COLUMNS,
    )
    detailed = [dataset, schema, columns, categories]
    overview = _build_overview(detailed)

    if not include_stable:
        detailed = [
            frame.loc[frame["Status"] != "Stable"].reset_index(drop=True)
            for frame in detailed
        ]
    report = {
        "overview": overview,
        "dataset_summary": detailed[0],
        "schema_changes": detailed[1],
        "column_changes": detailed[2],
        "category_changes": detailed[3],
    }
    report = {
        key: _round_report(frame.copy(), int(round_digits))
        for key, frame in report.items()
    }
    if display:
        _display_report(report)
    return report
