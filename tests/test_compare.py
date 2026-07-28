import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pandas_eda_check import __version__, compare


REPORT_KEYS = [
    "overview",
    "dataset_summary",
    "schema_changes",
    "column_changes",
    "category_changes",
]


def metric_row(report, section, metric, column=None):
    rows = report[section]
    mask = rows["Metric"].eq(metric)
    if column is not None:
        mask &= rows["Column"].eq(column)
    selected = rows.loc[mask]
    assert len(selected) == 1
    return selected.iloc[0]


def test_public_api_and_exact_return_structure():
    report = compare(pd.DataFrame(), pd.DataFrame(), display=False)

    assert __version__ == "0.3.0"
    assert list(report) == REPORT_KEYS
    assert all(isinstance(value, pd.DataFrame) for value in report.values())


def test_report_columns_are_stable():
    report = compare(pd.DataFrame(), pd.DataFrame(), display=False)

    assert list(report["overview"]) == ["Metric", "Value"]
    assert list(report["dataset_summary"]) == [
        "Metric",
        "Reference",
        "Current",
        "Absolute Change",
        "Percent Change",
        "Status",
        "Severity",
        "Note",
    ]
    assert list(report["schema_changes"]) == [
        "Column",
        "Change Type",
        "Reference Dtype",
        "Current Dtype",
        "Status",
        "Severity",
        "Note",
    ]
    assert list(report["column_changes"]) == [
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
    assert list(report["category_changes"]) == [
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


def test_matching_profiles_are_stable_and_filtered_by_default():
    data = pd.DataFrame({"value": [1, 2, 3], "kind": ["a", "b", "a"]})
    report = compare(data, data.copy(), display=False)

    assert report["dataset_summary"].empty
    assert report["schema_changes"].empty
    assert report["column_changes"].empty
    assert report["category_changes"].empty
    overview = report["overview"].set_index("Metric")["Value"]
    assert overview["Stable Findings"] > 0
    assert overview["Changed Findings"] == 0


def test_include_stable_true_includes_all_dataset_metrics():
    data = pd.DataFrame({"value": [1, 2]})
    report = compare(data, data.copy(), include_stable=True, display=False)

    assert report["dataset_summary"]["Metric"].tolist() == [
        "Row Count",
        "Column Count",
        "Total Missing Values",
        "Overall Missing Percentage",
        "Duplicate Row Count",
        "Duplicate Row Percentage",
    ]
    assert set(report["dataset_summary"]["Status"]) == {"Stable"}


def test_different_rows_indexes_and_order_are_profiled_without_alignment():
    reference = pd.DataFrame({"x": [1, 2, 2]}, index=[10, 20, 30])
    current = pd.DataFrame({"x": [2, 1, 3, 2]}, index=["d", "c", "b", "a"])
    report = compare(reference, current, include_stable=True, display=False)

    row = metric_row(report, "dataset_summary", "Row Count")
    assert (row["Reference"], row["Current"], row["Status"]) == (3, 4, "Changed")
    assert metric_row(report, "column_changes", "Minimum", "x")["Status"] == "Stable"
    assert metric_row(report, "column_changes", "Maximum", "x")["Current"] == 3.0


def test_added_removed_and_dtype_changes_have_fixed_severities_and_order():
    reference = pd.DataFrame(
        {"removed": [1], "changed": pd.Series([1], dtype="int64"), "same": ["a"]}
    )
    current = pd.DataFrame(
        {
            "changed": pd.Series([1], dtype="Int64"),
            "same": ["a"],
            "added": [True],
        }
    )
    report = compare(reference, current, include_stable=True, display=False)
    schema = report["schema_changes"]

    assert schema["Column"].tolist() == ["removed", "changed", "same", "added"]
    assert schema["Change Type"].tolist() == [
        "Removed",
        "Data Type Changed",
        "Unchanged",
        "Added",
    ]
    assert schema["Severity"].tolist() == ["High", "High", "None", "Medium"]
    assert pd.isna(schema.iloc[0]["Current Dtype"])
    assert pd.isna(schema.iloc[-1]["Reference Dtype"])


def test_compatible_numeric_dtype_change_still_compares_numeric_metrics():
    reference = pd.DataFrame({"x": pd.Series([1, 2], dtype="int64")})
    current = pd.DataFrame({"x": pd.Series([1, 3], dtype="Int64")})
    report = compare(reference, current, include_stable=True, display=False)

    assert report["schema_changes"].iloc[0]["Change Type"] == "Data Type Changed"
    assert "Mean" in report["column_changes"]["Metric"].tolist()


def test_compatible_object_and_string_dtype_compares_categories():
    reference = pd.DataFrame({"x": pd.Series(["a", "a"], dtype="object")})
    current = pd.DataFrame({"x": pd.Series(["a", "b"], dtype="string")})
    report = compare(reference, current, include_stable=True, display=False)

    assert report["schema_changes"].iloc[0]["Change Type"] == "Data Type Changed"
    assert "Most Frequent Value" in report["column_changes"]["Metric"].tolist()
    assert len(report["category_changes"]) == 1


def test_exact_categorical_dtype_metadata_change_is_reported():
    reference = pd.DataFrame(
        {"x": pd.Series(pd.Categorical(["a"], categories=["a", "b"]))}
    )
    current = pd.DataFrame(
        {"x": pd.Series(pd.Categorical(["a"], categories=["a", "c"]))}
    )
    report = compare(reference, current, include_stable=True, display=False)
    schema = report["schema_changes"].iloc[0]

    assert schema["Change Type"] == "Data Type Changed"
    assert schema["Reference Dtype"] == "category"
    assert schema["Current Dtype"] == "category"
    assert schema["Note"] == "Exact dtype metadata differs."


def test_incompatible_types_only_emit_common_metrics_with_note():
    reference = pd.DataFrame({"x": [1, 2]})
    current = pd.DataFrame({"x": ["1", "2"]})
    report = compare(reference, current, include_stable=True, display=False)
    rows = report["column_changes"]

    assert rows["Metric"].tolist() == [
        "Missing Count",
        "Missing Percentage",
        "Present Count",
        "Unique Count",
        "Unique Percentage",
    ]
    assert rows["Note"].str.contains("broad column types differ").all()


@pytest.mark.parametrize(
    ("reference", "current", "expected"),
    [
        ([1, 2, 3, 4], [1, None, None, 4], "Worsened"),
        ([1, None, None, 4], [1, 2, 3, 4], "Improved"),
    ],
)
def test_material_missing_percentage_direction(reference, current, expected):
    report = compare(
        pd.DataFrame({"x": reference}),
        pd.DataFrame({"x": current}),
        display=False,
    )

    row = metric_row(report, "column_changes", "Missing Percentage", "x")
    assert row["Status"] == expected
    assert row["Severity"] != "None"


def test_below_threshold_missing_percentage_is_stable_and_filtered():
    reference = pd.DataFrame({"x": range(100)})
    current = reference.astype("float64")
    current.loc[0, "x"] = np.nan
    report = compare(reference, current, display=False)

    assert not (
        report["column_changes"]["Metric"].eq("Missing Percentage")
        & report["column_changes"]["Column"].eq("x")
    ).any()


def test_zero_threshold_keeps_equal_values_stable():
    data = pd.DataFrame({"x": [0, 1], "kind": ["a", "b"]})
    report = compare(
        data,
        data.copy(),
        missing_change_threshold=0,
        numeric_change_threshold=0,
        unique_change_threshold=0,
        include_stable=True,
        display=False,
    )

    assert set(report["dataset_summary"]["Status"]) == {"Stable"}
    assert set(report["column_changes"]["Status"]) == {"Stable"}
    assert set(report["category_changes"]["Status"]) == {"Stable"}


def test_overall_missing_percentage_uses_percentage_points():
    reference = pd.DataFrame({"x": [1] * 10})
    current = pd.DataFrame({"x": [None] * 2 + [1] * 8})
    report = compare(reference, current, include_stable=True, display=False)
    row = metric_row(report, "dataset_summary", "Overall Missing Percentage")

    assert row["Absolute Change"] == 20.0
    assert row["Status"] == "Worsened"
    assert pd.isna(row["Percent Change"])
    assert "reference value is zero" in row["Note"]


@pytest.mark.parametrize(
    ("reference_values", "current_values", "expected"),
    [
        ([1, 2, 3, 4], [1, 1, 1, 1], "Worsened"),
        ([1, 1, 1, 1], [1, 2, 3, 4], "Improved"),
    ],
)
def test_duplicate_percentage_direction(reference_values, current_values, expected):
    report = compare(
        pd.DataFrame({"x": reference_values}),
        pd.DataFrame({"x": current_values}),
        display=False,
    )

    row = metric_row(report, "dataset_summary", "Duplicate Row Percentage")
    assert row["Status"] == expected


def test_duplicate_count_is_not_directional_when_row_counts_differ():
    reference = pd.DataFrame({"x": [1, 1]})
    current = pd.DataFrame({"x": [1, 1, 1]})
    report = compare(reference, current, include_stable=True, display=False)

    assert metric_row(report, "dataset_summary", "Duplicate Row Count")["Status"] == "Changed"


def test_numeric_profile_values_and_metric_order():
    reference = pd.DataFrame({"x": [-2.0, 0.0, 2.0, np.inf]})
    current = pd.DataFrame({"x": [-1.0, 0.0, 4.0, -np.inf]})
    report = compare(reference, current, include_stable=True, display=False)
    rows = report["column_changes"].query("Column == 'x'")

    assert rows["Metric"].tolist() == [
        "Missing Count",
        "Missing Percentage",
        "Present Count",
        "Unique Count",
        "Unique Percentage",
        "Mean",
        "Median",
        "Standard Deviation",
        "Minimum",
        "Maximum",
        "Zero Percentage",
        "Negative Percentage",
        "Infinite Count",
        "Infinite Percentage",
    ]
    assert metric_row(report, "column_changes", "Mean", "x")["Reference"] == 0.0
    assert metric_row(report, "column_changes", "Median", "x")["Current"] == 0.0
    assert metric_row(report, "column_changes", "Minimum", "x")["Current"] == -1.0
    assert metric_row(report, "column_changes", "Maximum", "x")["Current"] == 4.0
    assert metric_row(report, "column_changes", "Infinite Count", "x")["Current"] == 1
    assert metric_row(report, "column_changes", "Infinite Percentage", "x")[
        "Reference"
    ] == 25.0


def test_standard_deviation_uses_sample_ddof_one():
    report = compare(
        pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
        pd.DataFrame({"x": [1.0, 2.0, 4.0]}),
        include_stable=True,
        round_digits=6,
        display=False,
    )

    row = metric_row(report, "column_changes", "Standard Deviation", "x")
    assert row["Reference"] == 1.0
    assert row["Current"] == pytest.approx(1.527525, abs=1e-6)


def test_percentage_based_numeric_metrics_use_percentage_point_threshold():
    report = compare(
        pd.DataFrame({"x": [0, 1, 2, 3]}),
        pd.DataFrame({"x": [0, 0, -1, 3]}),
        numeric_change_threshold=20,
        include_stable=True,
        display=False,
    )

    zero = metric_row(report, "column_changes", "Zero Percentage", "x")
    negative = metric_row(report, "column_changes", "Negative Percentage", "x")
    assert zero["Absolute Change"] == 25.0
    assert zero["Status"] == "Changed"
    assert negative["Absolute Change"] == 25.0


@pytest.mark.parametrize("dtype", ["Int64", "Float64"])
def test_nullable_numeric_dtypes(dtype):
    reference = pd.DataFrame({"x": pd.Series([1, None, 3], dtype=dtype)})
    current = pd.DataFrame({"x": pd.Series([1, 2, 4], dtype=dtype)})
    report = compare(reference, current, include_stable=True, display=False)

    assert metric_row(report, "column_changes", "Mean", "x")["Reference"] == 2.0


def test_all_null_and_infinity_only_numeric_columns_are_safe():
    all_null = pd.Series([None, None], dtype="Float64")
    infinities = pd.Series([np.inf, -np.inf], dtype="Float64")
    report = compare(
        pd.DataFrame({"null": all_null, "inf": infinities}),
        pd.DataFrame({"null": all_null.copy(), "inf": infinities.copy()}),
        include_stable=True,
        display=False,
    )

    assert math.isnan(metric_row(report, "column_changes", "Mean", "null")["Reference"])
    assert math.isnan(metric_row(report, "column_changes", "Mean", "inf")["Reference"])
    assert metric_row(report, "column_changes", "Infinite Count", "inf")[
        "Reference"
    ] == 2
    assert metric_row(report, "column_changes", "Infinite Percentage", "inf")[
        "Reference"
    ] == 100.0


def test_zero_reference_percent_change_rules():
    report = compare(
        pd.DataFrame({"x": [0.0, 0.0]}),
        pd.DataFrame({"x": [1.0, 1.0]}),
        include_stable=True,
        display=False,
    )
    mean = metric_row(report, "column_changes", "Mean", "x")

    assert pd.isna(mean["Percent Change"])
    assert mean["Status"] == "Changed"
    assert mean["Severity"] == "Low"
    assert "reference value is zero" in mean["Note"]

    same = compare(
        pd.DataFrame({"x": [0.0]}),
        pd.DataFrame({"x": [0.0]}),
        include_stable=True,
        display=False,
    )
    mean = metric_row(same, "column_changes", "Mean", "x")
    assert mean["Percent Change"] == 0.0
    assert mean["Status"] == "Stable"


def test_one_nan_statistic_and_one_present_is_changed():
    report = compare(
        pd.DataFrame({"x": [1.0]}),
        pd.DataFrame({"x": [1.0, 2.0]}),
        include_stable=True,
        display=False,
    )

    row = metric_row(report, "column_changes", "Standard Deviation", "x")
    assert row["Status"] == "Changed"
    assert row["Severity"] == "Low"
    assert pd.isna(row["Percent Change"])


def test_datetime_metrics_and_order():
    reference = pd.DataFrame({"when": pd.to_datetime(["2024-01-01", "2024-01-11"])})
    current = pd.DataFrame({"when": pd.to_datetime(["2024-01-03", "2024-01-23"])})
    report = compare(reference, current, include_stable=True, display=False)
    rows = report["column_changes"].query("Column == 'when'")

    assert rows["Metric"].tolist()[-3:] == [
        "Earliest Date",
        "Latest Date",
        "Date Range Days",
    ]
    assert metric_row(report, "column_changes", "Earliest Date", "when")[
        "Absolute Change"
    ] == 2.0
    assert metric_row(report, "column_changes", "Latest Date", "when")[
        "Absolute Change"
    ] == 12.0
    assert metric_row(report, "column_changes", "Date Range Days", "when")[
        "Current"
    ] == 20.0


def test_timezone_aware_and_all_null_datetime_columns():
    aware = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True))
    nulls = pd.Series([pd.NaT, pd.NaT], dtype="datetime64[ns]")
    report = compare(
        pd.DataFrame({"aware": aware, "nulls": nulls}),
        pd.DataFrame({"aware": aware.copy(), "nulls": nulls.copy()}),
        include_stable=True,
        display=False,
    )

    assert metric_row(report, "column_changes", "Date Range Days", "aware")[
        "Reference"
    ] == 1.0
    assert metric_row(report, "column_changes", "Earliest Date", "nulls")[
        "Status"
    ] == "Stable"


def test_category_new_removed_top_and_dominance_changes():
    reference = pd.DataFrame({"kind": ["a", "a", "b", "old"]})
    current = pd.DataFrame({"kind": ["b", "b", "new", "new"]})
    report = compare(reference, current, include_stable=True, display=False)
    category = report["category_changes"].iloc[0]

    assert category["New Values"] == ("new",)
    assert category["Removed Values"] == ("a", "old")
    assert category["Reference Top Value"] == "a"
    assert category["Current Top Value"] == "b"
    assert category["Status"] == "Changed"
    assert category["Severity"] in {"Medium", "High"}
    assert metric_row(report, "column_changes", "Most Frequent Value", "kind")[
        "Status"
    ] == "Changed"


def test_tied_category_frequency_uses_first_seen_value():
    report = compare(
        pd.DataFrame({"kind": ["b", "a", "a", "b"]}),
        pd.DataFrame({"kind": ["z", "y", "y", "z"]}),
        include_stable=True,
        display=False,
    )
    category = report["category_changes"].iloc[0]

    assert category["Reference Top Value"] == "b"
    assert category["Current Top Value"] == "z"


def test_category_order_is_first_seen_not_sorted():
    report = compare(
        pd.DataFrame({"kind": ["z", 1, True]}),
        pd.DataFrame({"kind": [True, "new", 2]}),
        include_stable=True,
        display=False,
    )
    category = report["category_changes"].iloc[0]

    assert category["New Values"] == ("new", 2)
    assert category["Removed Values"] == ("z", 1)


def test_high_cardinality_category_set_is_skipped_without_truncation():
    report = compare(
        pd.DataFrame({"kind": ["a", "b", "c"]}),
        pd.DataFrame({"kind": ["a", "b", "d"]}),
        category_limit=2,
        include_stable=True,
        display=False,
    )
    row = report["category_changes"].iloc[0]

    assert row["Set Comparison"] == "Skipped"
    assert pd.isna(row["New Values"])
    assert pd.isna(row["Removed Values"])
    assert "cardinality exceeded category_limit" in row["Note"]


@pytest.mark.parametrize(
    ("limit", "expected"),
    [(3, "Completed"), (2, "Skipped")],
)
def test_category_limit_boundary(limit, expected):
    report = compare(
        pd.DataFrame({"kind": ["a", "b", "c"]}),
        pd.DataFrame({"kind": ["a", "b", "c"]}),
        category_limit=limit,
        include_stable=True,
        display=False,
    )

    assert report["category_changes"].iloc[0]["Set Comparison"] == expected


@pytest.mark.parametrize(
    ("reference", "current"),
    [
        (pd.DataFrame(), pd.DataFrame()),
        (pd.DataFrame(columns=["x"]), pd.DataFrame(columns=["x"])),
        (pd.DataFrame(columns=["x"]), pd.DataFrame(columns=["y"])),
        (pd.DataFrame(columns=["x"]), pd.DataFrame({"x": [1]})),
        (pd.DataFrame({"x": [1]}), pd.DataFrame(columns=["x"])),
        (pd.DataFrame(index=range(3)), pd.DataFrame(index=range(2))),
    ],
)
def test_empty_dataframe_combinations_return_all_sections(reference, current):
    report = compare(reference, current, include_stable=True, display=False)

    assert list(report) == REPORT_KEYS
    assert all(isinstance(value, pd.DataFrame) for value in report.values())
    overall = metric_row(
        report, "dataset_summary", "Overall Missing Percentage"
    )
    assert overall["Reference"] == 0.0
    assert overall["Current"] == 0.0


def test_all_null_common_column_profile():
    reference = pd.DataFrame({"x": [None, None]})
    current = reference.copy()
    report = compare(reference, current, include_stable=True, display=False)

    assert metric_row(report, "column_changes", "Missing Percentage", "x")[
        "Reference"
    ] == 100.0
    assert metric_row(report, "column_changes", "Present Count", "x")["Reference"] == 0
    assert metric_row(report, "column_changes", "Unique Count", "x")["Reference"] == 0
    assert metric_row(report, "column_changes", "Unique Percentage", "x")[
        "Reference"
    ] == 0.0


@pytest.mark.parametrize(
    "series",
    [
        pd.Series(["a", None], dtype="string"),
        pd.Series(["a", "b"], dtype="category"),
        pd.Series([True, False, True], dtype="boolean"),
    ],
)
def test_string_category_and_boolean_are_categorical_like(series):
    report = compare(
        pd.DataFrame({"x": series}),
        pd.DataFrame({"x": series.copy()}),
        include_stable=True,
        display=False,
    )

    assert "Most Frequent Value" in report["column_changes"]["Metric"].tolist()
    assert "Mean" not in report["column_changes"]["Metric"].tolist()
    assert len(report["category_changes"]) == 1


def test_mixed_and_unhashable_values_have_type_aware_unique_counts():
    values = ["1", 1, True, None, [1, 2], {"a": 1}, [1, 2]]
    data = pd.DataFrame({"x": values})
    report = compare(data, data.copy(), include_stable=True, display=False)

    assert metric_row(report, "column_changes", "Unique Count", "x")["Reference"] == 5


def test_duplicate_rows_with_unhashable_values_use_safe_fallback():
    data = pd.DataFrame({"x": [[1, 2], {"a": 1}, [1, 2], {"a": 1}]})
    report = compare(
        data.iloc[:2].copy(),
        data,
        include_stable=True,
        display=False,
    )

    assert metric_row(report, "dataset_summary", "Duplicate Row Count")["Current"] == 2


@pytest.mark.parametrize("which", ["reference", "current"])
def test_duplicate_column_labels_raise_clear_value_error(which):
    duplicate = pd.DataFrame([[1, 2]], columns=["x", "x"])
    normal = pd.DataFrame({"x": [1]})
    kwargs = {"reference": normal, "current": normal}
    kwargs[which] = duplicate

    with pytest.raises(
        ValueError, match=rf"^{which} contains duplicate column names: \['x'\]$"
    ):
        compare(display=False, **kwargs)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("reference", [], "reference must be a pandas DataFrame"),
        ("current", [], "current must be a pandas DataFrame"),
        ("display", 1, "display must be a bool"),
        ("include_stable", 0, "include_stable must be a bool"),
    ],
)
def test_invalid_primary_inputs(argument, value, message):
    kwargs = {
        "reference": pd.DataFrame(),
        "current": pd.DataFrame(),
        "display": False,
    }
    kwargs[argument] = value

    with pytest.raises(TypeError, match=message):
        compare(**kwargs)


@pytest.mark.parametrize(
    "name",
    [
        "missing_change_threshold",
        "numeric_change_threshold",
        "unique_change_threshold",
    ],
)
@pytest.mark.parametrize("value", [True, "1", None])
def test_invalid_threshold_types(name, value):
    with pytest.raises(TypeError, match=name):
        compare(
            pd.DataFrame(),
            pd.DataFrame(),
            display=False,
            **{name: value},
        )


@pytest.mark.parametrize(
    "name",
    [
        "missing_change_threshold",
        "numeric_change_threshold",
        "unique_change_threshold",
    ],
)
@pytest.mark.parametrize("value", [-1, float("inf"), float("-inf"), float("nan")])
def test_invalid_threshold_values(name, value):
    with pytest.raises(ValueError, match=name):
        compare(
            pd.DataFrame(),
            pd.DataFrame(),
            display=False,
            **{name: value},
        )


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_invalid_category_limit_type(value):
    with pytest.raises(TypeError, match="category_limit"):
        compare(pd.DataFrame(), pd.DataFrame(), category_limit=value, display=False)


def test_non_positive_category_limit():
    with pytest.raises(ValueError, match="category_limit"):
        compare(pd.DataFrame(), pd.DataFrame(), category_limit=0, display=False)


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_invalid_round_digits_type(value):
    with pytest.raises(TypeError, match="round_digits"):
        compare(pd.DataFrame(), pd.DataFrame(), round_digits=value, display=False)


def test_negative_round_digits():
    with pytest.raises(ValueError, match="round_digits"):
        compare(pd.DataFrame(), pd.DataFrame(), round_digits=-1, display=False)


def test_display_false_is_silent(capsys):
    compare(pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]}), display=False)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_display_true_has_fixed_headings_and_empty_messages(capsys):
    compare(pd.DataFrame(), pd.DataFrame(), display=True)
    output = capsys.readouterr().out

    headings = [
        "=== Overview ===",
        "=== Dataset Summary ===",
        "=== Schema Changes ===",
        "=== Column Changes ===",
        "=== Category Changes ===",
    ]
    assert [output.index(heading) for heading in headings] == sorted(
        output.index(heading) for heading in headings
    )
    assert output.count("No changes detected.") == 4


def test_common_column_and_category_order_follow_reference():
    reference = pd.DataFrame({"z": ["b", "a"], "a": ["x", "y"]})
    current = pd.DataFrame({"a": ["x", "z"], "z": ["b", "c"]})
    report = compare(reference, current, include_stable=True, display=False)

    assert report["column_changes"]["Column"].drop_duplicates().tolist() == ["z", "a"]
    assert report["category_changes"]["Column"].tolist() == ["z", "a"]


def test_inputs_are_not_mutated():
    reference = pd.DataFrame(
        {"x": [[1], {"a": 1}], "when": pd.to_datetime(["2024-01-01", None])},
        index=[5, 4],
    )
    current = pd.DataFrame(
        {"x": [[2], {"a": 1}], "when": pd.to_datetime(["2024-01-02", None])},
        index=[2, 1],
    )
    original_reference = reference.copy(deep=True)
    original_current = current.copy(deep=True)

    compare(reference, current, display=False)

    assert_frame_equal(reference, original_reference)
    assert_frame_equal(current, original_current)
