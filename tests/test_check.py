import pandas as pd
import pytest

from pandas_eda_check import __version__, check


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "name": ["Ada", "Bob", "Bob", None],
            "age": [36, None, 29, 36],
            "active": [True, True, False, True],
        }
    )


def test_normal_dataframe_has_expected_structure(sample_df):
    report = check(sample_df, display=False)

    assert list(report.columns) == [
        "Data Type",
        "Unique Values",
        "Values Present",
        "Missing Count",
        "Missing %",
    ]
    assert list(report.index) == ["name", "age", "active"]


def test_missing_percentages_are_calculated_and_rounded(sample_df):
    report = check(sample_df, round_digits=1, display=False)

    assert report.loc["name", "Missing %"] == 25.0
    assert report.loc["age", "Missing %"] == 25.0
    assert report.loc["active", "Missing %"] == 0.0


def test_unique_values_exclude_missing_values(sample_df):
    report = check(sample_df, display=False)

    assert report.loc["name", "Unique Values"] == 2
    assert report.loc["age", "Unique Values"] == 2
    assert report.loc["active", "Unique Values"] == 2


def test_values_present_are_non_missing_counts(sample_df):
    report = check(sample_df, display=False)

    assert report.loc["name", "Values Present"] == 3
    assert report.loc["age", "Values Present"] == 3
    assert report.loc["active", "Values Present"] == 4


def test_data_types_are_reported(sample_df):
    report = check(sample_df, display=False)

    for column in sample_df.columns:
        assert report.loc[column, "Data Type"] == str(sample_df[column].dtype)


def test_include_dtypes_false_removes_data_type(sample_df):
    report = check(sample_df, include_dtypes=False, display=False)

    assert "Data Type" not in report.columns


def test_include_complete_false_only_keeps_columns_with_missing_values(sample_df):
    report = check(sample_df, include_complete=False, display=False)

    assert list(report.index) == ["name", "age"]


@pytest.mark.parametrize(
    ("sort_by", "ascending", "expected_index"),
    [
        ("missing_pct", False, ["name", "age", "active"]),
        ("missing_count", True, ["active", "name", "age"]),
        ("unique", False, ["name", "age", "active"]),
        ("dtype", True, ["active", "age", "name"]),
    ],
)
def test_sorting(sample_df, sort_by, ascending, expected_index):
    report = check(
        sample_df,
        sort_by=sort_by,
        ascending=ascending,
        display=False,
    )

    assert list(report.index) == expected_index


def test_sort_by_dtype_works_when_dtype_column_is_excluded(sample_df):
    report = check(
        sample_df,
        include_dtypes=False,
        sort_by="dtype",
        ascending=True,
        display=False,
    )

    assert list(report.index) == ["active", "age", "name"]
    assert "Data Type" not in report.columns


def test_empty_dataframe_is_safe():
    report = check(pd.DataFrame(columns=["empty"]), display=False)

    assert report.loc["empty", "Missing %"] == 0.0
    assert report.attrs == {
        "shape": (0, 1),
        "total_missing_cells": 0,
        "rows_with_missing": 0,
        "overall_missing_percent": 0.0,
    }


def test_dataframe_without_columns_is_safe():
    report = check(pd.DataFrame(index=range(3)), display=False)

    assert report.empty
    assert list(report.columns) == [
        "Data Type",
        "Unique Values",
        "Values Present",
        "Missing Count",
        "Missing %",
    ]
    assert report.attrs["overall_missing_percent"] == 0.0


def test_invalid_non_dataframe_input():
    with pytest.raises(
        TypeError,
        match=r"^check\(\) expects a pandas DataFrame\.$",
    ):
        check([1, 2, 3])


@pytest.mark.parametrize("round_digits", [True, 1.5, "2", None])
def test_invalid_round_digits_type(round_digits):
    with pytest.raises(
        TypeError,
        match="round_digits must be a non-negative integer",
    ):
        check(pd.DataFrame(), round_digits=round_digits)


def test_negative_round_digits_is_invalid():
    with pytest.raises(
        ValueError,
        match="round_digits must be a non-negative integer",
    ):
        check(pd.DataFrame(), round_digits=-1)


@pytest.mark.parametrize("sort_by", ["missing", 42, ["missing_pct"]])
def test_invalid_sort_by(sample_df, sort_by):
    with pytest.raises(ValueError, match="sort_by must be one of"):
        check(sample_df, sort_by=sort_by)


def test_display_false_produces_no_console_output(sample_df, capsys):
    check(sample_df, display=False)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_display_summary_uses_expected_labels(sample_df, capsys):
    check(sample_df)

    assert capsys.readouterr().out.splitlines() == [
        "Data Shape: (4, 3)",
        "Total Missing Cells: 2",
        "Rows With Missing Values: 2",
        "Overall Missing Percentage: 16.67%",
    ]


def test_report_attrs_metadata(sample_df):
    report = check(sample_df, display=False)

    assert report.attrs == {
        "shape": (4, 3),
        "total_missing_cells": 2,
        "rows_with_missing": 2,
        "overall_missing_percent": 16.67,
    }


def test_duplicate_column_names_are_handled_by_position():
    data = pd.DataFrame(
        [[1, None], [1, 2]],
        columns=["duplicate", "duplicate"],
    )

    report = check(data, display=False)

    assert list(report.index) == ["duplicate", "duplicate"]
    assert report["Missing Count"].tolist() == [0, 1]
    assert report["Values Present"].tolist() == [2, 1]


def test_version_is_exposed():
    assert __version__ == "0.3.0"
