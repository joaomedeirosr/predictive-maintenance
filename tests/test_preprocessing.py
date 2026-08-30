import pandas as pd
import pytest
from src.preprocess import DataCleaner, Normalize


def test_remove_empty_columns():
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [None, None, None],
        "c": [4.0, 5.0, 6.0],
    })
    cleaned = DataCleaner.remove_empty_columns(df)
    assert "b" not in cleaned.columns
    assert list(cleaned.columns) == ["a", "c"]


def test_rename_columns():
    # 26 columns corresponding to ID, cycle, 3 operational settings, and 21 sensors
    dummy_data = {f"col_{i}": [i] for i in range(26)}
    df = pd.DataFrame(dummy_data)
    renamed = DataCleaner.rename_columns(df)
    assert "id" in renamed.columns
    assert "cycle" in renamed.columns
    assert "setting1" in renamed.columns
    assert "s21" in renamed.columns
    assert len(renamed.columns) == 26


def test_sort_by_engine():
    df = pd.DataFrame({
        "id": [2, 1, 1],
        "cycle": [1, 2, 1],
        "val": [10, 20, 30],
    })
    sorted_df = DataCleaner.sort_by_engine(df)
    assert sorted_df["id"].tolist() == [1, 1, 2]
    assert sorted_df["cycle"].tolist() == [1, 2, 1]


def test_normalize_scaler_fit_transform():
    df = pd.DataFrame({
        "id": [1, 1, 2],
        "cycle": [10, 20, 10],
        "sensor1": [100.0, 200.0, 150.0],
        "sensor2": [0.0, 50.0, 100.0],
        "RUL": [50, 40, 60],
        "failure_within_w1": [0, 0, 0],
    })

    normalizer = Normalize()
    transformed = normalizer.fit_transform(df)

    # Excluded columns should remain unmodified
    assert transformed["id"].tolist() == [1, 1, 2]
    assert transformed["RUL"].tolist() == [50, 40, 60]

    # Normalized sensor values should be scaled between 0 and 1
    assert transformed["sensor1"].min() == pytest.approx(0.0)
    assert transformed["sensor1"].max() == pytest.approx(1.0)
    assert transformed["sensor2"].min() == pytest.approx(0.0)
    assert transformed["sensor2"].max() == pytest.approx(1.0)
