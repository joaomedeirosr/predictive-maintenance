import numpy as np
import pandas as pd
import pytest
from src.feature_engineering import FeatureEngineering


def test_rul_with_numpy():
    train_df = pd.DataFrame({
        "id": [1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 10, 20],
    })

    result = FeatureEngineering.rul_with_numpy(train_df)

    # Motor 1 max cycle is 3: RULs should be 3-1=2, 3-2=1, 3-3=0
    assert result[result["id"] == 1]["RUL"].tolist() == [2, 1, 0]
    # Motor 2 max cycle is 20: RULs should be 20-10=10, 20-20=0
    assert result[result["id"] == 2]["RUL"].tolist() == [10, 0]


def test_generating_target_variable():
    df = pd.DataFrame({
        "RUL": [40, 30, 29, 0],
    })

    result = FeatureEngineering.generating_target_variable(df, w1=30)
    assert result["failure_within_w1"].tolist() == [0, 1, 1, 1]


def test_rul_test():
    test_df = pd.DataFrame({
        "id": [1, 1, 2],
        "cycle": [10, 20, 15],
    })

    truth_df = pd.DataFrame({
        0: [5, 10]  # Additional RUL for motor 1 (id=1) and motor 2 (id=2)
    })

    result = FeatureEngineering.rul_test(test_df, truth_df, w1=10)

    # For Motor 1: max cycle observed = 20, additional = 5 -> total max = 25
    # RUL at cycle 10: 25 - 10 = 15 -> failure_within_w1 = 0
    # RUL at cycle 20: 25 - 20 = 5 -> failure_within_w1 = 1
    motor1 = result[result["id"] == 1]
    assert motor1["RUL"].tolist() == [15, 5]
    assert motor1["failure_within_w1"].tolist() == [0, 1]

    # For Motor 2: max cycle observed = 15, additional = 10 -> total max = 25
    # RUL at cycle 15: 25 - 15 = 10 -> failure_within_w1 = 1 (since 10 <= 10)
    motor2 = result[result["id"] == 2]
    assert motor2["RUL"].tolist() == [10]
    assert motor2["failure_within_w1"].tolist() == [1]
