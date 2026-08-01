import pandas as pd
import pytest

from prep_presisi.evaluation.metrics import mae, mape, wmape


def test_mae_known_values():
    actual = pd.Series([10, 20, 30])
    predicted = pd.Series([12, 18, 33])
    # errors: 2, 2, 3 -> mean = 7/3
    assert mae(actual, predicted) == pytest.approx(7 / 3)


def test_mape_known_values():
    actual = pd.Series([10, 20, 40])
    predicted = pd.Series([12, 18, 44])
    # pct errors: 0.2, 0.1, 0.1 -> mean = 0.4/3
    assert mape(actual, predicted) == pytest.approx(0.4 / 3)


def test_mape_drops_zero_actual():
    actual = pd.Series([0, 20])
    predicted = pd.Series([5, 22])
    # only the actual=20 row counts: |20-22|/20 = 0.1
    assert mape(actual, predicted) == pytest.approx(0.1)


def test_wmape_known_values():
    actual = pd.Series([10, 20, 30])
    predicted = pd.Series([12, 18, 33])
    # total abs error = 7, total actual = 60 -> 7/60
    assert wmape(actual, predicted) == pytest.approx(7 / 60)
