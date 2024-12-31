import pytest
import pandas as pd
from main import load_data


@pytest.fixture
def loaded_data():
    """Фикстура для загрузки данных."""
    X, y = load_data("Main_Data.csv")
    return X, y


def test_load_data(loaded_data):
    """
    Проверка корректности загрузки данных.
    """
    X, y = loaded_data
    # Проверяем, что X — это DataFrame, а y — Series
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    # Проверяем, что данные не пустые
    assert not X.empty
    assert not y.empty
    # Проверяем, что в X есть ожидаемые колонки
    expected_columns = ["T", "E", "C", "FM", "Xfm", "AFM", "Xafm"]
    assert all(col in X.columns for col in expected_columns)
    # Проверяем, что y содержит только бинарные значения (0 и 1)
    assert set(y.unique()).issubset({0, 1})
    # Выводим информацию о данных
    print("\nМодульный тест 2.")
    print("Данные успешно загружены:")
    print(f"X (признаки): {X.shape[0]} строк, {X.shape[1]} столбцов")
    print(f"y (целевая переменная): {len(y)} значений")
