import pytest
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error

# Пути к файлам модели и данных
MODEL_PATH = "xgboost_model.pkl"
DATA_PATH = "Main_Data.csv"


# Фикстура для загрузки модели
@pytest.fixture
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


# Фикстура для загрузки тестовых данных
@pytest.fixture
def test_data():
    data = pd.read_csv(DATA_PATH)
    X = data[["T", "E", "C", "FM", "Xfm", "AFM", "Xafm"]]
    y = data["h"]
    return X, y


# Параметризованный тест: проверка работы модели на разных входных данных
@pytest.mark.parametrize(
    "test_input,expected",
    [
        ([4.345000, -0.504323, 0.138134, 0.000054, 0.104656, 0.000641, 0.830466], 0),
        ([3.713000, -0.611887, 0.208294, 0.000779, 1.380301, 0.000080, 0.106644], 0),
        ([1.106000, -1.784251, 0.159670, 0.381279, 0.354348, 0.002476, 19.960405], 1),
        ([0.395000, -1.962552, 0.229341, 0.074894, 0.362220, 0.924051, 0.411397], 1),
    ],
)
def test_model_with_various_inputs(load_model, test_input, expected):
    model = load_model

    # Преобразуем входные данные в формат numpy
    test_input_np = np.array([test_input])
    prediction = model.predict(test_input_np)  # Получаем предсказание

    # Отладочный вывод
    print(f"Параметрический тест: Вход: {test_input}, Предсказание: {prediction}, Ожидаемое: {expected}")

    # Проверка предсказания
    assert prediction == expected, (
        f"Ошибка предсказания для входа {test_input}. "
        f"Ожидалось: {expected}, Получено: {prediction}"
    )


# Проверка метрик модели
def test_model_metrics(load_model, test_data):
    model = load_model
    X, y = test_data

    # Получаем предсказания
    predictions = model.predict(X)

    # Отладочный вывод
    mse = mean_squared_error(y, predictions)
    print(f"Mean Squared Error (MSE): {mse}")

    # Проверяем метрику MSE как пример
    assert mse < 0.1, f"Слишком большое значение MSE: {mse}"


# Дополнительный тест для проверки фикстуры test_data
def test_data_fixture(test_data):
    X, y = test_data
    assert not X.empty, "Тестовые данные X пусты"
    assert not y.empty, "Тестовые данные y пусты"
    print("Параметрический тест: Фикстура test_data загружена корректно")
