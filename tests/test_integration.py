from fastapi.testclient import TestClient
from main import app
import pandas as pd
from main import load_data, load_model
from sklearn.model_selection import StratifiedKFold
import os

# Интеграционные тесты для эндпоинтов

client = TestClient(app)


def test_metrics_endpoint():
    """
    Проверка корректности работы эндпоинта /metrics/.
    """
    # Отправляем GET-запрос на эндпоинт /metrics/
    response = client.get("/metrics/")
    # Проверяем, что ответ успешный
    assert response.status_code == 200
    # Проверяем, что возвращаются все ожидаемые метрики
    metrics = response.json()
    expected_metrics = [
        "Accuracy", "Precision (Class 0)", "Precision (Class 1)",
        "Recall (Class 0)", "Recall (Class 1)", "F1-score (Class 0)",
        "F1-score (Class 1)", "ROC-AUC"
    ]
    for metric in expected_metrics:
        assert metric in metrics
    # Выводим информацию о тесте
    print("\nИнтеграционный тест: эндпоинт /metrics/ работает корректно.")


def test_roc_curve_endpoint():
    """
    Проверка корректности работы эндпоинта /roc_curve/.
    """
    # Отправляем GET-запрос на эндпоинт /roc_curve/
    response = client.get("/roc_curve/")
    # Проверяем, что ответ успешный и возвращается изображение
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    # Выводим информацию о тесте
    print("\nИнтеграционный тест: эндпоинт /roc_curve/ работает корректно.")


def test_cross_validation_endpoint():
    """
    Проверка корректности работы эндпоинта /cross_validation/.
    """
    # Отправляем GET-запрос на эндпоинт /cross_validation/
    response = client.get("/cross_validation/")
    # Проверяем, что ответ успешный
    assert response.status_code == 200
    # Проверяем, что возвращаются результаты кросс-валидации
    data = response.json()
    assert "results" in data
    assert "mean_roc_auc" in data
    # Проверяем, что результаты для каждого фолда содержат ожидаемые ключи
    for result in data["results"]:
        assert "fold" in result
        assert "roc_auc" in result
        assert "confusion_matrix" in result
    # Выводим информацию о тесте
    print("\nИнтеграционный тест: эндпоинт /cross_validation/ работает корректно.")


def test_feature_importance_endpoint():
    """
    Проверка корректности работы эндпоинта /feature_importance/.
    """
    # Отправляем GET-запрос на эндпоинт /feature_importance/
    response = client.get("/feature_importance/")
    # Проверяем, что ответ успешный и возвращается изображение
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    # Выводим информацию о тесте
    print("\nИнтеграционный тест: эндпоинт /feature_importance/ работает корректно.")


def test_shap_analysis_endpoint():
    """
    Проверка корректности работы эндпоинта /shap_analysis/.
    """
    # Отправляем GET-запрос на эндпоинт /shap_analysis/
    response = client.get("/shap_analysis/")
    # Проверяем, что ответ успешный и возвращается изображение
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    # Выводим информацию о тесте
    print("\nИнтеграционный тест: эндпоинт /shap_analysis/ работает корректно.")


# Интеграционные тесты для обработки ошибок
def test_predict_endpoint_error_handling():
    """
    Проверка обработки ошибок при загрузке некорректного файла.
    """
    # Создаем некорректный CSV файл (без необходимых колонок)
    invalid_data = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })
    invalid_data.to_csv("invalid_data.csv", index=False)
    # Отправляем некорректный файл на эндпоинт /predict/
    with open("invalid_data.csv", "rb") as f:
        response = client.post("/predict/", files={"file": f})
    # Проверяем, что возвращается ошибка 400
    assert response.status_code == 400
    assert "Отсутствует колонка" in response.json()["detail"]
    # Удаляем временный файл
    os.remove("invalid_data.csv")
    # Выводим информацию о тесте
    print("\nИнтеграционный тест: обработка ошибок при загрузке файла работает корректно.")


# Интеграционные тесты для взаимодействия между компонентами
def test_data_model_integration():
    """
    Проверка взаимодействия между загрузкой данных и модели.
    """
    # Загружаем данные и модель
    X, y = load_data("Main_Data.csv")
    model = load_model("xgboost_model.pkl")
    # Проверяем, что модель может делать предсказания на загруженных данных
    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})
    # Выводим информацию о тесте
    print("\nИнтеграционный тест: взаимодействие между загрузкой данных и модели работает корректно.")


def test_data_cross_validation_integration():
    """
    Проверка взаимодействия между загрузкой данных и кросс-валидацией.
    """
    # Загружаем данные
    X, y = load_data("Main_Data.csv")
    # Загружаем модель
    model = load_model("xgboost_model.pkl")
    # Выполняем кросс-валидацию
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kfold.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        # Обучаем модель и делаем предсказания
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        # Проверяем, что предсказания корректны
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})
    # Выводим информацию о тесте
    print("\nИнтеграционный тест: взаимодействие между загрузкой данных и кросс-валидацией работает корректно.")
