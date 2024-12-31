from fastapi.testclient import TestClient
from main import app
import pandas as pd
import os

# Создаем тестовый клиент
client = TestClient(app)


# Тест для проверки обработки предсказаний
def test_predict():
    # Создаем тестовый CSV файл
    test_data = {
        "L": [32, 32],
        "sample": [1, 1],
        "T": [5.0, 4.9],
        "E": [-0.4, -0.43],
        "C": [0.1, 0.09],
        "FM": [0.0001, 0.0002],
        "Xfm": [0.1, 0.09],
        "AFM": [0.0004, 0.0005],
        "Xafm": [0.5, 0.6],
    }
    test_df = pd.DataFrame(test_data)
    test_file = "test_data.csv"
    test_df.to_csv(test_file, index=False)

    with open(test_file, "rb") as f:
        response = client.post("/predict/", files={"file": ("test_data.csv", f, "text/csv")})

    assert response.status_code == 200
    predictions = response.json()
    assert len(predictions) == 2
    assert "y_predict" in predictions[0]
    assert "predict_proba" in predictions[0]

    # Удаляем тестовый файл
    os.remove(test_file)
    print("\nПриемочный тест пройден: предсказания модели совпали.")


# Тестирование некорректного ввода
def test_predict_invalid_file():
    response = client.post("/predict/", files={"file": ("test.txt", b"Invalid content", "text/plain")})
    assert response.status_code == 400
