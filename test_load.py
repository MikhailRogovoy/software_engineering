# Запуск pytest -s test_load.py

from fastapi.testclient import TestClient
from main import app
import time
from statistics import mean

# Нагрузочный тест для проверки стабильности системы и расчета времени отклика
client = TestClient(app)


def test_stability_with_multiple_requests():
    """
    Проверка стабильности системы при многократных запросах.
    """
    time_list = list()

    for _ in range(30):  # 30 запросов
        start = time.thread_time()
        response = client.get("/cross_validation_roc_auc/")
        finish = time.thread_time()
        time_response = finish-start
        time_list.append(time_response)
        assert response.status_code == 200
    avg_time = mean(time_list)

    # Выводим информацию о тесте
    print("\nНагрузочный тест: система стабильна при многократных запросах.")
    print("Среднее время отклика: " + str(avg_time))
