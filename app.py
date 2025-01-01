import streamlit as st
import requests
import pandas as pd
import io
from PIL import Image

# URL API (FastAPI)
API_URL = "http://fastapi:8000"

# Заголовок приложения
st.title("AI_phase_diagram_API")

# Раздел для отображения метрик модели
st.header("Метрики модели")
if st.button("Рассчитать метрики"):
    response = requests.get(f"{API_URL}/metrics/")
    if response.status_code == 200:
        st.json(response.json())
    else:
        st.error("Ошибка при получении метрик")

# Раздел для построения ROC-кривой
st.header("ROC-кривая")
if st.button("Построить ROC-кривую"):
    response = requests.get(f"{API_URL}/roc_curve/")
    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content))
        st.image(image, caption="ROC Curve", use_container_width=True)
    else:
        st.error("Ошибка при построении ROC-кривой")

# Раздел для кросс-валидации
st.header("Кросс-валидация")
if st.button("Запустить кросс-валидацию"):
    response = requests.get(f"{API_URL}/cross_validation/")
    if response.status_code == 200:
        results = response.json()["results"]
        mean_roc_auc = response.json()["mean_roc_auc"]

        st.subheader("Результаты по каждому фолду")
        for result in results:
            st.write(f"Fold {result['fold']}:")
            st.write(f"ROC-AUC: {result['roc_auc']:.4f}")
            cm_df = pd.DataFrame(
                result["confusion_matrix"],
                index=["Actual 0", "Actual 1"],
                columns=["Predicted 0", "Predicted 1"],
            )
            st.table(cm_df)

        st.subheader(f"Средний ROC-AUC: {mean_roc_auc:.4f}")
    else:
        st.error("Ошибка при выполнении кросс-валидации")

# Раздел для построения графика ROC-AUC для кросс-валидации
st.header("ROC-AUC для кросс-валидации")
if st.button("Построить график ROC-AUC"):
    response = requests.get(f"{API_URL}/cross_validation_roc_auc/")
    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content))
        st.image(image, caption="5-fold Cross-Validation ROC-AUC", use_container_width=True)
    else:
        st.error("Ошибка при построении графика ROC-AUC")

# Раздел для анализа важности признаков
st.header("Важность признаков")
if st.button("Показать важность признаков"):
    response = requests.get(f"{API_URL}/feature_importance/")
    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content))
        st.image(image, caption="Feature Importance", use_container_width=True)
    else:
        st.error("Ошибка при анализе важности признаков")

# Раздел для SHAP-анализа
st.header("SHAP-анализ")
if st.button("Построить SHAP-анализ"):
    response = requests.get(f"{API_URL}/shap_analysis/")
    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content))
        st.image(image, caption="SHAP Analysis", use_container_width=True)
    else:
        st.error("Ошибка при построении SHAP-анализа")

# Раздел для предсказаний
st.header("Предсказания")
uploaded_file = st.file_uploader("Загрузите файл CSV", type="csv")
if uploaded_file:
    try:
        # Отправка файла на API
        files = {"file": uploaded_file}
        response = requests.post(f"{API_URL}/predict/", files=files)

        if response.status_code == 200:
            # Преобразование результата в DataFrame
            data = pd.DataFrame(response.json())
            st.write(data)

            # Кнопка для скачивания результатов
            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Скачать результаты",
                data=csv,
                file_name="processed_results.csv",
                mime="text/csv",
            )
        else:
            st.error(f"Ошибка при обработке файла: {response.json()['detail']}")
    except Exception as e:
        st.error(f"Ошибка обработки файла: {str(e)}")
