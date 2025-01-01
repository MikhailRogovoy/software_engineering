from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import shap
import io
from fastapi import Response

# Инициализация FastAPI приложения
app = FastAPI()

# Пути к файлам модели и данных
MODEL_PATH = "xgboost_model.pkl"
DATA_PATH = "Main_Data.csv"


# Загрузка данных
def load_data(data_path):
    data = pd.read_csv(data_path)
    X = data[["T", "E", "C", "FM", "Xfm", "AFM", "Xafm"]]
    y = data["h"]
    return X, y


# Загрузка модели
def load_model(model_path):
    with open(model_path, "rb") as file:
        return pickle.load(file)


# Инициализация данных и модели
X, y = load_data(DATA_PATH)
model = load_model(MODEL_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Эндпоинт для получения метрик модели
@app.get("/metrics/")
async def get_metrics():
    """
    Возвращает метрики модели (Accuracy, Precision, Recall, F1-score, ROC-AUC).
    """
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (Class 0)": precision_score(y_test, y_pred, pos_label=0),
        "Precision (Class 1)": precision_score(y_test, y_pred, pos_label=1),
        "Recall (Class 0)": recall_score(y_test, y_pred, pos_label=0),
        "Recall (Class 1)": recall_score(y_test, y_pred, pos_label=1),
        "F1-score (Class 0)": f1_score(y_test, y_pred, pos_label=0),
        "F1-score (Class 1)": f1_score(y_test, y_pred, pos_label=1),
        "ROC-AUC": roc_auc_score(y_test, y_pred_prob),
    }
    return metrics


# Эндпоинт для построения ROC-кривой
@app.get("/roc_curve/")
async def get_roc_curve():
    """
    Возвращает ROC-кривую в виде изображения.
    """
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    plt.figure()
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


# Эндпоинт для кросс-валидации
@app.get("/cross_validation/")
async def get_cross_validation():
    """
    Выполняет кросс-валидацию и возвращает результаты для каждого фолда:
    - ROC-AUC для каждого фолда
    - Матрицу ошибок для каждого фолда
    - Средний ROC-AUC
    """
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    mean_roc_auc = 0

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)
        y_pred_prob_fold = model.predict_proba(X_test_fold)[:, 1]

        # Вычисление метрик
        cm = confusion_matrix(y_test_fold, y_pred_fold)
        roc_auc = roc_auc_score(y_test_fold, y_pred_prob_fold)
        mean_roc_auc += roc_auc

        # Сохранение результатов для текущего фолда
        results.append({
            "fold": fold,
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist(),  # Преобразуем в список для JSON
        })

    mean_roc_auc /= kfold.get_n_splits()

    return {
        "results": results,
        "mean_roc_auc": mean_roc_auc,
    }


# Эндпоинт для построения графика ROC-AUC для кросс-валидации
@app.get("/cross_validation_roc_auc/")
async def get_cross_validation_roc_auc():
    """
    Возвращает график ROC-AUC для каждого фолда кросс-валидации.
    """
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    roc_auc_scores = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train_fold, y_train_fold)
        y_pred_prob_fold = model.predict_proba(X_test_fold)[:, 1]
        roc_auc_scores.append(roc_auc_score(y_test_fold, y_pred_prob_fold))

    # Построение графика
    plt.figure(figsize=(10, 6))
    folds = np.arange(1, len(roc_auc_scores) + 1)
    plt.plot(folds, roc_auc_scores, marker='o', linestyle='-', color='royalblue', linewidth=2, label='ROC-AUC Score')
    plt.title("5-fold Cross-Validation ROC-AUC", fontsize=16)
    plt.xlabel("Fold Number", fontsize=14)
    plt.ylabel("ROC-AUC Score", fontsize=14)
    plt.xticks(folds)
    plt.grid(True)
    plt.legend(fontsize=12)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


# Эндпоинт для анализа важности признаков
@app.get("/feature_importance/")
async def get_feature_importance():
    """
    Возвращает график важности признаков.
    """
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


# Эндпоинт для SHAP-анализа
@app.get("/shap_analysis/")
async def get_shap_analysis():
    """
    Возвращает SHAP-анализ в виде изображения.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Принимает CSV-файл, возвращает предсказания модели с дополнительной логикой:
    - Предсказания для каждой строки (y_predict)
    - Вероятности предсказаний (predict_proba)
    - Групповые предсказания по колонке sample (sample_predict)
    - Проверка совпадений между y_predict и sample_predict
    """
    try:
        # Загрузка данных
        try:
            data = pd.read_csv(file.file)
        except pd.errors.EmptyDataError:
            raise HTTPException(status_code=400, detail="Файл пуст или не может быть прочитан.")
        except pd.errors.ParserError:
            raise HTTPException(status_code=400, detail="Файл имеет некорректный формат.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Ошибка при чтении файла: {str(e)}")

        # Проверка наличия необходимых колонок
        REQUIRED_COLUMNS = ["L", "sample", "T", "E", "C", "FM", "Xfm", "AFM", "Xafm"]
        for col in REQUIRED_COLUMNS:
            if col not in data.columns:
                raise HTTPException(status_code=400, detail=f"Отсутствует колонка: {col}")
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise HTTPException(status_code=400, detail=f"Колонка {col} должна содержать числовые значения.")

        # Получение предсказаний
        features = ["T", "E", "C", "FM", "Xfm", "AFM", "Xafm"]
        data["y_predict"] = model.predict(data[features])
        data["predict_proba"] = model.predict_proba(data[features])[:, 1]

        # Определение sample_predict
        sample_predictions = (
            data.groupby("sample")["y_predict"]
            .apply(lambda x: 1 if x.sum() > len(x) / 2 else 0)
            .rename("sample_predict")
        )
        data = data.merge(sample_predictions, on="sample")

        # Проверка совпадений
        data["y_predict_matches_sample"] = data["y_predict"] == data["sample_predict"]

        # Возвращаем результат
        return data.to_dict(orient="records")
    except HTTPException:
        raise  # Повторно вызываем HTTPException, чтобы не перехватывать его
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {str(e)}")
