# 🚀 Module 1 — Spaceship Titanic: EDA и бинарная классификация

**Автор:** Шунанов Кирилл Ерболатович, группа ПКТб-23-1  
**Дисциплина:** Машинное обучение и ИИ  
**Дата сдачи:** 2026-11-05  
**Датасет:** [Kaggle: Spaceship Titanic](https://www.kaggle.com/c/spaceship-titanic)  

---

## 📊 Результаты моделей на тестовой выборке

| Модель | Accuracy | Precision | Recall | F1-score | ROC-AUC | Время обучения |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7894 | 0.7915 | 0.7852 | 0.7883 | **0.8761** | 0.045 сек |
| Decision Tree | 0.8007 | 0.7986 | 0.8046 | 0.8016 | 0.8630 | 0.012 сек |

> [!NOTE]
> Задача бинарной классификации: предсказать, был ли пассажир перемещен в другое измерение (`Transported`: 0 или 1).
> Обе модели показали ROC-AUC > 0.86. Лучший результат по ROC-AUC показала **Logistic Regression** (0.876), а по Accuracy и F1-score — **Decision Tree** (0.80). 
> Для предсказаний в `predict_passenger` мы используем обе модели (LR с масштабированием, DT без масштабирования).

---

## 🚀 Быстрый старт (инференс из GitHub репозитория)

```python
import io
import json
import joblib
import requests
import numpy as np
import pandas as pd

# Базовый URL сырых файлов репозитория
BASE_URL = "https://raw.githubusercontent.com/shunanovke/ml-course-shunanov/main/module-1-titanic"

# 1. Загрузка обученной модели Logistic Regression
model_url = f"{BASE_URL}/models/lr_model.pkl"
lr_model = joblib.load(io.BytesIO(requests.get(model_url).content))

# 2. Загрузка StandardScaler (для LR)
scaler_url = f"{BASE_URL}/models/scaler.pkl"
scaler = joblib.load(io.BytesIO(requests.get(scaler_url).content))

# 3. Загрузка списка признаков
features_url = f"{BASE_URL}/models/feature_cols.json"
feature_cols = requests.get(features_url).json()

print("Модель и Scaler успешно загружены из репозитория!")