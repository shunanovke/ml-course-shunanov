# 🚀 Module 1 — Spaceship Titanic: EDA и бинарная классификация

**Автор:** Шунанов Кирилл Ерболатович, группа ПКТб-23-1  
**Дисциплина:** Машинное обучение и ИИ  
**Дата сдачи:** 2026-11-05  
**Датасет:** [Kaggle: Spaceship Titanic](https://www.kaggle.com/c/spaceship-titanic)  

---

## 📊 Результаты моделей на тестовой выборке

| Модель | Accuracy | Precision | Recall | F1-score | ROC-AUC | Время обучения |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7878 | 0.7877 | 0.7922 | 0.7900 | **0.8762** | ~0.01 сек |
| Decision Tree | **0.7993** | **0.7931** | **0.8139** | **0.8034** | 0.8631 | ~0.01 сек |

> [!NOTE]
> Задача бинарной классификации: предсказать, был ли пассажир перемещен в другое измерение (`Transported`: 0 или 1).
> Обе модели показали достойный результат. **Logistic Regression** лучше ранжирует вероятности (ROC-AUC = 0.8762), в то время как **Decision Tree** показывает более высокую точность (Accuracy = 0.7993) и F1-score (0.8034). 
> Для предсказаний в функции `predict_passenger` используются обе модели (LR с масштабированием, DT без масштабирования).

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