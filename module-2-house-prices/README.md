# 🏠 Module 2 — House Prices: Предобработка данных и регрессия

**Автор:** Шунанов Кирилл Ерболатович, группа ПКТб-23-1  
**Дисциплина:** Машинное обучение и ИИ  
**Дата сдачи:** 2026-11-05  
**Датасет:** [Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  
 
---

## 📊 Результаты моделей на тестовой выборке

| Модель | RMSE (log) | MAE (log) | $R^2$ | RMSE ($) | Время обучения |
|---|---|---|---|---|---|
| Linear Regression | 0.1378 | 0.0938 | 0.8873 | $22,688 | 0.029 сек |
| Ridge ($\alpha=10.0$) | 0.1276 | 0.0893 | 0.9034 | $21,631 | 0.004 сек |
| **Lasso ($\alpha=0.0005$)** | **0.1239** | **0.0861** | **0.9090** | **$20,952** | **0.183 сек** |
| ElasticNet ($\alpha=0.001, l1\_ratio=0.5$) | 0.1241 | 0.0862 | 0.9086 | $20,958 | 0.133 сек |

> [!NOTE]
> Все обученные регуляризованные модели преодолели целевой порог $R^2 \ge 0.90$.  
> Лучший результат показала **Lasso** ($R^2 = 0.9090$, RMSE в долларах = $20,952$).  
> Модель Lasso также выполнила отбор признаков, занулив 52 неинформативных признака из 238 (осталось 186 значимых).

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
BASE_URL = "https://raw.githubusercontent.com/shunanovke/ml-course-shunanov/main/module-2-house-prices"

# 1. Загрузка лучшей модели (Lasso)
model_url = f"{BASE_URL}/models/lasso_model.pkl"
lasso_model = joblib.load(io.BytesIO(requests.get(model_url).content))

# 2. Загрузка scaler
scaler_url = f"{BASE_URL}/models/scaler.pkl"
scaler = joblib.load(io.BytesIO(requests.get(scaler_url).content))

# 3. Загрузка списка признаков
features_url = f"{BASE_URL}/models/feature_cols.json"
feature_cols = requests.get(features_url).json()

# 4. Предсказание цены нового дома (упрощенная логика)
house = {
    'OverallQual': 9, 'TotalSF': 4600, 'GrLivArea': 2800,
    'YearBuilt': 2015, 'YearRemodAdd': 2015, 'YrSold': 2019,
    'TotalBath': 4.5, 'GarageCars': 3, 'Fireplaces': 2,
    'OverallQual_TotalSF': 41400, 'ExterQual': 4, 'KitchenQual': 5
}

df_new = pd.DataFrame([house])
for c in feature_cols:
    if c not in df_new.columns:
        df_new[c] = 0.0
df_new = df_new[feature_cols]

pred_log = lasso_model.predict(scaler.transform(df_new))[0]
pred_price = np.expm1(pred_log)
print(f"Оценочная стоимость дома: ${pred_price:,.0f}")