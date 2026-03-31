# 🎯 Sistema Inteligente de Predicción de Demanda

Sistema end-to-end para forecasting de demanda con Machine Learning y MLOps.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/ML-Ensemble-green)
![API](https://img.shields.io/badge/API-FastAPI-orange)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)

## 📊 Resultados

- **MAPE (Error):** 2.49% (industria estándar: 20-30%)
- **Dataset:** 73,000 registros de ventas
- **Features:** 23 variables
- **Modelo:** Ensemble (XGBoost + LightGBM + Gradient Boosting)

## 🏗️ Arquitectura


## 🚀 Cómo ejecutar

```bash
# 1. Generar datos
python src/data_generator.py

# 2. Crear base de datos
python src/database/init_db.py

# 3. Correr ETL
python src/etl/pipeline_sqlite.py

# 4. Feature engineering
python src/features/engineering_sqlite.py

# 5. Entrenar modelo
python src/models/train_ensemble.py

# 6. API (Terminal 1)
cd src/api && python -m uvicorn main:app --port 8000

# 7. Dashboard (Terminal 2)
python -m streamlit run src/dashboard/app.py
📡 Endpoints API
GET / - Info del sistema
GET /products - Listar productos
GET /stats/{product_id} - Estadísticas
POST /predict - Predicción de demanda
🛠️ Tech Stack
Python 3.10+
SQLite - Base de datos
scikit-learn - ML
XGBoost & LightGBM - Gradient Boosting
FastAPI - API REST
Streamlit - Dashboard
Plotly - Visualizaciones
👨‍💻 Autor
Sebastián Monje - Data Analyst
📧 monjesebas7@gmail.com
💼 LinkedIn

