import pandas as pd
import numpy as np
import sqlite3
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineerSQLite:
    def __init__(self, db_path='data/demand_forecast.db'):
        self.db_path = db_path
    
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def load_data(self):
        """Cargar datos de SQLite"""
        logger.info("Cargando datos de la base de datos...")
        conn = self.get_connection()
        
        query = """
        SELECT s.*, p.category, p.unit_cost, p.lead_time_days
        FROM sales_history s
        JOIN products p ON s.product_id = p.product_id
        ORDER BY s.product_id, s.date
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"Cargados {len(df)} registros")
        return df
    
    def create_features(self, df):
        """Crear features para el modelo"""
        logger.info("Creando features...")
        
        df = df.copy()
        df = df.sort_values(['product_id', 'date'])
        
        # 1. Features temporales básicas
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # 2. Features cíclicas
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 3. Lag features
        logger.info("  Creando lag features...")
        for lag in [1, 7, 14, 30]:
            df[f'lag_{lag}d'] = df.groupby('product_id')['quantity'].shift(lag)
        
        # 4. Rolling statistics
        logger.info("  Creando rolling features...")
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}d'] = df.groupby('product_id')['quantity'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'rolling_std_{window}d'] = df.groupby('product_id')['quantity'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            df[f'rolling_max_{window}d'] = df.groupby('product_id')['quantity'].transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )
        
        # 5. Expanding features
        df['expanding_mean'] = df.groupby('product_id')['quantity'].expanding().mean().reset_index(0, drop=True)
        
        # 6. Features por categoría
        category_dummies = pd.get_dummies(df['category'], prefix='cat')
        df = pd.concat([df, category_dummies], axis=1)
        
        # 7. Ratio de cambio
        df['change_1d'] = df.groupby('product_id')['quantity'].pct_change()
        
        logger.info(f"Features creadas. Total columnas: {len(df.columns)}")
        return df
    
    def prepare_training_data(self, df, target_col='quantity'):
        """Preparar X e y para entrenamiento"""
        logger.info("Preparando datos de entrenamiento...")
        
        # CORRECCIÓN: Manejar NaN de forma inteligente
        lag_cols = [f'lag_{lag}d' for lag in [1, 7, 14, 30]]
        
        # Llenar NaN de lags con 0 (asume no hubo ventas antes)
        df[lag_cols] = df[lag_cols].fillna(0)
        
        # Llenar otros NaN con la media de cada columna numérica
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Seleccionar features
        exclude_cols = ['quantity', 'date', 'revenue', 'product_id', 'sku', 'name', 
                       'category', 'subcategory', 'created_at', 'sale_id']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols 
                       and df[col].dtype in ['int64', 'float64', 'uint8']]
        
        X = df[feature_cols]
        y = df[target_col]
        
        logger.info(f"Dataset final: {len(df)} filas, {len(feature_cols)} features")
        
        return X, y, feature_cols, df
    
    def save_processed_data(self, df, output_path='data/processed/features.csv'):
        """Guardar datos procesados"""
        import os
        os.makedirs('data/processed', exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Datos guardados en {output_path}")
    
    def run(self):
        """Ejecutar todo el pipeline de features"""
        df = self.load_data()
        df = self.create_features(df)
        X, y, features, df_clean = self.prepare_training_data(df)
        self.save_processed_data(df_clean)
        
        # Guardar lista de features
        with open('data/processed/feature_list.json', 'w') as f:
            json.dump(features, f)
        
        logger.info("✅ Feature engineering completado")
        return X, y, features

if __name__ == "__main__":
    engineer = FeatureEngineerSQLite()
    X, y, features = engineer.run()
    
    print(f"\n📊 RESUMEN:")
    print(f"   • Muestras: {len(X):,}")
    print(f"   • Features: {len(features)}")
    print(f"   • Target (ventas) promedio: {y.mean():.1f}")
    print(f"\n🎯 Listo para entrenar modelos!")