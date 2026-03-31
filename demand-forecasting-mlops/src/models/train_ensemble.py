import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemandEnsemble:
    def __init__(self):
        self.models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        }
        self.weights = {'xgboost': 0.4, 'lightgbm': 0.4, 'gbm': 0.2}
        self.is_fitted = False
        
    def fit(self, X_train, y_train):
        """Entrenar todos los modelos"""
        logger.info("Entrenando modelos...")
        
        for name, model in self.models.items():
            logger.info(f"  Entrenando {name}...")
            model.fit(X_train, y_train)
            
            # Metricas de training
            train_pred = model.predict(X_train)
            train_mape = mean_absolute_percentage_error(y_train, train_pred)
            logger.info(f"    {name} - Train MAPE: {train_mape:.4f}")
        
        self.is_fitted = True
        logger.info("✅ Modelos entrenados")
            
    def predict(self, X):
        """Predicción ponderada del ensemble"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # Promedio ponderado
        ensemble_pred = sum(
            predictions[name] * weight 
            for name, weight in self.weights.items()
        )
        
        return ensemble_pred, predictions
    
    def evaluate(self, X_test, y_test):
        """Evaluar performance"""
        logger.info("Evaluando modelo...")
        ensemble_pred, individual_preds = self.predict(X_test)
        
        results = {
            'ensemble_mape': mean_absolute_percentage_error(y_test, ensemble_pred),
            'ensemble_rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred))
        }
        
        for name, pred in individual_preds.items():
            results[f'{name}_mape'] = mean_absolute_percentage_error(y_test, pred)
        
        # Mostrar resultados
        logger.info("\n📊 RESULTADOS:")
        logger.info(f"  Ensemble MAPE: {results['ensemble_mape']:.4f} ({results['ensemble_mape']*100:.2f}%)")
        logger.info(f"  Ensemble RMSE: {results['ensemble_rmse']:.2f}")
        
        for name in self.models.keys():
            logger.info(f"  {name} MAPE: {results[f'{name}_mape']:.4f} ({results[f'{name}_mape']*100:.2f}%)")
        
        return results
    
    def save(self, path='models/ensemble_model.pkl'):
        import os
        os.makedirs('models', exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"✅ Modelo guardado en {path}")
    
    @classmethod
    def load(cls, path='models/ensemble_model.pkl'):
        """Cargar modelo guardado"""
        import joblib
        return joblib.load(path)

def main():
    # Cargar datos
    logger.info("Cargando datos procesados...")
    df = pd.read_csv('data/processed/features.csv')
    
    with open('data/processed/feature_list.json', 'r') as f:
        feature_cols = json.load(f)
    
    # Quitar columnas que no son features
    feature_cols = [f for f in feature_cols if f not in ['quantity', 'sale_id']]
    
    X = df[feature_cols]
    y = df['quantity']
    
    # Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # False para series temporales
    )
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Entrenar
    model = DemandEnsemble()
    model.fit(X_train, y_train)
    
    # Evaluar
    results = model.evaluate(X_test, y_test)
    
    # Guardar
    model.save()
    
    # Guardar métricas
    with open('models/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n🎯 ENTRENAMIENTO COMPLETADO")

if __name__ == "__main__":
    main()