import pandas as pd
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ETLPipelineSQLite:
    def __init__(self, db_path='data/demand_forecast.db'):
        self.db_path = db_path
        
    def get_connection(self):
        return sqlite3.connect(self.db_path)
        
    def extract(self):
        """Cargar datos desde CSV"""
        logger.info("Extrayendo datos...")
        products = pd.read_csv('data/raw/products.csv')
        sales = pd.read_csv('data/raw/sales_history.csv')
        return products, sales
    
    def transform(self, products, sales):
        """Limpiar y transformar"""
        logger.info("Transformando datos...")
        
        sales = sales.dropna()
        sales['date'] = pd.to_datetime(sales['date'])
        
        # Convertir boolean a integer para SQLite
        if 'promotion_active' in sales.columns:
            sales['promotion_active'] = sales['promotion_active'].astype(int)
        
        return products, sales
    
    def load(self, products, sales):
        """Cargar a SQLite"""
        logger.info("Cargando a base de datos...")
        
        conn = self.get_connection()
        
        # Cargar productos
        products.to_sql('products', conn, if_exists='append', index=False)
        logger.info(f"  Cargados {len(products)} productos")
        
        # Cargar ventas en batches
        batch_size = 5000
        for i in range(0, len(sales), batch_size):
            batch = sales.iloc[i:i+batch_size]
            batch.to_sql('sales_history', conn, if_exists='append', index=False)
            logger.info(f"  Cargados {i + len(batch)} registros de ventas...")
        
        conn.close()
        logger.info("✅ ETL completado")
    
    def run(self):
        products, sales = self.extract()
        products, sales = self.transform(products, sales)
        self.load(products, sales)

if __name__ == "__main__":
    pipeline = ETLPipelineSQLite()
    pipeline.run()