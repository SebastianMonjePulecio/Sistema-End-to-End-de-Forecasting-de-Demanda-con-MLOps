import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

# Crear carpetas
os.makedirs('data/raw', exist_ok=True)

# Generar 100 productos
n_products = 100
products = pd.DataFrame({
    'product_id': range(1, n_products + 1),
    'sku': [f'SKU-{i:04d}' for i in range(1, n_products + 1)],
    'name': [f'Product {i}' for i in range(1, n_products + 1)],
    'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home'], n_products),
    'unit_cost': np.random.uniform(10, 500, n_products).round(2),
    'lead_time_days': np.random.randint(3, 30, n_products)
})

# Generar 2 años de ventas
date_range = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
sales_data = []

for product_id in range(1, n_products + 1):
    base_demand = {'Electronics': 50, 'Clothing': 120, 'Food': 300, 'Home': 80}[
        products.iloc[product_id-1]['category']
    ]
    
    for date in date_range:
        seasonal_factor = 1.5 if date.month == 12 else 1.0
        weekend_factor = 0.7 if date.weekday() >= 5 else 1.0
        trend = 1 + (date - date_range[0]).days / 365 * 0.1
        noise = np.random.normal(1, 0.2)
        
        quantity = int(base_demand * seasonal_factor * weekend_factor * trend * noise)
        quantity = max(0, quantity)
        
        sales_data.append({
            'product_id': product_id,
            'date': date.strftime('%Y-%m-%d'),
            'quantity': quantity,
            'revenue': round(quantity * products.iloc[product_id-1]['unit_cost'] * np.random.uniform(1.2, 1.5), 2),
            'promotion_active': 1 if np.random.random() < 0.05 else 0
        })

sales_df = pd.DataFrame(sales_data)

# Guardar
products.to_csv('data/raw/products.csv', index=False)
sales_df.to_csv('data/raw/sales_history.csv', index=False)

print(f"✅ Generados: {len(products)} productos, {len(sales_df)} registros de ventas")