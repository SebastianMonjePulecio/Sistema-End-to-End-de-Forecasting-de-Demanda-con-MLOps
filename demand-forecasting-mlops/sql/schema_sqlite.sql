-- Tabla de productos
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sku TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    category TEXT,
    subcategory TEXT,
    unit_cost REAL,
    supplier_id INTEGER,
    lead_time_days INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de ventas históricas
CREATE TABLE IF NOT EXISTS sales_history (
    sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER REFERENCES products(product_id),
    date DATE NOT NULL,
    quantity INTEGER NOT NULL,
    revenue REAL,
    store_id INTEGER,
    promotion_active INTEGER DEFAULT 0,  -- 0=FALSE, 1=TRUE
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de inventario actual
CREATE TABLE IF NOT EXISTS inventory (
    inventory_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER REFERENCES products(product_id),
    current_stock INTEGER NOT NULL,
    reorder_point INTEGER,
    max_stock INTEGER,
    warehouse_location TEXT,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de factores externos
CREATE TABLE IF NOT EXISTS external_factors (
    factor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    factor_type TEXT,  -- 'weather', 'holiday', 'event'
    factor_name TEXT,
    impact_score REAL,  -- -1.0 a 1.0
    location TEXT
);

-- Tabla de predicciones
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER REFERENCES products(product_id),
    forecast_date DATE NOT NULL,
    predicted_demand INTEGER NOT NULL,
    confidence_lower INTEGER,
    confidence_upper INTEGER,
    model_version TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(product_id, forecast_date, model_version)
);

-- Tabla de performance de modelos
CREATE TABLE IF NOT EXISTS model_performance (
    performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    product_id INTEGER REFERENCES products(product_id),
    metric_name TEXT,  -- 'MAPE', 'RMSE', 'MAE'
    metric_value REAL,
    training_date DATETIME,
    evaluation_date DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Índices para performance
CREATE INDEX IF NOT EXISTS idx_sales_date ON sales_history(date);
CREATE INDEX IF NOT EXISTS idx_sales_product ON sales_history(product_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_date ON forecasts(forecast_date);