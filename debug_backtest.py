from app.model_loader import load_artifacts
from app.prediction_service import create_prediction_service
import pandas as pd

model, scaler_x, scaler_y, config = load_artifacts('AAPL')
service = create_prediction_service(model, scaler_x, scaler_y, window=config.get('window', 30), device='cpu')

# Debug: fetch data and check
df = service.fetch_recent_data('AAPL', days=100)
print(f"Raw data shape: {df.shape}")
print(f"Raw data columns: {df.columns.tolist()}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

df_features = service.prepare_features(df)
print(f"\nFeatures data shape: {df_features.shape}")
print(f"Features columns: {df_features.columns.tolist()}")
print(f"Date range: {df_features['Date'].min()} to {df_features['Date'].max()}")
print(f"\nLast 5 rows:")
print(df_features[['Date', 'Close', 'log_return', 'target_next_log_return']].tail())
