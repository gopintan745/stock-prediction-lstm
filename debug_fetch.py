from app.prediction_service import PredictionService
import yfinance as yf
from data.pipeline import add_technical_indicators
import pandas as pd

# Test with different day counts
for days in [30, 50, 100, 200]:
    df = yf.download('AAPL', start=(pd.Timestamp.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d'), end=pd.Timestamp.now().strftime('%Y-%m-%d'), auto_adjust=True, progress=False)
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[0] != "" else c[1] for c in df.columns]
    
    df_feat = add_technical_indicators(df)
    print(f"Days={days}: raw={len(df)}, features={len(df_feat)}")
