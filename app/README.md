# Stock Prediction LSTM - Application

This directory contains the application layer for the trained LSTM stock prediction model.

## Components

- **`model_loader.py`** - Loads trained model, scalers, and configuration
- **`prediction_service.py`** - Handles data fetching, preprocessing, and predictions
- **`streamlit_app.py`** - Web UI built with Streamlit
- **`cli.py`** - Command-line interface for quick predictions
- **`__init__.py`** - Package exports

## Quick Start

### Web UI (Streamlit)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app/streamlit_app.py
```

The app will be available at `http://localhost:8501`

### Command Line Interface

```bash
# Single prediction
python -m app.cli predict --ticker AAPL

# With GPU
python -m app.cli predict --ticker AAPL --device cuda

# Backtest
python -m app.cli backtest --ticker AAPL --days 30
```

### Python API

```python
from app.model_loader import load_artifacts
from app.prediction_service import create_prediction_service

# Load model and artifacts
model, scaler_x, scaler_y, config = load_artifacts("AAPL")

# Create prediction service
service = create_prediction_service(
    model, scaler_x, scaler_y,
    window=config.get("window", 30),
    device="cpu"
)

# Make prediction
result = service.make_prediction("AAPL")
print(f"Predicted: {result['direction']} ({result['predicted_log_return']:.6f})")

# Run backtest
historical = service.get_historical_predictions("AAPL", test_days=30)
```

## Model Artifacts Required

The following files must exist in the project root (or specify `artifacts_dir`):

- `model_<TICKER>.pt` - Trained model weights
- `scalers_<TICKER>.pkl` - Fitted StandardScaler objects
- `config_<TICKER>.json` - Training configuration

## Features Used

The model uses 12 technical indicators as input features:

1. Open, High, Low, Close, Volume
2. SMA_10 (10-day Simple Moving Average)
3. SMA_30 (30-day Simple Moving Average)
4. EMA_10 (10-day Exponential Moving Average)
5. RSI_14 (14-day Relative Strength Index)
6. MACD (Moving Average Convergence Divergence)
7. MACD_signal (MACD Signal Line)
8. log_return (Previous day's log return)

## Target

The model predicts the **next-day log return**: `log(Close[t+1] / Close[t])`

## Disclaimer

⚠️ **This is NOT financial advice.** The predictions are based on a machine learning model trained on historical data. Past performance does not guarantee future results. Always do your own research before making investment decisions.
