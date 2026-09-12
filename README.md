# Stock Prediction LSTM

A PyTorch-based LSTM model for predicting next-day stock log returns using technical indicators.

## Project Structure

```
stock-prediction-lstm/
├── app/                    # Application layer
│   ├── model_loader.py     # Load trained model and artifacts
│   ├── prediction_service.py  # Data fetching & prediction logic
│   ├── streamlit_app.py    # Web UI (Streamlit)
│   ├── cli.py              # Command-line interface
│   └── README.md           # App documentation
├── data/
│   └── pipeline.py         # Data download, feature engineering, preprocessing
├── models/
│   └── lstm.py             # LSTM model architecture
├── training/
│   ├── loop.py             # Training loop, dataloaders, optimizers
│   ├── hpo.py              # Hyperparameter optimization (Optuna)
│   └── final_training.py   # Final training with best HPO params
├── utils/
│   └── metrics.py          # Evaluation metrics
├── config_AAPL.json        # Training configuration
├── model_AAPL.pt           # Trained model weights
├── scalers_AAPL.pkl        # Fitted scalers
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Features

- **LSTM Architecture**: Multi-layer LSTM with dropout for sequence modeling
- **Technical Indicators**: SMA, EMA, RSI, MACD, log returns
- **Proper Data Pipeline**: Chronological splits, no data leakage, scaling fit on train only
- **Hyperparameter Optimization**: Optuna for HPO
- **Comprehensive Metrics**: MSE, MAE, R², SMAPE, Directional Accuracy
- **Statistical Significance Testing**: Binomial test for directional accuracy
- **Web UI**: Streamlit app for interactive predictions
- **CLI**: Command-line interface for quick predictions and backtesting

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web Application (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

### Command Line

```bash
# Single prediction
python -m app.cli predict --ticker AAPL

# With GPU acceleration
python -m app.cli predict --ticker AAPL --device cuda

# Backtest on recent data
python -m app.cli backtest --ticker AAPL --days 30
```

### Python API

```python
from app.model_loader import load_artifacts
from app.prediction_service import create_prediction_service

# Load trained model
model, scaler_x, scaler_y, config = load_artifacts("AAPL")

# Create service
service = create_prediction_service(
    model, scaler_x, scaler_y,
    window=config.get("window", 30),
    device="cpu"
)

# Make prediction
result = service.make_prediction("AAPL")
print(f"Direction: {result['direction']}")
print(f"Predicted Log Return: {result['predicted_log_return']:.6f}")
print(f"Predicted Price: ${result['predicted_price']:.2f}")

# Backtest
historical = service.get_historical_predictions("AAPL", test_days=30)
```

## Model Details

### Architecture
- **Input Size**: 12 features (OHLCV + 7 technical indicators)
- **Hidden Size**: 250 (from Optuna HPO)
- **Layers**: 3 LSTM layers
- **Dropout**: 0.092
- **Output**: Single value (next-day log return)

### Training
- **Optimizer**: AdamW
- **Learning Rate**: 1.09e-5
- **Batch Size**: 16
- **Window Size**: 30 days
- **Epochs**: 100 (with early stopping)
- **Loss**: MSE

### Data
- **Source**: Yahoo Finance (yfinance)
- **Period**: 2015-01-01 to present
- **Split**: 70% train / 15% val / 15% test (chronological)
- **Target**: Next-day log return `log(Close[t+1] / Close[t])`

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MSE | Mean Squared Error |
| MAE | Mean Absolute Error |
| R² | Coefficient of Determination |
| SMAPE | Symmetric Mean Absolute Percentage Error |
| Directional Accuracy | Fraction of correct direction predictions |

## Requirements

- Python 3.9+
- PyTorch 2.0+
- See `requirements.txt` for full list

## Disclaimer

⚠️ **This is NOT financial advice.** The predictions are based on a machine learning model trained on historical data. Past performance does not guarantee future results. Always do your own research before making investment decisions.

## License

MIT License
