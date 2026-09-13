---
license: mit
library_name: pytorch
tags:
- stock-prediction
- lstm
- time-series
- pytorch
- financial-ml
model_name: stock-prediction-lstm
owner: RousingSea7309
---

# Stock Prediction LSTM Model

A PyTorch-based LSTM model for predicting next-day stock log returns using technical indicators.

## Model Details

### Architecture

- **Input Size**: 12 features (OHLCV + 7 technical indicators)
- **Hidden Size**: 250
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

### Features Used

1. Open
2. High
3. Low
4. Close
5. Volume
6. SMA_10
7. SMA_30
8. EMA_10
9. RSI_14
10. MACD
11. MACD_signal
12. log_return

## Usage

```python
import torch
import pickle
import json
from models.lstm import StockLSTM
from data.pipeline import FEATURE_COLUMNS

# Load config
with open("config_AAPL.json", "r") as f:
    config = json.load(f)

# Load scalers
with open("scalers_AAPL.pkl", "rb") as f:
    scalers = pickle.load(f)
scaler_x = scalers["scaler_x"]
scaler_y = scalers["scaler_y"]

# Load model
model = StockLSTM(
    input_size=12,
    hidden_size=config["hidden_size"],
    num_layers=config["num_layers"],
    dropout=config["dropout"],
)
model.load_state_dict(torch.load("model_AAPL.pt", map_location="cpu"))
model.eval()

# Make prediction
# (requires recent data with technical indicators)
# See app/prediction_service.py for full inference pipeline
```

## Evaluation Metrics

| Metric | Description |
| -------- | ------------- |
| MSE | Mean Squared Error |
| MAE | Mean Absolute Error |
| R² | Coefficient of Determination |
| SMAPE | Symmetric Mean Absolute Percentage Error |
| Directional Accuracy | Fraction of correct direction predictions |

## Limitations

- Trained on AAPL data only (2015-present)
- Predicts log returns, not raw prices
- Past performance does not guarantee future results
- **This is NOT financial advice**

## License

MIT License
