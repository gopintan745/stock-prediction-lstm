"""
Model loader utility for the Stock Prediction LSTM application.
Handles loading the trained model, scalers, and configuration.
"""

import json
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
from sklearn.preprocessing import StandardScaler

from models.lstm import StockLSTM


def load_artifacts(ticker: str = "AAPL", artifacts_dir: str = None) -> Tuple[StockLSTM, StandardScaler, StandardScaler, Dict[str, Any]]:
    """
    Load the trained model, scalers, and configuration for a given ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        artifacts_dir: Directory containing model artifacts. Defaults to project root.
    
    Returns:
        Tuple of (model, scaler_x, scaler_y, config)
    """
    if artifacts_dir is None:
        artifacts_dir = Path(__file__).parent.parent
    else:
        artifacts_dir = Path(artifacts_dir)
    
    # Load configuration
    config_path = artifacts_dir / f"config_{ticker}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Load scalers
    scalers_path = artifacts_dir / f"scalers_{ticker}.pkl"
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)
    scaler_x = scalers["scaler_x"]
    scaler_y = scalers["scaler_y"]
    
    # Load model with correct architecture from config/optuna
    # Use the best hyperparameters from the Optuna study
    model = StockLSTM(
        input_size=12,  # len(FEATURE_COLUMNS) from data.pipeline
        hidden_size=config.get("hidden_size", 250),
        num_layers=config.get("num_layers", 3),
        dropout=config.get("dropout", 0.09229481465680286),
    )
    
    model_path = artifacts_dir / f"model_{ticker}.pt"
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, scaler_x, scaler_y, config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration (from Optuna best params)."""
    return {
        "hidden_size": 250,
        "num_layers": 3,
        "dropout": 0.09229481465680286,
        "learning_rate": 1.0906196931243558e-05,
        "window": 30,
        "batch_size": 16,
        "optimizer": "adamw",
    }
